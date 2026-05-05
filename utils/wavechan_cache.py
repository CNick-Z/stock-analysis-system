# utils/wavechan_cache.py
# WaveChan 三层缓存管理器
# 
# 架构：
#   L1 历史归档层（Parquet，按年分区）- 2021-2024，永不重建
#   L2 热数据层（Parquet，按月分区）- 2025-2026，当年重建
#   L3 参数缓存层（SQLite）- (算法版本+参数hash) → 结果，LRU淘汰
#
# 收益：
#   换算法回测：7小时全量 → 只重建L2（1-2小时）
#   换参数回测：命中L3毫秒级 / 未命中分钟级
#   每日增量：7小时全量 → 秒级append

import os
import sys
import json
import hashlib
import sqlite3
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import OrderedDict
from contextlib import contextmanager

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
# 路径配置（统一从 paths.py 导入）
# ============================================================
from utils.paths import WAVECHAN_ROOT, WAVECHAN_L2_ROOT, WAVECHAN_L3_DB

# L1: 历史归档（2021-2024）
L1_COLD_YEARS = [2021, 2022, 2023, 2024]

# L2: 热数据（2025+）
L2_HOT_YEAR_START = 2025

# L3: 参数缓存（SQLite）
L3_DB_PATH = WAVECHAN_L3_DB  # from utils.paths
L3_MAX_ENTRIES = 100  # LRU 保留最近 100 组参数

# L2 cache base path (alias for WAVECHAN_L2_ROOT)
CACHE_BASE = WAVECHAN_L2_ROOT

# 默认参数版本
DEFAULT_ALGO_VERSION = "v3.0"


# ============================================================
# L3: 参数缓存层（SQLite LRU）
# ============================================================

class L3ParameterCache:
    """
    L3 参数缓存层
    - 存储：(algo_version + param_hash) → backtest_result
    - 淘汰：LRU，保留最近 N 组
    """

    def __init__(self, db_path: Path = L3_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS param_cache (
                    algo_version TEXT NOT NULL,
                    param_hash TEXT NOT NULL,
                    param_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    access_at REAL NOT NULL,
                    PRIMARY KEY (algo_version, param_hash)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_at ON param_cache(access_at)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _compute_hash(self, params: dict) -> str:
        """计算参数的 MD5 hash"""
        # 排序确保一致性
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def get(self, algo_version: str, params: dict) -> Optional[dict]:
        """获取缓存结果，命中则更新 access_at"""
        param_hash = self._compute_hash(params)

        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT result_json FROM param_cache
                WHERE algo_version = ? AND param_hash = ?
            """, (algo_version, param_hash)).fetchone()

            if row is None:
                return None

            # 更新访问时间
            conn.execute("""
                UPDATE param_cache SET access_at = ? 
                WHERE algo_version = ? AND param_hash = ?
            """, (time.time(), algo_version, param_hash))
            conn.commit()

            return json.loads(row["result_json"])

    def put(self, algo_version: str, params: dict, result: dict):
        """写入缓存，自动 LRU 淘汰"""
        param_hash = self._compute_hash(params)
        param_json = json.dumps(params, sort_keys=True, default=str)
        result_json = json.dumps(result, default=str)
        now = time.time()

        with self._get_conn() as conn:
            # Upsert
            conn.execute("""
                INSERT INTO param_cache (algo_version, param_hash, param_json, result_json, created_at, access_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(algo_version, param_hash) 
                DO UPDATE SET result_json = excluded.result_json, access_at = excluded.access_at
            """, (algo_version, param_hash, param_json, result_json, now, now))

            # LRU 淘汰：保留最近 L3_MAX_ENTRIES 条
            conn.execute("""
                DELETE FROM param_cache 
                WHERE algo_version = ? 
                AND rowid NOT IN (
                    SELECT rowid FROM param_cache 
                    WHERE algo_version = ?
                    ORDER BY access_at DESC
                    LIMIT ?
                )
            """, (algo_version, algo_version, L3_MAX_ENTRIES))

            conn.commit()

        logger.debug(f"[L3] 缓存写入: {algo_version}/{param_hash}")

    def clear(self, algo_version: Optional[str] = None):
        """清空缓存"""
        with self._get_conn() as conn:
            if algo_version:
                conn.execute("DELETE FROM param_cache WHERE algo_version = ?", (algo_version,))
            else:
                conn.execute("DELETE FROM param_cache")
            conn.commit()
        logger.info(f"[L3] 缓存已清空: {algo_version or 'all'}")

    def info(self) -> dict:
        """获取缓存状态"""
        with self._get_conn() as conn:
            if conn.execute("SELECT COUNT(*) FROM param_cache").fetchone()[0] == 0:
                return {"count": 0, "versions": []}

            count = conn.execute("SELECT COUNT(*) FROM param_cache").fetchone()[0]
            versions = [r[0] for r in conn.execute(
                "SELECT DISTINCT algo_version FROM param_cache"
            ).fetchall()]

            return {"count": count, "versions": versions}


# ============================================================
# L1+L2: Parquet 分层缓存
# ============================================================

class L1L2ParquetCache:
    """
    L1/L2 Parquet 分层存储
    - L1: 冷数据，按年分区，2021-2024（只读）
    - L2: 热数据，按年+月分区，2025+（可重建）
    """

    def __init__(self, base_path: Path = CACHE_BASE):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------ L1 历史归档（只读）------

    def _l1_path(self, year: int) -> Path:
        return self.base_path / f"l1_cold_year={year}"

    def _l2_path(self, year: int, month: int) -> Path:
        return self.base_path / f"l2_hot_year={year}_month={month:02d}"

    def read_l1(self, year: int) -> pd.DataFrame:
        """读取 L1 冷数据（单年）"""
        path = self._l1_path(year) / "data.parquet"
        if not path.exists():
            logger.warning(f"[L1] 文件不存在: {path}")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_l2(self, year: int, month: int) -> pd.DataFrame:
        """读取 L2 热数据（单月）"""
        path = self._l2_path(year, month) / "data.parquet"
        if not path.exists():
            logger.warning(f"[L2] 文件不存在: {path}")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        读取指定日期范围的数据（自动路由 L1/L2）
        2021-2024 → L1
        2025+ → L2
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dfs = []

        current = start_dt
        while current <= end_dt:
            year, month = current.year, current.month

            if year in L1_COLD_YEARS:
                df = self.read_l1(year)
                if not df.empty:
                    dfs.append(df)
            else:
                df = self.read_l2(year, month)
                if not df.empty:
                    dfs.append(df)

            # 下月
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        # 过滤日期范围
        result = result[
            (result['date'] >= start_date) &
            (result['date'] <= end_date)
        ]
        return result

    def write_l2(self, year: int, month: int, df: pd.DataFrame):
        """写入 L2 热数据（单月）"""
        path = self._l2_path(year, month)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "data.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        logger.info(f"[L2] 写入: {file_path} ({len(df):,} 行)")

    def write_l1(self, year: int, df: pd.DataFrame):
        """写入 L1 冷数据（单年，一次性）"""
        path = self._l1_path(year)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "data.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        logger.info(f"[L1] 写入: {file_path} ({len(df):,} 行)")

    def partition_info(self) -> dict:
        """返回各分区元信息（行数、日期范围）"""
        info = {}
        for year in L1_COLD_YEARS:
            path = self._l1_path(year) / "data.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                info[f"l1_year={year}"] = {
                    "rows": len(df),
                    "symbols": df['symbol'].nunique(),
                    "date_min": df['date'].min(),
                    "date_max": df['date'].max(),
                    "size_mb": path.stat().st_size / 1024 / 1024,
                }

        # L2: 检测所有年月
        for p in self.base_path.glob("l2_hot_year=*"):
            name = p.name
            pf = p / "data.parquet"
            if pf.exists():
                df = pd.read_parquet(pf)
                info[name] = {
                    "rows": len(df),
                    "symbols": df['symbol'].nunique(),
                    "date_min": df['date'].min(),
                    "date_max": df['date'].max(),
                    "size_mb": pf.stat().st_size / 1024 / 1024,
                }
        return info

    def get_l2_size_mb(self) -> float:
        """估算 L2 总大小（MB）"""
        total = 0
        for p in self.base_path.glob("l2_hot_year=*"):
            pf = p / "data.parquet"
            if pf.exists():
                total += pf.stat().st_size
        return total / 1024 / 1024


# ============================================================
# 三层缓存管理器（主入口）
# ============================================================

class WaveChanCacheManager:
    """
    WaveChan 三层缓存管理器

    对外接口：
        load(start_date, end_date, params) → DataFrame
        save_l2(year, month, df)
        get_cached_result(algo_version, params) → dict
        put_cached_result(algo_version, params, result)
        rebuild_l2(years, params, selector) → None
        daily_increment(date, df) → None
    """

    def __init__(
        self,
        base_path: Path = CACHE_BASE,
        l3_db_path: Path = L3_DB_PATH,
    ):
        self.l1l2 = L1L2ParquetCache(base_path)
        self.l3 = L3ParameterCache(l3_db_path)

        logger.info("[WaveChanCache] 三层缓存管理器初始化完成")
        logger.info(f"  L1 冷数据: {[f'{y}' for y in L1_COLD_YEARS]}")
        logger.info(f"  L2 热数据: {L2_HOT_YEAR_START}+")
        logger.info(f"  L3 参数缓存: {L3_MAX_ENTRIES} 组 LRU")

    # ------ 数据读取 ------

    def load(
        self,
        start_date: str,
        end_date: str,
        algo_version: str = DEFAULT_ALGO_VERSION,
        params: dict = None,
    ) -> pd.DataFrame:
        """
        读取指定日期范围的缓存信号数据
        自动路由 L1/L2，参数变化时查 L3
        """
        # 尝试 L3 参数缓存
        if params:
            cached = self.l3.get(algo_version, params)
            if cached is not None:
                logger.info(f"[WaveChanCache] L3 命中: {algo_version}/{self._hash_params(params)}")

        # L1+L2 读取
        df = self.l1l2.read_range(start_date, end_date)
        logger.info(f"[WaveChanCache] 加载 {start_date}~{end_date}: {len(df):,} 行")
        return df

    # ------ L2 写入（增量更新）------

    def daily_increment(self, date: str, df: pd.DataFrame):
        """
        每日增量：追加单日数据到 L2
        date: YYYY-MM-DD
        df: 当日信号数据（必须包含 date, symbol 字段）
        """
        dt = pd.to_datetime(date)
        year, month = dt.year, dt.month

        # 读取当月已有数据
        existing = self.l1l2.read_l2(year, month)

        if existing.empty:
            # 当月无数据，直接写入
            combined = df.copy()
        else:
            # 合并：去重（date + symbol）
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
            combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)

        self.l1l2.write_l2(year, month, combined)
        logger.info(f"[WaveChanCache] 每日增量完成: {date} → L2 year={year} month={month:02d}")

    # ------ L3 参数缓存 ------

    def get_cached_backtest_result(
        self,
        algo_version: str,
        params: dict,
    ) -> Optional[dict]:
        """查询 L3 缓存的回测结果"""
        return self.l3.get(algo_version, params)

    def put_cached_backtest_result(
        self,
        algo_version: str,
        params: dict,
        result: dict,
    ):
        """写入 L3 缓存的回测结果"""
        self.l3.put(algo_version, params, result)

    # ------ L2 重建（换算法时）------

    def rebuild_l2(
        self,
        years: List[int],
        params: dict,
        selector,  # WaveChanSelector 实例
        progress_callback=None,
    ):
        """
        重建 L2 热数据（指定年份）
        用于：换算法后重新计算信号

        Args:
            years: 要重建的年份列表 [2025, 2026]
            params: 算法参数字典
            selector: WaveChanSelector 实例（必须提供 _compute_symbol_scores 方法）
            progress_callback: (current, total, symbol) -> None
        """
        from strategies.wavechan_selector import WaveChanSelector

        total_symbols = 0
        for yr in years:
            yr_data = self.l1l2.read_l2(yr, 1)  # 检查是否存在
            if not yr_data.empty:
                total_symbols = yr_data['symbol'].nunique()
                break

        if total_symbols == 0:
            # 没有现有数据，估算
            total_symbols = 5400 * len(years)

        logger.info(f"[WaveChanCache] 开始重建 L2: {years}, 约 {total_symbols} 只股票")

        t0 = time.time()

        for yr in years:
            for month in range(1, 13):
                dt = datetime(yr, month, 1)
                if dt > datetime.now():
                    continue

                # 构建该月数据（需要 lookback）
                lookback_start = (dt - timedelta(days=150)).strftime("%Y-%m-%d")
                lookback_end = (dt.replace(day=28) + timedelta(days=4)).strftime("%Y-%m-%d")

                # 从 L1+L2 构建输入
                input_df = self.l1l2.read_range(lookback_start, lookback_end)
                if input_df.empty:
                    continue

                results = []
                for sym in input_df['symbol'].unique():
                    sym_df = input_df[input_df['symbol'] == sym].sort_values('date')
                    if len(sym_df) < 60:
                        continue

                    try:
                        scored = selector._compute_symbol_scores(sym, sym_df)
                        if scored is not None and not scored.empty:
                            # 过滤当月数据
                            month_start = f"{yr}-{month:02d}-01"
                            month_end = (dt.replace(day=28) + timedelta(days=4)).strftime("%Y-%m-%d")
                            scored = scored[
                                (scored['date'] >= month_start) &
                                (scored['date'] <= month_end)
                            ]
                            if not scored.empty:
                                results.append(scored)
                    except Exception as e:
                        logger.debug(f"[WaveChanCache] {sym} 异常: {e}")
                        continue

                    if progress_callback and len(results) % 100 == 0:
                        progress_callback(len(results), total_symbols, sym)

                if results:
                    combined = pd.concat(results, ignore_index=True)
                    self.l1l2.write_l2(yr, month, combined)

        elapsed = time.time() - t0
        logger.info(f"[WaveChanCache] L2 重建完成，耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ------ 状态查询 ------

    def status(self) -> dict:
        """返回三层缓存状态"""
        l1l2_info = self.l1l2.partition_info()
        l3_info = self.l3.info()
        l2_size_mb = self.l1l2.get_l2_size_mb()

        return {
            "l1_partitions": {k: v for k, v in l1l2_info.items() if k.startswith("l1_")},
            "l2_partitions": {k: v for k, v in l1l2_info.items() if k.startswith("l2_")},
            "l2_size_mb": round(l2_size_mb, 1),
            "l3_count": l3_info["count"],
            "l3_versions": l3_info["versions"],
        }

    @staticmethod
    def _hash_params(params: dict) -> str:
        """计算参数字典的短 hash"""
        s = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]


# ============================================================
# 兼容性别名（向后兼容）
# ============================================================

WaveChanSignalCache = WaveChanCacheManager
