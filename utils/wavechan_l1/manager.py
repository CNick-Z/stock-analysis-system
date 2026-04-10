"""
WaveChan L1 - 主管理器
=======================
统一管理 L1 缓存（极值点 + 趋势）的构建、查询

存储结构：
    /data/warehouse/wavechan_l1/extrema_year={year}/{symbol}.parquet
    /data/warehouse/wavechan_l1/trend_year={year}/{symbol}.parquet
"""

import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

from . import zigzag
from . import trend
from ._path import BASE_PATH, extrema_path, trend_path, ensure_dirs, list_available_years

logger = logging.getLogger(__name__)


class WaveChanL1Manager:
    """
    WaveChan L1 缓存管理器

    核心功能：
        build_year(year)    → 完整构建（极值 → 趋势）
        rebuild_year(year)  → 重建（先删后建）
        daily_increment()    → 增量更新
        get_extrema()        → 查询极值点
        get_trend()          → 查询趋势序列
        status()             → 查看缓存状态
    """

    def __init__(self, base_path: Optional[Path] = None, n_jobs: int = 8):
        self.base_path = base_path or BASE_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        logger.info(f"[WaveChanL1] 初始化完成，根目录: {self.base_path}, n_jobs={n_jobs}")

    # ============================================================
    # 构建
    # ============================================================

    def build_year(
        self,
        year: int,
        symbols: Optional[List[str]] = None,
        threshold: float = 0.05,
        lookback: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        完整构建某年的 L1 数据（极值 → 趋势）

        参数：
            year: 年份
            symbols: 股票列表，None = 全部
            threshold: ZigZag 回撤阈值（默认 5%）
            lookback: 趋势判断回看极值数（默认 5）
            verbose: 是否打印详情

        返回：
            dict: {
                'year': int,
                'extrema_count': int,    # 成功处理极值的股票数
                'trend_count': int,      # 成功处理趋势的股票数
                'symbols': List[str],
                'elapsed_seconds': float,
            }
        """
        import time
        t0 = time.time()

        logger.info(f"[WaveChanL1] 开始构建 {year} 年 L1 ...")

        # Step 1: 极值识别
        if verbose:
            logger.info(f"[WaveChanL1] Step 1: 识别 {year} 年极值点 (threshold={threshold}) ...")
        t1 = time.time()
        extrema_count = zigzag.process_year_extrema(year, threshold, symbols)
        if verbose:
            logger.info(f"[WaveChanL1] Step 1 完成: {extrema_count} 只股票，耗时 {time.time()-t1:.0f}s")

        # Step 2: 趋势判断
        if verbose:
            logger.info(f"[WaveChanL1] Step 2: 判断 {year} 年趋势 (lookback={lookback}) ...")
        t2 = time.time()
        trend_count = trend.process_year_trend(year, lookback, symbols)
        if verbose:
            logger.info(f"[WaveChanL1] Step 2 完成: {trend_count} 只股票，耗时 {time.time()-t2:.0f}s")

        elapsed = time.time() - t0

        # 获取实际处理的 symbols
        if symbols is None:
            extrema_dir = self.base_path / f"extrema_year={year}"
            actual_symbols = [p.stem for p in extrema_dir.glob("*.parquet")] if extrema_dir.exists() else []
        else:
            actual_symbols = symbols

        result = {
            "year": year,
            "extrema_count": extrema_count,
            "trend_count": trend_count,
            "symbols": actual_symbols,
            "threshold": threshold,
            "lookback": lookback,
            "elapsed_seconds": round(elapsed, 1),
        }
        if verbose:
            logger.info(f"[WaveChanL1] {year} 年 L1 构建完成，耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

        return result

    def rebuild_year(
        self,
        year: int,
        symbols: Optional[List[str]] = None,
        threshold: float = 0.05,
        lookback: int = 5,
    ) -> Dict:
        """
        重建某年 L1（先删后建）
        """
        logger.warning(f"[WaveChanL1] 重建 {year} 年 L1（删除旧数据）...")
        self._remove_year_data(year)
        return self.build_year(year, symbols, threshold, lookback)

    def _remove_year_data(self, year: int) -> None:
        """删除某年的所有 L1 数据"""
        for subdir in [f"extrema_year={year}", f"trend_year={year}"]:
            d = self.base_path / subdir
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"[WaveChanL1] 已删除: {d}")

    # ============================================================
    # 增量更新
    # ============================================================

    def daily_increment(
        self,
        date: str,
        daily_df: pd.DataFrame,
        threshold: float = 0.05,
        lookback: int = 5,
    ) -> None:
        """
        每日增量更新 L1

        参数：
            date: YYYY-MM-DD，当前交易日
            daily_df: 当日日线数据（必须包含 symbol, date, open, high, low, close, volume）
            threshold: ZigZag 回撤阈值
            lookback: 趋势判断回看极值数
        """
        import time
        t0 = time.time()

        dt = pd.to_datetime(date)
        year = dt.year

        logger.info(f"[WaveChanL1] 增量更新: {date} ...")

        # 目前增量逻辑与 build_year 相同（全量重算指定 symbol）
        # 后续可优化为 append 模式
        symbols = daily_df["symbol"].unique().tolist()

        extrema_count = zigzag.process_year_extrema(year, threshold, symbols)
        trend_count = trend.process_year_trend(year, lookback, symbols)

        logger.info(f"[WaveChanL1] 增量更新完成: {extrema_count} extrema, {trend_count} trends, 耗时 {time.time()-t0:.1f}s")

    # ============================================================
    # 查询
    # ============================================================

    def get_extrema(
        self,
        symbol: str,
        year: int,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取某股票某年的极值点

        参数：
            symbol: 股票代码
            year: 年份
            columns: 指定返回列，None = 全部

        返回：
            DataFrame 或 空 DataFrame（文件不存在时）
        """
        path = extrema_path(symbol, year)
        if not path.exists():
            logger.debug(f"[WaveChanL1] 极值文件不存在: {path}")
            return pd.DataFrame(columns=zigzag.EXTREMA_COLS)

        cols = columns or zigzag.EXTREMA_COLS
        return pd.read_parquet(path, columns=cols)

    def get_trend(
        self,
        symbol: str,
        year: int,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取某股票某年的趋势序列

        参数：
            symbol: 股票代码
            year: 年份
            columns: 指定返回列，None = 全部

        返回：
            DataFrame 或 空 DataFrame（文件不存在时）
        """
        path = trend_path(symbol, year)
        if not path.exists():
            logger.debug(f"[WaveChanL1] 趋势文件不存在: {path}")
            return pd.DataFrame(columns=trend.TREND_COLS)

        cols = columns or trend.TREND_COLS
        return pd.read_parquet(path, columns=cols)

    def get_current_trend(
        self,
        symbol: str,
        year: int,
        lookback: int = 5,
    ) -> str:
        """
        获取某股票当前的实时趋势（基于最新极值点）

        返回：
            'UP' / 'DOWN' / 'NEUTRAL'
        """
        extrema = self.get_extrema(symbol, year)
        if extrema.empty:
            return "NEUTRAL"
        return trend.determine_trend(extrema, lookback=lookback)

    def get_latest_extrema(
        self,
        symbol: str,
        year: int,
    ) -> Optional[Dict]:
        """
        获取某股票最新的极值点信息
        """
        extrema = self.get_extrema(symbol, year)
        if extrema.empty:
            return None
        latest = extrema.iloc[-1]
        return {
            "symbol": latest["symbol"],
            "date": latest["date"],
            "price": latest["price"],
            "type": latest["type"],
            "is_major": latest["is_major"],
        }

    def query(
        self,
        symbols: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        data_type: str = "all",
    ) -> Dict[str, Dict]:
        """
        批量查询 L1 数据

        参数：
            symbols: 股票列表，None = 全部已有
            years: 年份列表，None = 全部已有
            data_type: 'extrema' | 'trend' | 'all'

        返回：
            dict: {symbol: {'extrema': DataFrame, 'trend': DataFrame, 'year': int}}
        """
        if years is None:
            years = list_available_years(data_type="extrema")
        if not years:
            return {}

        if symbols is None:
            # 收集所有已有 symbol
            all_symbols = set()
            for yr in years:
                extrema_dir = self.base_path / f"extrema_year={yr}"
                if extrema_dir.exists():
                    all_symbols.update(p.stem for p in extrema_dir.glob("*.parquet"))
            symbols = sorted(all_symbols)

        result = {}
        for sym in symbols:
            sym_data = {"year": years}
            if data_type in ("extrema", "all"):
                dfs = [self.get_extrema(sym, yr) for yr in years]
                sym_data["extrema"] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if data_type in ("trend", "all"):
                dfs = [self.get_trend(sym, yr) for yr in years]
                sym_data["trend"] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            result[sym] = sym_data

        return result

    def status(self) -> Dict:
        """
        返回 L1 缓存状态
        """
        extrema_years = list_available_years("extrema")
        trend_years = list_available_years("trend")

        total_extrema_files = 0
        total_trend_files = 0
        total_extrema_size = 0
        total_trend_size = 0

        for yr in extrema_years:
            d = self.base_path / f"extrema_year={yr}"
            if d.exists():
                files = list(d.glob("*.parquet"))
                total_extrema_files += len(files)
                total_extrema_size += sum(f.stat().st_size for f in files)

        for yr in trend_years:
            d = self.base_path / f"trend_year={yr}"
            if d.exists():
                files = list(d.glob("*.parquet"))
                total_trend_files += len(files)
                total_trend_size += sum(f.stat().st_size for f in files)

        return {
            "base_path": str(self.base_path),
            "extrema_years": extrema_years,
            "trend_years": trend_years,
            "total_extrema_files": total_extrema_files,
            "total_trend_files": total_trend_files,
            "total_extrema_size_mb": round(total_extrema_size / 1024 / 1024, 2),
            "total_trend_size_mb": round(total_trend_size / 1024 / 1024, 2),
        }
