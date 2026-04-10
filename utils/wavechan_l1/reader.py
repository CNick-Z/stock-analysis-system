"""
WaveChan L1 - 读取接口模块
提供对 L1 缓存的高效读取
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ._path import (
    weekly_klines_path,
    weekly_bi_path,
    wave_labels_path,
    symbols_index_path,
    BASE_PATH,
)

logger = logging.getLogger(__name__)


def read_weekly_klines(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    """读取某只股票跨多年的周线K线"""
    dfs = []
    for yr in range(start_year, end_year + 1):
        p = weekly_klines_path(symbol, yr)
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
    # 去重
    if "date" in result.columns:
        result = result.drop_duplicates(subset=["date"], keep="last")
    return result


def read_weekly_bi(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    """读取某只股票跨多年的周线笔"""
    dfs = []
    for yr in range(start_year, end_year + 1):
        p = weekly_bi_path(symbol, yr)
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    # 去重（按 bi_index，保持最后）
    result = result.drop_duplicates(subset=["symbol", "bi_index"], keep="last")
    return result.sort_values("end_date").reset_index(drop=True)


def read_wave_labels(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    """读取某只股票跨多年的大浪标注"""
    dfs = []
    for yr in range(start_year, end_year + 1):
        p = wave_labels_path(symbol, yr)
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)


def read_symbol_l1(symbol: str, year: int) -> dict:
    """读取单只股票单年的完整 L1 数据"""
    return {
        "klines": _read(weekly_klines_path(symbol, year)),
        "bi": _read(weekly_bi_path(symbol, year)),
        "labels": _read(wave_labels_path(symbol, year)),
    }


def read_major_points(symbol: str, years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    获取某股票的所有主要拐点（用于画大级别浪）
    等价于：read_weekly_bi() → filter(is_major) → merge labels
    """
    if years is None:
        years = list_available_years()

    start_year = min(years)
    end_year = max(years)

    bi_df = read_weekly_bi(symbol, start_year, end_year)
    if bi_df.empty:
        return pd.DataFrame()

    major = bi_df[bi_df["is_major_high"] | bi_df["is_major_low"]].copy()
    if major.empty:
        return pd.DataFrame()

    # 合并 wave_label（取最新的）
    labels = read_wave_labels(symbol, start_year, end_year)
    if not labels.empty:
        # wave_label 在 labels 里，去掉 bi_index 可能重复
        latest_labels = labels.drop_duplicates(subset=["bi_index"], keep="last")
        major = major.merge(
            latest_labels[["bi_index", "wave_label", "trend_type"]],
            on="bi_index",
            how="left",
        )

    return major.sort_values("end_date").reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    """安全读取一个 Parquet 文件"""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def list_available_years() -> List[int]:
    """列出已有数据的年份"""
    from ._path import list_available_years as _list
    return _list()


def status() -> dict:
    """返回 L1 缓存状态"""
    info = {"years": {}}

    for yr in list_available_years():
        klines_dir = weekly_klines_path("", yr).parent
        bi_dir = weekly_bi_path("", yr).parent
        labels_dir = wave_labels_path("", yr).parent

        klines_files = list(klines_dir.glob("*.parquet")) if klines_dir.exists() else []
        bi_files = list(bi_dir.glob("*.parquet")) if bi_dir.exists() else []
        labels_files = list(labels_dir.glob("*.parquet")) if labels_dir.exists() else []

        # 估算大小
        def dir_size(d):
            if not d.exists():
                return 0.0
            return sum(f.stat().st_size for f in d.rglob("*.parquet")) / 1024 / 1024

        info["years"][yr] = {
            "symbols": len(klines_files),
            "klines_files": len(klines_files),
            "bi_files": len(bi_files),
            "labels_files": len(labels_files),
            "size_mb": round(dir_size(klines_dir) + dir_size(bi_dir) + dir_size(labels_dir), 1),
        }

    return info
