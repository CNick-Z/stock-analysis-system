# simulator/shared.py
# =============================================================
"""
backtest.py / simulate.py 公共模块
=============================================================
策略注册表、数据加载公共函数，两个入口共用。

用法:
    from simulator.shared import load_strategy, load_data_for_strategy
"""
# =============================================================

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ── 策略注册表 ─────────────────────────────────────────────
STRATEGY_REGISTRY = {
    "v8": {
        "class": "ScoreV8Strategy",
        "path": "strategies.score.v8.strategy",
    },
    "wavechan": {
        "class": "WaveChanStrategy",
        "path": "strategies.wavechan.v3_l2_cache.wavechan_strategy",
    },
    "wavechan_v3": {
        "class": "WaveChanStrategy",
        "path": "strategies.wavechan.v3_l2_cache.wavechan_strategy",
    },
}

# ── 波浪 L2 cache 路径 ─────────────────────────────────
WAVECHAN_L2_CACHE = Path("/data/warehouse/wavechan/wavechan_cache")


def load_strategy(name: str):
    """动态加载策略实例"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}. 可用: {list(STRATEGY_REGISTRY.keys())}")
    cfg = STRATEGY_REGISTRY[name]
    mod = __import__(cfg["path"], fromlist=[cfg["class"]])
    return getattr(mod, cfg["class"])()


def add_next_open(df: pd.DataFrame) -> pd.DataFrame:
    """添加 next_open 列（次日开盘价）"""
    df = df.copy()
    df["next_open"] = df.groupby("symbol")["open"].shift(-1)
    return df


def load_wavechan_cache(years: List[int]) -> pd.DataFrame:
    """
    从 L1/L2 cache 加载波浪信号数据。

    L1（冷数据）：l1_cold_year=YYYY/data.parquet（年度粒度）
    L2（热数据）：l2_hot_year=YYYY_month=MM/data.parquet（月度粒度）

    返回 DataFrame 包含：
        date, symbol, has_signal, total_score, signal_type,
        signal_status, wave_trend, wave_state, stop_loss 等字段。
    """
    parts = []
    for year in years:
        # L1 冷数据（优先）
        l1_path = WAVECHAN_L2_CACHE / f"l1_cold_year={year}" / "data.parquet"
        if l1_path.exists():
            df = pd.read_parquet(l1_path)
            parts.append(df)
            logger.info(f"  L1 cache loaded: {year}  {len(df):,} 行")
            continue

        # L2 热数据
        l2_dirs = list(WAVECHAN_L2_CACHE.glob(f"l2_hot_year={year}_month=*"))
        if l2_dirs:
            dfs = [
                pd.read_parquet(d / "data.parquet")
                for d in l2_dirs
                if (d / "data.parquet").exists()
            ]
            if dfs:
                parts.append(pd.concat(dfs, ignore_index=True))
                logger.info(f"  L2 cache loaded: {year}  {sum(len(d) for d in dfs):,} 行")
                continue

        logger.warning(f"  波浪缓存不存在: {year}")

    if not parts:
        return pd.DataFrame()

    cache_df = pd.concat(parts, ignore_index=True)
    cols = [
        "date", "symbol", "has_signal", "total_score",
        "signal_type", "signal_status", "wave_trend",
        "wave_state", "stop_loss",
    ]
    existing = [c for c in cols if c in cache_df.columns]
    cache_df = cache_df[existing].drop_duplicates(["date", "symbol"])
    return cache_df
