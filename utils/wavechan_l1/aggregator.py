"""
WaveChan L1 - 周线K线聚合模块
从日线数据聚合生成周线K线
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ._path import weekly_klines_path, daily_data_path, ensure_dirs

logger = logging.getLogger(__name__)


def aggregate_daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日线数据聚合为周线K线

    参数：
        daily_df: 必须包含 date(str), symbol, open, high, low, close, volume, amount

    返回：
        周线 DataFrame，列：
        date, symbol, open, high, low, close, volume, amount,
        change_pct, upper_shadow, lower_shadow, body_size
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 星期几（0=周一, 4=周五）
    df["week_id"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    # 用 year + week_id 作为周的唯一标识
    df["year_week"] = df["year"].astype(str) + "_" + df["week_id"].astype(str).str.zfill(2)

    # 按 symbol + 周 聚合
    agg = (
        df.groupby(["symbol", "year_week"], group_keys=False)
        .agg(
            date=("date", "max"),
            year=("year", "first"),
            week_id=("week_id", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        .reset_index()
    )

    agg = agg.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 过滤不足3个交易日的周（数据不完整）
    count_per_week = df.groupby(["symbol", "year_week"]).size().reset_index(name="day_count")
    agg = agg.merge(count_per_week, on=["symbol", "year_week"])
    agg = agg[agg["day_count"] >= 3].drop(columns=["year_week", "year", "week_id", "day_count"])

    # 衍生字段
    agg["change_pct"] = agg.groupby("symbol")["close"].pct_change() * 100
    agg["upper_shadow"] = agg["high"] - agg[["open", "close"]].max(axis=1)
    agg["lower_shadow"] = agg[["open", "close"]].min(axis=1) - agg["low"]
    agg["body_size"] = (agg["close"] - agg["open"]).abs()

    # 重命名 date 保持 timestamp
    agg["date"] = pd.to_datetime(agg["date"])

    return agg


def _read_daily_year(year: int, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """读取某年的日线数据（Parquet）"""
    dailypath = daily_data_path(year)
    if not dailypath.exists():
        logger.warning(f"日线数据不存在: {dailypath}")
        return pd.DataFrame()

    dataset = ds.dataset(str(dailypath), format="parquet")
    table = dataset.to_table()

    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    available = [c for c in cols if c in table.schema.names]
    table = table.select(available)

    df = table.to_pandas()
    if df.empty:
        return df

    if symbols:
        df = df[df["symbol"].isin(symbols)]

    return df


def aggregate_year_weekly(
    year: int,
    symbols: Optional[List[str]] = None,
    output_base: Optional[Path] = None,
) -> int:
    """
    聚合某年的日线数据为周线K线，按 symbol 写入独立 Parquet 文件。

    参数：
        year: 年份
        symbols: 要处理的股票列表（None = 全部）
        output_base: 输出根目录（默认用 _path 配置）

    返回：
        处理的股票数量
    """
    ensure_dirs(year)
    out_dir = weekly_klines_path("", year).parent

    # 读取日线数据
    logger.info(f"读取 {year} 年日线数据 ...")
    df = _read_daily_year(year, symbols)
    if df.empty:
        logger.warning(f"{year} 年无日线数据")
        return 0

    symbols_processed = df["symbol"].unique()
    logger.info(f"  {year} 年共 {len(symbols_processed)} 只股票，{len(df):,} 行日线")

    # 聚合
    weekly = aggregate_daily_to_weekly(df)
    logger.info(f"  聚合得到 {len(weekly):,} 行周线K线")

    # 按 symbol 写入
    written = 0
    for sym in symbols_processed:
        w = weekly[weekly["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if w.empty:
            continue

        out_path = weekly_klines_path(sym, year)
        w.to_parquet(out_path, index=False, engine="pyarrow")
        written += 1

    logger.info(f"  写入 {written} 只股票的周线K线到 {out_dir}")
    return written


def aggregate_multi_year_weekly(
    years: List[int],
    symbols: Optional[List[str]] = None,
) -> dict:
    """
    聚合多年日线数据为周线K线（不跨年合并，只是批量处理多年）
    用于预计算 lookback 数据

    返回：{year: count}
    """
    results = {}
    for yr in sorted(years):
        results[yr] = aggregate_year_weekly(yr, symbols)
    return results
