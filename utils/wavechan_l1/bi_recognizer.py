"""
WaveChan L1 - 周线笔识别模块 v3
使用 CZSC 在周线数据上识别笔（BI）

【核心修复 v3】is_major_high/low 用笔终点价格（end_price）的 running max/min
不再用 bars_raw 的 K 线高低来判断。
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from czsc import CZSC, RawBar, Freq, Direction
from ._path import weekly_klines_path, weekly_bi_path, ensure_dirs

logger = logging.getLogger(__name__)


def _build_raw_bars(klines_df: pd.DataFrame) -> List[RawBar]:
    """从周线K线 DataFrame 构建 RawBar 列表"""
    bars = []
    kl = klines_df.sort_values("date")
    for _, row in kl.iterrows():
        bar = RawBar(
            symbol=str(row["symbol"]),
            dt=pd.Timestamp(row["date"]).to_pydatetime(),
            freq=Freq.W,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            vol=float(row.get("volume", 0.0)),
            amount=float(row.get("amount", 0.0)),
        )
        bars.append(bar)
    return bars


def recognize_symbol_bi(
    symbol: str,
    klines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对单只股票的多年周线K线识别笔

    返回：
        bi_df，列：symbol, bi_index, start_date, end_date, start_price, end_price,
                  direction, power, length_weeks, is_major_high, is_major_low

    【修复 v3】is_major_high/low 用笔终点价格判断：
      - is_major_high = 当前笔终点价格 > 历史所有笔终点价格的最大值（仅针对 UP BI）
      - is_major_low  = 当前笔终点价格 < 历史所有笔终点价格的最小值（仅针对 DOWN BI）
    """

    if klines_df.empty:
        return pd.DataFrame()

    symbol_kl = klines_df[klines_df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    if len(symbol_kl) < 20:
        logger.warning(f"{symbol}: 周线不足 20 根，跳过 BI 识别")
        return pd.DataFrame()

    bars = _build_raw_bars(symbol_kl)
    c = CZSC(bars, max_bi_num=500)

    if not c.finished_bis:
        return pd.DataFrame()

    bi_records = []
    # 维护历史笔终点价格的 running max/min（全局，不分方向）
    # 用于判断"历史所有笔终点价格"的最大/最小值
    all_end_prices = []  # 所有已处理 BI 的终点价格（按时间顺序）

    for i, bi in enumerate(c.finished_bis):
        end_dt = pd.Timestamp(bi.fx_b.dt)
        end_price = bi.fx_b.fx
        direction = "up" if bi.direction == Direction.Up else "down"

        # ---- 【修复 v3】用笔终点价格判断 ----
        # 创新高 = 当前笔终点价格 > 历史所有笔终点价格的最大值
        # 创新低 = 当前笔终点价格 < 历史所有笔终点价格的最小值
        if all_end_prices:
            hist_max_price = max(all_end_prices)
            hist_min_price = min(all_end_prices)
            is_major_high = bool(end_price > hist_max_price)
            is_major_low = bool(end_price < hist_min_price)
        else:
            # 第一笔：既不是创新高也不是创新低
            is_major_high = False
            is_major_low = False

        # 互斥：不可能同时是创新高和创新低
        if is_major_high and is_major_low:
            is_major_high = False
            is_major_low = False

        # 更新历史终点价格列表
        all_end_prices.append(end_price)

        # 计算力度
        power = abs(end_price - bi.fx_a.fx) / bi.fx_a.fx * 100 if bi.fx_a.fx != 0 else 0
        start_dt = pd.Timestamp(bi.fx_a.dt)
        length_weeks = max(1, (end_dt - start_dt).days // 7)

        bi_records.append({
            "symbol": symbol,
            "bi_index": i,
            "start_date": start_dt,
            "end_date": end_dt,
            "start_price": bi.fx_a.fx,
            "end_price": end_price,
            "direction": direction,
            "power": power,
            "length_weeks": length_weeks,
            "is_major_high": is_major_high,
            "is_major_low": is_major_low,
        })

    bi_df = pd.DataFrame(bi_records)
    return bi_df


def recognize_year_bi(
    year: int,
    symbols: Optional[List[str]] = None,
    lookback_years: int = 3,
) -> int:
    """
    识别某年所有股票的周线笔
    """
    ensure_dirs(year)

    years_to_read = list(range(max(2018, year - lookback_years), year + 1))
    all_klines = {}

    for yr in years_to_read:
        syms = list_symbols_for_year(yr) if symbols is None else symbols
        for sym in syms:
            p = weekly_klines_path(sym, yr)
            if p.exists():
                df = pd.read_parquet(p)
                if sym not in all_klines:
                    all_klines[sym] = []
                all_klines[sym].append(df)

    written = 0
    for sym, dfs in all_klines.items():
        combined = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        bi_df = recognize_symbol_bi(sym, combined)

        if bi_df.empty:
            continue

        # 只保留该年及之后结束的 BI
        bi_year = bi_df[bi_df["end_date"].dt.year >= year].copy()
        if bi_year.empty:
            continue

        out_path = weekly_bi_path(sym, year)
        bi_year.to_parquet(out_path, index=False, engine="pyarrow")
        written += 1

    logger.info(f"识别 {year} 年 BI 完成，写入 {written} 只股票")
    return written


def list_symbols_for_year(year: int) -> List[str]:
    """从周线K线目录读取某年有哪些股票"""
    klines_dir = weekly_klines_path("", year).parent
    if not klines_dir.exists():
        return []
    return [p.stem for p in klines_dir.glob("*.parquet")]
