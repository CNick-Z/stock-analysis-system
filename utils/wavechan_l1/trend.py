"""
WaveChan L1 - 趋势判断模块
============================
基于极值点序列判断当前趋势（UP / DOWN / NEUTRAL）

判断规则：
- UP:    高点抬高（higher_high）+ 低点抬高（higher_low）
- DOWN:  高点降低（lower_high）+ 低点降低（lower_low）
- NEUTRAL: 其他情况（震荡）
"""

from typing import Optional, List, Dict
import logging

import pandas as pd

from ._path import trend_path, ensure_dirs

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
TREND_UP = "UP"
TREND_DOWN = "DOWN"
TREND_NEUTRAL = "NEUTRAL"

TREND_SCHEMA = {
    "symbol": "string",
    "year": "int16",
    "date": "timestamp[ns]",
    "trend": "string",        # UP / DOWN / NEUTRAL
    "trend_index": "int32",   # 趋势序号（每切换一次 +1）
    "lookback_extrema": "int16",  # 判定趋势时回看的极值点数量
    "higher_high": "bool",     # 高点是否抬高
    "higher_low": "bool",      # 低点是否抬高
    "lower_high": "bool",      # 高点是否降低
    "lower_low": "bool",       # 低点是否降低
    "swing_high_price": "float64",  # 当前波段高点价格
    "swing_low_price": "float64",   # 当前波段低点价格
    "strength": "float64",     # 趋势强度（0~1，高点/低点变化的幅度比）
}


# ============================================================
# 核心趋势判断算法
# ============================================================

def determine_trend(
    extrema: pd.DataFrame,
    lookback: int = 5,
) -> str:
    """
    基于极值变化判断当前趋势

    参数：
        extrema: 极值点 DataFrame（来自 zigzag.py identify_extrema）
                  必须包含 date, price, type 列
        lookback: 回看最近 N 个极值点（默认 5）

    返回：
        'UP' / 'DOWN' / 'NEUTRAL'

    规则：
        UP    = 高点抬高 + 低点抬高
        DOWN  = 高点降低 + 低点降低
        其他   = NEUTRAL
    """
    if extrema is None or len(extrema) < 4:
        return TREND_NEUTRAL

    # 只看最近的 lookback 个极值
    recent = extrema.tail(lookback).copy()

    # 提取高点序列和低点序列
    highs = recent[recent["type"] == "high"]["price"].values
    lows = recent[recent["type"] == "low"]["price"].values

    if len(highs) < 2 or len(lows) < 2:
        return TREND_NEUTRAL

    # 判断方向
    higher_high = highs[-1] > highs[0]   # 最近高点 > 最旧高点
    higher_low = lows[-1] > lows[0]        # 最近低点 > 最旧低点
    lower_high = highs[-1] < highs[0]     # 最近高点 < 最旧高点
    lower_low = lows[-1] < lows[0]         # 最近低点 < 最旧低点

    if higher_high and higher_low:
        return TREND_UP
    elif lower_high and lower_low:
        return TREND_DOWN
    else:
        return TREND_NEUTRAL


def determine_trend_detailed(
    extrema: pd.DataFrame,
    lookback: int = 5,
) -> Dict:
    """
    详细趋势判断，返回更多信息

    返回：
        dict: {
            'trend': 'UP' / 'DOWN' / 'NEUTRAL',
            'trend_index': int,
            'higher_high': bool,
            'higher_low': bool,
            'lower_high': bool,
            'lower_low': bool,
            'swing_high_price': float,
            'swing_low_price': float,
            'strength': float,
            'last_extrema_type': 'high' / 'low',
        }
    """
    if extrema is None or len(extrema) < 4:
        return {
            "trend": TREND_NEUTRAL,
            "trend_index": 0,
            "higher_high": False,
            "higher_low": False,
            "lower_high": False,
            "lower_low": False,
            "swing_high_price": 0.0,
            "swing_low_price": 0.0,
            "strength": 0.0,
            "last_extrema_type": None,
        }

    recent = extrema.tail(lookback).copy()

    highs = recent[recent["type"] == "high"]["price"].values
    lows = recent[recent["type"] == "low"]["price"].values

    higher_high = bool(highs[-1] > highs[0]) if len(highs) >= 2 else False
    higher_low = bool(lows[-1] > lows[0]) if len(lows) >= 2 else False
    lower_high = bool(highs[-1] < highs[0]) if len(highs) >= 2 else False
    lower_low = bool(lows[-1] < lows[0]) if len(lows) >= 2 else False

    if higher_high and higher_low:
        trend = TREND_UP
    elif lower_high and lower_low:
        trend = TREND_DOWN
    else:
        trend = TREND_NEUTRAL

    # 趋势强度：高点和低点变化的相对幅度
    swing_high_price = float(highs[-1]) if len(highs) > 0 else 0.0
    swing_low_price = float(lows[-1]) if len(lows) > 0 else 0.0

    strength = 0.0
    if len(highs) >= 2 and len(lows) >= 2:
        high_change = abs(highs[-1] - highs[0]) / (highs[0] + 1e-9) * 100
        low_change = abs(lows[-1] - lows[0]) / (lows[0] + 1e-9) * 100
        strength = min(1.0, (high_change + low_change) / 20.0)  # 归一化到 0~1

    last_extrema = extrema.iloc[-1] if len(extrema) > 0 else None
    last_extrema_type = last_extrema["type"] if last_extrema is not None else None

    return {
        "trend": trend,
        "trend_index": 0,  # 后续由 process_year_trend 填充
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "swing_high_price": swing_high_price,
        "swing_low_price": swing_low_price,
        "strength": round(strength, 4),
        "last_extrema_type": last_extrema_type,
    }


# ============================================================
# 趋势序列生成（带序号）
# ============================================================

def build_trend_series(
    extrema: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    生成完整趋势序列 DataFrame

    每个极值点对应一条记录，包含当时的趋势判断

    返回：
        DataFrame，列见 TREND_SCHEMA
    """
    if extrema is None or extrema.empty:
        return pd.DataFrame(columns=list(TREND_SCHEMA.keys()) + ["date"])

    extrema = extrema.copy().sort_values("date").reset_index(drop=True)
    symbol = extrema["symbol"].iloc[0]
    year = extrema["date"].dt.year.iloc[0]

    trend_records: List[Dict] = []
    trend_index = 0
    prev_trend = None

    # 需要至少 lookback 个极值点才能开始判断
    for i in range(lookback - 1, len(extrema)):
        window = extrema.iloc[:i + 1]
        detail = determine_trend_detailed(window, lookback=lookback)
        trend = detail["trend"]

        # 趋势切换时更新序号
        if trend != prev_trend:
            if prev_trend is not None:
                trend_index += 1
            prev_trend = trend

        detail["symbol"] = symbol
        detail["year"] = int(year)
        detail["date"] = extrema.iloc[i]["date"]
        detail["trend_index"] = trend_index
        detail["lookback_extrema"] = lookback

        trend_records.append(detail)

    if not trend_records:
        return pd.DataFrame(columns=list(TREND_SCHEMA.keys()) + ["date"])

    result = pd.DataFrame(trend_records)
    # 整理列顺序
    cols = ["symbol", "year", "date", "trend", "trend_index",
            "lookback_extrema", "higher_high", "higher_low",
            "lower_high", "lower_low", "swing_high_price",
            "swing_low_price", "strength", "last_extrema_type"]
    result = result[[c for c in cols if c in result.columns]]
    return result


# ============================================================
# 批量处理
# ============================================================

def process_year_trend(
    year: int,
    lookback: int = 5,
    symbols: Optional[List[str]] = None,
) -> int:
    """
    处理某年的趋势数据（读取极值 → 判断趋势 → 写入 Parquet）

    参数：
        year: 年份
        lookback: 回看极值点数
        symbols: 指定股票列表，None 则处理全部

    返回：
        处理成功的股票数量
    """
    from .zigzag import extrema_path, EXTREMA_COLS

    ensure_dirs(year)

    if symbols is None:
        # 从极值数据目录获取所有 symbol
        extrema_dir = extrema_path("", year).parent
        if extrema_dir.exists():
            symbols = [p.stem for p in extrema_dir.glob("*.parquet")]
        else:
            symbols = []

    success_count = 0
    for sym in symbols:
        try:
            path = extrema_path(sym, year)
            if not path.exists():
                continue

            extrema = pd.read_parquet(path, columns=EXTREMA_COLS)
            if extrema.empty:
                continue

            trend_df = build_trend_series(extrema, lookback=lookback)
            if trend_df.empty:
                continue

            # 写入
            out_path = trend_path(sym, year)
            trend_df.to_parquet(out_path, index=False)
            success_count += 1

        except Exception as e:
            logger.warning(f"[trend] 处理 {sym} 失败: {e}")

    logger.info(f"[trend] {year} 年趋势判断完成: {success_count} 只股票")
    return success_count


# ============================================================
# Schema 导出
# ============================================================
TREND_COLS = ["symbol", "year", "date", "trend", "trend_index",
              "higher_high", "higher_low", "lower_high", "lower_low",
              "swing_high_price", "swing_low_price", "strength",
              "last_extrema_type"]
