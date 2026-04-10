"""
WaveChan L1 - ZigZag 极值识别模块
===================================
识别局部显著极值点，类似 ZigZag 指标。

核心算法：
- 基于回撤阈值（threshold）识别显著极值
- 扫描价格序列，找到所有显著转折点
- 支持多symbol批量处理

存储：
- 输出到 /data/warehouse/wavechan_l1/extrema_year={year}/{symbol}.parquet
"""

from typing import Optional, Tuple, List
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ._path import extrema_path, ensure_dirs

logger = logging.getLogger(__name__)

# ============================================================
# Schema 定义
# ============================================================
EXTREMA_SCHEMA = {
    "symbol": "string",
    "date": "timestamp[ns]",
    "price": "float64",
    "type": "string",          # 'high' | 'low'
    "extrema_index": "int32",  # 序号（该symbol内全局唯一）
    "swing_high": "float64",  # 该极值所在波段的高点（用于计算回撤）
    "swing_low": "float64",   # 该极值所在波段的低点
    "retrace_pct": "float64", # 从前一个极值的回撤百分比
    "is_major": "bool",       # 是否为显著极值（创新高/新低）
}


# ============================================================
# 核心极值识别算法
# ============================================================

def identify_extrema(
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    识别局部显著极值点（类似 ZigZag 指标）

    参数：
        df: OHLCV 数据，必须包含 date, open, high, low, close 列
        threshold: 回撤阈值（默认 5%）
            - 只有当价格从波段高点/低点回撤超过此阈值时，才确认极值

    返回：
        DataFrame，列：
            - symbol: 股票代码
            - date: 极值日期
            - price: 极值价格
            - type: 'high'（局部高点）或 'low'（局部低点）
            - extrema_index: 序号
            - swing_high: 该波段高点
            - swing_low: 该波段低点
            - retrace_pct: 回撤百分比
            - is_major: 是否为显著极值
    """
    if len(df) < 3:
        return pd.DataFrame(columns=list(EXTREMA_SCHEMA.keys()))

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
    df = df.copy().sort_values("date").reset_index(drop=True)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    dates = pd.to_datetime(df["date"]).values

    extrema_list: List[dict] = []
    extrema_index = 0

    # 状态机状态
    # None -> ScanningUp -> ScanningDown -> ...
    state = None  # 'scan_up' | 'scan_down'
    last_extrema_idx = None
    last_extrema_price = None
    last_extrema_type = None

    # 波段高低价（动态维护）
    swing_high = highs[0]
    swing_high_idx = 0
    swing_low = lows[0]
    swing_low_idx = 0

    for i in range(1, len(highs)):
        price = closes[i]
        high = highs[i]
        low = lows[i]

        if state is None:
            # 初始化：找第一个显著高点或低点
            if high > highs[0]:
                state = "scan_up"
                swing_high = high
                swing_high_idx = i
            elif low < lows[0]:
                state = "scan_down"
                swing_low = low
                swing_low_idx = i
            continue

        if state == "scan_up":
            # 扫描上涨波段，寻找高点
            if high >= swing_high:
                swing_high = high
                swing_high_idx = i

            # 检查是否向下突破 threshold
            # 从 swing_high 回撤超过 threshold% → 确认高点
            if swing_high > 0:
                drawdown = (swing_high - low) / swing_high * 100
                # threshold 是小数（如 0.05），需要转百分比比较
                if drawdown >= threshold * 100:
                    # 确认局部高点
                    extrema_list.append({
                        "symbol": symbol,
                        "date": pd.Timestamp(dates[swing_high_idx]),
                        "price": float(swing_high),
                        "type": "high",
                        "extrema_index": extrema_index,
                        "swing_high": float(swing_high),
                        "swing_low": float(swing_low),
                        "retrace_pct": float((swing_high - lows[swing_high_idx]) / swing_high * 100)
                            if swing_high > 0 else 0.0,
                        "is_major": _is_major_point(closes, swing_high_idx, direction="high"),
                    })
                    extrema_index += 1

                    # 切换到 scan_down
                    state = "scan_down"
                    last_extrema_idx = swing_high_idx
                    last_extrema_price = swing_high
                    last_extrema_type = "high"
                    swing_low = low
                    swing_low_idx = i

        elif state == "scan_down":
            # 扫描下跌波段，寻找低点
            if low <= swing_low:
                swing_low = low
                swing_low_idx = i

            # 检查是否向上突破 threshold
            if swing_low > 0:
                rally = (high - swing_low) / swing_low * 100
                # threshold 是小数（如 0.05），需要转百分比比较
                if rally >= threshold * 100:
                    # 确认局部低点
                    extrema_list.append({
                        "symbol": symbol,
                        "date": pd.Timestamp(dates[swing_low_idx]),
                        "price": float(swing_low),
                        "type": "low",
                        "extrema_index": extrema_index,
                        "swing_high": float(swing_high),
                        "swing_low": float(swing_low),
                        "retrace_pct": float((highs[swing_low_idx] - swing_low) / swing_low * 100)
                            if swing_low > 0 else 0.0,
                        "is_major": _is_major_point(closes, swing_low_idx, direction="low"),
                    })
                    extrema_index += 1

                    # 切换到 scan_up
                    state = "scan_up"
                    last_extrema_idx = swing_low_idx
                    last_extrema_price = swing_low
                    last_extrema_type = "low"
                    swing_high = high
                    swing_high_idx = i

    # 如果最后还在 scan_up 且有有效高点，追加最后一个高点
    if state == "scan_up" and extrema_index > 0:
        last = extrema_list[-1]
        if last["type"] == "high":
            pass  # 最后一个高点已记录

    if not extrema_list:
        return pd.DataFrame(columns=list(EXTREMA_SCHEMA.keys()) + ["date"])

    result = pd.DataFrame(extrema_list)
    result["date"] = pd.to_datetime(result["date"])
    return result


def _is_major_point(closes: np.ndarray, idx: int, direction: str) -> bool:
    """
    判断某点是否为显著极值（创新高/新低）

    参数：
        closes: 收盘价数组
        idx: 当前索引
        direction: 'high' 或 'low'
    """
    if direction == "high":
        return closes[idx] == np.max(closes[:idx + 1])
    else:
        return closes[idx] == np.min(closes[:idx + 1])


# ============================================================
# 批量处理
# ============================================================

def identify_extrema_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    识别单只股票的极值点（统一入口）

    参数：
        symbol: 股票代码
        df: OHLCV 数据
        threshold: 回撤阈值

    返回：
        极值点 DataFrame
    """
    df = df.copy()
    df["symbol"] = symbol
    return identify_extrema(df, threshold=threshold)


def process_year_extrema(
    year: int,
    threshold: float = 0.05,
    symbols: Optional[List[str]] = None,
) -> int:
    """
    处理某年的极值数据（读取日线 → 识别极值 → 写入 Parquet）

    参数：
        year: 年份
        threshold: 回撤阈值
        symbols: 指定股票列表，None 则处理全部

    返回：
        处理成功的股票数量
    """
    from ..data_loader import DataLoader

    ensure_dirs(year)
    loader = DataLoader()

    if symbols is None:
        # 从日线数据目录获取所有 symbol
        daily_dir = _get_daily_data_dir(year)
        if daily_dir and daily_dir.exists():
            symbols = [p.stem for p in daily_dir.glob("*.parquet")]
        else:
            symbols = []

    success_count = 0
    for sym in symbols:
        try:
            # 读取多年日线数据（lookback 1年用于完整识别）
            start_year = max(year - 1, year)
            df = loader.load_daily(symbol=sym, start_year=start_year, end_year=year)
            if df is None or df.empty:
                continue

            # 识别极值
            extrema = identify_extrema(df, threshold=threshold)

            if extrema.empty:
                continue

            # 只保留当年数据（极值点可能在 lookback 年形成）
            extrema = extrema[extrema["date"].dt.year == year].copy()
            if extrema.empty:
                continue

            # 重新编号（当年序号从0开始）
            extrema["extrema_index"] = range(len(extrema))

            # 写入
            out_path = extrema_path(sym, year)
            extrema.to_parquet(out_path, index=False)
            success_count += 1

        except Exception as e:
            logger.warning(f"[zigzag] 处理 {sym} 失败: {e}")

    logger.info(f"[zigzag] {year} 年极值识别完成: {success_count} 只股票")
    return success_count


def _get_daily_data_dir(year: int):
    """获取日线数据目录（支持向上查找）"""
    from pathlib import Path
    DAILY_DATA_ROOT = Path("/root/.openclaw/workspace/data/warehouse")
    p = DAILY_DATA_ROOT / f"daily_data_year={year}"
    if p.exists():
        return p
    return None


# ============================================================
# Schema 导出
# ============================================================
EXTREMA_COLS = ["symbol", "date", "price", "type", "extrema_index",
                "swing_high", "swing_low", "retrace_pct", "is_major"]
