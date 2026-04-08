#!/usr/bin/env python3
"""
market_regime.py — 大盘牛熊过滤器
===================================
CSI300 数据源，计算 MA / RSI14 / MACD，判断市场状态并输出仓位上限。

三档仓位（永远不超过 100%，不融资）：
  BEAR（熊市）     → 30%（默认，可配置）
  NEUTRAL（震荡） → 70%（默认，可配置）
  BULL（牛市）    → 100%

信号判断（需连续 N 日确认）：
  1. BEAR：收盘 < MA20 AND RSI14 < 40
  2. NEUTRAL（RSI预警）：RSI14 < 50
  3. NEUTRAL（趋势确认）：MA5 < MA10 < MA20 AND MACD直方图 < 0
  4. BULL（退出熊市）：收盘 > MA20 AND RSI14 > 50 AND MA5 > MA10 > MA20
  5. 兜底：BULL

快速止损触发器：
  RSI14 < 40 AND 当日跌幅 > 2% → 强制 BEAR（不受 confirm_days 限制）
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """
    大盘牛熊过滤器

    Args:
        index_path: CSI300 parquet 路径
        confirm_days: 连续确认天数（默认1）
        neutral_position: 震荡仓位（默认0.70）
        bear_position: 熊市仓位（默认0.30）
        bull_position: 牛市仓位（默认1.00）
        regime_persist_days: 状态锁：regime 切换后最少保持天数（默认1）
    """

    def __init__(
        self,
        index_path: str = "/data/warehouse/indices/CSI300.parquet",
        confirm_days: int = 1,
        neutral_position: float = 0.70,
        bear_position: float = 0.30,
        bull_position: float = 1.00,
        regime_persist_days: int = 1,
    ):
        # 参数校验
        assert 0.0 < neutral_position <= 1.0, \
            f"neutral_position 必须介于 (0, 1]，当前值: {neutral_position}"
        assert 0.0 < bear_position <= neutral_position, \
            f"bear_position 必须介于 (0, neutral_position] 之间，当前值: {bear_position} > {neutral_position}"
        assert bull_position == 1.00, \
            f"bull_position 必须为 1.00（满仓），当前值: {bull_position}"
        assert 1 <= regime_persist_days <= 10, \
            f"regime_persist_days 必须介于 [1, 10]，当前值: {regime_persist_days}"

        self.index_path = index_path
        self.confirm_days = confirm_days
        self.neutral_position = neutral_position
        self.bear_position = bear_position
        self.bull_position = bull_position
        self.regime_persist_days = regime_persist_days

        # 状态锁相关状态
        self._last_confirmed_regime: Optional[str] = None

        # 预计算的 DataFrame（由 prepare() 填充）
        self._df: Optional[pd.DataFrame] = None

    # --------------------------------------------------------
    # 公开接口
    # --------------------------------------------------------

    def get_regime(self, date: str) -> dict:
        """
        查询某日期的市场状态（应用状态锁后）

        状态锁逻辑：直接使用 DataFrame 预计算的 consecutive_days，
        无需维护内部状态机。当 consecutive_days >= regime_persist_days 时
        确认为该 regime，否则保持上一个确认的 regime。

        Args:
            date: 查询日期（YYYY-MM-DD）

        Returns:
            dict with regime, position_limit, signal, consecutive_days, etc.
        """
        if self._df is None:
            raise ValueError("请先调用 prepare(start_date, end_date) 预计算数据")

        # 转换为与 index 匹配的格式
        dt = pd.to_datetime(date)
        row = self._df[self._df["date"] == dt]

        # 如果精确日期不存在（周末/节假日），取最近的前一个交易日
        if row.empty:
            candidates = self._df[self._df["date"] <= dt]
            if candidates.empty:
                raise ValueError(f"日期 {date} 早于数据起始日期 {self._df['date'].min().date()}")
            row = candidates
            date = str(self._df[self._df["date"] <= dt].iloc[-1]["date"].strftime("%Y-%m-%d"))
            dt = pd.to_datetime(date)
            row = self._df[self._df["date"] == dt]

        r = row.iloc[0]
        raw_regime = r["regime"]
        consecutive_days = int(r["consecutive_days"])

        # 状态锁：consecutive_days >= persist 时才确认为新 regime
        if consecutive_days >= self.regime_persist_days:
            confirmed_regime = raw_regime
        else:
            confirmed_regime = self._last_confirmed_regime

        # 更新上一个确认的 regime
        if confirmed_regime != self._last_confirmed_regime:
            self._last_confirmed_regime = confirmed_regime

        position_limit, signal = self._get_limit_and_signal(confirmed_regime, r)

        return {
            "date": date,
            "regime": confirmed_regime,
            "raw_regime": raw_regime,
            "csi300_close": r["close"],
            "ma20": r["ma20"],
            "rsi14": r["rsi14"],
            "position_limit": position_limit,
            "signal": signal,
            "consecutive_days": consecutive_days,
            "regime_persist_days": self.regime_persist_days,
        }

    def _get_limit_and_signal(self, regime: str, r: pd.Series) -> tuple:
        """根据 regime 返回 position_limit 和 signal"""
        # 安全兜底
        if regime is None:
            return self.neutral_position, "no_regime_data"
        n = self.confirm_days
        if regime == "BEAR":
            return self.bear_position, "bear_confirmed"
        elif regime == "NEUTRAL":
            # 判断是哪种 NEUTRAL
            if r["neutral_rsi_consec"] >= n:
                return self.neutral_position, "rsi_warning"
            else:
                return self.neutral_position, "trend_bear_confirmed"
        else:  # BULL
            if r["bull_consec"] >= n:
                return self.bull_position, "bull_confirmed"
            else:
                return self.bull_position, "bull"

    def prepare(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        预计算区间内所有交易日的状态

        Args:
            start_date: 区间开始（YYYY-MM-DD）
            end_date: 区间结束（YYYY-MM-DD）

        Returns:
            DataFrame: date, regime, csi300_close, ma20, ma10, ma5, rsi14,
                      macd, macd_signal, macd_hist, position_limit, signal,
                      consecutive_days, bear_cond, neutral_rsi_cond, neutral_trend_cond,
                      bull_cond
        """
        logger.info(f"MarketRegimeFilter: 加载数据 {start_date} ~ {end_date}")

        # 重置状态锁状态
        self._last_confirmed_regime = None

        # 1. 加载原始数据
        df = pd.read_parquet(self.index_path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # 取前后足够宽的窗口用于指标计算（向前多取 confirm_days + 30 天）
        warmup_days = self.confirm_days + 30
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
        end_dt = pd.to_datetime(end_date)

        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError(f"CSI300 数据为空: {start_dt} ~ {end_dt}")

        logger.info(f"  原始数据: {len(df)} 行, {df['date'].min().date()} ~ {df['date'].max().date()}")

        # 2. 计算技术指标
        close = df["close"]

        # MA
        df["ma5"] = close.rolling(5, min_periods=5).mean()
        df["ma10"] = close.rolling(10, min_periods=10).mean()
        df["ma20"] = close.rolling(20, min_periods=20).mean()

        # RSI14（Wilders EWM RSI）
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi14"] = (100 - 100 / (1 + rs)).clip(upper=100)

        # MACD（标准 12, 26, 9）
        ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal_line
        df["macd_hist"] = macd_line - macd_signal_line

        # ── 快速止损触发器：当日跌幅 > 2% ──
        # 计算当日涨跌（从前一日收盘到当日收盘的跌幅）
        prev_close = close.shift(1)
        df["pct_change"] = (close - prev_close) / prev_close * 100
        # 快速止损条件：RSI < 40 且 当日跌幅 > 2%
        df["fast_stop_loss"] = (df["rsi14"] < 40) & (df["pct_change"] < -2.0)

        # 3. 计算每日信号条件
        df["close_above_ma20"] = df["close"] > df["ma20"]
        df["rsi14_below_40"] = df["rsi14"] < 40
        df["rsi14_below_50"] = df["rsi14"] < 50
        df["rsi14_above_50"] = df["rsi14"] > 50
        df["ma5_above_ma10"] = df["ma5"] > df["ma10"]
        df["ma10_above_ma20"] = df["ma10"] > df["ma20"]
        df["ma5_below_ma10"] = df["ma5"] < df["ma10"]
        df["ma10_below_ma20"] = df["ma10"] < df["ma20"]
        df["macd_hist_negative"] = df["macd_hist"] < 0

        # BEAR 条件：收盘 < MA20 AND RSI14 < 40
        df["bear_cond"] = (~df["close_above_ma20"]) & df["rsi14_below_40"]

        # NEUTRAL RSI 预警条件：RSI14 < 50
        df["neutral_rsi_cond"] = df["rsi14_below_50"]

        # NEUTRAL 趋势确认条件：MA5 < MA10 < MA20 AND MACD直方图 < 0
        df["neutral_trend_cond"] = (
            df["ma5_below_ma10"] & df["ma10_below_ma20"] & df["macd_hist_negative"]
        )

        # BULL 退出条件：收盘 > MA20 AND RSI14 > 50 AND MA5 > MA10 > MA20
        df["bull_cond"] = (
            df["close_above_ma20"] & df["rsi14_above_50"] & df["ma5_above_ma10"] & df["ma10_above_ma20"]
        )

        # 4. 连续确认天数
        df["bear_consec"] = self._consecutive_sum(df["bear_cond"])
        df["neutral_rsi_consec"] = self._consecutive_sum(df["neutral_rsi_cond"])
        df["neutral_trend_consec"] = self._consecutive_sum(df["neutral_trend_cond"])
        df["bull_consec"] = self._consecutive_sum(df["bull_cond"])

        # 5. 判断 regime（按优先级）
        n = self.confirm_days

        def assign_regime(row):
            # 快速止损：RSI<40 且当日跌幅>2% → 强制 BEAR（不受 confirm_days 限制）
            if row["fast_stop_loss"]:
                return "BEAR"
            if row["bear_consec"] >= n:
                return "BEAR"
            elif row["neutral_rsi_consec"] >= n:
                return "NEUTRAL"
            elif row["neutral_trend_consec"] >= n:
                return "NEUTRAL"
            elif row["bull_consec"] >= n:
                return "BULL"
            else:
                return "NEUTRAL"

        df["regime"] = df.apply(assign_regime, axis=1)

        # 6. 分配 position_limit 和 signal（原始，不含状态锁）
        def assign_limit(row):
            if row["regime"] == "BEAR":
                return self.bear_position, "bear_confirmed", int(row["bear_consec"])
            elif row["regime"] == "NEUTRAL":
                # 判断是哪种 NEUTRAL
                if row["neutral_rsi_consec"] >= n:
                    return self.neutral_position, "rsi_warning", int(row["neutral_rsi_consec"])
                else:
                    return self.neutral_position, "trend_bear_confirmed", int(row["neutral_trend_consec"])
            else:  # BULL
                if row["bull_consec"] >= n:
                    return self.bull_position, "bull_confirmed", int(row["bull_consec"])
                else:
                    return self.bull_position, "bull", 0

        limits = df.apply(assign_limit, axis=1)
        df["position_limit"] = [x[0] for x in limits]
        df["signal"] = [x[1] for x in limits]
        df["consecutive_days"] = [x[2] for x in limits]

        # 7. 过滤到目标区间
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
        result = df[(df["date"] >= target_start) & (df["date"] <= target_end)].copy()
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")

        self._df = df  # 保存完整数据用于查询

        # 8. 打印分布
        regime_counts = result["regime"].value_counts()
        logger.info(f"  MarketRegimeFilter 准备完成: {len(result)} 个交易日")
        for regime, count in regime_counts.items():
            pct = count / len(result) * 100
            logger.info(f"    {regime}: {count} 天 ({pct:.1f}%)")

        return result

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------

    @staticmethod
    def _consecutive_sum(series: pd.Series) -> pd.Series:
        """
        计算每个位置向前连续满足条件的次数。

        例如：series = [F, F, T, T, T, F, T, T]
        结果    = [0, 0, 1, 2, 3, 0, 1, 2]
        """
        result = pd.Series(0, index=series.index)
        counter = 0
        for i in range(len(series)):
            if series.iloc[i]:
                counter += 1
            else:
                counter = 0
            result.iloc[i] = counter
        return result

    # --------------------------------------------------------
    # 便捷方法（用于 demo / 调试）
    # --------------------------------------------------------

    def regime_distribution(self, start_date: str, end_date: str) -> pd.DataFrame:
        """返回各窗口的 regime 分布（用于 demo 报告）"""
        result = self.prepare(start_date, end_date)
        counts = result["regime"].value_counts()
        total = len(result)
        dist = pd.DataFrame({
            "天数": counts,
            "占比": (counts / total * 100).round(1),
            "平均仓位上限": [
                result[result["regime"] == r]["position_limit"].mean() for r in counts.index
            ],
        })
        return dist
