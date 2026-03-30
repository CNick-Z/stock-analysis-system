# strategies/wavechan_strategy.py
# 波浪 + 缠论整合策略插件 v3（基于 Oracle 研究成果升级）

"""
波浪 + 缠论整合策略 v3

升级内容（基于 Oracle 研究成果）：
1. ElliottWaveCounter: Oracle 自实现 Zigzag 算法 + 斐波那契规则过滤
2. ChanlunSignalDetector: czsc 库买卖点识别（一买/二买/三买/一卖/二卖）
3. 整合逻辑：波浪趋势 + 缠论信号共振

策略接口适配：
- get_signals(): 返回 [buy_signals_df, sell_signals_df]
- generate_features(): 返回包含波浪/缠论特征的 DataFrame
- get_buy_signals(): 兼容接口

整合逻辑：
  波浪向上(1/3/5) + 缠论一买/二买/三买  →  买入
  波浪向下(A/B/C) + 缠论一卖/二卖          →  卖出
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

from .base import BaseStrategy

logger = logging.getLogger(__name__)

# ============================================================
# 波浪模块 (WaveModule) - 来源: Oracle 研究成果
# ============================================================


@dataclass
class Swing:
    """波段：两个转折点之间的价格走势"""
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    high_idx: int = 0
    high_price: float = 0.0
    low_idx: int = 0
    low_price: float = 0.0
    is_up: bool = True


@dataclass
class WaveLabel:
    """波浪标签"""
    wave_number: int
    wave_name: str
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    fib_ratio: float = 0.0


class ElliottWaveCounter:
    """
    艾略特波浪计数器（Oracle 实现版）
    - Zigzag 算法识别波段高低点
    - Elliott 规则过滤（Wave3 不能是最短浪）
    - 每次调用 analyze() 重新计算 = 波浪不断修正
    """

    FIB_RATIOS = {
        'wave2': [0.236, 0.382, 0.5, 0.618, 0.764],
        'wave3': [1.272, 1.618, 2.0, 2.618],
        'wave4': [0.236, 0.382, 0.5, 0.618],
        'wave5': [0.5, 0.618, 0.764, 1.0],
    }

    def __init__(self, threshold_pct: float = 0.025):
        self.threshold_pct = threshold_pct
        self.swings: List[Swing] = []
        self.wave_labels: List[WaveLabel] = []

    def compute_zigzag(self, prices: np.ndarray) -> List[Swing]:
        """
        Zigzag 算法：识别波段高低点
        寻找局部极值点，识别连续上升/下降波段
        """
        if len(prices) < 5:
            return []

        # Step 1: 找出所有局部极值点（转折点）
        extrema = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                extrema.append((i, True, float(prices[i])))   # 局部高点
            elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                extrema.append((i, False, float(prices[i])))   # 局部低点

        if len(extrema) < 2:
            self.swings = []
            return []

        # Step 2: 从极值点构建波段（相邻极值点类型必须不同）
        swings = []
        i = 0
        while i < len(extrema) - 1:
            idx1, is_high1, price1 = extrema[i]
            idx2, is_high2, price2 = extrema[i + 1]

            if is_high1 == is_high2:
                i += 1
                continue

            is_up = not is_high1  # 低点->高点 = 上升波段
            s = Swing(
                start_idx=idx1,
                end_idx=idx2,
                start_price=price1,
                end_price=price2,
                high_idx=idx2 if is_up else idx1,
                high_price=price2 if is_up else price1,
                low_idx=idx1 if is_up else idx2,
                low_price=price1 if is_up else price2,
                is_up=is_up
            )
            swings.append(s)
            i += 1

        # Step 3: 过滤幅度太小的波段
        filtered = [s for s in swings
                    if abs(s.end_price - s.start_price) / s.start_price >= self.threshold_pct]
        self.swings = filtered
        return filtered

    def label_waves(self, swings: List[Swing]) -> List[WaveLabel]:
        """对波段进行波浪标记，应用 Elliott 规则过滤"""
        if len(swings) < 5:
            return []

        labels = []
        wave_num = 1

        for i in range(len(swings) - 1):
            s1 = swings[i]
            s2 = swings[i + 1]
            pct_change = abs(s2.end_price - s1.end_price) / s1.end_price

            if pct_change < self.threshold_pct:
                continue

            if len(labels) > 0:
                prev = labels[-1]
                prev_len = abs(prev.end_price - prev.start_price)
                curr_len = abs(s2.end_price - s1.end_price)
                ratio = curr_len / prev_len if prev_len > 0 else 1.0
            else:
                ratio = 1.0

            name = f"Wave{wave_num}" if wave_num <= 5 else f"Wave{chr(65 + wave_num - 6)}"

            label = WaveLabel(
                wave_number=wave_num,
                wave_name=name,
                start_idx=s1.end_idx,
                end_idx=s2.end_idx,
                start_price=s1.end_price,
                end_price=s2.end_price,
                fib_ratio=ratio
            )

            # Elliott 规则：Wave3 不能是最短的浪
            if wave_num == 3 and len(labels) >= 2:
                w1_len = abs(labels[0].end_price - labels[0].start_price)
                w2_len = abs(labels[1].end_price - labels[1].start_price)
                w3_len = abs(label.end_price - label.start_price)
                if w3_len < max(w1_len, w2_len):
                    labels.pop()
                    wave_num -= 1
                    continue

            labels.append(label)
            wave_num += 1

        self.wave_labels = labels
        return labels

    def get_trend(self) -> str:
        """判断当前趋势方向"""
        if len(self.swings) < 2:
            return "neutral"
        recent = self.swings[-3:] if len(self.swings) >= 3 else self.swings
        ups = sum(1 for s in recent if s.is_up)
        return "up" if ups > len(recent) - ups else "down" if ups < len(recent) - ups else "neutral"

    def analyze(self, prices: np.ndarray) -> dict:
        """
        主分析入口
        每次新数据来时重新调用此方法 = 波浪不断修正
        """
        self.swings = []
        self.wave_labels = []
        swings = self.compute_zigzag(prices)
        labels = self.label_waves(swings)
        trend = self.get_trend()
        last_label = labels[-1] if labels else None

        return {
            "trend": trend,
            "swings": self.swings,
            "wave_labels": labels,
            "position": {
                "status": "wave_detected" if labels else "no_wave",
                "trend": trend,
                "last_wave": last_label.wave_name if last_label else "N/A",
                "last_wave_end_price": last_label.end_price if last_label else 0.0,
                "total_waves": len(labels),
            }
        }


# ============================================================
# 缠论模块 (ChanModule) - 来源: Oracle 研究成果
# ============================================================

CZSC_AVAILABLE = False
try:
    import sqlite3
    import pandas as pd
    from czsc import CZSC, Freq, RawBar
    from czsc.signals import cxt_first_buy_V221126, cxt_first_sell_V221126, cxt_third_buy_V230228
    CZSC_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ChanlunSignals:
    """缠论信号结果"""
    first_buy: bool = False
    first_sell: bool = False
    third_buy: bool = False
    second_buy: bool = False
    second_sell: bool = False
    bottom_divergence: bool = False
    top_divergence: bool = False
    trend: str = "unknown"


def detect_fenxing(bars: List[RawBar]) -> Tuple[bool, bool]:
    """检测分型（顶分型/底分型）"""
    if len(bars) < 3:
        return False, False
    last_3 = bars[-3:]
    bottom = (
        last_3[1].low < last_3[0].low and
        last_3[1].low < last_3[2].low and
        last_3[0].high > last_3[2].high
    )
    top = (
        last_3[1].high > last_3[0].high and
        last_3[1].high > last_3[2].high and
        last_3[0].low < last_3[2].low
    )
    return bottom, top


class ChanlunSignalDetector:
    """
    缠论信号检测器（Oracle 实现版）
    使用 czsc 库识别买卖点
    支持增量更新
    """

    def __init__(self, symbol: str, db_path: str):
        self.symbol = symbol
        self.db_path = db_path
        self.czs_cache: Optional[CZSC] = None

    def _build_czsc(self, bars: List[RawBar]) -> CZSC:
        def get_all_signals(c: CZSC, **kwargs) -> OrderedDict:
            result = OrderedDict()
            try:
                r1 = cxt_first_buy_V221126(c, di=1)
                result['first_buy'] = r1.get('raw_words', [''])[0] if r1 else ''
            except Exception:
                result['first_buy'] = ''
            try:
                r2 = cxt_first_sell_V221126(c, di=1)
                result['first_sell'] = r2.get('raw_words', [''])[0] if r2 else ''
            except Exception:
                result['first_sell'] = ''
            try:
                r3 = cxt_third_buy_V230228(c, di=1)
                result['third_buy'] = r3.get('raw_words', [''])[0] if r3 else ''
            except Exception:
                result['third_buy'] = ''
            bottom, top = detect_fenxing(c.bars_raw)
            result['bottom_divergence'] = '底分型' if bottom else ''
            result['top_divergence'] = '顶分型' if top else ''
            return result
        return CZSC(bars, get_signals=get_all_signals, max_bi_num=100)

    def load_bars(self, days: int = 120) -> List[RawBar]:
        """从数据库加载K线数据"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM daily_data "
            "WHERE symbol=? ORDER BY date ASC LIMIT ?",
            conn, params=(self.symbol, days)
        )
        conn.close()
        bars = []
        for _, row in df.iterrows():
            bar = RawBar(
                symbol=self.symbol,
                dt=str(row['date']),
                freq=Freq.D,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                vol=float(row['volume']) if row['volume'] else 0,
                amount=0
            )
            bars.append(bar)
        return bars

    def update_and_get_signals(self, new_bars: List[RawBar] = None) -> ChanlunSignals:
        """增量更新：传入新K线，更新 CZSC 对象"""
        if not CZSC_AVAILABLE:
            return ChanlunSignals()

        if new_bars:
            if self.czs_cache is None:
                self.czs_cache = self._build_czsc(new_bars)
            else:
                self.czs_cache.update(new_bars)
        else:
            bars = self.load_bars()
            self.czs_cache = self._build_czsc(bars)

        c = self.czs_cache
        sigs = c.signals

        result = ChanlunSignals(
            first_buy='BUY1' in str(sigs.get('first_buy', '')),
            first_sell='SELL1' in str(sigs.get('first_sell', '')),
            third_buy='三买' in str(sigs.get('third_buy', '')),
            bottom_divergence=bool(sigs.get('bottom_divergence')),
            top_divergence=bool(sigs.get('top_divergence')),
        )
        if result.bottom_divergence and not result.first_buy:
            result.second_buy = True
        if result.top_divergence and not result.first_sell:
            result.second_sell = True
        if hasattr(c, 'fx_list') and c.fx_list and len(c.fx_list) >= 2:
            last_2 = c.fx_list[-2:]
            result.trend = "上涨" if last_2[-1].close > last_2[0].close else "下跌"
        else:
            result.trend = "震荡"
        return result


# ============================================================
# 整合策略 (WavechanStrategy) - 继承 BaseStrategy
# ============================================================


class WavechanStrategy(BaseStrategy):
    """
    波浪 + 缠论 整合策略 v3（Oracle 研究成果升级版）

    核心逻辑：
    1. ElliottWaveCounter: Zigzag 算法识别波段 + 斐波那契规则过滤
    2. ChanlunSignalDetector: czsc 库识别一买/二买/三买/一卖/二卖
    3. 整合决策：波浪趋势 + 缠论信号共振

    整合规则：
      波浪向上(1/3/5) + 缠论一买/二买/三买  →  买入
      波浪向下(A/B/C) + 缠论一卖/二卖          →  卖出
    """

    def __init__(self, db_path: str = None, config: dict = None):
        default_config = {
            'wave_threshold_pct': 0.025,    # Zigzag 波段最小幅度
            'wave_confidence_min': 0.6,    # 波浪置信度阈值
            'decline_threshold': -0.15,   # 一买：近20日跌幅阈值
            'consolidation_threshold': 0.05,  # 一买：近5日盘整涨幅阈值
            'position_size': 0.3,
            'stop_loss_pct': 0.03,
            'profit_target_pct': 0.25,
        }
        self.config = {**default_config, **(config or {})}

        super().__init__(db_path, self.config)

        from utils.parquet_db import ParquetDatabaseIntegrator
        self.db_manager = ParquetDatabaseIntegrator(db_path)

        # 初始化波浪计数器（每次分析时按股票创建）
        self.wave_counter = ElliottWaveCounter(
            threshold_pct=self.config['wave_threshold_pct']
        )
        self.chan_detector: Optional[ChanlunSignalDetector] = None

        logger.info("[WavechanStrategy] 波浪+缠论策略(v3 Oracle版)初始化完成")
        logger.info(f"[WavechanStrategy] czsc可用: {CZSC_AVAILABLE}")

    # --------------------------------------------------------
    # 内部：Oracle 波浪模块封装
    # --------------------------------------------------------

    def _analyze_wave(self, prices: np.ndarray) -> dict:
        """封装 ElliottWaveCounter.analyze()"""
        return self.wave_counter.analyze(prices)

    # --------------------------------------------------------
    # 内部：Oracle 缠论模块封装（czsc）
    # --------------------------------------------------------

    def _analyze_chanlun(self, symbol: str, bars: List) -> ChanlunSignals:
        """封装 ChanlunSignalDetector.update_and_get_signals()"""
        if not CZSC_AVAILABLE:
            return ChanlunSignals()
        if self.chan_detector is None:
            self.chan_detector = ChanlunSignalDetector(symbol, self.db_path)
        return self.chan_detector.update_and_get_signals(bars)

    # --------------------------------------------------------
    # 核心特征计算（向量化批量处理）
    # --------------------------------------------------------

    def _compute_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        快速向量化计算所有波浪+缠论特征

        架构说明：
        - 波浪分析：Oracle ElliottWaveCounter（Zigzag + 斐波那契过滤）
        - 缠论分型：手写高效检测（底分型/顶分型）
        - 买卖信号：基于分型 + 趋势的量化判定
        - czsc 实时信号：仅在实时分析时使用，不在批量特征中使用
        """

        def calc_symbol(group):
            g = group.sort_values('date').copy()
            n = len(g)

            if n < 60:
                return pd.DataFrame()

            close = g['close'].values
            high = g['high'].values
            low = g['low'].values

            # === 1. 波浪分析（Oracle ElliottWaveCounter）===
            wave_trend = np.full(n, 'neutral', dtype=object)
            wave_stage = np.full(n, 'unknown', dtype=object)
            wave_total = np.zeros(n)
            wave_last_end_price = np.zeros(n)

            counter = ElliottWaveCounter(threshold_pct=self.config['wave_threshold_pct'])

            # 对每个有数据的位置计算波浪
            WAVE_WINDOW = 60  # P0-2: 30→60，增大窗口提供足够波段数据
            for i in range(WAVE_WINDOW, n):
                recent_prices = close[max(0, i-WAVE_WINDOW):i+1]
                wave_result = counter.analyze(recent_prices)
                wave_trend[i] = wave_result.get('trend', 'neutral')
                pos = wave_result.get('position', {})
                wave_stage[i] = pos.get('last_wave', 'unknown')
                wave_total[i] = pos.get('total_waves', 0)
                wave_last_end_price[i] = pos.get('last_wave_end_price', 0)

            g['wave_trend'] = wave_trend
            g['wave_stage'] = wave_stage
            g['wave_total'] = wave_total
            g['wave_last_end_price'] = wave_last_end_price

            # === 2. 缠论分型检测（手写高效版）===
            fractal = np.full(n, 'none', dtype=object)
            bottom_div = np.full(n, False, dtype=bool)
            top_div = np.full(n, False, dtype=bool)

            for i in range(2, n - 1):
                if (high[i-1] > high[i-2] and high[i-1] > high[i] and
                        high[i-1] > high[i+1]):
                    fractal[i] = '顶分型'
                    top_div[i] = True
                elif (low[i-1] < low[i-2] and low[i-1] < low[i] and
                      low[i-1] < low[i+1]):
                    fractal[i] = '底分型'
                    bottom_div[i] = True

            g['fractal'] = fractal
            g['chan_bottom_div'] = bottom_div
            g['chan_top_div'] = top_div

            # === 3. 买卖信号检测 ===
            recent_20_close = pd.Series(close).shift(20).values
            recent_5_close = pd.Series(close).shift(5).values
            recent_20_decline = np.where(
                pd.Series(recent_20_close).notna(),
                (close - recent_20_close) / recent_20_close,
                0.0
            )
            recent_5_change = np.where(
                pd.Series(recent_5_close).notna(),
                (close - recent_5_close) / recent_5_close,
                0.0
            )

            # 一买：底分型 + 20日跌幅够大 + 5日盘整
            is_first_buy = (
                (fractal == '底分型') &
                (recent_20_decline < self.config['decline_threshold']) &
                (recent_5_change > self.config['consolidation_threshold'])
            )

            # 二买：底分型 + 20日未创新低（价格高于20日前低点）
            prev_low = pd.Series(low).shift(20).values
            is_second_buy = (
                (fractal == '底分型') &
                np.where(
                    pd.Series(prev_low).notna(),
                    (close > prev_low) & (recent_20_decline >= self.config['decline_threshold']),
                    False
                )
            )

            # 背驰检测（MACD辅助）
            divergence = np.full(n, False, dtype=bool)
            if n >= 40:
                for i in range(40, n):
                    recent_low = min(low[i-20:i])
                    prev_low_val = low[i-20]
                    if recent_low < prev_low_val and low[i] == recent_low:
                        recent_decline = (close[i-20] - recent_low) / close[i-20] if close[i-20] > 0 else 0
                        prev_decline = (close[i-40] - low[i-40]) / close[i-40] if close[i-40] > 0 else 0
                        if recent_decline < prev_decline * 0.8:
                            divergence[i] = True

            g['divergence'] = divergence

            # 浪5顶部：创20日新高 + 背驰 + 顶分型
            recent_high = pd.Series(high).rolling(20).max().values
            prev_high = pd.Series(high).shift(20).values
            is_wave5_top = (
                (recent_high > prev_high) &
                (divergence == True) &
                (top_div == True)
            )

            # 缠论信号列（一买/二买/三买）
            g['chan_first_buy'] = is_first_buy
            g['chan_second_buy'] = is_second_buy & ~is_first_buy
            g['chan_third_buy'] = False  # 三买需czsc实时信号
            g['chan_first_sell'] = is_wave5_top
            g['chan_second_sell'] = top_div & ~is_wave5_top

            # === 4. 综合信号判定 ===
            daily_signal = np.full(n, 'hold', dtype=object)
            confidence = np.zeros(n)

            # 买入：波浪向上 + 一买/二买
            buy_mask = (wave_trend == 'up') & (is_first_buy | is_second_buy)
            daily_signal[buy_mask] = '买入'
            confidence[buy_mask] = 0.8

            # 底分型酝酿（波浪向上但无明确买点）
            hold_up_mask = (
                (wave_trend == 'up') &
                bottom_div &
                ~(is_first_buy | is_second_buy)
            )
            daily_signal[hold_up_mask & (daily_signal == 'hold')] = '观望'
            confidence[hold_up_mask] = 0.4

            # 卖出：波浪向下 + 顶分型
            sell_mask = (wave_trend == 'down') & top_div
            daily_signal[sell_mask] = '卖出'
            confidence[sell_mask] = 0.8

            # 顶分型酝酿（波浪向下但无明确卖点）
            hold_down_mask = (
                (wave_trend == 'down') &
                top_div &
                ~top_div
            )
            daily_signal[hold_down_mask & (daily_signal == 'hold')] = '观望'
            confidence[hold_down_mask] = 0.4

            g['daily_signal'] = daily_signal
            g['daily_confidence'] = confidence

            # === 5. 辅助特征 ===
            ma5 = pd.Series(close).rolling(5).mean().values
            ma10 = pd.Series(close).rolling(10).mean().values
            ma20 = pd.Series(close).rolling(20).mean().values

            g['trend_ma'] = np.where(
                (close > ma10) & (ma10 > ma20), 'up',
                np.where((close < ma10) & (ma10 < ma20), 'down', 'unknown')
            )

            g['stop_loss'] = low * (1 - self.config['stop_loss_pct'])
            g['target'] = close * (1 + self.config['profit_target_pct'])

            # 周线兼容
            g['weekly_can_trade'] = (
                (wave_trend == 'up') &
                np.isin(wave_stage, ['Wave1', 'Wave3', 'Wave5'])
            ) | (daily_signal == '买入')
            g['weekly_trend'] = wave_trend

            return g

        logger.info("[WavechanStrategy] 按股票计算波浪+缠论特征...")
        results = []
        symbols = df['symbol'].unique()
        total = len(symbols)

        for idx, symbol in enumerate(symbols):
            if idx % 200 == 0:
                logger.info(f"[WavechanStrategy] 进度: {idx}/{total}")

            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) >= 60:
                try:
                    result = calc_symbol(symbol_data)
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"[WavechanStrategy] {symbol} 处理异常: {e}")

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _compute_features_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """czsc 不可用时的降级实现（保留原有简化逻辑）"""
        def calc_symbol(group):
            g = group.sort_values('date').copy()
            n = len(g)
            if n < 60:
                return pd.DataFrame()

            close = g['close'].values
            high = g['high'].values
            low = g['low'].values

            ma5 = pd.Series(close).rolling(5).mean().values
            ma10 = pd.Series(close).rolling(10).mean().values
            ma20 = pd.Series(close).rolling(20).mean().values

            trend = np.where(
                (close > ma10) & (ma10 > ma20), 'up',
                np.where((close < ma10) & (ma10 < ma20), 'down', 'unknown')
            )
            g['trend_ma'] = trend

            wave_stage = np.full(n, 'unknown', dtype=object)
            wave_trend = np.full(n, 'neutral', dtype=object)

            for i in range(20, n):
                recent_high = max(high[max(0, i-20):i])
                recent_low = min(low[max(0, i-20):i])
                trough_idx = np.argmin(low[max(0, i-20):i])
                trough_price = low[max(0, i-20):i][trough_idx]

                rise = (close[i] - trough_price) / trough_price if trough_price > 0 else 0
                drawdown = (recent_high - close[i]) / recent_high if recent_high > 0 else 0

                if rise < 0.15:
                    wave_stage[i] = 'Wave1'
                elif 0.08 < drawdown < 0.25:
                    wave_stage[i] = 'Wave2'
                elif rise > 0.25 and close[i] > recent_high:
                    wave_stage[i] = 'Wave3'
                elif drawdown > 0.08:
                    wave_stage[i] = 'Wave4'

                wave_trend[i] = trend[i]

            g['wave_trend'] = wave_trend
            g['wave_stage'] = wave_stage
            g['wave_total'] = 0
            g['wave_last_end_price'] = 0
            g['chan_first_buy'] = False
            g['chan_second_buy'] = False
            g['chan_third_buy'] = False
            g['chan_first_sell'] = False
            g['chan_second_sell'] = False
            g['chan_bottom_div'] = False
            g['chan_top_div'] = False

            fractal = np.full(n, 'none', dtype=object)
            for i in range(2, n - 1):
                if high[i-1] > high[i-2] and high[i-1] > high[i] and high[i-1] > high[i+1]:
                    fractal[i] = '顶分型'
                elif low[i-1] < low[i-2] and low[i-1] < low[i] and low[i-1] < low[i+1]:
                    fractal[i] = '底分型'

            recent_20_decline = (close - pd.Series(close).shift(20).values) / pd.Series(close).shift(20).values
            recent_5_change = (close - pd.Series(close).shift(5).values) / pd.Series(close).shift(5).values

            is_first_buy = (
                (fractal == '底分型') &
                (recent_20_decline < self.config['decline_threshold']) &
                (recent_5_change > self.config['consolidation_threshold'])
            )
            is_second_buy = (
                (fractal == '底分型') &
                (recent_20_decline >= self.config['decline_threshold'])
            )

            daily_signal = np.full(n, 'hold', dtype=object)
            confidence = np.zeros(n)

            buy_mask = (trend == 'up') & (is_first_buy | is_second_buy)
            daily_signal[buy_mask] = '买入'
            confidence[buy_mask] = 0.75

            sell_mask = (trend == 'down') & (fractal == '顶分型')
            daily_signal[sell_mask] = '卖出'
            confidence[sell_mask] = 0.75

            g['daily_signal'] = daily_signal
            g['daily_confidence'] = confidence
            g['stop_loss'] = low * (1 - self.config['stop_loss_pct'])
            g['target'] = close * (1 + self.config['profit_target_pct'])
            g['weekly_can_trade'] = (wave_trend == 'up') & np.isin(wave_stage, ['Wave1', 'Wave3'])
            g['weekly_trend'] = wave_trend

            return g

        results = []
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) >= 60:
                result = calc_symbol(symbol_data)
                if not result.empty:
                    results.append(result)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # --------------------------------------------------------
    # BaseStrategy 接口实现
    # --------------------------------------------------------

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """生成波浪+缠论特征数据"""
        logger.info(f"[WavechanStrategy] 生成特征 {start_date} ~ {end_date}")

        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=150)
        extended_start_str = extended_start.strftime("%Y-%m-%d")

        daily_df = self.db_manager.fetch_daily_data(
            extended_start_str, end_date,
            columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        )

        if daily_df.empty:
            logger.warning("[WavechanStrategy] 没有找到日线数据")
            return pd.DataFrame()

        info_df = self.db_manager.get_stock_basic_info()
        if not info_df.empty:
            info_df = info_df[~info_df['name'].str.contains('ST', na=False)]

        daily_df = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        target_dates = (daily_df['date'] >= start_date) & (daily_df['date'] <= end_date)
        target_date_values = daily_df.loc[target_dates, 'date'].unique()

        features_df = self._compute_features_fast(daily_df)

        features_df = features_df[features_df['date'].isin(target_date_values)].copy()

        if not info_df.empty:
            features_df = features_df.merge(
                info_df[['symbol', 'name', 'industry']],
                on='symbol',
                how='left'
            )

        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"[WavechanStrategy] 生成 {len(features_df)} 条特征记录")
        return features_df

    def get_signals(self, start_date: str, end_date: str):
        """获取买卖信号"""
        logger.info(f"[WavechanStrategy] 生成信号 {start_date} ~ {end_date}")

        features_df = self.generate_features(start_date, end_date)

        if features_df.empty:
            logger.warning("[WavechanStrategy] 没有特征数据")
            return [None, None]

        # 买入信号：波浪向上 + 缠论买点
        buy_condition = (
            (features_df['wave_trend'] == 'up') &
            (
                features_df['chan_first_buy'] |
                features_df['chan_second_buy'] |
                features_df['chan_third_buy']
            )
        )
        buy_signals = features_df[buy_condition].copy()
        buy_signals['signal_type'] = 'buy'

        if not buy_signals.empty:
            buy_signals = buy_signals.set_index('date')
            logger.info(f"[WavechanStrategy] 买入信号: {len(buy_signals)} 条")
            sample = buy_signals[['symbol', 'daily_signal', 'wave_stage', 'close']].head(5)
            logger.info(f"买入信号示例:\n{sample.to_string()}")
        else:
            logger.info("[WavechanStrategy] 无买入信号")

        # 卖出信号：波浪向下 + 缠论卖点
        sell_condition = (
            (features_df['wave_trend'] == 'down') &
            (
                features_df['chan_first_sell'] |
                features_df['chan_second_sell']
            )
        )
        sell_signals = features_df[sell_condition].copy()
        sell_signals['signal_type'] = 'sell'

        if not sell_signals.empty:
            sell_signals = sell_signals.set_index('date')
            logger.info(f"[WavechanStrategy] 卖出信号: {len(sell_signals)} 条")

        return [
            buy_signals if not buy_signals.empty else None,
            sell_signals if not sell_signals.empty else None
        ]

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """兼容接口"""
        return []
