# strategies/wavechan_selector.py
# WaveChan 纯评分选股器 - 使用 WaveChan V3 引擎

"""
WaveChan 选股评分体系

完全基于 WaveChan V3 引擎（strategies/wavechan_v3.py）的波浪+缠论信号。
V3 引擎使用 CZSC 识别笔，WaveCounterV3 做波浪计数，
解决了简化版编号连续消失的问题。

评分维度：
  一、信号评分（40%）：C_BUY / W2_BUY / W4_BUY 及确认状态
  二、波浪结构评分（30%）：斐波那契回撤区间
  三、动能评分（20%）：RSI / MACD 背离 / 量价配合
  四、缠论确认（10%）：底分型 / 笔破坏

策略接口（适配 backtester.py）：
  generate_features() → DataFrame[date, symbol, ...features, scores]
  get_signals()        → [buy_signals_df, sell_signals_df]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================
# 评分结果数据结构
# ============================================================

@dataclass
class WaveChanScore:
    """单只股票单日评分结果"""
    symbol: str
    date: str

    # 信号评分（40%）
    signal_type: str = "none"       # C_BUY / W2_BUY / W4_BUY / none
    signal_status: str = "none"      # confirmed / warning / none
    signal_score: float = 0.0

    # 结构评分（30%）
    wave_retracement: float = 0.0   # 回撤比例
    structure_score: float = 0.0

    # 动能评分（20%）
    rsi: float = 50.0
    momentum_score: float = 0.0

    # 缠论评分（10%）
    chanlun_score: float = 0.0

    # 综合
    total_score: float = 0.0

    # 辅助信息
    wave_stage: str = "unknown"
    wave_trend: str = "neutral"
    stop_loss: float = 0.0
    fractal: str = "none"
    divergence: bool = False
    macd_hist: float = 0.0


# ============================================================
# WaveChan 选股器
# ============================================================

class WaveChanSelector:
    """
    WaveChan 选股评分器 - V3 引擎版

    完全基于 WaveChan V3 引擎的评分系统，不依赖简化波浪。

    评分体系：
      信号评分（40%）：C_BUY / W2_BUY / W4_BUY 及其确认状态
      结构评分（30%）：斐波那契回撤区间
      动能评分（20%）：RSI / MACD 背离 / 量价配合
      缠论评分（10%）：底分型 / 笔破坏
    """

    # ---------- 评分权重常数 ----------
    THRESHOLD_SCORE = 50          # 买入阈值

    # 信号评分表（满分40）
    SIGNAL_SCORES = {
        "C_BUY_confirmed": 40,    # 一买/熊市反转
        "W2_BUY_confirmed": 35,    # W2已确认
        "W2_BUY_alert": 25,       # W2预警（ALERT状态）
        "W4_BUY_confirmed": 25,    # W4已确认
        "W4_BUY_alert": 15,       # W4预警（ALERT状态）
    }

    # 结构评分表（满分30）
    STRUCTURE_SCORES = {
        "W2_optimal": 20,          # W2回撤 38-61.8%
        "W2_shallow": 15,          # W2回撤 23.6-38%
        "W2_deep": 10,             # W2回撤 61.8-78.6%
        "W4_normal": 15,           # W4回撤 23.6-38.2%
        "W3_exhausted": -20,       # W3创新高（动能耗尽）
    }

    # 动能评分表（满分20）
    MOMENTUM_SCORES = {
        "rsi_oversold": 15,        # RSI < 30
        "rsi_weak": 10,            # RSI 30-50
        "macd_divergence": 10,     # MACD底背离
        "volume_price": 5,         # 量价配合
    }

    # 缠论评分表（满分10）
    CHANLUN_SCORES = {
        "bottom_fractal": 5,       # 底分型形成
        "pen_break": 5,            # 笔破坏（向上）
    }

    def __init__(self, db_path: str = None, config: dict = None):
        """
        初始化选股器

        Args:
            db_path: 数据库路径（保留兼容性）
            config: 配置字典
        """
        default_config = {
            'wave_threshold_pct': 0.025,    # Zigzag 波段最小幅度（保留，V3不再使用）
            'wave_window': 60,             # 波浪分析窗口
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'top_n': 5,                    # 每日最多选股数量
            'stop_loss_pct': 0.08,         # 止损比例（默认8%）
            # 斐波那契区间
            'w2_optimal_min': 0.382,
            'w2_optimal_max': 0.618,
            'w2_shallow_max': 0.382,
            'w2_shallow_min': 0.236,
            'w2_deep_max': 0.786,
            'w4_normal_min': 0.236,
            'w4_normal_max': 0.382,
        }
        self.config = {**default_config, **(config or {})}

        from utils.parquet_db import ParquetDatabaseIntegrator
        self.db_manager = ParquetDatabaseIntegrator(db_path)

        logger.info("[WaveChanSelector] WaveChan V3 选股器初始化完成")
        logger.info(f"[WaveChanSelector] 配置: top_n={self.config['top_n']}, threshold={self.THRESHOLD_SCORE}")

    # --------------------------------------------------------
    # 核心：批量计算所有股票单日评分
    # --------------------------------------------------------

    def _compute_symbol_scores(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算单只股票所有日期的评分
        使用 WaveChan V3 引擎（WaveEngine）获取 W2/W4 信号

        Args:
            symbol: 股票代码
            df: 该股票的历史数据（已按 date 排序），包含 open/high/low/close/volume

        Returns:
            DataFrame：包含所有评分字段
        """
        n = len(df)
        if n < 60:
            return pd.DataFrame()

        g = df.sort_values('date').copy().reset_index(drop=True)

        close = g['close'].values.astype(float)
        high = g['high'].values.astype(float)
        low = g['low'].values.astype(float)
        volume = g['volume'].values.astype(float)

        # ---------- 预计算指标 ----------
        # RSI
        rsi = self._compute_rsi(close, self.config['rsi_period'])

        # MACD
        macd_hist = self._compute_macd(
            close,
            self.config['macd_fast'],
            self.config['macd_slow'],
            self.config['macd_signal']
        )

        # 缠论分型
        fractal = np.full(n, 'none', dtype=object)
        bottom_div = np.full(n, False, dtype=bool)
        for i in range(2, n - 1):
            if (low[i-1] < low[i-2] and low[i-1] < low[i] and low[i-1] < low[i+1]):
                fractal[i] = '底分型'
                bottom_div[i] = True
            elif (high[i-1] > high[i-2] and high[i-1] > high[i] and high[i-1] > high[i+1]):
                fractal[i] = '顶分型'

        # MACD 底背离检测
        divergence = self._compute_macd_divergence(close, low, macd_hist)

        # 量价配合（缩量见底）
        vol_ma5 = pd.Series(volume).rolling(5).mean().values
        vol_ratio = np.where(vol_ma5 > 0, volume / vol_ma5, 1.0)
        volume_price_bottom = (vol_ratio < 0.7) & bottom_div

        # ---------- WaveChan V3 引擎初始化 ----------
        from strategies.wavechan_v3 import WaveEngine

        engine = WaveEngine(symbol=symbol, cache_dir=None)
        # 不加载缓存，从头计算

        # 逐K线喂入 V3 引擎，收集每日信号
        wave_stage = np.full(n, 'unknown', dtype=object)
        wave_trend = np.full(n, 'neutral', dtype=object)
        v3_signal_type = np.full(n, 'none', dtype=object)
        v3_signal_status = np.full(n, 'none', dtype=object)
        v3_stop_loss = np.zeros(n)
        wave_retracement = np.full(n, np.nan)

        # V3 引擎产生信号的时间点标记
        WAVE_WINDOW = self.config['wave_window']

        for i in range(n):
            bar = {
                'date': str(g['date'].iloc[i]),
                'open': float(g['open'].iloc[i]),
                'high': float(g['high'].iloc[i]),
                'low': float(g['low'].iloc[i]),
                'close': float(g['close'].iloc[i]),
                'volume': float(g['volume'].iloc[i]),
            }
            snap = engine.feed_daily(bar)

            wave_stage[i] = snap.state if snap.state else 'unknown'
            wave_trend[i] = snap.direction if snap.direction else 'neutral'

            # 从 V3 引擎获取信号
            sig = engine.get_signal()

            if sig['signal'] in ('W2_BUY', 'W4_BUY', 'C_BUY'):
                v3_signal_type[i] = sig['signal']
                v3_signal_status[i] = sig['status'].lower()  # 'alert' / 'confirmed'
                if sig['stop_loss']:
                    v3_stop_loss[i] = sig['stop_loss']
                elif sig['price']:
                    v3_stop_loss[i] = sig['price'] * (1 - self.config['stop_loss_pct'])

            # V3 引擎的 wave_end_signal 用于结构评分参考
            # wave_retracement 暂用 V3 的 fib_618
            if snap.fib_618 and snap.w3_end and snap.w4_end:
                # W4 回撤比例
                w3_amp = snap.w3_end - (snap.w3_start or snap.w2_end or snap.w1_end or 0)
                if w3_amp > 0:
                    retr = (snap.w3_end - snap.w4_end) / w3_amp
                    if 0 <= retr <= 1.5:
                        wave_retracement[i] = retr

        g['wave_stage'] = wave_stage
        g['wave_trend'] = wave_trend
        g['v3_signal_type'] = v3_signal_type
        g['v3_signal_status'] = v3_signal_status
        g['wave_retracement'] = wave_retracement
        g['fractal'] = fractal
        g['bottom_div'] = bottom_div
        g['rsi'] = rsi
        g['macd_hist'] = macd_hist
        g['divergence'] = divergence
        g['volume_ratio'] = vol_ratio
        g['volume_price_bottom'] = volume_price_bottom
        g['v3_stop_loss'] = v3_stop_loss

        # ---------- 评分计算 ----------
        signal_type = np.full(n, 'none', dtype=object)
        signal_status = np.full(n, 'none', dtype=object)
        signal_score = np.zeros(n)
        structure_score = np.zeros(n)
        momentum_score = np.zeros(n)
        chanlun_score = np.zeros(n)
        total_score = np.zeros(n)
        stop_loss = np.zeros(n)

        for i in range(WAVE_WINDOW, n):
            cur_rsi = rsi[i]
            cur_fractal = fractal[i]
            bot_div = bottom_div[i]
            div_flag = divergence[i]
            vol_bottom = volume_price_bottom[i]
            cur_close = close[i]
            cur_low = low[i]
            trend = wave_trend[i]

            # ---- 信号评分（40%）----
            sig_type = 'none'
            sig_status = 'none'
            sig_sc = 0.0

            # V3 引擎信号：C_BUY / W2_BUY / W4_BUY
            v3_type = v3_signal_type[i]
            v3_st = v3_signal_status[i]

            if v3_type in ('C_BUY', 'W2_BUY', 'W4_BUY'):
                sig_type = v3_type
                sig_status = v3_st if v3_st != 'none' else 'alert'
                key = f"{sig_type}_{sig_status}"
                if key in self.SIGNAL_SCORES:
                    sig_sc = self.SIGNAL_SCORES[key]
                elif sig_status == 'confirmed':
                    key_confirmed = f"{sig_type}_confirmed"
                    sig_sc = self.SIGNAL_SCORES.get(key_confirmed, 20)
                elif sig_status == 'alert':
                    key_alert = f"{sig_type}_alert"
                    sig_sc = self.SIGNAL_SCORES.get(key_alert, 15)
                else:
                    sig_sc = 10

            # C_BUY：底分型 + 20日跌幅够大 + 波浪向上（V3没有发出C_BUY时补充）
            if sig_type == 'none' and cur_fractal == '底分型':
                recent_20_close = close[i - 20] if i >= 20 else close[0]
                decline_20 = (cur_close - recent_20_close) / recent_20_close if recent_20_close > 0 else 0
                decline_thresh = self.config.get('decline_threshold', -0.15)
                if decline_20 < decline_thresh and trend in ('long', 'neutral'):
                    sig_type = 'C_BUY'
                    sig_status = 'confirmed'
                    sig_sc = self.SIGNAL_SCORES["C_BUY_confirmed"]

            signal_type[i] = sig_type
            signal_status[i] = sig_status
            signal_score[i] = sig_sc

            # ---- 结构评分（30%）----
            struct_sc = 0.0
            retr = wave_retracement[i]

            # W2 回撤评分（基于 fib 回撤）
            if sig_type == 'W2_BUY' and not np.isnan(retr):
                cfg = self.config
                if cfg['w2_shallow_min'] <= retr < cfg['w2_shallow_max']:
                    struct_sc = self.STRUCTURE_SCORES["W2_shallow"]
                elif cfg['w2_optimal_min'] <= retr <= cfg['w2_optimal_max']:
                    struct_sc = self.STRUCTURE_SCORES["W2_optimal"]
                elif cfg['w2_optimal_max'] < retr <= cfg.get('w2_deep_max', 0.786):
                    struct_sc = self.STRUCTURE_SCORES["W2_deep"]

            # W4 回撤评分
            elif sig_type == 'W4_BUY' and not np.isnan(retr):
                cfg = self.config
                if cfg['w4_normal_min'] <= retr <= cfg['w4_normal_max']:
                    struct_sc = self.STRUCTURE_SCORES["W4_normal"]

            # W3 动能耗尽
            elif wave_stage[i] == 'w3_formed':
                if len(g) > i + 1 and i > 0:
                    prev_close = close[i - 1]
                    if cur_close < prev_close and cur_rsi < 50:
                        struct_sc = self.STRUCTURE_SCORES["W3_exhausted"]

            structure_score[i] = struct_sc

            # ---- 动能评分（20%）----
            mom_sc = 0.0
            if cur_rsi < 30:
                mom_sc += self.MOMENTUM_SCORES["rsi_oversold"]
            elif cur_rsi < 50:
                mom_sc += self.MOMENTUM_SCORES["rsi_weak"]

            if div_flag:
                mom_sc += self.MOMENTUM_SCORES["macd_divergence"]

            if vol_bottom:
                mom_sc += self.MOMENTUM_SCORES["volume_price"]

            momentum_score[i] = mom_sc

            # ---- 缠论评分（10%）----
            chan_sc = 0.0
            if cur_fractal == '底分型':
                chan_sc += self.CHANLUN_SCORES["bottom_fractal"]

            # 笔破坏（向上）：连续3根K线，低点依次抬高（简化版）
            if i >= 2:
                if low[i] > low[i-1] > low[i-2]:
                    chan_sc += self.CHANLUN_SCORES["pen_break"]

            chanlun_score[i] = chan_sc

            # ---- 综合评分 ----
            total = sig_sc + struct_sc + mom_sc + chan_sc
            total_score[i] = total

            # ---- 止损价 ----
            # 优先用 V3 引擎的止损，否则用默认逻辑
            if v3_stop_loss[i] > 0:
                stop_loss[i] = v3_stop_loss[i]
            else:
                recent_low = min(low[max(0, i-20):i+1])
                stop_loss[i] = recent_low * (1 - self.config['stop_loss_pct'])

        g['signal_type'] = signal_type
        g['signal_status'] = signal_status
        g['signal_score'] = signal_score
        g['structure_score'] = structure_score
        g['momentum_score'] = momentum_score
        g['chanlun_score'] = chanlun_score
        g['total_score'] = total_score
        g['stop_loss'] = stop_loss

        # 只保留有评分结果的行
        result = g[g['total_score'] > 0].copy()
        return result

    # --------------------------------------------------------
    # 指标计算辅助函数
    # --------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """计算 RSI"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))

        if len(gains) < period:
            return np.full(len(prices), 50.0)

        avg_gains[period] = np.mean(gains[1:period + 1])
        avg_losses[period] = np.mean(losses[1:period + 1])

        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period

        rs = np.where(avg_losses == 0, 100.0, avg_gains / avg_losses)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _compute_macd(prices: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
        """计算 MACD histogram"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        hist = macd_line - signal_line
        return hist

    @staticmethod
    def _compute_macd_divergence(prices: np.ndarray, lows: np.ndarray, macd_hist: np.ndarray, window: int = 20) -> np.ndarray:
        """检测 MACD 底背离"""
        n = len(prices)
        divergence = np.full(n, False, dtype=bool)

        for i in range(window, n):
            # 前一周期低点
            prev_low_idx = i - window + np.argmin(lows[i - window:i])
            prev_low_price = lows[prev_low_idx]
            prev_low_macd = macd_hist[prev_low_idx]

            # 当前低点
            curr_low_price = lows[i]
            curr_low_macd = macd_hist[i]

            # 背离：价格创新低，但 MACD 未创新低（或底背离）
            if curr_low_price < prev_low_price and curr_low_macd > prev_low_macd:
                divergence[i] = True

        return divergence

    # --------------------------------------------------------
    # 策略接口：生成特征
    # --------------------------------------------------------

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成 WaveChan 评分特征

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame：包含评分字段
        """
        logger.info(f"[WaveChanSelector] 生成特征 {start_date} ~ {end_date}")

        # 前推足够多的交易日（波浪分析需要历史数据）
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=self.config['wave_window'] + 60)
        extended_start_str = extended_start.strftime("%Y-%m-%d")

        # 加载数据
        daily_df = self.db_manager.fetch_daily_data(
            extended_start_str, end_date,
            columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        )

        if daily_df.empty:
            logger.warning("[WaveChanSelector] 没有找到日线数据")
            return pd.DataFrame()

        # 过滤 ST 股
        info_df = self.db_manager.get_stock_basic_info()
        if not info_df.empty:
            info_df = info_df[~info_df['name'].str.contains('ST', na=False)]
            valid_symbols = set(info_df['symbol'].unique())
            daily_df = daily_df[daily_df['symbol'].isin(valid_symbols)]

        daily_df = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 目标日期（排除 lookback 预热期）
        target_mask = (daily_df['date'] >= start_date) & (daily_df['date'] <= end_date)
        target_dates = daily_df.loc[target_mask, 'date'].unique()

        logger.info(f"[WaveChanSelector] 共 {len(daily_df['symbol'].unique())} 只股票，"
                     f"{len(daily_df)} 条数据，开始计算评分...")

        # 逐股票计算评分
        results = []
        symbols = daily_df['symbol'].unique()
        total = len(symbols)

        for idx, sym in enumerate(symbols):
            if idx % 200 == 0:
                logger.info(f"[WaveChanSelector] 进度: {idx}/{total}")

            sym_df = daily_df[daily_df['symbol'] == sym]
            if len(sym_df) < self.config['wave_window'] + 10:
                continue

            try:
                scored = self._compute_symbol_scores(sym, sym_df)
                if not scored.empty:
                    # 只保留目标日期范围
                    scored = scored[scored['date'].isin(target_dates)]
                    if not scored.empty:
                        results.append(scored)
            except Exception as e:
                logger.debug(f"[WaveChanSelector] {sym} 处理异常: {e}")

        if not results:
            return pd.DataFrame()

        features_df = pd.concat(results, ignore_index=True)

        # 合并基本信息
        if not info_df.empty:
            features_df = features_df.merge(
                info_df[['symbol', 'name', 'industry']],
                on='symbol',
                how='left'
            )

        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.sort_values(['date', 'symbol']).reset_index(drop=True)

        logger.info(f"[WaveChanSelector] 生成 {len(features_df)} 条评分记录，"
                     f"{len(features_df['date'].unique())} 个交易日")

        return features_df

    # --------------------------------------------------------
    # 策略接口：获取买卖信号
    # --------------------------------------------------------

    def get_signals(self, start_date: str, end_date: str):
        """
        获取 WaveChan 买卖信号

        Returns:
            [buy_signals_df, sell_signals_df]
            buy_signals_df: 评分 >= 50，按评分降序，每日最多 top_n 只
        """
        logger.info(f"[WaveChanSelector] 生成信号 {start_date} ~ {end_date}")

        features_df = self.generate_features(start_date, end_date)

        if features_df.empty:
            return [None, None]

        # ---------- 买入信号 ----------
        # 必须有实际信号类型（signal_type != 'none'）才能作为买入信号
        buy_df = features_df[
            (features_df['signal_type'] != 'none') &
            (features_df['total_score'] >= self.THRESHOLD_SCORE)
        ].copy()
        buy_df = buy_df.sort_values('total_score', ascending=False)

        # 每日最多选 top_n 只
        top_n = self.config['top_n']
        buy_signals_list = []

        for date, group in buy_df.groupby('date'):
            top = group.head(top_n)
            buy_signals_list.append(top)

        if buy_signals_list:
            buy_signals = pd.concat(buy_signals_list, ignore_index=True)
            buy_signals['signal_type_enum'] = buy_signals['signal_type'] + '_' + buy_signals['signal_status']
            buy_signals = buy_signals.set_index('date')
            logger.info(f"[WaveChanSelector] 买入信号: {len(buy_signals)} 条，"
                         f"{len(buy_signals.index.unique())} 个交易日有信号")
        else:
            buy_signals = None

        # ---------- 卖出信号 ----------
        # 卖出条件：波浪向下 + 顶分型
        sell_mask = (
            (features_df['wave_trend'] == 'down') &
            (features_df['fractal'] == '顶分型')
        )
        sell_signals = features_df[sell_mask].copy()

        if not sell_signals.empty:
            sell_signals['signal_type'] = 'sell'
            sell_signals = sell_signals.set_index('date')
            logger.info(f"[WaveChanSelector] 卖出信号: {len(sell_signals)} 条")
        else:
            sell_signals = None

        # 打印评分分布统计
        if not features_df.empty:
            logger.info(f"[WaveChanSelector] 评分分布（有效信号日）:")
            logger.info(f"  总分均值: {features_df['total_score'].mean():.1f}")
            logger.info(f"  总分中位数: {features_df['total_score'].median():.1f}")
            logger.info(f"  >=50分信号数: {len(buy_signals) if buy_signals is not None else 0}")

        return [buy_signals, sell_signals]

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        兼容接口：当日选股

        Args:
            date: 选股日期
            candidates: 当日候选股票 DataFrame（包含评分字段）

        Returns:
            list[dict]: 买入信号列表
        """
        if candidates.empty:
            return []

        # 过滤 >= 阈值
        qualified = candidates[candidates['total_score'] >= self.THRESHOLD_SCORE].copy()
        qualified = qualified.sort_values('total_score', ascending=False)
        qualified = qualified.head(self.config['top_n'])

        results = []
        for _, row in qualified.iterrows():
            results.append({
                'symbol': row['symbol'],
                'total_score': row['total_score'],
                'signal_type': row['signal_type'],
                'signal_status': row['signal_status'],
                'signal_score': row['signal_score'],
                'structure_score': row['structure_score'],
                'momentum_score': row['momentum_score'],
                'chanlun_score': row['chanlun_score'],
                'wave_stage': row.get('wave_stage', 'unknown'),
                'wave_trend': row.get('wave_trend', 'neutral'),
                'rsi': row.get('rsi', 50),
                'stop_loss': row.get('stop_loss', 0),
                'close': row.get('close', 0),
                'industry': row.get('industry', ''),
            })

        return results
