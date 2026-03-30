# strategies/wavechan_selector.py
# WaveChan 纯评分选股器 - 使用 WaveChan V3 引擎

"""
WaveChan 选股评分体系

完全基于 WaveChan V3 引擎（strategies/wavechan_v3.py）的波浪+缠论信号。
V3 引擎使用 CZSC 识别笔，WaveCounterV3 做波浪计数，
解决了简化版编号连续消失的问题。

评分维度（V3 Refactored - 2026-03-28）：
  一、信号评分（40%）：C_BUY / W2_BUY / W4_BUY 及确认状态
  二、结构评分（20%）：斐波那契回撤区间
  三、基本面评分（30%）：PE/PB/ROE/营收增长/净利润增长/股息率/流通市值
  四、市场环境评分（10%）：RSI多头区间 + 均线多头 + 量价配合

策略接口（适配 backtester.py）：
  generate_features() → DataFrame[date, symbol, ...features, scores]
  get_signals()        → [buy_signals_df, sell_signals_df]
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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

    # 结构评分（20%）
    wave_retracement: float = 0.0   # 回撤比例
    structure_score: float = 0.0

    # 基本面评分（30%）
    financial_score: float = 0.0

    # 市场环境评分（10%）
    market_score: float = 0.0

    # 综合
    total_score: float = 0.0

    # 辅助信息
    wave_stage: str = "unknown"
    wave_trend: str = "neutral"
    stop_loss: float = 0.0
    fractal: str = "none"
    rsi: float = 50.0
    macd_hist: float = 0.0


# ============================================================
# WaveChan 选股器
# ============================================================

class WaveChanSelector:
    """
    WaveChan 选股评分器 - V3 引擎版（重构版）

    评分体系：
      信号评分（40%）：C_BUY / W2_BUY / W4_BUY 及其确认状态
      结构评分（20%）：斐波那契回撤区间
      基本面评分（30%）：PE/PB/ROE/营收增长/净利润增长/股息率/流通市值
      市场环境评分（10%）：RSI多头 + 均线多头 + 量价配合
    """

    # ---------- 评分权重常数 ----------
    THRESHOLD_SCORE = 50          # 买入阈值（默认值，会被config覆盖）

    # 信号评分表（满分40）
    SIGNAL_SCORES = {
        "C_BUY_confirmed": 40,    # 一买/熊市反转
        "W2_BUY_confirmed": 35,    # W2已确认
        "W2_BUY_alert": 25,       # W2预警（ALERT状态）
        "W4_BUY_confirmed": 25,    # W4已确认
        "W4_BUY_alert": 15,       # W4预警（ALERT状态）
    }

    # 结构评分表（满分20）
    STRUCTURE_SCORES = {
        "W2_optimal": 20,          # W2回撤 38-61.8%
        "W2_shallow": 15,          # W2回撤 23.6-38%
        "W2_deep": 10,             # W2回撤 61.8-78.6%
        "W4_normal": 15,           # W4回撤 23.6-38.2%
        "W3_exhausted": -20,       # W3创新高（动能耗尽）
    }

    # 市场环境评分表（满分10）
    MARKET_SCORES = {
        "rsi_bullish": 5,          # RSI 50-70（多头区间）
        "ma_bullish": 3,           # 近5日均线多头排列
        "volume_surge": 2,         # 成交量放量 >1.5倍
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
            # 财务数据路径
            'financial_data_path': '/root/.openclaw/workspace/data/warehouse',
        }
        self.config = {**default_config, **(config or {})}

        from utils.parquet_db import ParquetDatabaseIntegrator
        self.db_manager = ParquetDatabaseIntegrator(db_path)

        # ---------- 加载财务数据（最新一期，按 symbol 匹配）----------
        self._financial_data: pd.DataFrame = self._load_financial_data()

        logger.info("[WaveChanSelector] WaveChan V3 选股器初始化完成")
        threshold_from_config = self.config.get('threshold', self.THRESHOLD_SCORE)
        logger.info(f"[WaveChanSelector] 配置: top_n={self.config['top_n']}, threshold={threshold_from_config}")

    # --------------------------------------------------------
    # 财务数据加载
    # --------------------------------------------------------

    def _load_financial_data(self) -> pd.DataFrame:
        """
        加载财务数据（最新一期，按 symbol 匹配）
        数据来源：stock_financial_year=*/data.parquet

        Returns:
            DataFrame：symbol -> 最新一期财务数据 dict
            空 dict 表示没有可用数据
        """
        warehouse_path = Path(self.config.get('financial_data_path', '/root/.openclaw/workspace/data/warehouse'))
        financial_dir = warehouse_path / "stock_financial_year=2025"

        if not financial_dir.exists():
            logger.warning("[WaveChanSelector] 财务数据目录不存在，使用空财务数据")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(financial_dir / "data.parquet")
            if df.empty:
                logger.warning("[WaveChanSelector] 财务数据为空")
                return pd.DataFrame()

            # 按 symbol 取最新一期财报（按 date 降序，取每个 symbol 第一条）
            df = df.sort_values('date', ascending=False).drop_duplicates(subset=['symbol'], keep='first')
            df = df.set_index('symbol')

            logger.info(f"[WaveChanSelector] 加载 {len(df)} 只股票的财务数据，最新财报日期: {df['date'].max()}")
            return df
        except Exception as e:
            logger.warning(f"[WaveChanSelector] 财务数据加载失败: {e}")
            return pd.DataFrame()

    # --------------------------------------------------------
    # 基本面评分
    # --------------------------------------------------------

    def _compute_financial_score(self, symbol: str) -> float:
        """
        计算单只股票的基本面评分（满分30）

        评分维度：
          - PE 市盈率（满分15）：合理区间10-30给满分，亏损或极高扣分
          - PB 市净率（满分8）：<3给分
          - ROE 净资产收益率（满分15）：>15%满分，<5%扣分
          - 营收增长率（满分5）：>20%加分，<0扣分
          - 净利润增长率（满分5）：>20%加分，<0扣分
          - 股息率（满分3）：>3%加分
          - 流通市值（满分8）：20-500亿加分

        注：revenue_growth / net_profit_growth / roe 为百分比（如 15.2 表示 15.2%）
            dividend_yield 也为百分比
            float_market_cap 单位为万元（如 30620368 = 306.2亿元）

        Returns:
            基本面评分（0-30）
        """
        if self._financial_data.empty or symbol not in self._financial_data.index:
            # 无财务数据时返回中间值，不给满分也不给0
            return 15.0

        row = self._financial_data.loc[symbol]
        score = 0.0

        # ---- PE 市盈率（满分15）----
        pe = row.get('pe_ratio', np.nan)
        if pd.notna(pe) and pe > 0:
            if 10 <= pe <= 30:
                score += 15.0
            elif 5 <= pe < 10:
                score += 10.0
            elif 30 < pe <= 50:
                score += 8.0
            elif pe < 5:
                score += 5.0
            else:  # pe > 50，极高
                score += 0.0
        # 亏损（pe <= 0）不加分

        # ---- PB 市净率（满分8）----
        pb = row.get('pb_ratio', np.nan)
        if pd.notna(pb) and 0 < pb < 3:
            score += 8.0
        elif pd.notna(pb) and 3 <= pb < 5:
            score += 4.0
        # pb <= 0 不加分

        # ---- ROE 净资产收益率（满分15，映射到 0-15）----
        roe = row.get('roe', np.nan)
        if pd.notna(roe):
            if roe >= 20:
                score += 15.0
            elif roe >= 15:
                score += 12.0
            elif roe >= 10:
                score += 8.0
            elif roe >= 5:
                score += 4.0
            else:  # roe < 5
                score += 0.0

        # ---- 营收增长率（满分5）----
        rev_growth = row.get('revenue_growth', np.nan)
        if pd.notna(rev_growth):
            if rev_growth >= 20:
                score += 5.0
            elif rev_growth >= 10:
                score += 3.0
            elif rev_growth >= 0:
                score += 1.0
            else:  # 负增长
                score += 0.0

        # ---- 净利润增长率（满分5）----
        np_growth = row.get('net_profit_growth', np.nan)
        if pd.notna(np_growth):
            if np_growth >= 20:
                score += 5.0
            elif np_growth >= 10:
                score += 3.0
            elif np_growth >= 0:
                score += 1.0
            else:  # 负增长
                score += 0.0

        # ---- 股息率（满分3）----
        div_yield = row.get('dividend_yield', np.nan)
        if pd.notna(div_yield) and div_yield > 0:
            if div_yield >= 3.0:
                score += 3.0
            elif div_yield >= 1.5:
                score += 2.0
            elif div_yield >= 0.5:
                score += 1.0

        # ---- 流通市值（满分8）----
        # float_market_cap 单位：万元，20-500亿 = 200000-5000000 万元
        float_cap = row.get('float_market_cap', np.nan)
        if pd.notna(float_cap) and float_cap > 0:
            if 200000 <= float_cap <= 5000000:
                score += 8.0
            elif 100000 <= float_cap < 200000:  # 10-20亿
                score += 5.0
            elif 5000000 < float_cap <= 10000000:  # 500-1000亿
                score += 5.0
            else:  # 太小或太大
                score += 0.0

        return max(0.0, min(30.0, score))

    # --------------------------------------------------------
    # 市场环境评分
    # --------------------------------------------------------

    def _compute_market_score(
        self,
        rsi: float,
        close: np.ndarray,
        volume: np.ndarray,
        i: int
    ) -> float:
        """
        计算单只股票单日的市场环境评分（满分10）

        替代原有的 momentum_score（移除 RSI<30 加分、MACD底背离加分）

        评分条件：
          - RSI 50-70（多头区间）: +5
          - 近5日均线多头排列: +3
          - 成交量放量（>5日均量1.5倍）: +2

        Args:
            rsi: 当日 RSI 值
            close: 收盘价数组
            volume: 成交量数组
            i: 当日索引

        Returns:
            市场环境评分（0-10）
        """
        score = 0.0

        # RSI 50-70 多头区间（+5）
        if 50 <= rsi <= 70:
            score += self.MARKET_SCORES["rsi_bullish"]

        # 近5日均线多头排列（+3）
        # 条件：MA5 > MA10 > MA20（简化版，使用日线 close）
        if i >= 20:
            ma5 = np.mean(close[i - 4:i + 1])   # 近5日
            ma10 = np.mean(close[i - 9:i + 1])  # 近10日
            if ma5 > ma10:
                score += self.MARKET_SCORES["ma_bullish"]

        # 成交量放量（>5日均量1.5倍）（+2）
        if i >= 5:
            vol_ma5 = np.mean(volume[i - 4:i + 1])
            if vol_ma5 > 0 and volume[i] >= vol_ma5 * 1.5:
                score += self.MARKET_SCORES["volume_surge"]

        return score

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

        # 缠论分型（用于止损辅助判断，不再用于独立评分）
        fractal = np.full(n, 'none', dtype=object)
        bottom_div = np.full(n, False, dtype=bool)
        for i in range(2, n - 1):
            if (low[i-1] < low[i-2] and low[i-1] < low[i] and low[i-1] < low[i+1]):
                fractal[i] = '底分型'
                bottom_div[i] = True
            elif (high[i-1] > high[i-2] and high[i-1] > high[i] and high[i-1] > high[i+1]):
                fractal[i] = '顶分型'

        # ---------- 涨跌停标记（提前计算）----------
        limit_up = np.zeros(n, dtype=bool)
        limit_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            prev_c = close[i - 1]
            chg = (close[i] - prev_c) / prev_c if prev_c > 0 else 0
            limit_up[i] = (chg >= 0.099) and (volume[i] > 0)
            limit_down[i] = (chg <= -0.099) and (volume[i] > 0)

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
        g['volume_ratio'] = np.where(
            pd.Series(volume).rolling(5).mean().values > 0,
            volume / pd.Series(volume).rolling(5).mean().values,
            1.0
        )
        g['v3_stop_loss'] = v3_stop_loss
        g['limit_up'] = limit_up
        g['limit_down'] = limit_down

        # ---------- 预计算基本面评分（所有日期相同，按 symbol）----------
        fin_score = self._compute_financial_score(symbol)

        # ---------- 评分计算 ----------
        signal_type = np.full(n, 'none', dtype=object)
        signal_status = np.full(n, 'none', dtype=object)
        signal_score = np.zeros(n)
        structure_score = np.zeros(n)
        market_score_arr = np.zeros(n)
        financial_score_arr = np.full(n, fin_score)
        total_score = np.zeros(n)
        stop_loss = np.zeros(n)

        for i in range(WAVE_WINDOW, n):
            cur_rsi = rsi[i]
            cur_fractal = fractal[i]
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
                # 【改进】W2/W4 BUY 只在周线上升浪（W1/W3/W5）中发信号
                # 周线下降趋势中的 W2/W4 BUY = 逆势抄底，确定性低
                if v3_type in ('W2_BUY', 'W4_BUY') and trend != 'up':
                    pass  # 周线不在上升浪，跳过
                else:
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

            # ---- 结构评分（20%）----
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

            # ---- 市场环境评分（10%）----
            # 【临时禁用】market_score 与 BiasFilter 矛盾（选强 = 容易涨停）
            # 移除后：signal(40) + structure(20) + financial(30) = 90
            mkt_sc = 0.0
            market_score_arr[i] = mkt_sc

            # ---- 基本面评分（30%）----
            # 已在循环前预计算，此处直接使用
            fin_sc = fin_score

            # ---- 综合评分 ----
            total = sig_sc + struct_sc + fin_sc  # 90分满分
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
        g['financial_score'] = financial_score_arr
        g['market_score'] = market_score_arr
        # backward compat: keep momentum_score alias for get_buy_signals()
        g['momentum_score'] = market_score_arr
        g['total_score'] = total_score
        g['stop_loss'] = stop_loss

        # 只保留有评分结果的行，且排除涨跌停股
        result = g[(g['total_score'] > 0) & (~g['limit_up']) & (~g['limit_down'])].copy()
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
        threshold = self.config.get('threshold', self.THRESHOLD_SCORE)
        buy_df = features_df[
            (features_df['signal_type'] != 'none') &
            (features_df['total_score'] >= threshold)
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
        #        # 卖出条件：波浪向下 + 顶分型
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

        # 过滤 >= 阈值，且排除涨停股
        threshold = self.config.get('threshold', self.THRESHOLD_SCORE)
        qualified = candidates[
            (candidates['total_score'] >= threshold) &
            (~candidates.get('limit_up', False)) &
            (~candidates.get('limit_down', False))
        ].copy()
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
                'financial_score': row['financial_score'],
                'market_score': row['market_score'],
                # backward compat
                'momentum_score': row.get('market_score', row.get('momentum_score', 0)),
                'wave_stage': row.get('wave_stage', 'unknown'),
                'wave_trend': row.get('wave_trend', 'neutral'),
                'rsi': row.get('rsi', 50),
                'stop_loss': row.get('stop_loss', 0),
                'close': row.get('close', 0),
                'industry': row.get('industry', ''),
            })

        return results
