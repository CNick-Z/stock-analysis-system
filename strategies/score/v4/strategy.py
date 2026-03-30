# strategies/score/v4/strategy.py
# V4 评分策略 — ScoreV4Strategy
# =============================================================
# 基于技术指标评分的选股策略（无跟踪止损版）
#
# 【选股条件】
#   - growth_condition: 收涨且涨幅 <= 6%
#   - ma_condition: SMA5 > SMA10 且 SMA10 < SMA20
#   - volume_condition: 放量1.5倍或超过5日均量1.2倍
#   - macd_condition: MACD < 0 且 > signal
#   - jc_condition: SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%
#   - angle_condition: SMA10角度 > 30°（V4独有）
#   - trend_condition: SMA20 < SMA55 且 SMA55 > SMA240
#   - rsi_filter: RSI 在 50-60 之间
#   - price_filter: 价格 3-15 元
#
# 【出场规则】
#   - 止损: 5%
#   - 止盈: 15%
#   - MA死叉: 入场时 sma20 <= sma55，当前 sma20 > sma55（需记录入场均线关系）
#   - 趋势破坏: 跌破 MA20 + 资金流为负
#
# 【持仓管理】
#   - 最多持仓: 5只（config: max_positions）
#   - 单只仓位上限: 20%（config: position_size）

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ============ 条件计算（内部函数）============

def _compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 V4 所有选股条件（DataFrame批量）

    Args:
        df: 包含技术指标的日线数据

    Returns:
        添加了各类条件列的 DataFrame
    """
    df = df.copy()

    # 涨幅条件：收涨且上影线不长
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (
        (df['close'] >= df['open']) &
        (df['high'] <= df['open'] * 1.06)
    )

    # 均线条件：SMA5 > SMA10 且 SMA10 < SMA20
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])

    # 成交量条件
    df['volume_condition'] = (
        (df['volume'] > df['volume'].shift(1) * 1.5) |
        (df['volume'] > df['vol_ma5'] * 1.2)
    )

    # MACD条件
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )

    # 金叉条件：SMA5连续2日上升 AND SMA20当日上升 AND 均线收敛<2%
    df['jc_condition'] = (
        (df['sma_5'] > df['sma_5'].shift(1)) &
        (df['sma_5'].shift(1) > df['sma_5'].shift(2)) &
        (df['sma_20'] > df['sma_20'].shift(1)) &
        (abs(df['sma_5'] - df['sma_20']) / df['sma_20'] < 0.02)
    )

    # 角度条件（V4独有）：SMA10角度 > 30°
    df['angle_condition'] = df['angle_ma_10'] > 30

    # 趋势条件：SMA20 < SMA55 且 SMA55 > SMA240
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])

    # RSI过滤：50-60
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)

    # 价格过滤：3-15元
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)

    return df


def _apply_buy_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    V4 买入条件过滤

    全部条件同时满足才买入：
      growth_condition & ma_condition & angle_condition &
      volume_condition & macd_condition &
      (jc_condition | macd_jc) &
      trend_condition & rsi_filter & price_filter

    Args:
        df: 经过 _compute_conditions 的 DataFrame

    Returns:
        满足全部买入条件的 DataFrame
    """
    df = df.copy()
    buy = (
        df['growth_condition'] &
        df['ma_condition'] &
        df['angle_condition'] &
        df['volume_condition'] &
        df['macd_condition'] &
        (df['jc_condition'] | df['macd_jc']) &
        df['trend_condition'] &
        df['rsi_filter'] &
        df['price_filter']
    )
    return df[buy].copy()


def _score_row(row: pd.Series) -> float:
    """
    对单行计算 V4 评分（用于模拟盘逐行评分）

    评分维度：
      - 技术面（均线、MACD、成交量、RSI、KDJ、CCI、布林带）
      - 资金流（资金量、量增幅、主生量等）
      - 市场热度

    Args:
        row: 单行数据（需已包含条件列和资金流列）

    Returns:
        V4 评分浮点数
    """
    score = 0.0

    # === 技术面 ===
    # 涨幅得分
    growth = row.get('growth', 0)
    if 0.5 <= growth <= 6.0:
        score += 0.06
    elif growth > 0:
        score += 0.03

    # 均线条件
    if row.get('ma_condition', False):
        score += 0.1622

    # 角度条件（SMA10 > 30°）
    if row.get('angle_ma_10', 0) > 30:
        score += 0.0854

    # MACD
    if row.get('macd_condition', False):
        score += 0.1366
    if row.get('macd_jc', False):
        score += 0.0873

    # 成交量
    vol_ratio = min(row.get('vol_ratio', row.get('volume', 0) / max(row.get('vol_ma5', 1), 1)), 3)
    score += vol_ratio * 0.1704

    # 超买超卖
    if row.get('rsi_14', 100) < 70:
        score += 0.0597
    if row.get('kdj_k', 100) < 80:
        score += 0.0597
    if row.get('cci_20', 100) < 100:
        score += 0.0597
    if row.get('close', 0) < row.get('bb_upper', float('inf')):
        score += 0.1191

    # === 资金流 ===
    if row.get('money_flow_positive', False):
        score += (-0.0072) * 1.4   # 资金量正 → 加分（负权重实际是抑制）
    if row.get('money_flow_increasing', False):
        score += 0.0108
    if row.get('money_flow_trend', False):
        main_vol = row.get('主生量', 0)
        base_vol = row.get('量基线', 1)
        score += 0.0147 * min(main_vol / max(base_vol, 1e-6), 2.0)
    if row.get('money_flow_weekly', False):
        score += 0.0072
    if row.get('money_flow_weekly_increasing', False):
        score += 0.0036 * 1.3

    # 量增幅调整
    vol_gain = row.get('量增幅', 0)
    if vol_gain > 1.3:
        score *= 0.3434
    elif vol_gain < -0.65:
        score *= 0.1717

    # 资金流量化
    score += min(vol_gain, 5) * 0.1524
    if row.get('量基线', 0) > 0:
        score += 0.0437
    main_vol = row.get('主生量', 0)
    base_vol = row.get('量基线', 1)
    score += min(main_vol / max(base_vol, 1e-6), 2.0) * 0.0159
    score += min(row.get('周增幅', 0), 5) * 0.2438

    # === 市场热度 ===
    market_heat = row.get('market_heat', 0)

    # 综合权重
    total = (
        score * 0.4483 +
        score * 0.2947 * 0.5 +   # 资金流权重简化
        market_heat * 0.257
    )

    return total


def _score_stocks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 批量计算 V4 评分（用于回测）

    Args:
        df: 经过 _apply_buy_filter 的 DataFrame

    Returns:
        添加了评分列的 DataFrame
    """
    df = df.copy()
    scores = []
    for _, row in df.iterrows():
        scores.append(_score_row(row))
    df['score'] = scores
    return df


# ============ 策略类 ============

class ScoreV4Strategy:
    """
    V4 评分策略类（无跟踪止损版 — 趋势跟踪）

    实现标准策略接口（filter_buy / score / should_sell / get_entry_conditions）。
    趋势跟踪风格：买入后一直持有，直到触发止损/止盈/MA死叉/趋势破坏。

    【入场条件】
      - growth_condition: 收涨且涨幅 <= 6%
      - ma_condition: SMA5 > SMA10 且 SMA10 < SMA20
      - angle_condition: SMA10角度 > 30°（V4独有）
      - volume_condition: 放量1.5倍或超过5日均量1.2倍
      - macd_condition: MACD < 0 且 > signal
      - jc_condition: SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%
      - trend_condition: SMA20 < SMA55 且 SMA55 > SMA240
      - rsi_filter: RSI 在 50-60 之间
      - price_filter: 价格 3-15 元

    【出场规则】
      - 止损: 5%（next_open < avg_cost * 0.95）
      - 止盈: 15%（next_open > avg_cost * 1.15）
      - MA死叉: 入场时 sma20 <= sma55，当前 sma20 > sma55
      - 趋势破坏: 跌破 MA20 + 资金流为负

    【持仓管理】
      - 最多持仓: 5只（config: max_positions）
      - 单只仓位上限: 20%（config: position_size）
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化 V4 策略

        Args:
            config: 配置字典，支持以下键:
                - stop_loss: 止损比例（默认 0.05）
                - take_profit: 止盈比例（默认 0.15）
                - max_positions: 最大持仓数（默认 5）
                - position_size: 单只仓位比例（默认 0.20）
        """
        self.config = config or {}
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.take_profit = self.config.get('take_profit', 0.15)
        self.max_positions = self.config.get('max_positions', 5)
        self.position_size = self.config.get('position_size', 0.20)

    # ------------------------------------------------------------------
    # 标准策略接口
    # ------------------------------------------------------------------

    def filter_buy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤候选股票（满足全部入场条件）

        调用 _compute_conditions() + _apply_buy_filter()，
        返回满足全部入场条件的候选股票。

        Args:
            df: 原始日线数据（需包含技术指标列）

        Returns:
            过滤后的 DataFrame（已通过全部条件）
        """
        df = _compute_conditions(df)
        df = _apply_buy_filter(df)
        return df

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评分排序（在 filter_buy 基础上评分）

        调用 _score_stocks_df()，返回添加了 score 列的 DataFrame。

        Args:
            df: 经过 filter_buy() 过滤后的 DataFrame

        Returns:
            添加了 'score' 列的 DataFrame，按 score 降序排列
        """
        df = _score_stocks_df(df)
        df = df.sort_values('score', ascending=False)
        return df

    def should_sell(
        self,
        row: pd.Series,
        pos: dict,
        market: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        判断是否应该出场

        出场条件（优先级：止损 > 止盈 > MA死叉 > 趋势破坏）：

          1. 止损: next_open < avg_cost * (1 - stop_loss)
          2. 止盈: next_open > avg_cost * (1 + take_profit)
          3. MA死叉: 入场时 sma20 <= sma55，当前 sma20 > sma55
                     ⚠️ 需要在建仓时记录 entry_sma20_le_sma55
          4. 趋势破坏: close < SMA20 且 money_flow_trend == False

        Args:
            row: 当日行情数据（需含 next_open, close, sma_20, sma_55,
                 money_flow_trend 等）
            pos: 持仓信息字典，必须包含:
                    - avg_cost: 入场成本价
                    - entry_sma20_le_sma55: bool，入场时 sma20 <= sma55 为 True
                    - entry_date: str，入场日期（可选，用于日志）
            market: 市场上下文（可选，当前未使用，保留接口兼容性）

        Returns:
            (should_sell: bool, reason: str)
        """
        # 处理 next_open 缺失：降级到 close，仍未获取则用 avg_cost
        next_open = row.get('next_open')
        if pd.isna(next_open) or next_open <= 0:
            next_open = row.get('close')
            if pd.isna(next_open) or next_open <= 0:
                next_open = pos.get('avg_cost', 0)

        entry_price = pos.get('avg_cost', 0)
        if entry_price <= 0:
            return False, "INVALID_POSITION avg_cost <= 0"

        # 1. 止损
        stop_price = entry_price * (1 - self.stop_loss)
        if next_open < stop_price:
            return True, f"STOP_LOSS @{next_open:.2f}"

        # 2. 止盈
        profit_price = entry_price * (1 + self.take_profit)
        if next_open > profit_price:
            return True, f"TAKE_PROFIT @{next_open:.2f}"

        # 3. MA死叉
        #    入场时 sma20 <= sma55（entry_sma20_le_sma55=True），
        #    当前 sma20 > sma55 = 死叉已发生，触发出场
        if pos.get('entry_sma20_le_sma55', False):
            if row.get('sma_20', 0) > row.get('sma_55', 0):
                return True, "MA_DEATH_CROSS"

        # 4. 趋势破坏：跌破 MA20 + 资金流为负
        if row.get('close', 0) < row.get('sma_20', float('inf')):
            if row.get('money_flow_trend', True) is False:
                return True, "TREND_BREAK ma20+money_flow"

        return False, ""

    def get_entry_conditions(self) -> dict:
        """
        返回当前生效的入场条件（用于日志/报告）

        Returns:
            选股条件字典
        """
        return {
            "growth_condition": "收涨且涨幅 <= 6%",
            "ma_condition": "SMA5 > SMA10 且 SMA10 < SMA20",
            "angle_condition": "SMA10角度 > 30°（V4独有）",
            "volume_condition": "放量1.5倍或超过5日均量1.2倍",
            "macd_condition": "MACD < 0 且 > signal",
            "jc_condition": "SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%",
            "trend_condition": "SMA20 < SMA55 且 SMA55 > SMA240",
            "rsi_filter": "RSI 在 50-60 之间",
            "price_filter": "价格 3-15 元",
            # 出场
            "stop_loss": f"{self.stop_loss:.0%}",
            "take_profit": f"{self.take_profit:.0%}",
            "ma_death_cross": "入场时 sma20 <= sma55，当前 sma20 > sma55",
            "trend_break": "close < SMA20 且 money_flow_trend == False",
            # 持仓
            "max_positions": self.max_positions,
            "position_size": f"{self.position_size:.0%}",
        }

    def __repr__(self) -> str:
        return (
            f"<ScoreV4Strategy "
            f"stop_loss={self.stop_loss:.0%} "
            f"take_profit={self.take_profit:.0%} "
            f"max_positions={self.max_positions}>"
        )


# =============================================================================
# 兼容层：保留原有的 ScoreStrategy 类和模块级逻辑
# 旧脚本（backtester.py）依赖 ScoreStrategy.get_signals()
# 新代码应使用 ScoreV4Strategy
# =============================================================================

class ScoreStrategy(ScoreV4Strategy):
    """
    兼容层：ScoreStrategy = ScoreV4Strategy

    保留原有类名，新代码请直接使用 ScoreV4Strategy。
    """
    pass


def compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _compute_conditions"""
    return _compute_conditions(df)


def apply_buy_filter(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _apply_buy_filter"""
    return _apply_buy_filter(df)


def score_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _score_stocks_df"""
    return _score_stocks_df(df)


def score_row(row: pd.Series) -> float:
    """兼容层：调用 _score_row"""
    return _score_row(row)


def should_sell(row: pd.Series, pos: dict, market: Optional[dict] = None) -> Tuple[bool, str]:
    """
    兼容层：全局 should_sell 函数

    注意：调用方需要自行保证 pos 中包含 entry_sma20_le_sma55 字段
    建议使用 ScoreV4Strategy.should_sell() 方法
    """
    strategy = ScoreV4Strategy()
    return strategy.should_sell(row, pos, market)
