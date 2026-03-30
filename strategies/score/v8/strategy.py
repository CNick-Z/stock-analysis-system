#!/usr/bin/env python3
"""
V8 策略核心模块 — ScoreV8Strategy
====================================
v6核心 + IC增强过滤

【选股条件】
  - growth_condition: 收涨且涨幅<=6%
  - ma_condition: SMA5>SMA10 且 SMA10<SMA20
  - volume_condition: 放量1.5倍或超过5日均量1.2倍
  - macd_condition: MACD<0 且 >signal
  - jc_condition: SMA5向上且SMA20向上
  - trend_condition: SMA20<SMA55 且 SMA55>SMA240
  - rsi_filter: RSI在50-60之间
  - price_filter: 价格3-15元

【IC增强过滤 - 剔除】
  - RSI>70 或 <25
  - 换手率>2.79%
  - WR>-95
  - CCI<-200

【IC增强过滤 - 加分】
  - CCI<-100: +0.10分
  - WR<-80: +0.05分
  - 换手率<0.42%: +0.05分
  - vol_ratio<0.71: +0.05分  # IC=+0.0037最优区间（2026-03-30调优）

【出场规则】
  - 止损: 5%
  - 止盈: 15%
  - MA死叉: SMA20从下穿上时入场，死叉触发（sma20>sma55 且 入场时sma20<sma55）

【持仓管理】
  - 最多5只
  - 每只仓位上限20%

此模块被以下脚本使用:
  - backtest_score_v8.py (回测)
  - v8_simulated_trading.py (模拟盘)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


# ============ 条件计算（内部函数）============

def _compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有选股条件（DataFrame批量）
    
    Args:
        df: 包含技术指标的日线数据
        
    Returns:
        添加了各类条件列的 DataFrame
    """
    df = df.copy()
    
    # 涨幅
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    
    # 均线
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    
    # 成交量
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | (df['volume'] > df['vol_ma5'] * 1.2)
    
    # MACD
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    # 金叉条件（SMA5连续2日上升 AND SMA20当日上升 AND 均线收敛<2%）
    df['jc_condition'] = (
        (df['sma_5'] > df['sma_5'].shift(1)) &
        (df['sma_5'].shift(1) > df['sma_5'].shift(2)) &
        (df['sma_20'] > df['sma_20'].shift(1)) &
        (abs(df['sma_5'] - df['sma_20']) / df['sma_20'] < 0.02)
    )
    
    # 趋势
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    
    # RSI
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    
    # 价格
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)
    
    return df


def _apply_ic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    IC增强过滤 - 剔除 + 加分
    
    Args:
        df: 经过条件计算的 DataFrame
        
    Returns:
        过滤后的 DataFrame，带 ic_bonus 列
    """
    df = df.copy()
    
    # 处理换手率重名字段（merge 后可能产生 _x/_y 后缀）
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    # ===== IC 增强过滤 — 剔除条件 =====
    exclude_mask = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 25) |
        (df['turnover_rate'] > 2.79) |
        (df['williams_r'] < -95) |
        (df['cci_20'] < -200)
    )
    df = df[~exclude_mask].copy()
    
    # ===== IC 增强过滤 — 买入条件 =====
    buy = (
        df['growth_condition'] &
        df['ma_condition'] &
        df['volume_condition'] &
        df['macd_condition'] &
        (df['jc_condition'] | df['macd_jc']) &
        df['trend_condition'] &
        df['rsi_filter'] &
        df['price_filter']
    )
    df = df[buy].copy()
    
    # ===== IC 加分 =====
    df['ic_bonus'] = 0.0
    df.loc[df['cci_20'] < -100, 'ic_bonus'] += 0.10
    df.loc[df['williams_r'] < -80, 'ic_bonus'] += 0.05
    df.loc[df['turnover_rate'] < 0.42, 'ic_bonus'] += 0.05
    df.loc[df['vol_ratio'] < 0.71, 'ic_bonus'] += 0.05
    
    return df


def _compute_v6_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算V6原始评分（DataFrame批量）
    
    Args:
        df: 经过 IC 过滤的 DataFrame
        
    Returns:
        添加了 v6_score 列的 DataFrame
    """
    df = df.copy()
    
    # 均线角度因子（SMA10变化率）
    df['_sma10_change'] = (df['sma_10'] / df['sma_10'].shift(1) - 1).abs().clip(0, 0.1) * 10 * 0.0854
    
    df['v6_score'] = (
        df['ma_condition'].astype(float) * 0.1622 +
        df['_sma10_change'].fillna(0) +
        df['macd_condition'].astype(float) * 0.1366 +
        df['volume_condition'].astype(float) * 0.1704 +
        (df['rsi_14'] < 70).astype(float) * 0.0597 +
        (df['kdj_k'] < 80).astype(float) * 0.0597 +
        (df['cci_20'] < 100).astype(float) * 0.0597 +
        (df['close'] < df['bb_upper']).astype(float) * 0.1191 +
        df['macd_jc'].astype(float) * 0.0873 +
        df['growth'].between(0.5, 6.0).astype(float) * 0.06
    )
    
    return df


def _compute_v8_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算V8总评分 = V6评分 + IC加分
    
    Args:
        df: 原始日线数据
        
    Returns:
        添加了 score（含 ic_bonus 的 v8 总分）列的 DataFrame
    """
    df = _compute_conditions(df)
    df = _apply_ic_filter(df)
    df = _compute_v6_score(df)
    df['score'] = df['v6_score'] + df['ic_bonus']
    return df


def _score_row_v8(row: pd.Series) -> float:
    """
    对单行计算V8评分（用于模拟盘逐行评分）
    与 DataFrame 批量计算结果完全一致
    
    Args:
        row: 单行数据（需已包含条件列）
        
    Returns:
        V8 评分浮点数
    """
    score = 0.0
    
    if not row.get('ma_condition', False): return 0
    if not row.get('growth_condition', False): return 0
    if not row.get('volume_condition', False): return 0
    if not row.get('macd_condition', False): return 0
    if not (row.get('jc_condition', False) or row.get('macd_jc', False)): return 0
    if not row.get('trend_condition', False): return 0
    if not row.get('rsi_filter', False): return 0
    if not row.get('price_filter', False): return 0
    
    # V6基础分
    score += 0.1622  # ma_condition
    
    # 均线角度
    sma10_prev = row.get('_sma10_change', 0)
    if not pd.isna(sma10_prev) and sma10_prev > 0:
        score += min(sma10_prev, 0.0854)
    
    # MACD
    if row.get('macd_condition', False): score += 0.1366
    if row.get('macd_jc', False): score += 0.0873
    
    # 成交量
    vr = min(row.get('vol_ratio', 0), 3)
    score += vr * 0.1704
    
    # 超买超卖指标
    if row.get('rsi_14', 100) < 70: score += 0.0597
    if row.get('kdj_k', 100) < 80: score += 0.0597
    if row.get('cci_20', 100) < 100: score += 0.0597
    if row.get('close', 0) < row.get('bb_upper', float('inf')): score += 0.1191
    
    # 涨幅
    g = row.get('growth', 0)
    if 0.5 <= g <= 6.0: score += 0.06
    elif g > 0: score += 0.03
    
    # IC加分
    if row.get('cci_20', 0) < -100: score += 0.10
    if row.get('williams_r', 0) < -80: score += 0.05
    if row.get('turnover_rate', 999) < 0.42: score += 0.05
    if row.get('vol_ratio', 999) < 0.71: score += 0.05
    
    return score


# ============ 策略类 ============

class ScoreV8Strategy:
    """
    V8 IC增强版策略类
    
    v6核心 + IC过滤增强，实现标准策略接口。
    
    【入场条件】
      - growth_condition: 收涨且涨幅 <= 6%
      - ma_condition: SMA5 > SMA10 且 SMA10 < SMA20
      - volume_condition: 放量1.5倍或超过5日均量1.2倍
      - macd_condition: MACD < 0 且 > signal
      - jc_condition: SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%
      - trend_condition: SMA20 < SMA55 且 SMA55 > SMA240
      - rsi_filter: RSI 在 50-60 之间
      - price_filter: 价格 3-15 元
    
    【IC增强过滤 - 剔除】
      - RSI > 70 或 < 25
      - 换手率 > 2.79%
      - WR < -95
      - CCI < -200
    
    【IC增强过滤 - 加分】
      - CCI < -100: +0.10
      - WR < -80: +0.05
      - 换手率 < 0.42%: +0.05
      - vol_ratio < 0.71: +0.05
    
    【出场规则】
      - 止损: 5%
      - 止盈: 15%
      - MA死叉: 入场时 sma20 < sma55，出场时 sma20 > sma55
    
    【持仓管理】
      - 最多持仓: 5只（config: max_positions）
      - 单只仓位上限: 20%（config: position_size）
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化 V8 策略
        
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
    # 公开接口
    # ------------------------------------------------------------------
    
    def filter_buy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤候选股票（满足全部入场条件）
        
        调用 _compute_conditions() + _apply_ic_filter()，
        返回满足全部 IC 过滤条件的候选股票。
        
        Args:
            df: 原始日线数据（需包含技术指标列）
            
        Returns:
            过滤后的 DataFrame（已通过全部条件，含 ic_bonus 列）
        """
        df = _compute_conditions(df)
        df = _apply_ic_filter(df)
        return df
    
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评分排序（在 filter_buy 基础上评分）
        
        调用 _compute_v8_score()，返回添加了 score 列的 DataFrame。
        score = v6_score + ic_bonus
        
        Args:
            df: 经过 filter_buy() 过滤后的 DataFrame
            
        Returns:
            添加了 'score'（和 'v6_score'）列的 DataFrame，按 score 降序排列
        """
        df = _compute_v8_score(df)
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
        
        出场条件（优先级：止损 > 止盈 > MA死叉）：
          1. 止损: next_open < 入场价 * (1 - stop_loss)
          2. 止盈: next_open > 入场价 * (1 + take_profit)
          3. MA死叉: 入场时 sma20 < sma55 且当前 sma20 > sma55
        
        Args:
            row: 当日行情数据（需含 next_open, sma_20, sma_55 等）
            pos: 持仓信息字典，必须包含:
                    - avg_cost: 入场成本价
                    - entry_sma20_le_sma55: bool，入场时 sma20 <= sma55 为 True
            market: 市场上下文（可选，当前未使用，保留接口兼容性）
            
        Returns:
            (should_sell: bool, reason: str)
        """
        # 处理 next_open 缺失：优先用 close，保留日志
        next_open = row.get('next_open')
        if pd.isna(next_open) or next_open <= 0:
            next_open = row.get('close')
            # next_open 缺失时应打印警告，但此处不破坏流程，静默降级
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
        
        # 3. MA死叉（入场时 sma20 < sma55，当前 sma20 > sma55 = 死叉已发生）
        #    pos['entry_sma20_le_sma55'] 在建仓时由调用方写入持仓记录
        if pos.get('entry_sma20_le_sma55', False):
            if row.get('sma_20', 0) > row.get('sma_55', 0):
                return True, "MA_DEATH_CROSS"
        
        return False, ""
    
    def get_entry_conditions(self) -> dict:
        """
        返回当前生效的入场条件（用于日志/报告）
        
        Returns:
            选股条件字典，键为条件名，值为阈值或描述
        """
        return {
            "growth_condition": "收涨且涨幅 <= 6%",
            "ma_condition": "SMA5 > SMA10 且 SMA10 < SMA20",
            "volume_condition": "放量1.5倍或超过5日均量1.2倍",
            "macd_condition": "MACD < 0 且 > signal",
            "jc_condition": "SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%",
            "trend_condition": "SMA20 < SMA55 且 SMA55 > SMA240",
            "rsi_filter": "RSI 在 50-60 之间",
            "price_filter": "价格 3-15 元",
            # IC 剔除
            "ic_exclude_rsi": "RSI > 70 或 < 25",
            "ic_exclude_turnover": "换手率 > 2.79%",
            "ic_exclude_wr": "WR < -95",
            "ic_exclude_cci": "CCI < -200",
            # IC 加分
            "ic_bonus_cci": "CCI < -100 → +0.10",
            "ic_bonus_wr": "WR < -80 → +0.05",
            "ic_bonus_turnover": "换手率 < 0.42% → +0.05",
            "ic_bonus_vol_ratio": "vol_ratio < 0.71 → +0.05",
            # 风控
            "stop_loss": f"{self.stop_loss:.0%}",
            "take_profit": f"{self.take_profit:.0%}",
            "max_positions": self.max_positions,
            "position_size": f"{self.position_size:.0%}",
        }
    
    def __repr__(self) -> str:
        return (
            f"<ScoreV8Strategy "
            f"stop_loss={self.stop_loss:.0%} "
            f"take_profit={self.take_profit:.0%} "
            f"max_positions={self.max_positions}>"
        )


# =============================================================================
# 兼容层：保留原有的模块级全局函数，供旧脚本使用
# 新代码应使用 ScoreV8Strategy 类
# =============================================================================

def compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _compute_conditions"""
    return _compute_conditions(df)


def apply_ic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _apply_ic_filter"""
    return _apply_ic_filter(df)


def compute_v6_score(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：调用 _compute_v6_score"""
    return _compute_v6_score(df)


def compute_v8_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容层：调用 _compute_v8_score
    注意：已移除 __self__ 调试残留（T0.2）
    """
    return _compute_v8_score(df)


def score_row_v8(row: pd.Series) -> float:
    """兼容层：调用 _score_row_v8"""
    return _score_row_v8(row)


def should_sell(row: pd.Series, pos: dict, market: Optional[dict] = None) -> Tuple[bool, str]:
    """
    兼容层：全局 should_sell 函数
    注意：调用方需要自行保证 pos 中包含 entry_sma20_le_sma55 字段
    建议使用 ScoreV8Strategy.should_sell() 方法
    """
    strategy = ScoreV8Strategy()
    return strategy.should_sell(row, pos, market)
