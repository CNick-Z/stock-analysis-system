#!/usr/bin/env python3
"""
V9 策略核心模块 — 从 IC 分析驱动的参数优化
====================
基于 IC/IR 分析结果（2018-2025）
- CCI信号: IC=+0.043, ICIR=2.52（最强因子）
- WR信号: IC=+0.015, ICIR=0.84
- 放量: IC=-0.048（陷阱）
- 高换手: IC=-0.10（陷阱）

策略目标：最大化 IC 有效因子，避免 IC 陷阱
"""

import pandas as pd
import numpy as np

# ============ 条件计算 ============

def compute_conditions(df):
    """计算所有选股条件"""
    df = df.copy()
    
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | (df['volume'] > df['vol_ma5'] * 1.2)
    
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    # 金叉（SMA5连续2日上升 AND SMA20当日上升 AND 均线收敛<2%）
    df['jc_condition'] = (
        (df['sma_5'] > df['sma_5'].shift(1)) &
        (df['sma_5'].shift(1) > df['sma_5'].shift(2)) &
        (df['sma_20'] > df['sma_20'].shift(1)) &
        (abs(df['sma_5'] - df['sma_20']) / df['sma_20'] < 0.02)
    )
    
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)
    
    return df


def apply_ic_filter(df, params=None):
    """
    V9 IC增强过滤
    params: dict 包含所有IC相关阈值
    """
    if params is None:
        params = {
            'rsi_high_exclude': 70,
            'rsi_low_exclude': 25,
            'turnover_exclude': 2.79,
            'vol_ratio_exclude': 1.25,
            'wr_exclude': -95,
            'cci_exclude': -200,
            'cci_bonus': -100,
            'wr_bonus': -80,
            'turnover_bonus': 0.42,
            'cci_bonus_score': 0.10,
            'wr_bonus_score': 0.05,
            'low_turnover_bonus_score': 0.05,
        }
    
    df = df.copy()
    
    # IC 剔除条件
    exclude_mask = (
        (df['rsi_14'] > params['rsi_high_exclude']) |
        (df['rsi_14'] < params['rsi_low_exclude']) |
        (df['turnover_rate'] > params['turnover_exclude']) |
        (df['vol_ratio'] > params['vol_ratio_exclude']) |
        (df['williams_r'] < params['wr_exclude']) |
        (df['cci_20'] < params['cci_exclude'])
    )
    df = df[~exclude_mask].copy()
    
    # 买入条件
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
    
    # IC 加分
    df['ic_bonus'] = 0.0
    df.loc[df['cci_20'] < params['cci_bonus'], 'ic_bonus'] += params['cci_bonus_score']
    df.loc[df['williams_r'] < params['wr_bonus'], 'ic_bonus'] += params['wr_bonus_score']
    df.loc[df['turnover_rate'] < params['turnover_bonus'], 'ic_bonus'] += params['low_turnover_bonus_score']
    
    return df


def compute_v6_score(df):
    """计算V6原始评分"""
    df = df.copy()
    
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


def should_sell(row, pos):
    """出场判断"""
    next_open = row.get('next_open')
    if pd.isna(next_open) or next_open <= 0:
        return False, ""
    
    if next_open < pos['avg_cost'] * 0.95:
        return True, "止损5%"
    if next_open > pos['avg_cost'] * 1.15:
        return True, "止盈15%"
    if row.get('sma_20', 0) > row.get('sma_55', 0):
        return True, "MA死叉"
    
    return False, ""
