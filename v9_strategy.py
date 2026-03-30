#!/usr/bin/env python3
"""
V9 策略核心 — V4核心 + IC增强
=============================
V4核心（18年+148%，807笔，49.4%胜率）:
  - 趋势: SMA5>SMA10 且 SMA20<SMA55 且 SMA55>SMA240
  - 买入: MACD<0 + (金叉jc OR MACD金叉macd_jc)
  - 布林带低位: close < bb_upper
  - 质量: growth(0.5~6%) + vol_ratio + RSI/KDJ/CCI过滤

IC增强（2018-2025数据分析）:
  - CCI<-100: IC=+0.043, ICIR=2.52 → 最强信号，+0.10
  - WR<-80: IC=+0.015, ICIR=0.84 → +0.05
  - 换手<0.42%: IC=+0.05 → +0.05

出场: 止损5% / 止盈15% / MA死叉(SMA20下穿SMA55)
"""

import pandas as pd
import numpy as np

# ============ 条件计算 ============

def compute_conditions(df):
    """计算所有选股条件"""
    df = df.copy()
    
    # 涨幅（当日收阳线）
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    
    # 质量条件
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    
    # MA质量: SMA5>SMA10 且 SMA10相对SMA20处于健康位置
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    
    # 放量
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | (df['volume'] > df['vol_ma5'] * 1.2)
    
    # MACD状态: MACD<0 且 MACD>signal（底部金叉酝酿中）
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    
    # MACD金叉
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    # 均线金叉: SMA5向上穿越SMA20
    df['jc_condition'] = (df['sma_5'] > df['sma_5'].shift(1)) & (df['sma_20'] > df['sma_20'].shift(1))
    
    # 趋势: SMA5>SMA10>SMA20>SMA55>SMA240多头排列
    # V4使用: SMA20<SMA55 且 SMA55>SMA240（20<55<240排序）
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    
    # RSI过滤器（V4原始：50-60区间）
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    
    # 价格过滤器（3-15元）
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)
    
    return df


def apply_ic_filter(df, params=None):
    """
    IC增强过滤
    - 剔除陷阱: RSI>70或<25, 换手>2.79%, vol_ratio>1.25, WR<-95, CCI<-200
    - 加分信号: CCI<-100 +0.10, WR<-80 +0.05, 换手<0.42% +0.05
    """
    if params is None:
        params = {
            # 剔除阈值
            'rsi_high': 70,
            'rsi_low': 25,
            'turnover_exclude': 2.79,
            'vol_ratio_exclude': 1.25,
            'wr_exclude': -95,
            'cci_exclude': -200,
            # 加分阈值
            'cci_bonus': -100,
            'wr_bonus': -80,
            'turnover_bonus': 0.42,
            # 加分分值
            'cci_bonus_score': 0.10,
            'wr_bonus_score': 0.05,
            'turnover_bonus_score': 0.05,
        }
    
    df = df.copy()
    
    # IC 陷阱剔除
    exclude_mask = (
        (df['rsi_14'] > params['rsi_high']) |
        (df['rsi_14'] < params['rsi_low']) |
        (df['turnover_rate'] > params['turnover_exclude']) |
        (df['vol_ratio'] > params['vol_ratio_exclude']) |
        (df['williams_r'] < params['wr_exclude']) |
        (df['cci_20'] < params['cci_exclude'])
    )
    df = df[~exclude_mask].copy()
    
    # V4 核心买入条件
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
    df.loc[df['turnover_rate'] < params['turnover_bonus'], 'ic_bonus'] += params['turnover_bonus_score']
    
    return df


def compute_v4_score(df):
    """
    V4 原始评分函数
    与paper_trading_18years.py中score_v4()完全一致
    """
    df = df.copy()
    
    # vol_ratio
    df['vr'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    
    def score_row(row):
        score = 0.0
        # 前置过滤
        if not row.get('macd_condition', False): return 0.0
        if not (row.get('jc_condition', False) or row.get('macd_jc', False)): return 0.0
        if not row.get('trend_condition', False): return 0.0
        
        # MA
        score += 0.1622
        
        # 涨幅
        g = row.get('growth', 0)
        if 0.5 <= g <= 6.0:
            score += 0.06
        elif g > 0:
            score += 0.03
        
        # 放量
        score += row.get('vr', 0) * 0.1704
        
        # MACD
        if row.get('macd_condition', False): score += 0.1366
        if row.get('macd_jc', False): score += 0.0873
        
        # RSI
        if row.get('rsi_14', 100) < 70: score += 0.0597
        
        # KDJ
        if row.get('kdj_k', 100) < 80: score += 0.0597
        
        # CCI
        if row.get('cci_20', 100) < 100: score += 0.0597
        
        # 布林带
        if row.get('close', 0) < row.get('bb_upper', float('inf')): score += 0.1191
        
        return score
    
    df['v4_score'] = df.apply(score_row, axis=1)
    
    # V9: v9_score = v4_score + ic_bonus
    df['v9_score'] = df['v4_score'] + df['ic_bonus']
    
    return df


def should_sell(row, pos):
    """
    出场判断: 止损5% / 止盈15% / MA死叉
    row需要有: next_open, sma_20, sma_55
    """
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
