#!/usr/bin/env python3
"""
V8 策略核心模块
====================
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
  - vol_ratio>1.25
  - WR<-95
  - CCI<-200

【IC增强过滤 - 加分】
  - CCI<-100: +0.10分
  - WR<-80: +0.05分
  - 换手率<0.42%: +0.05分

【出场规则】
  - 止损: 5%
  - 止盈: 15%
  - MA死叉: SMA20下穿SMA55

【持仓管理】
  - 最多5只
  - 每只仓位上限20%

此模块被以下脚本使用:
  - backtest_score_v8.py (回测)
  - v8_simulated_trading.py (模拟盘)
"""

import pandas as pd
import numpy as np

# ============ 条件计算 ============

def compute_conditions(df):
    """计算所有选股条件（DataFrame批量）"""
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


def apply_ic_filter(df):
    """
    IC增强过滤 - 剔除 + 加分
    返回过滤后的DataFrame和每只股票的IC加分
    """
    df = df.copy()
    
    # 处理换手率重名字段
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    # ===== IC 增强过滤 — 剔除条件 =====
    exclude_mask = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 25) |
        (df['turnover_rate'] > 2.79) |
        (df['vol_ratio'] > 1.25) |
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
    
    return df


def compute_v6_score(df):
    """
    计算V6原始评分（DataFrame批量）
    与backtest_score_v8.py完全一致
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


def compute_v8_score(df):
    """
    计算V8总评分 = V6评分 + IC加分
    """
    df = compute_v8_score.__self__ if hasattr(compute_v8_score, '__self__') else df
    df = compute_conditions(df)
    df = apply_ic_filter(df)
    df = compute_v6_score(df)
    df['v8_score'] = df['v6_score'] + df['ic_bonus']
    return df


def score_row_v8(row):
    """
    对单行计算V8评分（用于模拟盘逐行评分）
    与DataFrame批量计算结果完全一致
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
    
    return score


# ============ 出场判断 ============

def should_sell(row, pos):
    """
    判断是否应该出场
    返回 (should_sell, reason)
    """
    next_open = row.get('next_open')
    if pd.isna(next_open) or next_open <= 0:
        return False, ""
    
    # 止损5%
    if next_open < pos['avg_cost'] * 0.95:
        return True, "止损5%"
    
    # 止盈15%
    if next_open > pos['avg_cost'] * 1.15:
        return True, "止盈15%"
    
    # MA死叉
    if row.get('sma_20', 0) > row.get('sma_55', 0):
        return True, "MA死叉"
    
    return False, ""
