#!/usr/bin/env python3
"""
18年模拟盘 — V9 (V4核心 + IC增强评分)
基于V4框架，但评分时叠加IC因子加成

V4核心（18年+80%，807笔）:
  - 趋势: SMA20<SMA55 且 SMA55>SMA240
  - 买入: MACD<0 + (金叉jc OR MACD金叉macd_jc)
  - 布林带低位: close < bb_upper

IC增强（加分，不过滤）:
  - CCI<-100 +0.10（最强信号）
  - WR<-80 +0.05
  - 换手<0.42% +0.05
  - 陷阱标记: RSI>70/RSI<25/换手>2.79%/vol_ratio>1.25/WR<-95/CCI<-200
"""

import os, sys, gc, json
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/paper_trading'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 100_0000
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001
TOP_N = 5

def compute_conditions(df):
    df = df.copy()
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | (df['volume'] > df['vol_ma5'] * 1.2)
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['jc_condition'] = (df['sma_5'] > df['sma_5'].shift(1)) & (df['sma_20'] > df['sma_20'].shift(1))
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)
    # IC陷阱标记
    df['ic_trap'] = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 25) |
        (df['turnover_rate'] > 2.79) |
        (df['vol_ratio'] > 1.25) |
        (df['williams_r'] < -95) |
        (df['cci_20'] < -200)
    )
    return df

def score_v4(row):
    """V4原始评分"""
    score = 0.0
    if not row.get('macd_condition', False): return 0.0
    if not (row.get('jc_condition', False) or row.get('macd_jc', False)): return 0.0
    if not row.get('trend_condition', False): return 0.0
    score += 0.1622
    g = row.get('growth', 0)
    if 0.5 <= g <= 6.0: score += 0.06
    elif g > 0: score += 0.03
    score += row.get('vr', 0) * 0.1704
    if row.get('macd_condition', False): score += 0.1366
    if row.get('macd_jc', False): score += 0.0873
    if row.get('rsi_14', 100) < 70: score += 0.0597
    if row.get('kdj_k', 100) < 80: score += 0.0597
    if row.get('cci_20', 100) < 100: score += 0.0597
    if row.get('close', 0) < row.get('bb_upper', float('inf')): score += 0.1191
    return score

def score_v9(row):
    """V9评分: V4评分 + IC增强加分"""
    base = score_v4(row)
    if base == 0: return 0.0
    bonus = 0.0
    if not row.get('ic_trap', False):
        if row.get('cci_20', 0) < -100: bonus += 0.10
        if row.get('williams_r', 0) < -80: bonus += 0.05
        if row.get('turnover_rate', 999) < 0.42: bonus += 0.05
    return base + bonus

def run_v9_backtest(years):
    print("=" * 60)
    print("V9 回测 — V4核心 + IC增强 (2008-2026)")
    print("=" * 60)
    
    all_trades = []
    yearly_results = []
    
    for year in years:
        tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
        daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
        df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
        del tech, daily; gc.collect()
        
        df['vr'] = df['volume'] / (df['vol_ma5'] + 1e-10)
        df['vol_ratio'] = df['vr']  # alias for compute_conditions
        if 'turnover_rate_y' in df.columns: df['turnover_rate'] = df['turnover_rate_y']
        elif 'turnover_rate_x' in df.columns: df['turnover_rate'] = df['turnover_rate_x']
        df['next_open'] = df.groupby('symbol')['open'].shift(-1)
        
        df = compute_conditions(df)
        
        portfolio = {'cash': INITIAL_CASH, 'positions': {}, 'trades': []}
        dates = sorted(df['date'].unique())
        
        for i, date in enumerate(dates):
            nd = dates[i+1] if i+1 < len(dates) else None
            if not nd: continue
            
            d = df[df['date']==date]
            next_day = df[df['date']==nd] if i+1 < len(dates) else pd.DataFrame()
            
            # ===== 持仓出场 =====
            for sym in list(portfolio['positions'].keys()):
                today = d[d['symbol']==sym]
                if today.empty: continue
                row = today.iloc[0]
                no = row['next_open']
                if pd.isna(no) or no <= 0: continue
                
                should_sell = False; reason = ""
                if no < portfolio['positions'][sym]['avg_cost'] * 0.95:
                    should_sell = True; reason = "stop_loss"
                elif no > portfolio['positions'][sym]['avg_cost'] * 1.15:
                    should_sell = True; reason = "take_profit"
                elif row['sma_20'] > row['sma_55']:
                    should_sell = True; reason = "ma_death"
                
                if should_sell:
                    proceeds = portfolio['positions'][sym]['qty'] * no * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                    pnl = proceeds - portfolio['positions'][sym]['qty'] * portfolio['positions'][sym]['avg_cost']
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'sell',
                        'price': no, 'qty': portfolio['positions'][sym]['qty'],
                        'pnl': pnl, 'reason': reason, 'strategy': 'v9'
                    })
                    del portfolio['positions'][sym]
            
            # ===== 选股买入 =====
            if len(portfolio['positions']) < TOP_N:
                candidates = d.copy()
                candidates['v9_score'] = candidates.apply(score_v9, axis=1)
                candidates = candidates[candidates['v9_score'] > 0].sort_values('v9_score', ascending=False)
                
                slots = TOP_N - len(portfolio['positions'])
                bought = 0
                for _, row in candidates.iterrows():
                    if bought >= slots: break
                    sym = row['symbol']
                    if sym in portfolio['positions']: continue
                    no = row['next_open']
                    if pd.isna(no) or no <= 0: continue
                    qty = int(INITIAL_CASH * 0.20 / no / 100) * 100
                    if qty < 100: continue
                    cost = qty * no * (1 + COMMISSION + SLIPPAGE)
                    if cost > portfolio['cash']: continue
                    portfolio['cash'] -= cost
                    portfolio['positions'][sym] = {'qty': qty, 'avg_cost': no, 'entry_date': date}
                    portfolio['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'buy',
                        'price': no, 'qty': qty, 'price_cost': no,
                        'strategy': 'v9'
                    })
                    bought += 1
        
        # 年末估值
        fv = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            lr = df[df['symbol']==sym]
            if not lr.empty: fv += pos['qty'] * lr.iloc[-1]['close']
        
        ret = (fv - INITIAL_CASH) / INITIAL_CASH
        buys = sum(1 for t in portfolio['trades'] if t['action'] == 'buy')
        sells = [t for t in portfolio['trades'] if t['action'] == 'sell']
        wins = sum(1 for t in sells if t.get('pnl', 0) > 0)
        
        print(f"  {year}: {ret:+.2%} 买{buys} 卖{len(sells)} 胜{wins}/{len(sells)}={wins/max(1,len(sells)):.0%} "
              f"持仓{len(portfolio['positions'])} 最终{fv/1e4:.1f}万")
        
        yearly_results.append({'year': year, 'ret': ret, 'buys': buys, 'sells': len(sells),
                               'wins': wins, 'final': fv, 'pos': len(portfolio['positions'])})
        all_trades.extend(portfolio['trades'])
    
    # 汇总
    print("-" * 60)
    total_ret = 1.0
    for r in yearly_results:
        total_ret *= (1 + r['ret'])
    
    all_buys = sum(r['buys'] for r in yearly_results)
    all_sells = sum(r['sells'] for r in yearly_results)
    all_wins = sum(r['wins'] for r in yearly_results)
    
    print(f"V9 合计: {total_ret-1:+.2%} ({years[0]}-{years[-1]}共{len(years)}年)")
    print(f"总交易: 买{all_buys}笔 卖{all_sells}笔 胜率{all_wins}/{all_sells}={all_wins/max(1,all_sells):.1%}")
    
    return yearly_results, all_trades

if __name__ == '__main__':
    years = list(range(2008, 2027))
    results, trades = run_v9_backtest(years)
    
    # 保存
    out = {
        'strategy': 'v9',
        'desc': 'V4核心+IC增强(不过滤,只加分)',
        'years': years,
        'results': results,
        'total_trades': len(trades),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    fname = f"v9_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f'{OUT_DIR}/{fname}', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n💾 结果已保存: {OUT_DIR}/{fname}")
