#!/usr/bin/env python3
"""
双策略模拟盘 — v4(原始) vs v8(IC增强) 从2026年起同步运行
同时支持追加最新交易日数据

用法:
    python3 paper_trading_sim.py              # 从2026年初运行到最新数据
    python3 paper_trading_sim.py --date 2026-03-01  # 从指定日期运行
"""

import os, sys, gc, json, argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/paper_trading'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 100_0000  # 各100万
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001
TOP_N = 5

SIM_START = '2026-01-01'


def load_latest_data():
    """加载2026最新数据"""
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year=2026/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year=2026/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['next_close'] = df.groupby('symbol')['close'].shift(-1)
    # 处理合并后的重名字段(pandas默认用_x/_y)
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    # v4 条件
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | (df['volume'] > df['vol_ma5'] * 1.2)
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['jc_condition'] = (df['sma_5'] > df['sma_5'].shift(1)) & (df['sma_20'] > df['sma_20'].shift(1))
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)
    return df


def score_v4(row):
    """v4 原始评分（趋势跟踪）"""
    score = 0.0
    if not (row.get('ma_condition', False)): return 0
    if not row.get('growth_condition', False): return 0
    if not row.get('volume_condition', False): return 0
    if not row.get('macd_condition', False): return 0
    if not (row.get('jc_condition', False) or row.get('macd_jc', False)): return 0
    if not row.get('trend_condition', False): return 0
    if not row.get('rsi_filter', False): return 0
    if not row.get('price_filter', False): return 0
    # 基础分
    score += 0.1622  # ma
    # 涨幅
    g = row.get('growth', 0)
    if 0.5 <= g <= 6.0: score += 0.06
    elif g > 0: score += 0.03
    # 成交量
    vr = min(row.get('volume', 0) / max(row.get('vol_ma5', 1), 1), 3)
    score += vr * 0.1704
    # MACD
    if row.get('macd_condition', False): score += 0.1366
    if row.get('macd_jc', False): score += 0.0873
    # 超买超卖
    if row.get('rsi_14', 100) < 70: score += 0.0597
    if row.get('kdj_k', 100) < 80: score += 0.0597
    if row.get('cci_20', 100) < 100: score += 0.0597
    if row.get('close', 0) < row.get('bb_upper', float('inf')): score += 0.1191
    # 角度
    angle = abs(row.get('sma_10', 0) / row.get('sma_10', 1) - 1) * 100
    if angle > 30: score += 0.0854
    return score


def filter_v8(daily):
    """v8 IC增强过滤"""
    df = daily.copy()
    # 处理合并后的重名字段(pandas默认用_x/_y)
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    # IC 剔除条件
    mask = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 25) |
        (df['turnover_rate'] > 2.79) |
        (df['vol_ratio'] > 1.25) |
        (df['williams_r'] < -95) |
        (df['cci_20'] < -200)
    )
    df = df[~mask].copy()
    # v8 条件
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
    return df[buy]


def score_v8(row, v6_score):
    """v8 评分 = v6_score + IC增强分"""
    bonus = 0.0
    if row.get('cci_20', 0) < -100: bonus += 0.10
    if row.get('williams_r', 0) < -80: bonus += 0.05
    if row.get('turnover_rate', 999) < 0.42: bonus += 0.05
    return v6_score + bonus


def run_sim():
    print(f"{'='*70}")
    print(f"📈 双策略模拟盘 — v4 vs v8 (2026年起)")
    print(f"{'='*70}")
    
    df = load_latest_data()
    df = df[df['date'] >= SIM_START].sort_values('date').reset_index(drop=True)
    print(f"数据: {df['date'].min()} ~ {df['date'].max()}, {len(df):,}条")
    
    # 两个独立账户
    portfolios = {
        'v4': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []},
        'v8': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []}
    }
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    for i, date in enumerate(dates):
        daily = df[df['date'] == date].copy()
        next_day = dates[i+1] if i+1 < n_dates else None
        
        for strategy in ['v4', 'v8']:
            p = portfolios[strategy]
            
            # ===== 平仓 =====
            for sym, pos in list(p['positions'].items()):
                if not next_day:
                    continue
                today = daily[daily['symbol'] == sym]
                if today.empty:
                    continue
                row = today.iloc[0]
                next_open = row['next_open']
                if pd.isna(next_open) or next_open <= 0:
                    continue
                
                should_sell = False
                reason = ""
                if next_open < pos['avg_cost'] * 0.95:
                    should_sell = True
                    reason = "stop_loss"
                elif next_open > pos['avg_cost'] * 1.15:
                    should_sell = True
                    reason = "take_profit"
                elif row['sma_20'] > row['sma_55']:
                    should_sell = True
                    reason = "ma_death"
                
                if should_sell:
                    proceeds = pos['qty'] * next_open * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                    pnl = proceeds - pos['qty'] * pos['avg_cost']
                    p['cash'] += proceeds
                    p['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'sell',
                        'price': next_open, 'qty': pos['qty'], 'pnl': pnl, 'reason': reason,
                        'strategy': strategy
                    })
                    del p['positions'][sym]
            
            # ===== 选股买入 =====
            if next_day and len(p['positions']) < TOP_N:
                if strategy == 'v4':
                    candidates = daily.copy()
                    buy_mask = (
                        candidates['ma_condition'] &
                        candidates['growth_condition'] &
                        candidates['volume_condition'] &
                        candidates['macd_condition'] &
                        (candidates['jc_condition'] | candidates['macd_jc']) &
                        candidates['trend_condition'] &
                        candidates['rsi_filter'] &
                        candidates['price_filter']
                    )
                    candidates = candidates[buy_mask].copy()
                    candidates['score'] = candidates.apply(score_v4, axis=1)
                else:
                    candidates = filter_v8(daily)
                    if not candidates.empty:
                        # 快速v6 score
                        candidates['v6_score'] = (
                            candidates['ma_condition'].astype(float) * 0.1622 +
                            candidates['macd_condition'].astype(float) * 0.1366 +
                            candidates['volume_condition'].astype(float) * 0.1704 +
                            (candidates['rsi_14'] < 70).astype(float) * 0.0597 +
                            (candidates['kdj_k'] < 80).astype(float) * 0.0597 +
                            (candidates['cci_20'] < 100).astype(float) * 0.0597 +
                            (candidates['close'] < candidates['bb_upper']).astype(float) * 0.1191 +
                            candidates['macd_jc'].astype(float) * 0.0873 +
                            candidates['growth'].between(0.5, 6.0).astype(float) * 0.06
                        )
                        candidates['score'] = candidates.apply(
                            lambda r: score_v8(r, r['v6_score']), axis=1
                        )
                
                if not candidates.empty:
                    candidates = candidates.sort_values('score', ascending=False)
                
                slots = TOP_N - len(p['positions'])
                for _, row in candidates.head(slots).iterrows():
                    sym = row['symbol']
                    if sym in p['positions']:
                        continue
                    next_open = row['next_open']
                    if pd.isna(next_open) or next_open <= 0:
                        continue
                    
                    max_per = p['cash'] * 0.20
                    buy_qty = int(max_per / next_open / 100) * 100
                    if buy_qty < 100:
                        continue
                    
                    cost = buy_qty * next_open * (1 + COMMISSION + SLIPPAGE)
                    if cost > p['cash']:
                        continue
                    
                    p['cash'] -= cost
                    p['positions'][sym] = {
                        'qty': buy_qty, 'avg_cost': next_open,
                        'buy_date': date, 'buy_price': next_open
                    }
                    p['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'buy',
                        'price': next_open, 'qty': buy_qty,
                        'score': row.get('score', 0),
                        'cci': row.get('cci_20', 0),
                        'wr': row.get('williams_r', 0),
                        'rsi': row.get('rsi_14', 0),
                        'strategy': strategy
                    })
            
            # ===== 记录当日价值 =====
            pv = 0
            for sym, pos in p['positions'].items():
                today = daily[daily['symbol'] == sym]
                if not today.empty:
                    pv += today.iloc[0]['close'] * pos['qty']
                else:
                    pv += pos['qty'] * pos['avg_cost']
            
            total = p['cash'] + pv
            p['history'].append({
                'date': date,
                'cash': p['cash'],
                'positions_value': pv,
                'total_value': total,
                'n_positions': len(p['positions'])
            })
        
        # ===== 每10天打印进度 =====
        if i % 10 == 0 or i == n_dates - 1:
            v4_val = portfolios['v4']['history'][-1]['total_value']
            v8_val = portfolios['v8']['history'][-1]['total_value']
            print(f"  {date}: v4={v4_val/10000:.2f}万({(v4_val/INITIAL_CASH-1):+.2%}) | "
                  f"v8={v8_val/10000:.2f}万({(v8_val/INITIAL_CASH-1):+.2%})")
    
    # ===== 汇总报告 =====
    print(f"\n{'='*70}")
    print(f"📊 双策略模拟盘汇总 (2026-01 ~ 2026-03)")
    print(f"{'='*70}")
    
    results = {}
    for strat, p in portfolios.items():
        buys = [t for t in p['trades'] if t['action'] == 'buy']
        sells = [t for t in p['trades'] if t['action'] in ('sell',)]
        winners = [t for t in sells if t.get('pnl', 0) > 0]
        final_val = p['history'][-1]['total_value']
        total_ret = final_val / INITIAL_CASH - 1
        
        print(f"\n【{strat.upper()}】")
        print(f"  初始资金: {INITIAL_CASH/10000:.2f}万")
        print(f"  最新价值: {final_val/10000:.2f}万")
        print(f"  总收益率: {total_ret:+.2%}")
        print(f"  买入次数: {len(buys)}, 卖出次数: {len(sells)}")
        print(f"  胜率: {len(winners)/max(1,len(sells)):.0%}")
        print(f"  最新持仓: {list(p['positions'].keys())}")
        
        results[strat] = {
            'initial': INITIAL_CASH,
            'final': final_val,
            'return': total_ret,
            'n_buys': len(buys),
            'n_sells': len(sells),
            'win_rate': len(winners)/max(1,len(sells)),
            'trades': p['trades'],
            'history': p['history'],
            'positions': {sym: {'qty': pos['qty'], 'avg_cost': pos['avg_cost']} 
                          for sym, pos in p['positions'].items()}
        }
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f'{OUT_DIR}/paper_trading_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 结果已保存: {out_file}")
    
    # 打印持仓
    print(f"\n{'='*70}")
    print(f"📦 当前持仓")
    print(f"{'='*70}")
    for strat in ['v4', 'v8']:
        p = portfolios[strat]
        if p['positions']:
            print(f"\n【{strat.upper()}】")
            for sym, pos in p['positions'].items():
                today_data = df[df['date'] == dates[-1]]
                cur = today_data[today_data['symbol']==sym]
                cur_price = cur.iloc[0]['close'] if not cur.empty else pos['avg_cost']
                pnl_pct = (cur_price / pos['avg_cost'] - 1) * 100
                print(f"  {sym}: {pos['qty']}股 @买入{pos['avg_cost']:.2f} → 现价{cur_price:.2f}({pnl_pct:+.1f}%)")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='起始日期 YYYY-MM-DD')
    args = parser.parse_args()
    if args.date:
        SIM_START = args.date
    run_sim()
