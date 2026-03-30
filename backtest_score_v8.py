#!/usr/bin/env python3
"""
Score 策略 v8 回测 — v6 核心 + IC 增强过滤

【策略核心: v8_strategy.py (与模拟盘共用同一策略)】
"""

import os, sys, gc, psutil
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v8_strategy import compute_conditions, apply_ic_filter, compute_v6_score, should_sell

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/backtest_results'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 100_0000
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001
YEARS = range(2018, 2026)
TOP_N = 5


def check_mem(label=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] MEM={mem:.2f}GB | {label}")


def load_year(year):
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    for n in [5, 10, 20]:
        df[f'return_{n}d'] = df.groupby('symbol')['close'].pct_change(n).shift(-n)
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    # 使用v8_strategy统一计算选股条件
    df = compute_conditions(df)
    return df


# [apply_ic_filter 已迁移到 v8_strategy.py]


def run_backtest_year(year):
    print(f"\n{'='*60}")
    print(f"📅 {year} 年回测 (v8)")
    check_mem(f"{year}开始")
    
    df = load_year(year)
    print(f"  总记录: {len(df):,}")
    
    portfolio = {
        'cash': INITIAL_CASH,
        'positions': {},
        'trades': []
    }
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    for i, date in enumerate(dates):
        daily = df[df['date'] == date].copy()
        next_day = dates[i+1] if i+1 < n_dates else None
        
        # ===== 持仓止损/止盈 =====
        for sym, pos in list(portfolio['positions'].items()):
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
            
            # 止损 5%
            if next_open < pos['avg_cost'] * 0.95:
                should_sell = True
                reason = "stop_loss_5%"
            # 止盈 15%
            elif next_open > pos['avg_cost'] * 1.15:
                should_sell = True
                reason = "take_profit_15%"
            # MA20 死叉 + 资金流负
            elif row['sma_20'] > row['sma_55'] and row.get('money_flow_positive', True) == False:
                should_sell = True
                reason = "ma_death"
            
            if should_sell:
                proceeds = pos['qty'] * next_open * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                pnl = proceeds - pos['qty'] * pos['avg_cost']
                portfolio['cash'] += proceeds
                portfolio['trades'].append({
                    'date': date, 'symbol': sym, 'action': 'sell',
                    'price': next_open, 'qty': pos['qty'], 'pnl': pnl, 'reason': reason
                })
                del portfolio['positions'][sym]
        
        # ===== 选股买入 =====
        if next_day and len(portfolio['positions']) < TOP_N:
            candidates = apply_ic_filter(daily)
            
            # 计算V6评分 + V8总分（使用v8_strategy统一逻辑）
            if not candidates.empty:
                candidates = compute_v6_score(candidates)
                candidates['v8_score'] = candidates['v6_score'] + candidates['ic_bonus']
                candidates = candidates.sort_values('v8_score', ascending=False)
            
            slots = TOP_N - len(portfolio['positions'])
            for _, row in candidates.head(slots).iterrows():
                sym = row['symbol']
                if sym in portfolio['positions']:
                    continue
                next_open = row['next_open']
                if pd.isna(next_open) or next_open <= 0:
                    continue
                
                max_per = portfolio['cash'] * 0.20
                buy_qty = int(max_per / next_open / 100) * 100
                if buy_qty < 100:
                    continue
                
                cost = buy_qty * next_open * (1 + COMMISSION + SLIPPAGE)
                if cost > portfolio['cash']:
                    continue
                
                portfolio['cash'] -= cost
                portfolio['positions'][sym] = {
                    'qty': buy_qty, 'avg_cost': next_open,
                    'buy_date': date, 'buy_price': next_open
                }
                portfolio['trades'].append({
                    'date': date, 'symbol': sym, 'action': 'buy',
                    'price': next_open, 'qty': buy_qty,
                    'v8_score': row['v8_score'], 'v6_score': row['v6_score'],
                    'cci': row['cci_20'], 'wr': row['williams_r']
                })
        
        # ===== 进度 =====
        if i % 20 == 0 or i == n_dates - 1:
            pv = sum(
                daily[daily['symbol']==sym].iloc[0]['close'] * pos['qty']
                if not daily[daily['symbol']==sym].empty else pos['qty'] * pos['avg_cost']
                for sym, pos in portfolio['positions'].items()
            )
            total = portfolio['cash'] + pv
            print(f"  {date}: 持仓{len(portfolio['positions'])}只 现金{portfolio['cash']:.0f} 总{total:.0f}")
    
    # 年终清仓
    last = df[df['date'] == dates[-1]]
    for sym, pos in list(portfolio['positions'].items()):
        t = last[last['symbol'] == sym]
        if not t.empty:
            sp = t.iloc[0]['close']
            proceeds = pos['qty'] * sp * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
            portfolio['trades'].append({
                'date': dates[-1], 'symbol': sym, 'action': 'year_end_sell',
                'price': sp, 'qty': pos['qty'], 'pnl': proceeds - pos['qty'] * pos['avg_cost']
            })
            portfolio['cash'] += proceeds
        del portfolio['positions'][sym]
    
    total_val = portfolio['cash']
    ann_ret = (total_val - INITIAL_CASH) / INITIAL_CASH
    
    buys = [t for t in portfolio['trades'] if t['action'] == 'buy']
    sells = [t for t in portfolio['trades'] if t['action'] in ('sell', 'year_end_sell')]
    winners = [t for t in sells if t['pnl'] > 0]
    
    print(f"\n  📊 {year} 结果: 收益={ann_ret:.2%} 买入={len(buys)} 胜率={len(winners)/max(1,len(sells)):.0%}")
    del df; gc.collect()
    
    return {
        'year': year, 'annual_return': ann_ret,
        'n_buys': len(buys), 'n_sells': len(sells),
        'n_winners': len(winners), 'win_rate': len(winners)/max(1,len(sells)),
        'total_value': total_val, 'trades': portfolio['trades']
    }


def run():
    print(f"{'='*60}")
    print(f"🏃 Score v8 回测 — v6核心 + IC增强")
    print(f"{'='*60}")
    check_mem("启动")
    
    results = []
    for year in YEARS:
        r = run_backtest_year(year)
        results.append(r)
        check_mem(f"{year}完成")
    
    total_ret = 1.0
    for r in results:
        total_ret *= (1 + r['annual_return'])
    
    total_wins = sum(r['n_winners'] for r in results)
    total_sells = sum(r['n_sells'] for r in results)
    
    print(f"\n{'='*60}")
    print(f"🏆 v8 汇总 (2018-2025)")
    print(f"{'='*60}")
    print(f"{'年份':<8}{'收益':>10}{'买入':>6}{'卖出':>6}{'胜率':>8}")
    print("-"*45)
    for r in results:
        print(f"{r['year']:<8}{r['annual_return']:>10.2%}{r['n_buys']:>6}{r['n_sells']:>6}{r['win_rate']:>8.0%}")
    print("-"*45)
    print(f"{'合计':<8}{(total_ret-1):>10.2%}")
    print(f"最终: {results[-1]['total_value']/10000:.2f}万 vs 100万")
    
    return results


if __name__ == '__main__':
    run()
