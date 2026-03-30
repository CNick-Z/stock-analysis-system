#!/usr/bin/env python3
"""
18年模拟盘 — v4(原始) vs v8(IC增强) vs v4+广度调节器
从2008年开始，回测18年到2026年

仓位调节器规则:
  全市场MA55>MA240广度 > 50% → 满仓(5只)
  全市场MA55>MA240广度 20-50% → 半仓(3只)
  全市场MA55>MA240广度 5-20% → 轻仓(1-2只)
  全市场MA55>MA240广度 < 5% → 空仓
"""

import os, sys, gc, json
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/paper_trading'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 100_0000  # 各100万
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001

# 仓位调节器参数
BREADTH_FULL = 50    # >50% 满仓5只
BREADTH_HALF = 20    # >20% 半仓3只
BREADTH_LIGHT = 5    # >5% 轻仓1-2只
TOP_N_FULL = 5
TOP_N_HALF = 3
TOP_N_LIGHT = 2
TOP_N_MIN = 1

YEARS = list(range(2008, 2027))  # 2008-2026


def load_year(year):
    """加载某年的技术指标+日线数据"""
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    # v4 基础条件
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


def compute_breadth(df):
    """计算每日市场广度（MA55>MA240比例）"""
    breadth = {}
    for date, grp in df.groupby('date'):
        total = len(grp)
        valid = grp['sma_240'].notna().sum()
        if valid == 0:
            breadth[date] = 0.0
        else:
            up = (grp['sma_55'] > grp['sma_240']).sum()
            breadth[date] = up / total * 100
    return breadth


def get_top_n(breadth):
    """根据广度返回持仓上限"""
    if breadth > BREADTH_FULL:
        return TOP_N_FULL
    elif breadth > BREADTH_HALF:
        return TOP_N_HALF
    elif breadth > BREADTH_LIGHT:
        return TOP_N_LIGHT
    else:
        return 0  # 空仓


def score_v4(row):
    """v4 原始评分"""
    score = 0.0
    if not row.get('ma_condition', False): return 0
    if not row.get('growth_condition', False): return 0
    if not row.get('volume_condition', False): return 0
    if not row.get('macd_condition', False): return 0
    if not (row.get('jc_condition', False) or row.get('macd_jc', False)): return 0
    if not row.get('trend_condition', False): return 0
    if not row.get('rsi_filter', False): return 0
    if not row.get('price_filter', False): return 0
    
    score += 0.1622
    g = row.get('growth', 0)
    if 0.5 <= g <= 6.0: score += 0.06
    elif g > 0: score += 0.03
    vr = min(row.get('vol_ratio', 0), 3)
    score += vr * 0.1704
    if row.get('macd_condition', False): score += 0.1366
    if row.get('macd_jc', False): score += 0.0873
    if row.get('rsi_14', 100) < 70: score += 0.0597
    if row.get('kdj_k', 100) < 80: score += 0.0597
    if row.get('cci_20', 100) < 100: score += 0.0597
    if row.get('close', 0) < row.get('bb_upper', float('inf')): score += 0.1191
    return score


def filter_v8(daily):
    """v8 IC增强过滤"""
    df = daily.copy()
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    mask = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 25) |
        (df['turnover_rate'] > 2.79) |
        (df['vol_ratio'] > 0.8) |
        (df['williams_r'] < -95) |
        (df['cci_20'] < -200)
    )
    df = df[~mask].copy()
    
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


def run_year_backtest(year, portfolios, verbose=True):
    """回测单一年份"""
    df = load_year(year)
    
    # 计算当年每日广度
    breadth = compute_breadth(df)
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📅 {year} 年回测")
        print(f"  数据: {len(df):,}条, {len(dates)}个交易日, 广度区间{breadth[dates[0]]:.0f}%~{breadth[dates[-1]]:.0f}%")
    
    for i, date in enumerate(dates):
        daily = df[df['date'] == date].copy()
        next_day = dates[i+1] if i+1 < n_dates else None
        day_breadth = breadth.get(date, 50)
        
        for strat_name, p in portfolios.items():
            is_v8 = 'v8' in strat_name
            uses_breadth = '+breadth' in strat_name
            # 广度调节器: v4/v8固定5只, v4+breadth/v8+breadth随广度变化
            top_n = get_top_n(day_breadth) if uses_breadth else TOP_N_FULL
            
            # ===== 平仓 =====
            for sym, pos in list(p['positions'].items()):
                if not next_day:
                    continue
                today_data = daily[daily['symbol'] == sym]
                if today_data.empty:
                    continue
                row = today_data.iloc[0]
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
                        'strategy': strat_name
                    })
                    del p['positions'][sym]
            
            # ===== 选股买入 =====
            if next_day and len(p['positions']) < top_n and top_n > 0:
                if is_v8:
                    candidates = filter_v8(daily)
                    if candidates.empty:
                        continue
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
                else:
                    # v4或v4+调节器
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
                    if candidates.empty:
                        continue
                    candidates['score'] = candidates.apply(score_v4, axis=1)
                
                candidates = candidates.sort_values('score', ascending=False)
                slots = top_n - len(p['positions'])
                
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
                        'strategy': strat_name
                    })
            
            # ===== 记录当日价值 =====
            pv = 0
            for sym, pos in p['positions'].items():
                today_data = daily[daily['symbol'] == sym]
                if not today_data.empty:
                    pv += today_data.iloc[0]['close'] * pos['qty']
                else:
                    pv += pos['qty'] * pos['avg_cost']
            
            total = p['cash'] + pv
            p['history'].append({
                'date': date,
                'cash': round(p['cash'], 2),
                'positions_value': round(pv, 2),
                'total_value': round(total, 2),
                'breadth': round(day_breadth, 1),
                'top_n': top_n,
                'n_positions': len(p['positions'])
            })
    
    if verbose:
        # 年末持仓
        for strat_name, p in portfolios.items():
            pos_list = list(p['positions'].keys())
            last = p['history'][-1] if p['history'] else {}
            buys = sum(1 for t in p['trades'] if t['action'] == 'buy')
            sells = sum(1 for t in p['trades'] if t['action'] == 'sell')
            pnl_list = [t['pnl'] for t in p['trades'] if t['action'] == 'sell']
            win_rate = (sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100) if pnl_list else 0
            print(f"  {strat_name}: 持仓{len(pos_list)}只 交易{buys}买/{sells}卖 胜率{win_rate:.0f}%")
    
    return df


def summarize_portfolio(p, name):
    """汇总账户表现"""
    if not p['history']:
        return
    initial = INITIAL_CASH
    final = p['history'][-1]['total_value']
    total_return = (final - initial) / initial * 100
    
    buys = [t for t in p['trades'] if t['action'] == 'buy']
    sells = [t for t in p['trades'] if t['action'] == 'sell']
    pnl_list = [t['pnl'] for t in sells]
    win_rate = (sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100) if pnl_list else 0
    
    print(f"\n{'='*60}")
    print(f"🏆 {name}")
    print(f"{'='*60}")
    print(f"  初始: {initial/1e4:.1f}万 → 最终: {final/1e4:.1f}万")
    print(f"  总收益: {total_return:+.1f}%")
    print(f"  交易次数: {len(buys)}买/{len(sells)}卖")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  盈利交易: {sum(1 for p in pnl_list if p > 0)}笔, 亏损: {sum(1 for p in pnl_list if p < 0)}笔")
    if pnl_list:
        avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
        avg_loss = np.mean([p for p in pnl_list if p < 0]) if any(p < 0 for p in pnl_list) else 0
        print(f"  平均盈: {avg_win/1e4:.1f}万, 平均亏: {avg_loss/1e4:.1f}万")
    
    return {
        'name': name,
        'initial': initial,
        'final': final,
        'total_return': total_return,
        'n_buys': len(buys),
        'n_sells': len(sells),
        'win_rate': win_rate,
        'trades': p['trades'],
        'history': p['history']
    }


def main():
    print(f"{'='*70}")
    print(f"📈 18年模拟盘回测 — v4原始 vs v8(IC增强) vs v4+广度调节器")
    print(f"  期间: 2008-01-01 ~ 2026-03-27 ({len(YEARS)}年)")
    print(f"  仓位规则: 广度>{BREADTH_FULL}%→5只, >{BREADTH_HALF}%→3只, >{BREADTH_LIGHT}%→2只, <{BREADTH_LIGHT}%→空仓")
    print(f"{'='*70}")
    
    # 四个策略
    portfolios = {
        'v4': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []},
        'v8': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []},
        'v4+breadth': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []},
        'v8+breadth': {'cash': INITIAL_CASH, 'positions': {}, 'trades': [], 'history': []},
    }
    
    for year in YEARS:
        try:
            run_year_backtest(year, portfolios, verbose=True)
        except FileNotFoundError as e:
            print(f"\n⚠️ {year}年数据不存在，跳过: {e}")
            continue
    
    # 年化收益率计算
    results = {}
    for name, p in portfolios.items():
        r = summarize_portfolio(p, name)
        if r:
            results[name] = r
    
    # 打印对比表
    print(f"\n{'='*70}")
    print(f"📊 18年策略对比 (2008-2026)")
    print(f"{'='*70}")
    print(f"{'策略':<15} {'总收益':>10} {'年化':>8} {'交易次数':>10} {'胜率':>8}")
    print(f"{'-'*60}")
    
    all_data = []
    for name, r in results.items():
        n_years = 18
        cagr = ((r['final'] / r['initial']) ** (1/n_years) - 1) * 100
        print(f"{name:<15} {r['total_return']:>+9.1f}% {cagr:>+7.1f}% {r['n_buys']:>8}笔 {r['win_rate']:>7.1f}%")
        all_data.append({'name': name, 'return': r['total_return'], 'cagr': cagr,
                        'trades': r['n_buys'], 'win_rate': r['win_rate']})
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f'{OUT_DIR}/backtest_18years_{ts}.json'
    
    save_data = {
        'metadata': {
            'start': '2008-01-01',
            'end': '2026-03-27',
            'years': len(YEARS),
            'initial_cash': INITIAL_CASH,
            'breadth_rules': {
                'full': f'>{BREADTH_FULL}% → {TOP_N_FULL}只',
                'half': f'>{BREADTH_HALF}% → {TOP_N_HALF}只',
                'light': f'>{BREADTH_LIGHT}% → {TOP_N_LIGHT}只',
                'empty': f'<{BREADTH_LIGHT}% → 0只'
            }
        },
        'results': all_data
    }
    
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 结果已保存: {out_file}")
    
    return results


if __name__ == '__main__':
    main()
