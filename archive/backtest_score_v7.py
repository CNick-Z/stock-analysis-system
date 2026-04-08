#!/usr/bin/env python3
"""
Score 策略 v7 回测 — 基于 IC/IR 分析优化权重

【IC/IR 优化结论】
✅ 保留强化:
  - CCI超卖(CCI < -100): ICIR=2.52, 100%正IC → 最大权重
  - WR超卖(WR < -80): ICIR=0.84, 75%正IC → 第二权重
  - KDJ金叉: ICIR=0.50, 75%正IC → 保留降权

❌ 移除/反转:
  - RSI 50-60过滤: 撤掉（RSI超卖实际最差，IC=-0.057）
  - 均线多头加分: 移除（IC=-0.046，多头排列反而跌）
  - 放量买入: 反转为缩量优先（量比>1.25 IC=-0.048）
  - 高换手: 改为低换手加分（>2.79% IC=-0.100）
  - BOLL贴近下轨: 移除（中轨0.23~0.34最好）
  - MACD金叉: 降权（ICIR=0.04，几乎无效）

【新增正向逻辑】
  - 缩量信号: vol_ratio < 0.71 加分
  - 低换手: turnover < 0.42% 加分
  - BOLL中轨: boll_pos 0.23~0.34 加分
  - 低价股: 3-15元保留
"""

import os, sys, gc, tracemalloc, psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== 配置 =====
DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/backtest_results'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 100_0000  # 100万
COMMISSION = 0.0003       # 万三佣金
STAMP_TAX = 0.001         # 千一印花税（卖出时）
SLIPPAGE = 0.001          # 千分之一滑点

YEARS = range(2018, 2026)

# ===== 性能监控 =====
def check_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] MEM={mem_gb:.2f}GB | {label}")
    return mem_gb


def load_year(year):
    """加载单年技术指标+日线数据"""
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    
    # 计算辅助字段
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)   # 量比
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    
    # 未来收益率（label）
    for n in [5, 10, 20]:
        df[f'return_{n}d'] = df.groupby('symbol')['close'].pct_change(n).shift(-n)
    
    # 明日开盘价（用于模拟真实成交）
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    
    return df


def score_v7(row):
    """
    v7 评分函数 — 基于 IC/IR 优化后的权重
    返回: (score, signal_dict)
    """
    score = 0.0
    signals = {}
    
    # ===== CCI 超卖信号 — 最重要因子（权重最大）====
    if row['cci_20'] < -100:
        score += 0.35
        signals['cci_oversold'] = True
    elif row['cci_20'] < -50:
        score += 0.10
        signals['cci_weak_oversold'] = True
    else:
        signals['cci_oversold'] = False
    
    # ===== Williams%R 超卖信号 — 第二重要 =====
    if row['williams_r'] < -80:
        score += 0.20
        signals['wr_oversold'] = True
    elif row['williams_r'] < -60:
        score += 0.05
        signals['wr_weak_oversold'] = True
    else:
        signals['wr_oversold'] = False
    
    # ===== KDJ 金叉 — 保留但降权 =====
    kdj_cross = (
        (row['kdj_k'] > row['kdj_d']) and
        (row['kdj_k'] < 80)  # 避免在超买区金叉
    )
    if kdj_cross:
        score += 0.08
        signals['kdj_cross'] = True
    else:
        signals['kdj_cross'] = False
    
    # ===== MACD 金叉 — 降权（原权重很高但IC几乎为0）====
    macd_cross = (
        (row['macd'] > row['macd_signal']) and
        (row['macd'] < 0)  # 零轴下方金叉更有意义
    )
    if macd_cross:
        score += 0.05  # 降权处理
        signals['macd_cross'] = True
    else:
        signals['macd_cross'] = False
    
    # ===== 缩量信号（新增 — 反转原放量逻辑）====
    if row['vol_ratio'] < 0.71:
        score += 0.10
        signals['low_vol'] = True
    elif row['vol_ratio'] < 1.0:
        score += 0.03
        signals['slight_low_vol'] = True
    elif row['vol_ratio'] > 1.25:
        signals['high_vol_danger'] = True  # 警示
    else:
        signals['low_vol'] = False
    
    # ===== 低换手加分（新增 — 原高换手反而减分）====
    if row['turnover_rate'] < 0.42:
        score += 0.08
        signals['low_turnover'] = True
    elif row['turnover_rate'] < 1.0:
        score += 0.02
        signals['medium_turnover'] = True
    elif row['turnover_rate'] > 2.79:
        signals['high_turnover_danger'] = True  # 警示
    else:
        signals['low_turnover'] = False
    
    # ===== BOLL 中轨加分（新增 — 原逻辑是贴近下轨买）====
    if 0.23 <= row['boll_pos'] <= 0.34:
        score += 0.07
        signals['boll_mid'] = True
    elif row['boll_pos'] < 0.15:
        signals['boll_lower_danger'] = True  # 贴近下轨=陷阱
    else:
        signals['boll_mid'] = False
    
    # ===== 涨幅过滤（中性，±3%区间最好）====
    change = abs(row['change_pct']) if not pd.isna(row['change_pct']) else 0
    if 0.5 <= change <= 3.0:
        score += 0.04
        signals['healthy_change'] = True
    elif change > 6:
        signals['too_volatile'] = True  # 波动过大
    else:
        signals['healthy_change'] = False
    
    # ===== RSI 中性区加分（移除超卖逻辑，改为中性偏好）====
    rsi = row['rsi_14']
    if 42 <= rsi <= 48:
        score += 0.03  # IC显示这个区间最好，降权处理
        signals['rsi_neutral_good'] = True
    elif rsi > 70 or rsi < 30:
        signals['rsi_extreme'] = True  # 极值警示
    
    return score, signals


def filter_buy(signals_df):
    """v7 买入条件过滤"""
    buy_conditions = (
        # 最少要有CCI超卖信号（最重要因子）
        (signals_df['cci_20'] < -50) &
        # Williams%R 超卖
        (signals_df['williams_r'] < -60) &
        # 价格中等（3-15元保留）
        (signals_df['close'] >= 3) &
        (signals_df['close'] <= 15) &
        # KDJ 不要在超买区金叉
        (signals_df['kdj_k'] < 80) &
        # 避开高换手陷阱
        (signals_df['turnover_rate'] < 2.79) &
        # 避开极度超卖（IC显示极度超卖反转差）
        (signals_df['cci_20'] > -200)
    )
    
    # 按 score 排序取 top
    buy_df = signals_df[buy_conditions].copy()
    buy_df['v7_score'] = buy_df.apply(lambda r: score_v7(r)[0], axis=1)
    buy_df = buy_df.sort_values('v7_score', ascending=False)
    
    return buy_df


def run_backtest_year(year, top_n=5):
    """单年回测"""
    print(f"\n{'='*60}")
    print(f"📅 {year} 年回测")
    print(f"{'='*60}")
    
    df = load_year(year)
    check_memory(f"{year}加载")
    print(f"  总记录数: {len(df):,}")
    
    portfolio = {
        'cash': INITIAL_CASH,
        'positions': {},      # {symbol: {'qty': x, 'avg_cost': y}}
        'trades': [],
        'history': []
    }
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    annual_return = 0.0
    max_value = INITIAL_CASH
    max_drawdown = 0.0
    
    for i, date in enumerate(dates):
        daily = df[df['date'] == date].copy()
        next_day = dates[i+1] if i+1 < n_dates else None
        
        # ===== 持仓管理 =====
        for sym, pos in list(portfolio['positions'].items()):
            if next_day:
                today_data = daily[daily['symbol'] == sym]
                if len(today_data) == 0:
                    continue
                row = today_data.iloc[0]
                next_open = row['next_open']
                
                if pd.isna(next_open):
                    continue
                
                # 卖出条件（简化版）
                should_sell = False
                sell_reason = ""
                
                # 止损: 跌超5%止损
                if next_open < pos['avg_cost'] * 0.95:
                    should_sell = True
                    sell_reason = "stop_loss_5%"
                # 止盈: 涨超15%
                elif next_open > pos['avg_cost'] * 1.15:
                    should_sell = True
                    sell_reason = "take_profit_15%"
                # CCI死叉
                elif row['cci_20'] > 50 and row['cci_20'] > row.get('cci_prev', 0):
                    should_sell = True
                    sell_reason = "cci_death_cross"
                
                if should_sell:
                    proceeds = pos['qty'] * next_open * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                    pnl = proceeds - pos['qty'] * pos['avg_cost']
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'date': date,
                        'symbol': sym,
                        'action': 'sell',
                        'price': next_open,
                        'qty': pos['qty'],
                        'pnl': pnl,
                        'reason': sell_reason
                    })
                    del portfolio['positions'][sym]
        
        # ===== 选股买入 =====
        if next_day and len(portfolio['positions']) < top_n:
            buy_df = filter_buy(daily)
            
            available_slots = top_n - len(portfolio['positions'])
            candidates = buy_df.head(available_slots)
            
            for _, row in candidates.iterrows():
                sym = row['symbol']
                if sym in portfolio['positions']:
                    continue
                
                next_open = row['next_open']
                if pd.isna(next_open) or next_open <= 0:
                    continue
                
                # 每只股票最多投入20%仓位
                max_per_stock = portfolio['cash'] * 0.20
                buy_qty = int(max_per_stock / next_open / 100) * 100  # 按手买
                
                if buy_qty < 100:
                    continue
                
                cost = buy_qty * next_open * (1 + COMMISSION + SLIPPAGE)
                if cost > portfolio['cash']:
                    continue
                
                portfolio['cash'] -= cost
                portfolio['positions'][sym] = {
                    'qty': buy_qty,
                    'avg_cost': next_open,
                    'buy_date': date,
                    'buy_price': next_open
                }
                portfolio['trades'].append({
                    'date': date,
                    'symbol': sym,
                    'action': 'buy',
                    'price': next_open,
                    'qty': buy_qty,
                    'v7_score': row['v7_score']
                })
        
        # ===== 计算当日组合价值 =====
        positions_value = 0
        for sym, pos in portfolio['positions'].items():
            today_data = daily[daily['symbol'] == sym]
            if len(today_data) > 0:
                positions_value += today_data.iloc[0]['close'] * pos['qty']
            else:
                positions_value += pos['qty'] * pos['avg_cost']
        
        total_value = portfolio['cash'] + positions_value
        max_value = max(max_value, total_value)
        drawdown = (max_value - total_value) / max_value
        max_drawdown = max(max_drawdown, drawdown)
        
        if i % 20 == 0 or i == n_dates - 1:
            print(f"  {date}: 持仓{len(portfolio['positions'])}只, 现金{portfolio['cash']:.0f}, "
                  f"总价值{total_value:.0f}, 回撤{drawdown:.1%}")
    
    # 年终持仓清仓
    last_date = dates[-1]
    last_data = df[df['date'] == last_date]
    for sym, pos in list(portfolio['positions'].items()):
        today_data = last_data[last_data['symbol'] == sym]
        if len(today_data) > 0:
            sell_price = today_data.iloc[0]['close']
            proceeds = pos['qty'] * sell_price * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
            pnl = proceeds - pos['qty'] * pos['avg_cost']
            portfolio['trades'].append({
                'date': last_date,
                'symbol': sym,
                'action': 'year_end_sell',
                'price': sell_price,
                'qty': pos['qty'],
                'pnl': pnl
            })
            portfolio['cash'] += proceeds
        del portfolio['positions'][sym]
    
    total_value = portfolio['cash']
    annual_return = (total_value - INITIAL_CASH) / INITIAL_CASH
    
    # 统计
    buys = [t for t in portfolio['trades'] if t['action'] == 'buy']
    sells = [t for t in portfolio['trades'] if t['action'] in ('sell', 'year_end_sell')]
    winners = [t for t in sells if t['pnl'] > 0]
    
    print(f"\n  📊 {year} 年结果:")
    print(f"  年收益率: {annual_return:.2%}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  买入次数: {len(buys)}, 卖出次数: {len(sells)}")
    print(f"  盈利交易: {len(winners)}/{len(sells)} ({len(winners)/max(1,len(sells)):.0%})")
    
    del df; gc.collect()
    
    return {
        'year': year,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'total_value': total_value,
        'n_buys': len(buys),
        'n_sells': len(sells),
        'n_winners': len(winners),
        'win_rate': len(winners) / max(1, len(sells)),
        'trades': portfolio['trades']
    }


def run_full_backtest():
    print(f"{'='*60}")
    print(f"🏃 Score 策略 v7 回测 — IC/IR 优化版")
    print(f"{'='*60}")
    print(f"回测区间: {min(YEARS)}-{max(YEARS)}")
    print(f"初始资金: {INITIAL_CASH/10000:.0f}万")
    print(f"佣金: {COMMISSION:.2%}, 印花税: {STAMP_TAX:.1%}, 滑点: {SLIPPAGE:.1%}")
    print(f"每只股票仓位上限: 20%")
    print()
    
    check_memory("启动")
    
    results = []
    for year in YEARS:
        result = run_backtest_year(year)
        results.append(result)
        check_memory(f"{year}完成")
    
    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print(f"🏆 Score v7 完整回测汇总")
    print(f"{'='*60}")
    
    total_return = 1.0
    for r in results:
        total_return *= (1 + r['annual_return'])
    
    cumulative_max_dd = 0.0
    total_buys = sum(r['n_buys'] for r in results)
    total_winners = sum(r['n_winners'] for r in results)
    total_sells = sum(r['n_sells'] for r in results)
    
    print(f"\n{'年份':<8} {'收益率':>10} {'最大回撤':>10} {'买入':>6} {'卖出':>6} {'胜率':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['year']:<8} {r['annual_return']:>10.2%} {r['max_drawdown']:>10.2%} "
              f"{r['n_buys']:>6} {r['n_sells']:>6} {r['win_rate']:>8.0%}")
    
    print("-" * 55)
    print(f"{'合计':<8} {(total_return-1):>10.2%}")
    print(f"\n总买入次数: {total_buys}, 总卖出: {total_sells}")
    print(f"总胜率: {total_winners/max(1,total_sells):.2%}")
    print(f"最终资金: {results[-1]['total_value']/10000:.2f}万")
    print(f"vs 初始 {INITIAL_CASH/10000:.2f}万")
    
    return results


if __name__ == '__main__':
    results = run_full_backtest()
