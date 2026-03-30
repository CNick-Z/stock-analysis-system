#!/usr/bin/env python3
"""
详细检查 Wavechan 回测的交易记录 - 找异常
"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

db = ParquetDatabaseIntegrator()
df = db.fetch_daily_data(
    '2019-09-01', '2020-03-31',
    columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
)

config = {
    'wave_threshold_pct': 0.025,
    'decline_threshold': -0.15,
    'consolidation_threshold': 0.05,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
}

print("计算特征...")
features = compute_wavechan_features_fast(df, config)

# 过滤2020全年
year_df = features[features['date'].str.startswith('2020-')].copy()
dates = sorted(year_df['date'].unique())
print(f"2020年共 {len(dates)} 个交易日")

price_lu = year_df.pivot_table(index='date', columns='symbol', values='close')
date_to_idx = {d: i for i, d in enumerate(dates)}
date_set = set(dates)

def get_close(date, sym):
    try:
        if date in date_set and sym in price_lu.columns:
            p = price_lu.loc[date, sym]
            return p if not pd.isna(p) else None
    except: pass
    return None

def get_next_open(date, sym):
    i = date_to_idx.get(date, -1)
    if i >= 0 and i + 1 < len(dates):
        return get_close(dates[i+1], sym)
    return None

INITIAL = 1_000_000
CASH = INITIAL
POSITIONS = {}
TRADES = []

for date in dates:
    day_df = year_df[year_df['date'] == date]

    # ---- 卖出 ----
    for sym in list(POSITIONS.keys()):
        pos = POSITIONS[sym]
        pos['days'] += 1
        price = get_close(date, sym)
        if price is None:
            continue
        ret = (price - pos['buy_price']) / pos['buy_price']

        day_data = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
        reason = ''
        if not day_data.empty:
            sell_sig = day_data.iloc[0].get('chan_first_sell', False) or day_data.iloc[0].get('chan_second_sell', False)
            wave_trend = day_data.iloc[0].get('wave_trend', 'neutral')

            if ret <= -0.03:
                reason = f'止损{ret:.2%}'
            elif ret >= 0.25:
                reason = f'止盈{ret:.2%}'
            elif pos['days'] >= 10:
                reason = f'超时{pos["days"]}天'
            elif sell_sig:
                reason = f'缠论卖出(wave={wave_trend})'

        if reason:
            pnl = (price - pos['buy_price']) * pos['qty']
            CASH += price * pos['qty']
            TRADES.append({
                'date': date, 'symbol': sym, 'type': 'sell',
                'buy_date': pos['buy_date'], 'buy_price': pos['buy_price'],
                'sell_price': price, 'qty': pos['qty'],
                'ret': ret, 'pnl': pnl, 'reason': reason,
                'days': pos['days']
            })
            del POSITIONS[sym]

    # ---- 买入 ----
    if len(POSITIONS) < 5:
        candidates = day_df[
            (day_df['daily_signal'] == '买入') &
            (~day_df['symbol'].isin(POSITIONS.keys()))
        ].nlargest(3, 'daily_confidence')

        for _, row in candidates.iterrows():
            if len(POSITIONS) >= 5:
                break
            sym = row['symbol']
            buy_price = get_next_open(date, sym)
            if buy_price is None or buy_price <= 0:
                continue

            qty = int(CASH * 0.3 / (buy_price * 1.0003) / 100) * 100
            if qty < 100:
                continue
            cost = buy_price * qty * 1.0003
            if cost > CASH:
                continue

            CASH -= cost
            POSITIONS[sym] = {'qty': qty, 'buy_price': buy_price, 'buy_date': date, 'days': 0}
            TRADES.append({
                'date': date, 'symbol': sym, 'type': 'buy',
                'buy_price': buy_price, 'qty': qty,
                'close_price': row['close'],  # 当日收盘
                'ret_estimate': (row['close'] - buy_price) / buy_price  # 用收盘价的预估收益
            })

# 期末清算
last_date = dates[-1]
for sym, pos in list(POSITIONS.items()):
    price = get_close(last_date, sym)
    if price:
        CASH += price * pos['qty']

# ============================================================
# 分析交易记录
# ============================================================
buys = [t for t in TRADES if t['type'] == 'buy']
sells = [t for t in TRADES if t['type'] == 'sell']

print(f"\n{'='*70}")
print(f"总买入: {len(buys)} 笔 | 总卖出: {len(sells)} 笔 | 最终现金: {CASH:,.0f}")
print(f"{'='*70}")

# 1. 检查同日对冲（同一天买又卖）
print(f"\n🔍 异常1: 同一天内买又卖（当日T+0）")
for t1 in buys:
    for t2 in sells:
        if t2['symbol'] == t1['symbol'] and t2['date'] == t1['date']:
            print(f"  🚨 {t1['date']} {t1['symbol']} 当日买{t1['buy_price']} 又卖{t2['sell_price']}")

# 2. 检查当天涨停股是否能买到
print(f"\n🔍 异常2: 买入价 vs 当日收盘（验证是否涨停敢买）")
for t in buys[:20]:
    if 'close_price' in t:
        diff = (t['close_price'] - t['buy_price']) / t['buy_price']
        if abs(diff) > 0.095:
            print(f"  🚨 {t['date']} {t['symbol']} 买价:{t['buy_price']} 收盘:{t['close_price']:.2f} 涨幅:{diff:+.2%}")

# 3. 找最赚钱和最亏钱的交易
print(f"\n🔍 异常3: 最赚钱 Top10 交易")
sells_sorted = sorted(sells, key=lambda x: x['ret'], reverse=True)
for t in sells_sorted[:10]:
    print(f"  ✅ {t['date']} {t['symbol']} 买{t['buy_price']}→卖{t['sell_price']} 收益:{t['ret']:+.2%} 原因:{t['reason']}")

print(f"\n🔍 异常4: 最亏钱 Top10 交易")
for t in sells_sorted[-10:]:
    print(f"  ❌ {t['date']} {t['symbol']} 买{t['buy_price']}→卖{t['sell_price']} 收益:{t['ret']:+.2%} 原因:{t['reason']}")

# 4. 检查同一只股票被买入次数
print(f"\n🔍 异常5: 同一只股票被多次买入")
from collections import Counter
buy_counts = Counter(t['symbol'] for t in buys)
multi_buy = {s: c for s, c in buy_counts.items() if c > 3}
if multi_buy:
    for sym, count in sorted(multi_buy.items(), key=lambda x: -x[1])[:10]:
        buys_of_sym = [t for t in buys if t['symbol'] == sym]
        rets = []
        for b in buys_of_sym:
            for s in sells:
                if s['symbol'] == sym and s['date'] > b['date']:
                    rets.append(s['ret'])
                    break
        print(f"  {sym} 被买 {count} 次")
else:
    print(f"  无异常（最多每个股票买入1次）")

# 5. 胜率分布
print(f"\n🔍 异常6: 卖出收益分布")
if sells:
    rets = [t['ret'] for t in sells]
    print(f"  总卖出: {len(rets)} 笔")
    print(f"  胜: {sum(1 for r in rets if r>0)} 笔 ({sum(1 for r in rets if r>0)/len(rets):.0%})")
    print(f"  亏: {sum(1 for r in rets if r<=0)} 笔")
    print(f"  平均收益: {np.mean(rets):+.2%}")
    print(f"  平均盈利: {np.mean([r for r in rets if r>0]):+.2%}")
    print(f"  平均亏损: {np.mean([r for r in rets if r<=0]):+.2%}")
    print(f"  盈亏比: {abs(np.mean([r for r in rets if r>0])/np.mean([r for r in rets if r<=0])):.2f}")

# 6. 计算真实夏普（简化）
print(f"\n🔍 异常7: 复利效应验证")
equity_daily = []
cash = INITIAL
positions_hist = {}
daily_values = []

for date in dates:
    # 先清算当日持仓
    for sym in list(positions_hist.keys()):
        pos = positions_hist[sym]
        pos['days'] += 1
        price = get_close(date, sym)
        if price:
            pos['cur_price'] = price

    # 当日买入
    day_df = year_df[year_df['date'] == date]
    candidates = day_df[
        (day_df['daily_signal'] == '买入') &
        (~day_df['symbol'].isin(positions_hist.keys()))
    ].nlargest(3, 'daily_confidence')

    for _, row in candidates.iterrows():
        if len(positions_hist) >= 5:
            break
        sym = row['symbol']
        buy_price = get_next_open(date, sym)
        if buy_price is None or buy_price <= 0:
            continue
        qty = int(cash * 0.3 / (buy_price * 1.0003) / 100) * 100
        if qty < 100:
            continue
        cost = buy_price * qty * 1.0003
        if cost > cash:
            continue
        cash -= cost
        positions_hist[sym] = {'qty': qty, 'buy_price': buy_price, 'days': 0, 'cur_price': buy_price}

    # 当日卖出
    for sym in list(positions_hist.keys()):
        pos = positions_hist[sym]
        price = pos.get('cur_price', get_close(date, sym))
        if price is None:
            continue
        ret = (price - pos['buy_price']) / pos['buy_price']
        reason = ''
        day_data = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
        if not day_data.empty:
            sell_sig = day_data.iloc[0].get('chan_first_sell', False) or day_data.iloc[0].get('chan_second_sell', False)
            if ret <= -0.03:
                reason = '止损'
            elif ret >= 0.25:
                reason = '止盈'
            elif pos['days'] >= 10:
                reason = '超时'
            elif sell_sig:
                reason = '缠论'

        if reason:
            cash += price * pos['qty']
            del positions_hist[sym]

    total = cash
    for pos in positions_hist.values():
        if pos.get('cur_price'):
            total += pos['cur_price'] * pos['qty']
    daily_values.append({'date': date, 'total': total})

eq_df = pd.DataFrame(daily_values)
eq_df['ret'] = eq_df['total'].pct_change()
eq_df['cumret'] = (1 + eq_df['ret']).cumprod() - 1

sharpe = eq_df['ret'].mean() / eq_df['ret'].std() * np.sqrt(252) if eq_df['ret'].std() > 0 else 0
total_ret = eq_df['cumret'].iloc[-1] if len(eq_df) > 0 else 0

print(f"  2020全年最终收益: {total_ret:+.2%}")
print(f"  夏普比率(简化): {sharpe:.2f}")
print(f"  最大连续亏损天数: {(eq_df['ret']<0).astype(int).groupby((eq_df['ret']>=0).astype(int).cumsum()).cumsum().max() if len(eq_df)>0 else 0}")

print(f"\n📊 2020年每日权益（前20天）:")
for _, row in eq_df.head(20).iterrows():
    print(f"  {row['date']}: {row['total']:>12,.0f} ({row['cumret']:+.2%})")
