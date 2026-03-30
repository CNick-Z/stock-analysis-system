#!/usr/bin/env python3
"""
诊断 Wavechan 回测异常 - 逐笔分析
"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

# 只跑2020年前20天，逐笔打印
db = ParquetDatabaseIntegrator()
df = db.fetch_daily_data(
    '2019-09-01', '2020-01-31',
    columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
)

config = {
    'wave_threshold_pct': 0.025,
    'decline_threshold': -0.15,
    'consolidation_threshold': 0.05,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
}

print("计算特征中...")
features = compute_wavechan_features_fast(df, config)

# 过滤2020年1月
year_df = features[features['date'].str.startswith('2020-01')].copy()
dates = sorted(year_df['date'].unique())
print(f"\n2020年1月，共 {len(dates)} 个交易日")

# 只关注最早几天
INITIAL_CASH = 1_000_000
CASH = INITIAL_CASH
POSITIONS = {}
TRADES = []
date_to_idx = {d: i for i, d in enumerate(dates)}
price_lu = year_df.pivot_table(index='date', columns='symbol', values='close')
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

# 只跑前10天
for date in dates[:10]:
    day_df = year_df[year_df['date'] == date]
    day_buys = day_df[day_df['daily_signal'] == '买入']
    day_sells = day_df[day_df['daily_signal'] == '卖出']

    print(f"\n{'='*60}")
    print(f"📅 {date} | 持仓: {len(POSITIONS)} | 现金: {CASH:,.0f}")
    if day_buys.empty:
        print(f"  买入信号: 0")
    else:
        print(f"  买入信号: {len(day_buys)} 个")
        for _, r in day_buys.head(3).iterrows():
            print(f"    {r['symbol']} 收盘:{r['close']} 信号:{r['daily_signal']} 浪:{r['wave_trend']} 阶段:{r['wave_stage']}")

    # 打印当日持仓
    for sym, pos in POSITIONS.items():
        price = get_close(date, sym)
        if price:
            ret = (price - pos['buy_price']) / pos['buy_price']
            print(f"  持仓 {sym}: 买价{pos['buy_price']} 今收{price} 盈亏{ret:+.2%} 天数{pos['days']}")

    # ---- 卖出 ----
    for sym in list(POSITIONS.keys()):
        pos = POSITIONS[sym]
        pos['days'] += 1
        price = get_close(date, sym)
        if price is None:
            continue
        ret = (price - pos['buy_price']) / pos['buy_price']
        reason = ''

        day_data = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
        if not day_data.empty:
            wave_trend = day_data.iloc[0].get('wave_trend', 'neutral')
            sell_sig = day_data.iloc[0].get('chan_first_sell', False) or day_data.iloc[0].get('chan_second_sell', False)

            if ret <= -0.03:
                reason = f'止损{ret:.2%}'
            elif ret >= 0.25:
                reason = f'止盈{ret:.2%}'
            elif pos['days'] >= 10:
                reason = f'超时{pos["days"]}天'
            elif sell_sig:
                reason = '缠论卖出'

        if reason:
            pnl = (price - pos['buy_price']) * pos['qty']
            CASH += price * pos['qty']
            TRADES.append({'date': date, 'symbol': sym, 'type': 'sell',
                           'price': price, 'buy_price': pos['buy_price'],
                           'ret': ret, 'reason': reason, 'days': pos['days'], 'pnl': pnl})
            print(f"  🚫 卖出 {sym} 原因:{reason} 收益:{ret:+.2%} 天数:{pos['days']}")
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
            POSITIONS[sym] = {'qty': qty, 'buy_price': buy_price, 'days': 0}
            TRADES.append({'date': date, 'symbol': sym, 'type': 'buy',
                          'price': buy_price, 'reason': row['daily_signal']})
            print(f"  ✅ 买入 {sym} 价格:{buy_price} 数量:{qty}")

    # 当日总权益
    total = CASH
    for sym, pos in POSITIONS.items():
        p = get_close(date, sym)
        if p: total += p * pos['qty']
    print(f"  💰 当日总权益: {total:,.0f} ({total/INITIAL_CASH-1:+.2%})")

print(f"\n{'='*60}")
print(f"前10天交易汇总:")
buys = [t for t in TRADES if t['type']=='buy']
sells = [t for t in TRADES if t['type']=='sell']
print(f"买入: {len(buys)} 笔")
print(f"卖出: {len(sells)} 笔")
if sells:
    rets = [t['ret'] for t in sells]
    pnls = [t['pnl'] for t in sells]
    print(f"平均收益: {np.mean(rets):+.2%}")
    print(f"平均单笔PnL: {np.mean(pnls):+,.0f}")
    print(f"胜率: {sum(1 for r in rets if r>0)/len(rets):.0%}")
    print(f"\n卖出明细:")
    for t in sells[:10]:
        print(f"  {t['date']} {t['symbol']} {t['ret']:+.2%} 原因:{t['reason']} 天:{t['days']}")

# 重点：检查同一天同一只股票买卖
print(f"\n⚠️ 同日对冲检查（同一只股票先买后卖）:")
for i, t1 in enumerate(TRADES):
    if t1['type'] != 'buy': continue
    for t2 in TRADES[i+1:]:
        if t2['symbol'] == t1['symbol'] and t2['type'] == 'sell':
            d1 = dates.index(t1['date']) if t1['date'] in dates else -1
            d2 = dates.index(t2['date']) if t2['date'] in dates else -1
            print(f"  {t1['date']} 买 → {t2['date']} 卖 间隔:{d2-d1}天 收益:{t2['ret']:+.2%}")
            break
