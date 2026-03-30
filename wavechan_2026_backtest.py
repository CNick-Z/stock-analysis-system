#!/usr/bin/env python3
"""
Wavechan 策略 2026年至今 回测
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

CONFIG = {
    'initial_cash': 1_000_000,
    'position_size': 0.30,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
    'max_hold_days': 10,
    'max_positions': 3,
    'buy_slots_per_day': 3,
}


def run_backtest(year_df, dates, config):
    cash = config['initial_cash']
    positions = {}
    trades = []
    equity = []

    date_set = set(dates)
    date_idx = {d: i for i, d in enumerate(dates)}
    price_lu = year_df.pivot_table(index='date', columns='symbol', values='close')
    open_lu = year_df.pivot_table(index='date', columns='symbol', values='open')

    def get_close(date, sym):
        try:
            if date in date_set and sym in price_lu.columns:
                p = price_lu.loc[date, sym]
                return p if not pd.isna(p) else None
        except: pass
        return None

    def get_next_open(date, sym):
        """用真实的次日开盘价（非收盘价）"""
        i = date_idx.get(date, -1)
        if i >= 0 and i + 1 < len(dates):
            next_date = dates[i + 1]
            try:
                if next_date in date_set and sym in open_lu.columns:
                    p = open_lu.loc[next_date, sym]
                    return p if not pd.isna(p) else get_close(next_date, sym)
            except: pass
            return get_close(next_date, sym)
        return None

    for date in dates:
        day_df = year_df[year_df['date'] == date]

        # 卖出检查（用当日数据判断，次日开盘执行）
        for sym in list(positions.keys()):
            pos = positions[sym]
            pos['days'] += 1

            # 用当日收盘价判断是否触发卖出条件
            cur_price = get_close(date, sym)
            if cur_price is None:
                continue
            ret = (cur_price - pos['buy_price']) / pos['buy_price']
            reason = ''

            day_data = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
            if not day_data.empty:
                sell_sig = day_data.iloc[0].get('chan_first_sell', False) or day_data.iloc[0].get('chan_second_sell', False)
                if ret <= -config['stop_loss_pct']:
                    reason = f'止损{ret:.2%}'
                elif ret >= config['profit_target_pct']:
                    reason = f'止盈{ret:.2%}'
                elif pos['days'] >= config['max_hold_days']:
                    reason = f'超时{pos["days"]}天'
                elif sell_sig:
                    reason = '缠论卖出'

            # 如果触发卖出条件，用次日开盘价执行
            if reason:
                sell_price = get_next_open(date, sym)  # 改为次日开盘价
                if sell_price is None:
                    sell_price = cur_price  # fallback
                cash += sell_price * pos['qty']
                exec_ret = (sell_price - pos['buy_price']) / pos['buy_price']
                trades.append({
                    'date': date, 'symbol': sym, 'type': 'sell',
                    'buy_price': pos['buy_price'], 'sell_price': sell_price,
                    'ret': exec_ret, 'reason': reason, 'days': pos['days']
                })
                del positions[sym]

        # 买入
        if len(positions) < config['max_positions']:
            cands = day_df[
                (day_df['daily_signal'] == '买入') &
                (~day_df['symbol'].isin(positions.keys()))
            ].nlargest(config['buy_slots_per_day'], 'daily_confidence')

            for _, row in cands.iterrows():
                if len(positions) >= config['max_positions']:
                    break
                sym = row['symbol']
                buy_price = get_next_open(date, sym)
                if buy_price is None or buy_price <= 0:
                    continue
                qty = int(cash * config['position_size'] / (buy_price * 1.0003) / 100) * 100
                if qty < 100:
                    continue
                cost = buy_price * qty * 1.0003
                if cost > cash:
                    continue
                cash -= cost
                positions[sym] = {'qty': qty, 'buy_price': buy_price, 'days': 0}
                trades.append({
                    'date': date, 'symbol': sym, 'type': 'buy',
                    'buy_price': buy_price, 'price': buy_price,
                    'close_price': row['close'],
                    'wave_stage': row.get('wave_stage', ''),
                    'signal': row.get('daily_signal', ''),
                })

        # 权益
        total = cash
        for sym, pos in positions.items():
            p = get_close(date, sym)
            if p:
                total += p * pos['qty']
        equity.append({'date': date, 'total': total, 'cash': cash})

    # 清算
    last = dates[-1]
    for sym, pos in list(positions.items()):
        p = get_close(last, sym)
        if p:
            cash += p * pos['qty']

    total_value = cash
    peak = config['initial_cash']
    max_dd = 0.0
    for e in equity:
        if e['total'] > peak:
            peak = e['total']
        dd = (peak - e['total']) / peak
        if dd > max_dd:
            max_dd = dd

    sells = [t for t in trades if t['type'] == 'sell']
    wins = [t for t in sells if t.get('ret', 0) > 0]

    return {
        'initial': config['initial_cash'],
        'final': total_value,
        'ret': (total_value - config['initial_cash']) / config['initial_cash'],
        'max_drawdown': max_dd,
        'trades': trades,
        'buys': len([t for t in trades if t['type'] == 'buy']),
        'sells': len(sells),
        'win_rate': len(wins) / len(sells) if sells else 0,
        'equity': equity,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Wavechan 2026年至今 回测")
    print("=" * 60)

    db = ParquetDatabaseIntegrator()
    config = {
        'wave_threshold_pct': 0.025,
        'decline_threshold': -0.15,
        'consolidation_threshold': 0.05,
        'stop_loss_pct': 0.03,
        'profit_target_pct': 0.25,
    }

    # 加载数据
    t0 = time.time()
    df = db.fetch_daily_data(
        '2025-10-01', '2026-03-31',
        columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    )
    print(f"数据加载: {len(df)} 条, {df['symbol'].nunique()} 只, {time.time()-t0:.1f}s")

    # 特征计算
    t1 = time.time()
    features = compute_wavechan_features_fast(df, config)
    print(f"特征计算: {time.time()-t1:.1f}s")

    # 过滤2026年
    year_df = features[features['date'].str.startswith('2026-')].copy()
    dates = sorted(year_df['date'].unique())
    print(f"2026年: {dates[0]} ~ {dates[-1]}, {len(dates)} 个交易日\n")

    # 回测
    t2 = time.time()
    result = run_backtest(year_df, dates, CONFIG)
    print(f"回测: {time.time()-t2:.1f}s\n")

    # 输出
    print(f"=" * 60)
    print(f"📊 2026年至今 ({dates[0]} ~ {dates[-1]})")
    print(f"=" * 60)
    print(f"  初始资金: {result['initial']:,.0f}")
    print(f"  最终净值: {result['final']:,.0f}")
    print(f"  收益率:   {result['ret']:+.2%}")
    print(f"  最大回撤: {result['max_drawdown']:+.2%}")
    print(f"  买入次数: {result['buys']}")
    print(f"  卖出次数: {result['sells']}")
    print(f"  胜率:     {result['win_rate']:.0%}")

    sells = [t for t in result['trades'] if t['type'] == 'sell']
    if sells:
        rets = [t['ret'] for t in sells]
        print(f"  平均收益: {np.mean(rets):+.2%}")
        print(f"  平均盈利: {np.mean([r for r in rets if r>0]):+.2%}")
        print(f"  平均亏损: {np.mean([r for r in rets if r<=0]):+.2%}")

    print(f"\n--- 每日净值 ---")
    for e in result['equity']:
        cumret = (e['total'] - result['initial']) / result['initial']
        print(f"  {e['date']}: {e['total']:>12,.0f} ({cumret:+.2%})")

    print(f"\n--- 所有交易 ---")
    for t in result['trades']:
        if t['type'] == 'buy':
            print(f"  🟢 买入 {t['date']} {t['symbol']} @ {t['buy_price']} ({t.get('wave_stage','')})")
        else:
            emoji = "✅" if t['ret'] > 0 else "❌"
            print(f"  {emoji} 卖出 {t['date']} {t['symbol']} {t['buy_price']}→{t['sell_price']} {t['ret']:+.2%} 原因:{t['reason']} 持:{t['days']}天")
