#!/usr/bin/env python3
"""
WavechanFast 2020-2025 完整回测
使用极速版波浪缠论特征计算
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'initial_cash': 1_000_000,
    'position_size': 0.3,      # 30%仓位
    'stop_loss_pct': 0.03,    # 3%止损
    'profit_target_pct': 0.25, # 25%止盈
    'max_hold_days': 10,
    'max_positions': 5,
    'buy_slots_per_day': 3,
}

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# ============================================================
# 回测引擎
# ============================================================

def run_year_backtest(year: int, features_df: pd.DataFrame) -> dict:
    cfg = CONFIG
    cash = cfg['initial_cash']
    positions = {}
    trades = []
    equity = []

    # 过滤当年数据
    year_df = features_df[features_df['date'].str.startswith(str(year))].copy()
    if year_df.empty:
        return None

    dates = sorted(year_df['date'].unique())
    date_set = set(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    # 价格查询表
    priceLookup = year_df.pivot_table(index='date', columns='symbol', values='close')

    def get_close(date, sym):
        try:
            if date in date_set and sym in priceLookup.columns:
                p = priceLookup.loc[date, sym]
                return p if not pd.isna(p) else None
        except:
            pass
        return None

    def get_next_open(date, sym):
        i = date_idx.get(date, -1)
        if i >= 0 and i + 1 < len(dates):
            return get_close(dates[i + 1], sym)
        return None

    for date in dates:
        # ---- 卖出检查 ----
        for sym in list(positions.keys()):
            pos = positions[sym]
            pos['days'] += 1
            price = get_close(date, sym)
            if price is None:
                continue

            ret = (price - pos['buy_price']) / pos['buy_price']
            reason = ''

            # 当日分型/缠论信号作为辅助
            day_data = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
            if not day_data.empty:
                signal = day_data.iloc[0].get('daily_signal', 'hold')
                wave_trend = day_data.iloc[0].get('wave_trend', 'neutral')
                sell_signal = day_data.iloc[0].get('chan_first_sell', False) or day_data.iloc[0].get('chan_second_sell', False)

                if ret <= -cfg['stop_loss_pct']:
                    reason = f'止损{ret:.2%}'
                elif ret >= cfg['profit_target_pct']:
                    reason = f'止盈{ret:.2%}'
                elif pos['days'] >= cfg['max_hold_days']:
                    reason = f'超时{pos["days"]}天'
                elif signal == '卖出' and wave_trend == 'down':
                    reason = '波浪下跌+卖出信号'
                elif sell_signal:
                    reason = '缠论卖出信号'

            if reason:
                qty = pos['qty']
                pnl = (price - pos['buy_price']) * qty
                cash += price * qty
                trades.append({
                    'date': date, 'symbol': sym, 'type': 'sell',
                    'price': price, 'qty': qty, 'buy_price': pos['buy_price'],
                    'pnl': pnl, 'ret': ret, 'reason': reason, 'hold': pos['days']
                })
                del positions[sym]

        # ---- 买入信号 ----
        if len(positions) < cfg['max_positions']:
            day_df = year_df[year_df['date'] == date]
            buy_candidates = day_df[
                (day_df['daily_signal'] == '买入') &
                (~day_df['symbol'].isin(positions.keys()))
            ]

            if not buy_candidates.empty:
                buy_candidates = buy_candidates.nlargest(cfg['buy_slots_per_day'], 'daily_confidence')

                for _, row in buy_candidates.iterrows():
                    if len(positions) >= cfg['max_positions']:
                        break
                    sym = row['symbol']
                    if sym in positions:
                        continue

                    buy_price = get_next_open(date, sym)
                    if buy_price is None or buy_price <= 0:
                        continue

                    max_invest = cash * cfg['position_size']
                    qty = int(max_invest / (buy_price * 1.0003) / 100) * 100
                    if qty < 100:
                        continue

                    cost = buy_price * qty * 1.0003
                    if cost > cash:
                        continue

                    cash -= cost
                    positions[sym] = {
                        'qty': qty,
                        'buy_price': buy_price,
                        'buy_date': date,
                        'days': 0,
                        'signal': row.get('daily_signal', ''),
                        'wave_stage': row.get('wave_stage', ''),
                    }
                    trades.append({
                        'date': date, 'symbol': sym, 'type': 'buy',
                        'price': buy_price, 'qty': qty,
                        'reason': f"买入:{row.get('daily_signal','')} {row.get('wave_stage','')}"
                    })

        # ---- 权益记录 ----
        total = cash
        for sym, pos in positions.items():
            p = get_close(date, sym)
            if p:
                total += p * pos['qty']
        equity.append({'date': date, 'total': total})

    # 期末清算
    last_date = dates[-1]
    for sym, pos in list(positions.items()):
        p = get_close(last_date, sym)
        if p:
            cash += p * pos['qty']
            trades.append({
                'date': last_date, 'symbol': sym, 'type': 'final',
                'price': p, 'qty': pos['qty'],
                'ret': (p - pos['buy_price']) / pos['buy_price'],
                'reason': '期末清算'
            })

    total_value = cash
    peak = cfg['initial_cash']
    max_dd = 0.0
    for e in equity:
        if e['total'] > peak:
            peak = e['total']
        dd = (peak - e['total']) / peak
        if dd > max_dd:
            max_dd = dd

    sells = [t for t in trades if t['type'] in ('sell', 'final')]
    wins = [t for t in sells if t.get('pnl', 0) > 0]

    return {
        'year': year,
        'initial': cfg['initial_cash'],
        'final': total_value,
        'ret': (total_value - cfg['initial_cash']) / cfg['initial_cash'],
        'max_drawdown': max_dd,
        'trades': len([t for t in trades if t['type'] in ('buy', 'sell', 'final')]),
        'buys': len([t for t in trades if t['type'] == 'buy']),
        'sells': len(sells),
        'win_rate': len(wins) / len(sells) if sells else 0,
        'equity': equity,
    }


def print_result(r: dict):
    emoji = "🟢" if r['ret'] > 0 else "🔴"
    print(f"{emoji} {r['year']}年 | 收益: {r['ret']:+.2%} | 最大回撤: {r['max_drawdown']:+.2%} | "
          f"交易: {r['buys']}买/{r['sells']}卖 | 胜率: {r['win_rate']:.0%}")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("WavechanFast 完整回测 2020-2025")
    print("=" * 60)

    db = ParquetDatabaseIntegrator()
    config = {
        'wave_threshold_pct': 0.025,
        'decline_threshold': -0.15,
        'consolidation_threshold': 0.05,
        'stop_loss_pct': 0.03,
        'profit_target_pct': 0.25,
    }

    all_results = {}

    for year in YEARS:
        print(f"\n📅 {year} 年加载数据...")
        t0 = time.time()

        # 加载当年+预热数据(前150天)
        start = f'{year}-01-01'
        end = f'{year}-12-31'
        ext_start = f'{year - 1}-09-01'

        df = db.fetch_daily_data(
            ext_start, end,
            columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        )

        if df.empty:
            print(f"  ⚠️  无数据")
            continue

        print(f"  📊 {len(df)} 条, {df['symbol'].nunique()} 只股票, 加载 {time.time()-t0:.1f}s")

        t1 = time.time()
        features = compute_wavechan_features_fast(df, config)
        print(f"  ⚡ 特征计算 {time.time()-t1:.1f}s, {len(features)} 条")

        t2 = time.time()
        result = run_year_backtest(year, features)
        if result:
            all_results[year] = result
            print_result(result)
            print(f"  ⏱️  回测 {time.time()-t2:.1f}s")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("📊 汇总")
    print("=" * 60)
    print(f"{'年份':<6} {'收益':>10} {'最大回撤':>10} {'交易':>6} {'胜率':>6}")
    print("-" * 45)

    total_ret = 0
    for year in YEARS:
        if year not in all_results:
            print(f"{year:<6} {'无数据':>10}")
            continue
        r = all_results[year]
        total_ret = (1 + total_ret) * (1 + r['ret']) - 1
        print(f"{year:<6} {r['ret']:>+10.2%} {r['max_drawdown']:>+10.2%} "
              f"{r['buys']:>3}买/{r['sells']:>3}卖 {r['win_rate']:>6.0%}")

    if all_results:
        avg_ret = sum(r['ret'] for r in all_results.values()) / len(all_results)
        avg_dd = sum(r['max_drawdown'] for r in all_results.values()) / len(all_results)
        print(f"\n{'年均':<6} {avg_ret:>+10.2%} {avg_dd:>+10.2%}")
        print(f"{'累计':<6} {total_ret:>+10.2%}")

        # 对比Score
        print("\n" + "=" * 60)
        print("📊 vs Score策略")
        print("=" * 60)
        score = {2020:0.0129, 2021:-0.0098, 2022:-0.1868, 2023:0.0396, 2024:-0.1011, 2025:0.0507}
        print(f"{'年份':<6} {'Score':>10} {'Wavechan':>12} {'差异':>10}")
        print("-" * 42)
        for year in YEARS:
            if year not in all_results:
                continue
            sp = score.get(year, 0)
            wc = all_results[year]['ret']
            diff = wc - sp
            arrow = "⬆️" if diff > 0 else "⬇️"
            print(f"{year:<6} {sp:>+10.2%} {wc:>+12.2%} {diff:>+10.2%} {arrow}")
