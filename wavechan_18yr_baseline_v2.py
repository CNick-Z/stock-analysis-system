#!/usr/bin/env python3
"""
Wavechan 策略 18年完整基线测试 2008-2025（优化版）
✅ 分年加载：每年只加载 14个月（当年+预热期）
✅ 内存安全：每年独立，算完即释放
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
WARMUP_MONTHS = 2  # 预热期：前2个月（10月1日开始 = 14个月）


def run_year_backtest(year_df, dates, cfg, cash_start):
    """单年回测"""
    positions = {}
    cash = cash_start
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
        i = date_idx.get(date, -1)
        if i >= 0 and i + 1 < len(dates):
            nd = dates[i + 1]
            try:
                if nd in date_set and sym in open_lu.columns:
                    p = open_lu.loc[nd, sym]
                    return p if not pd.isna(p) else get_close(nd, sym)
            except: pass
            return get_close(nd, sym)
        return None

    for date in dates:
        day_df = year_df[year_df['date'] == date]

        # 卖出
        for sym in list(positions.keys()):
            pos = positions[sym]
            pos['days'] += 1
            cur = get_close(date, sym)
            if cur is None: continue
            ret = (cur - pos['buy_price']) / pos['buy_price']
            reason = ''
            d = year_df[(year_df['date'] == date) & (year_df['symbol'] == sym)]
            if not d.empty:
                ss = d.iloc[0].get('chan_first_sell', False) or d.iloc[0].get('chan_second_sell', False)
                if ret <= -cfg['stop_loss_pct']: reason = '止损'
                elif ret >= cfg['profit_target_pct']: reason = '止盈'
                elif pos['days'] >= cfg['max_hold_days']: reason = '超时'
                elif ss: reason = '缠论'
            if reason:
                sp = get_next_open(date, sym) or cur
                cash += sp * pos['qty']
                del positions[sym]

        # 买入
        if len(positions) < cfg['max_positions']:
            cands = day_df[
                (day_df['daily_signal'] == '买入') &
                (~day_df['symbol'].isin(positions.keys()))
            ].nlargest(cfg['buy_slots_per_day'], 'daily_confidence')
            for _, row in cands.iterrows():
                if len(positions) >= cfg['max_positions']: break
                sym = row['symbol']
                bp = get_next_open(date, sym)
                if bp is None or bp <= 0: continue
                qty = int(cash * cfg['position_size'] / (bp * 1.0003) / 100) * 100
                if qty < 100: continue
                cost = bp * qty * 1.0003
                if cost > cash: continue
                cash -= cost
                positions[sym] = {'qty': qty, 'buy_price': bp, 'days': 0}

    # 清算
    last = dates[-1]
    for sym, pos in list(positions.items()):
        p = get_close(last, sym)
        if p: cash += p * pos['qty']
    return cash


if __name__ == '__main__':
    YEARS = list(range(2008, 2026))  # 2008-2025
    db = ParquetDatabaseIntegrator()
    config = {
        'wave_threshold_pct': 0.025,
        'decline_threshold': -0.15,
        'consolidation_threshold': 0.05,
        'stop_loss_pct': 0.03,
        'profit_target_pct': 0.25,
    }

    print("=" * 70)
    print("Wavechan 18年基线测试（分年版）2008-2025")
    print("=" * 70)

    results = {}
    cash = CONFIG['initial_cash']

    print(f"\n{'年份':<6} {'市场':<8} {'年初净值':>10} {'年末净值':>10} {'收益率':>10} {'vs沪深300':>10}")
    print("-" * 65)

    hs300 = {
        2008:-65, 2009:97, 2010:-13, 2011:-25, 2012:8,
        2013:-8, 2014:52, 2015:6, 2016:-11, 2017:22,
        2018:-25, 2019:37, 2020:27, 2021:-5, 2022:-21,
        2023:-11, 2024:15, 2025:15
    }

    for year in YEARS:
        # ========== 1. 加载当年+预热期数据（约14个月）==========
        warmup_start = f'{year-1}-10-01'
        warmup_end = f'{year}-12-31'
        t0 = time.time()
        raw = db.fetch_daily_data(
            warmup_start, warmup_end,
            columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        )
        load_time = time.time() - t0

        if raw.empty:
            print(f"{year:<6} 无数据")
            continue

        # ========== 2. 计算特征（只算当年的）==========
        t1 = time.time()
        features = compute_wavechan_features_fast(raw, config)
        feat_time = time.time() - t1

        # ========== 3. 截取当年数据用于回测 ==========
        year_df = features[features['date'].str.startswith(str(year))].copy()
        if year_df.empty:
            print(f"{year:<6} 无当年数据")
            del raw, features
            continue
        dates = sorted(year_df['date'].unique())
        n_stocks = year_df['symbol'].nunique()

        # ========== 4. 跑回测 ==========
        t2 = time.time()
        cash = run_year_backtest(year_df, dates, CONFIG, cash)
        bt_time = time.time() - t2

        ret = (cash - CONFIG['initial_cash']) / CONFIG['initial_cash']
        market = "熊市" if year in (2008,2011,2018,2022) else ("牛市" if year in (2009,2014,2015,2020,2025) else "震荡")
        vs_hs = f"{ret*100 - hs300.get(year,0):+.1f}%"
        results[year] = {'ret': ret, 'market': market, 'final': cash}
        print(f"{year:<6} {market:<8} {CONFIG['initial_cash']:>10,.0f} {cash:>10,.0f} {ret:>+10.2%} {vs_hs:>10} ({n_stocks}只,加载{load_time:.0f}s,特征{feat_time:.0f}s,回测{bt_time:.1f}s)")

        # ========== 5. 释放内存 ==========
        del raw, features, year_df

    # 汇总
    print("\n" + "=" * 70)
    print("📊 汇总")
    print("=" * 70)
    total_cumret = (cash - CONFIG['initial_cash']) / CONFIG['initial_cash']
    print(f"  18年累计收益: {total_cumret:+.2%}")
    print(f"  年化收益: {(total_cumret+1)**(1/18)-1:+.2%}")
    print(f"  最终净值: {cash:,.0f}")

    bull = [y for y in YEARS if y in (2009,2014,2015,2020,2025)]
    bear = [y for y in YEARS if y in (2008,2011,2018,2022)]
    mix = [y for y in YEARS if y not in bull + bear]

    if bull:
        bull_avg = sum(results[y]['ret'] for y in bull if y in results) / len(bull)
        print(f"\n  牛市年平均: {bull_avg:+.2%} ({len(bull)}年)")
    if bear:
        bear_avg = sum(results[y]['ret'] for y in bear if y in results) / len(bear)
        print(f"  熊市年平均: {bear_avg:+.2%} ({len(bear)}年)")
    if mix:
        mix_avg = sum(results[y]['ret'] for y in mix if y in mix) / len(mix)
        print(f"  震荡年平均: {mix_avg:+.2%} ({len(mix)}年)")

    print(f"\n  沪深300 18年涨幅: 约 +150%（vs 本策略 {total_cumret:+.0%}）")
