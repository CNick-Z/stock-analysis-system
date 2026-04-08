#!/usr/bin/env python3
"""Debug script for position_limit not working in Windows B/E"""
import pandas as pd
import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

from simulator.market_regime import MarketRegimeFilter
from simulator.base_framework import BaseFramework as BacktestFramework
from utils.data_loader import load_strategy_data
import logging

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

# ── Step 1: 检查 MarketRegimeFilter 对 2022-09-30 的返回值 ──────────────
print("=" * 60)
print("STEP 1: 检查 MarketRegimeFilter.get_regime('2022-09-30')")
print("=" * 60)

mrf = MarketRegimeFilter(confirm_days=2, neutral_position=0.70, bear_position=0.30)
mrf.prepare('2022-01-01', '2023-12-31')

test_dates = ['2022-09-30', '2022-10-01', '2022-11-01', '2022-12-01']
for d in test_dates:
    try:
        ri = mrf.get_regime(d)
        print(f"\n{d}:")
        print(f"  regime           = {ri['regime']}")
        print(f"  raw_regime       = {ri['raw_regime']}")
        print(f"  position_limit   = {ri['position_limit']}")
        print(f"  signal           = {ri['signal']}")
        print(f"  rsi14            = {ri['rsi14']:.2f}")
        print(f"  csi300_close     = {ri['csi300_close']:.2f}")
        print(f"  ma20             = {ri['ma20']:.2f}")
        print(f"  consecutive_days = {ri['consecutive_days']}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ── Step 2: 检查 2018 年的几个日期 ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: 检查 2018 年 MarketRegimeFilter")
print("=" * 60)

mrf2018 = MarketRegimeFilter(confirm_days=2, neutral_position=0.70, bear_position=0.30)
mrf2018.prepare('2018-01-01', '2018-12-31')

test_dates_2018 = ['2018-02-01', '2018-06-01', '2018-10-01', '2018-12-01']
for d in test_dates_2018:
    try:
        ri = mrf2018.get_regime(d)
        print(f"\n{d}:")
        print(f"  regime           = {ri['regime']}")
        print(f"  raw_regime       = {ri['raw_regime']}")
        print(f"  position_limit   = {ri['position_limit']}")
        print(f"  signal           = {ri['signal']}")
        print(f"  rsi14            = {ri['rsi14']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ── Step 3: 跑单天 debug ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: 跑 2022-09-30 单天 debug（BaseFramework._process_buys）")
print("=" * 60)

from simulator.shared import load_strategy

df = load_strategy_data(years=[2022], add_money_flow=True)
df = df[(df['date'] >= '2022-09-01') & (df['date'] <= '2022-09-30')].copy()

# 添加 next_open
from simulator.shared import add_next_open
df = add_next_open(df)

strategy = load_strategy('v8')
if hasattr(strategy, 'prepare'):
    dates_list = sorted(df['date'].unique())
    strategy.prepare(dates_list, df)

# ── 自定义 DebugFramework ─────────────────────────────────────────────────
class DebugFramework(BacktestFramework):
    def _process_buys(self, full_df, daily, market):
        position_limit = 1.0
        ri_regime = None
        if self.market_regime_filter:
            ri_regime = self.market_regime_filter.get_regime(market["date"])
            position_limit = ri_regime["position_limit"]
            print(f"\n[DEBUG _process_buys] date={market['date']}")
            print(f"  regime={ri_regime['regime']}, raw_regime={ri_regime['raw_regime']}")
            print(f"  position_limit={position_limit}")
            print(f"  cash={self.cash}")
        
        avail = self.cash * position_limit
        print(f"  position_limit={position_limit} => avail_cash={avail:,.0f}")

        return super()._process_buys(full_df, daily, market)

framework = DebugFramework(
    initial_cash=1_000_000,
    market_regime_filter=mrf,
)
framework._strategy = strategy
framework.reset()

# 跑9月27日（9月30日前的交易日）
debug_date = '2022-09-30'
daily = df[df['date'] == debug_date].copy()
if daily.empty:
    # 找最近的可交易日
    all_dates = sorted(df['date'].unique())
    print(f"  9月30日无数据，可用日期: {all_dates[-3:]}")
    debug_date = all_dates[-1]
    daily = df[df['date'] == debug_date].copy()

print(f"\n  使用日期: {debug_date}")
print(f"  市场数据行数: {len(daily)}")

# 构建 market dict
market = {
    "date": debug_date,
    "cash": framework.cash,
    "total_value": framework.cash,
    "next_date": debug_date,
}

# 调用 _process_buys
framework._process_buys(df, daily, market)

print(f"\n  买完后 cash={framework.cash:,.0f}")
print(f"  持仓数: {len(framework.positions)}")
for sym, pos in framework.positions.items():
    print(f"    {sym}: qty={pos['qty']}, avg_cost={pos['avg_cost']:.2f}")

# ── Step 4: 检查 base_framework 的 _process_buys 中 avail_cash 计算 ──────
print("\n" + "=" * 60)
print("STEP 4: 检查 avail_cash 计算（关键路径）")
print("=" * 60)

# 模拟计算
cash = 1_000_000
slots = 5 - 0  # 0 positions
fill_count = 2

for limit in [1.0, 0.70, 0.30]:
    avail = cash * limit
    per_stock = min(avail / fill_count, cash * 0.20)
    cost = per_stock * 1.001  # approx commission
    print(f"\nlimit={limit}: avail={avail:,.0f}, per_stock={per_stock:,.0f}, cost~={cost:,.0f}")
    print(f"  是否超过cash: {cost > cash * limit * 0.9}")  # 大概判断

print("\n[结论] 如果 limit=0.30 但 per_stock 仍然是 200,000（0.2M），")
print("       说明 min(avail/1, cash*0.20) 取了后者，说明 position_size 掩盖了 limit 的效果。")
