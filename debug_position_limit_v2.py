#!/usr/bin/env python3
"""Debug v2: 深入分析 position_limit 不生效的真正原因"""
import pandas as pd
import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

from simulator.market_regime import MarketRegimeFilter
from simulator.base_framework import BaseFramework
from simulator.shared import load_strategy, add_next_open
from utils.data_loader import load_strategy_data
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── 加载完整 2022 数据（用于回测）───────────────────────────────────────
df = load_strategy_data(years=[2022], add_money_flow=True)
df = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2022-12-31')].copy()
df = add_next_open(df)
print(f"数据: {len(df):,} 行, {df['date'].min()} ~ {df['date'].max()}")

strategy = load_strategy('v8')

# 检查 filter_buy 在各日期返回多少候选股
dates_to_check = [
    '2022-03-01', '2022-04-01', '2022-06-01', '2022-09-01', '2022-09-30', '2022-10-01'
]

# 先 prepare
dates_list = sorted(df['date'].unique())
strategy.prepare(dates_list, df)

print("\nfilter_buy 候选数量:")
for d in dates_to_check:
    c = strategy.filter_buy(df, d)
    print(f"  {d}: {len(c)} 只候选")

# ── 关键测试：比较 单日 per_stock 金额 ────────────────────────────────────
print("\n" + "=" * 60)
print("关键分析：per_stock 计算 vs position_size")
print("=" * 60)

cash = 1_000_000
max_pos = 5
pos_size = 0.20

for limit, name in [(1.0, "无filter(BULL)"), (0.7, "NEUTRAL"), (0.3, "BEAR")]:
    avail = cash * limit
    slots = max_pos
    
    for fill in [1, 2, 3]:
        per = min(avail / fill, cash * pos_size)
        cost = per * 1.001
        exceeds_avail = cost > avail
        print(f"  {name:20s} fill={fill}: per_stock={per:>10,.0f}, cost~={cost:>10,.0f}, avail={avail:>10,.0f}, exceeds_avail={exceeds_avail}")

# 核心发现
print("\n" + "=" * 60)
print("【关键发现】")
print("=" * 60)
print("""
当 limit=0.30（BEAR）且只有1只候选时：
  - avail_cash = 300,000
  - per_stock = min(300,000/1, 200,000) = 200,000  ← 注意！
  - cost ≈ 200,200 > avail_cash (300,000)? 否！
  - 所以仍然按 200,000 元/只买入了 ≈ 1/5 仓位

这说明：当 fill_count < max_positions 时，per_stock 上限由
position_size (20%) 决定，position_limit (30%) 根本无法限制
单只股票的买入金额！

position_limit 的本意是限制【总仓位】(总持仓 / 总资金)，
但代码实现的是：position_limit × cash 然后再 min(..., cash×position_size)

即：总可买入 300,000，但每只最多 200,000 → 仍然买了 1 只 200,000
→ 总持仓 = 200,000 / 1,000,000 = 20%，不是预期的 30%×20%=6%！

position_limit 被 position_size 的 min() 运算抵消了大部分效果！
""")

# ── 验证：单日实际买入行为 ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("验证：框架实际 per_stock 行为")
print("=" * 60)

mrf = MarketRegimeFilter(confirm_days=2, neutral_position=0.70, bear_position=0.30)
mrf.prepare('2022-01-01', '2022-12-31')

class DebugFramework(BaseFramework):
    def _process_buys(self, full_df, daily, market):
        if self.market_regime_filter:
            ri = self.market_regime_filter.get_regime(market["date"])
            limit = ri["position_limit"]
            avail = self.cash * limit
            slots = self.max_positions - len(self.positions)
            print(f"\n[DEBUG] {market['date']} regime={ri['regime']} limit={limit}")
            print(f"  cash={self.cash:,.0f} avail_cash={avail:,.0f}")
            
            # 模拟 candidate 迭代
            candidates = self._strategy.filter_buy(full_df, market["date"])
            scored = self._strategy.score(candidates) if not candidates.empty else pd.DataFrame()
            
            if not scored.empty:
                fill_count = 0
                to_buy = []
                for _, row in scored.iterrows():
                    if fill_count >= slots:
                        break
                    sym = row["symbol"]
                    if sym in self.positions:
                        continue
                    day_row = daily[daily["symbol"] == sym]
                    if day_row.empty:
                        continue
                    r = day_row.iloc[0]
                    exec_price = r.get("next_open")
                    if pd.isna(exec_price):
                        exec_price = r.get("open", 10.0)
                    exec_price = exec_price * 1.0001
                    if exec_price <= 0:
                        continue
                    to_buy.append((sym, row, r, exec_price))
                    fill_count += 1
                
                if fill_count > 0:
                    per_stock_raw = avail / fill_count
                    per_stock_capped = min(per_stock_raw, self.cash * self.position_size)
                    print(f"  fill_count={fill_count} slots={slots}")
                    print(f"  per_stock_raw = {per_stock_raw:,.0f} (avail/fill)")
                    print(f"  per_stock_capped = {per_stock_capped:,.0f} (min(raw, cash*pos_size={self.cash*self.position_size:,.0f}))")
                    print(f"  【结论】per_stock 实际受 position_size 上限！position_limit 被 min() 抵消")
                    
                    for sym, row, r, exec_price in to_buy:
                        qty = int(per_stock_capped / exec_price)
                        qty = (qty // 100) * 100
                        cost = qty * exec_price * 1.001
                        print(f"    买入 {sym} @ {exec_price:.2f} x {qty}股 = {cost:,.0f}元")
        return super()._process_buys(full_df, daily, market)

framework = DebugFramework(
    initial_cash=1_000_000,
    max_positions=5,
    position_size=0.20,
    market_regime_filter=mrf,
)
framework._strategy = strategy
framework.reset()

# 测试几个日期
test_dates = ['2022-03-01', '2022-06-01', '2022-09-30']
for td in test_dates:
    daily = df[df['date'] == td].copy()
    if daily.empty:
        print(f"\n{td}: 无数据")
        continue
    market = {"date": td, "cash": framework.cash, "total_value": framework.cash, "next_date": td}
    framework._process_buys(df, daily, market)
