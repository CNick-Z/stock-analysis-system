#!/usr/bin/env python3
"""最终验证：追踪 Scenario B 中每日的 locked regime"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')
from simulator.market_regime import MarketRegimeFilter

f = MarketRegimeFilter(confirm_days=1, regime_persist_days=3)
f.prepare('2020-01-01', '2022-12-31')

# 追踪整个 2020-2022 的 locked regime
prev_locked = None
lock_changes = []

for d in sorted(f._df[f._df['date'] >= '2020-01-01']['date'].tolist()):
    d_str = str(d)[:10]
    r = f.get_regime(d_str)
    if r['regime'] != prev_locked:
        lock_changes.append((d_str, prev_locked, r['regime'], r['raw_regime'], r['consecutive_days']))
        prev_locked = r['regime']
    # Stop after we have enough
    if d_str > '2022-12-30':
        break

print("=== Scenario B locked regime 切换记录 ===")
for entry in lock_changes:
    print(f"  {entry[0]}: {entry[1]} → {entry[2]} (raw={entry[3]}, consec={entry[4]})")

# 检查 Scenario A
print("\n=== Scenario A locked regime 切换记录 ===")
fA = MarketRegimeFilter(confirm_days=1, regime_persist_days=3)
fA.prepare('2022-01-01', '2022-12-31')

prev_locked = None
for d in sorted(fA._df[fA._df['date'] >= '2022-01-01']['date'].tolist()):
    d_str = str(d)[:10]
    r = fA.get_regime(d_str)
    if r['regime'] != prev_locked:
        print(f"  {d_str}: {prev_locked} → {r['regime']} (raw={r['raw_regime']}, consec={r['consecutive_days']})")
        prev_locked = r['regime']
    if d_str > '2022-12-30':
        break

# 核心问题：为什么 BULL 一旦锁定就再也无法切换？
print("\n=== 核心问题：BEAR 连续天数永远不够 ===")
# 检查 2020-2022 年中所有 BEAR 连续天数
df_2022 = f._df[f._df['date'].between('2022-01-01', '2022-12-31')]
bear_days = df_2022[df_2022['regime'] == 'BEAR']
print(f"2022年 BEAR 总天数: {len(bear_days)}")
print(f"2022年 BEAR 最大连续天数: {bear_days['consecutive_days'].max() if len(bear_days) > 0 else 0}")
print(f"BEAR 触发锁定所需天数: 3")
print(f"\nBEAR 永远无法锁定！因此 position_limit 永远是 1.0 (BULL) 或 0.7 (NEUTRAL)")
print(f"而 NEUTRAL 的 consecutive_days 来自 neutral_rsi_consec，一累积就是几十天")
print(f"所以 NEUTRAL 一旦锁定，BEAR 的 1-3 天永远无法打破 3 天阈值")

# 检查 2020 年的 BEAR 连续天数
df_2020 = f._df[f._df['date'].between('2020-01-01', '2020-12-31')]
bear_days_2020 = df_2020[df_2020['regime'] == 'BEAR']
print(f"\n2020年 BEAR 总天数: {len(bear_days_2020)}")
print(f"2020年 BEAR 最大连续天数: {bear_days_2020['consecutive_days'].max() if len(bear_days_2020) > 0 else 0}")

print("\n=== 结论 ===")
print("场景B（2020-2022）: 首次交易日 2020-01-02，warmup 落在 2019-12-02")
print("  → 2019-12 已有 bull_cond=True，consecutive_days 不断累积")
print("  → 2020-01-02 首次调用: raw=BULL, consecutive_days=3 → BULL立即锁定")
print("  → 此后两年，BEAR 最多只有 1-3 天，从未达到 regime_persist_days=3")
print("  → BEAR 无法打破 BULL 的锁定，position_limit 永远是 1.0")
print("  → filter 完全失效！")
print()
print("场景A（2022 alone）: 首次交易日 2022-01-04，warmup 落在 2019-12-01")
print("  → 2022-01-04: 所有 cond 都是 False → regime=NEUTRAL")
print("  → 首次调用: raw=NEUTRAL, consecutive_days=0 → NEUTRAL立即锁定")
print("  → NEUTRAL 锁定后，BEAR 的 1-3 天同样无法打破")
print("  → BUT! NEUTRAL 的 position_limit=0.7 (不是 1.0)，所以 filter 有部分效果！")
print("  → 实际效果：position_limit=0.7（比无过滤的 1.0 低），所以 cash 更高")
