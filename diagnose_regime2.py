#!/usr/bin/env python3
"""深度诊断：找出场景A和场景B在2022年中具体哪些日期的regime不同"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pandas as pd
from simulator.market_regime import MarketRegimeFilter

INDEX_PATH = "/data/warehouse/indices/CSI300.parquet"

# 准备两个 DataFrame
fA = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
fB = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
dfA = fA.prepare('2022-01-01', '2022-12-31')
dfB = fB.prepare('2020-01-01', '2022-12-31')

# 只看 2022 年的数据
dfA_2022 = dfA[dfA['date'] >= '2022-01-01'].copy()
dfB_2022 = dfB[dfB['date'] >= '2022-01-01'].copy()

# 找出 regime 不同的日期
merged = dfA_2022.merge(dfB_2022, on='date', suffixes=('_A', '_B'))
diff = merged[merged['regime_A'] != merged['regime_B']]
print(f"2022年 regime 不同的日期数: {len(diff)}")
if len(diff) > 0:
    print("\n前30个不同的日期:")
    print(diff[['date', 'regime_A', 'regime_B', 'consecutive_days_A', 'consecutive_days_B']].head(30).to_string())

# 追踪场景B的状态机在2022年关键切换点
print("\n" + "=" * 80)
print("场景B 状态机追踪（2022年每一天）")
print("=" * 80)
fB2 = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
fB2.prepare('2020-01-01', '2022-12-31')

prev_locked = None
switches = []
for d in sorted(dfB_2022['date'].tolist()):
    result = fB2.get_regime(d)
    if result['regime'] != prev_locked:
        switches.append((d, prev_locked, result['regime'], result['raw_regime'], result['consecutive_days']))
        prev_locked = result['regime']

print(f"2022年 regime 切换次数: {len(switches)}")
for s in switches:
    print(f"  {s[0]}: {s[1]} → {s[2]} (raw={s[3]}, consec={s[4]})")

# 同样追踪场景A
print("\n" + "=" * 80)
print("场景A 状态机追踪（2022年每一天）")
print("=" * 80)
fA2 = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
fA2.prepare('2022-01-01', '2022-12-31')

prev_locked = None
switches_A = []
for d in sorted(dfA_2022['date'].tolist()):
    result = fA2.get_regime(d)
    if result['regime'] != prev_locked:
        switches_A.append((d, prev_locked, result['regime'], result['raw_regime'], result['consecutive_days']))
        prev_locked = result['regime']

print(f"2022年 regime 切换次数: {len(switches_A)}")
for s in switches_A:
    print(f"  {s[0]}: {s[1]} → {s[2]} (raw={s[3]}, consec={s[4]})")

# 比较两者的 locked regime 序列
print("\n" + "=" * 80)
print("对比两场景 locked regime 序列（抽查每10天）")
print("=" * 80)
dates_check = sorted(dfA_2022['date'].tolist())
for i, d in enumerate(dates_check):
    if i % 10 == 0:
        rA = fA2.get_regime(d)
        rB = fB2.get_regime(d)
        same = rA['regime'] == rB['regime']
        print(f"  {d}: A={rA['regime']}(raw={rA['raw_regime']}, consec={rA['consecutive_days']}) | "
              f"B={rB['regime']}(raw={rB['raw_regime']}, consec={rB['consecutive_days']}) | "
              f"{'SAME' if same else 'DIFF!'}")

# 深入检查 2022-10 附近的 BEAR 触发
print("\n" + "=" * 80)
print("2022年10月 regime 详细数据（DataFrame raw regime）")
print("=" * 80)
dfA_oct = dfA[dfA['date'].between('2022-09-20', '2022-11-10')]
dfB_oct = dfB[dfB['date'].between('2022-09-20', '2022-11-10')]
for d in sorted(dfA_oct['date'].tolist()):
    if d >= '2022-09-20':
        rA = dfA_oct[dfA_oct['date'] == d].iloc[0]
        rB = dfB_oct[dfB_oct['date'] == d].iloc[0]
        diff_mark = "DIFF!" if rA['regime'] != rB['regime'] else ""
        print(f"  {d}: A={rA['regime']}(c={rA['consecutive_days']}) | B={rB['regime']}(c={rB['consecutive_days']}) {diff_mark}")
