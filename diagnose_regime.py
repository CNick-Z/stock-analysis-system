#!/usr/bin/env python3
"""诊断 MarketRegimeFilter 在场景A和场景B中的行为差异"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pandas as pd
from simulator.market_regime import MarketRegimeFilter

INDEX_PATH = "/data/warehouse/indices/CSI300.parquet"
KEY_DATES = ['2022-01-04', '2022-04-01', '2022-10-01']

print("=" * 80)
print("场景A：单独跑 2022")
print("=" * 80)
fA = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
dfA = fA.prepare('2022-01-01', '2022-12-31')
print(f"\nDataFrame 区间: {dfA['date'].min()} ~ {dfA['date'].max()}")
print(f"DataFrame 总行数: {len(dfA)}")

# 检查 2022 年 regime 分布
dfA_2022 = dfA[dfA['date'] >= '2022-01-01']
print(f"\n2022 年 regime 分布:")
print(dfA_2022['regime'].value_counts())

print("\n关键日期详情 (场景A):")
for d in KEY_DATES:
    row = dfA[dfA['date'] == d]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {d}: regime={r['regime']}, consecutive_days={r['consecutive_days']}, "
              f"bear_cond={r.get('bear_cond', False)}, neutral_rsi_cond={r.get('neutral_rsi_cond', False)}, "
              f"neutral_trend_cond={r.get('neutral_trend_cond', False)}, bull_cond={r.get('bull_cond', False)}")

# 模拟 get_regime 状态机
print("\n场景A get_regime 状态机追踪:")
fA2 = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
fA2.prepare('2022-01-01', '2022-12-31')
for d in KEY_DATES:
    result = fA2.get_regime(d)
    print(f"  {d}: locked={result['regime']}, raw={result['raw_regime']}, "
          f"consec_days={result['consecutive_regime_days']}, limit={result['position_limit']}")

print("\n" + "=" * 80)
print("场景B：合并跑 2020-2022")
print("=" * 80)
fB = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
dfB = fB.prepare('2020-01-01', '2022-12-31')
print(f"\nDataFrame 区间: {dfB['date'].min()} ~ {dfB['date'].max()}")
print(f"DataFrame 总行数: {len(dfB)}")

# 检查 2022 年 regime 分布
dfB_2022 = dfB[dfB['date'] >= '2022-01-01']
print(f"\n2022 年 regime 分布:")
print(dfB_2022['regime'].value_counts())

print("\n关键日期详情 (场景B):")
for d in KEY_DATES:
    row = dfB[dfB['date'] == d]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {d}: regime={r['regime']}, consecutive_days={r['consecutive_days']}, "
              f"bear_cond={r.get('bear_cond', False)}, neutral_rsi_cond={r.get('neutral_rsi_cond', False)}, "
              f"neutral_trend_cond={r.get('neutral_trend_cond', False)}, bull_cond={r.get('bull_cond', False)}")

# 模拟 get_regime 状态机
print("\n场景B get_regime 状态机追踪:")
fB2 = MarketRegimeFilter(index_path=INDEX_PATH, confirm_days=1, regime_persist_days=3)
fB2.prepare('2020-01-01', '2022-12-31')
for d in KEY_DATES:
    result = fB2.get_regime(d)
    print(f"  {d}: locked={result['regime']}, raw={result['raw_regime']}, "
          f"consec_days={result['consecutive_regime_days']}, limit={result['position_limit']}")

print("\n" + "=" * 80)
print("对比 DataFrame consecutive_days 差异")
print("=" * 80)
for d in KEY_DATES:
    rowA = dfA[dfA['date'] == d]
    rowB = dfB[dfB['date'] == d]
    if not rowA.empty and not rowB.empty:
        rA = rowA.iloc[0]
        rB = rowB.iloc[0]
        same = (rA['consecutive_days'] == rB['consecutive_days'] and 
                rA['regime'] == rB['regime'])
        print(f"  {d}: A→regime={rA['regime']}, consec={rA['consecutive_days']} | "
              f"B→regime={rB['regime']}, consec={rB['consecutive_days']} | "
              f"{'SAME' if same else 'DIFFERENT!'}")

print("\n" + "=" * 80)
print("检查 _df 中 2022-01 附近的 regime 变化")
print("=" * 80)
for label, f, label2, f2 in [("A", fA, "B", fB)]:
    print(f"\n--- {label} ---")
    window = f._df[(f._df['date'] >= '2021-12-01') & (f._df['date'] <= '2022-01-15')]
    for _, r in window.iterrows():
        d = str(r['date'])
        if d[:10] >= '2021-12-20':
            print(f"  {d[:10]}: regime={r['regime']}, consec_days={r['consecutive_days']}, "
                  f"bear_cond={r.get('bear_cond', False)}, fast_stop={r.get('fast_stop_loss', False)}")
