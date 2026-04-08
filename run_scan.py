#!/usr/bin/env python3
"""快速参数扫描 - 直接跑框架，不用CLI"""
import sys, time, json
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

from simulator.market_regime import MarketRegimeFilter
from simulator.base_framework import BaseFramework
from simulator.shared import load_strategy
from backtest import load_data_for_strategy

INITIAL_CASH = 1_000_000.0

def run_window(strat, start, end, np_val, bp_val):
    strategy = load_strategy(strat)
    mrf = MarketRegimeFilter(
        confirm_days=1,
        neutral_position=np_val,
        bear_position=bp_val,
        regime_persist_days=3,
    )
    mrf.prepare(start, end)

    fw = BaseFramework(
        initial_cash=INITIAL_CASH,
        state_file=f"/tmp/scan_{np_val}_{bp_val}.json",
        market_regime_filter=mrf,
    )
    fw._strategy = strategy
    fw.reset()

    years = list(range(int(start[:4]), int(end[:4]) + 1))
    for year in years:
        y_start = max(f"{year}-01-01", start)
        y_end = min(f"{year}-12-31", end)
        if y_start > y_end:
            continue
        try:
            df = load_data_for_strategy(strat, y_start, y_end)
        except Exception as e:
            print(f"    数据失败 [{year}]: {e}")
            continue
        dates = sorted([d for d in df["date"].unique() if y_start <= d <= y_end])
        if hasattr(strategy, "prepare"):
            try:
                strategy.prepare(dates, df)
            except Exception:
                pass
        for date in dates:
            fw._on_day(date, df, dates)

        # 年度快照重置
        for attr in ("positions", "cash", "n_winning", "n_total"):
            if attr == "positions":
                setattr(fw, attr, {})
            else:
                setattr(fw, attr, INITIAL_CASH if attr == "cash" else 0)
        fw.trades = []
        del df

    final_val = fw.cash + sum(
        p.get("latest_price", p["avg_cost"]) * p["qty"]
        for p in fw.positions.values()
    )

    values = [s["total_value"] for s in fw.market_snapshots]
    peak = INITIAL_CASH
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    n_days = max(len(values), 1)
    n_years = n_days / 244
    annual = (final_val / INITIAL_CASH) ** (1 / n_years) - 1 if n_years > 0.5 else 0.0

    if len(values) > 2:
        rets = pd_series_pct(values)
        sharpe = rets.mean() / rets.std() * 244 ** 0.5 if rets.std() > 1e-10 else 0.0
    else:
        sharpe = 0.0

    return {
        "final_value": final_val,
        "annual_return": annual * 100,
        "max_drawdown": max_dd * 100,
        "sharpe": sharpe,
    }

def pd_series_pct(values):
    import pandas as pd
    return pd.Series(values).pct_change().dropna()

# 参数组合（精简4×2=8组）
NP_VALS = [0.6, 0.7, 0.8, 1.0]
BP_VALS = [0.2, 0.3]
PARAMS = [(np, bp) for np in NP_VALS for bp in BP_VALS if bp <= np]

results = []
WINDOW = ("2010-01-01", "2012-12-31", "F")

for i, (np_val, bp_val) in enumerate(PARAMS):
    label = f"NP={np_val}_BP={bp_val}"
    print(f"[{i+1}/{len(PARAMS)}] {WINDOW[2]}窗口 {label} ...", flush=True)
    t0 = time.time()
    r = run_window("v8", WINDOW[0], WINDOW[1], np_val, bp_val)
    elapsed = time.time() - t0
    print(f"  年化={r['annual_return']:.2f}% MaxDD={r['max_drawdown']:.2f}% 夏普={r['sharpe']:.2f} ({elapsed:.0f}s)")
    results.append({"np": np_val, "bp": bp_val, **r})

# 排序
results.sort(key=lambda x: x["max_drawdown"])
print(f"\n{'='*60}")
print(f"最优组合（MaxDD最浅）:")
best = results[0]
print(f"  NP={best['np']} BP={best['bp']} -> MaxDD={best['max_drawdown']:.2f}% 年化={best['annual_return']:.2f}%")
print(f"\n{'NP':>4} {'BP':>4} {'年化%':>8} {'MaxDD%':>9} {'夏普':>6}")
for r in results:
    print(f"{r['np']:>4.1f} {r['bp']:>4.1f} {r['annual_return']:>8.2f} {r['max_drawdown']:>9.2f} {r['sharpe']:>6.2f}")

with open("/tmp/scan_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n结果已写入 /tmp/scan_results.json")
