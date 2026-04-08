#!/usr/bin/env python3
"""F窗口参数扫描 - 12组参数 × 1窗口 = 12次回测，约15-20分钟"""
import itertools, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

NP_VALS = [0.6, 0.7, 0.8, 1.0]
BP_VALS = [0.2, 0.3, 0.4]
PARAMS = [{"np": np, "bp": bp} for np in NP_VALS for bp in BP_VALS if bp <= np]
print(f"参数组合: {len(PARAMS)} 组")

results = []

for i, p in enumerate(PARAMS):
    np_val, bp_val = p["np"], p["bp"]
    label = f"NP={np_val}_BP={bp_val}"
    print(f"\n[{i+1}/{len(PARAMS)}] {label}")

    t0 = time.time()
    import subprocess
    # 用 --neutral-position --bear-position 参数跑回测
    cmd = [
        "python3", "backtest.py",
        "--strategy", "v8",
        "--start", "2010-01-01",
        "--end", "2012-12-31",
        "--market-filter",
        "--neutral-position", str(np_val),
        "--bear-position", str(bp_val),
        "--filter-confirm-days", "1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    # 解析输出
    output = r.stdout + r.stderr
    final_val, annual_ret, max_dd, sharpe = None, None, None, None
    for line in output.split("\n"):
        if "最终价值" in line or "final" in line.lower():
            for tok in line.replace(",", "").split():
                if tok.replace(".", "").replace("-", "").isdigit():
                    if final_val is None:
                        try:
                            final_val = float(tok)
                        except:
                            pass
        if "年化" in line and "夏普" not in line:
            for tok in line.split():
                if tok.startswith(("+", "-")) and "%" in tok:
                    try:
                        annual_ret = float(tok.replace("%", ""))
                    except:
                        pass
        if "最大回撤" in line or "MaxDD" in line:
            for tok in line.replace("=", ":").split():
                if tok.startswith(("-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    try:
                        max_dd = float(tok.replace("%", "").replace("-", ""))
                        break
                    except:
                        pass
        if "夏普" in line:
            for tok in line.split():
                try:
                    v = float(tok)
                    sharpe = v
                except:
                    pass

    print(f"  → 年化{annual_ret}%, MaxDD={max_dd}%, 夏普={sharpe}, 耗时{elapsed:.0f}s")
    results.append({
        "np": np_val, "bp": bp_val,
        "annual_ret": annual_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "elapsed": elapsed,
    })

# 排序并输出
results.sort(key=lambda x: x["max_dd"])
print("\n\n=== F窗口 参数扫描结果（按MaxDD排序）===")
print(f"{'NP':>4} {'BP':>4} {'年化%':>8} {'MaxDD%':>8} {'夏普':>6}")
for r in results:
    print(f"{r['np']:>4.1f} {r['bp']:>4.1f} {r['annual_ret']:>8.2f} {r['max_dd']:>8.2f} {r['sharpe']:>6.2f}")

best = results[0]
print(f"\n最优参数: NP={best['np']}, BP={best['bp']}, MaxDD={best['max_dd']}%")
