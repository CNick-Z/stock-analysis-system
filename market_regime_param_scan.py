#!/usr/bin/env python3
"""
MarketRegimeFilter 参数扫描
============================
系统性扫描 216 组参数组合，评估 6 个回测窗口的加权综合得分。

参数空间：
    neutral_position   : 0.50 ~ 1.00, step 0.10  (6 值)
    bear_position      : 0.20 ~ 0.50, step 0.10  (4 值)
    confirm_days       : 1 ~ 3,      step 1        (3 值)
    regime_persist_days: 1 ~ 3,      step 1        (3 值)
共 6×4×3×3 = 216 组

用法：
    # 全部串行（调试用）
    python3 market_regime_param_scan.py --mode serial

    # 多核并行（推荐）
    python3 market_regime_param_scan.py --mode parallel --workers 4
"""

import argparse
import gc
import itertools
import json
import logging
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 导入回测组件 ──────────────────────────────────────────
from simulator.base_framework import BaseFramework, MarketSnapshot
from simulator.shared import load_strategy, add_next_open, STRATEGY_REGISTRY
from simulator.market_regime import MarketRegimeFilter
from utils.data_loader import load_strategy_data

# 从 backtest.py 导入 load_data_for_strategy（避免重复定义）
from backtest import load_data_for_strategy

# ── 回测窗口定义 ──────────────────────────────────────────
WINDOWS = [
    {"name": "A", "label": "牛市",  "start": "2024-01-01", "end": "2024-12-31",  "weight": 1},
    {"name": "B", "label": "震荡",  "start": "2022-01-01", "end": "2023-12-31",  "weight": 2},
    {"name": "C", "label": "震荡",  "start": "2020-01-01", "end": "2021-12-31",  "weight": 2},
    {"name": "D", "label": "熊市",  "start": "2015-06-01", "end": "2016-12-31",  "weight": 3},
    {"name": "E", "label": "熊市",  "start": "2018-01-01", "end": "2019-06-30",  "weight": 3},
    {"name": "F", "label": "熊市",  "start": "2010-01-01", "end": "2012-12-31",  "weight": 3},
]

STRATEGY_NAME = "v8"
INITIAL_CASH = 1_000_000.0


# ── 单窗口回测 ────────────────────────────────────────────

def run_single_window(
    strat_name: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
    mrf_params: Dict[str, Any],
) -> Dict[str, Any]:
    """在单个时间窗口运行带 MarketRegimeFilter 的回测，返回指标 dict。"""
    strategy = load_strategy(strat_name)
    all_years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))

    # 创建 MarketRegimeFilter（当 bear/neutral=1.0 时效果等同于无过滤器）
    mrf = MarketRegimeFilter(
        confirm_days=mrf_params["confirm_days"],
        neutral_position=mrf_params["neutral_position"],
        bear_position=mrf_params["bear_position"],
        bull_position=1.0,
        regime_persist_days=mrf_params["regime_persist_days"],
    )
    mrf.prepare(start_date, end_date)

    framework = BaseFramework(
        initial_cash=initial_cash,
        state_file=f"/tmp/mrf_scan_{id(mrf_params)}.json",
        market_regime_filter=mrf,
    )
    framework._strategy = strategy
    framework.reset()

    for year in all_years:
        y_start = max(f"{year}-01-01", start_date)
        y_end = min(f"{year}-12-31", end_date)
        if y_start > y_end:
            continue

        try:
            df = load_data_for_strategy(strat_name, y_start, y_end)
        except Exception as e:
            logger.warning(f"  数据加载失败 [{year}]: {e}")
            continue

        if hasattr(strategy, "prepare"):
            try:
                dates_for_prep = [d for d in sorted(df["date"].unique())
                                  if y_start <= d <= y_end]
                strategy.prepare(dates_for_prep, df)
            except Exception as prep_e:
                logger.debug(f"  prepare 失败 [{year}]: {prep_e}")

        dates_in_year = sorted([
            d for d in df["date"].unique()
            if y_start <= d <= y_end
        ])

        for date in dates_in_year:
            framework._on_day(date, df, dates_in_year)
            total_value = framework._calc_total_value(df, date)
            snap = MarketSnapshot(
                date=date,
                cash=framework.cash,
                total_value=total_value,
                n_positions=len(framework.positions),
                total_return=(total_value / framework.initial_cash - 1) * 100,
            )
            framework.market_snapshots.append(snap.__dict__)

        framework.trades = []
        framework.n_winning = 0
        framework.n_total = 0
        del df
        gc.collect()

    # ── 计算指标 ──────────────────────────────────────────
    final_value = framework.cash + sum(
        pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        for pos in framework.positions.values()
    )
    total_return = (final_value / framework.initial_cash - 1) * 100

    n_days = max(len(framework.market_snapshots), 1)
    n_years = n_days / 244
    annual_return = (final_value / framework.initial_cash) ** (1 / n_years) - 1 if n_years > 0.5 else 0.0

    values = [s["total_value"] for s in framework.market_snapshots]
    peak = framework.initial_cash
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    if len(values) > 1:
        daily_returns = pd.Series(values).pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * (244 ** 0.5)
                 if daily_returns.std() > 1e-10 else 0.0)
    else:
        sharpe = 0.0

    return {
        "annual_return_pct": annual_return * 100,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "n_trades": framework.n_total,
        "win_rate": framework.n_winning / max(framework.n_total, 1) * 100,
        "total_return_pct": total_return,
        "final_value": final_value,
    }


def run_all_windows(
    strat_name: str,
    windows: List[Dict],
    initial_cash: float,
    mrf_params: Dict[str, Any],
) -> Dict[str, Any]:
    """在所有窗口运行回测，返回各窗口指标 + 综合得分。"""
    window_results = {}
    weighted_sharpe_sum = 0.0
    weight_sum = 0.0

    for w in windows:
        key = w["name"]
        try:
            metrics = run_single_window(
                strat_name=strat_name,
                start_date=w["start"],
                end_date=w["end"],
                initial_cash=initial_cash,
                mrf_params=mrf_params,
            )
        except Exception as e:
            logger.error(f"  回测失败 [{key}]: {e}")
            metrics = {"sharpe": 0.0, "annual_return_pct": 0.0, "max_drawdown_pct": 0.0}

        window_results[key] = metrics
        weighted_sharpe_sum += metrics["sharpe"] * w["weight"]
        weight_sum += w["weight"]

    composite_score = weighted_sharpe_sum / weight_sum if weight_sum > 0 else 0.0

    return {
        "params": mrf_params,
        "window_results": window_results,
        "composite_score": composite_score,
    }


# ── 参数网格 ──────────────────────────────────────────────

def build_param_grid() -> List[Dict[str, Any]]:
    neutral_positions = [round(x, 2) for x in np.arange(0.50, 1.01, 0.10)]
    bear_positions    = [round(x, 2) for x in np.arange(0.20, 0.51, 0.10)]
    confirm_days_list = list(range(1, 4))
    persist_days_list = list(range(1, 4))

    combos = list(itertools.product(
        neutral_positions,
        bear_positions,
        confirm_days_list,
        persist_days_list,
    ))

    params_list = []
    for np_val, bp_val, cd_val, pd_val in combos:
        if bp_val > np_val:
            continue  # 跳过 bear > neutral 的非法组合
        params_list.append({
            "neutral_position":   np_val,
            "bear_position":       bp_val,
            "confirm_days":        cd_val,
            "regime_persist_days": pd_val,
        })
    return params_list


# ── Worker ────────────────────────────────────────────────

def _worker(params: Dict[str, Any]) -> Tuple[Dict, bool, str]:
    """单参数组合的全窗口评估 worker（供 multiprocessing 调用）。"""
    try:
        result = run_all_windows(
            strat_name=STRATEGY_NAME,
            windows=WINDOWS,
            initial_cash=INITIAL_CASH,
            mrf_params=params,
        )
        return result, True, ""
    except Exception as e:
        return {"params": params, "error": str(e)}, False, str(e)


# ── 主程序 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MarketRegimeFilter 参数扫描")
    parser.add_argument("--mode", choices=["serial", "parallel"], default="parallel",
                        help="运行模式（默认 parallel）")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="并行进程数（默认 CPU 核数）")
    args = parser.parse_args()

    t0 = time.time()
    params_list = build_param_grid()
    total = len(params_list)
    logger.info(f"参数组合总数 : {total}")
    logger.info(f"回测窗口数   : {len(WINDOWS)}")
    logger.info(f"预计总回测数 : {total * len(WINDOWS)} 次")

    # ── 基准（无过滤器 = NP=1.0, BP=1.0）─────────────────────
    logger.info("\n计算基准（无过滤器）...")
    baseline_params = {
        "neutral_position": 1.0,
        "bear_position": 1.0,
        "confirm_days": 1,
        "regime_persist_days": 1,
    }
    baseline = run_all_windows(STRATEGY_NAME, WINDOWS, INITIAL_CASH, baseline_params)
    baseline_score = baseline["composite_score"]
    logger.info(f"基准综合得分  : {baseline_score:.4f}")

    # ── 扫描 ────────────────────────────────────────────────
    if args.mode == "serial":
        raw_results: List[Dict] = []
        for i, p in enumerate(params_list):
            logger.info(f"[{i+1}/{total}] NP={p['neutral_position']:.2f} "
                        f"BP={p['bear_position']:.2f} CD={p['confirm_days']} PD={p['regime_persist_days']}")
            result, ok, err = _worker(p)
            if ok:
                raw_results.append(result)
                logger.info(f"  → score={result['composite_score']:.4f}")
            else:
                logger.error(f"  → 失败: {err}")
    else:
        n_workers = min(args.workers, total)
        logger.info(f"\n启动 {n_workers} 个并行进程扫描 {total} 组参数...")
        raw_results = []
        with mp.Pool(n_workers) as pool:
            completed = 0
            for res, ok, err in pool.imap_unordered(_worker, params_list, chunksize=4):
                completed += 1
                if ok:
                    raw_results.append(res)
                    p = res["params"]
                    logger.info(
                        f"[{completed:3d}/{total}] "
                        f"NP={p['neutral_position']:.2f} BP={p['bear_position']:.2f} "
                        f"CD={p['confirm_days']} PD={p['regime_persist_days']} "
                        f"→ score={res['composite_score']:.4f}"
                    )
                else:
                    logger.error(f"[{completed:3d}/{total}] 失败: {err}")

    elapsed = time.time() - t0
    logger.info(f"\n扫描完成！耗时: {elapsed/60:.1f} 分钟，共 {len(raw_results)}/{total} 组有效结果")

    if not raw_results:
        logger.error("没有有效的回测结果！")
        return

    # ── 排序 ────────────────────────────────────────────────
    raw_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top10 = raw_results[:10]
    best = raw_results[0]

    # ── Markdown 报告 ───────────────────────────────────────
    report = build_markdown_report(raw_results, top10, baseline, elapsed)
    out_path = PROJECT_ROOT / "SPEC_MarketRegimeFilter_PARAM_SCAN.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已写入: {out_path}")

    # ── JSON 详细结果 ───────────────────────────────────────
    json_path = PROJECT_ROOT / "SPEC_MarketRegimeFilter_PARAM_SCAN_results.json"
    serializable = []
    for r in raw_results:
        try:
            d = {
                "params": r["params"],
                "composite_score": round(r["composite_score"], 6),
                "windows": {
                    k: {
                        "sharpe": round(v["sharpe"], 4),
                        "annual_return_pct": round(v["annual_return_pct"], 2),
                        "max_drawdown_pct": round(v["max_drawdown_pct"], 2),
                        "n_trades": v["n_trades"],
                        "win_rate": round(v["win_rate"], 1),
                    }
                    for k, v in r["window_results"].items()
                }
            }
            serializable.append(d)
        except Exception:
            pass

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 结果已写入: {json_path}")

    # ── 打印最优结果 ────────────────────────────────────────
    p = best["params"]
    print("\n" + "=" * 60)
    print("  🏆 最优参数组合")
    print("=" * 60)
    print(f"  neutral_position    = {p['neutral_position']:.2f}")
    print(f"  bear_position      = {p['bear_position']:.2f}")
    print(f"  confirm_days       = {p['confirm_days']}")
    print(f"  regime_persist_days= {p['regime_persist_days']}")
    print(f"  综合得分           = {best['composite_score']:.4f}")
    print("-" * 60)
    print("  各窗口表现：")
    for w_name, w_label in [("A","牛市"),("B","震荡"),("C","震荡"),("D","熊市"),("E","熊市"),("F","熊市")]:
        m = best["window_results"].get(w_name, {})
        print(f"  {w_name} ({w_label}): 年化={m.get('annual_return_pct',0):+.2f}%, "
              f"MaxDD={m.get('max_drawdown_pct',0):.2f}%, 夏普={m.get('sharpe',0):.2f}")
    print("=" * 60)
    print(f"\n基准（无过滤器）得分 : {baseline_score:.4f}")
    print(f"最优相对基准提升     : {(best['composite_score']/baseline_score-1)*100:+.1f}%")


# ── 报告生成 ──────────────────────────────────────────────

def build_markdown_report(
    all_results: List[Dict],
    top10: List[Dict],
    baseline: Dict,
    elapsed_sec: float,
) -> str:
    total = len(all_results)
    best = top10[0]
    p = best["params"]

    # Top-10 表格行
    rows = []
    for rank, r in enumerate(top10, 1):
        pp = r["params"]
        anns = {k: v["annual_return_pct"] for k, v in r["window_results"].items()}
        dds  = {k: v["max_drawdown_pct"]  for k, v in r["window_results"].items()}
        rows.append(
            f"| {rank} | {pp['neutral_position']:.2f} | {pp['bear_position']:.2f} | "
            f"{pp['confirm_days']} | {pp['regime_persist_days']} | "
            f"{r['composite_score']:.4f} | "
            f"{anns.get('A',0):+.1f} | {anns.get('B',0):+.1f} | {anns.get('C',0):+.1f} | "
            f"{anns.get('D',0):+.1f} | {anns.get('E',0):+.1f} | {anns.get('F',0):+.1f} | "
            f"{dds.get('A',0):.1f} | {dds.get('D',0):.1f} | {dds.get('E',0):.1f} | {dds.get('F',0):.1f} |"
        )

    b_p   = baseline["params"]
    b_anns = {k: v["annual_return_pct"] for k, v in baseline["window_results"].items()}
    b_dds  = {k: v["max_drawdown_pct"]  for k, v in baseline["window_results"].items()}
    baseline_row = (
        f"| **基准** | {b_p['neutral_position']:.2f} | {b_p['bear_position']:.2f} | "
        f"{b_p['confirm_days']} | {b_p['regime_persist_days']} | "
        f"{baseline['composite_score']:.4f} | "
        f"{b_anns.get('A',0):+.1f} | {b_anns.get('B',0):+.1f} | {b_anns.get('C',0):+.1f} | "
        f"{b_anns.get('D',0):+.1f} | {b_anns.get('E',0):+.1f} | {b_anns.get('F',0):+.1f} | "
        f"{b_dds.get('A',0):.1f} | {b_dds.get('D',0):.1f} | {b_dds.get('E',0):.1f} | {b_dds.get('F',0):.1f} |"
    )

    # 敏感性分析
    def agg_effect(key: str) -> str:
        eff = {}
        for r in all_results:
            k = r["params"][key]
            if k not in eff:
                eff[k] = []
            eff[k].append(r["composite_score"])
        lines = []
        for k in sorted(eff.keys()):
            lines.append(f"| {k} | {np.mean(eff[k]):.4f} | {len(eff[k])} |")
        return "\n".join(lines)

    np_table = agg_effect("neutral_position")
    bp_table = agg_effect("bear_position")
    cd_table = agg_effect("confirm_days")
    pd_table = agg_effect("regime_persist_days")

    # 熊市改善
    bear_improvements = []
    for w_name, w_label in [("D","2015-16熊市"),("E","2018-19熊市"),("F","2010-12熊市")]:
        opt_mdd = best["window_results"].get(w_name, {}).get("max_drawdown_pct", 0)
        base_mdd = baseline["window_results"].get(w_name, {}).get("max_drawdown_pct", 0)
        delta = base_mdd - opt_mdd
        bear_improvements.append(
            f"- {w_label}：基准 MaxDD={base_mdd:.2f}% → 最优 MaxDD={opt_mdd:.2f}%（"
            f"{'改善' if delta > 0 else '劣化'} {abs(delta):.2f}个百分点）"
        )

    opt_ann_a = best["window_results"].get("A", {}).get("annual_return_pct", 0)
    base_ann_a = baseline["window_results"].get("A", {}).get("annual_return_pct", 0)
    ann_a_delta = opt_ann_a - base_ann_a
    bull_grade = ("✅ 牛市表现未明显劣化" if ann_a_delta > -5
                  else "⚠️ 牛市年化下降超过5%，需注意" if ann_a_delta > -10
                  else "🔴 牛市严重劣化，建议重新评估")

    report = f"""# MarketRegimeFilter 参数敏感性扫描报告

> 生成时间：2026-04-02
> 扫描范围：neutral_position × bear_position × confirm_days × regime_persist_days
> 总组合数：**{total}** 组 × **{len(WINDOWS)}** 窗口 = **{total * len(WINDOWS)}** 次回测
> 总耗时：{elapsed_sec/60:.1f} 分钟

---

## 1. 最优参数组合

```
neutral_position    = {p['neutral_position']:.2f}
bear_position       = {p['bear_position']:.2f}
confirm_days        = {p['confirm_days']}
regime_persist_days = {p['regime_persist_days']}
综合得分            = {best['composite_score']:.4f}
```

**相比基准（无过滤器）提升：{(best['composite_score']/baseline['composite_score']-1)*100:+.1f}%**

### 各窗口表现

| 窗口 | 类型 | 年化收益 | 最大回撤 | 夏普比率 | 权重 |
|------|------|---------|---------|---------|------|
| A | 牛市（2024） | {best['window_results'].get('A',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('A',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('A',{}).get('sharpe',0):.2f} | 1 |
| B | 震荡（2022-23） | {best['window_results'].get('B',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('B',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('B',{}).get('sharpe',0):.2f} | 2 |
| C | 震荡（2020-21） | {best['window_results'].get('C',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('C',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('C',{}).get('sharpe',0):.2f} | 2 |
| D | 熊市（2015-16） | {best['window_results'].get('D',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('D',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('D',{}).get('sharpe',0):.2f} | 3 |
| E | 熊市（2018-19） | {best['window_results'].get('E',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('E',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('E',{}).get('sharpe',0):.2f} | 3 |
| F | 熊市（2010-12） | {best['window_results'].get('F',{}).get('annual_return_pct',0):+.2f}% | {best['window_results'].get('F',{}).get('max_drawdown_pct',0):.2f}% | {best['window_results'].get('F',{}).get('sharpe',0):.2f} | 3 |

---

## 2. Top-10 参数组合

| 排名 | NP | BP | CD | PD | 综合得分 | A年化 | B年化 | C年化 | D年化 | E年化 | F年化 | A_MaxDD | D_MaxDD | E_MaxDD | F_MaxDD |
|------|----|----|----|----|---------|-------|-------|-------|-------|-------|-------|---------|---------|---------|---------|
{chr(10).join(rows)}
{baseline_row}

> 基准 = 无过滤器（neutral=1.0, bear=1.0, CD=1, PD=1）

---

## 3. 评分方法

```
综合得分 = Σ(年化夏普 × 权重) / Σ权重

权重分配：
  - 熊市窗口（F/E/D）：权重 3（更看重 MaxDD 改善）
  - 震荡窗口（B/C）  ：权重 2
  - 牛市窗口（A）    ：权重 1
```

---

## 4. 参数敏感性分析

### 4.1 neutral_position 影响

| neutral_position | 平均得分 | 样本数 |
|-----------------|---------|-------|
{np_table}

### 4.2 bear_position 影响

| bear_position | 平均得分 | 样本数 |
|--------------|---------|-------|
{bp_table}

### 4.3 confirm_days 影响

| confirm_days | 平均得分 | 样本数 |
|-------------|---------|-------|
{cd_table}

### 4.4 regime_persist_days 影响

| regime_persist_days | 平均得分 | 样本数 |
|--------------------|---------|-------|
{pd_table}

---

## 5. 关键发现

### 5.1 熊市保护效果

最优参数 vs 基准（MaxDD 越小越好）：
{chr(10).join(bear_improvements)}

### 5.2 牛市不劣化

- A 窗口（牛市2024）：基准年化={base_ann_a:+.2f}% → 最优年化={opt_ann_a:+.2f}%（{'提升' if ann_a_delta > 0 else '下降'} {abs(ann_a_delta):.2f}个百分点）
- 评估：{bull_grade}

### 5.3 NEUTRAL vs BEAR 区分度

最优参数（NP={p['neutral_position']:.2f}, BP={p['bear_position']:.2f}）：
- NEUTRAL（震荡）仓位上限：{p['neutral_position']*100:.0f}%
- BEAR（熊市）仓位上限：{p['bear_position']*100:.0f}%
- 仓位区分度：{(p['neutral_position']-p['bear_position'])*100:.0f}个百分点

---

## 6. 完整结果排名（前30）

| 排名 | NP | BP | CD | PD | 综合得分 | A年化 | D年化 | E年化 | F年化 |
|------|----|----|----|----|---------|-------|-------|-------|-------|
"""

    for rank, r in enumerate(all_results[:30], 1):
        pp = r["params"]
        anns = {k: v["annual_return_pct"] for k, v in r["window_results"].items()}
        report += (
            f"| {rank} | {pp['neutral_position']:.2f} | {pp['bear_position']:.2f} | "
            f"{pp['confirm_days']} | {pp['regime_persist_days']} | "
            f"{r['composite_score']:.4f} | "
            f"{anns.get('A',0):+.1f} | {anns.get('D',0):+.1f} | "
            f"{anns.get('E',0):+.1f} | {anns.get('F',0):+.1f} |\n"
        )

    report += f"""

---

## 7. 建议

基于以上分析，**最优参数组合**为：
- `neutral_position = {p['neutral_position']:.2f}`
- `bear_position = {p['bear_position']:.2f}`
- `confirm_days = {p['confirm_days']}`
- `regime_persist_days = {p['regime_persist_days']}`

综合得分：{best['composite_score']:.4f}（基准：{baseline['composite_score']:.4f}，提升 {(best['composite_score']/baseline['composite_score']-1)*100:+.1f}%）

**注意**：回测结果仅供参考，实盘需考虑滑点、流动性、交易费用等摩擦因素。

---
*报告由 Oracle 研究员自动生成 | 2026-04-02*
"""
    return report


if __name__ == "__main__":
    main()
