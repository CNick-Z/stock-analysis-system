#!/usr/bin/env python3
"""
tuner/parameter_tuner.py
========================
V8 策略批量参数调优器

设计原则
─────────
1. 单次最多加载 2 年数据（当前年 + 前一年 warm-up）
2. 每年单独 prepare → 回测 → 释放
3. 不一次加载全部历史

算法流程
─────────
对每个 2 年滑动窗口 (Y, Y+1):
    1. 加载 [Y, Y+1] 两年数据（统一一次加载，保证 shift/rolling 正确）
    2. 对每个参数组合:
         a. 实例化 V8 策略（注入参数）
         b. prepare(Y 年 dates)  → 缓存 Y 年候选
         c. 回测 Y 年（逐日 _on_day）
         d. prepare(Y+1 年 dates) → 缓存 Y+1 年候选
         e. 回测 Y+1 年（逐日 _on_day，现金/持仓跨年保持）
         f. 记录指标
    3. 跨窗口平均各参数组合表现

评分函数
─────────
composite = 年化收益率 × 0.4 + 胜率 × 0.3 + 夏普比率 × 0.3

用法
────
    from tuner import ParameterTuner
    from strategies.score.v8.strategy import ScoreV8Strategy

    grid = {
        'stop_loss':       [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        'take_profit':     [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        'max_positions':   [3, 4, 5, 6],
        'position_size':   [0.15, 0.20, 0.25],
        'rsi_filter_min':  [40, 45, 50],
        'rsi_filter_max':  [55, 60, 65],
    }
    tuner = ParameterTuner(ScoreV8Strategy, grid)
    best = tuner.tune(start_year=2020, end_year=2025)
    top10 = tuner.get_top_n(10)
"""

import gc
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework
from simulator.shared import add_next_open
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 结果分析（与 result_analyzer.py 共用）
# =============================================================================

def compute_metrics(framework: BaseFramework) -> Dict[str, Any]:
    """
    从 BaseFramework 运行结果计算绩效指标

    Returns:
        dict: {
            annual_return_pct,  # 年化收益率 %
            win_rate,            # 胜率 %
            sharpe,              # 夏普比率
            max_drawdown_pct,    # 最大回撤 %
            total_return_pct,    # 总收益率 %
            n_trades,            # 交易笔数
            final_value,         # 最终资产
        }
    """
    closed = [t for t in framework.trades if t["action"] == "sell"]
    total_pnl = sum(t["pnl"] for t in closed)
    n_winning = sum(1 for t in closed if t["pnl"] > 0)
    n_total = len(closed)
    win_rate = n_winning / max(n_total, 1) * 100

    final_value = framework.cash + sum(
        pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        for pos in framework.positions.values()
    )
    total_return = (final_value / framework.initial_cash - 1) * 100

    # 年化收益（按交易日天数估算）
    n_days = len(framework.market_snapshots)
    n_years = n_days / 244
    annual_return = (final_value / framework.initial_cash) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 最大回撤
    values = [s["total_value"] for s in framework.market_snapshots]
    peak = framework.initial_cash
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # 夏普比率（年化，日收益率标准差）
    if len(values) > 1:
        daily_returns = pd.Series(values).pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * (244 ** 0.5) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "annual_return_pct": annual_return * 100,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_return_pct": total_return,
        "n_trades": n_total,
        "final_value": final_value,
    }


# =============================================================================
# 参数化策略包装器
# =============================================================================

def _make_parametric_strategy(strategy_class, params: Dict[str, Any]):
    """
    为指定策略类创建参数化实例。

    核心机制：
      V8.prepare() 中有一行硬编码 RSI 过滤：
        df['_rsi_f'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
      我们把它替换成参数化版本，通过闭包捕获目标 rsi_min / rsi_max。

    步骤：
      1. 保存 V8 原始 prepare 方法
      2. 创建补丁函数：先用参数化值写入 df['_rsi_f']，再调用原始 prepare
         （闭包在定义时捕获当前 combo 的 rsi_min / rsi_max）
      3. 策略实例的 prepare 方法替换为补丁
      4. 后续调 prepare() → 补丁 → 写 _rsi_f → 原始 prepare（用新的 _rsi_f）

    Args:
        strategy_class: 原始策略类（ScoreV8Strategy）
        params: 参数字典（包含 rsi_filter_min / rsi_filter_max）

    Returns:
        注入了目标 RSI 参数的策略实例
    """
    # 构造完整 cfg（含 RSI 参数，V8.prepare() 已支持参数化）
    cfg = {k: v for k, v in params.items()
           if k in ("stop_loss", "take_profit", "max_positions",
                    "position_size", "rsi_filter_min", "rsi_filter_max")}

    # 创建策略实例（prepare() 直接读 cfg 中的 RSI 参数，无需补丁）
    instance = strategy_class(cfg)
    return instance


# =============================================================================
# ParameterTuner
# =============================================================================

class ParameterTuner:
    """
    策略参数批量调优器

    Args:
        strategy_class: 策略类（必须是 ScoreV8Strategy 或兼容接口）
        param_grid: 参数名 → 候选值列表

    Attributes:
        results: List[Dict]  所有 (窗口, 参数组合) 的原始回测结果
        summary: List[Dict]  每个参数组合跨窗口平均后的汇总
    """

    DEFAULT_GRID: Dict[str, List] = {
        "stop_loss":      [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        "take_profit":    [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        "max_positions":  [3, 4, 5, 6],
        "position_size":  [0.15, 0.20, 0.25],
        "rsi_filter_min": [40, 45, 50],
        "rsi_filter_max": [55, 60, 65],
    }

    def __init__(
        self,
        strategy_class,
        param_grid: Optional[Dict[str, List]] = None,
        initial_cash: float = 500_000,
        output_path: str = "/tmp/tuner_results.json",
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid or self.DEFAULT_GRID
        self.initial_cash = initial_cash
        self.output_path = Path(output_path)

        # 穷举全部组合
        self._param_names = list(self.param_grid.keys())
        self._param_values = [self.param_grid[k] for k in self._param_names]
        self._combinations: List[Dict[str, Any]] = []
        for combo in itertools.product(*self._param_values):
            self._combinations.append(dict(zip(self._param_names, combo)))

        self.results: List[Dict] = []     # 原始结果
        self.summary: List[Dict] = []       # 跨窗口平均后的汇总

        logger.info(
            f"ParameterTuner 初始化: {len(self._combinations):,} 组参数 "
            f"({', '.join(self._param_names)})"
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def tune(
        self,
        start_year: int,
        end_year: int,
        window_years: int = 2,
    ) -> Dict[str, Any]:
        """
        执行全部窗口 × 全部参数组合的网格搜索

        Args:
            start_year: 回测起始年（含）
            end_year:   回测结束年（含）
            window_years: 滑动窗口大小（默认 2）

        Returns:
            最佳参数组合（跨窗口平均后 composite score 最高）
        """
        if start_year > end_year:
            raise ValueError(f"start_year({start_year}) > end_year({end_year})")

        all_years = list(range(start_year, end_year + 1))
        n_windows = max(0, len(all_years) - window_years + 1)

        # 单年回测（start_year == end_year）：视为 1 个 1 年窗口，
        # load_strategy_data([y_start, y_end]) = [2024]（不翻倍）
        if start_year == end_year:
            n_windows = 1
            windows = [(start_year, start_year)]
        elif n_windows == 0:
            logger.warning(
                f"年份范围不足 {window_years} 年，扩大至 [{start_year}, {end_year}]"
            )
            n_windows = 1
            windows = [(start_year, end_year)]
        else:
            windows = [
                (all_years[i], all_years[i + window_years - 1])
                for i in range(n_windows)
            ]

        logger.info(
            f"滑动窗口: {n_windows} 个, 年份: {windows}"
        )
        logger.info(f"参数组合总数: {len(self._combinations):,} 组 × {n_windows} 窗口 "
                    f"= {len(self._combinations) * n_windows:,} 次回测")

        total_runs = len(self._combinations) * n_windows
        run_count = 0

        for w_idx, (y_start, y_end) in enumerate(windows):
            logger.info(f"\n{'='*60}")
            logger.info(f"  窗口 {w_idx+1}/{n_windows}: [{y_start}, {y_end}]")
            logger.info(f"{'='*60}")

            # ══════════════════════════════════════════════════════════════
            # 外层窗口循环：每个窗口只加载 1 次数据，所有参数组合共用
            # 数据加载 次数 = 窗口数（不是 窗口数 × 参数组合数）
            # ══════════════════════════════════════════════════════════════
            # ── 1. 加载两年数据（全量，对所有参数组合复用）──
            try:
                df = load_strategy_data(years=[y_start, y_end], add_money_flow=True)
            except Exception as e:
                logger.error(f"  数据加载失败 [{y_start}-{y_end}]: {e}")
                continue

            if df.empty:
                logger.error(f"  数据为空 [{y_start}-{y_end}]")
                continue

            df = add_next_open(df)
            all_dates_in_window = sorted(df["date"].unique())
            logger.info(
                f"  窗口数据: {len(df):,} 行, {df['symbol'].nunique()} 只股票, "
                f"{all_dates_in_window[0]} ~ {all_dates_in_window[-1]}"
            )

            # 分割各年 dates（用于分年 prepare）
            dates_by_year: Dict[int, List[str]] = {}
            for y in range(y_start, y_end + 1):
                dates_by_year[y] = sorted([
                    d for d in all_dates_in_window
                    if str(d).startswith(str(y))
                ])

            # ── 2. 参数组合循环（复用上方的 df，不重新加载）─────────────
            #    同一窗口内所有 combo 共用一份 df，del df 在窗口末尾（见 line ~515）
            for combo_idx, params in enumerate(self._combinations):
                run_count += 1

                # 进度日志（每 100 组）
                combo_key = json.dumps(params, sort_keys=True)
                if combo_idx == 0 or (run_count % 100 == 0) or combo_idx == len(self._combinations) - 1:
                    logger.info(
                        f"  进度 {run_count}/{total_runs} ({run_count/total_runs*100:.1f}%)  "
                        f"窗口{w_idx+1}/{n_windows}  "
                        f"参数[{combo_key[:80]}]"
                    )

                # 每组参数独立实例 + 独立框架
                strategy = _make_parametric_strategy(self.strategy_class, params)
                framework = self._create_framework(params, strategy)

                # ── prepare（一次性prepare所有年份，避免逐年被覆盖）──────
                all_year_dates = []
                for year in range(y_start, y_end + 1):
                    all_year_dates.extend(dates_by_year.get(year, []))
                if all_year_dates and hasattr(strategy, "prepare"):
                    try:
                        strategy.prepare(sorted(all_year_dates), df)
                    except Exception as prep_err:
                        logger.warning(f"  prepare() 失败: {prep_err}")

                # ── 逐年回测（使用本组合 prepare 后的缓存）─────────────
                for year in range(y_start, y_end + 1):
                    year_dates = dates_by_year.get(year, [])
                    if not year_dates:
                        continue

                    # 逐日回测（直接调用 _on_day，保持现金/持仓跨年连续）
                    for day_idx, date in enumerate(year_dates):
                        framework._on_day(date, df, year_dates)

                        # 每日快照
                        total_value = framework._calc_total_value(df, date)
                        snap = dict(
                            date=date,
                            cash=framework.cash,
                            total_value=total_value,
                            n_positions=len(framework.positions),
                            total_return=(total_value / framework.initial_cash - 1) * 100,
                        )
                        framework.market_snapshots.append(snap)

                    # 当年结束：保持现金/持仓跨年，
                    # trades / n_winning / n_total / market_snapshots
                    # 均保留至整个窗口结束（compute_metrics 需要完整数据）

                    logger.debug(f"  {year} 年完成: 持仓 {len(framework.positions)} "
                                 f"现金 {framework.cash:,.0f}")

                # ── 3. 计算窗口内指标 ──────────────────────────────────
                metrics = compute_metrics(framework)

                record = {
                    "window": (y_start, y_end),
                    "params": params,
                    **metrics,
                }
                self.results.append(record)

                # 清理
                del framework, strategy
                gc.collect()

            # 窗口结束，释放当年数据
            del df
            gc.collect()

        # ── 4. 跨窗口汇总 ─────────────────────────────────────────────
        self._summarize()

        # 保存到文件
        self._save()

        best = self.summary[0] if self.summary else {}
        logger.info(f"\n调优完成！最佳参数: {best.get('params', 'N/A')}")
        logger.info(f"  composite = {best.get('composite_score', 0):.4f}")
        logger.info(f"  年化收益  = {best.get('avg_annual_return_pct', 0):.2f}%")
        logger.info(f"  胜率      = {best.get('avg_win_rate', 0):.2f}%")
        logger.info(f"  夏普比率  = {best.get('avg_sharpe', 0):.4f}")
        logger.info(f"结果已保存: {self.output_path}")

        return best

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _create_framework(self, params: Dict[str, Any], strategy) -> BaseFramework:
        """根据参数创建回测框架"""
        fw = BaseFramework(
            initial_cash=self.initial_cash,
            max_positions=params.get("max_positions", 5),
            position_size=params.get("position_size", 0.20),
        )
        fw._strategy = strategy
        return fw

    def _summarize(self):
        """
        跨窗口平均：将 self.results 聚合成 self.summary

        对每个参数组合，收集它在不同窗口的指标，取平均值，
        然后按 composite = ann_ret×0.4 + win_rate×0.3 + sharpe×0.3 排序。
        """
        # 按参数字符串分组
        from collections import defaultdict

        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.results:
            key = json.dumps(r["params"], sort_keys=True)
            grouped[key].append(r)

        records = []
        for key, runs in grouped.items():
            params = runs[0]["params"]

            avg_ann_ret = np.mean([r["annual_return_pct"] for r in runs])
            avg_win_rate = np.mean([r["win_rate"] for r in runs])
            avg_sharpe = np.mean([r["sharpe"] for r in runs])
            avg_max_dd = np.mean([r["max_drawdown_pct"] for r in runs])
            avg_n_trades = int(np.mean([r["n_trades"] for r in runs]))

            # 综合评分（归一化到同一量级）
            composite = (
                avg_ann_ret   * 0.4  +   # 年化收益 × 0.4
                avg_win_rate  * 0.3  +   # 胜率 × 0.3（已是 %）
                avg_sharpe    * 0.3       # 夏普 × 0.3
            )

            records.append({
                "params": params,
                "avg_annual_return_pct": avg_ann_ret,
                "avg_win_rate": avg_win_rate,
                "avg_sharpe": avg_sharpe,
                "avg_max_drawdown_pct": avg_max_dd,
                "avg_n_trades": avg_n_trades,
                "composite_score": composite,
                "n_windows": len(runs),
            })

        # 降序排列
        records.sort(key=lambda x: x["composite_score"], reverse=True)
        self.summary = records

    def _save(self):
        """保存结果到 JSON 文件"""
        payload = {
            "param_grid": self.param_grid,
            "results": self.results,
            "summary": self.summary,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_top_n(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        返回跨窗口平均后综合得分最高的前 N 组参数

        Args:
            n: 返回数量（默认 10）

        Returns:
            List[Dict]，每项包含 params + avg_annual_return_pct / avg_win_rate /
            avg_sharpe / composite_score 等
        """
        if not self.summary:
            logger.warning("调优结果为空，请先调用 tune()")
            return []

        # 打印前 N 名
        header = (
            f"\n{'='*90}\n"
            f"  参数调优结果 — Top {min(n, len(self.summary))}\n"
            f"{'='*90}\n"
            f"  {'#':<3}  {'composite':>9}  {'年化收益':>9}  {'胜率':>7}  {'夏普':>7}  "
            f"{'最大回撤':>9}  {'参数'}\n"
            f"  {'-'*88}"
        )
        logger.info(header)

        lines = []
        for i, rec in enumerate(self.summary[:n], 1):
            p = rec["params"]
            param_str = (
                f"sl={p['stop_loss']}, tp={p['take_profit']}, "
                f"maxP={p['max_positions']}, ps={p['position_size']}, "
                f"rsi=[{p['rsi_filter_min']},{p['rsi_filter_max']}]"
            )
            line = (
                f"  {i:<3}  {rec['composite_score']:>9.4f}  "
                f"{rec['avg_annual_return_pct']:>+8.2f}%  "
                f"{rec['avg_win_rate']:>6.1f}%  "
                f"{rec['avg_sharpe']:>7.2f}  "
                f"{rec['avg_max_drawdown_pct']:>8.2f}%  "
                f"{param_str}"
            )
            logger.info(line)
            lines.append(rec)

        logger.info(f"{'='*90}")
        return lines

    def load(self, path: Optional[str] = None):
        """
        从 JSON 文件加载历史调优结果（不需要重新跑 tune()）

        Args:
            path: 结果文件路径，默认为构造时指定的 output_path
        """
        p = Path(path) if path else self.output_path
        if not p.exists():
            raise FileNotFoundError(f"结果文件不存在: {p}")

        with open(p, encoding="utf-8") as f:
            payload = json.load(f)

        self.param_grid = payload["param_grid"]
        self.results = payload["results"]
        self._summarize()   # 重新聚合并排序
        logger.info(f"已加载 {len(self.results)} 条历史记录，"
                    f" {len(self.summary)} 个参数组合")


# =============================================================================
# 快速入口（python -m tuner.parameter_tuner 直接运行）
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V8 策略参数调优")
    parser.add_argument("--start", type=int, default=2020, help="起始年份（含）")
    parser.add_argument("--end", type=int, default=2025, help="结束年份（含）")
    parser.add_argument("--cash", type=float, default=500_000, help="初始资金")
    parser.add_argument("--top", type=int, default=10, help="输出前 N 名")
    parser.add_argument(
        "--output", type=str, default="/tmp/tuner_results.json", help="结果文件路径"
    )
    args = parser.parse_args()

    # 延迟导入（避免顶层 import 拖慢 help）
    from strategies.score.v8.strategy import ScoreV8Strategy

    tuner = ParameterTuner(
        strategy_class=ScoreV8Strategy,
        param_grid=None,          # 使用默认网格
        initial_cash=args.cash,
        output_path=args.output,
    )

    tuner.tune(start_year=args.start, end_year=args.end)
    tuner.get_top_n(n=args.top)
