#!/usr/bin/env python3
"""
tuner/result_analyzer.py
========================
参数调优结果分析工具

功能
────
1. 加载历史调优结果（JSON）
2. 按单参数分组统计（其他参数固定时，该参数的变化对指标的影响）
3. 热力图 / 折线图可视化（文本表格）
4. 敏感性分析：哪些参数对 composite score 影响最大

用法
────
    from tuner import ResultAnalyzer
    analyzer = ResultAnalyzer("/tmp/tuner_results.json")
    analyzer.param_sensitivity()     # 打印参数敏感性排名
    analyzer.slice_by("rsi_filter_min")  # 固定其他参数，RSI下限对指标的影响
    analyzer.top_by("sharpe", n=5)   # 夏普最高的 Top-5
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 指标定义
# =============================================================================

METRIC_NAMES = [
    "annual_return_pct",
    "win_rate",
    "sharpe",
    "max_drawdown_pct",
    "composite_score",
]


# =============================================================================
# ResultAnalyzer
# =============================================================================

class ResultAnalyzer:
    """
    调优结果分析器

    Args:
        result_path: /tmp/tuner_results.json 路径
    """

    METRIC_LABELS = {
        "annual_return_pct":  "年化收益%",
        "win_rate":           "胜率%",
        "sharpe":             "夏普",
        "max_drawdown_pct":   "最大回撤%",
        "composite_score":    "综合评分",
        "avg_n_trades":       "平均交易次数",
    }

    def __init__(self, result_path: str):
        self.path = Path(result_path)
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"文件不存在: {self.path}")

        with open(self.path, encoding="utf-8") as f:
            payload = json.load(f)

        self.param_grid: Dict[str, List] = payload["param_grid"]
        self.results: List[Dict] = payload["results"]
        self.summary: List[Dict] = payload["summary"]

        logger.info(
            f"加载结果: {len(self.results)} 条记录, "
            f"{len(self.summary)} 个参数组合, "
            f"{len(self.param_grid)} 个参数"
        )

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------

    def top_by(
        self,
        metric: str = "composite_score",
        n: int = 10,
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        按指定指标排序返回 Top-N

        Args:
            metric: annual_return_pct / win_rate / sharpe /
                    max_drawdown_pct / composite_score
            n: 返回数量
            ascending: True 则返回最差的 N 个
        """
        if metric not in METRIC_NAMES and metric not in ["avg_n_trades"]:
            raise ValueError(f"未知指标: {metric}")

        sorted_summary = sorted(
            self.summary,
            key=lambda x: x.get(f"avg_{metric}", x.get(metric, 0)),
            reverse=not ascending,
        )

        label = self.METRIC_LABELS.get(metric, metric)
        direction = "最高" if not ascending else "最低"
        logger.info(f"\n{'='*80}")
        logger.info(f"  {direction} {n} 个参数组合（按 {label} 排序）")
        logger.info(f"{'='*80}")

        header = (
            f"  {'#':<3}  {label:>12}  年化收益  胜率  夏普  最大回撤  参数摘要"
        )
        logger.info(header)
        logger.info(f"  {'-'*75}")

        for i, rec in enumerate(sorted_summary[:n], 1):
            val = rec.get(f"avg_{metric}", rec.get(metric, 0))
            p = rec["params"]
            logger.info(
                f"  {i:<3}  {val:>12.4f}  "
                f"{rec['avg_annual_return_pct']:>+7.2f}%  "
                f"{rec['avg_win_rate']:>5.1f}%  "
                f"{rec['avg_sharpe']:>5.2f}  "
                f"{rec['avg_max_drawdown_pct']:>8.2f}%  "
                f"sl={p['stop_loss']} tp={p['take_profit']} "
                f"maxP={p['max_positions']} ps={p['position_size']} "
                f"rsi=[{p['rsi_filter_min']},{p['rsi_filter_max']}]"
            )
        logger.info(f"{'='*80}")

        return sorted_summary[:n]

    def get_best(self, metric: str = "composite_score") -> Dict[str, Any]:
        """返回单一指标最优的参数组合"""
        return self.top_by(metric=metric, n=1)[0]

    # ------------------------------------------------------------------
    # 敏感性分析
    # ------------------------------------------------------------------

    def param_sensitivity(
        self,
        metric: str = "composite_score",
        n_std: float = 3.0,
    ) -> Dict[str, Dict]:
        """
        参数敏感性分析：固定其他参数，观察每个参数的变化对指标的影响。

        原理：对于每个参数 p，收集所有包含 p 的参数组合，
        计算 p 的每个取值对应的指标均值，然后计算：
          - 指标range（最大值 - 最小值）
          - 指标 std
        range / std 越大，说明该参数越敏感。

        Returns:
            Dict[参数名 → {range, std, values: {值 → 均值}}]
        """
        metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric

        sensitivity: Dict[str, Dict] = {}

        for param_name, param_values in self.param_grid.items():
            # 按该参数的不同取值分组
            buckets: Dict[Any, List[float]] = {v: [] for v in param_values}

            for rec in self.summary:
                p = rec["params"]
                if param_name in p:
                    val = p[param_name]
                    m = rec.get(metric_key, 0)
                    if val in buckets:
                        buckets[val].append(m)

            # 汇总
            avg_by_value = {
                str(k): np.mean(v) if v else 0.0
                for k, v in buckets.items()
            }
            all_vals = list(avg_by_value.values())
            rng = max(all_vals) - min(all_vals) if all_vals else 0.0
            std = np.std(all_vals) if all_vals else 0.0

            sensitivity[param_name] = {
                "range": rng,
                "std": std,
                "avg_by_value": avg_by_value,
                "impact_score": rng / max(std, 0.001),  # 信噪比
            }

        # 按 impact_score 降序打印
        sorted_params = sorted(
            sensitivity.items(),
            key=lambda x: x[1]["impact_score"],
            reverse=True,
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"  参数敏感性分析（指标: {metric}）")
        logger.info(f"{'='*70}")
        logger.info(
            f"  {'参数':<20}  {'range':>10}  {'std':>10}  {'影响系数':>10}  取值→均值"
        )
        logger.info(f"  {'-'*68}")
        for name, info in sorted_params:
            vals_str = "  ".join(
                f"{k}→{v:.3f}" for k, v in info["avg_by_value"].items()
            )
            logger.info(
                f"  {name:<20}  {info['range']:>10.4f}  "
                f"{info['std']:>10.4f}  {info['impact_score']:>10.2f}  {vals_str}"
            )
        logger.info(f"{'='*70}")

        return sensitivity

    # ------------------------------------------------------------------
    # 单参数切片分析
    # ------------------------------------------------------------------

    def slice_by(
        self,
        param: str,
        metric: str = "composite_score",
        fix_other: bool = True,
    ) -> Dict[str, List]:
        """
        固定其他参数（取众数），只变化指定参数，观察指标变化。

        Args:
            param: 要分析的参数名
            metric: 评估指标
            fix_other: True 则固定其他参数为众数值；False 则展示所有组合的均值

        Returns:
            Dict[参数取值 → [指标值列表]]
        """
        metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric
        param_values = self.param_grid.get(param, [])

        if fix_other:
            # 找其他参数的众数
            fixed = {}
            other_params = [p for p in self.param_grid if p != param]
            for op in other_params:
                vals = [r["params"][op] for r in self.summary if op in r["params"]]
                if vals:
                    from collections import Counter
                    fixed[op] = Counter(vals).most_common(1)[0][0]

            # 按 param 取值分组
            buckets: Dict[Any, List[float]] = {v: [] for v in param_values}
            for rec in self.summary:
                p = rec["params"]
                # 检查是否等于众数（浮动误差忽略）
                if all(p.get(op) == fixed.get(op) for op in other_params):
                    if param in p and p[param] in buckets:
                        buckets[p[param]].append(rec.get(metric_key, 0))

        else:
            # 简单平均：所有组合中，该参数取每个值的指标均值
            buckets = {v: [] for v in param_values}
            for rec in self.summary:
                p = rec["params"]
                if param in p and p[param] in buckets:
                    buckets[p[param]].append(rec.get(metric_key, 0))

        # 打印表格
        label = self.METRIC_LABELS.get(metric, metric)
        logger.info(f"\n{'='*60}")
        logger.info(f"  参数切片分析: {param}（指标: {label}）")
        if fix_other:
            other_fixed = {k: v for k, v in fixed.items()}
            logger.info(f"  其他参数固定于: {other_fixed}")
        logger.info(f"{'='*60}")
        logger.info(
            f"  {'取值':>10}  {'样本数':>6}  {'均值':>12}  {'标准差':>10}  {'最小':>10}  {'最大':>10}"
        )
        logger.info(f"  {'-'*60}")

        result = {}
        for val in param_values:
            vals = buckets.get(val, [])
            if vals:
                result[str(val)] = {
                    "n": len(vals),
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                }
                logger.info(
                    f"  {str(val):>10}  {len(vals):>6}  "
                    f"{np.mean(vals):>12.4f}  "
                    f"{np.std(vals):>10.4f}  "
                    f"{np.min(vals):>10.4f}  "
                    f"{np.max(vals):>10.4f}"
                )
            else:
                result[str(val)] = {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
                logger.info(f"  {str(val):>10}  {'—':>6}  {'—':>12}  {'—':>10}  {'—':>10}  {'—':>10}")

        logger.info(f"{'='*60}")
        return result

    # ------------------------------------------------------------------
    # 两参数交叉分析（热力图文本版）
    # ------------------------------------------------------------------

    def cross_analysis(
        self,
        param_x: str,
        param_y: str,
        metric: str = "composite_score",
    ) -> Dict[str, Any]:
        """
        两参数交叉分析（文本热力图）

        打印一个矩阵：X 轴为 param_x 取值，Y 轴为 param_y 取值，
        单元格为指标均值。

        Args:
            param_x: X 轴参数
            param_y: Y 轴参数
            metric: 指标

        Returns:
            {"matrix": [[均值, ...], ...], "x_values": [...], "y_values": [...]}
        """
        metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric

        x_vals = self.param_grid.get(param_x, [])
        y_vals = self.param_grid.get(param_y, [])

        # 初始化矩阵
        matrix: List[List[float]] = []
        for _ in y_vals:
            matrix.append([0.0] * len(x_vals))

        # 填充
        for rec in self.summary:
            p = rec["params"]
            xv = p.get(param_x)
            yv = p.get(param_y)
            if xv in x_vals and yv in y_vals:
                col = x_vals.index(xv)
                row = y_vals.index(yv)
                matrix[row][col] = rec.get(metric_key, 0)

        # 打印
        label = self.METRIC_LABELS.get(metric, metric)
        logger.info(f"\n{'='*70}")
        logger.info(f"  两参数交叉分析: {param_x} × {param_y}（指标: {label}）")
        logger.info(f"{'='*70}")

        # 表头
        header = f"  {param_y + '/' + param_x:^20}"
        for xv in x_vals:
            header += f"  {str(xv):^8}"
        logger.info(header)
        logger.info(f"  {'-'*70}")

        for row_idx, yv in enumerate(y_vals):
            row_str = f"  {str(yv):^20}"
            for col_idx in range(len(x_vals)):
                cell = matrix[row_idx][col_idx]
                row_str += f"  {cell:>8.3f}"
            logger.info(row_str)

        logger.info(f"{'='*70}")

        return {"matrix": matrix, "x_values": list(x_vals), "y_values": list(y_vals)}

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_top_csv(self, path: str, n: int = 20):
        """
        将 Top-N 参数组合导出为 CSV（方便在 Excel 中进一步分析）
        """
        import csv

        rows = []
        for rec in self.summary[:n]:
            row = {**rec["params"]}
            row["avg_annual_return_pct"] = rec["avg_annual_return_pct"]
            row["avg_win_rate"] = rec["avg_win_rate"]
            row["avg_sharpe"] = rec["avg_sharpe"]
            row["avg_max_drawdown_pct"] = rec["avg_max_drawdown_pct"]
            row["composite_score"] = rec["composite_score"]
            row["n_windows"] = rec["n_windows"]
            rows.append(row)

        fieldnames = list(self.param_grid.keys()) + [
            "avg_annual_return_pct",
            "avg_win_rate",
            "avg_sharpe",
            "avg_max_drawdown_pct",
            "composite_score",
            "n_windows",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"已导出 Top-{n} 到: {path}")


# =============================================================================
# 快速入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调优结果分析")
    parser.add_argument("result_json", help="调优结果 JSON 文件路径")
    parser.add_argument("--top", type=int, default=10, help="Top-N")
    parser.add_argument("--sensitivity", action="store_true", help="参数敏感性分析")
    parser.add_argument("--slice", dest="slice_param", type=str, help="单参数切片分析")
    parser.add_argument("--cross", nargs=2, dest="cross_params", type=str,
                        metavar=("PARAM_X", "PARAM_Y"), help="两参数交叉分析")
    parser.add_argument("--metric", type=str, default="composite_score",
                        choices=METRIC_NAMES, help="分析所使用的指标")
    parser.add_argument("--export-csv", type=str, help="导出 CSV 路径")
    args = parser.parse_args()

    analyzer = ResultAnalyzer(args.result_json)

    if args.slice_param:
        analyzer.slice_by(args.slice_param, metric=args.metric)
    elif args.cross_params:
        analyzer.cross_analysis(args.cross_params[0], args.cross_params[1],
                               metric=args.metric)
    elif args.sensitivity:
        analyzer.param_sensitivity(metric=args.metric)
    else:
        # 默认：打印 top + 敏感性
        analyzer.top_by(metric=args.metric, n=args.top)
        print()
        analyzer.param_sensitivity(metric=args.metric)

    if args.export_csv:
        analyzer.export_top_csv(args.export_csv, n=args.top)
