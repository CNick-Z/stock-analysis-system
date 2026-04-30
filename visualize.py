#!/usr/bin/env python3
"""
可视化模块
==========
A. 单次回测结果：Equity Curve、Drawdown、Trade PnL分布、月度收益热力图
B. Walk-Forward 汇总：IS vs OOS 对比图
C. V3 + V8 模拟盘对比：双策略资金曲线、持仓分布

用法：
    python3 visualize.py --type backtest --input backtest_results.json --output figs/
    python3 visualize.py --type walkforward --input wf_results.csv --output figs/
    python3 visualize.py --type compare --input3 v3_sim.json --input8 v8_sim.json --output figs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 图 A: 单次回测可视化
# ============================================================

def plot_backtest(
    trades: list,
    market_snapshots: list,
    title: str = "Backtest Result",
    output_path: str = None,
    benchmark_df: pd.DataFrame = None,
    initial_cash: float = 500_000,
):
    """
    绘制回测结果：Equity Curve + Drawdown + Trade PnL + 月度热力图

    Args:
        trades: 交易记录列表
        market_snapshots: 每日市值快照
        title: 图表标题
        output_path: 保存路径（不含文件名）
        benchmark_df: 基准指数 DataFrame（date, close 列）
        initial_cash: 初始资金
    """
    import matplotlib
    matplotlib.use("Agg")
    # 静默字体找不到警告
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import colors as mcolors
    # 优先使用系统安装的中文字体
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "WenQuanYi Micro Hei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    if not market_snapshots:
        logging.warning("market_snapshots 为空，跳过绘图")
        return

    # ---- 整理数据 ----
    dates = [s["date"] for s in market_snapshots]
    values = [s["total_value"] for s in market_snapshots]
    cash_vals = [s.get("cash", s["total_value"]) for s in market_snapshots]

    equity_df = pd.DataFrame({"date": dates, "value": values, "cash": cash_vals})
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df.set_index("date", inplace=True)

    # 计算收益率
    equity_df["ret"] = equity_df["value"].pct_change().fillna(0)
    equity_df["equity_normalized"] = equity_df["value"] / initial_cash

    # 计算 Drawdown
    peak = equity_df["value"].cummax()
    drawdown = (equity_df["value"] - peak) / peak * 100

    # 整理交易
    sell_trades = [t for t in trades if t.get("action") == "sell"]
    trade_dates = [t["date"] for t in sell_trades]
    trade_pnls = [t["pnl"] for t in sell_trades]
    trade_rets = [t["return_pct"] / 100 for t in sell_trades if "return_pct" in t]

    # 月度收益
    monthly = (1 + equity_df["ret"]).resample("ME").prod() - 1
    monthly = monthly.dropna()

    # 持仓天数分布（从快照）
    holdings = [s.get("n_holdings", 0) for s in market_snapshots]

    # ---- 开始绘图 ----
    n_figs = 4
    fig = plt.figure(figsize=(16, 4 * n_figs), facecolor="white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    # 1. Equity Curve vs Benchmark
    ax1 = fig.add_subplot(n_figs, 1, 1)
    ax1.plot(equity_df.index, equity_df["equity_normalized"], color="#2196F3", linewidth=2, label="策略")
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

    if benchmark_df is not None and not benchmark_df.empty:
        bdf = benchmark_df.copy()
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf.set_index("date", inplace=True)
        bdf = bdf[bdf.index >= equity_df.index[0]]
        bdf = bdf[bdf.index <= equity_df.index[-1]]
        if not bdf.empty and "close" in bdf.columns:
            bnorm = bdf["close"] / bdf["close"].iloc[0]
            ax1.plot(bdf.index, bnorm, color="#FF5722", linewidth=1.5, alpha=0.7, label="基准(上证)")

    # 标注终点
    final_val = equity_df["equity_normalized"].iloc[-1]
    ax1.scatter([equity_df.index[-1]], [final_val], color="#2196F3", zorder=5, s=80)
    ax1.annotate(
        f"{final_val:.2f}x",
        xy=(equity_df.index[-1], final_val),
        xytext=(8, 0), textcoords="offset points",
        fontsize=11, fontweight="bold", color="#2196F3"
    )

    ax1.set_title("Equity Curve", fontsize=14, pad=8)
    ax1.set_ylabel("净值（初始=1）")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # 2. Drawdown
    ax2 = fig.add_subplot(n_figs, 1, 2, sharex=ax1)
    ax2.fill_between(equity_df.index, drawdown, 0, color="#F44336", alpha=0.4, label="回撤")
    ax2.plot(equity_df.index, drawdown, color="#F44336", linewidth=1)
    max_dd = drawdown.min()
    ax2.axhline(max_dd, color="#B71C1C", linewidth=0.8, linestyle="--")
    ax2.annotate(
        f"最大回撤: {max_dd:.1f}%",
        xy=(equity_df.index[drawdown.argmin()], max_dd),
        xytext=(8, -20), textcoords="offset points",
        fontsize=10, color="#B71C1C"
    )
    ax2.set_title("Drawdown", fontsize=14, pad=8)
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(drawdown.min() * 1.2, -30), 1)

    # 3. Trade PnL 分布
    ax3 = fig.add_subplot(n_figs, 1, 3)
    if sell_trades:
        color = ["#4CAF50" if p > 0 else "#F44336" for p in trade_pnls]
        ax3.bar(range(len(trade_pnls)), trade_pnls, color=color, alpha=0.8, width=0.6)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_title(f"Trade PnL（{len(trade_pnls)}笔，胜率{sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100:.0f}%）", fontsize=14, pad=8)
        ax3.set_ylabel("PnL (元)")
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "无平仓交易", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Trade PnL", fontsize=14)

    # 4. 月度收益热力图
    ax4 = fig.add_subplot(n_figs, 1, 4)
    if len(monthly) > 0:
        monthly_pct = monthly * 100
        month_labels = monthly.index.strftime("%Y-%m")
        bar_colors = ["#4CAF50" if v > 0 else "#F44336" for v in monthly_pct.values]
        bars = ax4.bar(range(len(monthly_pct)), monthly_pct.values, color=bar_colors, alpha=0.8, width=0.7)
        ax4.set_xticks(range(len(monthly_pct)))
        ax4.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.set_title("Monthly Returns (%)", fontsize=14, pad=8)
        ax4.set_ylabel("月收益 (%)")
        ax4.grid(True, alpha=0.3, axis="y")
        # 标注数值
        for bar, val in zip(bars, monthly_pct.values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1.5),
                f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=6, color="#333"
            )
    else:
        ax4.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Monthly Returns (%)", fontsize=14)

    plt.tight_layout()
    fig_path = _save_fig(fig, output_path, title)
    plt.close(fig)
    logging.info(f"回测图表已保存: {fig_path}")
    return fig_path


def _save_fig(fig, output_path: str, title: str) -> str:
    import matplotlib.pyplot as plt
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "-")[:50]
        fig_path = str(Path(output_path) / f"{safe_title}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig_path
    else:
        plt.show()
        return ""


# ============================================================
# 图 B: Walk-Forward 结果可视化
# ============================================================

def plot_walkforward(
    wf_df: pd.DataFrame,
    title: str = "Walk-Forward Validation",
    output_path: str = None,
):
    """
    绘制 Walk-Forward 汇总结果

    Args:
        wf_df: 包含 train_start, train_end, test_start, test_end,
               train_total_return, oos_total_return, oos_max_drawdown, wfe 等列
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "WenQuanYi Micro Hei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    if wf_df.empty:
        logging.warning("wf_df 为空，跳过绘图")
        return

    n = len(wf_df)
    x = np.arange(n)
    labels = [f"{r['train_start'][:7]}\n→\n{r['test_end'][:7]}" for _, r in wf_df.iterrows()]

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    # 1. IS vs OOS 收益对比（双柱状图）
    ax1 = fig.add_subplot(3, 1, 1)
    bar_w = 0.35
    bars1 = ax1.bar(x - bar_w/2, wf_df["train_total_return"], bar_w, color="#2196F3", alpha=0.8, label="训练集(IS)")
    bars2 = ax1.bar(x + bar_w/2, wf_df["oos_total_return"], bar_w, color="#FF9800", alpha=0.8, label="测试集(OOS)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("收益 (%)")
    ax1.set_title("IS vs OOS Returns by Window", fontsize=14, pad=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, wf_df["train_total_return"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -1),
                 f"{val:.1f}%", ha="center", fontsize=7,
                 color="#1565C0" if val >= 0 else "#B71C1C")
    for bar, val in zip(bars2, wf_df["oos_total_return"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -1),
                 f"{val:.1f}%", ha="center", fontsize=7,
                 color="#E65100" if val >= 0 else "#B71C1C")

    # 2. WFE 时序 + OOS 回撤
    ax2 = fig.add_subplot(3, 1, 2)
    wfe_colors = ["#4CAF50" if v > 0.5 else "#FF9800" if v > 0 else "#F44336" for v in wf_df["wfe"]]
    ax2.bar(x, wf_df["wfe"], color=wfe_colors, alpha=0.8, width=0.5)
    ax2.axhline(0.5, color="#4CAF50", linewidth=0.8, linestyle="--", label="稳健阈值(0.5)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("WFE")
    ax2.set_title("Walk-Forward Efficiency (WFE) = OOS_return / IS_return", fontsize=14, pad=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    for i, (xi, wfe) in enumerate(zip(x, wf_df["wfe"])):
        ax2.text(xi, wfe + (0.03 if wfe >= 0 else -0.08), f"{wfe:.2f}",
                 ha="center", fontsize=8, color="#333")

    # 3. OOS 回撤 + 胜率
    ax3 = fig.add_subplot(3, 1, 3)
    ax3_twin = ax3.twinx()
    ax3.bar(x, -wf_df["oos_max_drawdown"], color="#F44336", alpha=0.5, width=0.4, label="OOS回撤")
    ax3_twin.plot(x, wf_df["oos_win_rate"], "o-", color="#9C27B0", linewidth=2, markersize=6, label="OOS胜率")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylabel("回撤 (%)", color="#F44336")
    ax3_twin.set_ylabel("胜率 (%)", color="#9C27B0")
    ax3.set_title("OOS Drawdown & Win Rate", fontsize=14, pad=8)
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3, axis="y")

    # 汇总数据表
    summary_text = (
        f"窗口数: {n}\n"
        f"IS平均收益: {wf_df['train_total_return'].mean():+.1f}%\n"
        f"OOS平均收益: {wf_df['oos_total_return'].mean():+.1f}%\n"
        f"OOS平均胜率: {wf_df['oos_win_rate'].mean():.0f}%\n"
        f"OOS平均回撤: {wf_df['oos_max_drawdown'].mean():.1f}%\n"
        f"WFE均值: {wf_df['wfe'].mean():.2f}\n"
        f"WFE>0.5的窗口: {(wf_df['wfe'] > 0.5).sum()}/{n}\n"
        f"OOS正收益窗口: {(wf_df['oos_total_return'] > 0).sum()}/{n}"
    )
    fig.text(0.01, 0.01, summary_text, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    plt.tight_layout()
    fig_path = _save_fig(fig, output_path, title)
    plt.close(fig)
    logging.info(f"Walk-Forward 图表已保存: {fig_path}")
    return fig_path


# ============================================================
# 图 C: V3 vs V8 模拟盘对比
# ============================================================

def plot_simulation_comparison(
    sim_v3: dict,
    sim_v8: dict,
    output_path: str = None,
):
    """
    绘制 V3 和 V8 模拟盘的对比图

    Args:
        sim_v3: V3 模拟盘状态 dict
        sim_v8: V8 模拟盘状态 dict
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "WenQuanYi Micro Hei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    def extract_trades(sim):
        return sim.get("trade_history", [])

    def extract_snapshots(sim):
        return sim.get("market_snapshots", [])

    def build_equity(trade_history, snapshots, initial_cash=500_000):
        """从 snapshot 列表构建 equity series"""
        if snapshots:
            dates = [s["date"] for s in snapshots]
            vals = [s["total_value"] for s in snapshots]
        else:
            # 从 trade_history 重建
            dates, vals = [], []
            running_cash = initial_cash
            pos = {}
            last_date = None
            for t in sorted(trade_history, key=lambda x: x["date"]):
                if t["date"] != last_date and last_date is not None:
                    # 当日结算市值
                    daily_val = running_cash + sum(
                        t2.get("latest_price", 0) * t2.get("qty", 0)
                        for t2 in trade_history
                        if t2["date"] <= last_date and t2["action"] == "buy"
                        and t2.get("symbol") not in [tt.get("symbol") for tt in trade_history
                                                      if tt["date"] == last_date and tt["action"] == "sell"]
                    )
                    dates.append(last_date)
                    vals.append(daily_val)
                last_date = t["date"]
                if t["action"] == "buy":
                    pos[t["symbol"]] = {"qty": t["qty"], "price": t["price"]}
                elif t["action"] == "sell":
                    pos.pop(t["symbol"], None)
            if last_date:
                dates.append(last_date)
                vals.append(running_cash)

        if not dates:
            return pd.Series(dtype=float)
        return pd.Series(vals, index=pd.to_datetime(dates))

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle("V3 vs V8 模拟盘对比", fontsize=18, fontweight="bold", y=1.02)

    # 数据准备
    v3_trades = extract_trades(sim_v3)
    v8_trades = extract_trades(sim_v8)
    v3_snaps = extract_snapshots(sim_v3)
    v8_snaps = extract_snapshots(sim_v8)

    initial = 500_000

    # Equity Curve 对比
    ax1 = fig.add_subplot(3, 1, 1)

    if v3_snaps:
        v3_dates = [s["date"] for s in v3_snaps]
        v3_vals = [s["total_value"] for s in v3_snaps]
        v3_norm = pd.Series(np.array(v3_vals) / initial, index=pd.to_datetime(v3_dates))
        ax1.plot(v3_norm.index, v3_norm.values, color="#E91E63", linewidth=2, label="V3(WaveChan)", alpha=0.9)

    if v8_snaps:
        v8_dates = [s["date"] for s in v8_snaps]
        v8_vals = [s["total_value"] for s in v8_snaps]
        v8_norm = pd.Series(np.array(v8_vals) / initial, index=pd.to_datetime(v8_dates))
        ax1.plot(v8_norm.index, v8_norm.values, color="#2196F3", linewidth=2, label="V8(Score)", alpha=0.9)

    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

    # 标注终点
    if v3_snaps:
        ax1.scatter([v3_norm.index[-1]], [v3_norm.iloc[-1]], color="#E91E63", zorder=5, s=80)
        ax1.annotate(f"V3: {v3_norm.iloc[-1]:.2f}x", xy=(v3_norm.index[-1], v3_norm.iloc[-1]),
                     xytext=(8, 0), textcoords="offset points", fontsize=10, color="#E91E63", fontweight="bold")
    if v8_snaps:
        ax1.scatter([v8_norm.index[-1]], [v8_norm.iloc[-1]], color="#2196F3", zorder=5, s=80)
        ax1.annotate(f"V8: {v8_norm.iloc[-1]:.2f}x", xy=(v8_norm.index[-1], v8_norm.iloc[-1]),
                     xytext=(8, -15), textcoords="offset points", fontsize=10, color="#2196F3", fontweight="bold")

    ax1.set_title("Equity Curve（净值）", fontsize=14, pad=8)
    ax1.set_ylabel("净值（初始=1）")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # 持仓对比（表格）
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.axis("off")

    v3_pos = sim_v3.get("positions", {})
    v8_pos = sim_v8.get("positions", {})

    table_data = []
    for sym, pos in sorted(v3_pos.items()):
        table_data.append([sym, "V3", f"{pos.get('qty', 0)}股", f"¥{pos.get('latest_price', 0):.2f}",
                          f"{pos.get('current_value', 0):.0f}元", ""])
    for sym, pos in sorted(v8_pos.items()):
        table_data.append([sym, "V8", f"{pos.get('qty', 0)}股", f"¥{pos.get('latest_price', 0):.2f}",
                          f"{pos.get('current_value', 0):.0f}元", ""])

    if table_data:
        col_labels = ["股票", "策略", "数量", "现价", "市值", "持仓天数"]
        table = ax2.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper left",
            bbox=[0, 0, 0.9, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        # 表头颜色
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor("#37474F")
            table[(0, j)].set_text_props(color="white", fontweight="bold")
        # V3/V8 行颜色
        for i, row_data in enumerate(table_data):
            color = "#FFEBEE" if row_data[1] == "V3" else "#E3F2FD"
            for j in range(len(col_labels)):
                table[(i+1, j)].set_facecolor(color)
    else:
        ax2.text(0.5, 0.5, "当前无持仓", ha="center", va="center", transform=ax2.transAxes, fontsize=12)

    ax2.set_title("当前持仓（V3粉色 / V8蓝色）", fontsize=14, pad=8, loc="left")

    # 收益汇总对比
    ax3 = fig.add_subplot(3, 1, 3)

    def get_summary(sim):
        snaps = sim.get("market_snapshots", [])
        trades = sim.get("trade_history", [])
        if not snaps:
            return {}
        final_val = snaps[-1]["total_value"]
        peak = max(s["total_value"] for s in snaps)
        max_dd = (final_val - peak) / peak * 100 if peak > 0 else 0
        sell_trades = [t for t in trades if t.get("action") == "sell"]
        n_total = len(sell_trades)
        n_win = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
        return {
            "total_return": (final_val / initial - 1) * 100,
            "final_value": final_val,
            "max_drawdown": abs(max_dd),
            "n_trades": n_total,
            "win_rate": n_win / max(n_total, 1) * 100,
            "cash": snaps[-1].get("cash", final_val),
        }

    v3_s = get_summary(sim_v3)
    v8_s = get_summary(sim_v8)

    metrics = ["总收益", "最大回撤", "交易笔数", "胜率"]
    v3_vals = [v3_s.get("total_return", 0), v3_s.get("max_drawdown", 0),
               v3_s.get("n_trades", 0), v3_s.get("win_rate", 0)]
    v8_vals = [v8_s.get("total_return", 0), v8_s.get("max_drawdown", 0),
               v8_s.get("n_trades", 0), v8_s.get("win_rate", 0)]

    x = np.arange(len(metrics))
    bar_w = 0.3
    bars1 = ax3.bar(x - bar_w/2, v3_vals, bar_w, color="#E91E63", alpha=0.8, label="V3(WaveChan)")
    bars2 = ax3.bar(x + bar_w/2, v8_vals, bar_w, color="#2196F3", alpha=0.8, label="V8(Score)")

    for bar, val in zip(bars1, v3_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if val >= 0 else -4),
                 f"{val:.1f}", ha="center", fontsize=10, color="#880E4F")
    for bar, val in zip(bars2, v8_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if val >= 0 else -4),
                 f"{val:.1f}", ha="center", fontsize=10, color="#0D47A1")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=11)
    ax3.set_title("关键指标对比", fontsize=14, pad=8)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3, axis="y")

    # 汇总文字
    summary_text = (
        f"V3: 总收益{v3_s.get('total_return', 0):+.1f}% | 回撤{v3_s.get('max_drawdown', 0):.1f}% | "
        f"{v3_s.get('n_trades', 0)}笔 | 胜率{v3_s.get('win_rate', 0):.0f}%\n"
        f"V8: 总收益{v8_s.get('total_return', 0):+.1f}% | 回撤{v8_s.get('max_drawdown', 0):.1f}% | "
        f"{v8_s.get('n_trades', 0)}笔 | 胜率{v8_s.get('win_rate', 0):.0f}%"
    )
    fig.text(0.5, 0.01, summary_text, fontsize=10, ha="center", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#FAFAFA", alpha=0.9))

    plt.tight_layout()
    fig_path = _save_fig(fig, output_path, "v3_v8_comparison")
    plt.close(fig)
    logging.info(f"V3/V8 对比图已保存: {fig_path}")
    return fig_path


# ============================================================
# CLI 入口
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="回测/模拟盘可视化工具")
    parser.add_argument("--type", required=True, choices=["backtest", "walkforward", "compare"])
    parser.add_argument("--input", type=str, help="回测结果 JSON 或 WF CSV 路径")
    parser.add_argument("--input3", type=str, help="V3 模拟盘 JSON")
    parser.add_argument("--input8", type=str, help="V8 模拟盘 JSON")
    parser.add_argument("--benchmark", type=str, default=None, help="基准指数 Parquet 路径")
    parser.add_argument("--output", type=str, default="/tmp/visualize", help="图片输出目录")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    output_path = args.output
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if args.type == "backtest":
        with open(args.input) as f:
            data = json.load(f)
        trades = data.get("trades", [])
        snapshots = data.get("market_snapshots", [])
        initial = data.get("initial_cash", 500_000)

        benchmark_df = None
        if args.benchmark:
            import pandas as pd
            benchmark_df = pd.read_parquet(args.benchmark)

        title = args.title or "Backtest Result"
        plot_backtest(trades, snapshots, title, output_path, benchmark_df, initial)

    elif args.type == "walkforward":
        wf_df = pd.read_csv(args.input)
        title = args.title or "Walk-Forward Validation"
        plot_walkforward(wf_df, title, output_path)

    elif args.type == "compare":
        with open(args.input3) as f:
            sim_v3 = json.load(f)
        with open(args.input8) as f:
            sim_v8 = json.load(f)
        plot_simulation_comparison(sim_v3, sim_v8, output_path)


if __name__ == "__main__":
    main()
