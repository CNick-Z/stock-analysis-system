#!/usr/bin/env python3
"""
离线仓位管理验证 — Trader 方案
================================
对三个典型窗口：
  1. 跑基准回测（全量交易）
  2. 叠加 SequentialStopLossSizer 重算
  3. 对比年化/夏普/最大回撤/交易笔数
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from simulator.base_framework import BaseFramework
from strategies.score.v8.strategy import ScoreV8Strategy
from simulator.position_sizer import SequentialStopLossSizer, PositionSizeConfig, TradeResult
from utils.data_loader import load_strategy_data


def run_backtest_get_trades(y1, y2, params):
    """跑回测，返回交易序列"""
    df = load_strategy_data(years=[y1, y2])
    fw = BaseFramework(initial_cash=1_000_000, position_size=0.15, max_positions=3)
    strat = ScoreV8Strategy(params)
    fw.run_backtest(strat, df)

    # 提取平仓交易
    trades = []
    for t in fw.trades:
        if t.get("action") != "sell":
            continue
        pnl_pct = t.get("pnl_pct", 0) or 0
        # 判断是否止损出局（亏损且理由含止损关键词）
        reason = t.get("reason", "") or ""
        is_stop = ("止损" in reason or "stop" in reason.lower()) and pnl_pct < 0
        trades.append(TradeResult(
            date=t["date"],
            pnl_pct=pnl_pct,
            is_stop_loss=is_stop,
        ))

    # 提取权益曲线
    equity = [(s["date"], s["total_value"]) for s in fw.market_snapshots]
    return trades, equity, fw.initial_cash


def compute_stats(equity_curve, initial_cash):
    """从权益曲线算指标"""
    values = [e[1] for e in equity_curve]
    dates = [e[0] for e in equity_curve]
    if len(values) < 10:
        return {}
    total_ret = (values[-1] - initial_cash) / initial_cash * 100
    ann_ret = total_ret / (len(dates) / 244)
    daily_rets = np.diff(values) / values[:-1]
    std = np.std(daily_rets) if len(daily_rets) > 0 else 1.0
    sharpe = (np.mean(daily_rets) / std * np.sqrt(244)) if std > 0 else 0
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return {
        "annual_return_pct": ann_ret,
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "final_value": values[-1],
    }


def apply_sizer_equity(trades, equity_curve, initial_cash, config):
    """
    用 SequentialStopLossSizer 重算权益曲线。
    仓位变化只在下次买入时生效（等仓位生效后的第一笔新交易才改变持仓金额）。
    """
    sizer = SequentialStopLossSizer(config)

    # 按时间顺序重算
    # equity_curve: [(date, total_value), ...]
    # 策略：sizer 的仓位比例生效后，每次新交易的仓位按 effective_position 计算

    # 初始资金 fold
    cash = initial_cash
    positions = 0  # 当前持仓股数（简化：不跟踪具体持仓，只跟踪资金）
    # 更简单的方式：记录每笔交易时的仓位比例，在计算收益时乘以该比例

    # 实际上我们只需要把仓位比例乘到每笔交易的盈亏上
    # 假设每笔交易投入的资金 = 当前权益 * position_size（基准15%）
    # 加上仓位调整后，实际投入 = 当前权益 * position_size * sizer_ratio

    # 更简单：把 sizer 的仓位比例作为一个乘数，乘到每笔交易的资金上
    # 资金曲线 = 初始 * 累积(position_ratio_i * 盈亏比例_i)

    # 用基准仓位的收益序列，然后叠加仓位调整
    # 权益曲线日期对齐
    trade_idx = 0
    n_trades = len(trades)

    # 简单近似：每次新买入时，记录当时的 sizer 仓位
    # 我们没有逐日记录每笔新买入的日期，但可以用交易笔数来对齐
    # 这里用更简化的方式：直接对最终收益应用仓位乘数

    # 实际上最准确的做法：每笔交易结束后，记录下当时的 sizer 状态，
    # 下次买入时用那个状态对应的仓位。
    # 这里近似：用 sizer 的平均仓位作为整体乘数

    # 更实际：模拟逐笔交易
    equity_adjusted = [initial_cash]
    current_equity = initial_cash
    sizer.reset()

    # 建立交易日期到仓位的映射
    trade_by_date = {}
    for t in trades:
        trade_by_date[t.date] = t

    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    # 用日权益和交易序列做粗略对齐
    # 每天结束：检查有无交易，更新 sizer，如果下一天有新买入，用新仓位
    # 简化：按交易顺序，假设每笔交易的仓位 = 上一次平仓后的 sizer 仓位
    last_position_pct = 1.0  # 最近一次生效的仓位

    for i, (date, value) in enumerate(equity_curve):
        if date in trade_by_date:
            t = trade_by_date[date]
            sizer.record_trade(t)
            # 这次交易用上次的仓位
            applied_pos = last_position_pct
            if i > 0:
                # 用基准权益计算这笔交易带来的资金变化
                base_trade_pnl = (value - equity_adjusted[-1])
                adjusted_pnl = base_trade_pnl * applied_pos
                current_equity += adjusted_pnl
                equity_adjusted.append(current_equity)
            last_position_pct = sizer.get_position()  # 更新下次仓位
        else:
            # 无交易，权益按比例调整（简化）
            if i > 0:
                ratio = value / equity_curve[i-1][1] if equity_curve[i-1][1] > 0 else 1.0
                # 不持仓时，仓位调整不影响资金（实际是持仓期间才影响）
                # 这里直接用基准
                current_equity = equity_adjusted[-1] * ratio
                equity_adjusted.append(current_equity)

    return [(d, v) for d, v in zip(dates, equity_adjusted)]


def main():
    params = {
        "stop_loss": 0.04,
        "take_profit": 0.20,
        "rsi_filter_min": 50,
        "rsi_filter_max": 65,
    }

    windows = [
        (2010, 2011, "🔴熊市 2010-2011"),
        (2014, 2015, "🟢牛市 2014-2015"),
        (2021, 2022, "🟡震荡 2021-2022"),
    ]

    config = PositionSizeConfig(
        n1=3, n2=5,
        x=0.50, y=0.25,
        z1=2, z2=4,
    )

    print(f"{'='*75}")
    print(f"{'市场':<22} {'方案':<22} {'年化':>9} {'夏普':>7} {'最大回撤':>10} {'交易笔数':>8}")
    print(f"{'='*75}")

    all_results = []

    for y1, y2, label in windows:
        print(f"\n  {label}")

        # 跑基准
        trades, equity, ic = run_backtest_get_trades(y1, y2, params)
        s_base = compute_stats(equity, ic)
        n_base = len(trades)
        print(f"  基准{' '*17}: {s_base.get('annual_return_pct',0):>+8.2f}%  "
              f"{s_base.get('sharpe',0):>6.2f}  "
              f"{s_base.get('max_drawdown_pct',0):>+9.1f}%  {n_base:>6d}笔")

        # 叠加仓位管理
        sizer = SequentialStopLossSizer(config)
        # 按顺序记录交易
        for t in trades:
            sizer.record_trade(t)

        # 简化评估：把 sizer 的平均仓位作为整体乘数应用到最终收益
        # 更准确：用每笔交易的实际仓位
        # 这里用交易笔数来估算
        # 实际上我们需要一个更精细的重算

        # 精确重算：模拟逐笔
        # 假设：每笔交易金额 = 当日收盘指数 * 仓位比例
        # 平仓时：资金变化 = 交易金额 * pnl_pct
        # 但我们没有每笔交易的入场金额，只有 pnl_pct

        # 最简方法：直接把 sizer 的仓位乘到总收益上
        # 即：降仓50%相当于亏损也减少一半
        avg_pos = sizer.current_position  # 这是最终仓位，不能直接用

        # 用状态序列重算
        # 建立状态：[(连亏数, 连盈数, 仓位), ...]
        states = []
        sizer.reset()
        for t in trades:
            states.append((sizer.consecutive_losses, sizer.consecutive_wins, sizer.get_position()))
            sizer.record_trade(t)

        # 每笔交易后的仓位 = 下次买入时的仓位
        # 第一次交易用 100%
        position_seq = [1.0]  # 第一笔前的仓位
        for i, t in enumerate(trades):
            position_seq.append(states[i][2])  # 第i笔后的仓位

        # 简化：每笔交易的收益按该笔对应的仓位调整
        # 基准：equity curve 的逐日比值变化
        # 实际权益 = Σ(每笔交易 * 对应仓位)
        # 用每日权益变化 * 当日交易对应的仓位
        trade_dates = set(t.date for t in trades)

        # 重新计算：逐日权益 * 仓位
        equity_adjusted = [ic]
        sizer.reset()
        # 每天结束检查有无交易
        date_to_trade = {t.date: t for t in trades}
        prev_pos = 1.0

        for i, (date, val) in enumerate(equity):
            if i == 0:
                continue
            base_ret = (val - equity[i-1][1]) / equity[i-1][1] if equity[i-1][1] > 0 else 0

            if date in date_to_trade:
                t = date_to_trade[date]
                # 交易用上次的仓位
                effective_ret = base_ret * prev_pos
                sizer.record_trade(t)
                prev_pos = sizer.get_position()
            else:
                # 无新交易，仓位不变
                effective_ret = base_ret * prev_pos

            new_val = equity_adjusted[-1] * (1 + effective_ret)
            equity_adjusted.append(new_val)

        s_adjusted = compute_stats(list(zip([e[0] for e in equity], equity_adjusted)), ic)
        print(f"  连续止损降仓{' '*10}: {s_adjusted.get('annual_return_pct',0):>+8.2f}%  "
              f"{s_adjusted.get('sharpe',0):>6.2f}  "
              f"{s_adjusted.get('max_drawdown_pct',0):>+9.1f}%  {len(trades):>6d}笔")

        delta = s_adjusted.get('annual_return_pct', 0) - s_base.get('annual_return_pct', 0)
        delta_dd = s_adjusted.get('max_drawdown_pct', 0) - s_base.get('max_drawdown_pct', 0)
        print(f"  {'  改善:':>20} Δ年化{delta:>+7.2f}%  Δ回撤{delta_dd:>+7.1f}%")

        all_results.append({
            "window": f"{y1}-{y2}",
            "label": label,
            "baseline": s_base,
            "with_sizing": s_adjusted,
            "delta_ann": delta,
            "delta_dd": delta_dd,
            "n_trades": len(trades),
        })

    # 保存
    out = "/tmp/sequential_sizing_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out}")

    # 最终总结
    print(f"\n{'='*75}")
    print("结论：")
    for r in all_results:
        arrow = "↑" if r["delta_ann"] > 0 else "↓"
        print(f"  {r['label']}: 年化 {r['baseline']['annual_return_pct']:+.2f}% → "
              f"{r['with_sizing']['annual_return_pct']:+.2f}% ({arrow}{abs(r['delta_ann']):.2f}%)  "
              f"回撤 {r['baseline']['max_drawdown_pct']:.1f}% → {r['with_sizing']['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s  %(message)s")
    main()
