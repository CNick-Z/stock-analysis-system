#!/usr/bin/env python3
"""
backtest_representative.py
==========================
用指定股票列表对 WaveChan V3 进行回测（支持 with/without 周线过滤对比）

用法:
    python3 backtest_representative.py --stocks /tmp/stocks.txt --years 2024 2025 --output /tmp/result.json
    python3 backtest_representative.py --stocks /tmp/stocks.txt --years 2024 --weekly-filter --initial-cash 1000000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework, MarketSnapshot
from simulator.shared import load_wavechan_cache, add_next_open, STRATEGY_REGISTRY
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_for_strategy(stock_list, years, start_date, end_date):
    """加载指定股票列表的数据（仅波浪信号股票）"""
    # 加载日线 + 技术指标
    df = load_strategy_data(years=years, add_money_flow=True)
    logger.info(f"原始数据: {len(df):,} 行  {df['symbol'].nunique()} 只股票")

    # 过滤到指定股票
    if stock_list:
        df = df[df['symbol'].isin(stock_list)].copy()
        logger.info(f"股票列表过滤后: {len(df):,} 行  {df['symbol'].nunique()} 只股票")

    # 日期过滤
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    logger.info(f"日期过滤后: {len(df):,} 行  [{df['date'].min()} ~ {df['date'].max()}]")

    # next_open
    df = add_next_open(df)

    # 波浪信号合并（L1/L2 cache）
    wave_df = load_wavechan_cache(years)
    if not wave_df.empty:
        # 过滤到指定股票
        if stock_list:
            wave_df = wave_df[wave_df['symbol'].isin(stock_list)]
        wave_cols = [c for c in wave_df.columns if c not in ("date", "symbol")]
        for col in wave_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        wave_df = wave_df.copy()
        wave_df["date"] = wave_df["date"].astype(str)
        df = df.merge(wave_df, on=["date", "symbol"], how="left")
        logger.info(f"波浪信号合并后: {len(df):,} 行")
        n_signal = df["has_signal"].sum() if "has_signal" in df.columns else 0
        logger.info(f"  has_signal=True 行数: {n_signal:,}")
    else:
        logger.warning("波浪缓存为空！")

    return df


def run_backtest(
    strategy,
    df,
    dates,
    initial_cash=1_000_000,
    market_filter=False,
):
    """运行回测，返回结果"""
    framework = BaseFramework(
        initial_cash=initial_cash,
        state_file=f"/tmp/backtest_wavechan_{id(strategy)}.json",
        market_regime_filter=None,
    )
    # 先 reset，再设置 strategy（避免 run_backtest 内部覆盖）
    framework.reset()
    framework._strategy = strategy

    total_trades = []
    n_winning_total = 0
    n_total_total = 0

    for i, date in enumerate(dates):
        framework._on_day(date, df, dates)

        total_value = framework._calc_total_value(df, date)
        snap = MarketSnapshot(
            date=date,
            cash=framework.cash,
            total_value=total_value,
            n_positions=len(framework.positions),
            total_return=(total_value / framework.initial_cash - 1) * 100,
        )
        framework.market_snapshots.append(snap.__dict__)

        if (i + 1) % 50 == 0:
            logger.info(
                f"  {date} ({i+1}/{len(dates)}): "
                f"持仓{len(framework.positions)}只, 总值{total_value/10000:.1f}万"
            )

    # 汇总
    n_winning_total += framework.n_winning
    n_total_total += framework.n_total
    total_trades.extend(framework.trades)

    # 生成报告
    closed = [t for t in total_trades if t.get("action") == "sell"]
    total_pnl = sum(t["pnl"] for t in closed)
    win_rate = framework.n_winning / max(framework.n_total, 1) * 100

    final_value = framework.cash + sum(
        pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        for pos in framework.positions.values()
    )
    total_return = (final_value / framework.initial_cash - 1) * 100

    n_days = len(framework.market_snapshots)
    n_years = n_days / 244
    annual_return = (final_value / framework.initial_cash) ** (1 / n_years) - 1 if n_years > 0 else 0

    values = [s["total_value"] for s in framework.market_snapshots]
    peak = framework.initial_cash
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # 止损统计
    stop_loss_trades = [t for t in closed if '止损' in t.get('reason', '')]
    stop_loss_pct = len(stop_loss_trades) / max(len(closed), 1) * 100

    # 出场原因分布
    from collections import Counter
    reasons = Counter(t.get("reason", "unknown") for t in closed)

    # 周线过滤拒绝统计（从日志中提取）
    weekly_rejected = 0
    if hasattr(strategy, 'use_weekly_filter') and strategy.use_weekly_filter:
        weekly_filter_name = "周线开"
    else:
        weekly_filter_name = "周线关"

    result = {
        "weekly_filter": weekly_filter_name,
        "n_stocks": df['symbol'].nunique(),
        "start": str(dates[0]) if dates else "",
        "end": str(dates[-1]) if dates else "",
        "initial_cash": framework.initial_cash,
        "final_value": round(final_value, 0),
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(
            (pd.Series(values).pct_change().dropna().mean() /
             pd.Series(values).pct_change().dropna().std() * (244 ** 0.5))
            if len(values) > 1 and pd.Series(values).pct_change().dropna().std() > 0 else 0,
            2
        ),
        "n_trades": framework.n_total,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 0),
        "stop_loss_count": len(stop_loss_trades),
        "stop_loss_pct": round(stop_loss_pct, 1),
        "exit_reasons": dict(reasons.most_common(10)),
        "n_positions": len(framework.positions),
        "closed_trades": len(closed),
    }

    return result, framework, total_trades


def main():
    parser = argparse.ArgumentParser(description="WaveChan V3 代表性股票回测")
    parser.add_argument("--stocks", type=str, required=True, help="股票列表文件路径")
    parser.add_argument("--years", type=int, nargs="+", required=True, help="回测年份")
    parser.add_argument("--start", type=str, default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="结果JSON输出路径")
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="止损比例")
    parser.add_argument("--compare", action="store_true", help="同时运行周线开关对比")
    args = parser.parse_args()

    # 加载股票列表
    with open(args.stocks) as f:
        stock_list = [s.strip() for s in f if s.strip()]
    logger.info(f"加载 {len(stock_list)} 只股票")

    # 日期范围
    start_year = min(args.years)
    end_year = max(args.years)
    start_date = args.start or f"{start_year}-01-01"
    end_date = args.end or f"{end_year}-12-31"

    # 加载数据
    logger.info(f"加载数据: {start_date} ~ {end_date}")
    df = load_data_for_strategy(stock_list, args.years, start_date, end_date)
    dates = sorted([d for d in df['date'].unique() if start_date <= d <= end_date])
    logger.info(f"交易日: {len(dates)} 天, 股票: {df['symbol'].nunique()} 只")

    results = {}

    if args.compare:
        # 对比模式：分别运行周线开/关
        for wf in [True, False]:
            wf_name = "周线开" if wf else "周线关"
            logger.info(f"\n{'='*60}")
            logger.info(f"  开始回测 [{wf_name}]")
            logger.info(f"{'='*60}")

            from strategies.wavechan.v3_l2_cache.wavechan_strategy import WaveChanStrategy
            strat = WaveChanStrategy(
                use_weekly_filter=wf,
                stock_list=stock_list,
                stop_loss_pct=args.stop_loss,
            )

            # prepare
            strat._full_df = df.copy()
            strat._precompute_weekly_dirs_vectorized(dates)

            result, framework, trades = run_backtest(
                strat, df, dates,
                initial_cash=args.initial_cash,
            )
            results[wf_name] = result

            # 打印结果
            print_summary(result, wf_name)

    else:
        # 单次运行（默认周线开）
        logger.info(f"\n{'='*60}")
        logger.info(f"  开始回测 [周线开]")
        logger.info(f"{'='*60}")

        from strategies.wavechan.v3_l2_cache.wavechan_strategy import WaveChanStrategy
        strat = WaveChanStrategy(
            use_weekly_filter=True,
            stock_list=stock_list,
            stop_loss_pct=args.stop_loss,
        )

        strat._full_df = df.copy()
        strat._precompute_weekly_dirs_vectorized(dates)

        result, framework, trades = run_backtest(
            strat, df, dates,
            initial_cash=args.initial_cash,
        )
        results["周线开"] = result

        print_summary(result, "周线开")

    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"结果已保存: {args.output}")

    return results


def print_summary(result, label):
    print(f"\n{'=' * 60}")
    print(f"  回测报告  |  {label}")
    print(f"  区间: {result['start']} ~ {result['end']}")
    print(f"  股票数: {result['n_stocks']} 只")
    print(f"{'=' * 60}")
    print(f"  初始资金:      {result['initial_cash']:>14,.0f}")
    print(f"  最终价值:      {result['final_value']:>14,.0f}  ({result['total_return_pct']:+.2f}%)")
    print(f"  年化收益:      {result['annual_return_pct']:>14.2f}%")
    print(f"  最大回撤:      {result['max_drawdown_pct']:>14.2f}%")
    print(f"  夏普比率:      {result['sharpe']:>14.2f}")
    print(f"  总交易笔数:    {result['n_trades']:>14}  笔")
    print(f"  胜率:          {result['win_rate']:>14.1f}%")
    print(f"  累计盈亏:      {result['total_pnl']:>14,.0f}")
    print(f"  止损次数:      {result['stop_loss_count']:>14}  次  ({result['stop_loss_pct']:.1f}%)")
    print(f"  平仓次数:      {result['closed_trades']:>14}  次")
    print(f"  当前持仓:      {result['n_positions']:>14}  只")
    print(f"{'=' * 60}")
    if result['exit_reasons']:
        print(f"  出场原因分布:")
        for reason, count in result['exit_reasons'].items():
            print(f"    {reason:<30} {count:>4} 笔")


if __name__ == "__main__":
    main()
