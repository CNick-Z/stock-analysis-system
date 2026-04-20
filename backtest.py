#!/usr/bin/env python3
"""
backtest.py — 统一回测入口
===========================
用法:
    python3 backtest.py --strategy v8 --start 2024-01-01 --end 2024-12-31
    python3 backtest.py --strategy wavechan --start 2024-01-01 --end 2026-03-31
    python3 backtest.py --strategy all --start 2024-01-01 --end 2024-12-31
    python3 backtest.py --strategy v8 --start 2024-01-01 --end 2024-12-31 --output-state /tmp/state.json
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework, MarketSnapshot
from simulator.shared import (
    load_strategy,
    load_wavechan_cache,
    add_next_open,
    STRATEGY_REGISTRY,
    WAVECHAN_L2_CACHE,
)
from simulator.market_regime import MarketRegimeFilter
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_for_strategy(name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载策略所需数据，自动包含 next_open。
    WaveChan 策略会额外合并 L2 cache 的波浪信号。
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))

    # V3策略不需要资金流指标（省10秒计算）
    add_mf = "wavechan_v3" not in name
    logger.info(f"加载数据: {years} | 资金流: {'开启' if add_mf else '跳过（V3无需）'}")
    df = load_strategy_data(years=years, add_money_flow=add_mf)
    logger.info(f"原始数据: {len(df):,} 行  [{df['date'].min()} ~ {df['date'].max()}]")

    # 日期过滤
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    logger.info(f"日期过滤后: {len(df):,} 行")

    # next_open
    df = add_next_open(df)

    # WaveChan 额外加载波浪信号（包含 wavechan 和 wavechan_v3_strict）
    if "wavechan_v3_strict" in name:
        wave_df = load_wavechan_cache(years)
        if not wave_df.empty:
            # 删除可能冲突的列（保留原数据的技术指标，只覆盖波浪字段）
            wave_cols = [c for c in wave_df.columns if c not in ("date", "symbol")]
            for col in wave_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            # 统一日期格式为字符串，避免 merge 时类型不匹配
            wave_df = wave_df.copy()
            wave_df["date"] = wave_df["date"].astype(str)
            df = df.merge(wave_df, on=["date", "symbol"], how="left")
            logger.info(f"波浪信号合并后: {len(df):,} 行")
            n_signal = df["has_signal"].sum() if "has_signal" in df.columns else 0
            logger.info(f"  has_signal=True 行数: {n_signal:,}")
        else:
            logger.warning("  波浪缓存为空，请先构建 L2 cache！")

    return df


# ── 回测报告打印（与 base_framework._print_summary 风格一致） ───────────────

def print_summary(
    framework: BaseFramework,
    strategy_name: str,
    start_date: str,
    end_date: str,
) -> dict:
    """打印回测汇总，返回统计 dict（供 --strategy all 横向对比用）"""
    closed = [t for t in framework.trades if t["action"] == "sell"]
    total_pnl = sum(t["pnl"] for t in closed)
    win_rate = framework.n_winning / max(framework.n_total, 1) * 100

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

    print(f"\n{'=' * 54}")
    print(f"  回测报告  |  策略: {strategy_name}")
    print(f"  区间: {start_date} ~ {end_date}")
    print(f"{'=' * 54}")
    print(f"  初始资金:      {framework.initial_cash:>14,.0f}")
    print(f"  最终价值:      {final_value:>14,.0f}  ({total_return:+.2f}%)")
    print(f"  年化收益:      {annual_return * 100:>14.2f}%")
    print(f"  最大回撤:      {max_dd:>14.2f}%")
    print(f"  夏普比率:      {sharpe:>14.2f}")
    print(f"  总交易笔数:    {framework.n_total:>14}  笔")
    print(f"  胜率:          {win_rate:>14.1f}%")
    print(f"  累计盈亏:      {total_pnl:>14,.0f}")
    print(f"  当前持仓:      {len(framework.positions):>14}  只")
    print(f"{'=' * 54}")

    # 出场原因分布
    if closed:
        from collections import Counter
        reasons = Counter(t.get("reason", "") for t in closed)
        print(f"\n  出场原因分布:")
        for reason, count in reasons.most_common():
            print(f"    {(reason or 'unknown'):<30} {count:>4} 笔")

    return {
        "strategy": strategy_name,
        "start": start_date,
        "end": end_date,
        "initial_cash": framework.initial_cash,
        "final_value": final_value,
        "total_return_pct": total_return,
        "annual_return_pct": annual_return * 100,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "n_trades": framework.n_total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "n_positions": len(framework.positions),
    }


def print_comparison(results: list):
    """横向对比打印（--strategy all 时）"""
    print(f"\n{'=' * 90}")
    print(f"  策略横向对比  |  区间: {results[0]['start']} ~ {results[0]['end']}")
    print(f"{'=' * 90}")
    header = f"  {'策略':<10} {'最终价值':>14} {'总收益':>10} {'年化':>8} {'最大回撤':>9} {'夏普':>7} {'交易数':>7} {'胜率':>7}"
    print(header)
    print(f"  {'-'*88}")
    for r in results:
        print(
            f"  {r['strategy']:<10} "
            f"{r['final_value']:>14,.0f} "
            f"{r['total_return_pct']:>+9.2f}% "
            f"{r['annual_return_pct']:>+7.2f}% "
            f"{r['max_drawdown_pct']:>8.2f}% "
            f"{r['sharpe']:>7.2f} "
            f"{r['n_trades']:>7} "
            f"{r['win_rate']:>6.1f}%"
        )
    print(f"{'=' * 90}")


# ── 主入口 ─────────────────────────────────────────────────

def run_backtest_year_by_year(
    strat_name: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
    output_state: str = None,
    market_filter: bool = False,
    filter_confirm_days: int = 1,
    filter_neutral: float = 0.70,
    filter_bear: float = 0.30,
):
    """
    逐年回测（解决内存问题）：
    每年单独加载数据 + prepare，框架状态跨年保持（直接调用 _on_day 循环）。

    V8 需要前一年数据做 warm-up（shift/rolling 条件预计算），
    所以每年加载 [Y-1, Y] 两年，prepare 时只缓存 Y 年的日期。
    """
    strategy = load_strategy(strat_name)
    all_years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))

    # 初始化 MarketRegimeFilter（如果启用）
    market_regime_filter = None
    if market_filter:
        market_regime_filter = MarketRegimeFilter(
            confirm_days=filter_confirm_days,
            neutral_position=filter_neutral,
            bear_position=filter_bear,
        )
        market_regime_filter.prepare(start_date, end_date)
        logger.info(f"MarketRegimeFilter 已启用（连续 {filter_confirm_days} 日确认）")

    # 初始化框架（只初始化一次）
    # V3 分散持仓配置: max_positions=10, position_size=10%
    framework = BaseFramework(
        initial_cash=initial_cash,
        max_positions=10,
        position_size=0.10,
        state_file=f"/tmp/backtest_{strat_name}.json",
        market_regime_filter=market_regime_filter,
    )
    framework._strategy = strategy
    framework.reset()

    # 跨年汇总
    total_trades = []
    total_snapshots = []
    n_winning_total = 0
    n_total_total = 0

    for year in all_years:
        y_start = max(f"{year}-01-01", start_date)
        y_end = min(f"{year}-12-31", end_date)
        if y_start > y_end:
            continue

        # 只加载当年数据（每年 ~1.2M 行，避免 OOM）
        load_start = y_start
        load_end = y_end

        logger.info(f"\n  ── {year} 年回测（数据: {load_start} ~ {load_end}）──")
        try:
            df = load_data_for_strategy(strat_name, load_start, load_end)
        except Exception as e:
            logger.error(f"  数据加载失败 [{year}]: {e}")
            continue

        # ── prepare：只缓存当年日期的条件 ──────────────────────────
        if hasattr(strategy, 'prepare'):
            try:
                dates_for_prepare = [d for d in sorted(df['date'].unique())
                                     if y_start <= d <= y_end]
                strategy.prepare(dates_for_prepare, df)
            except Exception as prep_e:
                logger.warning(f"  策略 prepare() 失败 [{year}]: {prep_e}")

        # ── 逐年日内循环（直接调用 _on_day，不走 run_backtest 以避免 reset）──
        dates_in_year = sorted([
            d for d in df['date'].unique()
            if y_start <= d <= y_end
        ])

        logger.info(f"  {year} 年: {len(dates_in_year)} 个交易日")

        for i, date in enumerate(dates_in_year):
            framework._on_day(date, df, dates_in_year)

            # 每日快照
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
                    f"  {date} ({i+1}/{len(dates_in_year)}): "
                    f"持仓{len(framework.positions)}只, 总值{total_value/10000:.1f}万"
                )

        # 当年完成，汇总
        n_winning_total += framework.n_winning
        n_total_total += framework.n_total
        total_trades.extend(framework.trades)
        framework.trades = []          # 清空当年 trade 缓冲
        framework.n_winning = 0
        framework.n_total = 0

        logger.info(f"  {year} 年完成，持仓{len(framework.positions)}只，"
                    f"现金{framework.cash:,.0f}")

        # 释放内存
        del df
        import gc
        gc.collect()

    # ── 汇总报告 ─────────────────────────────────────────────
    framework.trades = total_trades
    framework.n_winning = n_winning_total
    framework.n_total = n_total_total

    result = _make_summary(framework, strat_name, start_date, end_date)

    if output_state:
        state = {
            "strategy": strat_name,
            "start": start_date,
            "end": end_date,
            "cash": framework.cash,
            "positions": framework.positions,
            "trades": total_trades,
            "n_winning": n_winning_total,
            "n_total": n_total_total,
        }
        import json
        with open(output_state, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"状态已保存: {output_state}")

    return result


def _make_summary(framework: BaseFramework, strat_name: str, start: str, end: str) -> dict:
    """生成回测汇总 dict（用于跨年汇总）"""
    closed = [t for t in framework.trades if t.get("action") == "sell"]
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

    if len(values) > 1:
        daily_returns = pd.Series(values).pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * (244 ** 0.5) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    from collections import Counter
    print(f"\n{'=' * 60}")
    print(f"  回测报告  |  策略: {strat_name}  |  区间: {start} ~ {end}")
    print(f"{'=' * 60}")
    print(f"  初始资金:  {framework.initial_cash:>14,.0f}")
    print(f"  最终价值:  {final_value:>14,.0f}  ({total_return:+.2f}%)")
    print(f"  年化收益:  {annual_return * 100:>14.2f}%")
    print(f"  最大回撤:  {max_dd:>14.2f}%")
    print(f"  夏普比率:  {sharpe:>14.2f}")
    print(f"  总交易:    {framework.n_total:>14}  笔")
    print(f"  胜率:      {win_rate:>14.1f}%")
    print(f"  累计盈亏:  {total_pnl:>14,.0f}")
    print(f"  当前持仓:  {len(framework.positions):>14}  只")
    print(f"{'=' * 60}")

    if closed:
        reasons = Counter(t.get("reason", "") for t in closed)
        print(f"\n  出场原因分布:")
        for reason, count in reasons.most_common():
            print(f"    {(reason or 'unknown'):<30} {count:>4} 笔")

    return {
        "strategy": strat_name,
        "start": start,
        "end": end,
        "initial_cash": framework.initial_cash,
        "final_value": final_value,
        "total_return_pct": total_return,
        "annual_return_pct": annual_return * 100,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "n_trades": framework.n_total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "n_positions": len(framework.positions),
    }


def main():
    parser = argparse.ArgumentParser(description="统一回测入口")
    parser.add_argument("--strategy", required=True,
                        choices=["v8", "wavechan_v3_strict", "all"],
                        help="选择策略: v8, wavechan_v3_strict（铁律过滤）, all")
    parser.add_argument("--start", required=True, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--output-state", help="可选：保存最终状态到文件")
    parser.add_argument("--initial-cash", type=float, default=1_000_000,
                        help="初始资金（默认 1,000,000）")
    parser.add_argument("--market-filter", action="store_true",
                        help="启用大盘牛熊过滤器（BEAR转40%%/NEUTRAL转80%%/BULL转100%%）")
    parser.add_argument("--filter-confirm-days", type=int, default=1,
                        help="过滤器连续确认天数（默认1）")
    parser.add_argument("--filter-neutral", type=float, default=0.70,
                        help="震荡市仓位上限（默认0.70）")
    parser.add_argument("--filter-bear", type=float, default=0.30,
                        help="熊市仓位上限（默认0.30）")
    args = parser.parse_args()

    strategies_to_run = ["v8", "wavechan"] if args.strategy == "all" else [args.strategy]
    results = []

    for strat_name in strategies_to_run:
        logger.info(f"\n{'='*50}\n  开始回测: {strat_name}\n{'='*50}")
        result = run_backtest_year_by_year(
            strat_name=strat_name,
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.initial_cash,
            output_state=args.output_state,
            market_filter=args.market_filter,
            filter_confirm_days=args.filter_confirm_days,
            filter_neutral=args.filter_neutral,
            filter_bear=args.filter_bear,
        )
        if result:
            results.append(result)

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
