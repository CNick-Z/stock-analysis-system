#!/usr/bin/env python3
"""
V8 ATR止损参数扫描
==================
串行扫描 ATR 倍数 (1.5 / 2.0 / 2.5) × 当日ATR
样本外：2007-2016
样本内：2018-2025

止损模式：
  - ATR止损：收盘价 < 入场价 - N×ATR
  - 硬下限：入场价 × 0.97（3% 硬止损，防止 ATR 过小时止损太紧）
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.score.v8.strategy import ScoreV8Strategy
from simulator.base_framework import BaseFramework
from simulator.shared import add_next_open
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_strategy(atr_multiplier: float, take_profit: float = 0.20,
                   tp_rsi_threshold: int = 65,
                   max_positions: int = 10,
                   position_size: float = 0.10) -> ScoreV8Strategy:
    """构建带 ATR 止损的 V8 策略"""
    config = {
        "stop_loss": 0.05,       # 固定止损（ATR 模式时不使用，但仍需传）
        "take_profit": take_profit,
        "tp_rsi_threshold": tp_rsi_threshold,
        "rsi_filter_min": 50,
        "rsi_filter_max": 65,
        "max_positions": max_positions,
        "position_size": position_size,
        "atr_multiplier": atr_multiplier,
        "atr_period": 14,
    }
    return ScoreV8Strategy(config=config)


def run_backtest(
    strategy: ScoreV8Strategy,
    start_date: str,
    end_date: str,
    initial_cash: float = 1_000_000,
    name: str = "v8",
) -> dict:
    """跑单次回测，返回汇总 dict"""
    all_years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))
    framework = BaseFramework(
        initial_cash=initial_cash,
        max_positions=strategy.config.get('max_positions', 10),
        position_size=strategy.config.get('position_size', 0.10),
        state_file=f"/tmp/backtest_{name}.json",
        market_regime_filter=None,
    )
    framework._strategy = strategy
    framework.reset()

    total_trades = []
    n_winning_total = 0
    n_total_total = 0

    for year in all_years:
        y_start = max(f"{year}-01-01", start_date)
        y_end = min(f"{year}-12-31", end_date)
        if y_start > y_end:
            continue

        # ATR仅需 ~14 日 warmup（单年加载即可，前2周用硬止损）
        load_years = [year]
        logger.info(f"  ── {year} 年（数据: {y_start} ~ {y_end}，ATR预热: 单年加载）──")
        try:
            df = load_strategy_data(years=load_years, add_money_flow=True)
        except Exception as e:
            logger.warning(f"  数据加载失败 [{year}]: {e}")
            continue

        df = df[(df["date"] >= y_start) & (df["date"] <= y_end)].copy()
        if df.empty:
            continue
        df = add_next_open(df)

        if hasattr(strategy, 'prepare'):
            try:
                dates_for_prepare = [d for d in sorted(df['date'].unique())
                                     if y_start <= d <= y_end]
                strategy.prepare(dates_for_prepare, df)
            except Exception as prep_e:
                logger.warning(f"  策略 prepare() 失败 [{year}]: {prep_e}")

        dates_in_year = sorted([
            d for d in df['date'].unique()
            if y_start <= d <= y_end
        ])
        logger.info(f"  {year} 年: {len(dates_in_year)} 个交易日")

        class _Market:
            def __init__(self, date):
                self.date = date
                self.next_date = None

        market = _Market(dates_in_year[0] if dates_in_year else y_start)

        for i, date in enumerate(dates_in_year):
            market.date = date
            # _on_day needs next_date in market dict
            _market = {"date": date, "next_date": dates_in_year[i + 1] if i + 1 < len(dates_in_year) else date}
            framework._on_day(date, df, dates_in_year)

            total_value = framework._calc_total_value(df, date)
            from simulator.base_framework import MarketSnapshot
            snap = MarketSnapshot(
                date=date,
                cash=framework.cash,
                total_value=total_value,
                n_positions=len(framework.positions),
                total_return=(total_value / framework.initial_cash - 1) * 100,
            )
            framework.market_snapshots.append(snap.__dict__)

            if (i + 1) % 50 == 0:
                logger.info(f"  {date} ({i+1}/{len(dates_in_year)}): "
                            f"持仓{len(framework.positions)}只, 总值{total_value/10000:.1f}万")

        n_winning_total += framework.n_winning
        n_total_total += framework.n_total
        total_trades.extend(framework.trades)
        framework.trades = []
        framework.n_winning = 0
        framework.n_total = 0
        logger.info(f"  {year} 年完成，持仓{len(framework.positions)}只，现金{framework.cash:,.0f}")

    # ── 汇总统计 ────────────────────────────────────────────────────────
    closed = [t for t in total_trades if t["action"] == "sell"]
    total_pnl = sum(t["pnl"] for t in closed)
    win_rate = n_winning_total / max(n_total_total, 1) * 100

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
    reasons = Counter(t.get("reason", "") for t in closed)

    return {
        "initial_cash": framework.initial_cash,
        "final_value": final_value,
        "total_return_pct": total_return,
        "annual_return_pct": annual_return * 100,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "n_trades": n_total_total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "n_positions": len(framework.positions),
        "exit_reasons": dict(reasons),
    }


def run_sweep():
    """执行 ATR 倍数扫描"""
    results_oos = []   # 样本外 2007-2016
    results_is = []    # 样本内 2018-2025

    atr_multipliers = [1.5, 2.0, 2.5]
    # 同时跑基准（固定5%止损，无ATR）
    run_baseline = True

    # ── 基准组：固定 5% 止损 ─────────────────────────────────────────
    if run_baseline:
        logger.info("=" * 60)
        logger.info("【基准】固定 5% 止损")
        logger.info("=" * 60)
        strat = build_strategy(atr_multiplier=0.0)
        r_oos = run_backtest(strat, "2007-01-01", "2016-12-31", name="v8_baseline_oos")
        r_oos["atr_multiplier"] = "固定5%止损"
        r_oos["mode"] = "OOS"
        results_oos.append(r_oos)
        print_summary("基准(固定5%)", r_oos)

        strat_is = build_strategy(atr_multiplier=0.0)
        r_is = run_backtest(strat_is, "2018-01-01", "2025-12-31", name="v8_baseline_is")
        r_is["atr_multiplier"] = "固定5%止损"
        r_is["mode"] = "IS"
        results_is.append(r_is)
        print_summary("基准(固定5%)", r_is)

    # ── ATR 倍数扫描 ─────────────────────────────────────────────────
    for atr_mult in atr_multipliers:
        logger.info("=" * 60)
        logger.info(f"【OOS】ATR × {atr_mult}")
        logger.info("=" * 60)
        strat = build_strategy(atr_multiplier=atr_mult)
        r_oos = run_backtest(strat, "2007-01-01", "2016-12-31", name=f"v8_atr{atr_mult}_oos")
        r_oos["atr_multiplier"] = f"ATR×{atr_mult}"
        r_oos["mode"] = "OOS"
        results_oos.append(r_oos)
        print_summary(f"ATR×{atr_mult}(OOS)", r_oos)

        logger.info("=" * 60)
        logger.info(f"【IS】ATR × {atr_mult}")
        logger.info("=" * 60)
        strat_is = build_strategy(atr_multiplier=atr_mult)
        r_is = run_backtest(strat_is, "2018-01-01", "2025-12-31", name=f"v8_atr{atr_mult}_is")
        r_is["atr_multiplier"] = f"ATR×{atr_mult}"
        r_is["mode"] = "IS"
        results_is.append(r_is)
        print_summary(f"ATR×{atr_mult}(IS)", r_is)

    return results_oos, results_is


def print_summary(label, r):
    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print(f"  年化收益: {r['annual_return_pct']:+.2f}%")
    print(f"  总收益:   {r['total_return_pct']:+.2f}%")
    print(f"  最大回撤: {r['max_drawdown_pct']:.2f}%")
    print(f"  夏普比率: {r['sharpe']:.2f}")
    print(f"  总交易:   {r['n_trades']} 笔")
    print(f"  胜率:     {r['win_rate']:.1f}%")
    if r.get('exit_reasons'):
        print(f"  出场分布: {r['exit_reasons']}")
    print(f"{'=' * 54}\n")


def write_report(results_oos, results_is, output_path: str):
    """写 Markdown 报告"""
    lines = [
        "# V8 ATR止损参数样本外验证",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | **验证框架**: Oracle V8",
        "",
        "## 参数扫描矩阵",
        "",
        "| 配置 | 年化收益 | 总收益 | 最大回撤 | 夏普 | 交易数 | 胜率 | 样本 |",
        "|------|---------|--------|---------|-----|--------|------|------|",
    ]

    # OOS
    for r in results_oos:
        lines.append(
            f"| {r['atr_multiplier']} | {r['annual_return_pct']:+.2f}% | "
            f"{r['total_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | "
            f"{r['sharpe']:.2f} | {r['n_trades']} | {r['win_rate']:.1f}% | OOS(07-16) |"
        )

    # IS
    for r in results_is:
        lines.append(
            f"| {r['atr_multiplier']} | {r['annual_return_pct']:+.2f}% | "
            f"{r['total_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | "
            f"{r['sharpe']:.2f} | {r['n_trades']} | {r['win_rate']:.1f}% | IS(18-25) |"
        )

    lines += ["", "## 出场原因分布", ""]

    # OOS 出场原因
    lines.append("### 样本外 (2007-2016)")
    for r in results_oos:
        reasons_str = str(r.get('exit_reasons', {}))
        lines.append(f"- **{r['atr_multiplier']}**: {reasons_str}")

    lines.append("")
    lines.append("### 样本内 (2018-2025)")
    for r in results_is:
        reasons_str = str(r.get('exit_reasons', {}))
        lines.append(f"- **{r['atr_multiplier']}**: {reasons_str}")

    # OOS vs IS 对比
    lines += [
        "",
        "## OOS vs IS 对比（过拟合检验）",
        "",
        "| 配置 | OOS年化 | IS年化 | OOS夏普 | IS夏普 | 年化差异 |",
        "|------|---------|--------|---------|--------|---------|",
    ]

    oos_dict = {r['atr_multiplier']: r for r in results_oos}
    for r_is in results_is:
        am = r_is['atr_multiplier']
        if am in oos_dict:
            r_oos = oos_dict[am]
            diff = r_is['annual_return_pct'] - r_oos['annual_return_pct']
            lines.append(
                f"| {am} | {r_oos['annual_return_pct']:+.2f}% | "
                f"{r_is['annual_return_pct']:+.2f}% | {r_oos['sharpe']:.2f} | "
                f"{r_is['sharpe']:.2f} | {diff:+.2f}% |"
            )

    # 最优推荐
    oos_with_returns = [r for r in results_oos if 'ATR' in r['atr_multiplier']]
    if oos_with_returns:
        best = max(oos_with_returns, key=lambda x: x['sharpe'])
        lines += [
            "",
            "## 最优推荐",
            "",
            f"**样本外最优**: {best['atr_multiplier']}（夏普 {best['sharpe']:.2f}，年化 {best['annual_return_pct']:+.2f}%，最大回撤 {best['max_drawdown_pct']:.2f}%）",
            "",
            "**推荐逻辑**: ATR动态止损根据市场波动率自适应调整止损位，",
            f"相比固定5%止损，在波动大时提供更宽的保护，在波动小时避免被震荡洗出。",
            "",
            "**过拟合检验**: 对比样本内(2018-2025)结果，年化收益差异在可接受范围内即视为无严重过拟合。",
        ]

    Path(output_path).write_text("\n".join(lines))
    logger.info(f"\n报告已写入: {output_path}")


if __name__ == "__main__":
    logger.info("开始 ATR 止损参数扫描...")
    output = "/root/.openclaw/workspace/memory/v8_atr_stop_loss_oos_validation.md"
    results_oos, results_is = run_sweep()
    write_report(results_oos, results_is, output)
    logger.info("全部完成!")
