#!/usr/bin/env python3
"""
V8 ATR止损 单配置验证脚本
==========================
接收命令行参数：ATR倍数 (1.5 / 2.0 / 2.5)
跑 样本外(2007-2016) + 样本内(2018-2025)
结果追加到 /tmp/atr_results.json，脚本退出（释放内存）

用法：
  python backtest_atr_single.py 1.5
  python backtest_atr_single.py 2.0
  python backtest_atr_single.py 2.5
"""

import sys
import json
import gc
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


def build_strategy(atr_multiplier: float,
                    take_profit: float = 0.20,
                    tp_rsi_threshold: int = 65,
                    max_positions: int = 10,
                    position_size: float = 0.10) -> ScoreV8Strategy:
    config = {
        "stop_loss": 0.05,
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

        load_years = [year]
        logger.info(f"  ── {year} 年（数据: {y_start} ~ {y_end}）──")

        try:
            df = load_strategy_data(years=load_years, add_money_flow=True)
        except Exception as e:
            logger.warning(f"  数据加载失败 [{year}]: {e}")
            continue

        df = df[(df["date"] >= y_start) & (df["date"] <= y_end)].copy()
        if df.empty:
            del df
            gc.collect()
            continue

        df = add_next_open(df)

        # ATR prepare (only if ATR enabled)
        if hasattr(strategy, 'prepare') and strategy.atr_multiplier > 0:
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
        logger.info(f"  {year} 年: {len(dates_in_year)} 个交易日，{len(df)} 行")

        for i, date in enumerate(dates_in_year):
            market_dict = {"date": date, "next_date": dates_in_year[i + 1] if i + 1 < len(dates_in_year) else date}
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

        # 释放今年数据
        del df
        gc.collect()

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python backtest_atr_single.py <atr_multiplier>")
        sys.exit(1)

    atr_mult = float(sys.argv[1])
    logger.info(f"=" * 60)
    logger.info(f"开始 ATR × {atr_mult} 验证")
    logger.info(f"=" * 60)

    results = {
        "atr_multiplier": atr_mult,
        "timestamp": datetime.now().isoformat(),
        "oos_2007_2016": None,
        "is_2018_2025": None,
    }

    # ── 样本外 2007-2016 ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"【OOS】ATR × {atr_mult} | 2007-2016")
    logger.info(f"{'='*60}")
    strat_oos = build_strategy(atr_multiplier=atr_mult)
    r_oos = run_backtest(strat_oos, "2007-01-01", "2016-12-31", name=f"v8_atr{atr_mult}_oos")
    r_oos["atr_multiplier"] = f"ATR×{atr_mult}"
    r_oos["mode"] = "OOS"
    results["oos_2007_2016"] = r_oos
    print_summary(f"ATR×{atr_mult}(OOS 07-16)", r_oos)

    # 释放内存
    del strat_oos
    gc.collect()

    # ── 样本内 2018-2025 ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"【IS】ATR × {atr_mult} | 2018-2025")
    logger.info(f"{'='*60}")
    strat_is = build_strategy(atr_multiplier=atr_mult)
    r_is = run_backtest(strat_is, "2018-01-01", "2025-12-31", name=f"v8_atr{atr_mult}_is")
    r_is["atr_multiplier"] = f"ATR×{atr_mult}"
    r_is["mode"] = "IS"
    results["is_2018_2025"] = r_is
    print_summary(f"ATR×{atr_mult}(IS 18-25)", r_is)

    del strat_is
    gc.collect()

    # ── 追加到 JSON ─────────────────────────────────────────────────
    json_path = Path("/tmp/atr_results.json")
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    else:
        existing = []

    existing.append(results)
    json_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.info(f"结果已追加到 {json_path}")

    logger.info(f"ATR × {atr_mult} 完成，进程即将退出（内存释放）")
