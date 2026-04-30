#!/usr/bin/env python3
"""
Walk-Forward 验证框架 v4
=======================
轻量版：每个窗口独立加载需要的数据子集，不预先合并全部数据，节省内存。

用法：
    python3 walk_forward.py --strategy v8
    python3 walk_forward.py --strategy v8 --start-year 2023 --end-year 2025
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework
from simulator.shared import load_strategy, add_next_open as shared_add_next_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_total_return: float
    train_n_trades: int
    train_win_rate: float
    oos_total_return: float
    oos_n_trades: int
    oos_win_rate: float
    oos_max_drawdown: float
    wfe: float


def _load_for_window(
    strategy_name: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """按年份加载数据子集（money_flow 缓存会复用）"""
    from utils.data_loader import load_strategy_data
    add_mf = "wavechan_v3" not in strategy_name
    years = list(range(start_year, end_year + 1))
    df = load_strategy_data(years=years, add_money_flow=add_mf)
    df = shared_add_next_open(df)
    return df


def _backtest_range(
    df: pd.DataFrame,
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_cash: float = 500_000,
) -> Optional[dict]:
    """对 df 切片运行回测"""
    try:
        window_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        if window_df.empty:
            return None

        framework = BaseFramework(initial_cash=initial_cash)
        strategy = load_strategy(strategy_name)

        dates = sorted(window_df["date"].unique())
        if len(dates) < 20:
            return None

        if hasattr(strategy, 'prepare'):
            strategy.prepare(dates, window_df)

        framework.run_backtest(strategy=strategy, df=window_df, start_date=start_date, end_date=end_date)

        n_winning = framework.n_winning
        n_total = framework.n_total
        win_rate = n_winning / max(n_total, 1)

        final_value = framework.cash + sum(
            pos["latest_price"] * pos["qty"] for pos in framework.positions.values()
        )
        total_return = (final_value / initial_cash - 1) * 100

        values = [s["total_value"] for s in framework.market_snapshots]
        peak = initial_cash
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        return {
            "total_return": total_return,
            "n_trades": n_total,
            "win_rate": win_rate,
            "max_drawdown": abs(max_dd),
        }
    except Exception as e:
        logger.warning(f"  回测失败 {start_date}~{end_date}: {e}")
        return None


def run_walk_forward(
    strategy_name: str,
    start_year: int = 2023,
    end_year: int = 2025,
    train_years: int = 1,
    test_months: int = 3,
    initial_cash: float = 500_000,
) -> pd.DataFrame:
    from dateutil.relativedelta import relativedelta

    logger.info(f"{'='*60}")
    logger.info(f"  Walk-Forward 验证: {strategy_name}")
    logger.info(f"  训练窗口: {train_years}年  |  测试窗口: {test_months}个月")
    logger.info(f"  数据范围: {start_year} ~ {end_year}")
    logger.info(f"{'='*60}")

    results: List[WindowResult] = []
    train_start_dt = datetime.date(start_year, 1, 1)
    max_train_start_dt = datetime.date(end_year, 1, 1) - relativedelta(years=train_years, months=test_months)

    window_num = 0
    while train_start_dt <= max_train_start_dt:
        train_end_dt = train_start_dt + relativedelta(years=train_years) - relativedelta(days=1)
        test_start_dt = train_end_dt + relativedelta(days=1)
        test_end_dt = test_start_dt + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end_dt.year > end_year:
            break

        train_start_str = train_start_dt.strftime("%Y-%m-%d")
        train_end_str = train_end_dt.strftime("%Y-%m-%d")
        test_start_str = test_start_dt.strftime("%Y-%m-%d")
        test_end_str = test_end_dt.strftime("%Y-%m-%d")

        # 计算这个窗口需要加载的数据年份范围
        load_start_year = max(train_start_dt.year, start_year)
        load_end_year = min(test_end_dt.year, end_year)

        window_num += 1
        logger.info(f"\n[{window_num}] 训练 {train_start_str} ~ {train_end_str} | 测试 {test_start_str} ~ {test_end_str}")

        # 加载这个窗口需要的数据
        t_load = time.time()
        df = _load_for_window(strategy_name, load_start_year, load_end_year)
        logger.info(f"  数据加载: {len(df):,}行, {time.time()-t_load:.0f}s")

        # 训练集回测
        t_train = time.time()
        train_res = _backtest_range(df, strategy_name, train_start_str, train_end_str, initial_cash)
        train_time = time.time() - t_train

        del df
        gc.collect()

        if train_res is None:
            train_start_dt = train_start_dt + relativedelta(months=test_months)
            continue

        # OOS 测试集回测（重新加载，因为数据可能不同年份）
        t_load2 = time.time()
        df2 = _load_for_window(strategy_name, load_start_year, load_end_year)
        logger.info(f"  数据加载: {len(df2):,}行, {time.time()-t_load2:.0f}s")

        t_oos = time.time()
        oos_res = _backtest_range(df2, strategy_name, test_start_str, test_end_str, initial_cash)
        oos_time = time.time() - t_oos

        del df2
        gc.collect()

        if oos_res is None:
            train_start_dt = train_start_dt + relativedelta(months=test_months)
            continue

        is_ret = train_res["total_return"]
        oos_ret = oos_res["total_return"]
        wfe = oos_ret / abs(is_ret) if is_ret != 0 else 0.0

        result = WindowResult(
            train_start=train_start_str,
            train_end=train_end_str,
            test_start=test_start_str,
            test_end=test_end_str,
            train_total_return=round(is_ret, 2),
            train_n_trades=train_res["n_trades"],
            train_win_rate=round(train_res["win_rate"] * 100, 1),
            oos_total_return=round(oos_ret, 2),
            oos_n_trades=oos_res["n_trades"],
            oos_win_rate=round(oos_res["win_rate"] * 100, 1),
            oos_max_drawdown=round(oos_res["max_drawdown"], 2),
            wfe=round(wfe, 3),
        )
        results.append(result)

        logger.info(
            f"  IS: {is_ret:+.1f}% ({train_res['n_trades']}笔, 胜{train_res['win_rate']*100:.0f}%) | "
            f"OOS: {oos_ret:+.1f}% ({oos_res['n_trades']}笔) | "
            f"回撤{oos_res['max_drawdown']:.1f}% | WFE={wfe:.2f} | 耗时{train_time+oos_time:.0f}s"
        )

        train_start_dt = train_start_dt + relativedelta(months=test_months)

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame([r.__dict__ for r in results])

    logger.info(f"\n{'='*60}")
    logger.info(f"  Walk-Forward 汇总 ({len(df_results)} 个窗口)")
    logger.info(f"{'='*60}")
    logger.info(f"  IS 平均收益:  {df_results['train_total_return'].mean():+.1f}%")
    logger.info(f"  OOS 平均收益: {df_results['oos_total_return'].mean():+.1f}%")
    logger.info(f"  OOS 平均胜率: {df_results['oos_win_rate'].mean():.0f}%")
    logger.info(f"  OOS 平均回撤: {df_results['oos_max_drawdown'].mean():.1f}%")
    wfe_pos = (df_results['wfe'] > 0).sum()
    wfe_robust = (df_results['wfe'] > 0.5).sum()
    logger.info(f"  WFE > 0 的窗口: {wfe_pos}/{len(df_results)}  |  WFE > 0.5: {wfe_robust}/{len(df_results)}")
    oos_positive = (df_results['oos_total_return'] > 0).sum()
    logger.info(f"  OOS 正收益窗口: {oos_positive}/{len(df_results)}")
    logger.info(f"{'='*60}")

    return df_results


def print_table(df: pd.DataFrame):
    if df.empty:
        return
    header = f"{'窗口':^22} | {'IS收益':>7} | {'IS笔':>4} | {'IS胜':>4} | {'OOS收益':>7} | {'OOS笔':>4} | {'OOS胜':>4} | {'OOS回撤':>6} | {'WFE':>5}"
    print(f"\n{header}")
    print("-" * 85)
    for _, r in df.iterrows():
        period = f"{r['train_start'][:7]}~{r['test_end'][:7]}"
        print(
            f"{period:^22} | {r['train_total_return']:>+6.1f}% | "
            f"{r['train_n_trades']:>4} | {r['train_win_rate']:>3.0f}% | "
            f"{r['oos_total_return']:>+6.1f}% | {r['oos_n_trades']:>4} | "
            f"{r['oos_win_rate']:>3.0f}% | {r['oos_max_drawdown']:>5.1f}% | {r['wFE']:>5.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 策略验证")
    parser.add_argument("--strategy", default="v8", choices=["v8", "wavechan_v3_strict"])
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--train-years", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    df = run_walk_forward(
        strategy_name=args.strategy,
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        test_months=args.test_months,
    )

    if df.empty:
        print("无结果")
        return

    print_table(df)

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
