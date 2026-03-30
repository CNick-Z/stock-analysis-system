#!/usr/bin/env python3
"""
Score 策略 v8 回测 — v6核心 + IC 增强过滤
============================================

与 paper_trading_sim.py 共用同一 ScoreV8Strategy 类。

策略逻辑完全由 strategies.score.v8.strategy.ScoreV8Strategy 提供：
  - 选股条件、IC增强过滤、评分、出场判断
  - 出场规则：止损5% / 止盈15% / MA死叉
  - 持仓上限：5只，单只仓位上限20%

数据加载每年独立（与 paper_trading_sim.py 保持一致），
回测循环通过 MultiSimulator 调度，与模拟盘共用同一引擎。

用法:
    python3 backtest_score_v8.py
"""

import os, sys, gc, argparse, psutil
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import MultiSimulator
from strategies.score.v8.strategy import ScoreV8Strategy

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/backtest_results'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 1_000_000   # 与模拟盘一致
DEFAULT_YEARS = range(2018, 2026)


def load_year_data(year: int) -> pd.DataFrame:
    """
    加载某年的技术指标+日线数据，并完成所有指标预处理。
    返回的 DataFrame 包含完整选股条件列，可直接供 ScoreV8Strategy 使用。
    """
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()

    # === 基础指标（MultiSimulator 内部也需要这些列） ===
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)

    # next_open 用于模拟次日开盘买入/卖出
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)

    # 处理合并后的重名字段（pandas 默认用 _x/_y 后缀）
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']

    df = df.sort_values('date').reset_index(drop=True)
    return df


def run_backtest_year(year: int) -> dict:
    """
    使用 MultiSimulator + ScoreV8Strategy 回测单一年份。

    返回与原 run_backtest_year() 相同的统计字典结构：
      year, annual_return, n_buys, n_sells, n_winners, win_rate,
      total_value, trades
    """
    print(f"\n{'='*60}")
    print(f"📅 {year} 年回测 (V8 + MultiSimulator)")
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] MEM={mem:.2f}GB | {year}开始")

    df = load_year_data(year)
    print(f"  总记录: {len(df):,}")

    dates = sorted(df['date'].unique())
    if not dates:
        return {'year': year, 'annual_return': 0, 'n_buys': 0, 'n_sells': 0,
                'n_winners': 0, 'win_rate': 0, 'total_value': INITIAL_CASH, 'trades': []}

    # === 构建 MultiSimulator ===
    sim = MultiSimulator(initial_cash_per_strategy=INITIAL_CASH)
    sim.add_strategy(
        "V8",
        ScoreV8Strategy(),
        config={
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'max_positions': 5,
            'position_size': 0.20,
        },
    )

    # === 运行 ===
    sim.run(
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        daily_data=df,
    )

    # === 提取结果 ===
    results = sim.get_results()
    v8_res = results.get("V8", {})

    trades = v8_res.get('trades', [])
    closed_trades = [t for t in trades if t['action'] == 'sell']
    winners = [t for t in closed_trades if t.get('pnl', 0) > 0]

    initial = v8_res.get('initial_cash', INITIAL_CASH)
    final = v8_res.get('final_value', initial)
    annual_return = (final - initial) / initial if initial > 0 else 0

    n_buys = sum(1 for t in trades if t['action'] == 'buy')
    n_sells = len(closed_trades)

    print(f"\n  📊 {year} 结果: 收益={annual_return:.2%} 买入={n_buys} 胜率={len(winners)/max(1,n_sells):.0%}")

    del df
    gc.collect()

    return {
        'year': year,
        'annual_return': annual_return,
        'n_buys': n_buys,
        'n_sells': n_sells,
        'n_winners': len(winners),
        'win_rate': len(winners) / max(1, n_sells),
        'total_value': final,
        'trades': trades,
    }


def run(years=DEFAULT_YEARS):
    """运行多年回测并打印汇总表"""
    print(f"{'='*60}")
    print(f"🏃 Score v8 回测 — v6核心 + IC增强 (MultiSimulator驱动)")
    print(f"{'='*60}")

    results = []
    for year in years:
        try:
            r = run_backtest_year(year)
            results.append(r)
        except FileNotFoundError as e:
            print(f"\n⚠️ {year}年数据不存在，跳过: {e}")
            continue

    if not results:
        print("没有可用的回测结果")
        return results

    # 汇总
    total_ret = 1.0
    for r in results:
        total_ret *= (1 + r['annual_return'])

    total_wins = sum(r['n_winners'] for r in results)
    total_sells = sum(r['n_sells'] for r in results)

    print(f"\n{'='*60}")
    print(f"🏆 v8 汇总 (2018-2025)")
    print(f"{'='*60}")
    print(f"{'年份':<8}{'收益':>10}{'买入':>6}{'卖出':>6}{'胜率':>8}")
    print("-"*45)
    for r in results:
        print(f"{r['year']:<8}{r['annual_return']:>10.2%}{r['n_buys']:>6}{r['n_sells']:>6}{r['win_rate']:>8.0%}")
    print("-"*45)
    print(f"{'合计':<8}{(total_ret-1):>10.2%}")
    print(f"最终: {results[-1]['total_value']/10000:.2f}万 vs 100万")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score v8 回测')
    parser.add_argument('--years', type=str, default='2018-2025',
                        help='年份范围，如 2018-2025')
    args = parser.parse_args()

    if '-' in args.years:
        start, end = args.years.split('-')
        years = range(int(start), int(end) + 1)
    else:
        years = DEFAULT_YEARS

    run(years=years)
