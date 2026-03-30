#!/usr/bin/env python3
"""
双策略模拟盘 — v4(原始) vs v8(IC增强)
使用 simulator/ 模块统一调度

用法:
    python3 paper_trading_sim.py                          # 2026年初至今（默认）
    python3 paper_trading_sim.py --date 2026-03-01      # 从指定日期运行
    python3 paper_trading_sim.py --start-year 2008 --end-year 2026  # 18年回测
"""

import os, sys, gc, json, argparse
import pandas as pd
import numpy as np
from datetime import datetime

# === 策略导入 ===
from simulator import MultiSimulator
from strategies.score.v4.strategy import ScoreV4Strategy
from strategies.score.v8.strategy import ScoreV8Strategy

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/paper_trading'
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_CASH = 1_000_000  # 各100万
SIM_START = '2026-01-01'
DEFAULT_START_YEAR = 2026
DEFAULT_END_YEAR = 2026


def load_data(year: int, start_date: str = None) -> pd.DataFrame:
    """
    加载指定年份的数据，并完成所有指标预处理。
    返回的 DataFrame 包含完整选股条件列，可直接供策略使用。
    """
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()

    # === 基础指标 ===
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)

    # next_open 用于模拟盘次日开盘买入/卖出
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['next_close'] = df.groupby('symbol')['close'].shift(-1)

    # 处理合并后的重名字段（pandas 默认用 _x/_y）
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']

    # === v4 选股条件列 ===
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100
    df['growth'] = df['growth'].round(1)
    df['growth_condition'] = (df['close'] >= df['open']) & (df['high'] <= df['open'] * 1.06)
    df['volume_condition'] = (
        (df['volume'] > df['volume'].shift(1) * 1.5) |
        (df['volume'] > df['vol_ma5'] * 1.2)
    )
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )
    df['jc_condition'] = (
        (df['sma_5'] > df['sma_5'].shift(1)) &
        (df['sma_20'] > df['sma_20'].shift(1))
    )
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)

    # v4 角度条件（SMA10 变化率，用于评分）
    df['_sma10_change'] = (df['sma_10'] / df['sma_10'].shift(1) - 1).abs().clip(0, 0.1)
    df['angle_ma_10'] = df['_sma10_change'] * 10 * (180 / np.pi)  # 近似角度

    # 按日期过滤
    if start_date:
        df = df[df['date'] >= start_date].sort_values('date').reset_index(drop=True)
    else:
        df = df.sort_values('date').reset_index(drop=True)

    return df


def run_sim_for_year(year: int, start_date: str = None, strategies=None):
    """
    对单一年份运行多策略模拟盘。

    Args:
        year: 年份
        start_date: 可选，起始日期（YYYY-MM-DD）
        strategies: 可选策略列表，默认 ['V4', 'V8']

    Returns:
        (sim: MultiSimulator, dates: list, df: pd.DataFrame)
    """
    if strategies is None:
        strategies = ['V4', 'V8']

    df = load_data(year, start_date)
    dates = sorted(df['date'].unique())

    if not dates:
        return None, [], None

    # --- 构建 simulator ---
    sim = MultiSimulator(initial_cash_per_strategy=INITIAL_CASH)

    if 'V4' in strategies:
        sim.add_strategy(
            "V4",
            ScoreV4Strategy(),
            config={
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'max_positions': 5,
                'position_size': 0.20,
            },
        )

    if 'V8' in strategies:
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

    # --- 运行 ---
    sim.run(
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        daily_data=df,
    )

    return sim, dates, df


def run_sim(start_date: str = None, start_year: int = None, end_year: int = None):
    """
    运行多策略模拟盘。

    多 year 模式（start_year/end_year 指定）：
      逐年运行，逐年打印结果，最终汇总。
    单 year 模式（start_date 指定）：
      运行指定日期范围，打印完整持仓和汇总。
    """
    # 确定运行模式
    multi_year = (start_year is not None) and (end_year is not None)

    if multi_year:
        years = list(range(start_year, end_year + 1))
        print(f"{'='*70}")
        print(f"📈 多策略模拟盘 — V4 vs V8 ({start_year}-{end_year}, 共{len(years)}年)")
        print(f"{'='*70}")

        all_results = {}
        year_summaries = []

        for year in years:
            try:
                sim, dates, _ = run_sim_for_year(year, start_date=None)
            except FileNotFoundError as e:
                print(f"\n⚠️ {year}年数据不存在，跳过: {e}")
                continue

            if sim is None:
                continue

            print(f"\n{'='*60}")
            print(f"📅 {year} 年结果 ({len(dates)}个交易日)")
            print(f"{'='*60}")

            sim.print_summary()

            results = sim.get_results()
            for strat, res in results.items():
                if strat not in all_results:
                    all_results[strat] = []
                all_results[strat].append({
                    'year': year,
                    **res,
                })

        # 汇总多年结果
        if all_results:
            print(f"\n{'='*70}")
            print(f"🏆 多年汇总 ({start_year}-{end_year})")
            print(f"{'='*70}")
            print(f"{'策略':<8}{'年份':<6}{'总收益':>10}{'最终市值':>12}{'交易次数':>10}{'胜率':>8}")
            print("-"*60)

            for strat in ['V4', 'V8']:
                if strat not in all_results:
                    continue
                strat_results = all_results[strat]
                for r in strat_results:
                    print(f"{strat:<8}{r['year']:<6}"
                          f"{r['total_return_pct']:>10}"
                          f"{r['final_value']/1e4:>10.1f}万"
                          f"{r['n_trades']:>10}"
                          f"{r['win_rate']:>7.1f}%")

            # 汇总行
            print("-"*60)
            for strat in ['V4', 'V8']:
                if strat not in all_results:
                    continue
                strat_results = all_results[strat]
                total_ret = 1.0
                for r in strat_results:
                    total_ret *= (1 + r['total_return'] / 100)
                final_val = strat_results[-1]['final_value'] if strat_results else INITIAL_CASH
                total_trades = sum(r['n_trades'] for r in strat_results)
                print(f"{strat:<8}{'(合计)':<6}"
                      f"{(total_ret-1)*100:>+9.1f}%"
                      f"{final_val/1e4:>10.1f}万"
                      f"{total_trades:>10}笔")

            # 保存
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_file = f'{OUT_DIR}/backtest_{start_year}_{end_year}_{ts}.json'
            flat_results = {
                strat: [
                    {k: v for k, v in r.items() if k != 'trades'}
                    for r in results
                ]
                for strat, results in all_results.items()
            }
            with open(out_file, 'w') as f:
                json.dump(flat_results, f, indent=2, default=str)
            print(f"\n💾 结果已保存: {out_file}")

        return all_results

    else:
        # 单年模式（原始行为）
        if start_year is None:
            start_year = DEFAULT_START_YEAR
        if end_year is None:
            end_year = DEFAULT_END_YEAR

        print(f"{'='*70}")
        print(f"📈 双策略模拟盘 — V4 vs V8 ({start_year}年)")
        print(f"{'='*70}")

        sim, dates, df = run_sim_for_year(start_year, start_date=start_date)
        if sim is None:
            print("没有可用数据")
            return

        # --- 汇总输出 ---
        print(f"\n{'='*70}")
        print(f"📊 双策略模拟盘汇总")
        print(f"{'='*70}")

        summary = sim.get_summary()
        print(summary.to_string(index=False))

        results = sim.get_results()

        # --- 打印持仓 ---
        print(f"\n{'='*70}")
        print(f"📦 当前持仓")
        print(f"{'='*70}")
        latest_date = str(dates[-1])
        latest_prices = df[df['date'] == latest_date].set_index('symbol')['close'].to_dict()

        for strat in ['V4', 'V8']:
            res = results.get(strat, {})
            positions = res.get('positions', [])
            if positions:
                print(f"\n【{strat.upper()}】")
                for pos in positions:
                    sym = pos['symbol']
                    qty = pos['qty']
                    avg_cost = pos['avg_cost']
                    cur_price = latest_prices.get(sym, avg_cost)
                    pnl_pct = (cur_price / avg_cost - 1) * 100
                    print(
                        f"  {sym}: {qty}股 @买入{avg_cost:.2f} → "
                        f"现价{cur_price:.2f}({pnl_pct:+.1f}%)"
                    )

        # --- 保存结果 ---
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = f'{OUT_DIR}/paper_trading_{ts}.json'
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 结果已保存: {out_file}")

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='双策略模拟盘（V4 vs V8）')
    parser.add_argument(
        '--date',
        default=None,
        help='起始日期 YYYY-MM-DD（默认 2026-01-01）'
    )
    parser.add_argument(
        '--start-year', type=int, default=None,
        help='起始年份（多 year 模式，如 --start-year 2008 --end-year 2026）'
    )
    parser.add_argument(
        '--end-year', type=int, default=None,
        help='结束年份'
    )
    args = parser.parse_args()

    run_sim(
        start_date=args.date,
        start_year=args.start_year,
        end_year=args.end_year,
    )
