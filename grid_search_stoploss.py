#!/usr/bin/env python3
"""
Stop Loss / Take Profit 网格扫描 — 2024-2025年快速验证
用 IC/IR 工具的逻辑：对每个参数档位跑回测，对比收益
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np, gc
from simulator import MultiSimulator
from strategies.score.v8.strategy import ScoreV8Strategy

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'

def load_data(years):
    dfs = []
    for y in years:
        tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={y}/data.parquet')
        daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={y}/data.parquet')
        df = pd.merge(tech, daily, on=['date','symbol'], how='inner')
        df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
        df['next_open'] = df.groupby('symbol')['open'].shift(-1)
        if 'turnover_rate_y' in df.columns: df['turnover_rate'] = df['turnover_rate_y']
        elif 'turnover_rate_x' in df.columns: df['turnover_rate'] = df['turnover_rate_x']
        dfs.append(df)
        del tech, daily; gc.collect()
    return pd.concat(dfs, ignore_index=True)

if __name__ == '__main__':
    print("加载2024-2025数据...")
    df = load_data([2024, 2025])
    print(f"数据: {len(df)}行")

    stop_losses = [0.03, 0.04, 0.05, 0.06]
    take_profits = [0.10, 0.15, 0.20, 0.25]

    print(f"\n{'='*60}")
    print(f"{'止损档位':<8} {'止盈档位':<8} {'收益':<10} {'最大回撤':<10} {'交易数':<6} {'胜率':<6}")
    print(f"{'='*60}")

    best = None
    best_ret = -999

    for sl in stop_losses:
        for tp in take_profits:
            # 创建策略实例，动态修改止损止盈
            class ConfigurableV8(ScoreV8Strategy):
                def __init__(self, sl, tp):
                    super().__init__()
                    self.stop_loss = sl
                    self.take_profit = tp

            strat = ConfigurableV8(sl, tp)
            sim = MultiSimulator(initial_cash_per_strategy=1_000_000)
            sim.add_strategy(f'v8_sl{sl}_tp{tp}', strat)
            sim.run('2024-01-01', '2025-12-31', daily_data=df)

            r = sim.get_summary().iloc[0]
            ret = float(r['total_return'])
            dd = r['max_drawdown_pct']
            trades = int(r['n_trades'])
            win = float(str(r['win_rate']).replace('%',''))

            print(f"SL={sl:<5} TP={tp:<5} {ret:+8.2%}  {dd:<10} {trades:<6} {win:5.1%}")

            if ret > best_ret:
                best_ret = ret
                best = (sl, tp, ret, dd, trades, win)

    print(f"{'='*60}")
    sl, tp, ret, dd, trades, win = best
    print(f"\n★ 最优: SL={sl} TP={tp}  收益={ret:+.2%}  回撤={dd}  交易={trades}笔")
