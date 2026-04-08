#!/usr/bin/env python3
"""单次运行脚本 - SL/TP参数验证"""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from simulator import MultiSimulator
from strategies.score.v8.strategy import ScoreV8Strategy

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
SL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
TP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15

class SimV8(ScoreV8Strategy):
    def __init__(self, sl, tp):
        super().__init__()
        self.stop_loss = sl
        self.take_profit = tp

# 加载缓存（只加载一次）
import pickle
CACHE = '/tmp/sim_data_2024_2025.pkl'
if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
else:
    dfs = []
    for y in [2024, 2025]:
        tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={y}/data.parquet')
        daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={y}/data.parquet')
        df_ = pd.merge(tech, daily, on=['date','symbol'], how='inner')
        df_['vol_ratio'] = df_['volume'] / (df_['vol_ma5'] + 1e-10)
        df_['next_open'] = df_.groupby('symbol')['open'].shift(-1)
        if 'turnover_rate_y' in df_.columns: df_['turnover_rate'] = df_['turnover_rate_y']
        elif 'turnover_rate_x' in df_.columns: df_['turnover_rate'] = df_['turnover_rate_x']
        dfs.append(df_)
        del tech, daily; gc.collect()
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(CACHE)

sim = MultiSimulator(initial_cash_per_strategy=1_000_000)
sim.add_strategy(f'v8', SimV8(SL, TP))
sim.run('2024-01-01', '2025-12-31', daily_data=df)
r = sim.get_summary().iloc[0]

def pct(v):
    try: return f"{float(v):+.2%}"
    except: return str(v)

print(f"SL={SL:.2f} TP={TP:.2f} | ret={pct(r['total_return'])} dd={pct(r['max_drawdown_pct'])} trades={int(r['n_trades'])} win={float(str(r['win_rate']).replace('%','')):5.1%}")
