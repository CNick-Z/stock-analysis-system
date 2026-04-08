#!/usr/bin/env python3
"""跟踪 _process_buys 中的实际 position_limit"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pandas as pd
import numpy as np
from simulator.base_framework import BaseFramework
from simulator.market_regime import MarketRegimeFilter
from simulator.shared import load_strategy
from utils.data_loader import load_strategy_data

# Patch _process_buys to log regime
original_process_buys = BaseFramework._process_buys

def patched_process_buys(self, full_df, daily, market):
    position_limit = 1.0
    if self.market_regime_filter:
        regime_info = self.market_regime_filter.get_regime(market["date"])
        position_limit = regime_info["position_limit"]
        if market["date"] in ['2022-01-04', '2022-03-15', '2022-06-01', '2022-10-13', '2022-11-01']:
            print(f"  DEBUG {market['date']}: locked={regime_info['regime']}, raw={regime_info['raw_regime']}, "
                  f"limit={position_limit:.2f}, RSI={regime_info['rsi14']:.1f}, "
                  f"consec={regime_info['consecutive_regime_days']}")
    return original_process_buys(self, full_df, daily, market)

BaseFramework._process_buys = patched_process_buys

# Run the 2020-2022 backtest with filter
print("=" * 80)
print("运行 2020-2022 回测（market_filter=True）")
print("=" * 80)

from simulator.base_framework import BaseFramework
from backtest import load_data_for_strategy

strategy = load_strategy('v8')
market_regime_filter = MarketRegimeFilter(confirm_days=1, neutral_position=0.70, bear_position=0.30)

# 一次性 prepare
print("\n>>> prepare(2020-01-01, 2022-12-31)")
market_regime_filter.prepare('2020-01-01', '2022-12-31')

# Test get_regime at key dates
print("\n>>> 关键日期 get_regime:")
for d in ['2022-01-04', '2022-03-15', '2022-06-01', '2022-10-13', '2022-11-01']:
    r = market_regime_filter.get_regime(d)
    print(f"  {d}: locked={r['regime']}, raw={r['raw_regime']}, "
          f"limit={r['position_limit']:.2f}, consec={r['consecutive_regime_days']}, "
          f"df_consec={r['consecutive_days']}")

framework = BaseFramework(
    initial_cash=1000000,
    state_file="/tmp/backtest_v8.json",
    market_regime_filter=market_regime_filter,
)
framework._strategy = strategy
framework.reset()

# 2022年只跟踪 regime
all_years = [2020, 2021, 2022]
for year in all_years:
    y_start = f"{year}-01-01"
    y_end = f"{year}-12-31"
    df = load_data_for_strategy('v8', y_start, y_end)
    dates_in_year = sorted(df['date'].unique())
    
    for i, date in enumerate(dates_in_year):
        framework._on_day(date, df, dates_in_year)
    
    # Summarize regime at year end
    if year == 2022:
        print(f"\n>>> {year}年完成: cash={framework.cash:,.0f}")
