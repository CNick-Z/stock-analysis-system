#!/usr/bin/env python3
"""带 regime debug 的 backtest"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)
# Only show regime debug, suppress other DEBUG
for h in logging.root.handlers:
    h.setLevel(logging.INFO)
logging.getLogger('simulator.market_regime').setLevel(logging.DEBUG)

# Monkey-patch _process_buys to log regime on every day
from simulator.base_framework import BaseFramework
from simulator import base_framework

orig = base_framework.BaseFramework._process_buys
def patched(self, full_df, daily, market):
    position_limit = 1.0
    regime_used = "NO_FILTER"
    if self.market_regime_filter:
        regime_info = self.market_regime_filter.get_regime(market["date"])
        position_limit = regime_info["position_limit"]
        regime_used = f"{regime_info['regime']}({position_limit:.0%})"
        # Only log 2022 and on first day of each month
        day = market["date"][-2:]
        if market["date"] >= "2022-01-01" and day == "01":
            print(f"  {market['date']}: regime={regime_used}, raw={regime_info['raw_regime']}, "
                  f"RSI={regime_info['rsi14']:.1f}, consec={regime_info['consecutive_regime_days']}")
    # Continue with original
    return orig(self, full_df, daily, market)

base_framework.BaseFramework._process_buys = patched

# Now run the actual backtest via subprocess
import subprocess
result = subprocess.run(
    ['python3', 'backtest.py', '--strategy', 'v8', 
     '--start', '2020-01-01', '--end', '2022-12-31',
     '--market-filter'],
    capture_output=True, text=True, timeout=300,
    cwd='/root/.openclaw/workspace/projects/stock-analysis-system'
)
# Extract regime log lines and summary
lines = result.stdout.split('\n')
regime_lines = [l for l in lines if 'regime=' in l or '年完成' in l or '年化' in l or '持仓' in l and '完成' in l]
for l in regime_lines[-20:]:
    print(l)
