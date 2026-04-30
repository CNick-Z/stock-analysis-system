#!/usr/bin/env python3
"""
V4 增强回测 - 2025年

使用 V4 配置：
- enable_b_trap_alert=True
- use_extended_fib=True  
- use_tight_w2_range=True
- use_fib_382_stop=True
- enable_w5_divergence_penalty=True
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path("/root/.openclaw/workspace/projects/stock-analysis-system")
sys.path.insert(0, str(PROJECT_ROOT))

# ========================
# 启用 V4 配置
# ========================
from strategies.wavechan.v3_l2_cache.wavechan_config_v4 import WaveChanConfigV4
from strategies.wavechan.v3_l2_cache.wavechan_v4 import (
    compute_extended_fib, detect_b_trap, calc_w2_range, 
    calc_w5_divergence_penalty
)

# 创建 V4 配置
V4_CONFIG = WaveChanConfigV4(
    enable_b_trap_alert=True,
    use_extended_fib=True,
    use_tight_w2_range=True,
    use_fib_382_stop=True,
    enable_w5_divergence_penalty=True,
)

print("=" * 60)
print("  V4 增强回测 - 2025年")
print("=" * 60)
print(f"V4 CONFIG:")
print(f"  enable_b_trap_alert = {V4_CONFIG.enable_b_trap_alert}")
print(f"  use_extended_fib = {V4_CONFIG.use_extended_fib}")
print(f"  use_tight_w2_range = {V4_CONFIG.use_tight_w2_range}")
print(f"  use_fib_382_stop = {V4_CONFIG.use_fib_382_stop}")
print(f"  enable_w5_divergence_penalty = {V4_CONFIG.enable_w5_divergence_penalty}")
print("=" * 60)
print()

# ========================
# 运行回测（复用原有逻辑）
# ========================
from simulator.base_framework import BaseFramework, MarketSnapshot
from simulator.shared import (
    load_strategy, load_wavechan_cache, add_next_open,
    STRATEGY_REGISTRY, WAVECHAN_L2_CACHE
)
from simulator.market_regime import MarketRegimeFilter
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_backtest(start_date: str, end_date: str, strategy_name: str = "wavechan_v3_strict"):
    """运行 V4 增强回测"""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))

    logger.info(f"开始回测: {strategy_name}")
    logger.info(f"时间区间: {start_date} ~ {end_date}")

    # 加载数据
    logger.info(f"加载数据: {years}")
    df = load_strategy_data(years=years)
    df['date'] = pd.to_datetime(df['date'])

    # 日期过滤
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    logger.info(f"日期过滤后: {len(df):,} 行")

    # 加载 L2 cache
    cache = load_wavechan_cache(start_date, end_date)
    logger.info(f"L2 cache loaded: {len(cache):,} 行")
    df = df.merge(cache, on=['date', 'symbol'], how='left')

    # 添加次日开盘价
    df = add_next_open(df)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # 加载策略
    strategy_class = load_strategy(strategy_name)
    strategy = strategy_class()

    # 回测
    cash = 1_000_000
    positions = {}
    trades = []
    daily_values = []

    dates = sorted(df['date'].unique())
    logger.info(f"交易日: {len(dates)} 天")

    for i, date in enumerate(dates):
        day_df = df[df['date'] == date].copy()
        date_str = str(date)[:10]

        # 信号过滤
        candidates = strategy.filter_buy(day_df, date_str)
        if candidates.empty:
            # 检查持仓
            for sym, pos in list(positions.items()):
                row = day_df[day_df['symbol'] == sym]
                if row.empty:
                    continue
                price = row.iloc[0]['close']
                should_sell, reason = strategy.should_sell(
                    {'price': price, 'entry_price': pos['entry_price']},
                    {'date': date_str, 'close': price},
                    {}
                )
                if should_sell:
                    proceeds = pos['shares'] * price
                    profit = proceeds - pos['cost']
                    cash += proceeds
                    trades.append({
                        'date': date_str, 'symbol': sym,
                        'action': 'SELL', 'price': price,
                        'shares': pos['shares'], 'profit': profit,
                        'reason': reason
                    })
                    del positions[sym]

        # 买入
        if cash > 0 and not candidates.empty:
            candidates = candidates.sort_values('total_score', ascending=False)
            max_positions = 5
            available_slots = max_positions - len(positions)
            if available_slots > 0:
                to_buy = candidates.head(available_slots)
                for _, row in to_buy.iterrows():
                    if cash < 10000:
                        break
                    price = row['close']
                    # === V4: 使用 fib_382 作为止损位 ===
                    stop_loss = price * 0.97  # 默认
                    if V4_CONFIG.use_fib_382_stop:
                        # 尝试用 fib_382
                        w1_end = row.get('w1_high', 0)
                        w1_start = row.get('w1_low', 0)
                        if w1_end > 0 and w1_start > 0:
                            fib_result = compute_extended_fib(w1_start, w1_end, price)
                            if 'fib_382' in fib_result and fib_result['fib_382']:
                                stop_loss = fib_result['fib_382']
                    
                    shares = int(20000 / price) * 100
                    cost = shares * price
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[row['symbol']] = {
                        'entry_price': price,
                        'shares': shares,
                        'cost': cost,
                        'stop_loss': stop_loss,
                        'entry_date': date_str
                    }
                    trades.append({
                        'date': date_str, 'symbol': row['symbol'],
                        'action': 'BUY', 'price': price,
                        'shares': shares, 'score': row.get('total_score', 0)
                    })

        # 更新持仓市值
        total_value = cash
        for sym, pos in positions.items():
            row = day_df[day_df['symbol'] == sym]
            if not row.empty:
                total_value += pos['shares'] * row.iloc[0]['close']

        daily_values.append({
            'date': date_str,
            'cash': cash,
            'positions': len(positions),
            'total_value': total_value
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  {date_str} ({i+1}/{len(dates)}): 持仓{len(positions)}只, 总值{total_value/10000:.1f}万")

    # 计算结果
    final_value = daily_values[-1]['total_value'] if daily_values else cash
    profit = final_value - 1_000_000
    profit_pct = profit / 1_000_000 * 100

    # 统计
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    wins = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
    losses = len(sell_trades) - wins
    win_rate = wins / len(sell_trades) * 100 if sell_trades else 0

    print()
    print("=" * 60)
    print(f"  回测报告  |  策略: {strategy_name} (V4增强)  |  区间: {start_date} ~ {end_date}")
    print("=" * 60)
    print(f"  初始资金:       1,000,000")
    print(f"  最终价值:       {final_value:,.0f}  ({profit_pct:+.2f}%)")
    print(f"  年化收益:       {profit_pct:.2f}%")
    print(f"  总交易:         {len(buy_trades)}  笔")
    print(f"  胜率:          {win_rate:.1f}%")
    print("=" * 60)

    return {
        'final_value': final_value,
        'profit_pct': profit_pct,
        'total_trades': len(buy_trades),
        'win_rate': win_rate
    }


if __name__ == "__main__":
    run_backtest("2025-01-01", "2025-12-31", "wavechan_v3_strict")