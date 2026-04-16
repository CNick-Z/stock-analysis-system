#!/usr/bin/env python3
"""
波浪买点回测
=============
买入：波浪识别W2/W4买点（回调>=5%后反弹）
卖出：原策略出场（止损5%、止盈15%、MA20出场）

用法:
    python3 wave_buy_backtest.py --years 2024
    python3 wave_buy_backtest.py --years 2024 --symbol 000001
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.score.v8.strategy import ScoreV8Strategy
from simulator.shared import load_wavechan_cache
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_wave_buy_signals(df: pd.DataFrame, cache_dir: str = '/data/warehouse/wavechan_l1') -> pd.DataFrame:
    """
    识别波浪买点信号
    
    在每个股票上：
    1. 按日期排序
    2. 识别调整浪（下跌>=5%）
    3. 调整后有反弹 = 买点
    """
    signals = []
    
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < 20:
            continue
        
        # 简化：用收盘价计算日涨跌
        prices = group['close'].values
        dates = group['date'].values
        
        # 计算日涨跌
        changes = np.diff(prices) / prices[:-1] * 100
        
        # 找调整浪：连续下跌>=5%
        i = 0
        while i < len(changes):
            if changes[i] < 0:
                # 开始下跌
                start_idx = i
                cumulative = 0
                
                while i < len(changes) and changes[i] < 0:
                    cumulative += changes[i]
                    i += 1
                
                end_idx = i - 1
                
                # 调整幅度>=5%
                if cumulative <= -5 and end_idx < len(changes) - 1:
                    # 检查之后是否有反弹
                    rebound_start = end_idx + 1
                    if rebound_start < len(changes):
                        rebound = changes[rebound_start]
                        if rebound > 0:
                            # 有反弹 = 买点
                            buy_date = dates[end_idx + 1]
                            buy_price = prices[end_idx + 1]
                            
                            signals.append({
                                'symbol': symbol,
                                'date': buy_date,
                                'price': buy_price,
                                'pullback_depth': cumulative,
                                'rebound': rebound
                            })
            else:
                i += 1
    
    return pd.DataFrame(signals)


def run_backtest(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    stop_loss: float = 0.05,
    take_profit: float = 0.15,
    holding_days: int = 20
) -> pd.DataFrame:
    """
    回测波浪买点
    
    买入：信号日开盘价
    卖出：止损/止盈/MA20/持有20天
    """
    trades = []
    
    for _, sig in signals.iterrows():
        symbol = sig['symbol']
        buy_date = sig['date']
        buy_price = sig['price']
        
        # 获取该股票的后续数据
        stock_df = df[(df['symbol'] == symbol) & (df['date'] > buy_date)].sort_values('date')
        
        if stock_df.empty:
            continue
        
        # 找开盘价
        first_open = stock_df.iloc[0]['open'] if 'open' in stock_df.columns else buy_price
        
        # 回测卖出
        sell_date = None
        sell_price = None
        sell_reason = None
        holding = 0
        
        for _, row in stock_df.iterrows():
            holding += 1
            price = row.get('close', buy_price)
            
            # 止损
            if price < first_open * (1 - stop_loss):
                sell_date = row['date']
                sell_price = price
                sell_reason = 'STOP_LOSS'
                break
            
            # 止盈
            if price > first_open * (1 + take_profit):
                sell_date = row['date']
                sell_price = price
                sell_reason = 'TAKE_PROFIT'
                break
            
            # MA20出场
            ma20 = row.get('sma_20')
            mf_trend = row.get('money_flow_trend', True)
            if not pd.isna(ma20) and price < ma20 and mf_trend is False:
                sell_date = row['date']
                sell_price = price
                sell_reason = 'MA20_EXIT'
                break
            
            # 持有20天强制卖出
            if holding >= holding_days:
                sell_date = row['date']
                sell_price = price
                sell_reason = 'HOLDING_DAYS'
                break
        
        if sell_date and sell_price:
            pnl_pct = (sell_price - first_open) / first_open * 100
            trades.append({
                'symbol': symbol,
                'buy_date': buy_date,
                'sell_date': sell_date,
                'buy_price': first_open,
                'sell_price': sell_price,
                'pnl_pct': pnl_pct,
                'holding_days': holding,
                'sell_reason': sell_reason
            })
    
    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description='波浪买点回测')
    parser.add_argument('--years', type=int, nargs='+', default=[2024], help='回测年份')
    parser.add_argument('--symbol', type=str, default=None, help='指定股票')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='止损比例')
    parser.add_argument('--take-profit', type=float, default=0.15, help='止盈比例')
    args = parser.parse_args()
    
    years = args.years
    logger.info(f"回测年份: {years}")
    
    # 加载数据
    df = load_strategy_data(years=years, add_money_flow=True)
    logger.info(f"数据: {len(df):,} 行")
    
    if args.symbol:
        df = df[df['symbol'] == args.symbol]
        logger.info(f"筛选后: {len(df):,} 行")
    
    # 找波浪买点
    signals = find_wave_buy_signals(df)
    logger.info(f"买入信号: {len(signals)} 个")
    
    if signals.empty:
        logger.warning("无买入信号!")
        return
    
    print("\n买入信号示例:")
    print(signals.head(10).to_string())
    
    # 回测
    trades = run_backtest(signals, df, args.stop_loss, args.take_profit)
    logger.info(f"有效交易: {len(trades)} 笔")
    
    if trades.empty:
        logger.warning("无有效交易!")
        return
    
    # 统计
    wins = trades[trades['pnl_pct'] > 0]
    losses = trades[trades['pnl_pct'] <= 0]
    
    print(f"\n=== 回测结果 ===")
    print(f"总交易: {len(trades)} 笔")
    print(f"盈利: {len(wins)} 笔 ({len(wins)/len(trades)*100:.1f}%)")
    print(f"亏损: {len(losses)} 笔 ({len(losses)/len(trades)*100:.1f}%)")
    print(f"平均收益: {trades['pnl_pct'].mean():.2f}%")
    print(f"总收益: {trades['pnl_pct'].sum():.2f}%")
    print(f"最大盈利: {trades['pnl_pct'].max():.2f}%")
    print(f"最大亏损: {trades['pnl_pct'].min():.2f}%")
    
    # 按卖出原因统计
    print(f"\n=== 卖出原因 ===")
    reason_stats = trades.groupby('sell_reason')['pnl_pct'].agg(['count', 'mean', 'sum'])
    print(reason_stats.to_string())
    
    # 保存结果
    output = f"/tmp/wave_backtest_{years[0]}.csv"
    trades.to_csv(output, index=False)
    logger.info(f"结果已保存: {output}")


if __name__ == '__main__':
    main()
