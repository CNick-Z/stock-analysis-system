#!/usr/bin/env python3
"""
波浪买点回测 + 基本面过滤
=======================
过滤条件：
- 市值 > min_market_cap (亿)
- PE > 0 (盈利股)
- 上市时间 > 1年

用法:
    python3 wave_buy_backtest_quality.py --years 2025 --min-market-cap 10 --min-pe 0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_wave_buy_signals(
    df: pd.DataFrame,
    min_market_cap: float = 10,  # 亿
    min_pe: float = 0,
    min_listing_years: float = 1,  # 年
) -> pd.DataFrame:
    """
    识别波浪买点信号 + 基本面过滤
    """
    # 基本面数据
    basic = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/stock_basic_info.parquet')
    basic['listing_date'] = pd.to_datetime(basic['listing_date'])
    
    # 计算市值(亿)
    if 'market_value' in basic.columns:
        basic['market_cap_yi'] = basic['market_value'] / 1e8
    elif 'total_shares' in basic.columns:
        # 需要用最新价格计算
        latest = df.groupby('symbol')['close'].last().reset_index()
        basic = basic.merge(latest, on='symbol', how='left')
        basic['market_cap_yi'] = basic['total_shares'] * basic['close'] / 1e8
    else:
        basic['market_cap_yi'] = 0
    
    # 计算上市年限
    ref_date = pd.Timestamp('2025-12-31')
    basic['listing_years'] = (ref_date - basic['listing_date']).dt.days / 365
    
    # 财务数据
    fin = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/financial_summary.parquet')
    # 取最新的PE
    latest_fin = fin.sort_values('date').groupby('symbol').last().reset_index()
    
    # 合并
    basic = basic.merge(latest_fin[['symbol', 'pe_ratio']], on='symbol', how='left')
    
    # 过滤条件
    mask = (
        (basic['market_cap_yi'] >= min_market_cap) &
        (basic['pe_ratio'].fillna(0) >= min_pe) &
        (basic['listing_years'] >= min_listing_years)
    )
    valid_symbols = set(basic[mask]['symbol'])
    logger.info(f"基本面过滤后: {len(valid_symbols)} 只股票")
    
    signals = []
    
    for symbol, group in df.groupby('symbol'):
        if symbol not in valid_symbols:
            continue
            
        group = group.sort_values('date').reset_index(drop=True)
        if len(group) < 20:
            continue
        
        prices = group['close'].values
        dates = group['date'].values
        changes = np.diff(prices) / prices[:-1] * 100
        
        i = 0
        while i < len(changes):
            if changes[i] < 0:
                start_idx = i
                cumulative = 0
                
                while i < len(changes) and changes[i] < 0:
                    cumulative += changes[i]
                    i += 1
                
                end_idx = i - 1
                
                if cumulative <= -5 and end_idx < len(changes) - 1:
                    rebound_start = end_idx + 1
                    if rebound_start < len(changes):
                        rebound = changes[rebound_start]
                        if rebound > 0:
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
    """回测"""
    trades = []
    
    for _, sig in signals.iterrows():
        symbol = sig['symbol']
        buy_date = sig['date']
        buy_price = sig['price']
        
        stock_df = df[(df['symbol'] == symbol) & (df['date'] > buy_date)].sort_values('date')
        if stock_df.empty:
            continue
        
        first_open = stock_df.iloc[0]['open'] if 'open' in stock_df.columns else buy_price
        
        sell_date = None
        sell_price = None
        sell_reason = None
        holding = 0
        
        for _, row in stock_df.iterrows():
            holding += 1
            price = row.get('close', buy_price)
            
            if price < first_open * (1 - stop_loss):
                sell_date = row['date']
                sell_price = price
                sell_reason = 'STOP_LOSS'
                break
            
            if price > first_open * (1 + take_profit):
                sell_date = row['date']
                sell_price = price
                sell_reason = 'TAKE_PROFIT'
                break
            
            ma20 = row.get('sma_20')
            mf_trend = row.get('money_flow_trend', True)
            if not pd.isna(ma20) and price < ma20 and mf_trend is False:
                sell_date = row['date']
                sell_price = price
                sell_reason = 'MA20_EXIT'
                break
            
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
    parser = argparse.ArgumentParser(description='波浪买点回测 + 基本面过滤')
    parser.add_argument('--years', type=int, nargs='+', default=[2025], help='回测年份')
    parser.add_argument('--symbol', type=str, default=None, help='指定股票')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='止损比例')
    parser.add_argument('--take-profit', type=float, default=0.15, help='止盈比例')
    parser.add_argument('--min-market-cap', type=float, default=10, help='最小市值(亿)')
    parser.add_argument('--min-pe', type=float, default=0, help='最小PE')
    parser.add_argument('--min-listing-years', type=float, default=1, help='最短上市时间(年)')
    args = parser.parse_args()
    
    years = args.years
    logger.info(f"回测年份: {years}")
    logger.info(f"基本面过滤: 市值>{args.min_market_cap}亿, PE>{args.min_pe}, 上市>{args.min_listing_years}年")
    
    df = load_strategy_data(years=years, add_money_flow=True)
    logger.info(f"数据: {len(df):,} 行")
    
    if args.symbol:
        df = df[df['symbol'] == args.symbol]
        logger.info(f"筛选后: {len(df):,} 行")
    
    signals = find_wave_buy_signals(
        df, 
        min_market_cap=args.min_market_cap,
        min_pe=args.min_pe,
        min_listing_years=args.min_listing_years
    )
    logger.info(f"买入信号: {len(signals)} 个")
    
    if signals.empty:
        logger.warning("无买入信号!")
        return
    
    trades = run_backtest(signals, df, args.stop_loss, args.take_profit)
    logger.info(f"有效交易: {len(trades)} 笔")
    
    if trades.empty:
        logger.warning("无有效交易!")
        return
    
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
    
    print(f"\n=== 卖出原因 ===")
    reason_stats = trades.groupby('sell_reason')['pnl_pct'].agg(['count', 'mean', 'sum'])
    print(reason_stats.to_string())
    
    output = f"/tmp/wave_backtest_quality_{years[0]}.csv"
    trades.to_csv(output, index=False)
    logger.info(f"结果已保存: {output}")


if __name__ == '__main__':
    main()
