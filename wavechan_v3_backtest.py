#!/usr/bin/env python3
"""
WaveChan V3 回测验证
测试买卖点信号的有效性
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from strategies.wavechan_v3 import WaveEngine, SymbolWaveCache, WaveCounterV3, BiRecord
from czsc import RawBar, Freq

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'initial_cash': 1_000_000,
    'position_size': 0.3,       # 30%仓位
    'stop_loss_pct': 0.05,      # 5%止损
    'profit_target_pct': 0.20,  # 20%止盈
    'max_hold_days': 20,        # 最大持仓20交易日
    'max_positions': 5,         # 最多5个持仓
    'test_symbols': [
        '600036', '600519', '601318', '000858', '002594', '300750',
        '000001', '000002', '600985', '600143'
    ],
    'start_date': '2025-01-01',
    'end_date': '2025-12-31',
}

# ============================================================
# 数据加载
# ============================================================

def load_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载股票数据"""
    all_data = []
    for year in [2025]:
        path = f'/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df[(df['symbol'] == symbol) & 
                    (df['date'] >= start_date) & 
                    (df['date'] <= end_date)]
            all_data.append(df)
    if all_data:
        return pd.concat(all_data).sort_values('date')
    return pd.DataFrame()

def df_to_rawbars(df: pd.DataFrame) -> list:
    """DataFrame转RawBar列表"""
    bars = []
    for _, row in df.iterrows():
        bar = RawBar(
            symbol=row['symbol'],
            dt=row['date'] if isinstance(row['date'], datetime) else pd.to_datetime(row['date']).to_pydatetime(),
            freq=Freq.D,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            vol=float(row['volume']) if pd.notna(row['volume']) else 0,
            amount=0.0,
        )
        bars.append(bar)
    return bars

# ============================================================
# 回测引擎
# ============================================================

def run_backtest(symbol: str, bars: list) -> dict:
    """单只股票回测"""
    cfg = CONFIG
    cache_dir = f"/tmp/wavechan_v3_backtest/{symbol}"
    os.makedirs(cache_dir, exist_ok=True)
    engine = WaveEngine(symbol=symbol, cache_dir=cache_dir)
    
    cash = cfg['initial_cash']
    position = None
    trades = []
    equity_curve = []
    
    for i, bar in enumerate(bars):
        date = bar.dt.strftime('%Y-%m-%d') if isinstance(bar.dt, datetime) else str(bar.dt)[:10]
        
        # 1. 喂数据
        snapshot = engine.feed_daily({
            'date': date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.vol,
        })
        
        # 2. 获取信号
        sig_dict = engine.get_signal()
        signal = sig_dict.get('signal', 'NO_SIGNAL')
        price = bar.close
        stop_loss = sig_dict.get('stop_loss')
        reason = sig_dict.get('reason', '')
        
        # 3. 卖出检查
        if position:
            position['days'] += 1
            ret = (price - position['buy_price']) / position['buy_price']
            exit_reason = ''
            
            # 止损
            if stop_loss and price <= stop_loss:
                exit_reason = 'STOP_LOSS'
            # 止盈
            elif ret >= cfg['profit_target_pct']:
                exit_reason = 'PROFIT_TAKE'
            # 时间止损
            elif position['days'] >= cfg['max_hold_days']:
                exit_reason = 'TIME_EXIT'
            # 信号卖出
            elif signal in ['W5_SELL', 'SELL']:
                exit_reason = 'WAVE_SIGNAL'
            
            if exit_reason:
                # 平仓
                pnl = (price - position['buy_price']) * position['shares']
                cash = cash + pnl
                trades.append({
                    'symbol': symbol,
                    'buy_date': position['buy_date'],
                    'sell_date': date,
                    'buy_price': position['buy_price'],
                    'sell_price': price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return': ret,
                    'exit_reason': exit_reason,
                    'buy_reason': position['buy_reason'],
                    'hold_days': position['days'],
                })
                position = None
        
        # 4. 买入检查
        if not position and signal in ['W2_BUY', 'W4_BUY_CONFIRMED', 'W4_BUY_ALERT', 'C_BUY']:
            # 仓位计算
            shares = int((cash * cfg['position_size']) / price / 100) * 100
            if shares >= 100:
                cost = shares * price
                if cost <= cash:
                    position = {
                        'symbol': symbol,
                        'buy_date': date,
                        'buy_price': price,
                        'shares': shares,
                        'stop_loss': stop_loss or price * (1 - cfg['stop_loss_pct']),
                        'days': 0,
                        'buy_reason': reason,
                    }
                    cash -= cost
        
        # 5. 记录 equity
        equity = cash + (position['shares'] * price - position['shares'] * position['buy_price']) if position else cash
        equity_curve.append({'date': date, 'equity': equity, 'position': position is not None})
    
    # 最终平仓
    if position:
        last_price = bars[-1].close
        pnl = (last_price - position['buy_price']) * position['shares']
        cash = cash + pnl
        trades.append({
            'symbol': symbol,
            'buy_date': position['buy_date'],
            'sell_date': bars[-1].dt.strftime('%Y-%m-%d') if isinstance(bars[-1].dt, datetime) else str(bars[-1].dt)[:10],
            'buy_price': position['buy_price'],
            'sell_price': last_price,
            'shares': position['shares'],
            'pnl': pnl,
            'return': (last_price - position['buy_price']) / position['buy_price'],
            'exit_reason': 'FINAL_EXIT',
            'signal_reason': position['buy_reason'],
            'hold_days': position['days'],
        })
    
    return {
        'symbol': symbol,
        'trades': trades,
        'equity_curve': equity_curve,
        'final_equity': cash,
        'total_trades': len(trades),
    }

# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("WaveChan V3 回测验证")
    print("=" * 60)
    
    cfg = CONFIG
    results = []
    
    for symbol in cfg['test_symbols']:
        print(f"\n正在回测 {symbol}...")
        
        # 加载数据
        df = load_stock_data(symbol, cfg['start_date'], cfg['end_date'])
        if df.empty:
            print(f"  无数据，跳过")
            continue
        
        bars = df_to_rawbars(df)
        if len(bars) < 60:
            print(f"  数据不足({len(bars)}天)，跳过")
            continue
        
        print(f"  数据: {len(bars)} 天")
        
        # 回测
        result = run_backtest(symbol, bars)
        results.append(result)
        
        # 单只结果
        total_pnl = sum(t['pnl'] for t in result['trades'])
        win_trades = [t for t in result['trades'] if t['pnl'] > 0]
        print(f"  交易次数: {result['total_trades']}")
        print(f"  盈利次数: {len(win_trades)}")
        print(f"  总盈亏: {total_pnl:+.2f}")
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    
    all_trades = []
    for r in results:
        all_trades.extend(r['trades'])
    
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        loss_trades = [t for t in all_trades if t['pnl'] <= 0]
        
        print(f"总交易次数: {len(all_trades)}")
        print(f"盈利次数: {len(win_trades)} ({len(win_trades)/len(all_trades)*100:.1f}%)")
        print(f"亏损次数: {len(loss_trades)}")
        print(f"总盈亏: {total_pnl:+.2f}")
        print(f"胜率: {len(win_trades)/len(all_trades)*100:.1f}%")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            print(f"平均盈利: {avg_win:+.2f}")
            print(f"平均亏损: {avg_loss:+.2f}")
            print(f"盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss else "N/A")
        
        # 按退出原因统计
        exit_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in all_trades:
            exit_stats[t['exit_reason']]['count'] += 1
            exit_stats[t['exit_reason']]['pnl'] += t['pnl']
        
        print("\n按退出原因统计:")
        for reason, stats in sorted(exit_stats.items()):
            print(f"  {reason}: {stats['count']}次, 盈亏{stats['pnl']:+.2f}")
        
        # 按信号类型统计
        signal_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in all_trades:
            reason = t.get('buy_reason', 'UNKNOWN')
            signal_stats[reason.split(':')[0]]['count'] += 1
            signal_stats[reason.split(':')[0]]['pnl'] += t['pnl']
        
        print("\n按买入信号统计:")
        for sig, stats in sorted(signal_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {sig}: {stats['count']}次, 盈亏{stats['pnl']:+.2f}")
    
    # 保存详细结果
    output_path = '/root/.openclaw/workspace/projects/stock-analysis-system/backtestresult/wavechan_v3_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': cfg,
            'results': [
                {
                    'symbol': r['symbol'],
                    'total_trades': r['total_trades'],
                    'final_equity': r['final_equity'],
                    'trades': r['trades'],
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {output_path}")

if __name__ == '__main__':
    main()