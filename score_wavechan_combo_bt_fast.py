#!/usr/bin/env python3
"""
Score + WaveChan V3 组合策略回测 - 高性能版（2年2024-2025）

优化：
- 启动时一次性加载所有市场数据到内存
- 批量获取 Score 信号（按年）
- WaveChan 引擎复用，不重复加载数据
"""

import sys, os, time, json, logging
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

from strategies.score_strategy import ScoreStrategy
from strategies.wavechan_v3 import WaveEngine

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'initial_cash': 1_000_000,
    'position_size': 0.3,
    'stop_loss_pct': 0.05,
    'profit_target_pct': 0.20,
    'max_hold_days': 20,
    'max_positions': 5,
    'top_n': 5,
    'wavechan_buy_signals': ['W2_BUY', 'W4_BUY', 'C_BUY'],
    'valid_statuses': ['alert', 'confirmed'],
    'test_start': '2024-01-01',
    'test_end': '2025-12-31',
    'warmup_days': 200,
    'cache_dir': '/tmp/score_wavechan_combo_bt_fast',
}

# ============================================================
# 数据加载
# ============================================================

def get_db_path():
    return os.environ.get('PARQUET_DB_PATH', '/root/.openclaw/workspace/data/warehouse')

def preload_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """一次性加载所有市场数据"""
    lookback = pd.to_datetime(start_date) - pd.Timedelta(days=400)
    lookback_str = lookback.strftime('%Y-%m-%d')
    
    all_data = []
    for year in range(2020, 2027):
        path = f"{get_db_path()}/daily_data_year={year}/"
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = df[df['date'] <= end_date]
                all_data.append(df)
            except Exception as e:
                pass
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.concat(all_data).sort_values(['symbol', 'date']).reset_index(drop=True)
    return df

# ============================================================
# WaveChan 管理器
# ============================================================

class WaveChanManager:
    def __init__(self, config, market_df: pd.DataFrame):
        self.config = config
        self.cache_dir = config['cache_dir']
        self.market_df = market_df  # 预加载数据
        self.engines: dict = {}
        self.warmed: set = set()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def warm_engine(self, symbol: str, cutoff_date: str):
        """用市场数据预热某股票的 WaveChan 引擎"""
        if symbol in self.engines:
            return
        
        sym_df = self.market_df[self.market_df['symbol'] == symbol].copy()
        sym_df = sym_df[sym_df['date'] < cutoff_date].tail(self.config['warmup_days'])
        
        engine = WaveEngine(symbol=symbol, cache_dir=f"{self.cache_dir}/{symbol}")
        
        for _, row in sym_df.iterrows():
            engine.feed_daily({
                'date': str(row['date'])[:10],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if pd.notna(row.get('volume')) else 0,
            })
        
        self.engines[symbol] = engine
        self.warmed.add(symbol)
    
    def feed_and_get_signal(self, symbol: str, bar: dict) -> dict:
        """喂入K线并返回信号"""
        if symbol not in self.engines:
            return {'signal': 'NO_SIGNAL'}
        self.engines[symbol].feed_daily(bar)
        return self.engines[symbol].get_signal()

# ============================================================
# 回测引擎
# ============================================================

class FastComboBacktester:
    def __init__(self, config: dict, market_df: pd.DataFrame):
        self.config = config
        self.market_df = market_df
        self.wave_manager = WaveChanManager(config, market_df)
        
        # 初始化 Score 策略
        self.score_strategy = ScoreStrategy(db_path=None, config={
            'top_n': config['top_n'],
        })
    
    def run(self) -> dict:
        cfg = self.config
        
        # 交易日历
        all_dates = sorted(self.market_df['date'].unique())
        dates = [d for d in all_dates if cfg['test_start'] <= d <= cfg['test_end']]
        warmup_dates = [d for d in all_dates if d < cfg['test_start']]
        
        print(f"预热期: {len(warmup_dates)} 天, 回测期: {len(dates)} 天")
        
        # 批量获取 Score 信号
        print("获取 Score 信号...")
        
        all_buy_dfs = []
        for year in [2024, 2025]:
            ystart = f"{year}-01-01"
            yend = f"{year}-12-31"
            if not (ystart <= cfg['test_end'] and yend >= cfg['test_start']):
                continue
            real_start = max(ystart, cfg['test_start'])
            real_end = min(yend, cfg['test_end'])
            print(f"  Score {year}: {real_start} ~ {real_end}...")
            buy_signals, _ = self.score_strategy.get_signals(real_start, real_end)
            if buy_signals is not None and not buy_signals.empty:
                buy_signals = buy_signals.reset_index()
                all_buy_dfs.append(buy_signals)
        
        if all_buy_dfs:
            score_buy_df = pd.concat(all_buy_dfs)
            score_buy_df['date'] = pd.to_datetime(score_buy_df['date'])
        else:
            score_buy_df = pd.DataFrame()
        
        print(f"  Score 买入候选总数: {len(score_buy_df)}")
        
        # 建立 symbol → 日期索引
        score_buy_df['date_str'] = score_buy_df['date'].astype(str).str[:10]
        
        # 建立 symbol → 日线数据索引
        symbol_dfs = {}
        for sym in score_buy_df['symbol'].unique():
            sym_df = self.market_df[self.market_df['symbol'] == sym].copy()
            sym_df['date_str'] = sym_df['date'].astype(str).str[:10]
            sym_df = sym_df.sort_values('date').reset_index(drop=True)
            symbol_dfs[sym] = sym_df
        
        # 预热 WaveChan（所有候选股）
        warmup_cutoff = cfg['test_start']
        print(f"预热 WaveChan {len(symbol_dfs)} 只股票...")
        warmed_count = 0
        for sym in score_buy_df['symbol'].unique():
            if sym not in symbol_dfs:
                continue
            # 预热到回测开始前
            sym_df = symbol_dfs[sym]
            warm_df = sym_df[sym_df['date'] < warmup_cutoff].tail(cfg['warmup_days'])
            
            engine = WaveEngine(symbol=sym, cache_dir=f"{cfg['cache_dir']}/{sym}")
            for _, row in warm_df.iterrows():
                engine.feed_daily({
                    'date': str(row['date'])[:10],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if pd.notna(row.get('volume')) else 0,
                })
            self.wave_manager.engines[sym] = engine
            self.wave_manager.warmed.add(sym)
            warmed_count += 1
            if warmed_count % 50 == 0:
                print(f"  已预热 {warmed_count} 只...")
        
        print(f"  WaveChan 预热完成: {warmed_count} 只股票")
        
        # ============================================================
        # 回测循环
        # ============================================================
        cash = cfg['initial_cash']
        positions = {}
        trades = []
        equity_curve = []
        
        stats = {
            'total_signals': 0,
            'wavechan_filtered': 0,
            'buy_executed': 0,
            'na_signals': 0,
            'by_signal': defaultdict(lambda: {'count': 0, 'executed': 0}),
        }
        
        # 按日期分组候选信号
        date_to_candidates = defaultdict(list)
        for _, row in score_buy_df.iterrows():
            date_to_candidates[row['date_str']].append(row.to_dict())
        
        for i, date in enumerate(dates):
            date_str = date if isinstance(date, str) else str(date)[:10]
            
            # 获取当日候选
            day_candidates = date_to_candidates.get(date_str, [])
            
            if day_candidates:
                stats['total_signals'] += len(day_candidates)
                
                # WaveChan 过滤
                for candidate in day_candidates:
                    symbol = candidate['symbol']
                    sig = self.wave_manager.get_signal(symbol)
                    signal = sig.get('signal', 'NO_SIGNAL')
                    status = sig.get('status', '')
                    
                    if signal != 'NO_SIGNAL':
                        stats['by_signal'][signal]['count'] += 1
                    
                    allowed = cfg['wavechan_buy_signals']
                    valid = cfg['valid_statuses']
                    
                    if signal in allowed and status in valid:
                        stats['wavechan_filtered'] += 1
                        stats['by_signal'][signal]['executed'] += 1
                        
                        # 执行买入
                        if len(positions) < cfg['max_positions'] and symbol not in positions:
                            sym_df = symbol_dfs.get(symbol)
                            if sym_df is None:
                                continue
                            day_rows = sym_df[sym_df['date_str'] == date_str]
                            if day_rows.empty:
                                continue
                            
                            price = float(day_rows.iloc[0]['close'])
                            position_value = cash * cfg['position_size']
                            shares = int(position_value / price / 100) * 100
                            
                            if shares >= 100 and shares * price <= cash:
                                stop_loss = sig.get('stop_loss') or price * (1 - cfg['stop_loss_pct'])
                                positions[symbol] = {
                                    'buy_date': date_str,
                                    'buy_price': price,
                                    'shares': shares,
                                    'stop_loss': stop_loss,
                                    'days': 0,
                                    'buy_reason': f"{signal}:{sig.get('reason', '')}",
                                }
                                cash -= shares * price
                                stats['buy_executed'] += 1
                                # print(f"✅ 买入 {symbol} @ {price:.2f} ({signal})")
                    elif signal == 'NO_SIGNAL':
                        stats['na_signals'] += 1
            
            # 处理持仓
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                pos['days'] += 1
                
                sym_df = symbol_dfs.get(symbol)
                if sym_df is None:
                    continue
                day_rows = sym_df[sym_df['date_str'] == date_str]
                if day_rows.empty:
                    continue
                close = float(day_rows.iloc[0]['close'])
                
                ret = (close - pos['buy_price']) / pos['buy_price']
                exit_reason = ''
                
                if close <= pos['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif ret >= cfg['profit_target_pct']:
                    exit_reason = 'PROFIT_TAKE'
                elif pos['days'] >= cfg['max_hold_days']:
                    exit_reason = 'TIME_EXIT'
                
                if exit_reason:
                    pnl = (close - pos['buy_price']) * pos['shares']
                    cash += pos['shares'] * close
                    trades.append({
                        'symbol': symbol,
                        'buy_date': pos['buy_date'],
                        'sell_date': date_str,
                        'buy_price': pos['buy_price'],
                        'sell_price': close,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'return': ret,
                        'exit_reason': exit_reason,
                        'buy_reason': pos['buy_reason'],
                        'hold_days': pos['days'],
                    })
                    del positions[symbol]
            
            # 喂入当日K线（给所有已预热的引擎）
            for symbol in self.wave_manager.warmed:
                sym_df = symbol_dfs.get(symbol)
                if sym_df is None:
                    continue
                day_rows = sym_df[sym_df['date_str'] == date_str]
                if not day_rows.empty:
                    row = day_rows.iloc[0]
                    bar = {
                        'date': date_str,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if pd.notna(row.get('volume')) else 0,
                    }
                    self.wave_manager.engines[symbol].feed_daily(bar)
            
            # 记录 equity
            position_value = 0
            for sym in positions:
                sym_df = symbol_dfs.get(sym)
                if sym_df is None:
                    continue
                day_rows = sym_df[sym_df['date_str'] == date_str]
                if not day_rows.empty:
                    position_value += float(day_rows.iloc[0]['close']) * positions[sym]['shares']
            
            total_equity = cash + position_value
            equity_curve.append({
                'date': date_str,
                'cash': cash,
                'position_value': position_value,
                'total_equity': total_equity,
                'positions': len(positions),
            })
            
            if (i + 1) % 100 == 0:
                print(f"  {date_str} | Equity: {total_equity:,.0f} | 持仓: {len(positions)} | {i+1}/{len(dates)}")
        
        # 最终平仓
        last_date = dates[-1]
        last_date_str = last_date if isinstance(last_date, str) else str(last_date)[:10]
        
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            sym_df = symbol_dfs.get(symbol)
            if sym_df is not None:
                last_rows = sym_df[sym_df['date_str'] == last_date_str]
                close = float(last_rows.iloc[0]['close']) if not last_rows.empty else pos['buy_price']
            else:
                close = pos['buy_price']
            pnl = (close - pos['buy_price']) * pos['shares']
            cash += pos['shares'] * close
            trades.append({
                'symbol': symbol,
                'buy_date': pos['buy_date'],
                'sell_date': last_date_str,
                'buy_price': pos['buy_price'],
                'sell_price': close,
                'shares': pos['shares'],
                'pnl': pnl,
                'return': (close - pos['buy_price']) / pos['buy_price'],
                'exit_reason': 'FINAL_EXIT',
                'buy_reason': pos['buy_reason'],
                'hold_days': pos['days'],
            })
            del positions[symbol]
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_equity': cash,
            'stats': {
                'total_signals': stats['total_signals'],
                'wavechan_filtered': stats['wavechan_filtered'],
                'buy_executed': stats['buy_executed'],
                'na_signals': stats['na_signals'],
            },
            'by_signal': {k: dict(v) for k, v in stats['by_signal'].items()},
        }

# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("Score + WaveChan V3 组合策略回测 (2024-2025) - 高性能版")
    print("=" * 60)
    
    cfg = CONFIG
    
    # 1. 加载所有市场数据
    print("\n📊 加载市场数据（一次性）...")
    t0 = time.time()
    market_df = preload_market_data(cfg['test_start'], cfg['test_end'])
    if market_df.empty:
        print("❌ 无市场数据")
        return
    print(f"  加载完成: {len(market_df):,} 条, {market_df['symbol'].nunique()} 只股票")
    print(f"  日期范围: {market_df['date'].min()} ~ {market_df['date'].max()}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    
    # 2. 运行回测
    print("\n🚀 运行回测...")
    t1 = time.time()
    tester = FastComboBacktester(cfg, market_df)
    result = tester.run()
    elapsed = time.time() - t1
    total_elapsed = time.time() - t0
    
    # ============================================================
    # 报告
    # ============================================================
    trades = result['trades']
    equity_curve = result['equity_curve']
    stats = result['stats']
    by_signal = result['by_signal']
    
    print("\n" + "=" * 60)
    print("回测报告 (2024-2025)")
    print("=" * 60)
    
    print(f"\n📈 信号统计:")
    print(f"  Score 信号总数: {stats['total_signals']}")
    print(f"  WaveChan 过滤后: {stats['wavechan_filtered']}")
    print(f"  实际买入次数: {stats['buy_executed']}")
    print(f"  WaveChan N/A（消失）: {stats['na_signals']}")
    if stats['total_signals'] > 0:
        print(f"  WaveChan 通过率: {stats['wavechan_filtered']/stats['total_signals']*100:.1f}%")
    
    print(f"\n🌊 按 WaveChan 信号:")
    for sig, s in sorted(by_signal.items(), key=lambda x: -x[1]['count']):
        print(f"  {sig}: 候选{s['count']}次, 买入{s['executed']}次")
    
    print(f"\n💰 收益概况:")
    init = cfg['initial_cash']
    final = result['final_equity']
    total_return_pct = (final / init - 1) * 100
    print(f"  初始资金: {init:,.0f}")
    print(f"  最终资金: {final:,.0f}")
    print(f"  总收益: {final - init:,.0f}")
    print(f"  总收益率: {total_return_pct:.2f}%")
    
    max_drawdown = 0.0
    if equity_curve:
        eq = pd.DataFrame(equity_curve)
        eq['peak'] = eq['total_equity'].cummax()
        eq['dd'] = (eq['total_equity'] - eq['peak']) / eq['peak'] * 100
        max_drawdown = eq['dd'].min()
        print(f"  最大回撤: {max_drawdown:.2f}%")
    
    if trades:
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        
        print(f"\n📊 交易统计:")
        print(f"  总交易次数: {len(trades)}")
        win_rate = len(win_trades) / len(trades) * 100
        print(f"  胜率: {win_rate:.1f}% ({len(win_trades)}/{len(trades)})")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            print(f"  平均盈利: {avg_win:,.0f}")
            print(f"  平均亏损: {avg_loss:,.0f}")
            if avg_loss != 0:
                print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")
        
        print(f"\n🚪 按退出原因:")
        exit_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            exit_stats[t['exit_reason']]['count'] += 1
            exit_stats[t['exit_reason']]['pnl'] += t['pnl']
        for reason, s in sorted(exit_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {reason}: {s['count']}次, 盈亏{s['pnl']:,.0f}")
        
        print(f"\n🌊 按买入信号:")
        sig_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'win': 0, 'loss': 0})
        for t in trades:
            sig = t['buy_reason'].split(':')[0]
            sig_stats[sig]['count'] += 1
            sig_stats[sig]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                sig_stats[sig]['win'] += 1
            else:
                sig_stats[sig]['loss'] += 1
        for sig, s in sorted(sig_stats.items(), key=lambda x: -x[1]['count']):
            wr = s['win'] / s['count'] * 100 if s['count'] > 0 else 0
            print(f"  {sig}: {s['count']}次, 胜率{wr:.0f}%, 盈亏{s['pnl']:,.0f}")
    else:
        print("\n⚠️ 无交易记录！")
    
    print(f"\n⏱️ 回测耗时: {elapsed:.1f}秒 (总: {total_elapsed:.1f}秒)")
    
    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/root/.openclaw/workspace/projects/stock-analysis-system/backtestresult/score_wavechan_combo_a_{ts}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_data = {
        'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in cfg.items()},
        'stats': stats,
        'by_signal': by_signal,
        'final_equity': float(final),
        'total_return': float(total_return_pct),
        'max_drawdown': float(max_drawdown),
        'trades': trades,
        'equity_curve': equity_curve[-30:] if len(equity_curve) > 30 else equity_curve,
        'elapsed_seconds': float(elapsed),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 结果已保存: {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
