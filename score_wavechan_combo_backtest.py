#!/usr/bin/env python3
"""
Score + WaveChan V3 组合策略回测

方向A：Score 选强势股 → WaveChan 找回调买点

回测逻辑：
1. 每日用 ScoreStrategy 选出 Top N 强势股
2. 用 WaveChan 检查是否处于 W2/W4 回调买点
3. 双确认 → 买入
"""

import sys, os, time, json, logging
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional, Set

from strategies.score_strategy import ScoreStrategy
from strategies.wavechan_v3 import WaveEngine

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
    'top_n': 5,                 # Score 每日选股数
    'wavechan_buy_signals': ['W2_BUY', 'W4_BUY', 'C_BUY'],
    
    # 测试参数
    'test_start': '2025-06-01',
    'test_end': '2025-06-30',
    
    # 预热
    'warmup_days': 200,  # 预热天数（需要足够长让缠论形成笔）
    'cache_dir': '/tmp/score_wavechan_combo_backtest_v2',
}

# ============================================================
# 数据加载
# ============================================================

def get_db_path():
    return os.environ.get('PARQUET_DB_PATH', '/root/.openclaw/workspace/data/warehouse')

def load_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """加载全市场数据"""
    all_data = []
    for year in range(2020, 2027):
        path = f"{get_db_path()}/daily_data_year={year}/"
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                all_data.append(df)
            except:
                pass
    if all_data:
        return pd.concat(all_data).sort_values(['symbol', 'date'])
    return pd.DataFrame()

# ============================================================
# WaveChan 管理器
# ============================================================

class WaveChanManager:
    """WaveChan 引擎管理器，按需初始化"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = config['cache_dir']
        self.engines: Dict[str, WaveEngine] = {}
        self.warmed: Set[str] = set()
        
    def ensure_engine(self, symbol: str, history_df: pd.DataFrame, warmup_days: int):
        """确保某只股票有 WaveChan 引擎（延迟初始化）"""
        if symbol in self.engines:
            return
        
        os.makedirs(f"{self.cache_dir}/{symbol}", exist_ok=True)
        engine = WaveEngine(symbol=symbol, cache_dir=f"{self.cache_dir}/{symbol}")
        
        # 用历史数据预热
        warmup_df = history_df.tail(warmup_days)
        for _, row in warmup_df.iterrows():
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
        
    def get_signal(self, symbol: str) -> dict:
        """获取信号"""
        if symbol not in self.engines:
            return {'signal': 'NO_SIGNAL'}
        return self.engines[symbol].get_signal()
    
    def feed_bar(self, symbol: str, bar: dict):
        """喂入新K线"""
        if symbol in self.engines:
            self.engines[symbol].feed_daily(bar)

# ============================================================
# 回测引擎
# ============================================================

class ComboBacktester:
    """组合策略回测器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化 Score 策略
        self.score_strategy = ScoreStrategy(db_path=None, config={
            'top_n': config['top_n'],
        })
        
        # WaveChan 管理器
        self.wave_manager = WaveChanManager(config)
        
        # 市场数据缓存
        self.symbol_data_cache: Dict[str, pd.DataFrame] = {}
        
    def load_symbol_data(self, symbols: list):
        """加载指定股票的历史数据"""
        db_path = get_db_path()
        warmup_start = pd.to_datetime(self.config['test_start']) - pd.Timedelta(days=self.config['warmup_days'] + 30)
        
        for sym in symbols:
            if sym in self.symbol_data_cache:
                continue
            
            all_data = []
            for year in range(2020, 2027):
                path = f"{db_path}/daily_data_year={year}/"
                if os.path.exists(path):
                    try:
                        df = pd.read_parquet(path)
                        df = df[(df['symbol'] == sym) & (df['date'] >= warmup_start.strftime('%Y-%m-%d'))]
                        all_data.append(df)
                    except:
                        pass
            
            if all_data:
                df = pd.concat(all_data).sort_values('date').reset_index(drop=True)
                self.symbol_data_cache[sym] = df
        
        self.logger.info(f"已加载 {len(self.symbol_data_cache)} 只股票的历史数据")
        
    def run(self) -> dict:
        """运行回测"""
        cfg = self.config
        
        # 获取交易日历（从 Parquet 数据仓库）
        db_path = get_db_path()
        date_set = set()
        for year in range(2020, 2027):
            path = f"{db_path}/daily_data_year={year}/"
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path, columns=['date'])
                    dates = df['date'].unique()
                    date_set.update(dates)
                except:
                    pass
        
        dates = sorted([d for d in date_set if cfg['test_start'] <= d <= cfg['test_end']])
        
        self.logger.info(f"开始回测 {len(dates)} 个交易日")
        
        # 初始化
        cash = cfg['initial_cash']
        positions = {}  # {symbol: {...}}
        trades = []
        equity_curve = []
        daily_stats = {
            'total_signals': 0,
            'wavechan_filtered': 0,
            'buy_executed': 0,
        }
        
        # 按日期迭代
        for i, date in enumerate(dates):
            date_str = date if isinstance(date, str) else pd.to_datetime(date).strftime('%Y-%m-%d')
            
            # =====================
            # 1. 获取 Score 信号
            # =====================
            score_signals = self.score_strategy.get_signals(date_str, date_str)
            buy_candidates, _ = score_signals
            
            candidates_with_wave = []
            
            if buy_candidates is not None and not buy_candidates.empty:
                daily_stats['total_signals'] += len(buy_candidates)
                
                # 获取候选股票的代码
                candidate_symbols = buy_candidates['symbol'].tolist()
                
                # 确保这些股票的数据已加载
                self.load_symbol_data(candidate_symbols)
                
                # 预热 WaveChan（如果还没预热过）
                for sym in candidate_symbols:
                    if sym not in self.wave_manager.warmed and sym in self.symbol_data_cache:
                        sym_df = self.symbol_data_cache[sym]
                        # 用到前一天为止的数据预热
                        warmup_df = sym_df[sym_df['date'] < date_str]
                        if len(warmup_df) >= 60:
                            self.wave_manager.ensure_engine(sym, warmup_df, len(warmup_df))
                
                # =====================
                # 2. WaveChan 过滤
                # =====================
                allowed_signals = cfg['wavechan_buy_signals']
                valid_statuses = ['ALERT', 'CONFIRMED']
                
                for _, row in buy_candidates.iterrows():
                    symbol = row['symbol']
                    
                    # 获取 WaveChan 信号
                    wave_sig = self.wave_manager.get_signal(symbol)
                    
                    # 只接受有效状态的买入信号
                    if wave_sig['signal'] in allowed_signals and wave_sig.get('status') in valid_statuses:
                        daily_stats['wavechan_filtered'] += 1
                        
                        # 合并信号
                        candidate = row.to_dict()
                        candidate['wave_signal'] = wave_sig['signal']
                        candidate['wave_status'] = wave_sig.get('status', '')
                        candidate['wave_reason'] = wave_sig.get('reason', '')
                        candidate['wave_stop_loss'] = wave_sig.get('stop_loss')
                        candidate['wave_price'] = wave_sig.get('price')
                        candidates_with_wave.append(candidate)
            
            # =====================
            # 3. 执行买入
            # =====================
            available_slots = cfg['max_positions'] - len(positions)
            
            for candidate in candidates_with_wave:
                if available_slots <= 0:
                    break
                symbol = candidate['symbol']
                if symbol in positions:
                    continue
                
                # 获取当日价格
                if symbol not in self.symbol_data_cache:
                    continue
                sym_df = self.symbol_data_cache[symbol]
                day_df = sym_df[sym_df['date'] == date_str]
                if day_df.empty:
                    continue
                
                price = float(day_df.iloc[0]['close'])
                position_value = cash * cfg['position_size']
                shares = int(position_value / price / 100) * 100
                
                if shares >= 100 and shares * price <= cash:
                    stop_loss = candidate.get('wave_stop_loss')
                    if not stop_loss:
                        stop_loss = price * (1 - cfg['stop_loss_pct'])
                    
                    positions[symbol] = {
                        'buy_date': date_str,
                        'buy_price': price,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'days': 0,
                        'buy_reason': f"{candidate.get('wave_signal', 'UNKNOWN')}:{candidate.get('wave_reason', '')}",
                    }
                    cash -= shares * price
                    daily_stats['buy_executed'] += 1
                    self.logger.info(f"✅ 买入 {symbol} @ {price:.2f} ({candidate.get('wave_signal', '')})")
            
            # =====================
            # 4. 处理持仓
            # =====================
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                pos['days'] += 1
                
                # 获取当日价格
                if symbol not in self.symbol_data_cache:
                    continue
                sym_df = self.symbol_data_cache[symbol]
                day_df = sym_df[sym_df['date'] == date_str]
                if day_df.empty:
                    continue
                close = float(day_df.iloc[0]['close'])
                
                # 计算收益
                ret = (close - pos['buy_price']) / pos['buy_price']
                exit_reason = ''
                
                # 止损检查
                if close <= pos['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                # 止盈检查
                elif ret >= cfg['profit_target_pct']:
                    exit_reason = 'PROFIT_TAKE'
                # 时间止损
                elif pos['days'] >= cfg['max_hold_days']:
                    exit_reason = 'TIME_EXIT'
                
                if exit_reason:
                    # 平仓
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
                    self.logger.info(f"📤 卖出 {symbol} @ {close:.2f} ({exit_reason}) | 收益: {pnl:+.2f}")
                    del positions[symbol]
            
            # =====================
            # 5. 喂入新K线给 WaveChan
            # =====================
            for sym, sym_df in self.symbol_data_cache.items():
                day_df = sym_df[sym_df['date'] == date_str]
                if not day_df.empty:
                    row = day_df.iloc[0]
                    bar = {
                        'date': date_str,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if pd.notna(row.get('volume')) else 0,
                    }
                    self.wave_manager.feed_bar(sym, bar)
            
            # =====================
            # 6. 记录 equity
            # =====================
            position_value = 0
            for sym in positions:
                if sym not in self.symbol_data_cache:
                    continue
                sym_df = self.symbol_data_cache[sym]
                day_df = sym_df[sym_df['date'] == date_str]
                if not day_df.empty:
                    position_value += float(day_df.iloc[0]['close']) * positions[sym]['shares']
            
            total_equity = cash + position_value
            equity_curve.append({
                'date': date_str,
                'cash': cash,
                'position_value': position_value,
                'total_equity': total_equity,
                'positions': len(positions),
            })
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"  {date_str} | Equity: {total_equity:,.0f} | 持仓: {len(positions)}")
        
        # 最终平仓
        last_date = dates[-1]
        last_date_str = last_date if isinstance(last_date, str) else pd.to_datetime(last_date).strftime('%Y-%m-%d')
        
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol in self.symbol_data_cache:
                sym_df = self.symbol_data_cache[symbol]
                last_df = sym_df[sym_df['date'] == last_date]
                close = float(last_df.iloc[0]['close']) if not last_df.empty else pos['buy_price']
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
            'daily_stats': daily_stats,
        }

# ============================================================
# 主程序
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Score + WaveChan V3 组合策略回测")
    print("=" * 60)
    
    cfg = CONFIG
    
    # 预加载市场数据（用于获取交易日历）
    print("\n📊 加载市场数据...")
    market_df = load_market_data('2020-01-01', cfg['test_end'])
    if market_df.empty:
        print("❌ 无市场数据，请检查数据路径")
        return
    
    print(f"  数据范围: {market_df['date'].min()} ~ {market_df['date'].max()}")
    print(f"  股票数量: {market_df['symbol'].nunique()}")
    print(f"  总记录数: {len(market_df)}")
    
    # 运行回测
    print("\n🚀 运行回测...")
    start_time = time.time()
    
    tester = ComboBacktester(cfg)
    
    # 只加载交易日历（不需要预热所有股票）
    # 先做一次 Score 信号获取，了解会选中哪些股票
    logger.info("探索 Score 信号分布...")
    
    # 简化处理：直接运行回测，按需加载数据
    result = tester.run()
    
    elapsed = time.time() - start_time
    
    # =====================
    # 输出报告
    # =====================
    print("\n" + "=" * 60)
    print("回测报告")
    print("=" * 60)
    
    trades = result['trades']
    equity_curve = result['equity_curve']
    stats = result['daily_stats']
    
    print(f"\n📈 每日统计:")
    print(f"  Score 信号总数: {stats['total_signals']}")
    print(f"  WaveChan 过滤后: {stats['wavechan_filtered']}")
    print(f"  实际买入次数: {stats['buy_executed']}")
    if stats['total_signals'] > 0:
        print(f"  WaveChan 过滤率: {(1 - stats['wavechan_filtered']/max(stats['total_signals'],1))*100:.1f}%")
    
    print(f"\n💰 收益概况:")
    print(f"  初始资金: {cfg['initial_cash']:,.0f}")
    print(f"  最终资金: {result['final_equity']:,.0f}")
    print(f"  总收益: {result['final_equity'] - cfg['initial_cash']:,.0f}")
    print(f"  收益率: {(result['final_equity']/cfg['initial_cash']-1)*100:.2f}%")
    
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        
        print(f"\n📊 交易统计:")
        print(f"  总交易次数: {len(trades)}")
        print(f"  盈利次数: {len(win_trades)} ({len(win_trades)/len(trades)*100:.1f}%)")
        print(f"  亏损次数: {len(loss_trades)}")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            print(f"  平均盈利: {avg_win:,.0f}")
            print(f"  平均亏损: {avg_loss:,.0f}")
            if avg_loss != 0:
                print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")
        
        # 按退出原因统计
        print(f"\n🚪 按退出原因:")
        exit_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            exit_stats[t['exit_reason']]['count'] += 1
            exit_stats[t['exit_reason']]['pnl'] += t['pnl']
        for reason, s in sorted(exit_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {reason}: {s['count']}次, 盈亏{s['pnl']:,.0f}")
        
        # 按买入信号统计
        print(f"\n🌊 按买入信号:")
        signal_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            sig = t['buy_reason'].split(':')[0]
            signal_stats[sig]['count'] += 1
            signal_stats[sig]['pnl'] += t['pnl']
        for sig, s in sorted(signal_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {sig}: {s['count']}次, 盈亏{s['pnl']:,.0f}")
    else:
        print("\n⚠️ 无交易记录！")
    
    print(f"\n⏱️ 回测耗时: {elapsed:.1f}秒")
    
    # 保存结果
    output_path = '/root/.openclaw/workspace/projects/stock-analysis-system/backtestresult/score_wavechan_combo_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_data = {
        'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in cfg.items()},
        'stats': stats,
        'final_equity': result['final_equity'],
        'total_return': (result['final_equity'] / cfg['initial_cash'] - 1) * 100,
        'trades': trades,
        'equity_curve': equity_curve[-30:] if len(equity_curve) > 30 else equity_curve,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 结果已保存: {output_path}")

if __name__ == '__main__':
    main()
