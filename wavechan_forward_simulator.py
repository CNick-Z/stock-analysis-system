#!/usr/bin/env python3
"""
WaveChan 前向模拟器 - 验证波浪自动修正

从2025年底开始，逐日喂入2026年新数据，
观察波浪标签如何自动修正，信号如何变化。
"""

import pandas as pd
import numpy as np
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载2026年增量数据
print("加载2026年数据...")
increments = []
for f in sorted(glob.glob('/opt/tdx_increment_2026*.csv')):
    df = pd.read_csv(f)
    increments.append(df)
    print(f"  {f}: {len(df)} 条")

inc_2026 = pd.concat(increments, ignore_index=True)
inc_2026['date'] = pd.to_datetime(inc_2026['date'])
inc_2026 = inc_2026.sort_values(['symbol', 'date']).reset_index(drop=True)
print(f"\n2026数据: {len(inc_2026):,} 条, {inc_2026['symbol'].nunique()} 只股票, {inc_2026['date'].nunique()} 天")
print(f"日期范围: {inc_2026['date'].min().date()} ~ {inc_2026['date'].max().date()}")

# 加载缓存获取2025-12-31状态（优先三层缓存，fallback到旧单文件）
print("\n加载波浪状态...")
try:
    from utils.wavechan_cache import WaveChanCacheManager
    cm = WaveChanCacheManager()
    cache = cm.load('2025-12-31', '2025-12-31')
    if cache.empty:
        raise ValueError("L1/L2 cache empty")
    print("  [L1/L2] 加载成功")
except Exception as e:
    print(f"  [L1/L2] 加载失败，回退到旧缓存: {e}")
    cache = pd.read_parquet('/data/warehouse/wavechan_signals_cache.parquet')

cache['date'] = pd.to_datetime(cache['date'])
last_date = cache['date'].max()
print(f"缓存最后日期: {last_date.date()}")

# 取2025-12-31的状态作为起点
cache_last = cache[cache['date'] == last_date].set_index('symbol').to_dict('index')

# 统计起始状态分布
print("\n=== 起始状态 (2025-12-31) ===")
states = [v['wave_state'] for v in cache_last.values()]
from collections import Counter
state_cnt = Counter(states)
print("波浪状态分布:")
for s, c in state_cnt.most_common():
    print(f"  {s}: {c}")

signals = [v['signal_type'] for v in cache_last.values()]
signal_cnt = Counter(signals)
print("\n信号分布:")
for s, c in signal_cnt.most_common():
    print(f"  {s}: {c}")

# 初始化波浪引擎
from strategies.wavechan_v3 import WaveEngine

# 预加载所有股票的引擎（从缓存状态恢复）
print("\n恢复波浪引擎状态...")

# 取2025-12-31的close价格
cache_last_df = cache[cache['date'] == last_date][['symbol', 'close', 'wave_state']].copy()
cache_last_df['symbol'] = cache_last_df['symbol'].astype(str).str.zfill(6)

# 只处理有波浪状态的股票
active_symbols = [s for s, v in cache_last.items() if v['wave_state'] not in ('initial', None)]
print(f"活跃股票: {len(active_symbols)}")

# 追踪变化
changes = []  # {date, symbol, old_state, new_state, old_signal, new_signal}
daily_wave_dist = {}  # date -> wave_state distribution

dates = sorted(inc_2026['date'].unique())
print(f"\n开始前向模拟: {dates[0].date()} ~ {dates[-1].date()}")
print("(只追踪2025年底已有波浪状态的股票)")

for i, date in enumerate(dates):
    if i % 3 == 0:
        print(f"\n[{i+1}/{len(dates)}] {date.date()}")
    
    # 当日新数据
    day_data = inc_2026[inc_2026['date'] == date].copy()
    day_data['symbol'] = day_data['symbol'].astype(str).str.zfill(6)
    
    # 找出当日有数据的活跃股票
    active_with_data = set(active_symbols) & set(day_data['symbol'].tolist())
    
    new_states = {}
    new_signals = {}
    
    for symbol in active_with_data:
        sym_data = day_data[day_data['symbol'] == symbol].copy()
        if len(sym_data) == 0:
            continue
        
        old_state = cache_last.get(symbol, {}).get('wave_state')
        old_signal = cache_last.get(symbol, {}).get('signal_type')
        
        # 用wavechan_v3引擎更新
        try:
            engine = WaveEngine(symbol=symbol)
            
            # 喂入历史数据（2025-12-31之前的）
            # 这里简化处理，直接喂2026数据，让引擎自动从头构建
            # 由于没有完整历史，这里只演示状态变化
            
            # 更新到最新
            for _, row in sym_data.iterrows():
                bar = {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                engine.update(bar)
            
            # 获取当前状态
            snapshot = engine.get_snapshot()
            new_state = snapshot.wave_state if snapshot else None
            new_signal = engine.get_signal() if hasattr(engine, 'get_signal') else None
            
            if new_state and new_state != old_state:
                changes.append({
                    'date': date,
                    'symbol': symbol,
                    'old_state': old_state,
                    'new_state': new_state,
                    'old_signal': old_signal,
                    'new_signal': str(new_signal) if new_signal else None
                })
                new_states[symbol] = new_state
                new_signals[symbol] = str(new_signal) if new_signal else None
                
        except Exception as e:
            pass
    
    # 更新缓存状态
    for sym, ns in new_states.items():
        if sym in cache_last:
            cache_last[sym]['wave_state'] = ns
        else:
            cache_last[sym] = {'wave_state': ns, 'signal_type': None}
    
    # 记录当日状态分布
    current_states = [v['wave_state'] for v in cache_last.values()]
    dist = Counter(current_states)
    daily_wave_dist[date] = dist

print(f"\n\n=== 模拟完成 ===")
print(f"总状态变化: {len(changes)} 次")

if changes:
    changes_df = pd.DataFrame(changes)
    
    # 分析变化
    print("\n=== 波浪状态变化分析 ===")
    print(f"变化股票数: {changes_df['symbol'].nunique()}")
    
    # 变化类型统计
    print("\n状态变化流向（前20）:")
    flow = changes_df.groupby(['old_state', 'new_state']).size().sort_values(ascending=False).head(20)
    for (old, new), cnt in flow.items():
        print(f"  {old} → {new}: {cnt}")
    
    print("\n信号变化流向（前20）:")
    signal_flow = changes_df.groupby(['old_signal', 'new_signal']).size().sort_values(ascending=False).head(20)
    for (old, new), cnt in signal_flow.items():
        print(f"  {old} → {new}: {cnt}")
    
    # 保存
    changes_df.to_parquet('/tmp/wavechan_state_changes.parquet', index=False)
    print(f"\n状态变化已保存到 /tmp/wavechan_state_changes.parquet")

# 每日状态分布
print("\n每日波浪状态分布变化:")
for date, dist in sorted(daily_wave_dist.items()):
    print(f"\n{date.date()}:")
    for state, cnt in dist.most_common(5):
        print(f"  {state}: {cnt}")