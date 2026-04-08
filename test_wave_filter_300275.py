#!/usr/bin/env python3
"""
快速测试：验证V3波浪过滤是否正常工作
测试股票: 300275
"""
import sys, os
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pandas as pd
import glob
from datetime import datetime

from strategies.wavechan_v3 import WaveEngine

# ======================
# 配置
# ======================
SYMBOL = '300275'
START_DATE = '2025-01-01'
END_DATE = '2025-12-31'
CACHE_DIR = f"/tmp/wave_filter_test/{SYMBOL}"

# ======================
# 数据加载
# ======================
def load_data(symbol, start_date, end_date):
    """加载股票数据"""
    path = '/root/.openclaw/workspace/data/warehouse/daily_data_year=2025/'
    parquet_file = glob.glob(path + '*.parquet')
    if not parquet_file:
        print("未找到数据文件")
        return pd.DataFrame()
    
    df = pd.read_parquet(parquet_file[0])
    df = df[(df['symbol'].astype(str) == symbol) & 
            (df['date'] >= start_date) & 
            (df['date'] <= end_date)]
    return df.sort_values('date')

# ======================
# 主测试
# ======================
print("=" * 60)
print(f"V3波浪过滤验证 - 股票 {SYMBOL}")
print("=" * 60)

# 加载数据
df = load_data(SYMBOL, START_DATE, END_DATE)
print(f"\n数据加载: {len(df)} 条K线")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

if df.empty:
    print("无数据，退出")
    sys.exit(1)

# 初始化引擎
os.makedirs(CACHE_DIR, exist_ok=True)
engine = WaveEngine(symbol=SYMBOL, cache_dir=CACHE_DIR)

# 存储所有信号
all_signals = []
all_states = []

print("\n开始回测...")
for idx, row in df.iterrows():
    bar = {
        'date': str(row['date'])[:10],
        'open': float(row['open']),
        'high': float(row['high']),
        'low': float(row['low']),
        'close': float(row['close']),
        'volume': float(row['volume']) if pd.notna(row['volume']) else 0,
    }
    
    # 喂数据
    snapshot = engine.feed_daily(bar)
    sig_dict = engine.get_signal()
    
    # 记录状态和信号
    state_str = engine.get_state_str()
    
    all_states.append({
        'date': bar['date'],
        'state': snapshot.state,
        'state_str': state_str,
        'w1_end': snapshot.w1_end,
        'w2_end': snapshot.w2_end,
        'w3_end': snapshot.w3_end,
        'w4_end': snapshot.w4_end,
        'w5_end': snapshot.w5_end,
        'wave_end_signal': snapshot.wave_end_signal,
        'wave_end_confidence': snapshot.wave_end_confidence,
    })
    
    # 只记录有信号的日子
    sig = sig_dict.get('signal', 'NO_SIGNAL')
    if sig != 'NO_SIGNAL':
        all_signals.append({
            'date': bar['date'],
            'signal': sig,
            'status': sig_dict.get('status', ''),
            'price': sig_dict.get('price'),
            'stop_loss': sig_dict.get('stop_loss'),
            'reason': sig_dict.get('reason', ''),
            'confidence': sig_dict.get('confidence', 0),
            'wave_type': sig_dict.get('wave_type', ''),
            'wave_structure_valid': sig_dict.get('wave_structure_valid', True),
            'large_trend': sig_dict.get('large_trend', 'neutral'),
        })
        print(f"  {bar['date']} | {sig:15s} | {sig_dict.get('price', 0):.2f} | {sig_dict.get('reason', '')[:50]}")

print(f"\n信号统计:")
print(f"  总信号数: {len(all_signals)}")

# 按信号类型统计
from collections import Counter
sig_counts = Counter([s['signal'] for s in all_signals])
for sig, count in sig_counts.items():
    print(f"  {sig}: {count}")

# 检查W2 Buy信号的波浪结构验证
print("\n" + "=" * 60)
print("W2 Buy 信号详情:")
print("=" * 60)
w2_signals = [s for s in all_signals if 'W2' in s['signal']]
if w2_signals:
    for sig in w2_signals:
        print(f"\n日期: {sig['date']}")
        print(f"  信号: {sig['signal']} ({sig['status']})")
        print(f"  价格: {sig['price']}")
        print(f"  止损: {sig['stop_loss']}")
        print(f"  置信度: {sig['confidence']}")
        print(f"  波浪结构验证: {sig['wave_structure_valid']}")
        print(f"  大级别趋势: {sig['large_trend']}")
        print(f"  原因: {sig['reason']}")
else:
    print("无W2 Buy信号")

# 检查是否有被过滤的信号
print("\n" + "=" * 60)
print("波浪过滤检查:")
print("=" * 60)
invalid_signals = [s for s in all_signals if not s.get('wave_structure_valid', True)]
print(f"被过滤的信号数: {len(invalid_signals)}")
if invalid_signals:
    for sig in invalid_signals:
        print(f"  {sig['date']} | {sig['signal']} | 原因: {sig.get('reason', 'N/A')[:60]}")

# 最终状态
final_state = all_states[-1] if all_states else None
if final_state:
    print(f"\n最终状态: {final_state['state_str']}")

print("\n测试完成")