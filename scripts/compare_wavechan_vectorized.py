"""
波浪缠论 V3 vs 向量化版本 对比测试
目标：验证向量化方案(fast)与逐K线方案(v3)的输出是否一致

测试方法：
1. 从已有 parquet 读取 2022年 v3 输出的信号(ground truth)
2. 对同样股票读取原始日线数据
3. 用 fast 版本计算
4. 对比 wave_trend、signal_type 一致率
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pyarrow.parquet as pq

# ========== 1. 加载 v3 ground truth (2022年从 parquet) ==========
print("=" * 60)
print("步骤1: 读取 v3 ground truth (2022年)")
print("=" * 60)

pf = pq.ParquetFile('/data/warehouse/wavechan_signals_cache.parquet')

# 读取包含2022的行组
v3_results = []
for i in range(pf.metadata.num_row_groups):
    table = pf.read_row_group(i, columns=['date', 'symbol', 'wave_trend', 'signal_type', 'wave_state', 'has_signal', 'close'])
    dates = table['date'].to_pandas()
    if '2022' in dates.astype(str).str[:4].values:
        df = table.to_pandas()
        df = df[df['date'].astype(str).str.startswith('2022')]
        v3_results.append(df)
        print(f"  行组{i}: 2022年 {len(df):,} 行")

v3_df = pd.concat(v3_results, ignore_index=True)
print(f"\n  v3 total: {len(v3_df):,} 行")
print(f"  股票数: {v3_df['symbol'].nunique()}")
print(f"  signal_type 分布:\n{v3_df['signal_type'].value_counts().head(10)}")

# ========== 2. 读取原始日线数据 for same stocks ==========
print("\n" + "=" * 60)
print("步骤2: 读取原始日线数据")
print("=" * 60)

daily_path = '/root/.openclaw/workspace/data/warehouse/daily_data_year=2022/data.parquet'
daily_df = pd.read_parquet(daily_path)
print(f"  总行数: {len(daily_df):,}")
print(f"  股票数: {daily_df['symbol'].nunique()}")

# 取共同股票(取前50只用于快速测试)
common_symbols = list(set(v3_df['symbol'].unique()) & set(daily_df['symbol'].unique()))[:50]
print(f"  共同股票数: {len(common_symbols)}")

v3_sample = v3_df[v3_df['symbol'].isin(common_symbols)].copy()
daily_sample = daily_df[daily_df['symbol'].isin(common_symbols)].copy()

print(f"  v3 样本: {len(v3_sample):,} 行")
print(f"  日线样本: {len(daily_sample):,} 行")

# ========== 3. 运行 fast 向量化版本 ==========
print("\n" + "=" * 60)
print("步骤3: 运行 fast 向量化版本")
print("=" * 60)

from strategies.wavechan_fast import compute_wavechan_features_fast

config = {
    'wave_threshold_pct': 0.025,
    'decline_threshold': -0.15,
    'consolidation_threshold': 0.05,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
}

print(f"  正在计算 {len(common_symbols)} 只股票...")
fast_result = compute_wavechan_features_fast(daily_sample, config)
print(f"  fast 输出: {len(fast_result):,} 行")

# ========== 4. 对比 wave_trend ==========
print("\n" + "=" * 60)
print("步骤4: 对比 wave_trend 一致率")
print("=" * 60)

# 合并两个结果
v3_sample = v3_sample.rename(columns={
    'signal_type': 'v3_signal_type',
    'wave_trend': 'v3_wave_trend',
    'wave_state': 'v3_wave_state',
})

# 合并(按date和symbol)
merged = fast_result[['date', 'symbol', 'wave_trend', 'wave_stage', 'daily_signal', 'close']].merge(
    v3_sample[['date', 'symbol', 'v3_wave_trend', 'v3_signal_type', 'v3_wave_state', 'close']],
    on=['date', 'symbol'],
    how='inner'
)
print(f"  匹配行数: {len(merged):,}")

# wave_trend 对比
# v3: 'up'/'down'/'neutral' vs fast: 'up'/'down'/'neutral'
merged['trend_match'] = merged['wave_trend'] == merged['v3_wave_trend']
trend_acc = merged['trend_match'].mean()
print(f"\n  wave_trend 一致率: {trend_acc:.2%}")
print(f"  v3 trend 分布: {merged['v3_wave_trend'].value_counts().to_dict()}")
print(f"  fast trend 分布: {merged['wave_trend'].value_counts().to_dict()}")

# 按趋势分类看准确率
for trend in merged['v3_wave_trend'].unique():
    subset = merged[merged['v3_wave_trend'] == trend]
    acc = subset['trend_match'].mean()
    cnt = len(subset)
    print(f"    v3={trend}: 一致率 {acc:.2%} ({cnt:,}行)")

# ========== 5. 对比 signal_type / buy_signal ==========
print("\n" + "=" * 60)
print("步骤5: 对比买卖信号")
print("=" * 60)

# v3 的 signal_type 有 BUY/SELL 等
# fast 的 daily_signal 有 '买入'/'卖出'/'观望'

# 统一成二值: 有买入信号 vs 无买入
merged['v3_has_buy'] = merged['v3_signal_type'].fillna('').str.contains('BUY', na=False)
merged['fast_has_buy'] = merged['daily_signal'] == '买入'

# 对比有无买入
buy_match = (merged['v3_has_buy'] == merged['fast_has_buy']).mean()
print(f"  买入信号一致率: {buy_match:.2%}")
print(f"  v3 买入天数: {merged['v3_has_buy'].sum():,}")
print(f"  fast 买入天数: {merged['fast_has_buy'].sum():,}")

# ========== 6. 详细差异分析 ==========
print("\n" + "=" * 60)
print("步骤6: 差异分析")
print("=" * 60)

# 找出趋势不一致的样本
diff = merged[~merged['trend_match']]
print(f"  趋势不一致行数: {len(diff):,}")
if len(diff) > 0:
    # 看不一致时 close 差异
    diff['close_diff_pct'] = abs(diff['close_x'] - diff['close_y']) / diff['close_y'] * 100
    print(f"  close 差异均值: {diff['close_diff_pct'].mean():.4f}% (应该都是0)")
    
    # 打印几个不一致的例子
    print(f"\n  不一致样本 (前10):")
    sample = diff[['date', 'symbol', 'v3_wave_trend', 'wave_trend', 'v3_signal_type', 'daily_signal']].head(10)
    print(sample.to_string(index=False))

# ========== 7. 信号位置偏移分析 ==========
print("\n" + "=" * 60)
print("步骤7: 买入信号时间偏移分析")
print("=" * 60)

# 对每只股票，找出第一次买入信号的时间，对比两个版本
signal_alignment = []
for sym in common_symbols[:20]:  # 前20只
    v3_sym = merged[(merged['symbol'] == sym) & (merged['v3_has_buy'])]
    fast_sym = merged[(merged['symbol'] == sym) & (merged['fast_has_buy'])]
    
    if len(v3_sym) > 0 and len(fast_sym) > 0:
        v3_first = v3_sym['date'].min()
        fast_first = fast_sym['date'].min()
        diff_days = (pd.to_datetime(fast_first) - pd.to_datetime(v3_first)).days
        signal_alignment.append({
            'symbol': sym,
            'v3_first_buy': v3_first,
            'fast_first_buy': fast_first,
            'days_diff': diff_days
        })

if signal_alignment:
    align_df = pd.DataFrame(signal_alignment)
    print(f"  有两只版本都有买入信号的股票: {len(align_df)}")
    print(f"  信号时间偏移(天)分布:\n{align_df['days_diff'].describe()}")
    print(f"\n  偏移前10:")
    print(align_df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("对比完成")
print("=" * 60)
