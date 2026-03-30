#!/usr/bin/env python3
"""
因子 IC 细化分析 v2 — 针对关键阈值做精细化分桶
补充方案A/C所需的数据支撑

【需要回答的问题】
1. RSI 到底多少以下是好的？30？40？50？
2. 量比的最佳买点是 >1（放量）还是 <1（缩量）？
3. 换手率低于多少是好的？
4. CCI -100 真的是最佳临界点吗？
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, gc, tracemalloc, psutil
from datetime import datetime

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
YEARS = range(2018, 2026)


def load_year_data(year):
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    return df


def spearman_ic(series, label):
    valid = pd.DataFrame({'s': series, 'l': label}).dropna()
    if len(valid) < 100:
        return np.nan
    return spearmanr(valid['s'], valid['l'])[0]


def bucket_ic(df, factor_col, label_col='return_5d', n_buckets=5):
    """分桶IC分析，返回每个桶的平均IC和样本比例"""
    valid = df[[factor_col, label_col]].dropna()
    if len(valid) < 500:
        return None
    
    try:
        labels = pd.qcut(valid[factor_col], q=n_buckets, duplicates='drop', retbins=True)
        buckets = labels[0]
        bins = labels[1]
    except:
        return None
    
    results = []
    for i, (idx, grp) in enumerate(valid.groupby(buckets)):
        ic = spearman_ic(grp[factor_col], grp[label_col])
        results.append({
            'bucket': i,
            'range': f'{bins[i]:.2f}~{bins[i+1]:.2f}',
            'ic': ic,
            'count': len(grp),
            'pct': len(grp) / len(valid) * 100
        })
    return results


def analyze_detailed():
    monitor = ResourceMonitor()
    print("=" * 70)
    print("🔍 因子阈值细化分析 — IC分桶")
    print("=" * 70)
    print(f"年份: {min(YEARS)}~{max(YEARS)}")
    print()
    
    # 收集所有年份数据
    all_years_ic = {}
    
    for year in YEARS:
        print(f"\n📅 {year} 年数据加载中...")
        df = load_year_data(year)
        df['return_5d'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
        df['my_vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)  # 自建量比
        # bollinger位置需要sma_5
        df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['boll_pos'] = df['boll_pos'].clip(0, 1)
        
        print(f"   记录数: {len(df):,}")
        monitor.check(f"{year}加载")
        
        # ===== 分桶分析 =====
        bucket_configs = [
            ('RSI分桶', 'rsi_14', 5),
            ('量比分桶', 'my_vol_ratio', 5),
            ('换手率分桶', 'turnover_rate', 5),
            ('CCI分桶(等距)', 'cci_20', 5),
            ('KDJ_K分桶', 'kdj_k', 5),
            ('MACD分桶', 'macd_histogram', 5),
            ('BOLL位置分桶', 'boll_pos', 5),
            ('WR分桶', 'williams_r', 5),
        ]
        
        for name, col, n_bk in bucket_configs:
            key = f'{year}_{name}'
            result = bucket_ic(df, col, 'return_5d', n_buckets=n_bk)
            if result:
                all_years_ic[key] = {
                    'factor': name,
                    'year': year,
                    'buckets': result
                }
                # 打印最关键信息
                if year == 2024:  # 只打印2024年的详情作为示例
                    ics = [r['ic'] for r in result if not np.isnan(r['ic'])]
                    if ics:
                        best = result[np.argmax(ics)]
                        worst = result[np.argmin(ics)]
                        print(f"  {name}: 最佳桶={best['range']} IC={best['ic']:.4f} | "
                              f"最差桶={worst['range']} IC={worst['ic']:.4f}")
        
        del df; gc.collect()
        monitor.check(f"{year}完成")
    
    # ===== 汇总跨年分桶IC =====
    print("\n" + "=" * 70)
    print("📊 跨年分桶IC汇总（各桶8年平均IC）")
    print("=" * 70)
    
    summary_cols = ['RSI分桶','量比分桶','换手率分桶','CCI分桶(等距)',
                     'KDJ_K分桶','MACD分桶','BOLL位置分桶','WR分桶']
    
    for fac in summary_cols:
        keys = [k for k in all_years_ic if fac in k]
        if not keys:
            continue
        
        # 按桶号汇总
        n_buckets = len(all_years_ic[keys[0]]['buckets'])
        bucket_avg = []
        for b in range(n_buckets):
            ics = []
            for k in keys:
                bkt = all_years_ic[k]['buckets']
                if b < len(bkt) and not np.isnan(bkt[b]['ic']):
                    ics.append(bkt[b]['ic'])
            if ics:
                bucket_avg.append({
                    'bucket': b,
                    'range': all_years_ic[keys[0]]['buckets'][b]['range'],
                    'avg_ic': np.mean(ics),
                    'ic_std': np.std(ics),
                    'years': len(ics)
                })
        
        print(f"\n🔹 {fac}（8年平均）:")
        print(f"  {'区间':<15} {'Avg IC':>8} {'Std':>6} {'年数':>5}")
        print(f"  {'-'*40}")
        for r in bucket_avg:
            print(f"  {r['range']:<15} {r['avg_ic']:>8.4f} {r['ic_std']:>6.4f} {r['years']:>5}")
        
        # 最佳区间标注
        best_bucket = max(bucket_avg, key=lambda x: x['avg_ic'])
        worst_bucket = min(bucket_avg, key=lambda x: x['avg_ic'])
        print(f"  ★ 最佳区间: {best_bucket['range']} (IC={best_bucket['avg_ic']:.4f})")
        print(f"  ✗ 最差区间: {worst_bucket['range']} (IC={worst_bucket['avg_ic']:.4f})")
    
    return all_years_ic


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak = 0
        
    def check(self, ctx=""):
        mem_gb = self.process.memory_info().rss / 1024**3
        self.peak = max(self.peak, mem_gb)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] MEM={mem_gb:.2f}GB(峰值={self.peak:.2f}GB) | {ctx}")
        return True


if __name__ == '__main__':
    results = analyze_detailed()
