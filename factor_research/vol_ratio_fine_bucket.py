#!/usr/bin/env python3
"""
vol_ratio 精细化分桶分析 — 回答：
1. vol_ratio 最优剔除阈值是多少？
2. vol_ratio 最优加分阈值是多少？
3. 剔除逻辑 vs 加分逻辑 哪个更好？

同时验证 turnover_rate 和 wr 的阈值
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, gc
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


def filter_ic(df, factor_col, threshold, direction, label_col='return_5d'):
    """剔除/加分逻辑IC计算
    direction: 'above' = 剔除>threshold, 'below' = 剔除<threshold
    返回: 全量IC, 过滤后IC, 过滤样本数
    """
    full_ic = spearman_ic(df[factor_col], df[label_col])
    
    if direction == 'above':
        filtered = df[df[factor_col] <= threshold]
    else:
        filtered = df[df[factor_col] >= threshold]
    
    filtered_ic = spearman_ic(filtered[factor_col], filtered[label_col]) if len(filtered) > 100 else np.nan
    return full_ic, filtered_ic, len(filtered)


def bonus_ic(df, factor_col, bonus_threshold, bonus_score, label_col='return_5d'):
    """加分逻辑：<threshold 的样本给予 bonus_score 加分后计算IC
    对比：原始IC vs 加分后IC
    """
    full_ic = spearman_ic(df[factor_col], df[label_col])
    bonus_df = df.copy()
    # 模拟加分效果：<threshold 的因子值提高
    bonus_df.loc[bonus_df[factor_col] < bonus_threshold, factor_col] *= (1 + bonus_score)
    bonus_ic = spearman_ic(bonus_df[factor_col], bonus_df[label_col])
    return full_ic, bonus_ic


def vol_ratio_fine_bucket():
    """vol_ratio 精细分桶: 0.3~3.0, 每0.2一档"""
    print("=" * 70)
    print("🔍 vol_ratio 精细化分桶分析 (2018-2025)")
    print("=" * 70)
    
    all_data = []
    for year in YEARS:
        print(f"\n📅 {year}...", end=' ')
        df = load_year_data(year)
        df['return_5d'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
        df['my_vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
        all_data.append(df[['date', 'symbol', 'my_vol_ratio', 'turnover_rate', 'williams_r', 'return_5d']])
        print(f"{len(df):,}条")
        del df; gc.collect()
    
    data = pd.concat(all_data, ignore_index=True)
    del all_data; gc.collect()
    
    valid = data[['my_vol_ratio', 'turnover_rate', 'williams_r', 'return_5d']].dropna()
    print(f"\n总有效样本: {len(valid):,}")
    print(f"vol_ratio 范围: {valid['my_vol_ratio'].min():.2f} ~ {valid['my_vol_ratio'].max():.2f}")
    print()
    
    # ===== 1. vol_ratio 精细分桶 IC =====
    print("=" * 70)
    print("📊 1. vol_ratio 精细分桶 IC (每0.2一档)")
    print("=" * 70)
    
    thresholds = np.arange(0.3, 3.05, 0.2)
    bucket_results = []
    
    for lo in thresholds:
        hi = lo + 0.2
        mask = (valid['my_vol_ratio'] >= lo) & (valid['my_vol_ratio'] < hi)
        bucket = valid[mask]
        ic = spearman_ic(bucket['my_vol_ratio'], bucket['return_5d'])
        pct = len(bucket) / len(valid) * 100
        bucket_results.append({
            'range': f'{lo:.1f}~{hi:.1f}',
            'ic': ic,
            'count': len(bucket),
            'pct': pct
        })
        print(f"  {lo:.1f}~{hi:.1f}: IC={ic:+.4f}  n={len(bucket):,} ({pct:.1f}%)")
    
    # 找最优/最差桶
    best = max(bucket_results, key=lambda x: x['ic'])
    worst = min(bucket_results, key=lambda x: x['ic'])
    print(f"\n  ★ 最优桶: {best['range']}  IC={best['ic']:+.4f}")
    print(f"  ✗ 最差桶: {worst['range']}  IC={worst['ic']:+.4f}")
    
    # ===== 2. 剔除逻辑测试 =====
    print("\n" + "=" * 70)
    print("📊 2. 剔除逻辑 — 剔除 vol_ratio > X 的样本后 IC")
    print("=" * 70)
    
    exclude_thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
    exclude_results = []
    
    full_ic = spearman_ic(valid['my_vol_ratio'], valid['return_5d'])
    print(f"\n  原始全量 IC = {full_ic:+.4f}")
    print(f"\n  {'剔除阈值':>10}  {'剔除后IC':>10}  {'IC提升':>8}  {'剔除样本%':>12}")
    print(f"  {'-'*45}")
    
    for thr in exclude_thresholds:
        _, filtered_ic, n_filtered = filter_ic(valid, 'my_vol_ratio', thr, 'above')
        pct_excl = n_filtered / len(valid) * 100
        ic_delta = filtered_ic - full_ic if not np.isnan(filtered_ic) else 0
        exclude_results.append({
            'threshold': thr,
            'filtered_ic': filtered_ic,
            'ic_delta': ic_delta,
            'pct_excluded': pct_excl
        })
        print(f"  >{thr:.1f}  剔除后  {filtered_ic:+.4f}  {ic_delta:+.4f}  ({pct_excl:.1f}%)")
    
    best_excl = max(exclude_results, key=lambda x: x['filtered_ic'])
    print(f"\n  ★ 最优剔除阈值: >{best_excl['threshold']:.1f}  IC={best_excl['filtered_ic']:+.4f} (提升{+best_excl['ic_delta']:.4f})")
    
    # ===== 3. 加分逻辑测试 =====
    print("\n" + "=" * 70)
    print("📊 3. 加分逻辑 — vol_ratio < X 给予加分，对比 IC 变化")
    print("=" * 70)
    
    bonus_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bonus_scores = [0.05, 0.10, 0.15, 0.20]
    
    print(f"\n  原始全量 IC = {full_ic:+.4f}")
    print(f"\n  {'加分阈值':>10}  {'加分幅度':>10}  {'加分后IC':>10}  {'IC提升':>8}  {'覆盖样本%':>12}")
    print(f"  {'-'*55}")
    
    bonus_detail = []
    for thr in bonus_thresholds:
        for score in bonus_scores:
            _, bonus_ic_val = bonus_ic(valid, 'my_vol_ratio', thr, score)
            covered_pct = (valid['my_vol_ratio'] < thr).sum() / len(valid) * 100
            ic_delta = bonus_ic_val - full_ic
            bonus_detail.append({
                'threshold': thr,
                'score': score,
                'bonus_ic': bonus_ic_val,
                'ic_delta': ic_delta,
                'covered_pct': covered_pct
            })
            print(f"  <{thr:.1f}   +{score:.0%}    {bonus_ic_val:+.4f}    {ic_delta:+.4f}    ({covered_pct:.1f}%)")
    
    best_bonus = max(bonus_detail, key=lambda x: x['bonus_ic'])
    print(f"\n  ★ 最优加分: <{best_bonus['threshold']:.1f}  +{best_bonus['score']:.0%}  IC={best_bonus['bonus_ic']:+.4f} (提升+{best_bonus['ic_delta']:.4f})")
    
    # ===== 4. 剔除 vs 加分 对比 =====
    print("\n" + "=" * 70)
    print("📊 4. 剔除 vs 加分 综合对比")
    print("=" * 70)
    
    # 对比：最优剔除 vs 最优加分 vs 两者结合
    print(f"\n  方案A（仅剔除）: >{best_excl['threshold']:.1f} 剔除 → IC={best_excl['filtered_ic']:+.4f}")
    print(f"  方案B（仅加分）: <{best_bonus['threshold']:.1f} +{best_bonus['score']:.0%} → IC={best_bonus['bonus_ic']:+.4f}")
    
    # 方案C：两者结合
    excl_th = best_excl['threshold']
    bon_th = best_bonus['threshold']
    bon_sc = best_bonus['score']
    
    combined = valid[
        (valid['my_vol_ratio'] <= excl_th) & 
        (valid['my_vol_ratio'] >= bon_th)
    ].copy() if bon_th < excl_th else valid[valid['my_vol_ratio'] <= excl_th].copy()
    
    # 模拟加分
    if bon_th < excl_th:
        combined.loc[combined['my_vol_ratio'] < bon_th, 'my_vol_ratio'] *= (1 + bon_sc)
    combined_ic = spearman_ic(combined['my_vol_ratio'], combined['return_5d'])
    combined_pct = len(combined) / len(valid) * 100
    print(f"  方案C（两者结合）: 剔除>{excl_th:.1f} + <{bon_th:.1f}加分 → IC={combined_ic:+.4f} (样本{combined_pct:.1f}%)")
    
    # ===== 5. 验证 turnover_rate 阈值 =====
    print("\n" + "=" * 70)
    print("📊 5. 验证 turnover_rate 阈值 (>2.79% 剔除)")
    print("=" * 70)
    
    tr_thresholds = [1.5, 2.0, 2.5, 2.79, 3.0, 3.5, 4.0]
    print(f"\n  {'剔除阈值':>10}  {'剔除后IC':>10}  {'IC提升':>8}  {'剔除样本%':>12}")
    print(f"  {'-'*45}")
    
    tr_results = []
    for thr in tr_thresholds:
        _, filtered_ic, n = filter_ic(valid, 'turnover_rate', thr, 'above')
        pct = n / len(valid) * 100
        delta = filtered_ic - full_ic
        tr_results.append({'threshold': thr, 'ic': filtered_ic, 'delta': delta, 'pct': pct})
        print(f"  >{thr:.2f}%  {filtered_ic:+.4f}  {delta:+.4f}  ({pct:.1f}%)")
    
    best_tr = max(tr_results, key=lambda x: x['ic'])
    print(f"\n  ★ 最优换手率剔除: >{best_tr['threshold']:.2f}%  IC={best_tr['ic']:+.4f}")
    
    # ===== 6. 验证 WR 阈值 (<-80 加分) =====
    print("\n" + "=" * 70)
    print("📊 6. 验证 WR 阈值 (<-80 加分)")
    print("=" * 70)
    
    wr_thresholds = [-60, -70, -80, -85, -90, -95]
    print(f"\n  {'加分阈值':>10}  {'加分后IC':>10}  {'IC提升':>8}  {'覆盖样本%':>12}")
    print(f"  {'-'*45}")
    
    wr_results = []
    for thr in wr_thresholds:
        # WR<threshold 表示超卖，对WR取反处理（WR越负表示越超卖）
        wr_corrected = -valid['williams_r']  # 转正：WR=-90 → 90（极度超卖）
        _, bonus_ic_val = bonus_ic(valid, 'williams_r', thr, 0.10)
        covered = (valid['williams_r'] < thr).sum() / len(valid) * 100
        wr_results.append({'threshold': thr, 'bonus_ic': bonus_ic_val, 'covered': covered})
        print(f"  <-{abs(thr):.0f}   {bonus_ic_val:+.4f}   {bonus_ic_val-full_ic:+.4f}    ({covered:.1f}%)")
    
    # ===== 7. 逐年验证最优剔除阈值 =====
    print("\n" + "=" * 70)
    print("📊 7. 逐年验证 vol_ratio 最优剔除阈值")
    print("=" * 70)
    
    candidate_thresholds = [1.0, 1.2, 1.4, 1.6, 2.0]
    
    for thr in candidate_thresholds:
        year_ics = []
        for year in YEARS:
            yr_data = valid[valid['date'].dt.year == year].copy()
            _, yr_ic, _ = filter_ic(yr_data, 'my_vol_ratio', thr, 'above')
            if not np.isnan(yr_ic):
                year_ics.append(yr_ic)
        avg = np.mean(year_ics) if year_ics else np.nan
        print(f"  >{thr:.1f}  8年平均IC={avg:+.4f}  (有效年数={len(year_ics)})")
    
    print("\n" + "=" * 70)
    print("📋 最终结论")
    print("=" * 70)
    
    return {
        'best_exclude': best_excl,
        'best_bonus': best_bonus,
        'combined_ic': combined_ic,
        'combined_pct': combined_pct,
        'best_tr': best_tr,
        'bucket_results': bucket_results,
        'exclude_results': exclude_results,
        'bonus_detail': bonus_detail,
    }


if __name__ == '__main__':
    results = vol_ratio_fine_bucket()
