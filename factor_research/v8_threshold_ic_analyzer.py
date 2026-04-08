#!/usr/bin/env python3
"""
V8 策略剩余参数 IC/IR 分析
分析三个关键阈值:
1. volume_condition 放量倍数 (>X.X)
2. RSI 超卖阈值 (<XX)
3. WR 超卖阈值 (<-XX)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, gc, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/factor_research/results'
YEARS = range(2018, 2026)


def load_year_data(year):
    """加载并合并技术指标 + 日线数据"""
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    return df


def spearman_ic(factor_values, returns):
    """计算 Spearman IC"""
    valid = pd.DataFrame({'f': factor_values, 'r': returns}).dropna()
    if len(valid) < 100:
        return np.nan, 0
    corr, pval = spearmanr(valid['f'], valid['r'])
    return corr, len(valid)


def analyze_threshold(name, df, factor_col, thresholds, direction='above', label_col='return_5d'):
    """
    对阈值扫描做 IC 分析
    direction: 'above' -> factor > threshold 为信号
               'below' -> factor < threshold 为信号
    """
    print(f"\n  {'='*55}")
    print(f"  📊 {name}")
    print(f"  {'='*55}")
    print(f"  {'阈值':<15} {'IC':>8} {'样本数':>10} {'信号率':>8}")
    print(f"  {'-'*45}")

    results = []
    for thresh in thresholds:
        # 构建二元信号
        if direction == 'above':
            signal = (df[factor_col] > thresh).astype(float)
        else:
            signal = (df[factor_col] < thresh).astype(float)

        ic, n = spearman_ic(signal.values, df[label_col].values)
        signal_rate = signal.mean() * 100

        marker = ""
        results.append({
            'threshold': thresh,
            'ic': ic,
            'n': n,
            'signal_rate': signal_rate
        })

        print(f"  {f'{thresh}':<15} {ic:>8.4f} {n:>10,} {signal_rate:>7.1f}%{marker}")

    # 找最优
    valid_results = [r for r in results if not np.isnan(r['ic'])]
    if valid_results:
        best = max(valid_results, key=lambda x: x['ic'])
        worst = min(valid_results, key=lambda x: x['ic'])
        print(f"  {'-'*45}")
        print(f"  ★ 最优阈值: {best['threshold']}  IC={best['ic']:.4f}  (信号率={best['signal_rate']:.1f}%)")
        print(f"  ✗ 最差阈值: {worst['threshold']}  IC={worst['ic']:.4f}")
    else:
        best = None

    return results, best


def main():
    print("=" * 65)
    print("🔍 V8 策略阈值 IC/IR 分析 — 2018~2025")
    print("=" * 65)
    print(f"数据目录: {DATA_DIR}")
    print()

    # ===== 按年份加载数据 =====
    all_dfs = {}
    for year in YEARS:
        tech_path = f'{DATA_DIR}/technical_indicators_year={year}/data.parquet'
        daily_path = f'{DATA_DIR}/daily_data_year={year}/data.parquet'
        if os.path.exists(tech_path) and os.path.exists(daily_path):
            df = load_year_data(year)
            df['return_5d'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
            df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
            all_dfs[year] = df
            print(f"  ✅ {year}: {len(df):,} 条记录")
        else:
            print(f"  ⚠️ {year}: 数据缺失")

    if not all_dfs:
        print("❌ 没有任何年份数据！")
        return

    # ===== 合并所有年份 =====
    combined = pd.concat(all_dfs.values(), ignore_index=True)
    del all_dfs
    gc.collect()
    print(f"\n📊 合并后总记录数: {len(combined):,}")
    print()

    # ============================================================
    # 1. VOL_RATIO 放量倍数阈值扫描
    # ============================================================
    print("\n" + "=" * 65)
    print("📦 1. volume_condition (放量倍数) IC 分析")
    print("    因子: vol_ratio = volume / vol_ma5")
    print("    信号: vol_ratio > threshold → 买入")
    print("=" * 65)

    vol_thresholds = [0.8, 1.0, 1.2, 1.5, 2.0]
    vol_results, vol_best = analyze_threshold(
        "放量倍数阈值 (vol_ratio > X)", combined,
        'vol_ratio', vol_thresholds, direction='above'
    )

    # 额外: 看看 <1 的缩量区间（反向信号）
    print(f"\n  --- 补充：缩量区间（反向信号）---")
    below_thresholds = [0.5, 0.6, 0.7, 0.8]
    below_results = []
    for thresh in below_thresholds:
        signal = (combined['vol_ratio'] < thresh).astype(float)
        ic, n = spearman_ic(signal.values, combined['return_5d'].values)
        signal_rate = signal.mean() * 100
        below_results.append({'threshold': thresh, 'ic': ic, 'n': n, 'signal_rate': signal_rate})
        print(f"  {'<'+str(thresh):<15} {ic:>8.4f} {n:>10,} {signal_rate:>7.1f}%")

    # ============================================================
    # 2. RSI 超卖阈值扫描
    # ============================================================
    print("\n" + "=" * 65)
    print("📉 2. RSI 超卖阈值 IC 分析")
    print("    因子: rsi_14")
    print("    信号: rsi_14 < threshold → 超卖区间（考虑剔除）")
    print("    注意: RSI绝对值 IC=-1.92 最差，RSI超卖可能是陷阱")
    print("=" * 65)

    rsi_thresholds = [20, 25, 30, 35, 40]
    rsi_results, rsi_best = analyze_threshold(
        "RSI 超卖阈值 (rsi_14 < X)", combined,
        'rsi_14', rsi_thresholds, direction='below'
    )

    # 也看看 RSI > threshold（反向，RSI不是超卖）是否有更好 IC
    print(f"\n  --- 补充：RSI 适中/偏高区间（反向）---")
    above_rsi_thresholds = [40, 50, 60, 70]
    above_rsi_results = []
    for thresh in above_rsi_thresholds:
        signal = (combined['rsi_14'] > thresh).astype(float)
        ic, n = spearman_ic(signal.values, combined['return_5d'].values)
        signal_rate = signal.mean() * 100
        above_rsi_results.append({'threshold': thresh, 'ic': ic, 'n': n, 'signal_rate': signal_rate})
        print(f"  {'>'+str(thresh):<15} {ic:>8.4f} {n:>10,} {signal_rate:>7.1f}%")

    # ============================================================
    # 3. WR 超卖阈值扫描
    # ============================================================
    print("\n" + "=" * 65)
    print("📊 3. WR (Williams%R) 超卖阈值 IC 分析")
    print("    因子: williams_r")
    print("    信号1: williams_r < threshold → 超卖加分")
    print("    信号2: williams_r < threshold → 剔除（<=-95 当前）")
    print("    注意: IC/IR显示 WR<-80 加分有效(IC=+0.015)")
    print("=" * 65)

    wr_thresholds = [-70, -80, -85, -90, -95]
    wr_results, wr_best = analyze_threshold(
        "WR 超卖阈值 (williams_r < X)", combined,
        'williams_r', wr_thresholds, direction='below'
    )

    # 也看看 > threshold（WR 不是超卖）是否有更好 IC
    print(f"\n  --- 补充：WR 偏高区间（反向信号）---")
    above_wr_thresholds = [-60, -50, -40, -20]
    above_wr_results = []
    for thresh in above_wr_thresholds:
        signal = (combined['williams_r'] > thresh).astype(float)
        ic, n = spearman_ic(signal.values, combined['return_5d'].values)
        signal_rate = signal.mean() * 100
        above_wr_results.append({'threshold': thresh, 'ic': ic, 'n': n, 'signal_rate': signal_rate})
        print(f"  {'>'+str(thresh):<15} {ic:>8.4f} {n:>10,} {signal_rate:>7.1f}%")

    # ============================================================
    # 年度 IC 稳定性分析（按年份分解）
    # ============================================================
    print("\n" + "=" * 65)
    print("📅 关键阈值年度 IC 稳定性分析")
    print("=" * 65)

    yearly_keys = [
        ('vol_ratio > 1.2', 'vol_ratio', 1.2, 'above'),
        ('vol_ratio > 1.5', 'vol_ratio', 1.5, 'above'),
        ('rsi_14 < 30', 'rsi_14', 30, 'below'),
        ('rsi_14 < 40', 'rsi_14', 40, 'below'),
        ('williams_r < -80', 'williams_r', -80, 'below'),
        ('williams_r < -90', 'williams_r', -90, 'below'),
    ]

    print(f"\n  {'阈值':<20} {'2018':>7} {'2019':>7} {'2020':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'Avg':>7}")
    print(f"  {'-'*80}")

    stability_results = {}
    for label, col, thresh, direction in yearly_keys:
        year_ics = []
        for year in YEARS:
            if year in [2025]:  # skip incomplete years
                continue
            df_yr = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet') if os.path.exists(f'{DATA_DIR}/technical_indicators_year={year}/data.parquet') else None
            if df_yr is None:
                continue
            daily_yr = pd.read_parquet(f'{DATA_DIR}/daily_data_year={year}/data.parquet')
            df_yr = pd.merge(df_yr, daily_yr, on=['date', 'symbol'], how='inner')
            df_yr['return_5d'] = df_yr.groupby('symbol')['close'].pct_change(5).shift(-5)
            df_yr['vol_ratio'] = df_yr['volume'] / (df_yr['vol_ma5'] + 1e-10)
            if direction == 'above':
                signal = (df_yr[col] > thresh).astype(float)
            else:
                signal = (df_yr[col] < thresh).astype(float)
            ic, _ = spearman_ic(signal.values, df_yr['return_5d'].values)
            year_ics.append((year, ic))
            del df_yr, daily_yr; gc.collect()

        # Print row
        row_str = f"  {label:<20}"
        ic_vals = []
        for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            yr_ic = next((ic for y, ic in year_ics if y == year), np.nan)
            row_str += f" {yr_ic:>7.3f}" if not np.isnan(yr_ic) else f" {'N/A':>7}"
            ic_vals.append(yr_ic)
        avg_ic = np.nanmean(ic_vals)
        row_str += f" {avg_ic:>7.3f}"
        print(row_str)
        stability_results[label] = {'yearly': year_ics, 'avg': avg_ic}

    # ============================================================
    # 最终汇总报告
    # ============================================================
    print("\n" + "=" * 65)
    print("🏆 V8 策略阈值 IC 分析最终结论")
    print("=" * 65)

    print(f"""
volume_condition 阈值 IC 结论：
  • >0.8: IC={next((r['ic'] for r in vol_results if r['threshold']==0.8), np.nan):.4f}
  • >1.0: IC={next((r['ic'] for r in vol_results if r['threshold']==1.0), np.nan):.4f}
  • >1.2: IC={next((r['ic'] for r in vol_results if r['threshold']==1.2), np.nan):.4f} ← 当前
  • >1.5: IC={next((r['ic'] for r in vol_results if r['threshold']==1.5), np.nan):.4f}
  • >2.0: IC={next((r['ic'] for r in vol_results if r['threshold']==2.0), np.nan):.4f}
★ 最优阈值：>{(vol_best['threshold'] if vol_best else '?')}  IC={vol_best['ic']:.4f if vol_best else 'N/A'}

RSI 超卖 IC 结论：
  • <20: IC={next((r['ic'] for r in rsi_results if r['threshold']==20), np.nan):.4f}
  • <25: IC={next((r['ic'] for r in rsi_results if r['threshold']==25), np.nan):.4f} ← 当前（剔除）
  • <30: IC={next((r['ic'] for r in rsi_results if r['threshold']==30), np.nan):.4f}
  • <35: IC={next((r['ic'] for r in rsi_results if r['threshold']==35), np.nan):.4f}
  • <40: IC={next((r['ic'] for r in rsi_results if r['threshold']==40), np.nan):.4f}
★ 最优阈值：<{(rsi_best['threshold'] if rsi_best else '?')}  IC={rsi_best['ic']:.4f if rsi_best else 'N/A'}
  ⚠️ 验证 RSI 超卖是否陷阱：需对比非超卖区间 IC

WR 超卖 IC 结论：
  • <-70: IC={next((r['ic'] for r in wr_results if r['threshold']==-70), np.nan):.4f}
  • <-80: IC={next((r['ic'] for r in wr_results if r['threshold']==-80), np.nan):.4f} ← 当前（加分）
  • <-85: IC={next((r['ic'] for r in wr_results if r['threshold']==-85), np.nan):.4f}
  • <-90: IC={next((r['ic'] for r in wr_results if r['threshold']==-90), np.nan):.4f}
  • <-95: IC={next((r['ic'] for r in wr_results if r['threshold']==-95), np.nan):.4f} ← 当前（剔除）
★ 最优阈值：<{(wr_best['threshold'] if wr_best else '?')}  IC={wr_best['ic']:.4f if wr_best else 'N/A'}
""")

    # 保存结果
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = f'{OUT_DIR}/v8_threshold_ic_results.json'
    import json
    all_results = {
        'vol_results': vol_results,
        'rsi_results': rsi_results,
        'wr_results': wr_results,
        'stability': {k: {'yearly': v['yearly'], 'avg': v['avg']} for k, v in stability_results.items()}
    }
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"💾 结果已保存: {out_file}")


if __name__ == '__main__':
    main()
