#!/usr/bin/env python3
"""
重建历史年份的 money_flow_indicators 并写入 technical_indicators parquet
============================================================================
用法：
  python3 scripts/rebuild_money_flow.py --year 2024    # 重建单一年份
  python3 scripts/rebuild_money_flow.py --years 2024 2025  # 重建多年

原理：
  money_flow_trend 等指标的计算只依赖 OHLCV + total_shares，
  这些在 daily_data_year=YYYY/data.parquet 和 stock_basic_info.parquet 中已完整保存。
  计算后只向 technical_indicators parquet 追加新的 money_flow 列，不改动原有数据。
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import calculate_money_flow_indicators

DATA_DIR = Path('/root/.openclaw/workspace/data/warehouse')


def rebuild_year(year: int) -> dict:
    """
    重建单一年份的 money_flow_indicators 并写入 technical_indicators parquet

    Returns:
        dict: 统计信息
    """
    print(f"\n{'='*60}")
    print(f"📅 重建 {year} 年 money_flow_indicators")
    print(f"{'='*60}")

    tech_path = DATA_DIR / f'technical_indicators_year={year}' / 'data.parquet'
    daily_path = DATA_DIR / f'daily_data_year={year}' / 'data.parquet'
    basic_path = DATA_DIR / 'stock_basic_info.parquet'

    if not tech_path.exists():
        print(f"  ⚠️ 技术指标文件不存在: {tech_path}")
        return {'year': year, 'status': 'skip', 'reason': 'tech_file_missing'}
    if not daily_path.exists():
        print(f"  ⚠️ 日线数据文件不存在: {daily_path}")
        return {'year': year, 'status': 'skip', 'reason': 'daily_file_missing'}

    # ── 1. 加载数据 ──────────────────────────────────────────
    print(f"  [1/4] 加载数据...")
    tech = pd.read_parquet(tech_path)
    daily = pd.read_parquet(daily_path)
    basic = pd.read_parquet(basic_path)[['symbol', 'total_shares']]

    # 只保留需要的日线列（避免重复）
    daily_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    daily = daily[daily_cols].copy()

    # ── 2. 合并 total_shares ────────────────────────────────
    print(f"  [2/4] 合并 total_shares...")
    df = pd.merge(daily, basic, on='symbol', how='left')

    # ── 3. 计算资金流指标 ───────────────────────────────────
    print(f"  [3/4] 计算 money_flow_indicators (约{len(df):,}行)...")
    df = calculate_money_flow_indicators(df)
    gc.collect()

    # ── 4. 提取 money_flow 列，合并写入 tech parquet ───────
    mf_cols = [
        'date', 'symbol',
        'XVL', 'LIJIN', 'LLJX', '主生量', '量基线', '量增幅',
        'money_flow_positive', 'money_flow_trend',
        'money_flow_increasing', 'money_flow_weekly', 'money_flow_weekly_increasing',
    ]
    mf_cols = [c for c in mf_cols if c in df.columns]
    mf_df = df[mf_cols].copy()

    # 检查是否已有 money_flow_trend 列
    existing_cols = set(tech.columns)
    has_mf = 'money_flow_trend' in existing_cols

    if has_mf:
        # 已有列 → 覆盖更新
        cols_to_drop = [c for c in mf_cols if c in tech.columns and c not in ('date', 'symbol')]
        tech = tech.drop(columns=cols_to_drop)
        print(f"  [4/4] 覆盖更新已有 money_flow 列: {cols_to_drop}")
    else:
        print(f"  [4/4] 新增 money_flow 列: {[c for c in mf_cols if c not in ('date','symbol')]}")

    # 合并
    tech = pd.merge(tech, mf_df, on=['date', 'symbol'], how='left')

    # 写入（覆盖原文件）
    tech_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(tech)
    n_cols = len(tech.columns)
    new_mf = tech['money_flow_trend'].notna().mean()
    tech.to_parquet(tech_path, index=False)
    print(f"  ✅ 写入完成: {tech_path} ({n_rows:,}行, {n_cols}列)")
    print(f"  验证: money_flow_trend 非空率 = {new_mf:.1%}")

    del df, tech, daily, mf_df
    gc.collect()

    return {
        'year': year,
        'status': 'ok',
        'n_rows': n_rows,
        'money_flow_trend_coverage': new_mf,
    }


def main():
    parser = argparse.ArgumentParser(description='重建历史年份 money_flow_indicators')
    parser.add_argument('--years', type=int, nargs='+', default=[2024],
                        help='要重建的年份，如 --years 2024 2025')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印计划，不实际写入')
    args = parser.parse_args()

    results = []
    for year in sorted(args.years):
        if args.dry_run:
            print(f"  [DRY RUN] 会重建 {year} 年")
        else:
            r = rebuild_year(year)
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print("📊 汇总")
        for r in results:
            if r['status'] == 'ok':
                print(f"  {r['year']}: ✅ money_flow_trend 非空率 {r['money_flow_trend_coverage']:.1%}")
            else:
                print(f"  {r['year']}: ⚠️  {r.get('reason','?')}")


if __name__ == '__main__':
    main()
