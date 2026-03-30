#!/usr/bin/env python3
"""
WaveChan 缓存分区迁移脚本

功能：
  将现有的单一 Parquet 缓存（/data/warehouse/wavechan_signals_cache.parquet）
  迁移到三层缓存架构：

  L1 历史归档层（按年分区）- 2021-2024
  L2 热数据层（按年分区）- 2025+
  L3 参数缓存层（SQLite）- 自动初始化

用法：
  python3 wavechan_cache_migrate.py --dry-run
  python3 wavechan_cache_migrate.py
"""

import argparse
import logging
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---- paths ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "strategies"))

from utils.wavechan_cache import (
    WaveChanCacheManager,
    L1_COLD_YEARS,
    L2_HOT_YEAR_START,
    CACHE_BASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- 常量 ----
SOURCE_CACHE = Path("/data/warehouse/wavechan_signals_cache.parquet")
BACKUP_CACHE = Path("/data/warehouse/wavechan_signals_cache_backup.parquet")


def load_source() -> pd.DataFrame:
    """加载源缓存"""
    if not SOURCE_CACHE.exists():
        raise FileNotFoundError(f"源缓存不存在: {SOURCE_CACHE}")

    logger.info(f"加载源缓存: {SOURCE_CACHE}")
    df = pd.read_parquet(SOURCE_CACHE)
    df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    logger.info(f"  记录: {len(df):,} | 股票: {df['symbol'].nunique()} | "
                 f"日期: {df['date'].min()} ~ {df['date'].max()}")
    return df


def migrate_l1(df: pd.DataFrame, cache_mgr: WaveChanCacheManager):
    """迁移 L1 历史归档（2021-2024，按年分区）"""
    logger.info("=" * 60)
    logger.info("迁移 L1 历史归档层（2021-2024）...")

    for year in L1_COLD_YEARS:
        year_df = df[df['date'].str.startswith(str(year))]
        if year_df.empty:
            logger.info(f"  {year}: 无数据，跳过")
            continue

        cache_mgr.l1l2.write_l1(year, year_df)
        logger.info(f"  {year}: {len(year_df):,} 行 → L1")

    logger.info("L1 迁移完成")


def migrate_l2(df: pd.DataFrame, cache_mgr: WaveChanCacheManager):
    """迁移 L2 热数据层（2025+，按年分区）"""
    logger.info("=" * 60)
    logger.info("迁移 L2 热数据层（2025+）...")

    # 2025 按月分区
    year = 2025
    year_df = df[df['date'].str.startswith(str(year))]
    if year_df.empty:
        logger.warning(f"  {year}: 无数据，跳过")
    else:
        # 按月写入
        for month in range(1, 13):
            month_df = year_df[year_df['date'].str[5:7] == f"{month:02d}"]
            if month_df.empty:
                continue
            cache_mgr.l1l2.write_l2(year, month, month_df)
            logger.info(f"  {year}-{month:02d}: {len(month_df):,} 行 → L2")

    # 2026 按月分区
    year = 2026
    year_df = df[df['date'].str.startswith(str(year))]
    if not year_df.empty:
        for month in range(1, 13):
            month_df = year_df[year_df['date'].str[5:7] == f"{month:02d}"]
            if month_df.empty:
                continue
            cache_mgr.l1l2.write_l2(year, month, month_df)
            logger.info(f"  {year}-{month:02d}: {len(month_df):,} 行 → L2")
    else:
        logger.warning(f"  {year}: 无数据，跳过")

    logger.info("L2 迁移完成")


def init_l3(cache_mgr: WaveChanCacheManager):
    """初始化 L3 参数缓存"""
    logger.info("=" * 60)
    logger.info("初始化 L3 参数缓存层...")
    cache_mgr.l3.clear()
    logger.info("L3 初始化完成（空缓存）")


def verify(cache_mgr: WaveChanCacheManager, source_df: pd.DataFrame):
    """验证迁移结果"""
    logger.info("=" * 60)
    logger.info("验证迁移结果...")

    status = cache_mgr.status()
    print(f"\n缓存状态:")
    print(f"  L1 分区数: {len(status['l1_partitions'])}")
    print(f"  L2 分区数: {len(status['l2_partitions'])}")
    print(f"  L2 大小: {status['l2_size_mb']} MB")
    print(f"  L3 条目: {status['l3_count']}")

    # 抽样验证：2023年随机抽查
    logger.info("\n抽样验证：2023年数据...")
    sample_date = "2023-06-15"
    try:
        l1_data = cache_mgr.l1l2.read_l1(2023)
        if not l1_data.empty:
            sample = l1_data[l1_data['date'] == sample_date].head(3)
            logger.info(f"  L1 2023-{sample_date}: {len(sample)} 条样本")
            if not sample.empty:
                logger.info(f"  Sample: {sample[['date','symbol','total_score','signal_type']].to_string()}")
    except Exception as e:
        logger.warning(f"  L1 验证失败: {e}")

    # 2025年抽查
    logger.info("\n抽样验证：2025年数据...")
    try:
        l2_data = cache_mgr.l1l2.read_l2(2025, 6)
        if not l2_data.empty:
            sample = l2_data[l2_data['date'] == f"2025-06-15"].head(3)
            logger.info(f"  L2 2025-06: {len(l2_data):,} 行, 抽查 {sample_date}: {len(sample)} 条")
            if not sample.empty:
                logger.info(f"  Sample: {sample[['date','symbol','total_score','signal_type']].to_string()}")
    except Exception as e:
        logger.warning(f"  L2 验证失败: {e}")

    logger.info("\n✅ 迁移验证完成")


def main():
    parser = argparse.ArgumentParser(description="WaveChan 缓存分区迁移")
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不执行")
    parser.add_argument("--skip-backup", action="store_true", help="跳过备份源文件")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WaveChan 缓存分区迁移")
    logger.info(f"源文件: {SOURCE_CACHE}")
    logger.info(f"目标: {CACHE_BASE}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] 仅显示计划，不执行迁移")
        # 显示源数据概览
        try:
            df = load_source()
            for year in sorted(df['date'].str[:4].unique()):
                ydf = df[df['date'].str.startswith(year)]
                logger.info(f"  {year}: {len(ydf):,} 行, {ydf['symbol'].nunique()} 只股票")
        except FileNotFoundError as e:
            logger.error(str(e))
        return

    t_start = time.time()

    # 1. 加载源数据
    df = load_source()

    # 2. 创建缓存管理器（自动初始化 L3）
    cache_mgr = WaveChanCacheManager()

    # 3. 迁移 L1（如果不存在）
    if not cache_mgr.l1l2._l1_path(2021).exists():
        migrate_l1(df, cache_mgr)
    else:
        logger.info("L1 已存在，跳过迁移")

    # 4. 迁移 L2（如果不存在）
    if not cache_mgr.l1l2._l2_path(2025, 1).exists():
        migrate_l2(df, cache_mgr)
    else:
        logger.info("L2 已存在，跳过迁移")

    # 5. 初始化 L3
    init_l3(cache_mgr)

    # 6. 验证
    verify(cache_mgr, df)

    # 7. 可选：备份源文件
    if not args.skip_backup and BACKUP_CACHE.exists():
        logger.info(f"\n⚠️  备份文件已存在: {BACKUP_CACHE}")
        logger.info("  跳过备份（可手动删除源文件以释放空间）")
    elif not args.skip_backup:
        logger.info("\n备份源文件...")
        shutil.copy2(SOURCE_CACHE, BACKUP_CACHE)
        logger.info(f"  备份完成: {BACKUP_CACHE}")

    elapsed = time.time() - t_start
    logger.info(f"\n总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)
    logger.info("✅ 迁移完成!")
    logger.info(f"  旧缓存: {SOURCE_CACHE} （可手动删除）")
    logger.info(f"  新缓存: {CACHE_BASE}")
    logger.info(f"  备份: {BACKUP_CACHE}")


if __name__ == "__main__":
    main()
