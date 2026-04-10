#!/usr/bin/env python3
"""
重建波浪缓存 L1（2021-2024 冷数据）

整合自：
- batch_wave_builder.py (Phase 3.1 批量构建)
- utils/wavechan_cache.py (L1/L2 分层存储)

用法：
    # 重建指定年份
    python scripts/rebuild_wavechan_l1.py --years 2022 2023 2024
    
    # 重建单年
    python scripts/rebuild_wavechan_l1.py --years 2024
    
    # 查看进度
    python scripts/rebuild_wavechan_l1.py --progress --years 2022
"""

import argparse
import logging
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.wavechan.v3_l2_cache.strategy import BatchWaveBuilder
from utils.wavechan_cache import L1L2ParquetCache, CACHE_BASE
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

DEFAULT_DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
DEFAULT_CACHE_BASE = '/data/warehouse/wavechan/wavechan_cache'
DEFAULT_BATCH_SIZE = 100

# ============================================================
# 重建函数
# ============================================================

def rebuild_year(year: int,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 data_dir: str = DEFAULT_DATA_DIR,
                 cache_base: str = DEFAULT_CACHE_BASE,
                 resume: bool = False) -> int:
    from pathlib import Path
    """
    重建指定年份的 L1 缓存
    
    Args:
        year: 年份
        batch_size: 每批处理股票数
        data_dir: 数据目录
        cache_base: 缓存基础目录
        resume: 是否断点续建
    
    Returns:
        处理的股票数量
    """
    l1l2 = L1L2ParquetCache(Path(cache_base))
    l1_path = l1l2._l1_path(year)
    
    # 删除旧缓存（除非断点续建）
    if l1_path.exists() and not resume:
        logger.info(f"删除旧缓存：{l1_path}")
        shutil.rmtree(l1_path)
    
    # 临时工作目录
    temp_cache_dir = f'/tmp/wavechan_rebuild_{year}'
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    # 进度文件
    progress_file = os.path.join(temp_cache_dir, '_progress.json')
    
    # 创建 Builder
    builder = BatchWaveBuilder(cache_dir=temp_cache_dir, data_dir=data_dir)
    
    if not builder.symbols:
        logger.error(f"未找到任何股票数据，请检查：{data_dir}")
        return 0
    
    logger.info(f"[{year}] 共找到 {len(builder.symbols):,} 只股票")
    
    # 日期范围
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    # 批量构建
    results = builder.run_batch(
        symbols=None,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        progress_file=progress_file
    )
    
    # 合并信号
    all_signals = []
    for cache in results:
        if cache.signal_history:
            for sig in cache.signal_history:
                sig['symbol'] = cache.symbol
                all_signals.append(sig)
    
    # 写入 L1 缓存
    if all_signals:
        df = pd.DataFrame(all_signals)
        l1l2.write_l1(year, df)
        logger.info(f"[{year}] ✅ 写入 L1: {len(df):,} 行")
    else:
        logger.warning(f"[{year}] ⚠️ 无信号数据")
    
    # 清理临时目录
    shutil.rmtree(temp_cache_dir)
    logger.info(f"清理临时目录：{temp_cache_dir}")
    
    return len(results)


def rebuild_years(years: list, **kwargs):
    """重建多个年份"""
    total = 0
    completed = []
    for year in years:
        logger.info("=" * 60)
        logger.info(f"开始重建：{year}")
        logger.info("=" * 60)
        
        count = rebuild_year(year, **kwargs)
        total += count
        completed.append((year, count))
        
        logger.info(f"[{year}] 完成，处理 {count:,} 只股票\n")
        
        # 每完成一年，更新任务看板
        try:
            completed_str = ", ".join([f"{y}({c:,})" for y, c in completed])
            os.system(f"bash /root/.openclaw/team-fairy/scripts/update_todo.sh --agent main --task 'V3 波浪缓存重建 (2018-2025 共 8 年)' --status in-progress --note '已完成：{completed_str}'")
        except Exception as e:
            logger.warning(f"更新任务看板失败：{e}")
    
    logger.info("=" * 60)
    logger.info(f"全部完成！总计 {total:,} 只股票")
    logger.info("=" * 60)


def show_progress(years: list, cache_base: str = DEFAULT_CACHE_BASE):
    """显示进度"""
    l1l2 = L1L2ParquetCache(Path(cache_base))
    
    for year in years:
        l1_path = l1l2._l1_path(year)
        if l1_path.exists():
            try:
                df = pd.read_parquet(l1_path / "data.parquet")
                symbols = df['symbol'].nunique() if 'symbol' in df.columns else 0
                signals = len(df)
                logger.info(f"[{year}] ✅ L1 缓存：{symbols:,} 只股票，{signals:,} 条信号")
            except Exception as e:
                logger.info(f"[{year}] ⚠️ 读取失败：{e}")
        else:
            logger.info(f"[{year}] ❌ 无缓存")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='重建波浪缓存 L1（冷数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/rebuild_wavechan_l1.py --years 2022 2023 2024  # 重建三年
  python scripts/rebuild_wavechan_l1.py --years 2024           # 重建单年
  python scripts/rebuild_wavechan_l1.py --progress             # 查看进度
  python scripts/rebuild_wavechan_l1.py --years 2022 --resume  # 断点续建
        """
    )
    
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='要重建的年份列表')
    parser.add_argument('--progress', action='store_true',
                        help='显示进度')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'每批处理股票数 (默认{DEFAULT_BATCH_SIZE})')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'数据目录 (默认{DEFAULT_DATA_DIR})')
    parser.add_argument('--cache-base', type=str, default=DEFAULT_CACHE_BASE,
                        help=f'缓存基础目录 (默认{DEFAULT_CACHE_BASE})')
    parser.add_argument('--resume', action='store_true',
                        help='断点续建（不删除已有缓存）')
    
    args = parser.parse_args()
    
    # 默认年份
    if args.years is None:
        args.years = [2022, 2023, 2024]
    
    if args.progress:
        show_progress(args.years, args.cache_base)
        return
    
    rebuild_years(
        years=args.years,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        cache_base=args.cache_base,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
