#!/usr/bin/env python3
"""
BatchWaveBuilder - 批量波浪构建脚本 (Phase 3.1)

功能：
- 读取多只股票日线数据
- 批量计算波浪特征
- 结果缓存到 /root/.openclaw/workspace/data/wavechan_cache/
- 分批处理，每次处理100只股票

用法：
    # 全量构建
    python batch_wave_builder.py --full

    # 增量更新（今日）
    python batch_wave_builder.py --incremental

    # 指定股票列表
    python batch_wave_builder.py --symbols 600985 000001 000002

    # 指定日期范围
    python batch_wave_builder.py --start 2025-01-01 --end 2026-03-27
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# 添加 strategies 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategies'))

from wavechan_v3 import BatchWaveBuilder, SymbolWaveCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = '/root/.openclaw/workspace/data/wavechan_cache'
DEFAULT_DATA_DIR = '/root/.openclaw/workspace/data'

# 默认每批股票数量 (Phase 3.1 要求)
DEFAULT_BATCH_SIZE = 100


def run_full(batch_size: int = DEFAULT_BATCH_SIZE,
             cache_dir: str = DEFAULT_CACHE_DIR,
             data_dir: str = DEFAULT_DATA_DIR,
             start_date: str = None,
             end_date: str = None):
    """全量构建所有股票"""
    logger.info(f"开始全量构建: cache_dir={cache_dir}, batch_size={batch_size}")

    builder = BatchWaveBuilder(cache_dir=cache_dir, data_dir=data_dir)

    if not builder.symbols:
        logger.error("未找到任何股票数据，请检查数据目录")
        return

    progress_file = os.path.join(cache_dir, '_progress_full.json')
    results = builder.run_batch(
        symbols=None,  # 全市场
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        progress_file=progress_file
    )

    logger.info(f"全量构建完成: {len(results)} 只股票")
    return results


def run_incremental(date: str = None,
                     batch_size: int = DEFAULT_BATCH_SIZE,
                     cache_dir: str = DEFAULT_CACHE_DIR,
                     data_dir: str = DEFAULT_DATA_DIR):
    """增量更新指定日期的数据"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"增量更新: date={date}, batch_size={batch_size}")

    builder = BatchWaveBuilder(cache_dir=cache_dir, data_dir=data_dir)
    results = builder.run_incremental(date=date, batch_size=batch_size)

    logger.info(f"增量更新完成: {len(results)} 只股票")
    return results


def run_symbols(symbols: list,
                batch_size: int = DEFAULT_BATCH_SIZE,
                cache_dir: str = DEFAULT_CACHE_DIR,
                data_dir: str = DEFAULT_DATA_DIR,
                start_date: str = None,
                end_date: str = None):
    """构建指定股票列表"""
    logger.info(f"构建指定股票: {len(symbols)} 只")

    builder = BatchWaveBuilder(cache_dir=cache_dir, data_dir=data_dir)
    progress_file = os.path.join(cache_dir, '_progress_symbols.json')
    results = builder.run_batch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        progress_file=progress_file
    )

    logger.info(f"指定股票构建完成: {len(results)} 只")
    return results


def show_progress(cache_dir: str = DEFAULT_CACHE_DIR):
    """显示构建进度"""
    progress_file = os.path.join(cache_dir, '_progress_full.json')

    if not os.path.exists(progress_file):
        logger.info("无进度记录")
        return

    import json
    with open(progress_file, 'r') as f:
        data = json.load(f)
    done = set(data.get('done', []))

    builder = BatchWaveBuilder(cache_dir=cache_dir)
    total = len(builder.symbols)

    logger.info(f"进度: {len(done)}/{total} ({100*len(done)/total:.1f}%)")

    if done:
        logger.info(f"已完成示例: {list(done)[:5]}")
    if total > len(done):
        remaining = set(builder.symbols) - done
        logger.info(f"剩余示例: {list(remaining)[:5]}")


def main():
    parser = argparse.ArgumentParser(
        description='WaveChan V3 批量波浪构建脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_wave_builder.py --full                  # 全量构建
  python batch_wave_builder.py --incremental          # 增量更新今日
  python batch_wave_builder.py --incremental --date 2026-03-26  # 增量更新指定日
  python batch_wave_builder.py --symbols 600985 000001  # 指定股票
  python batch_wave_builder.py --progress              # 查看进度
  python batch_wave_builder.py --batch-size 50        # 每批50只(默认100)
        """
    )

    parser.add_argument('--full', action='store_true',
                        help='全量构建所有股票')
    parser.add_argument('--incremental', action='store_true',
                        help='增量更新(默认今日)')
    parser.add_argument('--date', type=str, default=None,
                        help='增量更新的日期 (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        help='指定股票代码列表')
    parser.add_argument('--progress', action='store_true',
                        help='显示构建进度')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'每批处理股票数量 (默认{DEFAULT_BATCH_SIZE})')
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'缓存目录 (默认{DEFAULT_CACHE_DIR})')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'数据目录 (默认{DEFAULT_DATA_DIR})')
    parser.add_argument('--start', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.progress:
        show_progress(cache_dir=args.cache_dir)
        return

    if args.full:
        run_full(
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            start_date=args.start,
            end_date=args.end
        )
    elif args.incremental:
        run_incremental(
            date=args.date,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir
        )
    elif args.symbols:
        run_symbols(
            symbols=args.symbols,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            start_date=args.start,
            end_date=args.end
        )
    else:
        parser.print_help()
        print("\n请指定 --full, --incremental, --symbols 或 --progress")


if __name__ == '__main__':
    main()
