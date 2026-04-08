#!/usr/bin/env python3
"""
WaveChan Weekly Batch - 周线波浪预计算入库
==========================================

功能：
- 每日收盘后增量计算全市场周线波浪状态，存入 parquet
- 供 filter_buy 直接读取，避免重复计算

存储路径：/data/warehouse/wavechan_weekly/{year}/week_wave_{year}.parquet

输出字段：
    symbol, week_end_date, weekly_dir, state,
    w1_start, w1_end, w2_end, w3_end, w4_end, w5_end,
    w3_start, fib_382, fib_500, fib_618, fib_target,
    bi_count, bi_direction, bi_high, bi_low,
    last_fx_mark, last_fx_price, last_fx_date,
    wave_end_signal, wave_end_confidence,
    w5_divergence, w5_failed, w5_ending_diagonal,
    stop_loss, stop_loss_type,
    last_bi_close

用法：
    # 全量构建（近3年）
    python wavechan_weekly_batch.py --full

    # 增量更新（今日）
    python wavechan_weekly_batch.py --incremental

    # 增量更新指定日期
    python wavechan_weekly_batch.py --incremental --date 2026-04-04

    # 测试模式（5只股票）
    python wavechan_weekly_batch.py --test

    # 指定日期范围
    python wavechan_weekly_batch.py --start 2024-01-01 --end 2026-04-04
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 添加 strategies 目录到 path
STRATEGY_DIR = Path(__file__).parent / 'strategies'
sys.path.insert(0, str(STRATEGY_DIR))

from czsc import CZSC, RawBar, Freq
from wavechan.v3_l2_cache.strategy import SymbolWaveCache, WaveSnapshot, BiRecord

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 数据路径配置
# 注意：实际数据仓库在 /root/.openclaw/workspace/data/warehouse/
# /data/warehouse/ 是符号链接或不同挂载点，兼容两种路径
WAREHOUSE_DIR = Path('/root/.openclaw/workspace/data/warehouse')
if not WAREHOUSE_DIR.exists():
    WAREHOUSE_DIR = Path('/data/warehouse')
WEEKLY_WAREHOUSE = WAREHOUSE_DIR / 'wavechan_weekly'
DAILY_DATA_DIR = WAREHOUSE_DIR / 'daily_data_year={year}' / 'data.parquet'

# 默认缓存目录
DEFAULT_CACHE_DIR = '/root/.openclaw/workspace/data/wavechan_cache'

# 默认每批股票数量
DEFAULT_BATCH_SIZE = 100


# ======================
# 核心函数
# ======================

def aggregate_daily_to_weekly(daily_bars: List[dict]) -> List[dict]:
    """
    将日线 bars 聚合为周线 bars

    Args:
        daily_bars: [{date, open, high, low, close, volume}, ...]

    Returns:
        weekly_bars: [{date, open, high, low, close, volume}, ...]
        - date: 取该周最后一天的日期（即周收盘日，周五或月末）
    """
    if not daily_bars:
        return []

    df = pd.DataFrame(daily_bars)
    df['date'] = pd.to_datetime(df['date'])

    # 按周分组，取每周最后一个交易日
    df['week_end'] = df['date'].dt.to_period('W').apply(lambda x: x.end_time.date())
    df['week_start'] = df['date'].dt.to_period('W').apply(lambda x: x.start_time.date())

    weekly = df.groupby('week_end').agg({
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'week_start': 'first',
    }).reset_index()

    weekly.columns = ['week_end_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'week_start_date']
    # week_end_date is already a datetime.date after to_period().end_time.date()
    weekly['week_end_date'] = weekly['week_end_date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
    weekly['week_start_date'] = weekly['week_start_date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))

    return weekly.to_dict('records')


def _aggregate_weekly_bar_from_daily(daily_bars: List[dict]) -> Optional[dict]:
    """
    将日线 bars 聚合为一根周线 bar
    用于增量计算：取最后 N 根日线聚合

    Args:
        daily_bars: 日线 bars

    Returns:
        weekly_bar: {date, open, high, low, close, volume}
    """
    if not daily_bars or len(daily_bars) < 1:
        return None

    bars_df = pd.DataFrame(daily_bars)
    bars_df['date'] = pd.to_datetime(bars_df['date'])

    return {
        'date': bars_df['date'].max().strftime('%Y-%m-%d'),
        'open': float(bars_df['open'].iloc[0]),
        'high': float(bars_df['high'].max()),
        'low': float(bars_df['low'].min()),
        'close': float(bars_df['close'].iloc[-1]),
        'volume': float(bars_df['volume'].sum()),
    }


def compute_weekly_wave(symbol: str,
                        daily_bars: List[dict],
                        cache_dir: str = None) -> List[dict]:
    """
    计算一只股票的周线波浪序列

    Args:
        symbol: 股票代码
        daily_bars: 日线 bars（按日期升序）
        cache_dir: 缓存目录

    Returns:
        weekly_records: [{week_end_date, weekly_dir, state, w1_end, ...}, ...]
    """
    if not daily_bars or len(daily_bars) < 9:
        return []

    # 聚合日线为周线
    weekly_bars = aggregate_daily_to_weekly(daily_bars)
    if not weekly_bars:
        return []

    # 创建周线波浪缓存
    cache_dir = cache_dir or f"/tmp/wavechan_weekly_cache/{symbol}"
    os.makedirs(cache_dir, exist_ok=True)

    weekly_cache = SymbolWaveCache(f"{symbol}_WEEKLY", cache_dir)
    weekly_cache.load()

    records = []

    for wbar in weekly_bars:
        # feed_bar expects bar['date'], not bar['week_end_date']
        wbar_feed = {
            'date': wbar['week_end_date'],
            'open': wbar['open'],
            'high': wbar['high'],
            'low': wbar['low'],
            'close': wbar['close'],
            'volume': wbar['volume'],
        }
        snap = weekly_cache.feed_bar(wbar_feed, freq=Freq.W)
        snap = weekly_cache.counter.get_snapshot()
        snap.date = wbar['week_end_date']

        # 获取最后一笔的收盘价
        last_bi_close = None
        if weekly_cache.completed_bis:
            last_bi = weekly_cache.completed_bis[-1]
            last_bi_close = last_bi.end_price

        # 判断周线方向
        weekly_dir = _infer_weekly_dir(weekly_cache)

        record = {
            'symbol': symbol,
            'week_end_date': wbar['week_end_date'],
            'weekly_dir': weekly_dir,
            'state': snap.state,
            'w1_start': snap.w1_start,
            'w1_end': snap.w1_end,
            'w2_end': snap.w2_end,
            'w3_end': snap.w3_end,
            'w4_end': snap.w4_end,
            'w5_end': snap.w5_end,
            'w3_start': snap.w3_start,
            'fib_382': snap.fib_382,
            'fib_500': snap.fib_500,
            'fib_618': snap.fib_618,
            'fib_target': snap.fib_target,
            'bi_count': snap.bi_count,
            'bi_direction': snap.bi_direction,
            'bi_high': snap.bi_high,
            'bi_low': snap.bi_low,
            'last_fx_mark': snap.last_fx_mark,
            'last_fx_price': snap.last_fx_price,
            'last_fx_date': snap.last_fx_date,
            'wave_end_signal': snap.wave_end_signal,
            'wave_end_confidence': snap.wave_end_confidence,
            'w5_divergence': snap.w5_divergence,
            'w5_failed': snap.w5_failed,
            'w5_ending_diagonal': snap.w5_ending_diagonal,
            'stop_loss': snap.stop_loss,
            'stop_loss_type': snap.stop_loss_type,
            'last_bi_close': last_bi_close,
        }
        records.append(record)

    return records


def _infer_weekly_dir(weekly_cache: SymbolWaveCache) -> str:
    """
    判断周线方向（复用 WaveCounterV3 状态 + 笔序列分析）
    """
    try:
        counter = weekly_cache.counter
        state = counter.state
        bis = weekly_cache.completed_bis

        # ── 优先使用 WaveCounterV3 波浪状态 ─────────────────────────
        if state in ('w3_formed', 'w4_formed', 'w4_in_progress'):
            return 'up'
        elif state == 'w5_formed':
            if len(bis) >= 1:
                last_bi = bis[-1]
                if last_bi.direction == 'up':
                    return 'neutral'
                else:
                    return 'down'
            return 'neutral'

        # ── 备选：分析已完成笔序列的方向趋势 ─────────────────────
        recent_bis = bis[-5:] if len(bis) >= 5 else bis
        if not recent_bis:
            return 'neutral'

        up_count = sum(1 for b in recent_bis if b.direction == 'up')
        down_count = len(recent_bis) - up_count
        dir_score = up_count - down_count

        if len(recent_bis) >= 2:
            first_bi = recent_bis[0]
            last_bi = recent_bis[-1]
            first_range = (min(first_bi.start_price, first_bi.end_price),
                           max(first_bi.start_price, first_bi.end_price))
            last_range = (min(last_bi.start_price, last_bi.end_price),
                          max(last_bi.start_price, last_bi.end_price))
            overall_up = last_range[0] > first_range[0] and last_range[1] > first_range[1]
            overall_down = last_range[1] < first_range[0]
        else:
            overall_up = overall_down = False

        if dir_score >= 2 or (dir_score >= 1 and overall_up):
            return 'up'
        elif dir_score <= -2 or (dir_score <= -1 and overall_down):
            return 'down'
        else:
            return 'neutral'

    except Exception:
        return 'neutral'


def get_all_symbols(years: List[int] = None) -> List[str]:
    """获取所有股票代码列表"""
    if years is None:
        years = [2024, 2025, 2026]

    symbols = set()
    for y in years:
        path = DAILY_DATA_DIR.format(year=y)
        if Path(path).exists():
            df = pd.read_parquet(path, columns=['symbol'])
            symbols.update(df['symbol'].unique().tolist())
        else:
            logger.warning(f"日线数据不存在: {path}")

    return sorted(symbols)


def load_daily_bars(symbol: str, start_date: str = None, end_date: str = None) -> List[dict]:
    """加载指定股票日线数据"""
    all_bars = []

    # 加载所有年份的数据
    for year_dir in sorted(WAREHOUSE_DIR.glob('daily_data_year=*')):
        if not year_dir.is_dir():
            continue
        parquet_path = year_dir / 'data.parquet'
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        df = df[df['symbol'] == symbol]

        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

        all_bars.append(df)

    if not all_bars:
        return []

    df = pd.concat(all_bars, ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)

    return df.to_dict('records')


def save_weekly_records(records: List[dict], year: int):
    """
    保存周线波浪记录到 parquet，按年分目录

    Args:
        records: 周线波浪记录列表
        year: 年份
    """
    if not records:
        return

    year_dir = WEEKLY_WAREHOUSE / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    output_path = year_dir / f'week_wave_{year}.parquet'

    # 读取现有数据
    existing_df = pd.DataFrame()
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)

    new_df = pd.DataFrame(records)

    if existing_df.empty:
        final_df = new_df
    else:
        # 合并去重：同一 symbol + week_end_date 保留最新
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=['symbol', 'week_end_date'],
            keep='last'
        )
        final_df = final_df.sort_values(['symbol', 'week_end_date']).reset_index(drop=True)

    # 保存
    final_df.to_parquet(output_path, index=False)
    logger.info(f"已保存 {len(new_df)} 条新记录到 {output_path}（共 {len(final_df)} 条）")


# ======================
# 批量处理
# ======================

def run_full(years: List[int] = None,
             batch_size: int = DEFAULT_BATCH_SIZE,
             cache_dir: str = DEFAULT_CACHE_DIR):
    """
    全量构建：计算多年周线波浪

    Args:
        years: 要计算的年份列表，默认 [2023, 2024, 2025, 2026]
        batch_size: 每批处理的股票数量
        cache_dir: 临时缓存目录
    """
    if years is None:
        years = [2023, 2024, 2025, 2026]

    logger.info(f"开始全量构建: years={years}, batch_size={batch_size}")

    symbols = get_all_symbols(years)
    logger.info(f"共 {len(symbols)} 只股票")

    all_records = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if i % batch_size == 0:
            logger.info(f"进度: {i}/{total} ({100*i/total:.1f}%)")

        try:
            # 加载该股票所有日线数据
            start_year = min(years)
            end_date = f"{max(years)}-12-31"
            daily_bars = load_daily_bars(symbol, start_date=f"{start_year}-01-01", end_date=end_date)

            if not daily_bars:
                continue

            records = compute_weekly_wave(symbol, daily_bars, cache_dir=cache_dir)
            all_records.extend(records)

        except Exception as e:
            logger.warning(f"[{symbol}] 计算失败: {e}")

    # 按年保存
    by_year = {}
    for rec in all_records:
        year = int(rec['week_end_date'][:4])
        by_year.setdefault(year, []).append(rec)

    for year, recs in by_year.items():
        save_weekly_records(recs, year)

    logger.info(f"全量构建完成: 共 {len(all_records)} 条周线波浪记录")
    return all_records


def run_incremental(date: str = None,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    cache_dir: str = DEFAULT_CACHE_DIR):
    """
    增量更新：计算指定日期所属周的周线波浪

    逻辑：
    - 确定日期所属周的结束日（周五）
    - 加载该股票近几周日线数据重新计算
    - 只有周五之后的周线才能入库（凝固）

    Args:
        date: 参考日期，默认今天
        batch_size: 每批处理的股票数量
        cache_dir: 临时缓存目录
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    ref_date = pd.to_datetime(date)

    # 检查是否是周五之后（周线凝固条件）
    is_friday_or_after = ref_date.dayofweek >= 4  # Mon=0, Fri=4

    # 计算该周结束日
    week_end = ref_date - timedelta(days=ref_date.dayofweek - 4)  # 定位到上周五
    if ref_date.dayofweek == 4:  # 恰好是周五
        week_end = ref_date  # 今天就是周五

    week_end_str = week_end.strftime('%Y-%m-%d')

    # 只有周五之后才能更新周线（凝固规则）
    if not is_friday_or_after:
        logger.info(f"今天 ({date}) 是 {ref_date.strftime('%A')}，周线尚未凝固，跳过更新")
        return []

    logger.info(f"增量更新: date={date}, week_end={week_end_str}, batch_size={batch_size}")

    symbols = get_all_symbols([ref_date.year])
    logger.info(f"共 {len(symbols)} 只股票需要更新")

    all_records = []
    total = len(symbols)

    # 近3个月日线数据（足够重新计算波浪）
    lookback_days = 90
    start_date = (ref_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    for i, symbol in enumerate(symbols):
        if i % batch_size == 0:
            logger.info(f"进度: {i}/{total} ({100*i/total:.1f}%)")

        try:
            daily_bars = load_daily_bars(symbol, start_date=start_date, end_date=date)
            if not daily_bars:
                continue

            records = compute_weekly_wave(symbol, daily_bars, cache_dir=cache_dir)
            # 只保留目标周的记录
            target_week_recs = [r for r in records if r['week_end_date'] == week_end_str]
            all_records.extend(target_week_recs)

        except Exception as e:
            logger.warning(f"[{symbol}] 计算失败: {e}")

    # 保存
    if all_records:
        year = int(week_end_str[:4])
        save_weekly_records(all_records, year)

    logger.info(f"增量更新完成: {len(all_records)} 条新记录")
    return all_records


def run_test(batch_size: int = 5,
             cache_dir: str = DEFAULT_CACHE_DIR):
    """
    测试模式：选5只股票验证周线方向识别
    """
    logger.info("=" * 60)
    logger.info("测试模式：验证周线波浪计算")
    logger.info("=" * 60)

    # 选5只有代表性的股票
    test_symbols = ['000001', '000002', '600519', '600985', '601318']

    all_records = []

    for symbol in test_symbols:
        try:
            # 加载近3年日线
            daily_bars = load_daily_bars(
                symbol,
                start_date='2023-01-01',
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if not daily_bars:
                logger.warning(f"[{symbol}] 无日线数据")
                continue

            logger.info(f"[{symbol}] 加载 {len(daily_bars)} 根日线")

            records = compute_weekly_wave(symbol, daily_bars, cache_dir=cache_dir)

            if records:
                latest = records[-1]
                logger.info(f"  → 最新周线: week={latest['week_end_date']}, "
                           f"dir={latest['weekly_dir']}, state={latest['state']}, "
                           f"w1={latest['w1_end']}, w2={latest['w2_end']}, "
                           f"w3={latest['w3_end']}, w4={latest['w4_end']}, "
                           f"w5={latest['w5_end']}")
                logger.info(f"  → bi_count={latest['bi_count']}, "
                           f"last_fx={latest['last_fx_mark']}@{latest['last_fx_price']}, "
                           f"wave_end={latest['wave_end_signal']}")
            else:
                logger.warning(f"[{symbol}] 无周线波浪记录（数据不足）")

            all_records.extend(records)

        except Exception as e:
            logger.error(f"[{symbol}] 测试失败: {e}")

    # 保存测试结果
    if all_records:
        by_year = {}
        for rec in all_records:
            year = int(rec['week_end_date'][:4])
            by_year.setdefault(year, []).append(rec)

        for year, recs in by_year.items():
            save_weekly_records(recs, year)

        # 打印统计
        for symbol in test_symbols:
            symbol_recs = [r for r in all_records if r['symbol'] == symbol]
            if symbol_recs:
                dir_counts = pd.Series([r['weekly_dir'] for r in symbol_recs]).value_counts()
                logger.info(f"[{symbol}] 周线方向分布: {dict(dir_counts)}")

    logger.info(f"测试完成: 共 {len(all_records)} 条记录")
    return all_records


# ======================
# CLI
# ======================

def main():
    parser = argparse.ArgumentParser(description='WaveChan Weekly Batch - 周线波浪预计算入库')
    parser.add_argument('--full', action='store_true', help='全量构建（近3年）')
    parser.add_argument('--incremental', action='store_true', help='增量更新')
    parser.add_argument('--test', action='store_true', help='测试模式（5只股票）')
    parser.add_argument('--date', type=str, default=None, help='增量更新指定日期 (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, default=None, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--years', type=int, nargs='+', default=[2023, 2024, 2025, 2026],
                        help='全量构建的年份列表')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'每批处理的股票数量 (默认 {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'缓存目录 (默认 {DEFAULT_CACHE_DIR})')

    args = parser.parse_args()

    if args.test:
        run_test(batch_size=5, cache_dir=args.cache_dir)
    elif args.full:
        run_full(years=args.years, batch_size=args.batch_size, cache_dir=args.cache_dir)
    elif args.incremental:
        run_incremental(date=args.date, batch_size=args.batch_size, cache_dir=args.cache_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
