#!/usr/bin/env python3
"""
WaveChan 信号预计算缓存生成器

功能：
  预计算全市场所有交易日的 WaveChan 信号评分，
  输出 Parquet 缓存文件供 Optuna 优化加速。

输出路径：/data/warehouse/wavechan_signals_cache.parquet

信号字段：
  - date, symbol
  - 信号评分维度：signal_score, structure_score, momentum_score, chan_score
  - 总分：total_score
  - 是否有信号：has_signal
  - 详细：wave_state, rsi, macd底背离, volume_ratio 等

用法：
  python3 wavechan_signal_precompute.py
  python3 wavechan_signal_precompute.py --start 2020-01-01 --end 2026-03-27
  python3 wavechan_signal_precompute.py --years 2021 2022 2023 2024 2025
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---- paths ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "strategies"))

from strategies.wavechan_selector import WaveChanSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ---- 常量 ----
WAREHOUSE_PATH = Path("/root/.openclaw/workspace/data/warehouse")
OUTPUT_PATH = Path("/data/warehouse/wavechan_signals_cache.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 预热期（波浪分析需要）
LOOKBACK_DAYS = 120  # 确保波浪分析有足够历史

# 默认计算年份（最近5年，数据量与质量平衡）
DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]

# ---- 缓存字段（按任务要求）----
OUTPUT_COLUMNS = [
    # 基础
    "date",
    "symbol",
    # 评分维度
    "signal_score",      # 信号评分（40%权重）
    "structure_score",   # 结构评分（30%权重）
    "momentum_score",    # 动能评分（20%权重）
    "chan_score",        # 缠论评分（10%权重）== chanlun_score
    # 总分
    "total_score",
    # 信号判断
    "has_signal",
    "signal_type",
    "signal_status",
    # 详细指标
    "wave_state",         # wave_stage
    "wave_trend",
    "wave_retracement",
    "rsi",
    "macd_hist",
    "divergence",         # MACD底背离
    "volume_ratio",
    "fractal",
    "bottom_div",
    "stop_loss",
    # 价格（辅助）
    "close",
    "open",
    "high",
    "low",
    "volume",
]


def load_all_daily_data(years: list) -> pd.DataFrame:
    """加载指定年份的所有日线数据"""
    logger.info(f"加载日线数据: years={years}")
    dfs = []
    for yr in years:
        pf = WAREHOUSE_PATH / f"daily_data_year={yr}" / "data.parquet"
        if not pf.exists():
            logger.warning(f"数据文件不存在: {pf}")
            continue
        df = pd.read_parquet(pf)
        dfs.append(df)
        logger.info(f"  {yr}: {df['symbol'].nunique()} 只股票, {len(df)} 行")

    if not dfs:
        raise FileNotFoundError("没有任何日线数据！")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"合计: {data['symbol'].nunique()} 只股票, {len(data)} 行, "
                 f"日期 {data['date'].min()} ~ {data['date'].max()}")
    return data


def precompute_wavechan_signals(
    data: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
    batch_log: int = 200,
) -> pd.DataFrame:
    """
    预计算全市场 WaveChan 信号评分

    Args:
        data: 所有股票的日线数据（已拼接多年）
        lookback_days: 预热期天数
        batch_log: 每处理多少只股票打印一次进度

    Returns:
        DataFrame: 所有评分字段
    """
    selector = WaveChanSelector()

    # 获取所有股票列表
    all_symbols = sorted(data["symbol"].unique())
    total_symbols = len(all_symbols)
    logger.info(f"开始预计算: {total_symbols} 只股票")

    # 预热截止日期（跳过这段时间的输出）
    # lookback_days 前的日期作为正式输出范围
    all_dates = sorted(data["date"].unique())
    if len(all_dates) < lookback_days:
        warmup_end_date = all_dates[0]
    else:
        warmup_end_date = all_dates[lookback_days]

    logger.info(f"预热截止: {warmup_end_date}（该日期之前的数据仅用于计算指标，不输出）")

    results = []
    errors = 0
    t0 = time.time()

    for idx, sym in enumerate(all_symbols):
        # 进度打印
        if idx % batch_log == 0 or idx == total_symbols - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_symbols - idx - 1) / rate if rate > 0 else 0
            logger.info(f"  进度: {idx+1}/{total_symbols} ({100*(idx+1)/total_symbols:.1f}%) | "
                         f"速度: {rate:.0f} sym/s | 剩余: {eta:.0f}s | 错误: {errors}")

        # 取该股票全部历史
        sym_df = data[data["symbol"] == sym].sort_values("date").reset_index(drop=True)

        if len(sym_df) < 60:
            continue

        try:
            scored = selector._compute_symbol_scores(sym, sym_df)
            if scored is None or scored.empty:
                continue

            # 过滤预热期
            scored = scored[scored["date"] > str(warmup_end_date)].copy()
            if scored.empty:
                continue

            results.append(scored)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  [{sym}] 处理异常: {e}")
            continue

    if not results:
        logger.warning("没有任何评分结果！")
        return pd.DataFrame()

    logger.info(f"合并结果: {len(results)} 只股票有评分数据")

    # 合并
    df = pd.concat(results, ignore_index=True)

    # ---- 添加 has_signal（任务要求）----
    df["has_signal"] = df["signal_type"] != "none"

    # ---- 重命名以符合输出规范 ----
    df = df.rename(columns={
        "wave_stage": "wave_state",
        "chanlun_score": "chan_score",
        "bottom_div": "bottom_div_flag",
    })

    # ---- 添加日期字符串（统一格式）----
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # ---- 选择输出列（按规范顺序）----
    available = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    if extra:
        logger.info(f"额外字段（保留）: {extra}")

    df_out = df[available].copy()

    # ---- 按 date + symbol 排序 ----
    df_out = df_out.sort_values(["date", "symbol"]).reset_index(drop=True)

    return df_out


def validate_output(df: pd.DataFrame):
    """验证输出数据"""
    logger.info("=== 输出数据验证 ===")
    logger.info(f"  总行数: {len(df):,}")
    logger.info(f"  总列数: {len(df.columns)}")
    logger.info(f"  股票数: {df['symbol'].nunique():,}")
    logger.info(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"  日期数: {df['date'].nunique():,}")

    logger.info(f"  has_signal=True: {df['has_signal'].sum():,} "
                 f"({100*df['has_signal'].mean():.1f}%)")

    logger.info(f"  各列空值:")
    for col in ["signal_score", "structure_score", "momentum_score",
                "chan_score", "total_score", "rsi", "wave_retracement"]:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            logger.info(f"    {col}: {null_pct:.1f}% 空值")

    logger.info(f"  signal_type 分布:")
    for st, cnt in df["signal_type"].value_counts().head(10).items():
        logger.info(f"    {st}: {cnt:,}")

    logger.info(f"  total_score 统计:")
    logger.info(f"    mean={df['total_score'].mean():.1f}, "
                 f"std={df['total_score'].std():.1f}, "
                 f"max={df['total_score'].max():.1f}")

    logger.info(f"  has_signal 日均信号数:")
    daily = df.groupby("date")["has_signal"].sum()
    logger.info(f"    mean={daily.mean():.0f}, "
                 f"min={daily.min():.0f}, "
                 f"max={daily.max():.0f}")


def main():
    parser = argparse.ArgumentParser(description="WaveChan 信号预计算缓存生成")
    parser.add_argument("--start", type=str, default=None,
                        help="开始日期 YYYY-MM-DD（默认: lookback前推120天）")
    parser.add_argument("--end", type=str, default="2026-03-27",
                        help="结束日期 YYYY-MM-DD（默认: 2026-03-27）")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                        help=f"指定年份列表（默认: {DEFAULT_YEARS}）")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                        help=f"预热期天数（默认: {LOOKBACK_DAYS}）")
    parser.add_argument("--batch-log", type=int, default=200,
                        help=f"进度打印间隔（默认: 200）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"输出路径（默认: {OUTPUT_PATH}）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WaveChan 信号预计算缓存生成器")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"计算年份: {args.years}")
    logger.info("=" * 60)

    t_start = time.time()

    # 1. 加载数据
    data = load_all_daily_data(args.years)

    # 2. 预计算
    df_out = precompute_wavechan_signals(
        data,
        lookback_days=args.lookback,
        batch_log=args.batch_log,
    )

    if df_out.empty:
        logger.error("没有生成任何数据，退出")
        sys.exit(1)

    # 3. 验证
    validate_output(df_out)

    # 4. 写入 Parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写 Parquet（按 date 分区有利于查询）
    df_out.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✅ 已保存: {output_path} ({size_mb:.1f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)
    logger.info("预计算完成！")
    logger.info(f"文件: {output_path}")
    logger.info(f"记录: {len(df_out):,} 行 | 股票: {df_out['symbol'].nunique():,} | "
                 f"交易日: {df_out['date'].nunique():,}")


if __name__ == "__main__":
    main()
