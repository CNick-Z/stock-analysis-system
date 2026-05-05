#!/usr/bin/env python3
"""
WaveChan 每日增量信号脚本（正式版）
=====================================

集成三层缓存架构的每日增量计算：
- L1 历史归档层（Parquet，按年分区）- 2021-2024，只读
- L2 热数据层（Parquet，按月分区）- 2025+，每日增量
- L3 参数缓存层（SQLite，LRU）

核心逻辑：
1. 从 L2 读取最新日期，确定增量起点
2. 从 ParquetDB 读取增量数据（lookback 确保波浪计算正确）
3. 用 WaveChanSelector（V3引擎）批量计算所有股票评分
4. 结果写入 L2 当月分区（自动去重合并）
5. 生成当日信号报告

信号字段（L2 Schema）：
  date, symbol, signal_score(40%), structure_score(30%),
  momentum_score(20%), chan_score(10%), total_score,
  has_signal, signal_type, signal_status, wave_state,
  wave_trend, wave_retracement, rsi, macd_hist, divergence,
  volume_ratio, fractal, stop_loss, close, open, high, low, volume

用法：
  python3 wavechan_daily_incremental.py              # 增量计算今日
  python3 wavechan_daily_incremental.py --date 2026-03-27  # 指定日期
  python3 wavechan_daily_incremental.py --dry-run    # 仅预览，不写入
  python3 wavechan_daily_incremental.py --rebuild-month 2026-03  # 重建某月
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ---- paths ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.parquet_db import ParquetDatabaseIntegrator
from utils.wavechan_cache import WaveChanCacheManager
from strategies.wavechan_selector import WaveChanSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Schema：L2 缓存期望的字段（与 precompute 一致）
# ============================================================
L2_COLUMNS = [
    "date", "symbol",
    "signal_score", "structure_score", "momentum_score", "chan_score",
    "total_score", "has_signal", "signal_type", "signal_status",
    "wave_state", "wave_trend", "wave_retracement",
    "rsi", "macd_hist", "divergence", "volume_ratio",
    "fractal", "stop_loss",
    "close", "open", "high", "low", "volume",
]

# WaveChanSelector 输出的字段名 → L2 字段名 映射
FIELD_RENAME = {
    "wave_stage": "wave_state",
    "chanlun_score": "chan_score",
    "bottom_div": "bottom_div_flag",
}


# ============================================================
# 核心增量计算
# ============================================================

def _compute_batch(args):
    """
    计算一批股票（供多进程调用）
    每个进程独立创建 selector 实例
    """
    symbols, df_dict, target_dates, config, progress_offset, progress_total = args
    # 每个进程创建自己的 selector（避免跨进程 pickle 问题）
    selector = WaveChanSelector(config=config)
    results = []

    for idx, sym in enumerate(symbols):
        sym_key = str(sym)
        if sym_key not in df_dict:
            continue
        sym_df = df_dict[sym_key]
        if len(sym_df) < 60:
            continue

        try:
            scored = selector._compute_symbol_scores(sym, sym_df)
            if scored is None or scored.empty:
                continue
            scored = scored[scored["date"].isin(target_dates)]
            if not scored.empty:
                results.append(scored)
        except Exception:
            continue

    return results


def compute_incremental_signals(
    start_date: str,
    end_date: str,
    selector: WaveChanSelector,
    db: ParquetDatabaseIntegrator,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    计算指定日期范围的增量信号（多进程加速）

    Args:
        start_date: 增量开始日期（YYYY-MM-DD）
        end_date: 增量结束日期（YYYY-MM-DD）
        selector: WaveChanSelector 实例（用于获取配置）
        db: ParquetDatabaseIntegrator 实例
        n_workers: 并行进程数

    Returns:
        DataFrame: L2 Schema 的信号数据
    """
    # lookback：波浪分析需要足够历史
    lookback_days = 120
    extended_start = (
        pd.to_datetime(start_date) - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    logger.info(f"加载数据: {extended_start} ~ {end_date} (lookback={lookback_days}天)")

    t0 = time.time()
    df = db.fetch_daily_data(
        extended_start, end_date,
        columns=["date", "symbol", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        logger.warning("没有数据")
        return pd.DataFrame()

    # 过滤 ST 股（可选）
    try:
        info = db.get_stock_basic_info()
        if not info.empty:
            st_mask = ~info["name"].str.contains("ST", na=False)
            valid_symbols = set(info.loc[st_mask, "symbol"].unique())
            df = df[df["symbol"].isin(valid_symbols)]
            logger.info(f"过滤ST后: {df['symbol'].nunique()} 只股票")
    except Exception:
        pass

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"数据: {len(df):,} 条, {df['symbol'].nunique()} 只, "
                f"{time.time()-t0:.1f}s")

    # 目标日期
    target_dates = set(df[df["date"] >= start_date]["date"].unique())
    target_dates_sorted = sorted(target_dates)
    logger.info(f"目标日期: {target_dates_sorted[0]} ~ {target_dates_sorted[-1]} "
                f"({len(target_dates_sorted)}天)")

    # 构建每只股票的数据字典（便于多进程分发）
    symbols = sorted(df["symbol"].unique())
    df_dict = {}
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        df_dict[str(sym)] = sym_df

    config = selector.config
    total = len(symbols)
    t1 = time.time()

    # ---- 多进程分发 ----
    import multiprocessing as mp
    n_workers = min(n_workers, mp.cpu_count(), 8)
    chunk_size = max(1, total // n_workers)
    batches = []
    for i in range(0, total, chunk_size):
        chunk_syms = symbols[i:i + chunk_size]
        batches.append((chunk_syms, df_dict, target_dates, config, i, total))

    logger.info(f"启动 {n_workers} 进程并行计算...")
    with mp.Pool(n_workers) as pool:
        batch_results = pool.map(_compute_batch, batches)

    results = []
    for br in batch_results:
        results.extend(br)

    if not results:
        logger.warning("没有任何评分结果")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t1
    rate = total / elapsed if elapsed > 0 else 0
    logger.info(f"评分完成: {len(combined):,} 条, {elapsed:.0f}s, "
                f"{rate:.0f} sym/s")

    # ---- 字段映射 → L2 Schema ----
    for old_name, new_name in FIELD_RENAME.items():
        if old_name in combined.columns and new_name not in combined.columns:
            combined = combined.rename(columns={old_name: new_name})

    # 添加 has_signal
    combined["has_signal"] = combined["signal_type"] != "none"

    # 确保日期格式一致
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

    # 选择 L2 列
    extra_cols = [c for c in combined.columns if c not in L2_COLUMNS]
    if extra_cols:
        logger.info(f"额外字段（保留）: {extra_cols}")

    output_cols = [c for c in L2_COLUMNS if c in combined.columns]
    combined = combined[output_cols].copy()

    # ── 从技术指标库读取 RSI（覆盖 WaveChanSelector 计算的 RSI）─────────────
    if 'rsi' not in combined.columns or combined['rsi'].isna().any():
        try:
            # 确定年份范围
            dates = pd.to_datetime(combined['date']).dt.year.unique()
            tech_frames = []
            for yr in dates:
                tp = Path(f'/root/.openclaw/workspace/data/warehouse/technical_indicators_year={yr}/data.parquet')
                if tp.exists():
                    tdf = pd.read_parquet(tp, columns=['date','symbol','rsi_14'])
                    tdf = tdf.rename(columns={'rsi_14': 'rsi'})
                    tech_frames.append(tdf)
            if tech_frames:
                tech_all = pd.concat(tech_frames, ignore_index=True)
                tech_all['date'] = pd.to_datetime(tech_all['date']).dt.strftime('%Y-%m-%d')
                # 合并到 combined（技术指标库优先级）
                merged = combined.merge(tech_all[['date','symbol','rsi']], on=['date','symbol'], how='left', suffixes=('','_tech'))
                combined['rsi'] = merged['rsi_tech'].combine_first(combined['rsi'])
                n_fixed = combined['rsi'].notna().sum()
                logger.info(f"[RSI] 从技术指标库补入 {n_fixed} 个值")
        except Exception as e:
            logger.warning(f"[RSI] 技术指标库读取失败: {e}")

    return combined


def get_latest_l2_date(cm: WaveChanCacheManager) -> str | None:
    """
    获取 L2 缓存中最新已有信号行的日期。
    
    关键：不能用 date 列最大值判断（铁律批量写入会覆盖整个月分区，
    iron_laws 列延伸到月末但 signal 列在特定日期后不再更新）。
    必须用 has_signal==True 的最大日期来判断实际信号覆盖范围。
    """
    l1l2 = cm.l1l2
    latest_date = None
    for p in sorted(l1l2.base_path.glob("l2_hot_year=*")):
        pf = p / "data.parquet"
        if not pf.exists():
            continue
        try:
            # 同时读 date 和 has_signal，用后者过滤
            df = pd.read_parquet(pf, columns=["date", "has_signal"])
            if df.empty:
                continue
            # 只看有信号的行
            sig_df = df[df["has_signal"] == True]
            if sig_df.empty:
                continue
            dmax = sig_df["date"].max()
            if latest_date is None or dmax > latest_date:
                latest_date = dmax
        except Exception:
            continue
    return latest_date




def report_signals(df: pd.DataFrame, trade_date: str) -> dict:
    """
    生成信号报告并打印
    """
    day_df = df[df["date"] == trade_date]
    if day_df.empty:
        logger.warning(f"{trade_date} 无数据")
        return None

    buy_cands = day_df[
        (day_df["signal_type"] != "none") &
        (day_df["total_score"] >= 50)
    ].sort_values("total_score", ascending=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 {trade_date} WaveChan 信号报告")
    logger.info(f"{'='*60}")
    logger.info(f"市场状态: 共 {len(day_df)} 只股票")
    logger.info(f"有信号: {int(day_df['has_signal'].sum())} 个")
    logger.info(f"买入候选: {len(buy_cands)} 个")

    if len(buy_cands) > 0:
        logger.info(f"\n📈 Top买入信号 (total_score >= 50):")
        header = (f"{'代码':<8} {'信号类型':<12} {'状态':<10} "
                  f"{'总分':<6} {'信号分':<6} {'结构分':<6} "
                  f"{'动能分':<6} {'缠论分':<6} {'RSI':<6} "
                  f"{'止损':<8} {'最新价':<8}")
        logger.info(header)
        logger.info("-" * 110)

        for _, r in buy_cands.head(15).iterrows():
            sl = r.get("stop_loss", 0) or 0
            row_str = (
                f"{str(r['symbol']):<8} "
                f"{str(r.get('signal_type','')):<12} "
                f"{str(r.get('signal_status','')):<10} "
                f"{r.get('total_score', 0):.0f}     "
                f"{r.get('signal_score', 0):.0f}     "
                f"{r.get('structure_score', 0):.0f}     "
                f"{r.get('momentum_score', 0):.0f}     "
                f"{r.get('chan_score', 0):.0f}     "
                f"{r.get('rsi', 0):.0f}     "
                f"{sl:.2f}     "
                f"{r.get('close', 0):.2f}"
            )
            logger.info(row_str)

    # 信号类型分布
    sig_mask = day_df["signal_type"] != "none"
    sig_dist = day_df.loc[sig_mask, "signal_type"].value_counts()
    if not sig_dist.empty:
        logger.info(f"\n信号类型分布:")
        for sig_type, cnt in sig_dist.head(5).items():
            logger.info(f"  {sig_type}: {cnt}")

    result = {
        "date": trade_date,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": int(len(day_df)),
        "signals_count": int(day_df["has_signal"].sum()),
        "buy_candidates": int(len(buy_cands)),
        "top_signals": [],
    }

    wave_state_col = "wave_state" if "wave_state" in buy_cands.columns else "wave_stage"
    for _, r in buy_cands.head(10).iterrows():
        sl = r.get("stop_loss", 0) or 0
        result["top_signals"].append({
            "symbol": str(r["symbol"]),
            "signal_type": str(r.get("signal_type", "")),
            "signal_status": str(r.get("signal_status", "")),
            "total_score": float(r.get("total_score", 0)),
            "signal_score": float(r.get("signal_score", 0)),
            "structure_score": float(r.get("structure_score", 0)),
            "momentum_score": float(r.get("momentum_score", 0)),
            "chan_score": float(r.get("chan_score", 0)),
            "rsi": float(r.get("rsi", 0)),
            "stop_loss": float(sl) if sl else None,
            "close": float(r.get("close", 0)),
            "wave_state": str(r.get(wave_state_col, "")),
            "wave_trend": str(r.get("wave_trend", "")),
        })

    return result


# ============================================================
# 每日增量主流程
# ============================================================

def run_daily_incremental(
    target_date: str | None = None,
    dry_run: bool = False,
    rebuild_month: str | None = None,
    top_n: int = 27,
    threshold: float = 23.3,
):
    """
    每日增量主流程

    Args:
        target_date: 目标日期（YYYY-MM-DD），默认今日
        dry_run: True=仅预览不写入
        rebuild_month: YYYY-MM 格式，重建某月数据
    """
    t_start = time.time()
    today_str = datetime.now().strftime("%Y-%m-%d")
    target_date = target_date or today_str

    logger.info("=" * 60)
    logger.info("WaveChan 每日增量信号")
    logger.info(f"目标日期: {target_date} | 干跑: {dry_run}")
    logger.info("=" * 60)

    # ---- 初始化管理器 ----
    cm = WaveChanCacheManager()
    db = ParquetDatabaseIntegrator()
    selector_config = {'top_n': top_n, 'threshold': threshold}
    selector = WaveChanSelector(config=selector_config)

    # ---- 重建某月（跳过日期确定）----
    if rebuild_month:
        year, month = map(int, rebuild_month.split("-"))
        start_date = f"{year}-{month:02d}-01"
        # 月末
        import calendar
        _, last_day = calendar.monthrange(year, month)
        end_date = f"{year}-{month:02d}-{last_day:02d}"

        logger.info(f"重建月份: {rebuild_month}, 范围: {start_date} ~ {end_date}")

        signals = compute_incremental_signals(start_date, end_date, selector, db)
        if signals.empty:
            logger.error("计算结果为空，退出")
            return

        # 过滤当月
        signals["_month"] = signals["date"].str[:7]
        signals = signals[signals["_month"] == rebuild_month].drop(columns=["_month"])

        if signals.empty:
            logger.error(f"{rebuild_month} 范围内无数据")
            return

        if dry_run:
            logger.info(f"[干跑] 共 {len(signals)} 条信号预览:")
            print(signals.head(10).to_string())
            return

        # 逐日写入 L2
        for date in sorted(signals["date"].unique()):
            day_signals = signals[signals["date"] == date]
            cm.daily_increment(date, day_signals)
            logger.info(f"  ✅ {date}: {len(day_signals)} 条 → L2")

        logger.info(f"重建完成: {len(signals)} 条, 耗时: {time.time()-t_start:.0f}s")
        return

    # ---- 正常增量：确定起始日期 ----
    latest_l2_date = get_latest_l2_date(cm)
    logger.info(f"L2 最新日期: {latest_l2_date}")

    # 已有 L2 数据：增量计算
    if latest_l2_date:
        if target_date <= latest_l2_date:
            # --date 明确指定时：target_date==latest_l2_date 时发报告，target_date<latest_l2_date 时强制重算
            if target_date == latest_l2_date:
                logger.info(f"目标日期 {target_date} == L2最新，跳过重算，仅报告")
                signals = cm.load(latest_l2_date, latest_l2_date)
                if not signals.empty:
                    report_signals(signals, latest_l2_date)
                return
            else:
                # target_date < latest_l2_date：强制重算（用于修复历史缺失数据）
                logger.info(f"目标日期 {target_date} < L2最新 {latest_l2_date}，强制重算")
                start_date = target_date
                logger.info(f"重算模式: {start_date} ~ {target_date}")

        else:
            start_date = (
                pd.to_datetime(latest_l2_date) + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            logger.info(f"增量模式: {start_date} ~ {target_date}")

    else:
        # 无 L2 数据：从头计算（从仓库数据最早日期开始）
        logger.info("L2 为空，从头计算...")
        # 默认从 2025-01-01 开始（预热 120 天后，输出从 2025-05 开始）
        start_date = "2025-01-01"

    # ---- 增量计算 ----
    signals = compute_incremental_signals(start_date, target_date, selector, db)

    if signals.empty:
        logger.warning("增量计算结果为空")
        return

    if dry_run:
        logger.info(f"[干跑] 共 {len(signals)} 条信号:")
        print(signals.head(10).to_string())
        return

    # ---- 写入 L2（按日增量）----
    for date in sorted(signals["date"].unique()):
        day_signals = signals[signals["date"] == date]
        cm.daily_increment(date, day_signals)

    logger.info(f"\n✅ 增量写入完成: {len(signals)} 条, {time.time()-t_start:.0f}s")

    # ---- 信号报告 ----
    report = report_signals(signals, target_date)
    if report:
        signal_file = f"/tmp/wavechan_signals_{target_date}.json"
        with open(signal_file, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {signal_file}")

    logger.info(f"总耗时: {time.time()-t_start:.0f}s")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="WaveChan 每日增量信号")
    parser.add_argument(
        "--date", type=str, default=None,
        help="目标日期 YYYY-MM-DD（默认: 今日）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不写入 L2"
    )
    parser.add_argument(
        "--rebuild-month", type=str, default=None,
        help="重建某月数据，格式 YYYY-MM"
    )
    parser.add_argument(
        "--cache-status", action="store_true",
        help="仅显示缓存状态"
    )
    parser.add_argument(
        "--top-n", type=int, default=27,
        help="每日最大持仓数（默认27）"
    )
    parser.add_argument(
        "--threshold", type=float, default=23.3,
        help="信号总分阈值（默认23.3）"
    )
    args = parser.parse_args()

    if args.cache_status:
        cm = WaveChanCacheManager()
        status = cm.status()
        print("=" * 60)
        print("WaveChan L1/L2/L3 缓存状态")
        print("=" * 60)
        print("\nL1 历史归档（2021-2024）:")
        for k, v in status.get("l1_partitions", {}).items():
            print(f"  {k}: {v['rows']:,} 行, {v['size_mb']:.1f} MB")
        print("\nL2 热数据（2025+）:")
        for k, v in status.get("l2_partitions", {}).items():
            print(f"  {k}: {v['rows']:,} 行, {v['size_mb']:.1f} MB")
        print(f"\nL2 总大小: {status.get('l2_size_mb', 0):.1f} MB")
        print(f"L3 参数缓存: {status.get('l3_count', 0)} 条")
        return

    run_daily_incremental(
        target_date=args.date,
        dry_run=args.dry_run,
        rebuild_month=args.rebuild_month,
        top_n=args.top_n,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
