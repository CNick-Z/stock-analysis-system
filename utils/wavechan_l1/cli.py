#!/usr/bin/env python
"""
WaveChan L1 - CLI 入口
======================
用法：
    python -m utils.wavechan_l1.cli build --year 2024
    python -m utils.wavechan_l1.cli build --year 2024 --symbols 600368,000001
    python -m utils.wavechan_l1.cli build --all
    python -m utils.wavechan_l1.cli rebuild --year 2024
    python -m utils.wavechan_l1.cli status
    python -m utils.wavechan_l1.cli read --symbol 600368 --year 2024
    python -m utils.wavechan_l1.cli trend --symbol 600368 --year 2024
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wavechan_l1.manager import WaveChanL1Manager
from utils.wavechan_l1 import zigzag, trend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wavechan_l1.cli")


def cmd_build(args):
    """构建 L1 数据"""
    mgr = WaveChanL1Manager(n_jobs=args.jobs)

    if args.all:
        # 构建所有已有年份
        years = list(range(args.start_year or 2020, (args.end_year or 2026) + 1))
        for yr in years:
            result = mgr.build_year(yr, symbols=args.symbols,
                                    threshold=args.threshold,
                                    lookback=args.lookback)
            print(f"  {yr}: extrema={result['extrema_count']}, trend={result['trend_count']}, "
                  f"time={result['elapsed_seconds']}s")
    else:
        result = mgr.build_year(
            year=args.year,
            symbols=args.symbols,
            threshold=args.threshold,
            lookback=args.lookback,
        )
        print(f"✅ {args.year} 年 L1 构建完成:")
        print(f"   extrema: {result['extrema_count']} 只股票")
        print(f"   trend:   {result['trend_count']} 只股票")
        print(f"   耗时:    {result['elapsed_seconds']}s")


def cmd_rebuild(args):
    """重建 L1 数据"""
    mgr = WaveChanL1Manager(n_jobs=args.jobs)
    result = mgr.rebuild_year(
        year=args.year,
        symbols=args.symbols,
        threshold=args.threshold,
        lookback=args.lookback,
    )
    print(f"✅ {args.year} 年 L1 重建完成:")
    print(f"   extrema: {result['extrema_count']} 只股票")
    print(f"   trend:   {result['trend_count']} 只股票")


def cmd_status(args):
    """查看缓存状态"""
    mgr = WaveChanL1Manager()
    st = mgr.status()
    print(f"根目录: {st['base_path']}")
    print(f"极值数据年份: {st['extrema_years']}")
    print(f"趋势数据年份: {st['trend_years']}")
    print(f"极值文件: {st['total_extrema_files']} 个 ({st['total_extrema_size_mb']} MB)")
    print(f"趋势文件: {st['total_trend_files']} 个 ({st['total_trend_size_mb']} MB)")


def cmd_read(args):
    """读取极值数据"""
    mgr = WaveChanL1Manager()
    df = mgr.get_extrema(args.symbol, args.year)
    if df.empty:
        print(f"⚠️  未找到 {args.symbol} {args.year} 年极值数据")
        return
    print(f"=== {args.symbol} {args.year} 年极值点 ({len(df)} 条) ===")
    print(df.to_string(index=False))


def cmd_trend(args):
    """读取趋势数据"""
    mgr = WaveChanL1Manager()

    if args.current_only:
        current = mgr.get_current_trend(args.symbol, args.year, lookback=args.lookback)
        print(f"{args.symbol} 当前趋势: {current}")
        latest = mgr.get_latest_extrema(args.symbol, args.year)
        if latest:
            print(f"最新极值: {latest['date'].date() if hasattr(latest['date'], 'date') else latest['date']} "
                  f"{latest['type']} @ {latest['price']}")
        return

    df = mgr.get_trend(args.symbol, args.year)
    if df.empty:
        print(f"⚠️  未找到 {args.symbol} {args.year} 年趋势数据")
        return
    print(f"=== {args.symbol} {args.year} 年趋势序列 ({len(df)} 条) ===")
    print(df.to_string(index=False))


def cmd_demo(args):
    """演示：生成测试数据的极值和趋势"""
    import pandas as pd
    import numpy as np

    # 构造模拟数据
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    n = len(dates)

    # 生成带波动的高点/低点序列
    np.random.seed(42)
    trend_signal = np.cumsum(np.random.randn(n) * 0.02)
    highs = 100 + trend_signal + np.abs(np.random.randn(n) * 2)
    lows = 100 + trend_signal - np.abs(np.random.randn(n) * 2)
    closes = 100 + trend_signal + np.random.randn(n)

    df = pd.DataFrame({
        "date": dates,
        "symbol": args.symbol or "DEMO",
        "open": closes - 0.5,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(1e6, 1e7, n),
    })

    # 极值识别
    extrema = zigzag.identify_extrema(df, threshold=args.threshold)
    print(f"=== 演示：极值点 ({len(extrema)} 条) ===")
    print(extrema.to_string(index=False))

    # 趋势判断
    if not extrema.empty:
        current_trend = trend.determine_trend(extrema, lookback=args.lookback)
        print(f"\n当前趋势: {current_trend}")
        trend_df = trend.build_trend_series(extrema, lookback=args.lookback)
        print(f"=== 趋势序列 ({len(trend_df)} 条) ===")
        print(trend_df[["date", "trend", "trend_index", "strength", "last_extrema_type"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="WaveChan L1 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = sub.add_parser("build", help="构建 L1 数据")
    p_build.add_argument("--year", type=int, help="年份")
    p_build.add_argument("--all", action="store_true", help="构建所有年份")
    p_build.add_argument("--start-year", type=int, default=2020, help="起始年份（--all 时用）")
    p_build.add_argument("--end-year", type=int, default=2026, help="结束年份（--all 时用）")
    p_build.add_argument("--symbols", type=lambda s: s.split(","), default=None, help="股票列表，逗号分隔")
    p_build.add_argument("--threshold", type=float, default=0.05, help="ZigZag 回撤阈值 (默认 0.05)")
    p_build.add_argument("--lookback", type=int, default=5, help="趋势回看极值数 (默认 5)")
    p_build.add_argument("--jobs", type=int, default=8, help="并行 worker 数")
    p_build.set_defaults(func=cmd_build)

    # rebuild
    p_rebuild = sub.add_parser("rebuild", help="重建 L1 数据")
    p_rebuild.add_argument("--year", type=int, required=True, help="年份")
    p_rebuild.add_argument("--symbols", type=lambda s: s.split(","), default=None)
    p_rebuild.add_argument("--threshold", type=float, default=0.05)
    p_rebuild.add_argument("--lookback", type=int, default=5)
    p_rebuild.add_argument("--jobs", type=int, default=8)
    p_rebuild.set_defaults(func=cmd_rebuild)

    # status
    p_status = sub.add_parser("status", help="查看缓存状态")
    p_status.set_defaults(func=cmd_status)

    # read
    p_read = sub.add_parser("read", help="读取极值数据")
    p_read.add_argument("--symbol", required=True, help="股票代码")
    p_read.add_argument("--year", type=int, required=True, help="年份")
    p_read.set_defaults(func=cmd_read)

    # trend
    p_trend = sub.add_parser("trend", help="读取趋势数据")
    p_trend.add_argument("--symbol", required=True, help="股票代码")
    p_trend.add_argument("--year", type=int, required=True, help="年份")
    p_trend.add_argument("--lookback", type=int, default=5, help="回看极值数")
    p_trend.add_argument("--current-only", action="store_true", help="只显示当前趋势")
    p_trend.set_defaults(func=cmd_trend)

    # demo
    p_demo = sub.add_parser("demo", help="演示：测试数据极值和趋势")
    p_demo.add_argument("--symbol", default="DEMO", help="股票代码")
    p_demo.add_argument("--threshold", type=float, default=0.05)
    p_demo.add_argument("--lookback", type=int, default=5)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
