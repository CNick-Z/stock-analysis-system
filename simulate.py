#!/usr/bin/env python3
"""
simulate.py — 统一模拟盘入口
=============================
用法:
    python3 simulate.py --strategy v8                      # 每天运行（自动读取上次状态继续）
    python3 simulate.py --strategy v8 --date 2026-03-31    # 指定日期运行
    python3 simulate.py --strategy v8 --reset              # 重置状态
    python3 simulate.py --strategy v8 --show-state         # 查看当前持仓和资金
    python3 simulate.py --strategy v8 --show-trades        # 查看交易记录
    python3 simulate.py --strategy v8 --state-file /tmp/my_state.json  # 自定义状态文件
"""

import argparse
import fcntl
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework
from simulator.shared import (
    load_strategy,
    load_wavechan_cache,
    add_next_open,
    STRATEGY_REGISTRY,
    WAVECHAN_L2_CACHE,
)
from simulator.market_regime import MarketRegimeFilter
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_for_date(name: str, target_date: str) -> tuple:
    """
    加载目标日期所在年份的数据。
    返回 (year_df, target_df)：全年数据 和 目标日期过滤后的数据。
    模拟盘需要全年数据用于 prepare，target_df 用于单日模拟。
    """
    # V3策略不需要资金流指标（省10秒计算）
    add_mf = name != "wavechan_v3_strict"
    year = int(target_date[:4])
    logger.info(f"加载数据: year={year} | 资金流: {'开启' if add_mf else '跳过（V3无需）'}")
    year_df = load_strategy_data(years=[year], add_money_flow=add_mf)
    logger.info(f"原始数据: {len(year_df):,} 行  [{year_df['date'].min()} ~ {year_df['date'].max()}]")

    # 先加 next_open（必须在过滤前，否则 groupby shift(-1) 找不到下一行）
    year_df = add_next_open(year_df)

    # 再过滤到目标日期
    df = year_df[year_df["date"] == target_date].copy()
    if df.empty:
        # 数据里没有 target_date，用最后一天（数据延迟的情况）
        last_date = year_df["date"].max()
        logger.warning(f"  目标日期 {target_date} 无数据，用最后一天 {last_date} 代替")
        df = year_df[year_df["date"] == last_date].copy()

    if name == "wavechan_v3_strict":
        wave_df = load_wavechan_cache([year])
        if not wave_df.empty:
            for col in [c for c in wave_df.columns if c not in ("date", "symbol")]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            wave_df = wave_df.copy()
            wave_df["date"] = wave_df["date"].astype(str)
            df = df.merge(wave_df, on=["date", "symbol"], how="left")

    return year_df, df


def load_year_df(target_date: str) -> pd.DataFrame:
    """
    加载完整年份数据（供 V8 等策略的 prepare 使用）。
    返回全年数据，不过滤到单日。
    """
    year = int(target_date[:4])
    df = load_strategy_data(years=[year], add_money_flow=True)
    logger.info(f"Prepare 全年数据: {len(df):,} 行  [{df['date'].min()} ~ {df['date'].max()}]")
    return df


def show_state(framework: BaseFramework, strategy_name: str, format: str = "text"):
    """打印当前持仓和资金状态
    format: 'text' 纯文本 / 'markdown' QQmarkdown格式
    """
    final_value = framework.cash + sum(
        pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        for pos in framework.positions.values()
    )
    total_return = (final_value / framework.initial_cash - 1) * 100
    position_value = final_value - framework.cash

    # 交易统计
    closed = [t for t in framework.trades if t["action"] == "sell"]
    win_rate = framework.n_winning / max(framework.n_total, 1) * 100
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0

    if format == "markdown":
        # ---- Markdown 格式 ----
        lines = []
        strategy_display = {
            "v8": "V8模拟盘", "wavechan_v3_strict": "WaveChanV3铁律模拟盘"
        }
        display_name = strategy_display.get(strategy_name, f"{strategy_name}模拟盘")
        lines.append(f"**{display_name}状态**  |  {strategy_name}")
        lines.append(f"初始 **{framework.initial_cash:,.0f}** | 现金 **{framework.cash:,.0f}** | 持仓 **{position_value:,.0f}**")
        lines.append(f"总价值 **{final_value:,.0f}** ({total_return:+.2f}%) | {len(framework.positions)} 只 | 胜率 {win_rate:.0f}%")
        lines.append("")

        if framework.positions:
            # 表头
            rows = []
            for sym, pos in sorted(framework.positions.items()):
                latest = pos.get("latest_price", pos["avg_cost"])
                pnl = (latest - pos["avg_cost"]) * pos["qty"]
                pnl_pct = (latest / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
                mkt_val = latest * pos["qty"]
                pos_pct = mkt_val / final_value * 100 if final_value > 0 else 0
                # 简化 MACD/KDJ 状态（从 extra 或默认）
                extra = pos.get("extra", {})
                macd_s = extra.get("macd_state", "-")
                kdj_s = extra.get("kdj_state", "-")
                limit_mark = "X" if pos.get("limit_up") or pos.get("limit_down") else "-"
                rows.append((
                    sym,
                    pos.get("name", pos.get("industry", sym)),
                    pos["qty"],
                    pos["avg_cost"],
                    latest,
                    pnl,
                    pnl_pct,
                    pos.get("days_held", 0),
                    f"{pos_pct:.1f}%",
                    macd_s,
                    kdj_s,
                    limit_mark,
                ))

            header = "| # | 代码 | 名称 | 持仓量 | 成本 | 现价 | 盈亏额 | 盈亏% | 天 | 仓位% | MACD | KDJ | 涨停 |"
            sep   = "|---|------|------|--------|------|------|--------|-------|---|------|-----|-----|------|"
            lines.append(header)
            lines.append(sep)
            for i, row in enumerate(rows, 1):
                sym, name, qty, cost, latest, pnl, pnl_pct, days, pos_pct, macd_s, kdj_s, limit = row
                lines.append(
                    f"| {i} | {sym} | {name} | {qty:,} | {cost:.2f} | {latest:.2f} "
                    f"| {pnl:+,.0f} | {pnl_pct:+.2f}% | {days} | {pos_pct} | {macd_s} | {kdj_s} | {limit} |"
                )
        else:
            lines.append("*当前无持仓*")

        print("\n".join(lines))
        return

    # ---- 纯文本格式 ----
    _display_map = {
        "v8": "V8模拟盘", "wavechan_v3_strict": "WaveChanV3铁律模拟盘"
    }
    _dname = _display_map.get(strategy_name, f"{strategy_name}模拟盘")
    print(f"\n{'=' * 50}")
    print(f"  {_dname}状态  |  {strategy_name}")
    print(f"{'=' * 50}")
    print(f"  初始资金: {framework.initial_cash:>14,.0f}")
    print(f"  当前现金: {framework.cash:>14,.0f}")
    print(f"  总价值:   {final_value:>14,.0f}  ({total_return:+.2f}%)")
    print(f"  持仓数:   {len(framework.positions):>14}  只")
    print(f"{'=' * 50}")

    if framework.positions:
        print(f"\n  {'代码':<8} {'持仓量':>8} {'成本':>10} {'现价':>10} {'盈亏额':>12} {'盈亏%':>8} {'持仓天数':>8}")
        print(f"  {'-'*70}")
        for sym, pos in sorted(framework.positions.items()):
            latest = pos.get("latest_price", pos["avg_cost"])
            pnl = (latest - pos["avg_cost"]) * pos["qty"]
            pnl_pct = (latest / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
            print(
                f"  {sym:<8} {pos['qty']:>8} {pos['avg_cost']:>10.2f} "
                f"{latest:>10.2f} {pnl:>+12,.0f} {pnl_pct:>+7.2f}% {pos.get('days_held', 0):>8}"
            )
    else:
        print("\n  当前无持仓")

    if closed:
        print(f"\n  交易统计: {framework.n_total} 笔 | 胜率: {win_rate:.1f}% | 累计盈亏: {total_pnl:+,.0f}")
    print(f"{'=' * 50}")


def show_trades(framework: BaseFramework, strategy_name: str):
    """打印交易记录"""
    if not framework.trades:
        print("\n  暂无交易记录")
        return

    print(f"\n{'=' * 70}")
    print(f"  交易记录  |  策略: {strategy_name}")
    print(f"{'=' * 70}")
    print(f"  {'日期':<12} {'代码':<8} {'方向':<6} {'价格':>10} {'数量':>8} {'盈亏':>12} {'原因'}")
    print(f"  {'-'*70}")
    for t in framework.trades:
        print(
            f"  {t['date']:<12} {t['symbol']:<8} {t['action']:<6} "
            f"{t['price']:>10.2f} {t['qty']:>8} "
            f"{t.get('pnl', 0):>+12,.0f}  {t.get('reason', '')}"
        )
    print(f"{'=' * 70}")

    # 汇总
    closed = [t for t in framework.trades if t["action"] == "sell"]
    if closed:
        win_rate = framework.n_winning / max(framework.n_total, 1) * 100
        total_pnl = sum(t["pnl"] for t in closed)
        print(f"\n  共 {len(closed)} 笔卖出 | 胜率: {win_rate:.1f}% | 累计盈亏: {total_pnl:+,.0f}")


def show_candidates(framework: BaseFramework, strategy_name: str, date: str = None, format: str = "text"):
    """打印候选信号（markdown格式）
    date: None 表示最新一天
    format: 'text' / 'markdown'
    """
    if not framework.candidates:
        print("\n  暂无候选信号记录")
        return

    # 取最新日期的候选
    if date:
        filtered = [c for c in framework.candidates if c["date"] == date]
    else:
        dates = sorted(set(c["date"] for c in framework.candidates))
        if not dates:
            print("\n  暂无候选信号记录")
            return
        latest_date = dates[-1]
        filtered = [c for c in framework.candidates if c["date"] == latest_date]
        date = latest_date

    if not filtered:
        print(f"\n  {date} 无候选信号记录")
        return

    # 按评分排序
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    if format == "markdown":
        lines = []
        bought = [c for c in filtered if c["status"].startswith("OK")]
        not_bought = [c for c in filtered if not c["status"].startswith("OK")]
        n_buy = len(bought)
        n_total = len(filtered)
        lines.append(f"**V8 候选信号**  |  {date}（买前过滤后 {n_total} 只，{n_buy} 只买入）")
        lines.append("")
        lines.append("| 排名 | 评分 | 代码 | 名称 | 行业 | 状态 | 主要信号 |")
        lines.append("|------|------|------|------|------|------|----------|")
        for i, c in enumerate(filtered, 1):
            # 状态标签
            st = c["status"]
            if st.startswith("OK"):
                st_label = "OK"
            elif "已持仓" in st:
                st_label = "X"
            elif "涨停" in st:
                st_label = "X"
            elif "跌停" in st:
                st_label = "X"
            elif "已满仓" in st:
                st_label = "X"
            elif "资金" in st:
                st_label = "X"
            else:
                st_label = "X"

            # 主要信号
            signals = []
            if c.get("cci", 0) < -80:
                signals.append(f"CCI{int(c['cci'])}")
            if c.get("wr", 0) < -80:
                signals.append(f"WR{int(c['wr'])}")
            if c.get("rsi", 0) < 35:
                signals.append(f"RSI{int(c['rsi'])}")
            if c.get("macd_state") == "金叉":
                signals.append("MACD金叉")
            sig_str = ", ".join(signals) if signals else "-"

            name = c.get("name", c["symbol"])
            industry = c.get("industry", "-")
            lines.append(
                f"| {i} | {c['score']:+.2f} | {c['symbol']} | {name} | {industry} | {st_label} | {sig_str} |"
            )
        print("\n".join(lines))
        return

    # ---- 纯文本格式 ----
    print(f"\n{'=' * 70}")
    print(f"  候选信号  |  日期: {date}  |  策略: {strategy_name}")
    print(f"{'=' * 70}")
    print(f"  {'排名':<4} {'评分':>6} {'代码':<8} {'名称':<10} {'状态':<16} {'信号'}")
    print(f"  {'-'*70}")
    for i, c in enumerate(filtered, 1):
        signals = []
        if c.get("cci", 0) < -80:
            signals.append(f"CCI{int(c['cci'])}")
        if c.get("wr", 0) < -80:
            signals.append(f"WR{int(c['wr'])}")
        if c.get("rsi", 0) < 35:
            signals.append(f"RSI{int(c['rsi'])}")
        sig_str = ", ".join(signals) if signals else "-"
        print(
            f"  {i:<4} {c['score']:>+6.2f} {c['symbol']:<8} "
            f"{c.get('name', c['symbol']):<10} {c['status']:<16} {sig_str}"
        )
    print(f"{'=' * 70}")


# ── 主入口 ─────────────────────────────────────────────────

def main():
    # ── 文件锁：防止并发重复启动 ────────────────────────────────
    lock_path = Path("/tmp/simulate.lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("【警告】另一个模拟盘进程正在运行，退出。"
              "如确认无进程在跑，请手动删除: rm /tmp/simulate.lock")
        os.close(lock_fd)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="统一模拟盘入口")
    parser.add_argument("--strategy", required=True,
                        choices=["v8", "wavechan_v3_strict"],
                        help="选择策略: v8 / wavechan_v3_strict（铁律过滤）")
    parser.add_argument("--date", default=None,
                        help="运行日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--reset", action="store_true",
                        help="重置状态（删除状态文件）")
    parser.add_argument("--show-state", action="store_true",
                        help="查看当前持仓和资金")
    parser.add_argument("--show-trades", action="store_true",
                        help="查看交易记录")
    parser.add_argument("--show-candidates", action="store_true",
                        help="查看候选信号（默认markdown格式）")
    parser.add_argument("--report-format", default="text", choices=["text", "markdown"],
                        help="报告格式（默认 text，show-state/candidates 时生效）")
    parser.add_argument("--state-file",
                        help="自定义状态文件路径")
    parser.add_argument("--initial-cash", type=float, default=1_000_000,
                        help="初始资金（默认 1,000,000）")
    parser.add_argument("--show-regime", action="store_true",
                        help="显示当日大盘状态（BEAR/NEUTRAL/BULL 及仓位上限）")
    parser.add_argument("--no-regime", action="store_true",
                        help="禁用 MarketRegimeFilter（全天候固定仓位）")
    parser.add_argument("--start", default=None,
                        help="区间起始日期 YYYY-MM-DD（需配合 --end 使用，数据只加载一次）")
    parser.add_argument("--end", default=None,
                        help="区间结束日期 YYYY-MM-DD（需配合 --start 使用）")
    args = parser.parse_args()

    # ── 日期区间模式（--start + --end）────────────────────────────────
    if args.start and args.end:
        strategy_name = args.strategy
        portfolio_dir = Path("/root/.openclaw/workspace/portfolio")
        state_file = args.state_file or str(portfolio_dir / f"simulate_{strategy_name}.json")
        _s = date.fromisoformat(args.start)
        _e = date.fromisoformat(args.end)
        if _s > _e:
            print(f"错误: --start ({args.start}) 不能晚于 --end ({args.end})")
            return

        # 加载一次数据（用起始日期确定年份）
        print(f"加载数据: {args.start} ~ {args.end}")
        year_df, _ = load_data_for_date(strategy_name, args.start)
        all_dates = sorted(year_df["date"].unique().tolist())
        date_list = [d for d in all_dates if d >= args.start and d <= args.end]
        print(f"交易日: {date_list[0]} ~ {date_list[-1]}，共 {len(date_list)} 天")

        # 初始化框架
        V3_MAX_POSITIONS = 5
        V3_POSITION_SIZE = 0.10
        params = STRATEGY_REGISTRY.get(strategy_name, {}).get("params", {})
        mrf = None
        if strategy_name == "wavechan_v3_strict":
            # V3 启用 MarketRegimeFilter（牛熊市仓位控制）
            mrf = MarketRegimeFilter(
                confirm_days=1,
                neutral_position=0.70,
                bear_position=0.30,
            )
            mrf.prepare(args.start, args.end)
            framework = BaseFramework(
                initial_cash=args.initial_cash,
                max_positions=V3_MAX_POSITIONS,
                position_size=V3_POSITION_SIZE,
                state_file=state_file,
                market_regime_filter=mrf,
            )
        else:
            framework = BaseFramework(
                initial_cash=args.initial_cash,
                max_positions=params.get("max_positions", 3),
                position_size=params.get("position_size", 0.20),
                state_file=state_file,
            )
        framework.load_state()
        if args.reset:
            framework.reset()
        strategy = load_strategy(strategy_name)
        strategy.prepare(date_list, year_df)

        print(f"\n{'='*50}")
        print(f"  {strategy_name}  区间: {date_list[0]} ~ {date_list[-1]}")
        print(f"{'='*50}")

        for i, td in enumerate(date_list):
            try:
                framework.run_simulate(strategy=strategy, df=year_df, target_date=td)
            except Exception as e:
                logger.error(f"{td} 模拟失败: {e}")
                continue
            total = framework.cash + sum(
                p.get("qty", 0) * p.get("latest_price", p.get("avg_cost", 0))
                for p in framework.positions.values()
            )
            pct = (total / args.initial_cash - 1) * 100
            n_win = framework.n_winning
            n_tot = framework.n_total
            wr = n_win / n_tot * 100 if n_tot > 0 else 0
            print(f"  [{i+1}/{len(date_list)}] {td} | 持仓:{len(framework.positions):>2} | 交易:{len(framework.trades):>3} | 胜率:{wr:>5.1f}% | 总值:{total:>14,.0f} ({pct:+.2f}%)")

        framework.save_state()
        print(f"\n完成，状态已保存: {state_file}")
        print(f"最终: 现金 {framework.cash:,.0f} | 持仓 {len(framework.positions)} 只 | 总交易 {len(framework.trades)} 笔")
        return

    # 默认今天
    target_date = args.date or date.today().strftime("%Y-%m-%d")
    strategy_name = args.strategy

    # 状态文件
    portfolio_dir = Path("/root/.openclaw/workspace/portfolio")
    default_state = str(portfolio_dir / f"simulate_{strategy_name}.json")
    state_file = args.state_file or default_state

    # 显示名称映射
    strategy_display = {
        "v8": "V8模拟盘", "wavechan_v3_strict": "WaveChanV3铁律模拟盘"
    }
    display_name = strategy_display.get(strategy_name, f"{strategy_name}模拟盘")

    # V3 波浪缠论配置（与 backtest.py 保持一致）
    V3_MAX_POSITIONS = 5
    V3_POSITION_SIZE = 0.10

    # 初始化框架，根据策略读取对应配置
    strategy_params = STRATEGY_REGISTRY.get(strategy_name, {})
    params = strategy_params.get("params", {})

    mrf = None
    # V3 使用特殊配置（与 backtest.py 对齐）
    if strategy_name == "wavechan_v3_strict":
        mrf = MarketRegimeFilter(
            confirm_days=1,
            neutral_position=0.70,
            bear_position=0.30,
        )
        mrf.prepare(target_date, target_date)
        framework = BaseFramework(
            initial_cash=args.initial_cash,
            max_positions=V3_MAX_POSITIONS,
            position_size=V3_POSITION_SIZE,
            state_file=state_file,
            market_regime_filter=mrf,
        )
    elif strategy_name == "v8":
        # V8 使用 shared.py 中的配置
        framework = BaseFramework(
            initial_cash=args.initial_cash,
            max_positions=params.get("max_positions", 3),
            position_size=params.get("position_size", 0.20),
            state_file=state_file,
        )
    else:
        framework = BaseFramework(
            initial_cash=args.initial_cash,
            state_file=state_file,
        )

    # --reset
    if args.reset:
        framework.reset()
        print(f"状态已重置: {state_file}")
        return

    # --show-state / --show-trades / --show-candidates（只读，不运行交易）
    if args.show_state or args.show_trades or args.show_candidates:
        loaded = framework.load_state()
        if not loaded:
            print(f"状态文件不存在: {state_file}，无可用状态信息")
            return
        if args.show_state:
            show_state(framework, strategy_name, format=args.report_format)
        if args.show_trades:
            show_trades(framework, strategy_name)
        if args.show_candidates:
            show_candidates(framework, strategy_name, format=args.report_format)
        return

    # 正常模拟盘运行
    logger.info(f"\n{'='*50}\n  {display_name}  日期: {target_date}\n{'='*50}")

    try:
        strategy = load_strategy(strategy_name)
        year_df, df = load_data_for_date(strategy_name, target_date)

        # MarketRegimeFilter（按需初始化）
        mrf = None
        if not args.no_regime:
            mrf = MarketRegimeFilter()
            mrf.prepare(target_date, target_date)

        # --show-regime：只查询大盘状态，不运行交易
        if args.show_regime:
            if mrf is None:
                print("MarketRegimeFilter 已禁用（--no-regime）")
                return
            info = mrf.get_regime(target_date)
            print(f"\n{'=' * 50}")
            print(f"  大盘状态  |  日期: {target_date}")
            print(f"{'=' * 50}")
            print(f"  指数:     CSI300")
            print(f"  收盘:     {info['csi300_close']:.2f}")
            print(f"  MA20:     {info['ma20']:.2f}")
            print(f"  RSI14:    {info['rsi14']:.1f}")
            print(f"  Regime:   {info['regime']}")
            print(f"  Signal:  {info['signal']}")
            print(f"  仓位上限: {info['position_limit']:.0%}")
            print(f"{'=' * 50}")
            return

        # ── 策略数据准备（prepare 接口）────────────────────────────
        # 注意：传入完整 year df（不要过滤到单日），V8 等策略需要预计算全场日期条件
        if hasattr(strategy, 'prepare'):
            try:
                strategy.prepare([target_date], year_df)   # year_df 已在上面加载
            except Exception as prep_e:
                logger.warning(f"  策略 prepare() 失败: {prep_e}")
    except Exception as e:
        logger.error(f"数据/策略加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 先尝试恢复状态
    was_loaded = framework.load_state()
    if was_loaded:
        logger.info(f"状态已恢复: 现金 {framework.cash:,.0f} | 持仓 {len(framework.positions)} 只")
    else:
        logger.info("未找到历史状态，从初始资金开始")

    # MarketRegimeFilter
    if mrf is not None:
        framework.market_regime_filter = mrf
        regime_info = mrf.get_regime(target_date)
        logger.info(
            f"  大盘状态: {regime_info['regime']} | {regime_info['signal']} | "
            f"仓位上限: {regime_info['position_limit']:.0%} | RSI14: {regime_info['rsi14']:.1f}"
        )
    else:
        framework.market_regime_filter = None
        logger.info("  MarketRegimeFilter: 已禁用（全天候固定仓位）")

    try:
        all_dates = sorted(year_df["date"].unique().tolist())
        framework.run_simulate(
            strategy=strategy,
            df=year_df,
            target_date=target_date,
        )
        # 打印当日状态
        show_state(framework, strategy_name)
        logger.info(f"\n状态文件: {state_file}")
    except Exception as e:
        logger.error(f"模拟盘运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        os.close(lock_fd)

if __name__ == "__main__":
    main()
