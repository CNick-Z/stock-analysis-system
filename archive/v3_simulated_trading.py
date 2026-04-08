#!/usr/bin/env python3
"""
V3 波浪缠论模拟盘 - 统一框架版
============================================================

统一框架：BasePortfolio（所有策略共用）
策略信号：WaveChanStrategy（波浪缠论自有逻辑）

用法：
  python3 v3_simulated_trading.py                    # 日常运行（自动读取上次状态继续）
  python3 v3_simulated_trading.py --reset          # 重置模拟盘（从50万重新开始）
  python3 v3_simulated_trading.py --show-state      # 查看当前持仓状态）
  python3 v3_simulated_trading.py --backtest        # 从2026-01-01回测到今天
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_portfolio import BasePortfolio
from strategies.wavechan.v3_l2_cache.wavechan_strategy import WaveChanStrategy

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/v3_simulated_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---- 配置 ----
STATE_FILE = "/tmp/v3_sim_state.json"
INITIAL_CAPITAL = 500000
QQ_RECIPIENT = "c2c:B1DE0C2788382B67AF73F1C189A6A5C5"

# L2 缓存根目录（支持年月子目录）
L2_CACHE_ROOT = "/data/warehouse/wavechan/wavechan_cache"

# 回测参数
THRESHOLD = 50          # 买入阈值（total_score >= THRESHOLD）
POSITION_LIMIT = 5      # 最大持仓数
POSITION_SIZE = 0.20   # 每只股票仓位比例

# 涨跌停边界（9.5%容错）
LIMIT_UP_PCT = 0.095
LIMIT_DOWN_PCT = 0.095


# ================================================================
# L2 缓存数据加载
# ================================================================

def find_l2_cache_dir(year: int) -> Optional[Path]:
    """查找指定年份的L2缓存目录（支持 year=2026 → l2_hot_year=2026_month=XX）"""
    if not Path(L2_CACHE_ROOT).exists():
        return None
    # 优先精确匹配 l2_hot_year=YYYY 目录
    exact = Path(L2_CACHE_ROOT) / f"l2_hot_year={year}"
    if exact.exists():
        return exact
    # 查找年月子目录
    candidates = sorted(
        [p for p in Path(L2_CACHE_ROOT).iterdir()
         if p.name.startswith(f"l2_hot_year={year}_month=")],
        key=lambda p: p.name
    )
    return candidates[-1] if candidates else None


def load_l2_signals_for_date(target_date: str) -> pd.DataFrame:
    """
    从L2缓存加载指定日期的信号数据

    Returns:
        DataFrame with columns: date, symbol, total_score, has_signal,
        signal_type, wave_trend, stop_loss, close, open, high, low, volume, ...
    """
    year = int(target_date[:4])
    month = target_date[5:7]

    # 直接定位到对应的月份 data.parquet
    cache_path = Path(L2_CACHE_ROOT) / f"l2_hot_year={year}_month={month}" / "data.parquet"

    if not cache_path.exists():
        logger.debug(f"L2缓存不存在: {cache_path}")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            return pd.DataFrame()

        result = df[df["date"] == target_date].copy()

        # 过滤有信号的股票
        if "has_signal" in result.columns:
            result = result[result["has_signal"] == True]

        return result
    except Exception as e:
        logger.warning(f"读取 {cache_path} 失败: {e}")
        return pd.DataFrame()


def get_latest_signal_date() -> Optional[str]:
    """获取L2缓存中最新的有信号日期"""
    cache_root = Path(L2_CACHE_ROOT)
    if not cache_root.exists():
        return None

    # 扫描所有月份目录找最新文件
    latest_date = None
    for month_dir in sorted(cache_root.iterdir(), reverse=True):
        if not month_dir.name.startswith("l2_hot_year="):
            continue
        data_file = month_dir / "data.parquet"
        if data_file.exists():
            try:
                df = pd.read_parquet(data_file, columns=["date"])
                if "date" in df.columns and not df.empty:
                    d = df["date"].max()
                    if latest_date is None or d > latest_date:
                        latest_date = d
                    break  # 目录内文件已是最新的，直接用
            except:
                continue
    return latest_date


def load_all_signals_for_backtest(start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载回测区间内所有日期的信号
    返回: DataFrame with date, symbol, total_score, has_signal, signal_type,
          wave_trend, stop_loss, close, open, next_open, high, low, volume 等
    """
    all_dfs = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        df = load_l2_signals_for_date(date_str)
        if not df.empty:
            df['next_open'] = df.get('open', df['close'])
            all_dfs.append(df)
        current += timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# ================================================================
# 持仓状态管理（与 BasePortfolio 兼容）
# ================================================================

def load_state() -> Optional[dict]:
    if not Path(STATE_FILE).exists():
        return None
    with open(STATE_FILE, 'r') as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def reset_state():
    state = {
        "last_date": None,
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "history": [],
        "total_trades": 0,
        "winning_trades": 0,
    }
    save_state(state)
    logger.info(f"模拟盘已重置，初始资金 {INITIAL_CAPITAL:,.2f}")


def show_state():
    state = load_state()
    if not state:
        print("模拟盘尚未初始化")
        return

    print(f"\n{'='*50}")
    print(f"V3 模拟盘状态（{state.get('last_date', 'N/A')}）")
    print(f"{'='*50}")
    print(f"现金: {state['cash']:,.2f}")
    print(f"持仓数: {len(state['positions'])}/5")

    if state['positions']:
        print(f"\n持仓明细:")
        for symbol, pos in state['positions'].items():
            qty = pos.get('qty', 0)
            entry_price = pos.get('avg_cost', 0)
            latest_price = pos.get('latest_price', entry_price)
            pnl = (latest_price - entry_price) * qty
            pnl_pct = (latest_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            entry_date = pos.get('entry_date', 'N/A')
            print(f"  {symbol} | 成本{entry_price:.2f} | 现价{latest_price:.2f} | {pnl_pct:+.1f}%({pnl:+,.0f}) | 入场{entry_date}")

    pos_value = sum(pos.get('latest_price', pos['cost']/pos['qty']) * pos['qty']
                     for pos in state['positions'].values())
    total = state['cash'] + pos_value
    print(f"\n总资产: {total:,.2f}（初始 {INITIAL_CAPITAL:,.2f}，{((total/INITIAL_CAPITAL-1)*100):+.1f}%）")
    print(f"盈利交易: {state['winning_trades']}笔 / 总交易: {state['total_trades']}笔")


# ================================================================
# QQ 推送
# ================================================================

def send_qq_notification(date: str, portfolio: BasePortfolio, bought: list, sold: list):
    if not bought and not sold:
        return

    positions = portfolio.positions
    lines = [f"📊 **V3波浪模拟盘 {date}**\n"]

    if bought:
        lines.append(f"🟢 **买入 ({len(bought)}只)**\n")
        lines.append("| 代码 | 数量 | 成本 | 信号 | 评分 |")
        lines.append("|:----:|-----:|-----:|:----:|-----:|")
        for sym in bought:
            pos = positions.get(sym, {})
            lines.append(f"| {sym} | {pos.get('qty', 0)}股 | {pos.get('avg_cost', 0):.2f} | {pos.get('signal_type', '')} | {pos.get('signal_score', 0):.0f} |")

    if sold:
        lines.append(f"\n🔴 **卖出 ({len(sold)}只)**")
        lines.append("| 代码 | 盈亏 | 原因 |")
        lines.append("|:----:|-----:|:----:|")
        for t in sold:
            lines.append(f"| {t['symbol']} | {t.get('pnl', 0):+,.0f} | {t.get('reason', '')} |")

    # 持仓明细
    if positions:
        lines.append(f"\n📋 **持仓明细**\n")
        lines.append("| 代码 | 成本 | 现价 | 盈亏% | 盈亏额 | 入场日期 |")
        lines.append("|:----:|-----:|-----:|------:|------:|:--------:|")
        for sym, pos in positions.items():
            qty = pos.get('qty', 0)
            entry_price = pos.get('avg_cost', 0)
            latest_price = pos.get('latest_price', entry_price)
            pnl = (latest_price - entry_price) * qty
            pnl_pct = (latest_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            lines.append(f"| {sym} | {entry_price:.2f} | {latest_price:.2f} | {pnl_pct:+.1f}% | {pnl:+,.0f} | {pos.get('entry_date', 'N/A')} |")

    # 资金概况
    stats = portfolio.get_stats()
    lines.append(f"\n💰 **资金概况**\n")
    lines.append(f"- 现金：{stats['cash']:,.0f}")
    lines.append(f"- 总资产：**{stats['total_value']:,.0f}**（{stats['total_return_pct']}）")
    lines.append(f"- 持仓：{stats['n_positions']}/5只 | 胜率：{stats['win_rate']:.0f}%（{stats['n_winning']}/{stats['n_trades']}笔）")

    msg = "\n".join(lines)

    qq_sender = "/root/.openclaw/workspace/skills/timetree/qqbot_direct_sender.py"
    import subprocess
    try:
        subprocess.run(
            [sys.executable, qq_sender, "--to", QQ_RECIPIENT, "--message", msg],
            capture_output=True, timeout=30
        )
        logger.info(f"[V3] QQ通知已发送")
    except Exception as e:
        logger.warning(f"[V3] QQ通知失败: {e}")


# ================================================================
# 统一框架：基于 BasePortfolio 的每日调度
# ================================================================

def run_with_portfolio(target_date: str, portfolio: BasePortfolio):
    """
    使用 BasePortfolio 统一框架执行当日交易
    """
    strategy = portfolio.strategy

    # 加载当日信号
    daily_signals = load_l2_signals_for_date(target_date)
    if daily_signals.empty:
        logger.info(f"今日（{target_date}）无信号")
        return

    # 给每个持仓的股票附加最新价格和wave_trend
    for sym, pos in list(portfolio.positions.items()):
        row = daily_signals[daily_signals['symbol'] == sym]
        if not row.empty:
            r = row.iloc[0]
            pos['latest_price'] = r.get('close', pos.get('avg_cost', 0))
            pos['wave_trend'] = r.get('wave_trend', 'neutral')
            pos['signal_type'] = r.get('signal_type', '')

    # 调用框架的 on_day（自动处理：先出场，再买入）
    trades = portfolio.on_day(target_date, daily_signals)

    # 拆分买卖交易
    bought = [t for t in trades if t['action'] == 'buy']
    sold = [t for t in trades if t['action'] == 'sell']

    # 更新持仓的最新价格
    for sym, pos in portfolio.positions.items():
        row = daily_signals[daily_signals['symbol'] == sym]
        if not row.empty:
            pos['latest_price'] = row.iloc[0].get('close', pos.get('avg_cost', 0))

    # QQ 推送
    if bought or sold:
        send_qq_notification(target_date, portfolio, [t['symbol'] for t in bought], sold)

    # 日志输出
    sell_value = sum(t['pnl'] for t in sold)
    stats = portfolio.get_stats()
    logger.info(
        f"今日结束：现金 {stats['cash']:,.0f} | "
        f"持仓 {stats['n_positions']}只 | "
        f"总资产 {stats['total_value']:,.0f}（{stats['total_return_pct']}） | "
        f"胜率 {stats['win_rate']:.0f}%"
    )


# ================================================================
# 回测模式
# ================================================================

def run_backtest(start_date: str, end_date: str):
    """
    回测模式：从 start_date 到 end_date 逐日运行
    使用 BasePortfolio 统一框架
    """
    logger.info(f"=" * 60)
    logger.info(f"V3 波浪缠论模拟盘 回测模式")
    logger.info(f"区间：{start_date} ~ {end_date}")
    logger.info(f"=" * 60)

    # 创建策略和组合
    strategy = WaveChanStrategy(
        threshold=THRESHOLD,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        max_hold_days=20,
        max_positions=5,
        position_size=0.20,
    )

    portfolio = BasePortfolio(
        name="V3_WaveChan",
        initial_cash=INITIAL_CAPITAL,
        strategy=strategy,
    )

    # 预加载所有信号（按日期分组）
    all_signals = load_all_signals_for_backtest(start_date, end_date)
    if all_signals.empty:
        logger.error("无信号数据，请检查L2缓存")
        return

    dates = sorted(all_signals['date'].unique())
    logger.info(f"回测区间共 {len(dates)} 个交易日")

    for i, date in enumerate(dates):
        daily = all_signals[all_signals['date'] == date].copy()

        # 更新持仓信息（最新价格）
        for sym, pos in list(portfolio.positions.items()):
            row = daily[daily['symbol'] == sym]
            if not row.empty:
                r = row.iloc[0]
                pos['latest_price'] = r.get('close', pos.get('avg_cost', 0))
                pos['wave_trend'] = r.get('wave_trend', 'neutral')
                pos['signal_type'] = r.get('signal_type', '')
                pos['stop_loss'] = r.get('stop_loss', 0)

        # 框架处理
        trades = portfolio.on_day(date, daily)
        bought = [t for t in trades if t['action'] == 'buy']
        sold = [t for t in trades if t['action'] == 'sell']

        # 更新最新价格
        for sym, pos in portfolio.positions.items():
            row = daily[daily['symbol'] == sym]
            if not row.empty:
                pos['latest_price'] = row.iloc[0].get('close', pos.get('avg_cost', 0))

        stats = portfolio.get_stats()
        logger.info(
            f"📅 {date} ({i+1}/{len(dates)}): "
            f"{stats['total_value']/10000:.2f}万 "
            f"({stats['total_return_pct']}) "
            f"| 持仓{stats['n_positions']}只 | "
            f"买入{len(bought)}笔 卖出{len(sold)}笔"
        )

    # 回测结束汇总
    stats = portfolio.get_stats()
    closed_trades = [t for t in portfolio.trades if t['action'] == 'sell']

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 V3 波浪缠论 回测汇总（{start_date} ~ {end_date}）")
    logger.info(f"{'='*60}")
    logger.info(f"  最终价值: {stats['total_value']:,.0f}（{stats['total_return_pct']}）")
    logger.info(f"  初始资金: {stats['initial_cash']:,.0f}")
    logger.info(f"  交易次数: {stats['n_trades']}笔")
    logger.info(f"  胜率: {stats['win_rate']:.1f}%（{stats['n_winning']}赢/{stats['n_trades']}总）")
    logger.info(f"  最大回撤: ---（需单独计算）")

    # 按出场原因统计
    if closed_trades:
        from collections import defaultdict
        reason_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in closed_trades:
            reason = t.get('reason', 'UNKNOWN')
            reason_stats[reason]['count'] += 1
            reason_stats[reason]['pnl'] += t.get('pnl', 0)
        logger.info(f"\n出场原因分布:")
        for reason, s in sorted(reason_stats.items(), key=lambda x: -x[1]['count']):
            logger.info(f"  {reason}: {s['count']}笔, 盈亏{s['pnl']:+,.0f}")


# ================================================================
# 每日运行模式
# ================================================================

def run_daily(target_date: str = None):
    """每日运行（从状态文件恢复，继续交易）"""
    if target_date is None:
        target_date = get_latest_signal_date()
        if not target_date:
            logger.error("无法获取最新日期，请手动指定 --date")
            return

    logger.info(f"=" * 50)
    logger.info(f"V3 波浪缠论模拟盘 {target_date}")
    logger.info(f"=" * 50)

    # 加载或创建组合
    state = load_state()
    strategy = WaveChanStrategy()

    portfolio = BasePortfolio(
        name="V3_WaveChan",
        initial_cash=INITIAL_CAPITAL,
        strategy=strategy,
    )

    if state:
        # 从状态恢复
        portfolio.cash = state.get('cash', INITIAL_CAPITAL)
        portfolio.positions = state.get('positions', {})
        # 恢复持仓的额外字段
        for sym, pos in portfolio.positions.items():
            pos['signal_type'] = pos.get('signal_type', '')
            pos['wave_trend'] = pos.get('wave_trend', 'neutral')
            pos['days_held'] = pos.get('days_held', 0)
            pos['consecutive_bad_days'] = pos.get('consecutive_bad_days', 0)

    # 执行当日交易
    run_with_portfolio(target_date, portfolio)

    # 保存状态
    save_state({
        "last_date": target_date,
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "history": [],
        "total_trades": len([t for t in portfolio.trades if t['action'] == 'sell']),
        "winning_trades": len([t for t in portfolio.trades if t['action'] == 'sell' and t.get('pnl', 0) > 0]),
    })


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 波浪缠论模拟盘（统一框架版）")
    parser.add_argument("--date", type=str, default=None, help="指定日期（默认自动获取最新）")
    parser.add_argument("--reset", action="store_true", help="重置模拟盘")
    parser.add_argument("--show-state", action="store_true", help="查看当前持仓状态")
    parser.add_argument("--backtest", action="store_true", help="回测模式")
    parser.add_argument("--start", type=str, default="2026-01-01", help="回测开始日期")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期")
    args = parser.parse_args()

    if args.reset:
        reset_state()
    elif args.show_state:
        show_state()
    elif args.backtest:
        end_date = args.end or (datetime.now().strftime("%Y-%m-%d"))
        run_backtest(args.start, end_date)
    else:
        run_daily(args.date)
