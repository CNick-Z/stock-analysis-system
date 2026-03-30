#!/usr/bin/env python3
"""
V3 模拟盘每日运行脚本
===============================
每日收盘后运行，持续跟踪模拟盘持仓

用法：
  python3 v3_simulated_trading.py                    # 日常运行（自动读取上次状态继续）
  python3 v3_simulated_trading.py --reset            # 重置模拟盘（从50万重新开始）
  python3 v3_simulated_trading.py --show-state       # 查看当前持仓状态

信号来源：L2 缓存（wavechan_daily_incremental.py 每日生成）
持仓跟踪：--save-state / --load-state 机制
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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
L2_CACHE_DIR = "/data/warehouse/wavechan/wavechan_cache/l2_hot_year=2026"
SIGNAL_FILE = "/tmp/wavechan_signals.json"
QQ_RECIPIENT = "c2c:B1DE0C2788382B67AF73F1C189A6A5C5"

# 回测参数（与 wavechan_selector 模拟盘一致）
INITIAL_CAPITAL = 500000
POSITION_LIMIT = 5
THRESHOLD = 50
TOP_N = 30


def get_latest_date_from_l2() -> str:
    """从 L2 缓存获取最新可用日期"""
    if not Path(L2_CACHE_DIR).exists():
        return None
    months = sorted(Path(L2_CACHE_DIR).iterdir(), reverse=True)
    for month_dir in months:
        for f in month_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(f, columns=["date"])
                return df["date"].max()
            except:
                continue
    return None


def load_today_signals(date: str) -> pd.DataFrame:
    """从 L2 缓存加载指定日期的信号"""
    if not Path(L2_CACHE_DIR).exists():
        return pd.DataFrame()

    year = date[:4]
    month_dir = Path(L2_CACHE_DIR) / f"year={year}"
    if not month_dir.exists():
        return pd.DataFrame()

    dfs = []
    for f in month_dir.glob(f"*.parquet"):
        try:
            df = pd.read_parquet(f)
            if "date" in df.columns:
                dfs.append(df[df["date"] == date])
        except:
            continue

    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    # 筛选有信号的股票
    if "has_signal" in result.columns:
        result = result[result["has_signal"] == True]
    if "total_score" in result.columns:
        result = result.sort_values("total_score", ascending=False)
    return result


def load_state() -> dict:
    """加载持仓状态"""
    if not Path(STATE_FILE).exists():
        return None
    with open(STATE_FILE, 'r') as f:
        return json.load(f)


def save_state(state: dict):
    """保存持仓状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def reset_state():
    """重置模拟盘（从头开始）"""
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
    """显示当前持仓状态"""
    state = load_state()
    if not state:
        print("模拟盘尚未初始化")
        return

    print(f"\n{'='*50}")
    print(f"V3 模拟盘状态（{state.get('last_date', 'N/A')}）")
    print(f"{'='*50}")
    print(f"现金: {state['cash']:,.2f}")
    print(f"持仓数: {len(state['positions'])}/{POSITION_LIMIT}")

    if state['positions']:
        print(f"\n持仓明细:")
        for symbol, pos in state['positions'].items():
            qty = pos['qty']
            entry_price = pos.get('entry_price', pos['cost']/qty)
            latest_price = pos.get('latest_price', entry_price)
            pnl = (latest_price - entry_price) * qty
            pnl_pct = (latest_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            entry_date = pos.get('entry_date', 'N/A')
            print(f"  {symbol} | 成本{entry_price:.2f} | 现价{latest_price:.2f} | {pnl_pct:+.1f}%({pnl:+,.0f}) | 入场{entry_date}")

    # 计算总资产
    pos_value = sum(pos.get('latest_price', pos['cost']/pos['qty']) * pos['qty']
                     for pos in state['positions'].values())
    total = state['cash'] + pos_value
    print(f"\n总资产: {total:,.2f}（初始 {INITIAL_CAPITAL:,.2f}，{((total/INITIAL_CAPITAL-1)*100):+.1f}%）")
    print(f"盈利交易: {state['winning_trades']}笔 / 总交易: {state['total_trades']}笔")
    print()


def run_daily_trading(target_date: str = None):
    """每日模拟交易"""
    if target_date is None:
        target_date = get_latest_date_from_l2()
        if not target_date:
            logger.error("无法获取最新日期，请手动指定 --date")
            return

    logger.info(f"="*50)
    logger.info(f"V3 模拟盘 {target_date}")
    logger.info(f"="*50)

    # 加载今日信号
    signals = load_today_signals(target_date)
    if signals.empty:
        logger.info(f"今日（{target_date}）无新信号")
        return

    logger.info(f"今日信号: {len(signals)} 只股票有信号")

    # 取 top_n
    top_signals = signals.head(TOP_N)
    actionable = []
    for _, row in top_signals.iterrows():
        symbol = row.get("symbol", row.get("code", ""))
        score = row.get("total_score", 0)
        sig_type = row.get("signal_type", "")
        if symbol and score >= THRESHOLD:
            actionable.append({
                "symbol": symbol,
                "score": score,
                "signal_type": sig_type,
                "close": row.get("close", 0),
                "wave_trend": row.get("wave_trend", ""),
            })

    if not actionable:
        logger.info(f"无达到阈值的信号（threshold={THRESHOLD}）")
        return

    # 加载状态
    state = load_state()
    if not state:
        logger.info("模拟盘未初始化，自动初始化...")
        state = {
            "last_date": None,
            "cash": INITIAL_CAPITAL,
            "positions": {},
            "history": [],
            "total_trades": 0,
            "winning_trades": 0,
        }

    # 恢复持仓信息（最新价从信号中获取）
    for symbol, pos in state["positions"].items():
        for row in signals.itertuples():
            if row.symbol == symbol:
                pos["latest_price"] = getattr(row, "close", pos.get("cost", 0) / pos["qty"] if pos["qty"] > 0 else 0)
                break

    # 计算当前总资产
    pos_value = sum(pos.get("latest_price", pos["cost"]/pos["qty"] if pos["qty"] > 0 else 0) * pos["qty"]
                     for pos in state["positions"].values())
    total_value = state["cash"] + pos_value

    logger.info(f"当前持仓: {list(state['positions'].keys())}")
    logger.info(f"现金: {state['cash']:,.2f} 总资产: {total_value:,.2f}")

    # 买入信号
    positions = state["positions"]
    slots = POSITION_LIMIT - len(positions)
    bought = []
    if slots > 0:
        for item in actionable:
            symbol = item["symbol"]
            if symbol in positions:
                continue
            if slots <= 0:
                break

            price = item["close"]
            # 每只股票分配等权仓位（总资产的20%，最多5只）
            per_stock_value = total_value * 0.2
            qty = int(per_stock_value / price / 100) * 100  # 按手买
            if qty < 100:
                continue

            cost = qty * price * 1.0003  # 含佣金
            if cost > state["cash"]:
                qty = int(state["cash"] / price / 100 / 1.0003) * 100
                cost = qty * price * 1.0003

            if qty < 100:
                continue

            state["cash"] -= cost
            positions[symbol] = {
                "qty": qty,
                "cost": qty * price,
                "entry_date": target_date,
                "entry_price": price,
                "latest_price": price,
                "signal_type": item["signal_type"],
                "signal_score": item["score"],
            }
            bought.append(symbol)
            slots -= 1
            logger.info(f"  买入 {symbol} @ {price:.2f} × {qty}股（信号:{item['signal_type']} score:{item['score']:.0f}）")

    # 卖出信号（止损/止盈/趋势破坏）
    sold = []
    for symbol, pos in list(positions.items()):
        # 查找该股票今日信号
        row_data = None
        for _, row in signals.iterrows():
            if row.get("symbol") == symbol or row.get("code") == symbol:
                row_data = row
                break

        latest = pos.get("latest_price", pos["cost"] / pos["qty"])
        cost_basis = pos["cost"] / pos["qty"]
        pnl_pct = (latest - cost_basis) / cost_basis if cost_basis > 0 else 0

        should_sell = False
        reason = ""

        # 止损：亏损超8%
        if pnl_pct <= -0.08:
            should_sell = True
            reason = f"止损({pnl_pct:.1%})"
        # 止盈：盈利超20%
        elif pnl_pct >= 0.20:
            should_sell = True
            reason = f"止盈({pnl_pct:.1%})"
        # 趋势破坏：wave_trend 变成 down
        elif row_data is not None:
            wave_trend = row_data.get("wave_trend", "")
            if wave_trend == "down" and pos.get("wave_trend") == "up":
                should_sell = True
                reason = f"趋势破坏({wave_trend})"

        if should_sell:
            sell_price = latest
            proceeds = pos["qty"] * sell_price * (1 - 0.0003 - 0.001)  # 佣金+印花税
            profit = proceeds - pos["cost"]
            state["cash"] += proceeds
            del positions[symbol]
            sold.append(symbol)
            state["total_trades"] += 1
            if profit > 0:
                state["winning_trades"] += 1
            logger.info(f"  卖出 {symbol} @ {sell_price:.2f}（{reason}，盈亏:{profit:+,.2f}）")

    # 更新状态
    state["last_date"] = target_date
    state["positions"] = positions
    save_state(state)

    # QQ 推送
    if bought or sold:
        send_qq_notification(target_date, state, bought, sold)

    # 汇总
    final_pos_value = sum(pos.get("latest_price", pos["cost"]/pos["qty"]) * pos["qty"]
                          for pos in state["positions"].values())
    final_total = state["cash"] + final_pos_value
    logger.info(f"今日结束：现金 {state['cash']:,.2f} 持仓 {len(positions)}只 总资产 {final_total:,.2f}")


def send_qq_notification(date: str, state: dict, bought: list, sold: list):
    """发送 QQ 推送（Markdown格式）"""
    if not bought and not sold:
        return

    lines = [f"📊 **V3模拟盘 {date}**\n"]

    if bought:
        lines.append(f"🟢 **买入 ({len(bought)}只)**\n")
        lines.append("| 代码 | 数量 | 成本 | 信号 | 评分 |")
        lines.append("|:----:|-----:|-----:|:----:|-----:|")
        for symbol in bought:
            pos = state["positions"].get(symbol, {})
            qty = pos.get('qty', 0)
            entry_price = pos.get('entry_price', 0)
            sig_type = pos.get('signal_type', '')
            score = pos.get('signal_score', 0)
            lines.append(f"| {symbol} | {qty}股 | {entry_price:.2f} | {sig_type} | {score:.0f} |")

    if sold:
        lines.append(f"\n🔴 **卖出 ({len(sold)}只)**")
        lines.append("| 代码 |")
        lines.append("|:----:|")
        for symbol in sold:
            lines.append(f"| {symbol} |")

    # 持仓明细
    if state["positions"]:
        lines.append(f"\n📋 **持仓明细**\n")
        lines.append("| 代码 | 成本 | 现价 | 盈亏% | 盈亏额 | 入场日期 |")
        lines.append("|:----:|-----:|-----:|------:|------:|:--------:|")
        for symbol, pos in state["positions"].items():
            qty = pos.get('qty', 0)
            entry_price = pos.get('entry_price', 0)
            latest_price = pos.get('latest_price', entry_price)
            cost_basis = pos['cost'] / qty if qty > 0 else 0
            pnl = (latest_price - cost_basis) * qty if qty > 0 else 0
            pnl_pct = (latest_price - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
            entry_date = pos.get('entry_date', 'N/A')
            lines.append(f"| {symbol} | {entry_price:.2f} | {latest_price:.2f} | {pnl_pct:+.1f}% | {pnl:+,.0f} | {entry_date} |")

    # 汇总
    pos_value = sum(pos.get('latest_price', pos['cost']/pos['qty']) * pos['qty']
                   for pos in state['positions'].values())
    total = state['cash'] + pos_value
    lines.append(f"\n💰 **资金概况**\n")
    lines.append(f"- 现金：{state['cash']:,.0f}")
    lines.append(f"- 总资产：**{total:,.0f}**（{((total/INITIAL_CAPITAL-1)*100):+.1f}%）")
    lines.append(f"- 持仓：{len(state['positions'])}/{POSITION_LIMIT}只 | 盈利交易：{state['winning_trades']}/{state['total_trades']}笔")

    msg = "\n".join(lines)

    # 通过 qqbot_direct_sender.py 发送（统一QQ接口）
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 模拟盘每日运行")
    parser.add_argument("--date", type=str, default=None, help="指定日期（默认自动获取最新）")
    parser.add_argument("--reset", action="store_true", help="重置模拟盘")
    parser.add_argument("--show-state", action="store_true", help="查看当前持仓状态")
    args = parser.parse_args()

    if args.reset:
        reset_state()
    elif args.show_state:
        show_state()
    else:
        run_daily_trading(args.date)
