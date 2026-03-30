#!/usr/bin/env python3
"""
V8 模拟盘每日运行脚本
===============================
每日收盘后运行，持续跟踪 v8(IC增强) 模拟盘

策略来源: v8_strategy.py (与回测脚本共用同一策略逻辑)

用法：
  python3 v8_simulated_trading.py              # 日常运行（自动读取上次状态继续）
  python3 v8_simulated_trading.py --reset      # 重置模拟盘（从100万重新开始）
  python3 v8_simulated_trading.py --show-state  # 查看当前持仓状态
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import gc
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from v8_strategy import (
    compute_conditions, apply_ic_filter, compute_v6_score,
    should_sell
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/v8_simulated_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---- 配置 ----
STATE_FILE = "/tmp/v8_sim_state.json"
OUT_DIR = "/root/.openclaw/workspace/projects/stock-analysis-system/paper_trading"
QQ_RECIPIENT = "c2c:B1DE0C2788382B67AF73F1C189A6A5C5"
DATA_DIR = '/root/.openclaw/workspace/data/warehouse'

INITIAL_CASH = 100_0000  # 100万
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001
TOP_N = 5


def load_latest_data():
    """加载2026年最新技术指标+日线数据"""
    tech = pd.read_parquet(f'{DATA_DIR}/technical_indicators_year=2026/data.parquet')
    daily = pd.read_parquet(f'{DATA_DIR}/daily_data_year=2026/data.parquet')
    df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
    del tech, daily
    gc.collect()
    
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    
    # 计算选股条件（使用共享模块）
    df = compute_conditions(df)
    
    return df


def run_day(state, daily, date, dates, i, n_dates):
    """每日交易逻辑"""
    p = state['portfolio']
    next_day = dates[i+1] if i+1 < n_dates else None
    
    # ===== 平仓 =====
    sold_today = []
    for sym, pos in list(p['positions'].items()):
        if not next_day:
            continue
        today = daily[daily['symbol'] == sym]
        if today.empty:
            continue
        row = today.iloc[0].to_dict()
        row['next_open'] = row.get('next_open', np.nan)
        
        sell, reason = should_sell(row, pos)
        if sell:
            next_open = row['next_open']
            proceeds = pos['qty'] * next_open * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
            pnl = proceeds - pos['qty'] * pos['avg_cost']
            p['cash'] += proceeds
            p['trades'].append({
                'date': date, 'symbol': sym, 'action': 'sell',
                'price': next_open, 'qty': pos['qty'], 'pnl': pnl, 'reason': reason
            })
            sold_today.append({'symbol': sym, 'pnl': pnl, 'reason': reason})
            del p['positions'][sym]
            logger.info(f"  卖出 {sym} @ {next_open:.2f} ({reason}), 盈亏 {pnl/10000:+.2f}万")
    
    # ===== 选股买入 =====
    bought_today = []
    if next_day and len(p['positions']) < TOP_N:
        # IC过滤 + V6条件
        candidates = apply_ic_filter(daily)
        if candidates.empty:
            return [], []
        
        # 计算V6评分 + IC评分（使用共享模块）
        candidates = compute_v6_score(candidates)
        candidates['v8_score'] = candidates['v6_score'] + candidates['ic_bonus']
        candidates = candidates.sort_values('v8_score', ascending=False)
        
        slots = TOP_N - len(p['positions'])
        for _, row in candidates.head(slots).iterrows():
            sym = row['symbol']
            if sym in p['positions']:
                continue
            next_open = row['next_open']
            if pd.isna(next_open) or next_open <= 0:
                continue
            
            max_per = p['cash'] * 0.20
            buy_qty = int(max_per / next_open / 100) * 100
            if buy_qty < 100:
                continue
            
            cost = buy_qty * next_open * (1 + COMMISSION + SLIPPAGE)
            if cost > p['cash']:
                continue
            
            p['cash'] -= cost
            p['positions'][sym] = {
                'qty': buy_qty,
                'avg_cost': next_open,
                'buy_date': date,
                'buy_price': next_open
            }
            p['trades'].append({
                'date': date, 'symbol': sym, 'action': 'buy',
                'price': next_open, 'qty': buy_qty,
                'v8_score': row['v8_score'],
                'v6_score': row['v6_score'],
                'ic_bonus': row['ic_bonus'],
                'cci': row['cci_20'],
                'wr': row['williams_r']
            })
            bought_today.append(sym)
            logger.info(f"  买入 {sym} @ {next_open:.2f}, score={row['v8_score']:.3f} (v6={row['v6_score']:.3f}+ic={row['ic_bonus']:.3f})")
    
    # ===== 记录当日价值 =====
    pv = 0
    for sym, pos in p['positions'].items():
        today = daily[daily['symbol'] == sym]
        if not today.empty:
            pv += today.iloc[0]['close'] * pos['qty']
        else:
            pv += pos['qty'] * pos['avg_cost']
    
    total = p['cash'] + pv
    state['history'].append({
        'date': date,
        'cash': round(p['cash'], 2),
        'total_value': round(total, 2),
        'n_positions': len(p['positions'])
    })
    
    return bought_today, sold_today


def save_state(state, filepath=STATE_FILE):
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_state(filepath=STATE_FILE):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def send_qq_notification(date: str, state: dict):
    """发送 QQ 推送"""
    p = state['portfolio']
    history = state['history']
    if not history:
        return
    
    initial = INITIAL_CASH
    final = history[-1]['total_value']
    total_ret = (final - initial) / initial * 100
    
    sells = [t for t in p['trades'] if t['action'] == 'sell']
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / max(1, len(sells)) * 100
    
    positions = []
    for sym, pos in p['positions'].items():
        positions.append({
            'symbol': sym,
            'qty': pos['qty'],
            'avg_cost': pos['avg_cost']
        })
    
    msg_lines = [
        f"📊 **V8 模拟盘日报**",
        f"📅 {date}",
        f"━━━━━━━━━━━━━━",
        f"💰 账户: {final/10000:.2f}万 ({total_ret:+.2f}%)",
        f"📈 交易: {len(p['trades'])}笔 | 胜率: {win_rate:.0f}%",
    ]
    
    if positions:
        msg_lines.append(f"📦 持仓 ({len(positions)}只):")
        for pos in positions:
            msg_lines.append(f"  • {pos['symbol']}: {pos['qty']}股 @ {pos['avg_cost']:.2f}")
    else:
        msg_lines.append("📦 持仓: 空仓")
    
    msg_lines.append(f"━━━━━━━━━━━━━━")
    msg_lines.append(f"策略: v6核心+IC增强过滤")
    msg_lines.append(f"止损5% | 止盈15% | MA死叉出场")
    
    msg = '\n'.join(msg_lines)
    
    try:
        qq_sender = "/root/.openclaw/workspace/skills/timetree/qqbot_direct_sender.py"
        import subprocess
        cmd = [sys.executable, qq_sender, "--to", QQ_RECIPIENT, "--message", msg]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("[V8] QQ通知已发送")
        else:
            logger.warning(f"[V8] QQ通知失败: {result.stderr}")
    except Exception as e:
        logger.warning(f"[V8] QQ通知异常: {e}")


def main():
    parser = argparse.ArgumentParser(description='V8 模拟盘每日运行')
    parser.add_argument('--reset', action='store_true', help='重置模拟盘')
    parser.add_argument('--show-state', action='store_true', help='显示当前状态')
    args = parser.parse_args()
    
    logger.info("加载数据...")
    df = load_latest_data()
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    logger.info(f"数据: {dates[0]} ~ {dates[-1]}, {n_dates}个交易日")
    
    if args.show_state:
        state = load_state()
        if state:
            h = state['history']
            if h:
                last = h[-1]
                print(f"\n📊 V8 模拟盘当前状态 ({last['date']})")
                print(f"  总价值: {last['total_value']/10000:.2f}万")
                print(f"  持仓数: {last['n_positions']}只")
                print(f"  持仓: {list(state['portfolio']['positions'].keys())}")
        else:
            print("无状态记录")
        return
    
    if args.reset:
        state = {
            'portfolio': {'cash': INITIAL_CASH, 'positions': {}, 'trades': []},
            'history': [],
            'last_date': None
        }
        logger.info("模拟盘已重置")
    else:
        state = load_state()
        if state is None:
            state = {
                'portfolio': {'cash': INITIAL_CASH, 'positions': {}, 'trades': []},
                'history': [],
                'last_date': None
            }
            logger.info("无历史状态，创建新模拟盘")
    
    last_date = state.get('last_date')
    if last_date and last_date in dates:
        start_idx = dates.index(last_date) + 1
        logger.info(f"从 {last_date} 继续运行 (索引 {start_idx})")
    else:
        start_idx = 0
        logger.info("从头开始运行")
    
    for i in range(start_idx, n_dates):
        date = dates[i]
        daily = df[df['date'] == date].copy()
        run_day(state, daily, date, dates, i, n_dates)
        state['last_date'] = date
        
        if i % 10 == 0 or i == n_dates - 1:
            h = state['history'][-1]
            logger.info(f"📅 {date}: {h['total_value']/10000:.2f}万 ({h['n_positions']}持仓)")
    
    save_state(state)
    
    if dates:
        send_qq_notification(dates[-1], state)
    
    h = state['history'][-1]
    sells = [t for t in state['portfolio']['trades'] if t['action'] == 'sell']
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / max(1, len(sells)) * 100
    
    print(f"\n{'='*50}")
    print(f"📊 V8 模拟盘 ({dates[0]} ~ {dates[-1]})")
    print(f"  最新价值: {h['total_value']/10000:.2f}万 ({(h['total_value']/INITIAL_CASH-1):+.2%})")
    print(f"  交易次数: {len(state['portfolio']['trades'])}笔")
    print(f"  胜率: {win_rate:.0f}%")
    print(f"  持仓: {list(state['portfolio']['positions'].keys())}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
