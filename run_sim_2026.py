"""
批量运行 2026 年 WaveChan V3 模拟盘
全年数据只加载一次，逐日运行
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import logging
import json
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from simulator.base_framework import BaseFramework
from simulator.shared import load_strategy
from simulator.market_regime import MarketRegimeFilter
from utils.data_loader import load_strategy_data
from simulator.shared import add_next_open

# ── 命令行参数 ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--l2-only', action='store_true', help='仅用L2信号，不触发L1 Fallback')
parser.add_argument('--reset', action='store_true', help='重置状态')
parser.add_argument('--date', type=str, default=None, help='目标日期（预留）')
parser.add_argument('--strategy', type=str, default='wavechan_v3_strict', help='策略名称: v8 / wavechan_v3_strict')
parser.add_argument('--state-file', type=str, default=None, help='状态文件路径')
args, _ = parser.parse_known_args()

# ── 加载 2026 全年数据（只加载一次）─────────────────────────
logger.info("加载 2026 年数据...")
year_df = load_strategy_data(years=[2026], add_money_flow=True)
year_df = add_next_open(year_df)
all_dates = sorted(year_df['date'].unique().tolist())
logger.info(f"数据: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日, {len(year_df):,} 行")

strategy_name = args.strategy or 'wavechan_v3_strict'
# 状态文件
if args.state_file:
    state_file = args.state_file
else:
    state_file = f"/tmp/simulate_{strategy_name}_{2026}.json"

# ── 初始化框架 ─────────────────────────────────────────────
framework = BaseFramework(initial_cash=1_000_000, state_file=state_file)
if args.reset:
    framework.reset()

strategy = load_strategy(strategy_name)
if args.l2_only:
    strategy._l2_only_mode = True
    logger.info("[L2-ONLY] 模式已启用，不使用L1 Fallback")
mrf = MarketRegimeFilter()
mrf.prepare(all_dates[0], all_dates[-1])

initial_cash = 1_000_000.0
wins, losses = 0, 0
equity_curve = []

# ── 逐日运行 ───────────────────────────────────────────────
for i, date in enumerate(all_dates):
    try:
        df_day = year_df[year_df["date"] == date].copy()
        if df_day.empty:
            logger.warning(f"[{date}] 无数据，跳过")
            continue

        # MarketRegime
        framework.market_regime_filter = mrf

        # 策略 prepare
        if hasattr(strategy, 'prepare'):
            strategy.prepare([date], df_day)

        # 运行当日
        framework.run_simulate(strategy=strategy, df=df_day, target_date=date, dates=all_dates)

        # 收集结果
        total_value = framework._calc_total_value(df_day, date)
        equity_curve.append({'date': date, 'value': total_value})

        pos_count = len(framework.positions)
        ret = total_value / initial_cash - 1

        # 每20天打印进度
        if (i + 1) % 20 == 0 or i == 0 or date == all_dates[-1]:
            logger.info(f"[{date}] #{i+1}/{len(all_dates)} 现金:{framework.cash:,.0f} 持仓:{pos_count}只 总值:{total_value:,.0f} ({ret:+.2%})")

    except ValueError as e:
        logger.warning(f"[{date}] 跳过: {e}")
    except Exception as e:
        logger.error(f"[{date}] 错误: {e}")
        import traceback; traceback.print_exc()
        break

# ── 最终报告 ───────────────────────────────────────────────
final_value = equity_curve[-1]['value'] if equity_curve else initial_cash
total_return = final_value / initial_cash - 1
closed_trades = [t for t in framework.trades if t['action'] == 'sell']
wins = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
losses = sum(1 for t in closed_trades if t.get('pnl', 0) <= 0)

logger.info("=" * 60)
logger.info(f"2026年 WaveChan V3 模拟完成  ({all_dates[0]} ~ {all_dates[-1]})")
logger.info(f"初始资金: {initial_cash:,.0f}")
logger.info(f"最终价值: {final_value:,.0f}  ({total_return:+.2%})")
logger.info(f"总交易笔数: {len(closed_trades)}  胜: {wins}  负: {losses}  胜率: {wins/max(len(closed_trades),1):.1%}")

# 保存详细结果
result = {
    'strategy': 'wavechan_v3_strict',
    'period': f"{all_dates[0]} ~ {all_dates[-1]}",
    'initial_cash': initial_cash,
    'final_value': final_value,
    'total_return': float(total_return),
    'total_trades': len(closed_trades),
    'wins': wins,
    'losses': losses,
    'win_rate': wins / max(len(closed_trades), 1),
    'equity_curve': [{'date': str(e['date']), 'value': float(e['value'])} for e in equity_curve],
    'trades': [dict(t, pnl=float(t.get('pnl', 0))) for t in closed_trades],
}
out_path = '/tmp/sim_2026_wavechan_v3_result.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
logger.info(f"详细结果已保存: {out_path}")
