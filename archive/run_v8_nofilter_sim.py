#!/usr/bin/env python3
"""
run_v8_nofilter_sim.py — V8 No-Regime 每日模拟盘
====================================================
复用 paper_trading_sim.py 的 load_data（含条件列），
用 BaseFramework 运行，禁用 MarketRegimeFilter。

流程：
  1. load_data 加载全年 + 基础条件列
  2. 注入 turnover_rate 修复（parquet 原值全为0）
  3. 注入 IC 条件（ic_exclude / ic_bonus / _ic_buy）
  4. 构建 _ic_cache（策略只依赖缓存，不再依赖 prepare）
  5. BaseFramework.run_simulate() 运行模拟盘

用法:
  # 今日运行:
  python3 run_v8_nofilter_sim.py

  # 追数（一次性）:
  python3 run_v8_nofilter_sim.py --catchup

  # 单日运行:
  python3 run_v8_nofilter_sim.py --date 2026-04-02

  # 查看状态:
  python3 run_v8_nofilter_sim.py --show-state

  # 重置:
  python3 run_v8_nofilter_sim.py --reset
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework
from strategies.score.v8.strategy import ScoreV8Strategy
from paper_trading_sim import load_data   # 含基础条件列

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STATE_FILE = "/tmp/simulate_v8_nofilter.json"
INITIAL_CASH = 800_000
SIGNAL_DIR = Path("/root/.openclaw/workspace/portfolio/signals/v8_nofilter")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def patch_ic_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    注入 V8 IC 条件列：ic_exclude / ic_bonus / _ic_buy
    同时修复 turnover_rate（全为0时从量价重建）。
    返回带完整条件列的 df。
    """
    df = df.copy()

    # ── turnover_rate 修复 ────────────────────────────────────────
    # 两个 parquet 合并后产生 _x/_y 后缀，且值全为0
    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']
    # 如果仍全为0，从量价重建：turnover = volume*100 / (amount/close) * 100
    if df['turnover_rate'].sum() == 0 and 'amount' in df.columns:
        df['turnover_rate'] = (
            df['volume'] * 100 * df['close'] /
            df['amount'].replace(0, float('nan'))
        ).fillna(0).clip(0, 30)
        logger.debug("turnover_rate 从量价重建")

    # ── IC 剔除条件 ───────────────────────────────────────────────
    df['ic_exclude'] = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 35) |
        (df['turnover_rate'] > 2.79) |
        (df['williams_r'] < -90) |
        (df['cci_20'] < -200)
    )

    # ── IC 加分 ──────────────────────────────────────────────────
    df['ic_bonus'] = 0.0
    df.loc[df['cci_20'] < -100, 'ic_bonus'] += 0.10
    df.loc[df['williams_r'] < -80, 'ic_bonus'] += 0.05
    df.loc[df['turnover_rate'] < 0.42, 'ic_bonus'] += 0.05
    df.loc[df['vol_ratio'] < 0.71, 'ic_bonus'] += 0.05

    # ── _ic_buy（V8 全部条件AND）─────────────────────────────────
    df['_ic_buy'] = (
        (~df['ic_exclude']) &
        (df['close'] >= df['open']) &
        (df['high'] <= df['open'] * 1.06) &
        df['ma_condition'] &
        df['volume_condition'] &
        df['macd_condition'] &
        (df['jc_condition'] | df['macd_jc']) &
        df['trend_condition'] &
        df['rsi_filter'] &
        df['price_filter']
    )

    return df


def build_ic_cache(year_df: pd.DataFrame) -> dict:
    """
    构建 _ic_cache {date → DataFrame(index=symbol)}。
    V8 filter_buy 依赖此缓存，从缓存读取 _ic_buy / ic_bonus 等。
    """
    # 添加下划线别名（V8 filter_buy 依赖下划线前缀）
    alias_map = {
        'ma_condition': '_ma_cond',
        'macd_condition': '_macd_cond',
        'volume_condition': '_vol_cond',
        'macd_jc': '_macd_jc',
        'jc_condition': '_jc_cond',
    }
    for src, dst in alias_map.items():
        if src in year_df.columns and dst not in year_df.columns:
            year_df[dst] = year_df[src]

    cache_cols = [
        'symbol', 'date',
        '_ic_buy', 'ic_bonus', '_sma10_change',
        '_ma_cond', '_macd_cond', '_vol_cond', '_macd_jc',
        'growth',
    ]
    # 确保这些列存在
    existing = [c for c in cache_cols if c in year_df.columns]
    missing = set(cache_cols) - set(existing)
    if missing:
        logger.warning(f"缺少列，跳过: {missing}")

    cache_df = year_df[existing].copy()
    ic_cache = {}
    for dt, grp in cache_df.groupby('date'):
        ic_cache[str(dt)] = grp.set_index('symbol')
    logger.info(f"IC缓存构建完成: {len(ic_cache)} 个交易日")
    return ic_cache


def write_daily_signal(target_date: str, framework: BaseFramework):
    """将当日状态写入信号文件（供 Reporter 读取）"""
    total_value = framework.cash + sum(
        pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        for pos in framework.positions.values()
    )
    positions = []
    for sym, pos in framework.positions.items():
        latest = pos.get("latest_price", pos["avg_cost"])
        pnl = (latest / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
        positions.append({
            "symbol": sym,
            "qty": pos["qty"],
            "avg_cost": pos["avg_cost"],
            "latest_price": latest,
            "pnl_pct": round(pnl, 2),
            "days_held": pos.get("days_held", 0),
        })

    closed = [t for t in framework.trades if t["action"] == "sell"]
    win_rate = framework.n_winning / max(framework.n_total, 1) * 100
    total_pnl = sum(t["pnl"] for t in closed)

    signal = {
        "date": target_date,
        "strategy": "V8(NoRegime)",
        "state": {
            "cash": round(framework.cash, 2),
            "positions_count": len(framework.positions),
            "total_value": round(total_value, 2),
            "total_return_pct": round((total_value / INITIAL_CASH - 1) * 100, 2),
            "n_trades": framework.n_total,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
        },
        "positions": positions,
        "buy_signals": [],   # 当日无新信号（下个交易日才入场）
        "sell_signals": [],  # 出场信号从 trades 里看
    }

    out_file = SIGNAL_DIR / f"{target_date}.json"
    with open(out_file, 'w') as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)
    logger.info(f"信号已写入: {out_file}")


def run_sim_for_date(target_date: str, year_df: pd.DataFrame,
                     ic_cache: dict, reset: bool = False) -> BaseFramework:
    """对单个日期运行模拟盘"""
    strategy = ScoreV8Strategy(config={
        'stop_loss': 0.05,
        'take_profit': 0.20,
        'max_positions': 5,
        'position_size': 0.20,
    })
    # 注入预建缓存（策略内部不再依赖 prepare）
    strategy._ic_cache = ic_cache

    framework = BaseFramework(
        initial_cash=INITIAL_CASH,
        max_positions=5,
        position_size=0.20,
        state_file=STATE_FILE,
        market_regime_filter=None,   # 禁用 regime filter
    )

    if reset:
        framework.reset()

    was_loaded = framework.load_state()
    if was_loaded:
        logger.info(f"状态已恢复: 现金 {framework.cash:,.0f} | 持仓 {len(framework.positions)} 只")
    else:
        logger.info("未找到历史状态，从初始资金开始")

    # 过滤到目标日期
    daily_df = year_df[year_df["date"] == target_date].copy()
    if daily_df.empty:
        raise ValueError(f"{target_date} 无数据")

    # 注入最新价（用于计算浮动盈亏）
    latest_prices = daily_df.set_index('symbol')['close'].to_dict()
    for sym, pos in framework.positions.items():
        pos['latest_price'] = latest_prices.get(sym, pos.get('latest_price', pos['avg_cost']))

    framework.run_simulate(strategy=strategy, df=daily_df, target_date=target_date)
    framework.save_state()

    return framework


def run_catchup(start_date: str = "2026-01-01", end_date: str = None):
    """逐日追数（一次性）"""
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    year = int(start_date[:4])
    logger.info(f"追数: {start_date} → {end_date}（year={year}）")

    # 加载全年数据 + 计算 IC 条件 + 构建缓存
    year_df = load_data(year, None)
    year_df = patch_ic_conditions(year_df)
    ic_cache = build_ic_cache(year_df)

    # 重置状态
    framework = BaseFramework(
        initial_cash=INITIAL_CASH,
        state_file=STATE_FILE,
        market_regime_filter=None,
    )
    framework.reset()
    logger.info(f"状态已重置: {STATE_FILE}")

    all_dates = sorted(year_df['date'].unique())
    dates = [d for d in all_dates if start_date <= d <= end_date]
    logger.info(f"共 {len(dates)} 个交易日")

    for i, dt in enumerate(dates):
        try:
            fw = run_sim_for_date(dt, year_df, ic_cache)
        except Exception as e:
            logger.error(f"{dt}: 模拟失败 - {e}")
            continue

        if (i + 1) % 20 == 0:
            total_val = fw.cash + sum(
                pos.get('latest_price', pos['avg_cost']) * pos['qty']
                for pos in fw.positions.values()
            )
            logger.info(
                f"  [{i+1}/{len(dates)}] {dt}: "
                f"总值{total_val/1e4:.1f}万 | 交易{fw.n_total}笔 | "
                f"持仓{len(fw.positions)}只"
            )

    # 最终状态
    fw = BaseFramework(initial_cash=INITIAL_CASH, state_file=STATE_FILE,
                       market_regime_filter=None)
    fw.load_state()
    total_val = fw.cash + sum(
        pos.get('latest_price', pos['avg_cost']) * pos['qty']
        for pos in fw.positions.values()
    )
    logger.info(f"\n追数完成! 最终日期: {dates[-1]}")
    logger.info(f"  总交易: {fw.n_total} 笔")
    logger.info(f"  最终市值: {total_val/1e4:.2f}万 ({(total_val/INITIAL_CASH-1)*100:+.2f}%)")

    write_daily_signal(dates[-1], fw)
    return fw


def show_state():
    framework = BaseFramework(initial_cash=INITIAL_CASH, state_file=STATE_FILE,
                              market_regime_filter=None)
    if not framework.load_state():
        print(f"状态文件不存在: {STATE_FILE}")
        return

    total_val = framework.cash + sum(
        pos.get('latest_price', pos['avg_cost']) * pos['qty']
        for pos in framework.positions.values()
    )
    ret = (total_val / INITIAL_CASH - 1) * 100

    print(f"\n{'='*50}")
    print(f"  V8(NoRegime) 模拟盘状态")
    print(f"{'='*50}")
    print(f"  初始资金: {INITIAL_CASH:>14,.0f}")
    print(f"  当前现金: {framework.cash:>14,.0f}")
    print(f"  总价值:   {total_val:>14,.0f}  ({ret:+.2f}%)")
    print(f"  持仓数:   {len(framework.positions):>14}  只")
    print(f"  交易次数: {framework.n_total:>14}  笔")
    print(f"  胜率:     {framework.n_winning/max(framework.n_total,1)*100:>14.1f}%")
    print(f"{'='*50}")

    if framework.positions:
        print(f"\n  {'代码':<8} {'持仓量':>8} {'成本':>10} {'现价':>10} {'盈亏%':>8} {'持仓天数':>8}")
        print(f"  {'-'*56}")
        for sym, pos in sorted(framework.positions.items()):
            latest = pos.get('latest_price', pos['avg_cost'])
            pnl_pct = (latest / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] > 0 else 0
            print(f"  {sym:<8} {pos['qty']:>8} {pos['avg_cost']:>10.2f} "
                  f"{latest:>10.2f} {pnl_pct:>+7.2f}% {pos.get('days_held', 0):>8}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V8(NoRegime) 模拟盘")
    parser.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--catchup", action="store_true", help="追数（2026-01-01 → 昨天）")
    parser.add_argument("--start-date", default="2026-01-01", help="追数起始")
    parser.add_argument("--end-date", default=None, help="追数结束")
    parser.add_argument("--reset", action="store_true", help="重置状态")
    parser.add_argument("--show-state", action="store_true", help="查看状态")
    args = parser.parse_args()

    if args.show_state:
        show_state()
    elif args.catchup:
        run_catchup(args.start_date, args.end_date)
    else:
        target_date = args.date or date.today().strftime("%Y-%m-%d")
        logger.info(f"\n{'='*50}\n  V8(NoRegime)  日期: {target_date}\n{'='*50}")

        year = int(target_date[:4])
        year_df = load_data(year, None)
        year_df = patch_ic_conditions(year_df)
        ic_cache = build_ic_cache(year_df)

        fw = run_sim_for_date(target_date, year_df, ic_cache, reset=args.reset)
        write_daily_signal(target_date, fw)
        show_state()
