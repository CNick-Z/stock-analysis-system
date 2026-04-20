#!/usr/bin/env python3
"""
WaveChan V3 周线过滤对比回测
=============================
对比有/无周线过滤的效果（10只代表性股票，2024-01-01 ~ 2025-12-31）

股票列表（不同行业、不同市值）：
  000001 平安银行（银行，大盘）
  000002 万科A（房地产，大盘）
  600519 贵州茅台（酿酒，大盘）
  600036 招商银行（银行，大盘）
  000858 五粮液（酿酒，大盘）
  300750 宁德时代（电池，大盘）
  688981 中芯国际（半导体，大盘）
  002475 立讯精密（消费电子，中盘）
  300059 东方财富（互联网，中盘）
  600585 海螺水泥（建材，中盘）
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.base_framework import BaseFramework
# 保留 v3_weekly_filter_compare 自己的佣金逻辑（万3固定，用于横向对比）
_CORRECT_COMMISSION = 0.0003  # 原框架默认值，对比分析用
from simulator.shared import load_wavechan_cache, add_next_open
from utils.data_loader import load_strategy_data
from strategies.wavechan.v3_l2_cache.wavechan_strategy import WaveChanStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 测试股票 ─────────────────────────────────────────────────
TEST_STOCKS = [
    # 10只代表性股票（不同行业、不同市值）
    "000001",  # 平安银行（银行，大盘）
    "000002",  # 万科A（房地产，大盘）
    "600519",  # 贵州茅台（酿酒，大盘）
    "600036",  # 招商银行（银行，大盘）
    "000858",  # 五粮液（酿酒，大盘）
    "300750",  # 宁德时代（电池，大盘）
    "688981",  # 中芯国际（半导体，大盘）
    "002475",  # 立讯精密（消费电子，中盘）
    "300059",  # 东方财富（互联网，中盘）
    "600585",  # 海螺水泥（建材，中盘）
]

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
INITIAL_CASH = 1_000_000.0


# ─────────────────────────────────────────────────────────────
# 无周线过滤策略（仅覆盖 filter_buy，跳过 step 2 周线判断）
# ─────────────────────────────────────────────────────────────
class WaveChanStrategyNoWeekly(WaveChanStrategy):
    """
    WaveChan V3 无周线过滤版本
    等同于把 filter_buy 的 step 2 完全禁用
    """

    def filter_buy(self, daily_df: pd.DataFrame, date: str = None) -> pd.DataFrame:
        """只做日线基础过滤，不做周线方向判断"""
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()

        _false = pd.Series(False, index=df.index)
        _zero = pd.Series(0, index=df.index)
        _empty = pd.Series('', index=df.columns)
        has_signal = df['has_signal'] if 'has_signal' in df.columns else _false
        total_score = df['total_score'] if 'total_score' in df.columns else _zero
        wave_trend = df['wave_trend'] if 'wave_trend' in df.columns else _empty

        mask = (
            has_signal.eq(True) &
            total_score.ge(self.threshold) &
            wave_trend.isin(['long', 'neutral', ''])
        )

        buy_signals = {'W2_BUY', 'W4_BUY', 'C_BUY', 'W4_BUY_ALERT', 'W4_BUY_CONFIRMED'}
        if 'signal_type' in df.columns:
            mask &= df['signal_type'].isin(buy_signals)
        if 'signal_status' in df.columns:
            mask &= df['signal_status'].eq('confirmed')

        candidates = df[mask].copy()
        return candidates


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
def load_data(stocks, start_date, end_date):
    years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))

    df = load_strategy_data(years=years, add_money_flow=True)
    df = df[df['symbol'].isin(stocks)].copy()
    logger.info(f"原始数据: {len(df):,} 行, {df['symbol'].nunique()} 只股票")

    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    logger.info(f"日期过滤后: {len(df):,} 行")

    df = add_next_open(df)

    wave_df = load_wavechan_cache(years)
    if not wave_df.empty:
        wave_df = wave_df[wave_df['symbol'].isin(stocks)].copy()
        wave_cols = [c for c in wave_df.columns if c not in ('date', 'symbol')]
        for col in wave_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        wave_df = wave_df.copy()
        wave_df['date'] = wave_df['date'].astype(str)
        df = df.merge(wave_df, on=['date', 'symbol'], how='left')
        if 'has_signal' in df.columns:
            df['has_signal'] = df['has_signal'].fillna(False)
            n_signal = df['has_signal'].sum()
            n_sym_sig = df[df['has_signal']]['symbol'].nunique()
            logger.info(f"波浪信号合并: {len(df):,} 行, has_signal=True: {n_signal:,}, 覆盖股票: {n_sym_sig}")
        else:
            logger.warning("波浪信号列不存在！")
    else:
        logger.warning("波浪缓存为空！")

    return df


# ─────────────────────────────────────────────────────────────
# 回测引擎（使用 BaseFramework）
# ─────────────────────────────────────────────────────────────
def run_backtest(strategy, df, initial_cash, label):
    """
    使用 BaseFramework 逐年回测
    """
    framework = BaseFramework(
        initial_cash=initial_cash,
        state_file=f"/tmp/bkt_{label.replace(' ', '_')}.json",
        market_regime_filter=None,
    )
    framework._strategy = strategy
    framework.reset()

    all_years = sorted(df['date'].str[:4].unique())

    for year in all_years:
        y_df = df[df['date'].str.startswith(year)].copy()
        if y_df.empty:
            continue

        logger.info(f"\n  ── {label} | {year} 年 ──")
        dates = sorted(y_df['date'].unique())
        logger.info(f"  {year} 年: {len(dates)} 个交易日, {len(y_df):,} 行数据")

        # prepare 策略（传入当年完整数据，策略内部预计算周线等）
        try:
            strategy.prepare(dates=dates, loader=y_df)
        except Exception as e:
            logger.warning(f"prepare 异常 ({label} {year}): {e}")

        # 逐年运行（用 BaseFramework._on_day 逻辑）
        y_df = y_df.sort_values('date').reset_index(drop=True)
        y_dates = sorted(y_df['date'].unique())

        for i, date in enumerate(y_dates):
            day_df = y_df[y_df['date'] == date].copy()
            if day_df.empty:
                continue

            market = {
                'date': date,
                'cash': framework.cash,
                'total_value': framework.cash + sum(
                    p.get('latest_price', p['avg_cost']) * p['qty']
                    for p in framework.positions.values()
                ),
            }

            # 更新持仓状态 & on_tick
            for sym, pos in list(framework.positions.items()):
                row = day_df[day_df['symbol'] == sym]
                if row.empty:
                    continue
                r = row.iloc[0]
                pos['latest_price'] = r.get('close', pos.get('avg_cost', 0))
                pos['days_held'] = pos.get('days_held', 0) + 1

                try:
                    strategy.on_tick(r, pos, market)
                except TypeError:
                    try:
                        strategy.on_tick(r, pos)
                    except Exception:
                        pass

            # 出场
            for sym in list(framework.positions.keys()):
                row = day_df[day_df['symbol'] == sym]
                if row.empty:
                    continue
                r = row.iloc[0]
                pos = framework.positions[sym]

                should_sell, reason = strategy.should_sell(r, pos, market)

                if not should_sell:
                    continue

                # 涨跌停检查
                if r.get('limit_up') or r.get('limit_down'):
                    continue

                exec_price = r.get('next_open')
                if pd.isna(exec_price) or exec_price <= 0:
                    exec_price = r.get('close', pos['avg_cost'])

                exec_price = exec_price * (1 - framework.slippage_pct)
                pnl = (exec_price - pos['avg_cost']) * pos['qty']
                pnl_pct = (exec_price - pos['avg_cost']) / pos['avg_cost'] * 100
                sell_value = exec_price * pos['qty'] * (1 - _CORRECT_COMMISSION)

                framework.cash += sell_value
                framework.trades.append({
                    'date': date,
                    'symbol': sym,
                    'action': 'sell',
                    'price': exec_price,
                    'qty': pos['qty'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                })
                framework.n_total += 1
                if pnl > 0:
                    framework.n_winning += 1
                logger.info(f"  卖出 {sym} @{exec_price:.2f} ({reason}, {pnl_pct:+.1f}%)")
                del framework.positions[sym]

            # 入场
            if len(framework.positions) < strategy.max_positions:
                avail_cash = framework.cash
                slots = strategy.max_positions - len(framework.positions)

                candidates = strategy.filter_buy(y_df, market.get('date'))
                if candidates.empty:
                    pass
                else:
                    try:
                        scored = strategy.score(candidates)
                    except Exception:
                        scored = candidates

                    if not scored.empty:
                        fill_count = 0
                        to_buy = []

                        for _, row in scored.iterrows():
                            if fill_count >= slots:
                                break
                            sym = row.get('symbol', '')
                            if sym in framework.positions:
                                continue

                            day_row = day_df[day_df['symbol'] == sym]
                            if day_row.empty:
                                continue
                            r = day_row.iloc[0]

                            if r.get('limit_up') or r.get('limit_down'):
                                continue

                            exec_price = r.get('next_open')
                            if pd.isna(exec_price):
                                exec_price = r.get('open', r.get('close', 0))

                            if exec_price <= 0 or avail_cash < exec_price * 100:
                                continue

                            to_buy.append((sym, row, r, exec_price))
                            fill_count += 1

                        if fill_count > 0:
                            per_stock_cash = min(
                                avail_cash / fill_count,
                                framework.cash * strategy.position_size
                            )

                            for sym, row, r, exec_price in to_buy:
                                buy_qty = int(per_stock_cash / exec_price)
                                buy_qty = (buy_qty // 100) * 100
                                if buy_qty < 100:
                                    continue

                                cost = buy_qty * exec_price * (1 + _CORRECT_COMMISSION)
                                if cost > framework.cash:
                                    continue

                                framework.cash -= cost
                                framework.positions[sym] = {
                                    'qty': buy_qty,
                                    'avg_cost': exec_price,
                                    'entry_date': date,
                                    'entry_price': exec_price,
                                    'latest_price': exec_price,
                                    'days_held': 0,
                                    'consecutive_bad_days': 0,
                                    'extra': {},
                                }
                                framework.trades.append({
                                    'date': date,
                                    'symbol': sym,
                                    'action': 'buy',
                                    'price': exec_price,
                                    'qty': buy_qty,
                                    'reason': row.get('signal_type', ''),
                                    'total_score': row.get('total_score', 0),
                                })
                                logger.info(
                                    f"  买入 {sym} @{exec_price:.2f} × {buy_qty}股 "
                                    f"(评分:{row.get('score', row.get('total_score', 0)):.0f})"
                                )

            # 更新市值快照
            total_value = framework.cash + sum(
                p.get('latest_price', p['avg_cost']) * p['qty']
                for p in framework.positions.values()
            )
            framework.market_snapshots.append({
                'date': date,
                'total_value': total_value,
                'n_positions': len(framework.positions),
            })

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  {date} ({i + 1}/{len(y_dates)}): "
                    f"持仓{len(framework.positions)}只, 总值{total_value / 10000:.1f}万"
                )

    # ── 汇总 ───────────────────────────────────────────────────
    closed = [t for t in framework.trades if t['action'] == 'sell']
    total_pnl = sum(t['pnl'] for t in closed)
    win_rate = framework.n_winning / max(framework.n_total, 1) * 100

    final_value = framework.cash + sum(
        p.get('latest_price', p['avg_cost']) * p['qty']
        for p in framework.positions.values()
    )
    total_return = (final_value / initial_cash - 1) * 100

    values = [s['total_value'] for s in framework.market_snapshots]
    n_days = len(values)
    n_years = n_days / 244
    annual_return = (final_value / initial_cash) ** (1 / n_years) - 1 if n_years > 0 else 0

    peak = initial_cash
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    if len(values) > 1:
        daily_returns = pd.Series(values).pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * (244 ** 0.5) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # 止损统计
    stop_loss_keywords = ['止损', 'stop_loss', 'STOP_LOSS', '止损出局']
    stop_loss_trades = [
        t for t in closed
        if any(kw in t.get('reason', '') for kw in stop_loss_keywords)
    ]
    n_stop_loss = len(stop_loss_trades)
    stop_loss_rate = n_stop_loss / max(len(closed), 1) * 100

    reason_counter = Counter(t.get('reason', 'unknown') for t in closed)

    result = {
        'label': label,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return_pct': total_return,
        'annual_return_pct': annual_return * 100,
        'max_drawdown_pct': max_dd,
        'sharpe': sharpe,
        'n_trades': framework.n_total,
        'n_buy': len([t for t in framework.trades if t['action'] == 'buy']),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'n_positions_current': len(framework.positions),
        'n_stop_loss': n_stop_loss,
        'stop_loss_rate': stop_loss_rate,
        'reason_counter': dict(reason_counter),
        'closed_trades': closed,
        'all_trades': framework.trades,
        'market_snapshots': framework.market_snapshots,
    }

    return result


# ─────────────────────────────────────────────────────────────
# 打印对比
# ─────────────────────────────────────────────────────────────
def print_comparison(with_w, without_w):
    print(f"\n\n{'=' * 74}")
    print(f"  WaveChan V3 周线过滤对比回测  |  区间: {START_DATE} ~ {END_DATE}")
    print(f"  股票: {', '.join(TEST_STOCKS)}")
    print(f"{'=' * 74}")

    for r, label in [(with_w, '有周线过滤（V3原版）'), (without_w, '无周线过滤（对比组）')]:
        print(f"\n{'─' * 74}")
        print(f"  【{label}】")
        print(f"  初始资金:      {r['initial_cash']:>14,.0f}")
        print(f"  最终价值:      {r['final_value']:>14,.0f}  ({r['total_return_pct']:+.2f}%)")
        print(f"  年化收益:      {r['annual_return_pct']:>14.2f}%")
        print(f"  最大回撤:      {r['max_drawdown_pct']:>14.2f}%")
        print(f"  夏普比率:      {r['sharpe']:>14.2f}")
        print(f"  总卖出交易:    {r['n_trades']:>14}  笔")
        print(f"  买入次数:      {r['n_buy']:>14}  次")
        print(f"  胜率:          {r['win_rate']:>14.1f}%")
        print(f"  止损次数:      {r['n_stop_loss']:>14}  次  ({r['stop_loss_rate']:.1f}%)")
        print(f"  累计盈亏:      {r['total_pnl']:>14,.0f}")
        print(f"  当前持仓:      {r['n_positions_current']:>14}  只")

        if r['reason_counter']:
            print(f"\n  出场原因分布:")
            for reason, count in sorted(r['reason_counter'].items(), key=lambda x: -x[1]):
                print(f"    {(reason or 'unknown'):<32} {count:>4}  笔")

    # 横向对比表
    print(f"\n{'─' * 74}")
    print(f"  【横向对比摘要】")
    hdr = f"  {'版本':<22} {'最终价值':>12} {'总收益':>9} {'年化':>8} {'最大回撤':>9} {'交易数':>7} {'止损':>6} {'止损率':>7} {'胜率':>7}"
    print(hdr)
    print(f"  {'-' * 72}")
    for r, label in [(with_w, '有周线过滤（V3原版）'), (without_w, '无周线过滤（对比组）')]:
        print(
            f"  {label:<22} "
            f"{r['final_value']:>12,.0f} "
            f"{r['total_return_pct']:>+8.2f}% "
            f"{r['annual_return_pct']:>+7.2f}% "
            f"{r['max_drawdown_pct']:>8.2f}% "
            f"{r['n_trades']:>7} "
            f"{r['n_stop_loss']:>6} "
            f"{r['stop_loss_rate']:>6.1f}% "
            f"{r['win_rate']:>6.1f}%"
        )

    # 差异分析
    dw = with_w['total_return_pct'] - without_w['total_return_pct']
    dn_sl = with_w['n_stop_loss'] - without_w['n_stop_loss']
    dn_tr = with_w['n_trades'] - without_w['n_trades']
    dsr = with_w['stop_loss_rate'] - without_w['stop_loss_rate']
    dwr = with_w['win_rate'] - without_w['win_rate']

    print(f"\n  【差异分析：有周线 - 无周线】")
    print(f"  总收益差异:    {dw:>+10.2f}%")
    print(f"  止损次数差异:  {dn_sl:>+10} 次  {'✅ 周线过滤减少止损' if dn_sl < 0 else '⚠️ 周线过滤增加止损' if dn_sl > 0 else '= 无差异'}")
    print(f"  止损率差异:    {dsr:>+10.1f}%")
    print(f"  总交易次数差异: {dn_tr:>+10} 笔")
    print(f"  胜率差异:      {dwr:>+10.1f}%")
    print(f"\n{'=' * 74}")

    return {
        'with_weekly': with_w,
        'without_weekly': without_w,
    }


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{'=' * 74}")
    print(f"  WaveChan V3 周线过滤对比回测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 74}")

    # 加载数据
    logger.info("加载数据...")
    df = load_data(TEST_STOCKS, START_DATE, END_DATE)
    if df.empty:
        logger.error("数据为空")
        return None

    logger.info(f"\n数据概览: {len(df):,} 行, {df['symbol'].nunique()} 只股票")
    logger.info(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    # ── 有周线过滤 ─────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("  开始回测：有周线过滤")
    logger.info("=" * 50)

    strat_with = WaveChanStrategy()
    result_with = run_backtest(strat_with, df, INITIAL_CASH, "有周线过滤（V3原版）")

    # ── 无周线过滤 ─────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("  开始回测：无周线过滤")
    logger.info("=" * 50)

    strat_without = WaveChanStrategyNoWeekly()
    result_without = run_backtest(strat_without, df, INITIAL_CASH, "无周线过滤（对比组）")

    # 打印对比
    comparison = print_comparison(result_with, result_without)

    return comparison


if __name__ == "__main__":
    main()
