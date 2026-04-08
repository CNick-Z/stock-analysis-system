#!/usr/bin/env python3
"""
V3 B浪仓位逻辑正式回测
======================
基于预计算波浪信号的仓位逻辑回测

策略设置：
- V3原策略（无B浪降仓）：所有信号同仓位（50%）
- V3新策略（B浪降仓）：根据波浪结构调整仓位 30-70%

仓位规则（新策略）：
- bearish + W1<5浪 → B浪反弹：30-50%（根据W1子浪数）
- bullish + W1推动 + W2回调：50-70%（根据回撤幅度）
- neutral趋势：40-50%

回测设置：
- 时间范围：2024年全年 + 2025年全年
- 标的：10只代表性A股
- 对比指标：总收益率、最大回撤、夏普比率、胜率、B浪信号数量

Author: Byte
Date: 2026-04-08
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# ================================================================
# 项目路径
# ================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.shared import load_wavechan_cache, add_next_open
from utils.data_loader import load_strategy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ================================================================
# 测试股票（10只代表性股票）
# ================================================================
TEST_STOCKS = [
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

# ================================================================
# B浪仓位计算器
# ================================================================
class BWavePositionCalculator:
    """
    B浪仓位计算器

    根据波浪结构计算 position_size_ratio：
    - bearish + W1<5浪 → B浪反弹：30-50%（根据W1子浪数）
    - bullish + W1推动 + W2回调：50-70%（根据回撤幅度）
    - neutral趋势：40-50%

    注意：这里使用 wave_state 和 wave_trend 近似判断波浪结构
    """

    def __init__(self):
        pass

    def calculate_position_ratio(
        self,
        wave_state: str,
        wave_trend: str,
        wave_retracement: float,
        signal_type: str = None,
        symbol: str = None,
        full_df: pd.DataFrame = None,
        date: str = None
    ) -> Tuple[float, str, bool]:
        """
        计算仓位比例

        基于 signal_type 和 wave_state 判断仓位：
        - C_BUY 信号（W1/C浪反弹后回调入场）→ 30-50%：B浪反弹
        - W2_BUY 信号（W2回调入场）→ 50-70%：推动浪入场
        - W4_BUY 信号（W4调整入场）→ 40-50%：中性仓位

        Args:
            wave_state: 波浪状态（如 'w2_formed', 'w3_formed' 等）
            wave_trend: 波浪趋势（如 'long', 'down' 等）
            wave_retracement: 回撤比例（已废弃，使用signal_type代替）
            signal_type: 信号类型（C_BUY, W2_BUY, W4_BUY）

        Returns:
            (position_size_ratio, reason, is_b_wave_rebound)
        """
        # 根据信号类型判断仓位
        if signal_type == 'C_BUY':
            # C_BUY = W1反弹后的回调买入 = B浪反弹
            # 使用较低仓位：30-50%
            is_b_rebound = True
            # w1_formed 和 initial 表示W1刚开始或未完成，B浪特征明显
            if wave_state in ('w1_formed', 'initial'):
                ratio = 0.30
                reason = f"C_BUY+B浪反弹(W1未完成,仓位30%)"
            elif wave_state == 'w2_formed':
                ratio = 0.40
                reason = f"C_BUY+B浪反弹(W2入场,仓位40%)"
            else:
                ratio = 0.50
                reason = f"C_BUY+B浪反弹(W3入场,仓位50%)"
            return ratio, reason, is_b_rebound

        elif signal_type == 'W2_BUY':
            # W2_BUY = W2回调买入 = 推动浪入场
            # 使用较高仓位：50-70%
            is_b_rebound = False
            # w2_formed 表示W2调整完成，浅回撤
            if wave_state == 'w2_formed':
                ratio = 0.70
                reason = f"W2_BUY+推动浪(W2完成,仓位70%)"
            elif wave_state in ('w3_formed',):
                ratio = 0.60
                reason = f"W2_BUY+推动浪(W3中,仓位60%)"
            else:
                ratio = 0.50
                reason = f"W2_BUY+推动浪(其他,仓位50%)"
            return ratio, reason, is_b_rebound

        elif signal_type == 'W4_BUY':
            # W4_BUY = W4调整买入 = 中性仓位
            # 使用中等仓位：40-50%
            is_b_rebound = False
            if wave_state in ('w4_formed',):
                ratio = 0.50
                reason = f"W4_BUY+调整浪(W4完成,仓位50%)"
            elif wave_state == 'w4_in_progress':
                ratio = 0.40
                reason = f"W4_BUY+调整浪(W4进行中,仓位40%)"
            else:
                ratio = 0.50
                reason = f"W4_BUY+调整浪(其他,仓位50%)"
            return ratio, reason, is_b_rebound

        # 默认
        return 0.50, "默认仓位50%", False

    def _determine_large_trend(self, wave_state: str, wave_trend: str) -> str:
        """判断大级别趋势"""
        # bullish: 推动浪中（W2完成=W3进行中，或W4完成=W5进行中）
        if wave_state in ('w2_formed', 'w4_formed'):
            return 'bullish'

        # bearish: 调整浪中（W3完成=W4进行中，或W5完成=新一轮下跌）
        if wave_state in ('w3_formed', 'w4_in_progress', 'w5_formed'):
            if wave_trend == 'long':
                return 'neutral'
            return 'bearish'

        return 'neutral'

    def _estimate_w1_segments(self, wave_state: str, wave_retracement: float) -> int:
        """
        估算W1内部子浪数

        简化估算：
        - w2_formed, w3_formed, w4_formed, w5_formed: W1有5浪
        - w4_in_progress: W1有4浪
        - 其他: 使用回撤幅度辅助判断
        """
        if wave_state in ('w2_formed', 'w3_formed', 'w4_formed', 'w5_formed'):
            return 5
        elif wave_state == 'w4_in_progress':
            return 4
        else:
            if wave_retracement > 0.618:
                return 3
            elif wave_retracement > 0.382:
                return 4
            else:
                return 5


# ================================================================
# 数据加载
# ================================================================
def load_data(stocks, start_date, end_date):
    """加载数据"""
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

    return df


# ================================================================
# 回测引擎
# ================================================================
class BacktestEngine:
    """
    简化回测引擎
    """

    def __init__(
        self,
        initial_cash: float,
        commission_pct: float = 0.0003,
        slippage_pct: float = 0.0001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.20,
        max_hold_days: int = 20,
        max_positions: int = 5,
    ):
        self.initial_cash = initial_cash
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days
        self.max_positions = max_positions

        self.reset()

    def reset(self):
        """重置状态"""
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.market_snapshots = []
        self.n_total = 0
        self.n_winning = 0

    def run_backtest(
        self,
        df: pd.DataFrame,
        use_b_wave_position: bool = False,
        position_calculator: BWavePositionCalculator = None,
    ) -> Dict:
        """
        运行回测
        """
        self.reset()
        b_wave_calculator = position_calculator or BWavePositionCalculator()

        df = df.sort_values('date').reset_index(drop=True)
        dates = sorted(df['date'].unique())

        for i, date in enumerate(dates):
            day_df = df[df['date'] == date].copy()
            if day_df.empty:
                continue

            total_value = self.cash + sum(
                p['latest_price'] * p['qty']
                for p in self.positions.values()
            )

            # 更新持仓状态
            for sym, pos in list(self.positions.items()):
                row = day_df[day_df['symbol'] == sym]
                if row.empty:
                    continue
                r = row.iloc[0]
                pos['latest_price'] = r.get('close', pos['avg_cost'])
                pos['days_held'] = pos.get('days_held', 0) + 1

            # 出场判断
            for sym in list(self.positions.keys()):
                row = day_df[day_df['symbol'] == sym]
                if row.empty:
                    continue
                r = row.iloc[0]
                pos = self.positions[sym]

                should_sell, reason = self._should_sell(r, pos, date)
                if not should_sell:
                    continue

                if r.get('limit_up') or r.get('limit_down'):
                    continue

                exec_price = r.get('next_open')
                if pd.isna(exec_price) or exec_price <= 0:
                    exec_price = r.get('close', pos['avg_cost'])

                exec_price = exec_price * (1 - self.slippage_pct)

                pnl = (exec_price - pos['avg_cost']) * pos['qty']
                pnl_pct = (exec_price - pos['avg_cost']) / pos['avg_cost'] * 100
                sell_value = exec_price * pos['qty'] * (1 - self.commission_pct)

                self.cash += sell_value
                self.trades.append({
                    'date': date,
                    'symbol': sym,
                    'action': 'sell',
                    'price': exec_price,
                    'qty': pos['qty'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'position_size_ratio': pos.get('position_size_ratio', 1.0),
                    'is_b_wave_rebound': pos.get('is_b_wave_rebound', False),
                })
                self.n_total += 1
                if pnl > 0:
                    self.n_winning += 1

                del self.positions[sym]

            # 入场判断
            if len(self.positions) < self.max_positions:
                slots = self.max_positions - len(self.positions)

                candidates = day_df[
                    (day_df.get('has_signal', False) == True) &
                    (day_df['signal_type'].isin(['W2_BUY', 'W4_BUY', 'C_BUY'])) &
                    (day_df['total_score'] >= 15)
                ].copy()

                if not candidates.empty:
                    candidates = candidates.sort_values('total_score', ascending=False)

                    fill_count = 0
                    for _, row in candidates.iterrows():
                        if fill_count >= slots:
                            break
                        sym = row.get('symbol', '')
                        if sym in self.positions:
                            continue

                        if row.get('limit_up') or row.get('limit_down'):
                            continue

                        exec_price = row.get('next_open')
                        if pd.isna(exec_price):
                            exec_price = row.get('open', row.get('close', 0))

                        if exec_price <= 0 or self.cash < exec_price * 100:
                            continue

                        if use_b_wave_position:
                            position_ratio, reason, is_b_rebound = b_wave_calculator.calculate_position_ratio(
                                wave_state=str(row.get('wave_state', '')),
                                wave_trend=str(row.get('wave_trend', '')),
                                wave_retracement=float(row.get('wave_retracement', 0.382)),
                                signal_type=str(row.get('signal_type', '')),
                                symbol=sym,
                                full_df=df,
                                date=date,
                            )
                        else:
                            position_ratio = 0.50
                            is_b_rebound = False
                            reason = "固定仓位50%"

                        per_stock_cash = min(
                            self.cash * 0.5 / (fill_count + 1),
                            self.cash * 0.2
                        )
                        actual_cash = per_stock_cash * position_ratio
                        buy_qty = int(actual_cash / exec_price)
                        buy_qty = (buy_qty // 100) * 100
                        if buy_qty < 100:
                            continue

                        cost = buy_qty * exec_price * (1 + self.commission_pct)
                        if cost > self.cash:
                            continue

                        self.cash -= cost
                        self.positions[sym] = {
                            'qty': buy_qty,
                            'avg_cost': exec_price,
                            'entry_date': date,
                            'entry_price': exec_price,
                            'latest_price': exec_price,
                            'days_held': 0,
                            'position_size_ratio': position_ratio,
                            'is_b_wave_rebound': is_b_rebound,
                            'entry_signal': row.get('signal_type', ''),
                        }
                        self.trades.append({
                            'date': date,
                            'symbol': sym,
                            'action': 'buy',
                            'price': exec_price,
                            'qty': buy_qty,
                            'reason': row.get('signal_type', ''),
                            'total_score': row.get('total_score', 0),
                            'position_size_ratio': position_ratio,
                            'is_b_wave_rebound': is_b_rebound,
                        })
                        fill_count += 1

            total_value = self.cash + sum(
                p['latest_price'] * p['qty']
                for p in self.positions.values()
            )
            self.market_snapshots.append({
                'date': date,
                'total_value': total_value,
                'n_positions': len(self.positions),
            })

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  {date} ({i + 1}/{len(dates)}): "
                    f"持仓{len(self.positions)}只, 总值{total_value / 10000:.1f}万"
                )

        return self._summarize()

    def _should_sell(self, row: pd.Series, pos: dict, date: str) -> Tuple[bool, str]:
        """判断是否出场"""
        entry_price = pos.get('avg_cost', 0)
        if entry_price <= 0:
            return False, "INVALID_POSITION"

        next_open = row.get('next_open')
        if pd.isna(next_open) or next_open <= 0:
            next_open = row.get('close', entry_price)

        # 止损
        stop_price = entry_price * (1 - self.stop_loss_pct)
        if next_open < stop_price:
            return True, f"STOP_LOSS @{stop_price:.2f}"

        # 止盈
        profit_price = entry_price * (1 + self.take_profit_pct)
        if next_open >= profit_price:
            return True, f"TAKE_PROFIT @{profit_price:.2f}"

        # 时间止损
        days_held = pos.get('days_held', 0)
        if days_held >= self.max_hold_days:
            return True, f"TIME_EXIT({days_held}天)"

        # 波浪出场信号
        signal_type = row.get('signal_type', '')
        if signal_type in ('W5_SELL', 'SELL'):
            return True, f"WAVE_SIGNAL({signal_type})"

        # 趋势转空出场
        wave_trend = row.get('wave_trend', '')
        if wave_trend == 'down' and days_held >= 3:
            return True, f"TREND_DOWN({wave_trend})"

        return False, ""

    def _summarize(self) -> Dict:
        """汇总结果"""
        closed = [t for t in self.trades if t['action'] == 'sell']
        total_pnl = sum(t['pnl'] for t in closed)
        win_rate = self.n_winning / max(self.n_total, 1) * 100

        final_value = self.cash + sum(
            p['latest_price'] * p['qty']
            for p in self.positions.values()
        )
        total_return = (final_value / self.initial_cash - 1) * 100

        values = [s['total_value'] for s in self.market_snapshots]
        n_days = len(values)
        n_years = n_days / 244
        annual_return = (final_value / self.initial_cash) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 最大回撤
        peak = self.initial_cash
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        # 夏普比率
        if len(values) > 1:
            daily_returns = pd.Series(values).pct_change().dropna()
            sharpe = daily_returns.mean() / daily_returns.std() * (244 ** 0.5) if daily_returns.std() > 0 else 0.0
        else:
            sharpe = 0.0

        # 止损统计
        stop_loss_trades = [t for t in closed if '止损' in t.get('reason', '') or 'STOP_LOSS' in t.get('reason', '')]
        n_stop_loss = len(stop_loss_trades)
        stop_loss_rate = n_stop_loss / max(len(closed), 1) * 100

        # B浪交易统计
        b_wave_trades = [t for t in closed if t.get('is_b_wave_rebound', False)]
        n_b_wave = len(b_wave_trades)
        b_wave_pnl = sum(t['pnl'] for t in b_wave_trades)
        b_wave_win_rate = sum(1 for t in b_wave_trades if t['pnl'] > 0) / max(n_b_wave, 1) * 100 if n_b_wave > 0 else 0

        # W2回调交易统计
        w2_correction_trades = [t for t in closed if not t.get('is_b_wave_rebound', False)]
        n_w2 = len(w2_correction_trades)
        w2_pnl = sum(t['pnl'] for t in w2_correction_trades)
        w2_win_rate = sum(1 for t in w2_correction_trades if t['pnl'] > 0) / max(n_w2, 1) * 100 if n_w2 > 0 else 0

        # 出场原因分布
        reason_counter = Counter(t.get('reason', 'unknown') for t in closed)

        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return * 100,
            'max_drawdown_pct': max_dd,
            'sharpe': sharpe,
            'n_trades': self.n_total,
            'n_buy': len([t for t in self.trades if t['action'] == 'buy']),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'n_positions_current': len(self.positions),
            'n_stop_loss': n_stop_loss,
            'stop_loss_rate': stop_loss_rate,
            'n_b_wave_trades': n_b_wave,
            'b_wave_pnl': b_wave_pnl,
            'b_wave_win_rate': b_wave_win_rate,
            'n_w2_correction_trades': n_w2,
            'w2_correction_pnl': w2_pnl,
            'w2_correction_win_rate': w2_win_rate,
            'reason_counter': dict(reason_counter),
            'closed_trades': closed,
            'all_trades': self.trades,
            'market_snapshots': self.market_snapshots,
        }


# ================================================================
# 打印对比
# ================================================================
def print_comparison(original, new_strategy):
    """打印策略对比结果"""
    print(f"\n\n{'=' * 80}")
    print(f"  V3 B浪仓位逻辑正式回测  |  区间: {START_DATE} ~ {END_DATE}")
    print(f"  股票: {', '.join(TEST_STOCKS)}")
    print(f"{'=' * 80}")

    for r, label in [
        (original, 'V3原策略（无B浪降仓，所有信号同仓位50%）'),
        (new_strategy, 'V3新策略（B浪降仓，根据波浪结构调整仓位30-70%）')
    ]:
        print(f"\n{'─' * 80}")
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

        print(f"\n  ── B浪信号统计 ──")
        print(f"  B浪交易次数:   {r['n_b_wave_trades']:>14}  次")
        print(f"  B浪累计盈亏:   {r['b_wave_pnl']:>14,.0f}")
        print(f"  B浪胜率:       {r['b_wave_win_rate']:>14.1f}%")
        print(f"  W2回调交易次数: {r['n_w2_correction_trades']:>14}  次")
        print(f"  W2回调累计盈亏: {r['w2_correction_pnl']:>14,.0f}")
        print(f"  W2回调胜率:    {r['w2_correction_win_rate']:>14.1f}%")

        if r['reason_counter']:
            print(f"\n  出场原因分布:")
            for reason, count in sorted(r['reason_counter'].items(), key=lambda x: -x[1]):
                print(f"    {(reason or 'unknown'):<40} {count:>4}  笔")

    # 横向对比表
    print(f"\n{'─' * 80}")
    print(f"  【横向对比摘要】")
    hdr = f"  {'版本':<42} {'最终价值':>12} {'总收益':>9} {'年化':>8} {'最大回撤':>9} {'交易数':>7} {'止损':>6} {'止损率':>7} {'胜率':>7}"
    print(hdr)
    print(f"  {'-' * 78}")
    for r, label in [
        (original, 'V3原策略（无B浪降仓）'),
        (new_strategy, 'V3新策略（B浪降仓）')
    ]:
        print(
            f"  {label:<42} "
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
    dw = new_strategy['total_return_pct'] - original['total_return_pct']
    ddd = new_strategy['max_drawdown_pct'] - original['max_drawdown_pct']
    dn_sl = new_strategy['n_stop_loss'] - original['n_stop_loss']
    dsr = new_strategy['stop_loss_rate'] - original['stop_loss_rate']
    dwr = new_strategy['win_rate'] - original['win_rate']

    print(f"\n  【差异分析：新策略 - 原策略】")
    print(f"  总收益差异:      {dw:>+10.2f}%  {'✅ B浪降仓提升收益' if dw > 0 else '⚠️ B浪降仓降低收益' if dw < 0 else '= 无差异'}")
    print(f"  最大回撤差异:    {ddd:>+10.2f}%  {'✅ B浪降仓减少回撤' if ddd < 0 else '⚠️ B浪降仓增加回撤' if ddd > 0 else '= 无差异'}")
    print(f"  夏普比率差异:    {new_strategy['sharpe'] - original['sharpe']:>+10.2f}")
    print(f"  止损次数差异:    {dn_sl:>+10} 次  {'✅ B浪降仓减少止损' if dn_sl < 0 else '⚠️ B浪降仓增加止损' if dn_sl > 0 else '= 无差异'}")
    print(f"  止损率差异:      {dsr:>+10.1f}%")
    print(f"  胜率差异:        {dwr:>+10.1f}%  {'✅ B浪降仓提升胜率' if dwr > 0 else '⚠️ B浪降仓降低胜率' if dwr < 0 else '= 无差异'}")

    print(f"\n  【B浪信号 vs W2回调信号】")
    print(f"  原策略B浪交易次数:   {original['n_b_wave_trades']:>10} 次")
    print(f"  新策略B浪交易次数:   {new_strategy['n_b_wave_trades']:>10} 次")
    print(f"  原策略W2回调次数:    {original['n_w2_correction_trades']:>10} 次")
    print(f"  新策略W2回调次数:    {new_strategy['n_w2_correction_trades']:>10} 次")

    print(f"\n{'=' * 80}")

    return {
        'original': original,
        'new_strategy': new_strategy,
    }


# ================================================================
# 保存结果
# ================================================================
def save_results(comparison, output_dir=None):
    """保存回测结果"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "backtestresult"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        'timestamp': timestamp,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'test_stocks': TEST_STOCKS,
        'initial_cash': INITIAL_CASH,
        'original_strategy': {
            'label': 'V3原策略（无B浪降仓）',
            'final_value': comparison['original']['final_value'],
            'total_return_pct': comparison['original']['total_return_pct'],
            'annual_return_pct': comparison['original']['annual_return_pct'],
            'max_drawdown_pct': comparison['original']['max_drawdown_pct'],
            'sharpe': comparison['original']['sharpe'],
            'n_trades': comparison['original']['n_trades'],
            'win_rate': comparison['original']['win_rate'],
            'n_b_wave_trades': comparison['original']['n_b_wave_trades'],
            'b_wave_pnl': comparison['original']['b_wave_pnl'],
            'b_wave_win_rate': comparison['original']['b_wave_win_rate'],
        },
        'new_strategy': {
            'label': 'V3新策略（B浪降仓）',
            'final_value': comparison['new_strategy']['final_value'],
            'total_return_pct': comparison['new_strategy']['total_return_pct'],
            'annual_return_pct': comparison['new_strategy']['annual_return_pct'],
            'max_drawdown_pct': comparison['new_strategy']['max_drawdown_pct'],
            'sharpe': comparison['new_strategy']['sharpe'],
            'n_trades': comparison['new_strategy']['n_trades'],
            'win_rate': comparison['new_strategy']['win_rate'],
            'n_b_wave_trades': comparison['new_strategy']['n_b_wave_trades'],
            'b_wave_pnl': comparison['new_strategy']['b_wave_pnl'],
            'b_wave_win_rate': comparison['new_strategy']['b_wave_win_rate'],
        },
        'difference': {
            'total_return_diff': comparison['new_strategy']['total_return_pct'] - comparison['original']['total_return_pct'],
            'max_drawdown_diff': comparison['new_strategy']['max_drawdown_pct'] - comparison['original']['max_drawdown_pct'],
            'sharpe_diff': comparison['new_strategy']['sharpe'] - comparison['original']['sharpe'],
            'win_rate_diff': comparison['new_strategy']['win_rate'] - comparison['original']['win_rate'],
        }
    }

    json_path = output_dir / f"bwave_position_backtest_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {json_path}")

    # 保存交易记录CSV
    all_trades = comparison['new_strategy']['all_trades']
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        csv_path = output_dir / f"bwave_trades_{timestamp}.csv"
        trades_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"交易记录已保存: {csv_path}")

    return json_path, csv_path


# ================================================================
# 主函数
# ================================================================
def main():
    print(f"\n{'=' * 80}")
    print(f"  V3 B浪仓位逻辑正式回测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 80}")

    # 加载数据
    logger.info("加载数据...")
    df = load_data(TEST_STOCKS, START_DATE, END_DATE)
    if df.empty:
        logger.error("数据为空")
        return None

    logger.info(f"\n数据概览: {len(df):,} 行, {df['symbol'].nunique()} 只股票")
    logger.info(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    # 创建回测引擎
    engine = BacktestEngine(
        initial_cash=INITIAL_CASH,
        commission_pct=0.0003,
        slippage_pct=0.0001,
        stop_loss_pct=0.05,
        take_profit_pct=0.20,
        max_hold_days=20,
        max_positions=5,
    )

    # ── V3原策略（无B浪降仓）─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  开始回测：V3原策略（无B浪降仓）")
    logger.info("=" * 60)

    result_original = engine.run_backtest(df, use_b_wave_position=False)

    # ── V3新策略（B浪降仓）─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  开始回测：V3新策略（B浪降仓）")
    logger.info("=" * 60)

    result_new = engine.run_backtest(df, use_b_wave_position=True, position_calculator=BWavePositionCalculator())

    # 打印对比
    comparison = print_comparison(result_original, result_new)

    # 保存结果
    save_results(comparison)

    return comparison


if __name__ == "__main__":
    main()