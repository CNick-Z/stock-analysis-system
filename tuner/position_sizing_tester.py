#!/usr/bin/env python3
"""
连续止损仓位管理回测验证
==============================
Trader 方案：连续N次止损触发降仓，连续Z次盈利恢复满仓

规则：
  - 默认仓位：100%
  - 连续3次止损 → 降仓50%
  - 连续5次止损 → 降仓25%
  - 连续2次盈利 → 恢复+25%
  - 连续4次盈利 → 恢复100%

对比三个市场：
  - 熊市：2010-2011
  - 牛市：2014-2015
  - 震荡：2021-2022
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from simulator.base_framework import BaseFramework
from strategies.score.v8.strategy import ScoreV8Strategy


class SequentialSizingFramework(BaseFramework):
    """叠加连续止损降仓规则的回测框架"""

    def __init__(
        self,
        initial_cash=1_000_000,
        n1=3, n2=5,
        x=0.50, y=0.25,  # 降仓比例
        z1=2, z2=4,       # 恢复触发
        **kwargs
    ):
        super().__init__(initial_cash=initial_cash, **kwargs)
        self.n1 = n1
        self.n2 = n2
        self.x = x
        self.y = y
        self.z1 = z1
        self.z2 = z2

        # 状态
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self._base_position_size = self.position_size  # 原始仓位比例

    def _effective_position_size(self):
        """根据连续盈亏计算当前仓位"""
        if self.consecutive_losses >= self.n2:
            return self.y  # 25%
        elif self.consecutive_losses >= self.n1:
            return self.x  # 50%
        return 1.0  # 100%

    def reset(self):
        super().reset()
        self.consecutive_losses = 0
        self.consecutive_wins = 0

    def _record_trade_result(self, pnl_pct: float):
        """每笔交易结束后记录结果，用于更新连续计数器"""
        if pnl_pct > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def _on_close(self, sym: str, pos: dict, exec_price: float,
                  pnl: float, pnl_pct: float, reason: str):
        """平仓时更新连续盈亏计数"""
        super()._on_close(sym, pos, exec_price, pnl, pnl_pct, reason)
        self._record_trade_result(pnl_pct)

    def _process_buys(self, full_df, daily, market):
        """处理买入——动态调整仓位"""
        if len(self.positions) >= self.max_positions:
            return

        # 关键：用 effective_position_size 覆盖 position_size
        effective_pct = self._effective_position_size()
        avail_cash = self.cash * effective_pct
        slots = self.max_positions - len(self.positions)

        candidates = self._strategy.filter_buy(full_df, market.get("date"))
        if candidates.empty:
            return

        scored = self._strategy.score(candidates)
        if scored.empty:
            return

        fill_count = 0
        to_buy = []

        for _, row in scored.iterrows():
            if fill_count >= slots:
                break
            sym = row["symbol"]
            if sym in self.positions:
                continue
            day_row = daily[daily["symbol"] == sym]
            if day_row.empty:
                continue
            r = day_row.iloc[0]
            if r.get("limit_up", False) or r.get("limit_down", False):
                continue
            exec_price = r.get("next_open")
            if pd.isna(exec_price):
                exec_price = r.get("open")
            exec_price = exec_price * (1 + self.slippage_pct)
            if exec_price <= 0 or avail_cash < exec_price * 100:
                continue
            to_buy.append((sym, row, r, exec_price))
            fill_count += 1

        if fill_count == 0:
            return

        per_stock_cash = avail_cash / fill_count
        for sym, row, r, exec_price in to_buy:
            buy_qty = int(per_stock_cash / exec_price)
            buy_qty = (buy_qty // 100) * 100
            if buy_qty < 100:
                continue
            cost = buy_qty * exec_price * (1 + self.commission_pct)
            if cost > self.cash:
                continue
            self.cash -= cost
            self.positions[sym] = {
                "qty": buy_qty,
                "avg_cost": exec_price,
                "entry_date": market["date"],
                "entry_price": exec_price,
                "latest_price": exec_price,
                "days_held": 0,
                "consecutive_bad_days": 0,
                "extra": {},
            }
            trade = {
                "date": market.get("next_date", market["date"]),
                "symbol": sym,
                "entry_price": exec_price,
                "exit_price": None,
                "pnl": None,
                "pnl_pct": None,
                "reason": "BUY",
            }
            self.trades.append(trade)
            self.n_total += 1
            logger.debug(f"  买入 {sym} @{exec_price:.2f} (仓位:{effective_pct:.0%})")

    def get_stats(self):
        """返回统计字典"""
        equity = [s["total_value"] for s in self.market_snapshots]
        dates = [s["date"] for s in self.market_snapshots]
        if len(equity) < 2:
            return {}
        total_return = (equity[-1] - self.initial_cash) / self.initial_cash * 100
        ann_return = total_return / (len(dates) / 244)
        # 夏普
        daily_rets = np.diff(equity) / equity[:-1]
        std = np.std(daily_rets) if len(daily_rets) > 0 else 1.0
        sharpe = (np.mean(daily_rets) / std * np.sqrt(244)) if std > 0 else 0
        # 最大回撤
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (e - peak) / peak
            if dd < max_dd:
                max_dd = dd
        # 胜率（只统计平仓交易）
        closed = [t for t in self.trades if t.get("action") == "sell"]
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        win_rate = wins / len(closed) * 100 if closed else 0
        return {
            "annual_return_pct": ann_return,
            "total_return_pct": total_return,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "win_rate": win_rate,
            "n_trades": len(closed),
        }


def load_data_for_window(y1, y2):
    """加载指定年份窗口的数据"""
    from utils.data_loader import load_strategy_data
    # 加载两年窗口（和 tuner 一样）
    df = load_strategy_data(years=[y1, y2])
    return df


def compute_stats(fw):
    """从 framework 对象提取统计"""
    equity = [s["total_value"] for s in fw.market_snapshots]
    dates = [s["date"] for s in fw.market_snapshots]
    if len(equity) < 2:
        return {}
    total_ret = (equity[-1] - fw.initial_cash) / fw.initial_cash * 100
    ann_ret = total_ret / (len(dates) / 244)
    daily_rets = np.diff(equity) / equity[:-1]
    std = np.std(daily_rets) if len(daily_rets) > 0 else 1.0
    sharpe = (np.mean(daily_rets) / std * np.sqrt(244)) if std > 0 else 0
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak
        if dd < max_dd:
            max_dd = dd
    closed = [t for t in fw.trades if t.get("action") == "sell"]
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    win_rate = wins / len(closed) * 100 if closed else 0
    return {
        "annual_return_pct": ann_ret,
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate": win_rate,
        "n_trades": len(closed),
    }


def run_comparison():
    test_windows = [
        (2010, 2011, "🔴熊市 2010-2011", -9.34, 19.5, -0.81),
        (2014, 2015, "🟢牛市 2014-2015", +13.51, 37.6, 1.04),
        (2021, 2022, "🟡震荡 2021-2022", +13.30, 25.5, 1.14),
    ]

    params = {
        "stop_loss": 0.04,
        "take_profit": 0.20,
        "rsi_filter_min": 50,
        "rsi_filter_max": 65,
    }

    print(f"{'='*75}")
    print(f"{'市场':<22} {'方案':<18} {'年化':>9} {'胜率':>7} {'夏普':>7} {'回撤':>9} {'交易':>6}")
    print(f"{'='*75}")

    all_results = []

    for y1, y2, label, base_ann, base_win, base_sharpe in test_windows:
        print(f"\n  {label}")
        print(f"  {'─'*60}")

        # 加载数据
        print(f"  加载数据中...")
        df = load_data_for_window(y1, y2)
        # date 和 symbol 列已存在于 data_loader 输出中

        # 基准回测（不传日期，用数据全量）
        fw_base = BaseFramework(
            initial_cash=1_000_000,
            position_size=0.15,
            max_positions=3,
        )
        strat_base = ScoreV8Strategy(params)
        fw_base.run_backtest(strat_base, df)
        s_base = compute_stats(fw_base)
        print(f"  基准(无仓位管理)    : {s_base.get('annual_return_pct',0):>+8.2f}%  "
              f"{s_base.get('win_rate',0):>5.1f}%  {s_base.get('sharpe',0):>6.2f}  "
              f"{s_base.get('max_drawdown_pct',0):>+8.1f}%  {s_base.get('n_trades',0):>5d}笔")
        all_results.append({
            "window": f"{y1}-{y2}", "label": label,
            "scheme": "基准", **s_base
        })

        # 连续止损降仓
        fw_ps = SequentialSizingFramework(
            initial_cash=1_000_000,
            position_size=0.15,
            max_positions=3,
            n1=3, n2=5, x=0.50, y=0.25, z1=2, z2=4,
        )
        strat_ps = ScoreV8Strategy(params)
        fw_ps.run_backtest(strat_ps, df)
        s_ps = compute_stats(fw_ps)
        print(f"  连续止损降仓        : {s_ps.get('annual_return_pct',0):>+8.2f}%  "
              f"{s_ps.get('win_rate',0):>5.1f}%  {s_ps.get('sharpe',0):>6.2f}  "
              f"{s_ps.get('max_drawdown_pct',0):>+8.1f}%  {s_ps.get('n_trades',0):>5d}笔")

        # 改善
        delta = s_ps.get('annual_return_pct', 0) - s_base.get('annual_return_pct', 0)
        delta_dd = s_ps.get('max_drawdown_pct', 0) - s_base.get('max_drawdown_pct', 0)
        print(f"  {'  改善:':>20} Δ年化{delta:>+7.2f}%  Δ回撤{delta_dd:>+7.1f}%")
        all_results.append({
            "window": f"{y1}-{y2}", "label": label,
            "scheme": "连续止损降仓", **s_ps
        })

    # 保存结果
    out = "/tmp/position_sizing_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out}")
    return all_results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s  %(message)s")
    run_comparison()
