# simulator/multi_simulator.py
# 多策略并行调度器
# =============================================================
"""
多策略并行模拟盘。

职责：
  - 管理多个独立的 BasePortfolio（每个策略独立账户）
  - 逐年/逐日运行调度
  - 汇总各策略结果

使用方式：
  sim = MultiSimulator(initial_cash_per_strategy=1_000_000)
  sim.add_strategy("V4", ScoreV4Strategy())
  sim.add_strategy("V8", ScoreV8Strategy())
  results = sim.run("2026-01-01", "2026-03-30", data_loader=load_daily_data)
"""

import logging
import math
from typing import Callable, Dict, List, Optional, Any, Tuple
import pandas as pd

from .base_portfolio import BasePortfolio

logger = logging.getLogger(__name__)


class MultiSimulator:
    """
    多策略并行调度器

    各策略拥有独立账户（独立现金、持仓、交易记录）。
    每日同步调度：同一日期数据分发给所有 portfolio.on_day()。

    Attributes:
        initial_cash_per_strategy: 每个策略的初始资金
        portfolios: {strategy_name: BasePortfolio} 字典
    """

    def __init__(
        self,
        initial_cash_per_strategy: float = 1_000_000,
    ):
        """
        Args:
            initial_cash_per_strategy: 每个策略账户的初始资金（默认 100万）
        """
        self.initial_cash_per_strategy = initial_cash_per_strategy
        self.portfolios: Dict[str, BasePortfolio] = {}

        # 全局交易记录（所有策略合并）
        self.all_trades: List[dict] = []

        # 每日权益曲线 {strategy_name: [{date, total_value}, ...]}
        self._daily_values: Dict[str, List[dict]] = {}

        # 运行元信息
        self._start_date: Optional[str] = None
        self._end_date: Optional[str] = None
        self._completed = False

    def add_strategy(
        self,
        name: str,
        strategy: Any,
        config: Optional[dict] = None,
        initial_cash: Optional[float] = None,
    ) -> BasePortfolio:
        """
        添加一个策略（创建独立账户）

        Args:
            name: 策略名称（唯一标识）
            strategy: 策略实例（需实现 filter_buy / score / should_sell 接口）
            config: 策略配置（传入 strategy.__init__(config)）
                   支持 stop_loss, take_profit, max_positions, position_size
            initial_cash: 该策略的初始资金（默认用 self.initial_cash_per_strategy）

        Returns:
            创建的 BasePortfolio 实例
        """
        if name in self.portfolios:
            raise ValueError(f"策略名称 '{name}' 已存在，请使用唯一名称")

        cfg = {
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'max_positions': 5,
            'position_size': 0.20,
            **(config or {}),
        }

        cash = initial_cash if initial_cash is not None else self.initial_cash_per_strategy
        portfolio = BasePortfolio(
            name=name,
            initial_cash=cash,
            strategy=strategy,
        )
        self.portfolios[name] = portfolio

        logger.info(
            f"[MultiSimulator] 添加策略 '{name}': "
            f"初始资金={cash:,.0f}, stop_loss={cfg['stop_loss']:.0%}, "
            f"take_profit={cfg['take_profit']:.0%}, "
            f"max_positions={cfg['max_positions']}"
        )
        return portfolio

    def run(
        self,
        start_date: str,
        end_date: str,
        data_loader: Optional[Callable[[str, str], pd.DataFrame]] = None,
        daily_data: Optional[pd.DataFrame] = None,
        date_column: str = 'date',
        symbol_column: str = 'symbol',
    ):
        """
        运行多策略模拟

        两种数据提供方式（任选其一）：
          1. data_loader: Callable[[start, end] -> pd.DataFrame]
          2. daily_data: pd.DataFrame（直接传入完整数据）

        每日调度逻辑：
          1. 按 date_column 遍历每个交易日
          2. 构建 next_open 映射（当日收盘后次日开盘价）
          3. 分发给各 portfolio.on_day()

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            data_loader: 数据加载函数（可选）
            daily_data: 日线数据 DataFrame（可选，与 data_loader 二选一）
            date_column: 日期列名（默认 'date'）
            symbol_column: 股票代码列名（默认 'symbol'）

        Raises:
            ValueError: 既无 data_loader 也无 daily_data
        """
        self._start_date = start_date
        self._end_date = end_date
        self._completed = False

        # ---- 加载数据 ----
        if daily_data is not None:
            df = daily_data.copy()
        elif data_loader is not None:
            logger.info(f"[MultiSimulator] 加载数据 {start_date} ~ {end_date} ...")
            df = data_loader(start_date, end_date)
            logger.info(f"[MultiSimulator] 加载完成，共 {len(df)} 条记录")
        else:
            raise ValueError(
                "必须提供 data_loader 或 daily_data 之一"
            )

        if df.empty:
            logger.warning("[MultiSimulator] 数据为空，跳过运行")
            return self

        # ---- 构建 next_open 映射 ----
        df = df.sort_values([symbol_column, date_column])
        df['next_open'] = df.groupby(symbol_column)['open'].shift(-1)

        # ---- 按日期遍历 ----
        dates = sorted(df[date_column].unique())
        dates_in_range = [
            d for d in dates
            if str(d) >= str(start_date) and str(d) <= str(end_date)
        ]

        logger.info(
            f"[MultiSimulator] 开始运行: {dates_in_range[0]} ~ {dates_in_range[-1]} "
            f"共 {len(dates_in_range)} 个交易日, {len(self.portfolios)} 个策略"
        )

        for i, date in enumerate(dates_in_range):
            daily_df = df[df[date_column] == date].copy()

            if daily_df.empty:
                continue

            market_ctx = {'date': date, 'breadth': None}  # 可扩展市场宽度

            for name, portfolio in self.portfolios.items():
                try:
                    trades = portfolio.on_day(str(date), daily_df, market=market_ctx)
                    self.all_trades.extend(trades)
                except Exception as e:
                    logger.error(
                        f"[MultiSimulator] {name} @ {date} 异常: {e}",
                        exc_info=True,
                    )

            # ---- 记录每日权益曲线 ----
            # 构建当日收盘价映射（用于计算持仓市值）
            close_prices = {}
            for _, r in daily_df.iterrows():
                close_prices[r[symbol_column]] = r['close']

            for name, portfolio in self.portfolios.items():
                if name not in self._daily_values:
                    self._daily_values[name] = []
                total_val = portfolio.get_total_value(close_prices)
                self._daily_values[name].append({
                    'date': str(date),
                    'total_value': total_val,
                })

            # 每50天打印一次进度
            if (i + 1) % 50 == 0:
                logger.info(
                    f"[MultiSimulator] 进度 {i + 1}/{len(dates_in_range)} "
                    f"({(i+1)/len(dates_in_range)*100:.0f}%)"
                )

        self._completed = True
        logger.info(f"[MultiSimulator] 运行完成，共 {len(self.all_trades)} 笔交易")

        return self

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        daily_values: List[dict],
        trades: List[dict],
        initial_cash: float,
        n_days: int = 0,
    ) -> dict:
        """
        计算性能指标（最大回撤、夏普、年化等）

        Args:
            daily_values: 每日权益列表 [{date, total_value}, ...]
            trades: 交易历史
            initial_cash: 初始资金
            n_days: 交易天数（用于年化）

        Returns:
            dict，包含 max_drawdown, sharpe_ratio, annualized_return 等
        """
        if not daily_values or len(daily_values) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': '0.00%',
                'sharpe_ratio': 0.0,
                'annualized_return': 0.0,
                'annualized_return_pct': '0.00%',
            }

        values = [v['total_value'] for v in daily_values]
        peak = values[0]
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        # 计算最大回撤
        for v in values:
            if v > peak:
                peak = v
            dd = peak - v
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
                max_drawdown_pct = dd_pct

        # 计算夏普比率（假设无风险利率 3%）
        risk_free_rate = 0.03
        if len(values) >= 2:
            # 日收益率（基于每日权益曲线计算）
            daily_returns = []
            for i in range(1, len(values)):
                if values[i - 1] > 0:
                    ret = (values[i] - values[i - 1]) / values[i - 1]
                    daily_returns.append(ret)

            if daily_returns:
                mean_ret = sum(daily_returns) / len(daily_returns)
                # 标准差
                variance = sum((r - mean_ret) ** 2 for r in daily_returns) / max(len(daily_returns) - 1, 1)
                std_dev = math.sqrt(variance) if variance > 0 else 0

                # 年化（假设252交易日）
                annual_ret = mean_ret * 252
                annual_std = std_dev * math.sqrt(252)

                if annual_std > 0:
                    sharpe = (annual_ret - risk_free_rate) / annual_std
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # 年化收益率
        # 注意：n_days 应该是实际交易天数（非日历天）
        # dates_in_range = sorted(df['date'].unique()) 已经过 date_column 过滤
        # 每日权益曲线 _daily_values 按交易日记录，故 len(daily_values) = 交易天数
        # 年化系数使用 252（A股每年实际交易天数约 240-252 天）
        if values[-1] > 0 and initial_cash > 0:
            total_return = values[-1] / initial_cash - 1
            n = n_days if n_days > 0 else len(daily_values)
            if n > 0:
                years = n / 252  # n 是交易天数，252 是年均交易天数
                if years > 0:
                    annualized = (1 + total_return) ** (1 / years) - 1
                else:
                    annualized = total_return
            else:
                annualized = total_return
        else:
            annualized = 0.0

        return {
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': f"{max_drawdown_pct * 100:+.2f}%",
            'sharpe_ratio': round(sharpe, 2),
            'annualized_return': round(annualized, 4),
            'annualized_return_pct': f"{annualized * 100:+.2f}%",
        }

    def get_results(self) -> dict:
        """
        返回各策略的独立结果

        Returns:
            dict，格式：
            {
              'strategy_name': {
                'initial_cash': float,
                'final_value': float,
                'total_return': float,
                'total_return_pct': str,
                'n_positions': int,
                'n_trades': int,
                'win_rate': float,
                'total_pnl': float,
                'positions': [...],
                'trades': [...],
              },
              ...
            }
        """
        results = {}

        for name, portfolio in self.portfolios.items():
            stats = portfolio.get_stats()

            # 获取未平仓持仓（取每日最后一条行情的收盘价）
            positions_detail = []
            for sym, pos in portfolio.positions.items():
                positions_detail.append({
                    'symbol': sym,
                    'qty': pos['qty'],
                    'avg_cost': pos['avg_cost'],
                    'entry_date': pos.get('entry_date', ''),
                    'entry_sma20_le_sma55': pos.get('entry_sma20_le_sma55', False),
                })

            # 计算增强指标
            daily_vals = self._daily_values.get(name, [])
            n_trading_days = len(daily_vals)
            metrics = self._compute_metrics(
                daily_vals,
                portfolio.trades,
                stats['initial_cash'],
                n_trading_days,
            )

            results[name] = {
                'strategy': name,
                'initial_cash': stats['initial_cash'],
                'final_value': stats['total_value'],
                'cash': stats['cash'],
                'total_return': stats['total_return'],
                'total_return_pct': stats['total_return_pct'],
                'n_positions': stats['n_positions'],
                'n_trades': stats['n_trades'],
                'win_rate': stats['win_rate'],
                'total_pnl': stats['total_pnl'],
                # 新增指标
                'max_drawdown': metrics['max_drawdown'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'annualized_return': metrics['annualized_return'],
                'annualized_return_pct': metrics['annualized_return_pct'],
                'n_trading_days': n_trading_days,
                'positions': positions_detail,
                'trades': [t for t in portfolio.trades],
            }

        return results

    def get_summary(self) -> pd.DataFrame:
        """
        返回多策略对比摘要表

        Returns:
            pd.DataFrame，每行一个策略，列包括收益、回撤、夏普等
        """
        results = self.get_results()
        rows = []

        for name, res in results.items():
            closed_trades = [t for t in res['trades'] if t['action'] == 'sell']
            pnls = [t.get('pnl_pct', 0) for t in closed_trades]

            # 区分止盈/止损次数
            take_profit_trades = [t for t in closed_trades if 'TAKE_PROFIT' in t.get('reason', '')]
            stop_loss_trades = [t for t in closed_trades if 'STOP_LOSS' in t.get('reason', '')]
            ma_death_cross_trades = [t for t in closed_trades if 'MA_DEATH_CROSS' in t.get('reason', '')]
            trend_break_trades = [t for t in closed_trades if 'TREND_BREAK' in t.get('reason', '')]

            row = {
                'strategy': name,
                'initial_cash': res['initial_cash'],
                'final_value': res['final_value'],
                'total_return_pct': res['total_return_pct'],
                'total_return': res['total_return'],
                'max_drawdown_pct': res['max_drawdown_pct'],
                'sharpe_ratio': res['sharpe_ratio'],
                'annualized_return_pct': res['annualized_return_pct'],
                'n_trades': res['n_trades'],
                'win_rate': f"{res['win_rate']:.1f}%",
                'n_positions': res['n_positions'],
                'total_pnl': res['total_pnl'],
            }

            if pnls:
                row['avg_hold_pnl'] = f"{sum(pnls)/len(pnls):.2f}%"
                row['best_trade'] = f"{max(pnls):.2f}%"
                row['worst_trade'] = f"{min(pnls):.2f}%"
            else:
                row['avg_hold_pnl'] = 'N/A'
                row['best_trade'] = 'N/A'
                row['worst_trade'] = 'N/A'

            # 止盈/止损统计
            row['stop_loss_count'] = len(stop_loss_trades)
            row['take_profit_count'] = len(take_profit_trades)
            row['ma_death_cross_count'] = len(ma_death_cross_trades)
            row['trend_break_count'] = len(trend_break_trades)

            # 分开计算止盈/止损的平均盈亏
            if stop_loss_trades:
                sl_pnls = [t.get('pnl_pct', 0) for t in stop_loss_trades]
                row['avg_stop_loss_pnl'] = f"{sum(sl_pnls)/len(sl_pnls):.2f}%"
            else:
                row['avg_stop_loss_pnl'] = 'N/A'

            if take_profit_trades:
                tp_pnls = [t.get('pnl_pct', 0) for t in take_profit_trades]
                row['avg_take_profit_pnl'] = f"{sum(tp_pnls)/len(tp_pnls):.2f}%"
            else:
                row['avg_take_profit_pnl'] = 'N/A'

            rows.append(row)

        return pd.DataFrame(rows)

    def print_summary(self):
        """打印多策略对比摘要"""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("多策略模拟盘 — 结果摘要")
        print("=" * 80)
        if self._start_date and self._end_date:
            print(f"运行区间: {self._start_date} ~ {self._end_date}")
        print(f"策略数量: {len(self.portfolios)}")
        print("-" * 80)
        print(summary.to_string(index=False))
        print("=" * 80)

    def __repr__(self) -> str:
        return (
            f"<MultiSimulator "
            f"strategies={list(self.portfolios.keys())} "
            f"cash_per_strategy={self.initial_cash_per_strategy:,.0f}>"
        )
