# wavechan_backtest.py
# WaveChan 选股策略回测引擎
# 基于 wavechan_selector.py 的纯 WaveChan 评分体系

"""
WaveChan 选股策略回测

评分体系：
  一、信号评分（40%）：C_BUY / W2_BUY / W4_BUY 及确认状态
  二、波浪结构评分（30%）：斐波那契回撤区间
  三、动能评分（20%）：RSI / MACD 背离 / 量价配合
  四、缠论确认（10%）：底分型 / 笔破坏

执行规则：
  - 买入阈值：总分 >= 50
  - Top N：每日最多选 N 只
  - T+1 执行：信号日后第一个开盘价买入
  - 止损：使用 WaveChanSelector 返回的 stop_loss
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Parquet data warehouse
WAREHOUSE_PATH = Path("/root/.openclaw/workspace/data/warehouse")

# ============================================================
# 基础模块
# ============================================================

from utils.parquet_db import ParquetDatabaseIntegrator
from config.risk_config import RiskManager


# ============================================================
# Bias 修正器（涨跌停/停牌过滤）
# ============================================================

class BiasCorrector:
    """涨跌停/停牌过滤，避免盘口无法成交的假信号"""

    def __init__(self, price_matrix: pd.DataFrame, trading_dates):
        self.price_matrix = price_matrix
        self.trading_dates = trading_dates
        self._precompute_limits()

    def _precompute_limits(self):
        """预计算涨跌停标记"""
        df = self.price_matrix.copy()
        # 涨跌幅 = (close - prev_close) / prev_close
        close = df['close'] if 'close' in df.columns else df['close']
        prev_close = close.shift(1)
        self.pct_change = (close - prev_close) / prev_close
        self.pct_change = self.pct_change.replace([np.inf, -np.inf], np.nan)

    def is_limit_up(self, symbol: str, date: str) -> bool:
        """是否涨停（真实盘口无法买入）"""
        try:
            pct = self.pct_change.loc[date, symbol]
            return pd.notna(pct) and pct >= 0.095
        except Exception:
            return False

    def is_limit_down(self, symbol: str, date: str) -> bool:
        """是否跌停"""
        try:
            pct = self.pct_change.loc[date, symbol]
            return pd.notna(pct) and pct <= -0.095
        except Exception:
            return False

    def is停牌(self, symbol: str, date: str) -> bool:
        """是否停牌（成交量为0或价格为nan）"""
        try:
            v = self.price_matrix.loc[date, ('volume', symbol)]
            return pd.isna(v) or v == 0
        except Exception:
            return False

    def can_buy(self, symbol: str, date: str, price: float) -> tuple:
        """检查是否可买入"""
        if self.is停牌(symbol, date):
            return False, "停牌"
        if self.is_limit_up(symbol, date):
            return False, "涨停"
        return True, "ok"

    def can_sell(self, symbol: str, date: str) -> tuple:
        """检查是否可卖出"""
        if self.is停牌(symbol, date):
            return False, "停牌"
        if self.is_limit_down(symbol, date):
            return False, "跌停"
        return True, "ok"

    def filter_buy_signals(self, date, candidates: pd.DataFrame) -> pd.DataFrame:
        """过滤涨跌停/停牌候选股"""
        if candidates.empty:
            return candidates
        result = []
        for _, row in candidates.iterrows():
            sym = row['symbol']
            can, reason = self.can_buy(sym, date, row.get('close', 0))
            if can:
                result.append(row)
        if not result:
            return pd.DataFrame()
        return pd.DataFrame(result)


# ============================================================
# 交易模拟器
# ============================================================

class TradingSimulator:
    """WaveChan 策略交易模拟器"""

    DEFAULT_SLIPPAGE = 0.001   # 0.1%/笔
    DEFAULT_COMMISSION = 0.0003  # 0.03%/笔

    def __init__(self, initial_capital: float = 5e5,
                 commission: float = 0.0003,
                 position_limit: int = 3,
                 slippage: float = 0.001):
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: {qty, cost, entry_date, latest_price, stop_loss}}
            'history': []
        }
        self.commission_rate = commission
        self.slippage_rate = slippage
        self.position_limit = position_limit
        self.buy_signals = {}   # {date: [(symbol, price, date, qty, signal_info), ...]}
        self.sell_signals = {}  # {date: [(symbol, price, date, qty, signal_info), ...]}
        self.initial_capital = initial_capital

    def _execute_buy(self, symbol: str, price: float, date: str,
                     quantity: int, signal_info: dict) -> bool:
        """执行买入"""
        exec_price = price * (1 + self.slippage_rate)
        commission = exec_price * quantity * self.commission_rate
        total_cost = exec_price * quantity + commission

        if total_cost > self.portfolio['cash']:
            return False

        stop_loss = signal_info.get('stop_loss', price * 0.92)

        if symbol in self.portfolio['positions']:
            pos = self.portfolio['positions'][symbol]
            pos['qty'] += quantity
            pos['cost'] += total_cost
            pos['latest_price'] = price
            pos['stop_loss'] = stop_loss
        else:
            self.portfolio['positions'][symbol] = {
                'qty': quantity,
                'cost': total_cost,
                'entry_date': date,
                'latest_price': price,
                'stop_loss': stop_loss,
            }

        self.portfolio['cash'] -= total_cost

        self.portfolio['history'].append({
            'date': date,
            'symbol': symbol,
            'type': 'buy',
            'price': price,
            'execution_price': exec_price,
            'quantity': quantity,
            'commission': commission,
            'signal_info': signal_info,
            'signal_type': signal_info.get('signal_type', 'unknown'),
            'total_score': signal_info.get('total_score', 0),
        })

        logging.info(
            f"  ✅ 买入 {symbol} | 价格:{price:.2f} | 数量:{quantity} | "
            f"评分:{signal_info.get('total_score', 0):.0f} | "
            f"信号:{signal_info.get('signal_type', '')}/{signal_info.get('signal_status', '')}"
        )
        return True

    def _execute_sell(self, symbol: str, price: float, date: str,
                       quantity: int, signal_info: dict) -> bool:
        """执行卖出"""
        if symbol not in self.portfolio['positions']:
            return False

        exec_price = price * (1 - self.slippage_rate)
        proceeds = exec_price * quantity
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        pos = self.portfolio['positions'][symbol]
        cost_basis = pos['cost'] / pos['qty']
        pnl_per_share = exec_price - cost_basis
        total_pnl = pnl_per_share * quantity

        self.portfolio['cash'] += net_proceeds

        if pos['qty'] == quantity:
            del self.portfolio['positions'][symbol]
        else:
            pos['qty'] -= quantity
            pos['cost'] = cost_basis * pos['qty']

        self.portfolio['history'].append({
            'date': date,
            'symbol': symbol,
            'type': 'sell',
            'price': price,
            'execution_price': exec_price,
            'quantity': quantity,
            'commission': commission,
            'pnl': total_pnl,
            'signal_info': signal_info,
            'reason': signal_info.get('reason', 'signal'),
        })

        logging.info(
            f"  🔴 卖出 {symbol} | 价格:{price:.2f} | 数量:{quantity} | "
            f"浮盈:{total_pnl:.2f} | 原因:{signal_info.get('reason', 'signal')}"
        )
        return True

    def execute_order(self, order_type: str, symbol: str, price: float,
                      date: str, quantity: int, signal_info: dict) -> bool:
        if order_type == 'buy':
            return self._execute_buy(symbol, price, date, quantity, signal_info)
        elif order_type == 'sell':
            return self._execute_sell(symbol, price, date, quantity, signal_info)
        return False

    def total_value(self, date: str) -> float:
        """计算当前组合总价值"""
        pos_value = sum(
            pos['latest_price'] * pos['qty']
            for pos in self.portfolio['positions'].values()
        )
        return self.portfolio['cash'] + pos_value


# ============================================================
# 回测引擎
# ============================================================

class WaveChanBacktester:
    """
    WaveChan 选股策略回测引擎

    使用方法：
      python wavechan_backtest.py [--start 2024-01-01] [--end 2024-12-31]
    """

    def __init__(self,
                 initial_capital: float = 5e5,
                 position_limit: int = 3,
                 commission: float = 0.0003,
                 slippage: float = 0.001,
                 top_n: int = 5,
                 db_path: str = None):
        self.initial_capital = initial_capital
        self.position_limit = position_limit
        self.commission = commission
        self.slippage = slippage
        self.top_n = top_n

        # 数据
        self.db = ParquetDatabaseIntegrator(db_path)
        self.price_matrix = None
        self.trading_dates = None
        self.bias_corrector = None

        # 策略
        from strategies.wavechan_selector import WaveChanSelector
        self.selector = WaveChanSelector(
            db_path=db_path,
            config={
                'top_n': top_n,
                'stop_loss_pct': 0.08,
            }
        )

        # 风控
        self.risk_manager = RiskManager(
            use_tiered_tp=True,
            use_trailing_stop=True,
            use_time_stop=True,
            stop_loss_pct=-0.08,
            max_hold_days=20,
        )

        # 结果
        self.results = {}

        logging.info(
            f"[WaveChanBacktester] 初始化完成 | "
            f"初始资金:{initial_capital:.0f} | 持仓上限:{position_limit} | TopN:{top_n}"
        )

    # --------------------------------------------------------
    # 数据加载
    # --------------------------------------------------------

    def _load_data(self, start_date: str, end_date: str):
        """加载回测数据（向前延伸 lookback 用于涨跌停计算）"""
        lookback_start = (pd.to_datetime(start_date) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')

        logging.info(f"加载数据: {lookback_start} ~ {end_date}（含 lookback）...")
        self.trading_dates, self.price_matrix = self.db.fetch_trading_dates_and_price_matrix(
            lookback_start, end_date
        )

        # 仅保留回测期内的交易日
        self.trading_dates = self.trading_dates[self.trading_dates >= pd.to_datetime(start_date)]

        # 构建 Bias 修正器
        self.bias_corrector = BiasCorrector(self.price_matrix, self.trading_dates)
        logging.info(f"✅ 数据加载完成 | {len(self.trading_dates)} 个交易日")

    # --------------------------------------------------------
    # 辅助函数
    # --------------------------------------------------------

    def _get_next_open_price(self, date: str, symbol: str) -> Tuple[float, str]:
        """获取下一个有效开盘价"""
        next_date = self._get_next_trading_day(date)
        max_look = 10
        for _ in range(max_look):
            try:
                p = self.price_matrix.loc[next_date, ('open', symbol)]
                if pd.notna(p):
                    return float(p), next_date
            except Exception:
                pass
            next_date = self._get_next_trading_day(next_date)
        return None, None

    def _get_next_trading_day(self, date: str) -> str:
        """下一个交易日"""
        if isinstance(date, str):
            d = datetime.strptime(date, '%Y-%m-%d').date()
        else:
            d = date
        nd = d + timedelta(days=1)
        for _ in range(30):
            nd_str = nd.strftime('%Y-%m-%d')
            if nd_str in self.price_matrix.index:
                return nd_str
            nd += timedelta(days=1)
        return None

    def _get_close_price(self, date: str, symbol: str) -> float:
        """获取收盘价"""
        try:
            p = self.price_matrix.loc[date, ('close', symbol)]
            return float(p) if pd.notna(p) else None
        except Exception:
            return None

    # --------------------------------------------------------
    # 回测主循环
    # --------------------------------------------------------

    def run(self, start_date: str, end_date: str) -> dict:
        """
        运行回测

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            dict: 回测结果统计
        """
        self._load_data(start_date, end_date)

        simulator = TradingSimulator(
            initial_capital=self.initial_capital,
            commission=self.commission,
            position_limit=self.position_limit,
            slippage=self.slippage,
        )

        # ========== 预计算全年信号 ==========
        logging.info(f"[回测] 生成全年信号 {start_date} ~ {end_date}...")
        buy_signals_df, sell_signals_df = self.selector.get_signals(start_date, end_date)

        if buy_signals_df is None or buy_signals_df.empty:
            logging.warning("[回测] 全年无买入信号！")
            return self._generate_report(simulator)

        logging.info(f"[回测] 买入信号 {len(buy_signals_df)} 条，"
                     f"{len(buy_signals_df.index.unique())} 个交易日有信号")

        # ========== 逐日回测 ==========
        trading_dates_list = sorted(self.trading_dates.tolist())
        dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
                     for d in trading_dates_list]

        logging.info(f"[回测] 开始逐日回测，共 {len(dates_str)} 个交易日...")

        for i, date in enumerate(dates_str):
            # ---- 处理卖出计划 ----
            if str(date) in simulator.sell_signals:
                for item in simulator.sell_signals[str(date)]:
                    sym, price, day, qty, info = item
                    simulator.execute_order('sell', sym, price, day, qty, info)
                del simulator.sell_signals[str(date)]

            # ---- 处理买入计划 ----
            if str(date) in simulator.buy_signals:
                for item in simulator.buy_signals[str(date)]:
                    sym, price, day, qty, info = item
                    simulator.execute_order('buy', sym, price, day, qty, info)
                del simulator.buy_signals[str(date)]

            # ---- 更新最新价 ----
            for sym, pos in simulator.portfolio['positions'].items():
                close_p = self._get_close_price(date, sym)
                if close_p:
                    pos['latest_price'] = close_p

            # ---- 止损检查 ----
            self._check_stop_loss(date, simulator)

            # ---- 风控止盈检查 ----
            self._check_take_profit(date, simulator)

            # ---- 卖出信号处理 ----
            self._process_sell_signals(date, simulator, sell_signals_df)

            # ---- 买入信号处理 ----
            self._process_buy_signals(date, simulator, buy_signals_df)

            # ---- 记录净值 ----
            total_val = simulator.total_value(date)
            simulator.portfolio['history'].append({
                'date': date,
                'value': total_val,
            })

            if (i + 1) % 60 == 0:
                logging.info(f"  进度 {i+1}/{len(dates_str)} | 净值:{total_val:.0f} | "
                             f"持仓:{len(simulator.portfolio['positions'])}")

        return self._generate_report(simulator)

    # --------------------------------------------------------
    # 每日处理
    # --------------------------------------------------------

    def _check_stop_loss(self, date, simulator):
        """止损检查：使用 WaveChanSelector 返回的 stop_loss"""
        for sym, pos in list(simulator.portfolio['positions'].items()):
            # 跳过今日买入
            if pos.get('entry_date') == date:
                continue

            current_price = pos['latest_price']
            stop_loss = pos.get('stop_loss', 0)

            if stop_loss > 0 and current_price <= stop_loss:
                try:
                    next_price, next_day = self._get_next_open_price(date, sym)
                    if next_price is None:
                        continue
                    can_sell, _ = self.bias_corrector.can_sell(sym, next_day)
                    if not can_sell:
                        continue

                    info = {
                        'reason': f'stop_loss',
                        'type': 'stop_loss',
                        'stop_loss': stop_loss,
                        'entry': pos['cost'] / pos['qty'],
                        'current': current_price,
                    }
                    if next_day not in simulator.sell_signals:
                        simulator.sell_signals[next_day] = []
                    simulator.sell_signals[next_day].append([
                        sym, next_price, next_day, pos['qty'], info
                    ])
                    logging.info(f"  ⚠️ 止损触发 {sym} | 现价:{current_price:.2f} | 止损:{stop_loss:.2f}")
                except Exception as e:
                    logging.debug(f"止损处理失败 {sym}: {e}")

    def _check_take_profit(self, date, simulator):
        """分档止盈 + 追踪止损"""
        for sym, pos in list(simulator.portfolio['positions'].items()):
            if pos.get('entry_date') == date:
                continue

            entry_price = pos['cost'] / pos['qty']
            current_price = pos['latest_price']
            hold_days = (datetime.strptime(date, '%Y-%m-%d').date() -
                         datetime.strptime(pos['entry_date'], '%Y-%m-%d').date()).days

            peak_price = pos.get('highest_price', current_price)
            if current_price > peak_price:
                peak_price = current_price
                pos['highest_price'] = peak_price

            should_sell, reason, sell_pct = self.risk_manager.check(
                current_price=current_price,
                entry_price=entry_price,
                peak_price=peak_price,
                hold_days=hold_days,
            )

            if should_sell:
                try:
                    next_price, next_day = self._get_next_open_price(date, sym)
                    if next_price is None:
                        continue
                    can_sell, _ = self.bias_corrector.can_sell(sym, next_day)
                    if not can_sell:
                        continue

                    sell_qty = int(pos['qty'] * sell_pct)
                    if sell_qty <= 0:
                        sell_qty = pos['qty']

                    info = {
                        'reason': reason,
                        'type': 'take_profit',
                        'sell_pct': sell_pct,
                    }
                    if next_day not in simulator.sell_signals:
                        simulator.sell_signals[next_day] = []
                    simulator.sell_signals[next_day].append([
                        sym, next_price, next_day, sell_qty, info
                    ])
                except Exception:
                    pass

    def _process_sell_signals(self, date, simulator, sell_signals_df):
        """处理卖出信号"""
        if sell_signals_df is None or sell_signals_df.empty:
            return

        holding_symbols = list(simulator.portfolio['positions'].keys())
        if not holding_symbols:
            return

        try:
            day_sells = sell_signals_df.loc[date]
            if isinstance(day_sells, pd.DataFrame):
                day_sells = day_sells[day_sells['symbol'].isin(holding_symbols)]
            else:
                day_sells = pd.DataFrame([day_sells]) if day_sells['symbol'] in holding_symbols else pd.DataFrame()
        except KeyError:
            return

        if day_sells.empty:
            return

        for _, row in day_sells.iterrows():
            sym = row['symbol']
            if sym not in simulator.portfolio['positions']:
                continue

            # 只对盈利持仓发出卖出计划
            pos = simulator.portfolio['positions'][sym]
            entry = pos['cost'] / pos['qty']
            if pos['latest_price'] <= entry:
                continue

            try:
                next_price, next_day = self._get_next_open_price(date, sym)
                if next_day is None:
                    continue
                info = {
                    'reason': 'wavechan_sell_signal',
                    'type': 'sell_signal',
                    'wave_trend': row.get('wave_trend', 'down'),
                    'fractal': row.get('fractal', ''),
                }
                if next_day not in simulator.sell_signals:
                    simulator.sell_signals[next_day] = []
                simulator.sell_signals[next_day].append([
                    sym, next_price, next_day, pos['qty'], info
                ])
            except Exception:
                pass

    def _process_buy_signals(self, date, simulator, buy_signals_df):
        """处理买入信号"""
        available_slots = self.position_limit - len(simulator.portfolio['positions'])
        if available_slots <= 0:
            return

        try:
            day_buys = buy_signals_df.loc[date]
            if isinstance(day_buys, pd.Series):
                day_buys = pd.DataFrame([day_buys])
        except KeyError:
            return

        if day_buys.empty:
            return

        holding = list(simulator.portfolio['positions'].keys())

        # 过滤涨跌停
        day_buys = self.bias_corrector.filter_buy_signals(date, day_buys)
        if day_buys.empty:
            return

        # 按评分降序，优先高评分
        day_buys = day_buys.sort_values('total_score', ascending=False)

        selected = 0
        for _, row in day_buys.iterrows():
            if selected >= available_slots:
                break
            sym = row['symbol']
            if sym in holding:
                continue

            try:
                next_price, next_day = self._get_next_open_price(date, sym)
                if next_day is None:
                    continue

                # Bias 修正：涨跌停过滤
                can_buy, reason = self.bias_corrector.can_buy(sym, next_day, next_price)
                if not can_buy:
                    continue

                # 资金检查
                max_investment = simulator.portfolio['cash'] // 1.1  # 预留手续费
                if max_investment <= 0:
                    break
                max_afford = max_investment // (next_price * (1 + self.commission))
                max_afford = (max_afford // 100) * 100
                if max_afford <= 0:
                    continue

                signal_info = {
                    'type': 'wavechan_buy',
                    'signal_type': row.get('signal_type', 'unknown'),
                    'signal_status': row.get('signal_status', 'unknown'),
                    'total_score': row.get('total_score', 0),
                    'signal_score': row.get('signal_score', 0),
                    'structure_score': row.get('structure_score', 0),
                    'momentum_score': row.get('momentum_score', 0),
                    'chanlun_score': row.get('chanlun_score', 0),
                    'wave_stage': row.get('wave_stage', ''),
                    'wave_trend': row.get('wave_trend', ''),
                    'stop_loss': row.get('stop_loss', next_price * 0.92),
                    'rsi': row.get('rsi', 50),
                    'symbol': sym,
                    'close': row.get('close', 0),
                    'industry': row.get('industry', ''),
                    'fractal': row.get('fractal', ''),
                }

                if next_day not in simulator.buy_signals:
                    simulator.buy_signals[next_day] = []
                simulator.buy_signals[next_day].append([
                    sym, next_price, next_day, max_afford, signal_info
                ])
                selected += 1

            except Exception as e:
                logging.debug(f"买入处理失败 {sym}: {e}")

    # --------------------------------------------------------
    # 报告生成
    # --------------------------------------------------------

    def _generate_report(self, simulator: TradingSimulator) -> dict:
        """生成回测报告"""
        history = simulator.portfolio['history']

        # 分离净值记录和交易记录
        daily_records = [r for r in history if 'value' in r and 'type' not in r]
        trade_records = [r for r in history if 'type' in r]

        df_value = pd.DataFrame(daily_records).set_index('date')
        if len(df_value) > 1:
            df_value['returns'] = df_value['value'].pct_change()
            df_value['cummax'] = df_value['value'].cummax()
            df_value['drawdown'] = df_value['value'] / df_value['cummax'] - 1

        trades = pd.DataFrame(trade_records)

        # === 核心指标 ===
        initial = self.initial_capital
        final_value = df_value['value'].iloc[-1] if not df_value.empty else initial
        total_return = final_value / initial - 1
        max_drawdown = df_value['drawdown'].min() if not df_value.empty else 0

        # 年化收益（假设252交易日）
        n_days = len(df_value)
        if n_days > 0:
            annualized = (final_value / initial) ** (252 / n_days) - 1
        else:
            annualized = 0

        # 交易统计
        buy_trades = trades[trades['type'] == 'buy'] if not trades.empty else pd.DataFrame()
        sell_trades = trades[trades['type'] == 'sell'] if not trades.empty else pd.DataFrame()

        n_buy = len(buy_trades)
        n_sell = len(sell_trades)
        win_trades = sell_trades[sell_trades['pnl'] > 0] if not sell_trades.empty else pd.DataFrame()
        n_win = len(win_trades)
        win_rate = n_win / n_sell if n_sell > 0 else 0

        avg_pnl = sell_trades['pnl'].mean() if not sell_trades.empty else 0
        total_pnl = sell_trades['pnl'].sum() if not sell_trades.empty else 0

        # === 打印报告 ===
        report = {
            'initial_capital': initial,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized,
            'max_drawdown': max_drawdown,
            'n_buy': n_buy,
            'n_sell': n_sell,
            'n_win': n_win,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'df_value': df_value,
            'trades': trades,
        }

        self.results = report

        # 打印
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  WaveChan 选股策略回测报告")
        print(f"{separator}")
        print(f"  初始资金:   {initial:,.0f} 元")
        print(f"  最终净值:   {final_value:,.0f} 元")
        print(f"  总收益率:   {total_return:.2%}")
        print(f"  年化收益:   {annualized:.2%}")
        print(f"  最大回撤:   {max_drawdown:.2%}")
        print(f"  买入次数:   {n_buy}")
        print(f"  卖出次数:   {n_sell}")
        print(f"  盈利次数:   {n_win}")
        print(f"  胜率:       {win_rate:.2%}")
        print(f"  平均盈亏:   {avg_pnl:,.0f} 元")
        print(f"  总盈亏:     {total_pnl:,.0f} 元")
        print(f"{separator}\n")

        # === 保存文件 ===
        os.makedirs('./backtestresult', exist_ok=True)
        ts = str(int(time.time()))

        # Excel 报告
        xlsx_path = f'./backtestresult/wavechan_backtest_{ts}.xlsx'
        with pd.ExcelWriter(xlsx_path) as writer:
            if not trades.empty:
                trades.to_excel(writer, sheet_name='交易记录', index=False)
            if not df_value.empty:
                df_value.to_excel(writer, sheet_name='每日净值')

            summary = pd.DataFrame([{
                '指标': '初始资金',
                '数值': f"{initial:,.0f}",
            }, {
                '指标': '最终净值',
                '数值': f"{final_value:,.0f}",
            }, {
                '指标': '总收益率',
                '数值': f"{total_return:.2%}",
            }, {
                '指标': '年化收益',
                '数值': f"{annualized:.2%}",
            }, {
                '指标': '最大回撤',
                '数值': f"{max_drawdown:.2%}",
            }, {
                '指标': '买入次数',
                '数值': str(n_buy),
            }, {
                '指标': '卖出次数',
                '数值': str(n_sell),
            }, {
                '指标': '胜率',
                '数值': f"{win_rate:.2%}",
            }, {
                '指标': '平均盈亏',
                '数值': f"{avg_pnl:,.0f}",
            }, {
                '指标': '总盈亏',
                '数值': f"{total_pnl:,.0f}",
            }])
            summary.to_excel(writer, sheet_name='指标汇总', index=False)

        logging.info(f"✅ 回测报告已保存: {xlsx_path}")

        # CSV 交易记录
        if not trades.empty:
            csv_path = f'./backtestresult/wavechan_trades_{ts}.csv'
            trades.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"✅ 交易记录已保存: {csv_path}")

        return report


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='WaveChan 选股策略回测')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--capital', type=float, default=500000, help='初始资金')
    parser.add_argument('--top_n', type=int, default=5, help='每日选股数量')
    parser.add_argument('--position_limit', type=int, default=3, help='最大持仓数')

    args = parser.parse_args()

    bt = WaveChanBacktester(
        initial_capital=args.capital,
        position_limit=args.position_limit,
        top_n=args.top_n,
    )

    result = bt.run(args.start, args.end)
