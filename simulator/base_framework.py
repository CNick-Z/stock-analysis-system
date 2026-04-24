#!/usr/bin/env python3
"""
BaseFramework — 统一回测/模拟盘框架
=====================================
所有策略共用此框架，只需实现 Strategy 接口。

接口：
    Strategy.filter_buy(daily_df) → DataFrame
    Strategy.score(candidates) → DataFrame（带 score 列）
    Strategy.should_sell(row, pos, market) → (bool, str)
    Strategy.on_tick(row, pos)  → 更新持仓状态

用法：
    # 回测
    framework = BaseFramework(initial_cash=500_000)
    framework.run_backtest(strategy=V8Strategy(), start='2024-01-01', end='2024-12-31')

    # 模拟盘
    framework = BaseFramework(initial_cash=500_000)
    framework.run_simulate(strategy=V8Strategy(), target_date='2026-03-31')

框架职责：
    ✅ 逐日循环（先出场再入场）
    ✅ 涨跌停过滤
    ✅ 仓位分配
    ✅ 资金管理
    ✅ 状态持久化
    ✅ 交易通知

策略职责：
    ✅ filter_buy / score / should_sell / on_tick
    ✅ 提供入场/出场信号
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class Position:
    """持仓"""
    symbol: str
    qty: int
    avg_cost: float
    entry_date: str
    entry_price: float
    # 框架维护的状态
    days_held: int = 0
    consecutive_bad_days: int = 0
    # 额外字段（策略自行读写）
    extra: Dict = field(default_factory=dict)


@dataclass
class Trade:
    """交易记录"""
    date: str
    symbol: str
    action: str          # buy / sell
    price: float
    qty: int
    pnl: float = 0.0    # 平仓盈亏（卖出时计算）
    signal_type: str = ""  # 买入时的信号类型
    reason: str = ''     # 出场原因
    pnl_pct: float = 0.0


@dataclass
class MarketSnapshot:
    """市场快照（每个交易日更新）"""
    date: str
    cash: float
    total_value: float
    n_positions: int
    total_return: float


@dataclass
class CandidateSignal:
    """候选交易信号"""
    date: str
    symbol: str
    score: float
    # 技术指标
    rsi: float = 0.0
    cci: float = 0.0
    wr: float = 0.0
    macd_state: str = ""
    kdj_state: str = ""
    boll_state: str = ""
    # 基本信息
    industry: str = ""
    name: str = ""
    close: float = 0.0
    limit_up: bool = False
    limit_down: bool = False
    # 状态
    status: str = ""  # OK-买入 / X-未买-原因
    reason: str = ""  # 未买原因


# ── 工具函数 ──────────────────────────────────────────────────────
def _infer_macd_state(r: pd.Series) -> str:
    """从行情行推断 MACD 状态"""
    macd = r.get("macd", 0) or 0
    macd_sig = r.get("macd_signal", 0) or 0
    if macd > macd_sig:
        return "金叉"
    elif macd > 0:
        return "红柱"
    else:
        return "绿柱"


def _infer_kdj_state(r: pd.Series) -> str:
    """从行情行推断 KDJ 状态"""
    kdj_j = r.get("kdj_j", 0) or 0
    if kdj_j > 80:
        return "高位"
    elif kdj_j < 20:
        return "低位"
    else:
        return "中性"


# ============================================================
# 策略接口（抽象）
# ============================================================

class Strategy:
    """
    策略基类 — 所有策略必须实现此接口

    接口方法由框架调用，策略提供信号判断逻辑。
    """

    name: str = "Strategy"

    # 策略声明需要哪些额外列（框架在 prepare 时加载并合并到 daily_df）
    REQUIRED_COLUMNS: list = []

    def prepare(self, dates: list, loader: 'DataLoader'):
        """
        框架调用：策略按需加载私有数据

        Args:
            dates: 交易日列表 [start_date, ..., end_date]
            loader: 数据加载器（DataLoader 实例）
        """
        self._extra_data = {}

    def filter_buy(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        入场过滤：返回候选股票（必须包含 symbol 列）

        Args:
            daily_df: 当日全市场数据

        Returns:
            DataFrame with symbol column
        """
        raise NotImplementedError

    def score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        候选评分排序：返回带 score 列的 DataFrame

        Args:
            candidates: filter_buy() 的结果

        Returns:
            DataFrame with symbol and score columns, sorted descending by score
        """
        raise NotImplementedError

    def should_sell(self, row: pd.Series, pos: dict, market: dict) -> tuple:
        """
        判断是否出场

        Args:
            row: 当日行情（Series）
            pos: 持仓信息（dict），框架维护
            market: 市场快照（dict）

        Returns:
            (should_sell: bool, reason: str)
        """
        raise NotImplementedError

    def on_tick(self, row: pd.Series, pos: dict, market: dict) -> None:
        """
        每日 tick 回调 — 更新持仓状态

        Args:
            row: 当日行情
            pos: 持仓信息（框架维护，策略修改）
            market: 市场快照
        """
        # 默认实现：仅更新持仓天数
        pos['days_held'] = pos.get('days_held', 0) + 1


# ============================================================
# BaseFramework
# ============================================================

# A股税费标准（从实盘交易记录反推）
# https://www.investopedia.com/terms/s/stamptax.asp
STOCK_BUY_COMMISSION = 0.00011   # 买入佣金 万1.1
STOCK_SELL_COMMISSION = 0.00011  # 卖出佣金 万1.1
STOCK_STAMP_TAX = 0.0005        # 印花税（仅卖出）万5
ETF_COMMISSION = 0.0001          # ETF 买卖佣金 万1（无印花税）


class BaseFramework:
    """
    统一回测/模拟盘框架

    A股税费规则（从实盘交易记录反推）：
      个股买入：佣金 万1.1
      个股卖出：佣金 万1.1 + 印花税 万5 = 万6.1
      ETF    买入/卖出：佣金 万1（无印花税）

    Args:
        initial_cash: 初始资金
        max_positions: 最大持仓数
        position_size: 每只股票仓位比例
        slippage_pct: 滑点（默认 0.0001）
    """

    def __init__(
        self,
        initial_cash: float = 500_000,
        max_positions: int = 5,
        position_size: float = 0.20,
        slippage_pct: float = 0.0001,
        state_file: str = "/tmp/framework_state.json",
        market_regime_filter = None,
    ):
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.position_size = position_size
        self.slippage_pct = slippage_pct
        self.state_file = Path(state_file)
        self.market_regime_filter = market_regime_filter

        # 运行时状态
        self.cash: float = initial_cash
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.market_snapshots: List[dict] = []
        self.n_winning: int = 0
        self.n_total: int = 0
        self.candidates: List[dict] = []  # 候选信号列表
        self._strategy: Optional[Strategy] = None

    # ============================================================
    # 税费计算（A股实盘标准）
    # ============================================================

    def _is_etf(self, symbol: str, category: str = "") -> bool:
        """判断是否为ETF（category优先，其次按代码前缀）"""
        if category in ("ETF", "etf"):
            return True
        if category in ("Stock", "stock"):
            return False
        # ETF代码前缀：15/51/56/58/59 开头
        return symbol.startswith(("15", "51", "56", "58", "59"))

    def _buy_cost(self, amount: float, is_etf: bool) -> float:
        """买入成本（扣除佣金后净成本）"""
        commission = ETF_COMMISSION if is_etf else STOCK_BUY_COMMISSION
        return amount * (1 + commission)

    def _sell_proceeds(self, amount: float, is_etf: bool) -> float:
        """卖出收入（扣除佣金+印花税后净收入）"""
        if is_etf:
            return amount * (1 - ETF_COMMISSION)
        else:
            return amount * (1 - STOCK_SELL_COMMISSION - STOCK_STAMP_TAX)

    # ============================================================
    # 状态持久化
    # ============================================================

    def save_state(self):
        """保存当前状态到文件"""
        state = {
            "cash": self.cash,
            "positions": {
                sym: {**pos, "extra": pos.get("extra", {})}
                for sym, pos in self.positions.items()
            },
            "trades": self.trades[-100:],  # 只保留最近100条
            "n_winning": self.n_winning,
            "n_total": self.n_total,
            "candidates": self.candidates[-50:],  # 只保留最近50天的候选
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"状态已保存: {self.state_file}")

    def load_state(self) -> bool:
        """从文件恢复状态，返回是否成功"""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.cash = state["cash"]
            self.positions = state["positions"]
            self.trades = state.get("trades", [])
            self.n_winning = state.get("n_winning", 0)
            self.n_total = state.get("n_total", 0)
            self.candidates = state.get("candidates", [])
            logger.info(f"状态已恢复: 现金{self.cash:,.0f}, 持仓{len(self.positions)}只")
            return True
        except Exception as e:
            logger.warning(f"状态恢复失败: {e}")
            return False

    def reset(self):
        """重置状态"""
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.market_snapshots = []
        self.n_winning = 0
        self.n_total = 0
        self.candidates = []
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("状态已重置")

    # ============================================================
    # 回测
    # ============================================================

    def run_backtest(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        运行回测

        Args:
            strategy: 策略实例（实现 Strategy 接口）
            df: 预加载的全市场数据（date, symbol, open, high, low, close, volume 等）
            start_date: 回测开始日期（可选，从 df 中过滤）
            end_date: 回测结束日期（可选）
        """
        self._strategy = strategy
        self.reset()

        # 过滤日期范围
        if start_date:
            df = df[df["date"] >= start_date].copy()
        if end_date:
            df = df[df["date"] <= end_date].copy()

        dates = sorted(df["date"].unique())
        logger.info(f"回测区间: {dates[0]} ~ {dates[-1]}（{len(dates)}个交易日）")

        for i, date in enumerate(dates):
            self._on_day(date, df, dates)

            # 每日快照
            total_value = self._calc_total_value(df, date)
            snap = MarketSnapshot(
                date=date,
                cash=self.cash,
                total_value=total_value,
                n_positions=len(self.positions),
                total_return=(total_value / self.initial_cash - 1) * 100,
            )
            self.market_snapshots.append(snap.__dict__)

            if (i + 1) % 20 == 0:
                logger.info(
                    f"  {date} ({i+1}/{len(dates)}): "
                    f"持仓{len(self.positions)}只, "
                    f"总值{total_value/10000:.1f}万({snap.total_return:+.1f}%)"
                )

        # 最终报告
        self._print_summary()

    # ============================================================
    # 模拟盘
    # ============================================================

    def run_simulate(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        target_date: str,
        dates: list = None,
    ):
        """
        运行单日模拟盘（从状态文件恢复，继续交易）

        Args:
            strategy: 策略实例
            df: 预加载数据
            target_date: 目标日期
            dates: 完整交易日列表（用于计算 next_date）
        """
        self._strategy = strategy

        # 尝试恢复状态
        if not self.load_state():
            self.reset()

        # 传入完整日期列表，以便计算 next_date
        if dates is None:
            dates = sorted(df["date"].unique())
        self._on_day(target_date, df, dates=dates)
        self.save_state()

        total_value = self._calc_total_value(df, target_date)
        logger.info(
            f"模拟完成: {target_date} | "
            f"现金{self.cash:,.0f} | "
            f"持仓{len(self.positions)}只 | "
            f"总值{total_value/10000:.1f}万"
        )

    # ============================================================
    # 核心逻辑
    # ============================================================

    def _on_day(self, date: str, df: pd.DataFrame, dates: Optional[List[str]] = None):
        """
        每日处理逻辑：
          1. 更新持仓信息
          2. 处理出场
          3. 处理入场
        """
        daily = df[df["date"] == date].copy()
        if daily.empty:
            return

        # 计算次日日期（用于成交日期）
        next_date = None
        if dates:
            idx = dates.index(date) if date in dates else -1
            if 0 <= idx < len(dates) - 1:
                next_date = dates[idx + 1]
        else:
            # 如果没有传入 dates，从 df 中获取所有日期并计算 next_date
            all_dates = sorted(df["date"].unique())
            if date in all_dates:
                idx = all_dates.index(date)
                if idx < len(all_dates) - 1:
                    next_date = all_dates[idx + 1]

        market = {
            "date": date,
            "cash": self.cash,
            "total_value": self._calc_total_value(df, date),
            "next_date": next_date,
        }

        # ── 1. 更新持仓状态 ──
        for sym, pos in list(self.positions.items()):
            row = daily[daily["symbol"] == sym]
            if row.empty:
                continue
            r = row.iloc[0]
            pos["latest_price"] = r.get("close", pos.get("avg_cost", 0))
            # 存储技术指标到 extra（供报告使用）
            if "extra" not in pos:
                pos["extra"] = {}
            pos["extra"]["rsi"] = r.get("rsi_14")
            pos["extra"]["macd_state"] = _infer_macd_state(r)
            pos["extra"]["kdj_state"] = _infer_kdj_state(r)
            pos["extra"]["industry"] = r.get("industry", "")
            pos["extra"]["name"] = r.get("name", "")
            _strategy_on_tick = getattr(self._strategy, 'on_tick', None)
            if _strategy_on_tick is not None:
                try:
                    _strategy_on_tick(r, pos, market)
                except TypeError:
                    try:
                        _strategy_on_tick(r, pos)  # 兼容2参数策略
                    except Exception:
                        pass
            else:
                # 策略无 on_tick，框架使用默认实现
                pos['days_held'] = pos.get('days_held', 0) + 1

        # ── 2. 处理出场 ──
        self._process_sells(daily, market)

        # ── 3. 处理入场 ──
        self._process_buys(df, daily, market)

    def _process_sells(self, daily: pd.DataFrame, market: dict):
        """处理所有持仓的出场"""
        for sym in list(self.positions.keys()):
            row = daily[daily["symbol"] == sym]
            if row.empty:
                continue
            r = row.iloc[0]
            pos = self.positions[sym]

            should_sell, reason = self._strategy.should_sell(r, pos, market)
            if not should_sell:
                continue

            # 卖出价：回测用次日开盘，模拟盘（单日）用当日收盘
            exec_price = r.get("next_open")
            if pd.isna(exec_price):
                exec_price = r.get("close", pos["avg_cost"])

            exec_price = exec_price * (1 - self.slippage_pct)  # 滑点

            # 次日涨跌停检查（用 next_limit_up/down 字段，次日涨停则无法卖出）
            if r.get("next_limit_up", False) or r.get("next_limit_down", False):
                continue  # 次日涨跌停，无法卖出，跳过

            # 计算收益（用持仓时的成本，不扣税费）
            pnl = (exec_price - pos["avg_cost"]) * pos["qty"]
            pnl_pct = (exec_price - pos["avg_cost"]) / pos["avg_cost"] * 100

            # 扣除税费（A 股：印花税只在卖出时收取）
            is_etf = pos.get("is_etf", False)
            sell_value = self._sell_proceeds(exec_price * pos["qty"], is_etf)
            self.cash += sell_value

            # 记录
            trade = Trade(
                date=market.get("next_date", market["date"]),
                symbol=sym,
                action="sell",
                price=exec_price,
                qty=pos["qty"],
                pnl=pnl,
                signal_type=pos.get("signal_type", ""),
                reason=reason,
                pnl_pct=pnl_pct,
            )
            self.trades.append(trade.__dict__)
            self.n_total += 1
            if pnl > 0:
                self.n_winning += 1

            del self.positions[sym]
            logger.info(f"  卖出 {sym} @{exec_price:.2f} ({reason}, {pnl_pct:+.1f}%)")

    def _process_buys(self, full_df: pd.DataFrame, daily: pd.DataFrame, market: dict):
        """处理新入场"""
        if len(self.positions) >= self.max_positions:
            return

        # 查询大盘仓位上限
        position_limit = 1.0
        if self.market_regime_filter:
            regime_info = self.market_regime_filter.get_regime(market["date"])
            position_limit = regime_info["position_limit"]
            logger.debug(
                f"  大盘状态: {regime_info['regime']} | "
                f"{regime_info['signal']} | "
                f"仓位上限: {position_limit:.0%} | "
                f"RSI14: {regime_info['rsi14']:.1f}"
            )

        # 剩余可用现金 = 总资产×regime比例 - 持仓市值
        total_value = self.cash + sum(
            p.get("qty", 0) * p.get("latest_price", p.get("avg_cost", 0))
            for p in self.positions.values()
        )
        remaining_slots = self.max_positions - len(self.positions)
        if remaining_slots <= 0:
            return
        position_budget = total_value * position_limit
        used_budget = sum(
            p.get("qty", 0) * p.get("latest_price", p.get("avg_cost", 0))
            for p in self.positions.values()
        )
        remaining_cash = max(0.0, position_budget - used_budget)
        per_stock_budget = remaining_cash / remaining_slots

        # 策略选股
        candidates = self._strategy.filter_buy(daily, market.get("date"))
        if candidates.empty:
            return

        scored = self._strategy.score(candidates)
        if scored.empty:
            return

        # TOP 5 候选
        top5 = scored.nlargest(5, "score") if len(scored) > 5 else scored

        # ---- 第一遍：构建所有候选信号记录 ----
        today_candidates = []
        for _, row in top5.iterrows():
            sym = row["symbol"]
            day_row = daily[daily["symbol"] == sym]
            if day_row.empty:
                continue
            r = day_row.iloc[0]

            macd = r.get("macd", 0) or 0
            macd_sig = r.get("macd_signal", 0) or 0
            macd_prev = macd
            try:
                prev_df = full_df[(full_df["symbol"] == sym) & (full_df["date"] < market["date"])]
                if not prev_df.empty:
                    macd_prev = prev_df.iloc[-1].get("macd", 0) or 0
            except:
                pass
            macd_cross = (macd > macd_sig) and (macd_prev <= macd_sig)
            macd_state = "金叉" if macd_cross else ("红柱" if macd > 0 else "绿柱")

            kdj_j = r.get("kdj_j", 0) or 0
            kdj_state = "高位" if kdj_j > 80 else ("低位" if kdj_j < 20 else "中性")

            cand = {
                "date": market["date"],
                "symbol": sym,
                "score": round(float(row.get("score", 0)), 2),
                "signal_type": str(row.get("signal_type", "")),
                "wave_state": str(row.get("wave_state", "")),
                "wave_trend": str(row.get("wave_trend", "")),
                "iron_law_1": bool(row.get("iron_law_1", False)),
                "iron_law_2": bool(row.get("iron_law_2", False)),
                "iron_law_3": bool(row.get("iron_law_3", False)),
                "all_verified": bool(row.get("all_verified", False)),
                "rsi": round(float(r.get("rsi_14", 0) or 0), 1),
                "cci": round(float(r.get("cci_20", 0) or 0), 1),
                "macd_state": macd_state,
                "kdj_state": kdj_state,
                "industry": str(r.get("industry", "")),
                "name": str(r.get("name", "")),
                "close": round(float(r.get("close", 0) or 0), 2),
                "limit_up": bool(r.get("limit_up", False)),
                "limit_down": bool(row.get("limit_down", False)),
                "status": "pending",
                "reason": "",
            }
            today_candidates.append(cand)

        # ---- 第二遍：标记未买入原因 ----
        for cand in today_candidates:
            sym = cand["symbol"]
            if sym in self.positions:
                cand["status"] = "X0 - 已持仓"
                cand["reason"] = "已持有该股票"
                continue
            day_row = daily[daily["symbol"] == sym]
            if day_row.empty:
                cand["status"] = "X1 - 数据异常"
                cand["reason"] = "行情数据缺失"
                continue
            r = day_row.iloc[0]
            if r.get("limit_up", False):
                cand["status"] = "X2 - 涨停"
                cand["reason"] = "涨停无法买入"
                continue
            if r.get("limit_down", False):
                cand["status"] = "X3 - 跌停"
                cand["reason"] = "跌停风险"
                continue
            exec_price = r.get("next_open")
            if pd.isna(exec_price):
                exec_price = r.get("open", 0) or 0
            exec_price = exec_price * (1 + self.slippage_pct)
            if remaining_slots <= 0:
                cand["status"] = "X4 - 已满仓"
                cand["reason"] = f"仓位已满({len(self.positions)}/{self.max_positions})"
                continue
            if exec_price <= 0 or self.cash < exec_price * 100:
                cand["status"] = "X5 - 资金不足"
                cand["reason"] = "可用资金不足"
                continue

        # ---- 第三遍：执行买入 ----
        to_buy = []
        buy_count = 0
        for cand in today_candidates:
            if cand["status"] != "pending":
                continue
            sym = cand["symbol"]
            if buy_count >= remaining_slots:
                cand["status"] = "X4 - 已满仓"
                cand["reason"] = f"仓位已满({len(self.positions)}/{self.max_positions})"
                continue
            day_row = daily[daily["symbol"] == sym]
            if day_row.empty:
                continue
            r = day_row.iloc[0]
            exec_price = r.get("next_open")
            if pd.isna(exec_price):
                exec_price = r.get("open", 0) or 0
            exec_price = exec_price * (1 + self.slippage_pct)
            to_buy.append((sym, cand, r, exec_price))
            buy_count += 1

        if not to_buy:
            self.candidates = [c for c in self.candidates if c["date"] != market["date"]]
            self.candidates.extend(today_candidates)
            return

        per_stock_cash = per_stock_budget  # 每只股票可用金额

        logger.info(
            f"[仓位预算] 总资产={total_value/1e4:.1f}万 | "
            f"Regime仓位={position_limit:.0%} | "
            f"理论预算={position_budget/1e4:.1f}万 | "
            f"持仓市值={used_budget/1e4:.1f}万 | "
            f"剩余现金={remaining_cash/1e4:.1f}万 | "
            f"可用slot={remaining_slots} | "
            f"每只预算={per_stock_budget/1e4:.1f}万 | "
            f"实际现金={self.cash/1e4:.1f}万"
        )

        for sym, cand, r, exec_price in to_buy:
            buy_qty = int(per_stock_cash / exec_price)
            buy_qty = (buy_qty // 100) * 100
            if buy_qty < 100:
                cand["status"] = "X5 - 资金不足"
                cand["reason"] = f"资金不足(买入{buy_qty}股<1手)"
                continue

            # 判断ETF（优先用category，否则按代码前缀）
            category = str(cand.get("category", r.get("category", "")))
            is_etf = self._is_etf(sym, category)

            # A股税费：买入扣佣金，卖出还要扣印花税
            cost = self._buy_cost(buy_qty * exec_price, is_etf)
            if cost > self.cash:
                cand["status"] = "X5 - 资金不足"
                cand["reason"] = "资金不足"
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
                "is_etf": is_etf,
                "signal_type": cand.get("signal_type", ""),
                "extra": {},
            }
            for c in today_candidates:
                if c["symbol"] == sym:
                    c["status"] = "OK - 买入"
                    break

            trade = Trade(
                date=market.get("next_date", market["date"]),
                symbol=sym,
                action="buy",
                price=exec_price,
                qty=buy_qty,
                signal_type=cand.get("signal_type", ""),
            )
            self.trades.append(trade.__dict__)
            logger.info(
                f"  买入 {sym} @{exec_price:.2f} x {buy_qty}股 "
                f"(评分:{cand.get('score', 0):.0f})"
            )

        # ---- 合并候选 ----
        self.candidates = [c for c in self.candidates if c["date"] != market["date"]]
        self.candidates.extend(today_candidates)

    def _calc_total_value(self, df: pd.DataFrame, date: str) -> float:
        """计算当日总资产"""
        total = self.cash
        daily = df[df["date"] == date]
        for sym, pos in self.positions.items():
            row = daily[daily["symbol"] == sym]
            if not row.empty:
                price = row.iloc[0].get("close", pos.get("avg_cost", 0))
                total += price * pos["qty"]
            else:
                total += pos.get("latest_price", pos["avg_cost"]) * pos["qty"]
        return total

    def _print_summary(self):
        """打印回测汇总"""
        closed = [t for t in self.trades if t["action"] == "sell"]
        total_pnl = sum(t["pnl"] for t in closed)
        win_rate = self.n_winning / max(self.n_total, 1) * 100

        final_value = self.cash + sum(
            pos["latest_price"] * pos["qty"]
            for pos in self.positions.values()
        )
        total_return = (final_value / self.initial_cash - 1) * 100

        # 最大回撤
        values = [s["total_value"] for s in self.market_snapshots]
        peak = self.initial_cash
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        print(f"\n{'='*50}")
        print(f"  回测汇总")
        print(f"{'='*50}")
        print(f"  初始资金:  {self.initial_cash:>12,.0f}")
        print(f"  最终价值:  {final_value:>12,.0f} ({total_return:+.2f}%)")
        print(f"  最大回撤:  {max_dd:>12.2f}%")
        print(f"  总交易数:  {self.n_total:>12} 笔")
        print(f"  胜率:      {win_rate:>12.1f}%")
        print(f"  累计盈亏:  {total_pnl:>12,.0f}")
        print(f"  当前持仓:  {len(self.positions):>12} 只")
        print(f"{'='*50}")

        # 出场原因分布
        if closed:
            from collections import Counter
            reasons = Counter(t.get("reason", "") for t in closed)
            print(f"\n出场原因分布:")
            for reason, count in reasons.most_common():
                print(f"  {reason or 'unknown'}: {count}笔")
