"""
仓位管理器 — Trader 方案
==========================
参考 DynamicPositionAdjuster 的接口风格

方案：连续止损触发降仓，连续盈利恢复满仓
  - 连续 N1 次止损 → 降仓至 X%
  - 连续 N2 次止损 → 降仓至 Y%
  - 连续 Z1 次盈利 → 恢复 +25%
  - 连续 Z2 次盈利 → 恢复 100%
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PositionSizeConfig:
    """仓位管理配置"""
    n1: int = 3          # 触发第一次降仓的连续止损次数
    n2: int = 5          # 触发第二次降仓的连续止损次数
    x: float = 0.50      # 第一次降仓后的目标仓位（50%）
    y: float = 0.25      # 第二次降仓后的目标仓位（25%）
    z1: int = 2          # 触发第一次恢复的连续盈利次数
    z2: int = 4          # 触发第二次恢复的连续盈利次数
    step: float = 0.25   # 每次恢复加仓的幅度（25%）
    max_position: float = 1.0   # 最高仓位上限
    min_position: float = 0.10  # 最低仓位下限（保险）


@dataclass
class TradeResult:
    """交易结果记录"""
    date: str
    pnl_pct: float      # 盈亏比例（正=盈利，负=亏损）
    is_stop_loss: bool = False  # 是否为止损出局


class SequentialStopLossSizer:
    """
    连续止损仓位管理器

    核心思想：不预判市场，只看自己的交易结果。
    连续亏损本身就是最真实的信号。

    使用方式：
        sizer = SequentialStopLossSizer()
        for trade in trades:
            sizer.record_trade(trade)
        current_position = sizer.get_position()
    """

    def __init__(self, config: Optional[PositionSizeConfig] = None):
        self.config = config or PositionSizeConfig()
        self.consecutive_losses = 0     # 连续亏损次数
        self.consecutive_wins = 0      # 连续盈利次数
        self.current_position = 1.0    # 当前仓位（默认满仓）
        self.trade_history: List[TradeResult] = []

    def record_trade(self, trade: TradeResult):
        """记录一笔交易结果，用于更新计数器"""
        self.trade_history.append(trade)

        if trade.is_stop_loss:
            # 止损：连续亏损+1，连续盈利归零
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        elif trade.pnl_pct > 0:
            # 盈利：连续盈利+1，连续亏损归零
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            # 持平（0 < pnl <= 0）：双方都归零，不累积
            self.consecutive_losses = 0
            self.consecutive_wins = 0

        self._update_position()

    def _update_position(self):
        """根据计数器计算目标仓位"""
        c = self.config

        # 止损降仓（优先级高，同时只生效一个）
        if self.consecutive_losses >= c.n2:
            # 连续5次止损 → 25%
            self.current_position = c.y
        elif self.consecutive_losses >= c.n1:
            # 连续3次止损 → 50%
            self.current_position = c.x
        else:
            # 正常状态：根据连续盈利恢复仓位
            if self.consecutive_wins >= c.z2:
                # 连续4次盈利 → 100%
                self.current_position = c.max_position
            elif self.consecutive_wins >= c.z1:
                # 连续2次盈利 → +25%（分步恢复）
                self.current_position = min(
                    c.max_position,
                    self.current_position + c.step
                )
            # else: 仓位不变

        # 保险：不超过上下限
        self.current_position = max(c.min_position, min(c.max_position, self.current_position))

    def get_position(self) -> float:
        """获取当前目标仓位比例"""
        return self.current_position

    def get_state(self) -> dict:
        """获取当前状态（用于调试）"""
        return {
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "current_position": f"{self.current_position:.0%}",
            "n_trades": len(self.trade_history),
        }

    def reset(self):
        """重置状态"""
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_position = 1.0
        self.trade_history.clear()

    def __repr__(self):
        s = self.get_state()
        return (f"SequentialStopLossSizer("
                f"连亏={s['consecutive_losses']} "
                f"连盈={s['consecutive_wins']} "
                f"仓位={s['current_position']} "
                f"交易={s['n_trades']})")


# ── 快速测试 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sizer = SequentialStopLossSizer()

    # 模拟：连续3次止损
    print("=== 连续3次止损 ===")
    for i in range(3):
        t = TradeResult(date=f"day{i}", pnl_pct=-0.04, is_stop_loss=True)
        sizer.record_trade(t)
        print(f"  {t}: 仓位 {sizer.get_position():.0%}")

    # 再来2次止损（累计5次）
    print("=== 再来2次（累计5次）===")
    for i in range(3, 5):
        t = TradeResult(date=f"day{i}", pnl_pct=-0.04, is_stop_loss=True)
        sizer.record_trade(t)
        print(f"  {t}: 仓位 {sizer.get_position():.0%}")

    # 2次盈利 → 恢复+25%
    print("=== 连续2次盈利 ===")
    for i in range(5, 7):
        t = TradeResult(date=f"day{i}", pnl_pct=0.05, is_stop_loss=False)
        sizer.record_trade(t)
        print(f"  {t}: 仓位 {sizer.get_position():.0%}")

    # 再2次盈利 → 恢复100%
    print("=== 再来2次（累计4次）===")
    for i in range(7, 9):
        t = TradeResult(date=f"day{i}", pnl_pct=0.05, is_stop_loss=False)
        sizer.record_trade(t)
        print(f"  {t}: 仓位 {sizer.get_position():.0%}")

    print(f"\n最终状态: {sizer}")
