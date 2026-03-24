# -*- coding: utf-8 -*-
"""
止盈止损独立配置模块
===========================
与策略信号分离，可被多个策略复用

设计理念：
- 配置驱动：通过配置文件设置规则，无需修改策略代码
- 分档止盈：根据涨幅动态调整回撤比例
- 追踪止损：跌破最高点时触发
- 时间止损：持仓超期自动退出

使用示例：
    from config.risk_config import TieredTakeProfit, TrailingStop, TimeStop
    
    # 分档止盈
    ttp = TieredTakeProfit()
    should_sell, reason = ttp.check(current_price, entry_price, peak_price)
    
    # 追踪止损
    ts = TrailingStop(trailing_pct=0.05)
    should_sell, reason = ts.check(current_price, peak_price, entry_price)
    
    # 时间止损
    tst = TimeStop(max_hold_days=20)
    should_sell, reason = tst.check(hold_days)
"""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


# ============================================================
# 分档止盈配置
# ============================================================

# 分档止盈规则：(涨幅下限, 涨幅上限) -> {mode: 模式, sell_pct: 卖出仓位比例, withdraw_pct: 允许回撤比例}
TIERED_TAKE_PROFIT_RULES: Dict[Tuple[float, float], Dict[str, Any]] = {
    # 优化版：让利润奔跑，大涨后放宽止盈
    (0.00, 0.05): {'mode': 'breakeven', 'sell_pct': 0.30, 'withdraw_pct': 0.003, 'desc': '保本微盈卖30%'},
    (0.05, 0.10): {'mode': 'trailing', 'sell_pct': 0.50, 'withdraw_pct': 0.50, 'desc': '允许回撤50%卖50%'},
    (0.10, 0.20): {'mode': 'trailing', 'sell_pct': 0.50, 'withdraw_pct': 0.35, 'desc': '允许回撤35%卖50%'},
    (0.20, 0.30): {'mode': 'trailing', 'sell_pct': 0.70, 'withdraw_pct': 0.25, 'desc': '允许回撤25%卖70%'},
    (0.30, 0.50): {'mode': 'trailing', 'sell_pct': 0.80, 'withdraw_pct': 0.30, 'desc': '允许回撤30%卖80%'},
    (0.50, 999.0): {'mode': 'trailing', 'sell_pct': 1.0, 'withdraw_pct': 0.40, 'desc': '允许回撤40%卖全部'},
}

# 默认止损配置
DEFAULT_STOP_LOSS_PCT = -0.05  # 固定止损 -5%

# 时间止损配置
DEFAULT_MAX_HOLD_DAYS = 20  # 默认最大持仓20天


class TieredTakeProfit:
    """
    分档止盈器
    -----------
    根据持仓涨幅动态调整止盈策略
    
    逻辑：
    - 涨 ≤5%：保本微盈（赚0.3%）
    - 涨 5-10%：允许回撤50%卖出
    - 涨 10-20%：允许回撤30%卖出
    - 涨 >20%：允许回撤10%卖出
    """
    
    def __init__(self, rules: Dict[Tuple[float, float], Dict[str, Any]] = None):
        self.rules = rules or TIERED_TAKE_PROFIT_RULES
    
    def get_tier(self, gain_pct: float) -> Dict[str, Any]:
        """根据涨幅获取对应档位配置"""
        for (low, high), config in self.rules.items():
            if low <= gain_pct < high:
                return config
        # 默认返回最后一档
        return self.rules[(0.20, 999.0)]
    
    def check(self, current_price: float, entry_price: float, peak_price: float, hold_days: int = 0) -> Tuple[bool, str, float]:
        """
        检查是否触发止盈
        
        Args:
            current_price: 当前价格
            entry_price: 入场价格
            peak_price: 持仓期间最高价
            hold_days: 已持仓天数
        
        Returns:
            (should_sell, reason, sell_pct): 是否应该卖出, 原因, 卖出仓位比例(0-1)
        
        核心逻辑：
        - 持仓 < 20天：不触发止盈（让利润奔跑，0-20天是主要亏损期）
        - 20天+：根据涨幅分档止盈，每次只卖一部分
        """
        if entry_price <= 0 or current_price <= 0 or peak_price <= 0:
            return False, "", 0.0
        
        # 关键：根据最高点涨幅确定档位，而非当前涨幅
        peak_gain_pct = (peak_price - entry_price) / entry_price
        tier = self.get_tier(peak_gain_pct)
        
        if tier['mode'] == 'breakeven':
            # 保本模式：只要不跌破保本线就不卖
            breakeven_price = entry_price * 1.003
            if current_price < breakeven_price:
                return False, "", 0.0
            if current_price < breakeven_price * (1 - tier['withdraw_pct']):
                return True, f"take_profit:保本触发(现价{current_price:.2f})", 1.0
            return False, "", 0.0
        
        elif tier['mode'] == 'partial':
            # 分批卖出模式：涨到目标位就卖指定比例
            # 触发条件：当前价 >= 峰值 * (1 - withdraw_pct)
            trailing_trigger = peak_price * (1 - tier['withdraw_pct'])
            if current_price >= trailing_trigger:
                return True, f"take_profit:{tier['desc']}(最高{peak_price:.2f}, 现{current_price:.2f})", tier['sell_pct']
            return False, "", 0.0
        
        elif tier['mode'] == 'trailing':
            # 移动止盈模式：跌破最高点 * (1 - 回撤比例) 则卖
            trailing_trigger = peak_price * (1 - tier['withdraw_pct'])
            if current_price < trailing_trigger:
                sell_pct = tier.get('sell_pct', 1.0)  # 默认为全卖
                return True, f"take_profit:{tier['desc']}(最高{peak_price:.2f}, 现{current_price:.2f})", sell_pct
            return False, "", 0.0
        
        return False, "", 0.0
    
    def get_status(self, current_price: float, entry_price: float, peak_price: float) -> str:
        """获取当前止盈状态（用于调试/显示）"""
        if entry_price <= 0 or current_price <= 0:
            return "无效价格"
        
        gain_pct = (current_price - entry_price) / entry_price * 100
        tier = self.get_tier(gain_pct)
        trailing_trigger = peak_price * (1 - tier['withdraw_pct'])
        
        return f"涨{gain_pct:.1f}% | 档位:{tier['desc']} | 触发线:{trailing_trigger:.2f}"


class TrailingStop:
    """
    追踪止损器
    -----------
    跌破最高点回撤比例时触发
    
    特点：
    - 不关心涨跌，只关心从最高点回撤了多少
    - 会随价格上涨不断抬高止损线
    """
    
    def __init__(self, trailing_pct: float = 0.05):
        """
        Args:
            trailing_pct: 回撤比例（默认5%）
        """
        self.trailing_pct = trailing_pct
    
    def check(self, current_price: float, peak_price: float, entry_price: float) -> Tuple[bool, str]:
        """
        检查是否触发追踪止损
        
        Args:
            current_price: 当前价格
            peak_price: 持仓期间最高价
            entry_price: 入场价格
        
        Returns:
            (should_sell, reason): 是否应该卖出, 原因
        """
        if peak_price <= 0 or current_price <= 0:
            return False, ""
        
        # 触发价：最高价 * (1 - 回撤比例)
        trigger_price = peak_price * (1 - self.trailing_pct)
        
        # 如果当前价低于触发价，且低于入场价，则触发止损
        if current_price < trigger_price and current_price < entry_price:
            drawdown = (peak_price - current_price) / peak_price * 100
            return True, f"trailing_stop:回撤{drawdown:.1f}%触发(高{peak_price:.2f}, 现{current_price:.2f})"
        
        return False, ""
    
    def get_trigger_price(self, peak_price: float) -> float:
        """获取当前触发价格"""
        return peak_price * (1 - self.trailing_pct)


class TimeStop:
    """
    时间止损器
    -----------
    持仓超期自动退出
    """
    
    def __init__(self, max_hold_days: int = None):
        """
        Args:
            max_hold_days: 最大持仓天数
        """
        self.max_hold_days = max_hold_days or DEFAULT_MAX_HOLD_DAYS
    
    def check(self, hold_days: int, current_price: float = None, entry_price: float = None) -> Tuple[bool, str]:
        """
        检查是否触发时间止损
        
        Args:
            hold_days: 已持仓天数
            current_price: 当前价格（可选）
            entry_price: 入场价格（可选）
        
        Returns:
            (should_sell, reason): 是否应该卖出, 原因
        """
        if hold_days >= self.max_hold_days:
            pnl_info = ""
            if current_price and entry_price:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl_info = f", 盈亏{pnl_pct:.1f}%"
            return True, f"time_stop:持仓超期({hold_days}天){pnl_info}"
        return False, ""


class RiskManager:
    """
    风险管理器
    -----------
    整合分档止盈、追踪止损、时间止损
    
    使用方式：
        rm = RiskManager()
        should_sell, reason = rm.check(
            current_price=8.5,
            entry_price=8.0,
            peak_price=8.8,
            hold_days=15
        )
    """
    
    def __init__(
        self,
        use_tiered_tp: bool = True,
        use_trailing_stop: bool = True,
        use_time_stop: bool = True,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        max_hold_days: int = DEFAULT_MAX_HOLD_DAYS
    ):
        self.use_tiered_tp = use_tiered_tp
        self.use_trailing_stop = use_trailing_stop
        self.use_time_stop = use_time_stop
        self.stop_loss_pct = stop_loss_pct
        self.tiered_tp = TieredTakeProfit() if use_tiered_tp else None
        self.trailing_stop = TrailingStop() if use_trailing_stop else None
        self.time_stop = TimeStop(max_hold_days) if use_time_stop else None
    
    def check(
        self,
        current_price: float,
        entry_price: float,
        peak_price: float,
        hold_days: int
    ) -> Tuple[bool, str, float]:
        """
        综合风控检查
        
        检查顺序：
        1. 固定止损（全卖）
        2. 分档止盈（部分卖）
        3. 追踪止损（部分卖）
        4. 时间止损（全卖）
        
        Returns:
            (should_sell, reason, sell_pct): 是否应该卖出, 原因, 卖出仓位比例(0-1)
        """
        # 1. 固定止损（全卖）
        if current_price <= entry_price * (1 + self.stop_loss_pct):
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return True, f"stop_loss:亏损{pnl_pct:.1f}%触发", 1.0
        
        # 2. 分档止盈（部分卖）
        if self.tiered_tp:
            should_sell, reason, sell_pct = self.tiered_tp.check(current_price, entry_price, peak_price, hold_days)
            if should_sell:
                return True, reason, sell_pct
        
        # 3. 追踪止损（部分卖）
        if self.trailing_stop:
            should_sell, reason = self.trailing_stop.check(current_price, peak_price, entry_price)
            if should_sell:
                return True, reason, 1.0  # 追踪止损全卖
        
        # 4. 时间止损（全卖）
        if self.time_stop:
            should_sell, reason = self.time_stop.check(hold_days, current_price, entry_price)
            if should_sell:
                return True, reason, 1.0
        
        return False, "", 0.0
    
    def get_status(
        self,
        current_price: float,
        entry_price: float,
        peak_price: float,
        hold_days: int
    ) -> Dict[str, Any]:
        """获取当前风控状态（用于调试/显示）"""
        gain_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        status = {
            '持仓天数': f"{hold_days}天",
            '浮盈': f"{gain_pct:.1f}%",
            '最高价': f"{peak_price:.2f}",
            '当前价': f"{current_price:.2f}",
        }
        
        if self.tiered_tp:
            status['止盈状态'] = self.tiered_tp.get_status(current_price, entry_price, peak_price)
        
        if self.trailing_stop:
            status['追踪止损'] = f"触发线{self.trailing_stop.get_trigger_price(peak_price):.2f}"
        
        if self.time_stop:
            status['剩余天数'] = f"{self.time_stop.max_hold_days - hold_days}天"
        
        return status


# ============================================================
# 预设配置
# ============================================================

# 保守策略：快速锁定利润
CONSERVATIVE_CONFIG = {
    'use_tiered_tp': True,
    'use_trailing_stop': True,
    'use_time_stop': True,
    'stop_loss_pct': -0.03,  # 3%止损
    'max_hold_days': 15,
}

# 激进策略：让利润奔跑
AGGRESSIVE_CONFIG = {
    'use_tiered_tp': True,
    'use_trailing_stop': True,
    'use_time_stop': True,
    'stop_loss_pct': -0.05,  # 5%止损
    'max_hold_days': 30,
}

# 平衡策略
BALANCED_CONFIG = {
    'use_tiered_tp': True,
    'use_trailing_stop': True,
    'use_time_stop': True,
    'stop_loss_pct': -0.05,  # 5%止损
    'max_hold_days': 20,
}


if __name__ == '__main__':
    # 简单测试
    print("=== 止盈止损模块测试 ===\n")
    
    # 测试场景
    entry_price = 10.0  # 入场价
    peak_price = 11.5   # 最高价
    current_price = 11.0  # 当前价
    hold_days = 10
    
    rm = RiskManager()
    
    should_sell, reason = rm.check(current_price, entry_price, peak_price, hold_days)
    print(f"持仓情况: 入场{entry_price}, 最高{peak_price}, 现价{current_price}")
    print(f"是否卖出: {should_sell}, 原因: {reason}")
    print(f"状态: {rm.get_status(current_price, entry_price, peak_price, hold_days)}")
