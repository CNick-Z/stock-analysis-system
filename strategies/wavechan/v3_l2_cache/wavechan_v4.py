# -*- coding: utf-8 -*-
"""
WaveChan V4 增强工具包

独立函数，不继承原版代码，通过配置控制：
- compute_extended_fib() - 斐波那契多档
- detect_b_trap() - B 浪陷阱检测  
- calc_tight_w2_range() - W2 收紧区间

所有新功能通过 config 开关控制，默认关闭不影响现有 V3
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ========================
# 斐波那契多档计算
# ========================


def compute_extended_fib(w1_start: float, w1_end: float, w2_end: float) -> Dict[str, float]:
    """
    计算多档斐波那契位
    
    Args:
        w1_start: W1 起点
        w1_end: W1 终点
        w2_end: W2 终点（当前价格）
    
    Returns:
        dict: {fib_382, fib_500, fib_618, fib_1618, fib_2618}
    """
    w1_len = w1_end - w1_start
    
    if w1_len <= 0:
        return {}
    
    return {
        'fib_382': round(w1_end - w1_len * 0.382, 2),
        'fib_500': round(w1_end - w1_len * 0.500, 2),
        'fib_618': round(w1_end - w1_len * 0.618, 2),
        'fib_1618': round(w1_end + w1_len * 1.618, 2),
        'fib_2618': round(w1_end + w1_len * 2.618, 2),
    }


# ========================
# B 浪陷阱检测
# ========================


def detect_b_trap(
    wave_state: str,
    w1_is_impulse: bool,
    w1_internal_segments: int,
    volume_trend: str,
    config_enabled: bool = False
) -> bool:
    """
    检测 B 浪反弹陷阱
    
    DSA 波浪理论：B 浪反弹是"陷阱"
    - 大级别下跌趋势中，W1 不是 5 浪 = B 浪
    - 反弹弱、缩量、很快被打回来
    
    Args:
        wave_state: 当前波浪状态
        w1_is_impulse: W1 是否为推动浪（5浪）
        w1_internal_segments: W1 内部子浪数
        volume_trend: 'expanding' / 'shrinking' / 'stable'
        config_enabled: config.enable_b_trap_alert
    
    Returns:
        True = B 浪陷阱，建议减仓
    """
    if not config_enabled:
        return False
    
    # B 浪特征：下跌趋势中，W1 不是 5 浪
    if wave_state in ['w1_formed', 'w2_in_progress'] and not w1_is_impulse:
        # W1 只有 3 浪 = B 浪���弹
        if w1_internal_segments <= 3:
            logger.info(f"[V4] B浪陷阱检测: W1内部={w1_internal_segments}浪")
            return True
    
    return False


# ========================
# W2 收紧区间
# ========================


def calc_w2_range(use_tight: bool = False) -> Tuple[float, float]:
    """
    计算 W2 回撤区间
    
    Args:
        use_tight: 是否收紧（config.use_tight_w2_range）
    
    Returns:
        (min_retracement, max_retracement)
    """
    if use_tight:
        return (0.50, 0.65)  # V4 收紧区间
    else:
        return (0.40, 0.70)  # V3 原版区间


# ========================
# W5 顶背驰评分
# ========================


def calc_w5_divergence_penalty(
    w5_divergence_detected: bool,
    config_enabled: bool = False
) -> int:
    """
    W5 顶背驰评分
    
    DSA 波浪理论：第5浪量能 < 第3浪 = 顶背驰
    
    Args:
        w5_divergence_detected: 是否检测到 W5 顶背驰
        config_enabled: config.enable_w5_divergence_penalty
    
    Returns:
        评分调整值（负数 = 减分）
    """
    if not config_enabled:
        return 0
    
    if w5_divergence_detected:
        logger.info(f"[V4] W5顶背驰: -10分")
        return -10
    
    return 0


# ========================
# 便捷封装
# ========================


def create_v4_config(
    enable_b_trap: bool = False,
    use_extended_fib: bool = False,
    use_tight_w2: bool = False,
    use_fib_382_stop: bool = False,
    enable_w5_penalty: bool = False,
) -> dict:
    """
    创建 V4 配置（与 wavechan_config_v4 兼容）
    """
    return {
        'enable_b_trap_alert': enable_b_trap,
        'use_extended_fib': use_extended_fib,
        'use_tight_w2_range': use_tight_w2,
        'use_fib_382_stop': use_fib_382_stop,
        'enable_w5_divergence_penalty': enable_w5_penalty,
    }