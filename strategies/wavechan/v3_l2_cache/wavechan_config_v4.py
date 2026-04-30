# -*- coding: utf-8 -*-
"""
WaveChan V4 配置扩展

新增 Flag 默认关闭，不影响现有 V3
用法：改一行配置即可开启

    from wavechan_v4 import WaveCounterV4
    # 或
    from wavechan_strategy_v4 import WaveChanStrategyV4
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WaveChanConfigV4:
    """
    V4 扩展配置
    
    所有新功能默认 False，丝滑接入不影响现有 V3
    """
    # ====== B 浪陷阱警报 ======
    enable_b_trap_alert: bool = False
    """B浪反弹时返回警告（建议减仓）"""
    
    # ====== 斐波那契多档 ======
    use_extended_fib: bool = False
    """使用多档斐波那契止损位（fib_382/fib_500/fib_618）"""
    
    use_fib_382_stop: bool = False
    """止损用 fib_382 而非 fib_618（更紧止损）"""
    
    # ====== W2 精确区间 ======
    use_tight_w2_range: bool = False
    """W2 回撤缩窄至 50%-65%（原 40%-70%）"""
    
    # ====== 顶背驰评分 ======
    enable_w5_divergence_penalty: bool = False
    """W5 顶背驰时 scoring -10"""
    
    # ====== 共振加分 ======
    enable_wave_resonance_bonus: bool = False
    """W2+W4 同时满足时 +5 分"""
    
    def __post_init__(self):
        """打印配置状态"""
        enabled = [k for k, v in self.__dict__.items() 
                  if k.startswith('enable') and v]
        if enabled:
            logger.info(f"[V4] 开启功能: {enabled}")
        else:
            logger.info(f"[V4] 无额外功能，保持 V3 兼容")


# 全局默认配置（不影响现有）
DEFAULT_CONFIG_V4 = WaveChanConfigV4()