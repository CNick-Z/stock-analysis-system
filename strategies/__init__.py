# Strategies Plugin System
# 策略插件系统

from .base import BaseStrategy
from .score_strategy import ScoreStrategy
from .resonance_strategy import ResonanceStrategy

# 策略注册表，方便通过名称选择
STRATEGY_REGISTRY = {
    'score': ScoreStrategy,
    'resonance': ResonanceStrategy,
}

__all__ = ['BaseStrategy', 'ScoreStrategy', 'ResonanceStrategy', 'STRATEGY_REGISTRY']
