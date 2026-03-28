# Strategies Plugin System
# 策略插件系统

from .base import BaseStrategy
from .score_strategy import ScoreStrategy
from .resonance_strategy import ResonanceStrategy
from .wavechan_strategy import WavechanStrategy
from .qlib_strategy import QlibStrategy
from .resonance_v2_strategy import ResonanceV2Strategy
from .score_wavechan_combo import ScoreWaveChanComboStrategy
from .wavechan_selector import WaveChanSelector

# 策略注册表，方便通过名称选择
STRATEGY_REGISTRY = {
    'score': ScoreStrategy,
    'resonance': ResonanceStrategy,
    'resonance_v2': ResonanceV2Strategy,  # 共振策略v2（优化版）
    'wavechan': WavechanStrategy,
    'qlib': QlibStrategy,  # Qlib增强策略
    'score_wavechan_combo': ScoreWaveChanComboStrategy,  # Score + WaveChan 组合策略（方向A）
    'wavechan_selector': WaveChanSelector,  # WaveChan纯评分选股策略
}

__all__ = ['BaseStrategy', 'ScoreStrategy', 'ResonanceStrategy', 'WavechanStrategy', 'QlibStrategy', 'ResonanceV2Strategy', 'ScoreWaveChanComboStrategy', 'WaveChanSelector', 'STRATEGY_REGISTRY']
