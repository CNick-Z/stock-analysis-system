# utils/wave_backtrack_corrector.py
# =============================================================
"""
波浪序列回溯修正器 v2.0
======================

基于正确波浪理论的回溯修正算法。

正确波浪序列：
  推动浪：Wave1 → Wave2 → Wave3 → Wave4 → Wave5
  调整浪：WaveA → WaveB → WaveC
  新周期：Wave1 → Wave2 → ...

  完整序列：Wave1→Wave2→Wave3→Wave4→Wave5→WaveA→WaveB→WaveC→Wave1

L2 缓存现状：
  有：Wave1-5, WaveA, WaveB
  缺少：WaveC（用 unknown 占位）

合法转换规则：
  Wave1 → Wave2
  Wave2 → Wave3
  Wave3 → Wave4
  Wave4 → Wave5
  Wave5 → WaveA      ← 修正：不是 Wave1
  WaveA → WaveB
  WaveB → WaveC 或 unknown（WaveC 缺失时用 unknown 占位）
  WaveC → Wave1（新周期）
  unknown → Wave1（作为 WaveC 的占位符）

测试案例：603700 (2020年5-7月)
  原始：Wave2 → Wave1 → Wave5
  期望：Wave2 → Wave3 → Wave4
"""
# =============================================================

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

# 波浪状态序号（用于序号比较）
STATE_ID = {
    # 推动浪 (1-5)
    'Wave1': 1, 'Wave2': 2, 'Wave3': 3, 'Wave4': 4, 'Wave5': 5,
    # 调整浪 (6-8)
    'WaveA': 6, 'WaveB': 7, 'WaveC': 8,
    # 兼容旧格式
    'w1_formed': 1, 'w2_formed': 2, 'w3_formed': 3,
    'w4_formed': 4, 'w5_formed': 5, 'w4_in_progress': 4,
    'initial': 0, 'unknown': 0, 'neutral': 0,
}

# 标准波浪序列
WAVE_SEQUENCE = ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5',
                  'WaveA', 'WaveB', 'WaveC', 'unknown']

# 合法转换表
VALID_TRANSITIONS = {
    'Wave1': ['Wave2'],
    'Wave2': ['Wave3'],
    'Wave3': ['Wave4'],
    'Wave4': ['Wave5'],
    'Wave5': ['WaveA'],      # 推动浪结束，进入调整浪
    'WaveA': ['WaveB'],
    'WaveB': ['WaveC', 'unknown'],  # WaveC 缺失时用 unknown 占位
    'WaveC': ['Wave1'],     # 调整浪结束，新周期
    'unknown': ['Wave1'],   # unknown 作为 WaveC 的占位符
}

# 反转阈值（价格变化百分比）
REVERSAL_THRESHOLD = -0.02  # -2%


# ============================================================
# 数据结构
# ============================================================

@dataclass
class WaveJump:
    """波浪非法跳变"""
    idx: int
    prev_state: str
    curr_state: str
    prev_price: float
    curr_price: float
    reversal_pct: float  # 价格变化百分比
    is_reversal: bool    # 是否为合法反转
    correction: str       # 建议的修正状态


# ============================================================
# WaveBacktrackCorrector
# ============================================================

class WaveBacktrackCorrector:
    """
    波浪序列回溯修正器 v2.0

    基于正确波浪理论和价格方向，对 L2 cache 中的 wave_state 进行回溯修正。
    """

    def __init__(self, reversal_threshold: float = REVERSAL_THRESHOLD, 
                 lookback: int = 20):
        """
        Args:
            reversal_threshold: 价格反转阈值（默认 -2%）
            lookback: 回溯最大窗口天数
        """
        self.reversal_threshold = reversal_threshold
        self.lookback = lookback

    # ============================================================
    # 公开 API
    # ============================================================

    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 应用回溯修正。

        Args:
            df: 包含 date, symbol, wave_state, close 列的 DataFrame
                可选: wave1_start_price, wave2_start_price, ..., wave_last_end_price

        Returns:
            修正后的 DataFrame（新增 wave_state_corrected 列）
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 标准化 state 名称（处理 "Wave1" 和 "w1_formed" 两种格式）
        if 'wave_state' in df.columns:
            df['wave_state_raw'] = df['wave_state'].copy()
            df['wave_state'] = df['wave_state'].apply(self._normalize_state)

        # 确保必要的列存在
        required = ['date', 'symbol', 'wave_state', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"[Backtrack] 缺少列 {missing}，跳过修正")
            df['wave_state_corrected'] = df['wave_state']
            return df

        # 初始化输出列
        df['wave_state_corrected'] = df['wave_state'].copy()

        n_symbols = df['symbol'].nunique()
        logger.info(f"[Backtrack v2.0] 开始修正: {n_symbols} 只股票, {len(df):,} 行")

        results = []
        for symbol, sym_df in df.groupby('symbol'):
            sym_df = sym_df.reset_index(drop=True)
            corrected = self._correct_symbol(sym_df)
            sym_df['wave_state_corrected'] = corrected
            results.append(sym_df)

        result_df = pd.concat(results, ignore_index=True)

        # 统计修正数量
        n_corrected = (result_df['wave_state_corrected'] != result_df['wave_state']).sum()
        logger.info(f"[Backtrack v2.0] 修正完成: {n_corrected} 处修改")

        return result_df

    # ============================================================
    # 核心修正逻辑
    # ============================================================

    def _correct_symbol(self, sym_df: pd.DataFrame) -> List[str]:
        """
        对单只股票的波浪序列进行修正。
        
        算法：
        1. 扫描整个序列，标记非法跳变点
        2. 根据正确波浪理论确定修正方案
        3. 应用修正并传播
        
        Returns:
            修正后的 wave_state 列表
        """
        states = sym_df['wave_state'].tolist()
        closes = sym_df['close'].values if 'close' in sym_df.columns else None
        
        n = len(states)
        corrected = states.copy()

        # 步骤1：检测所有非法跳变并确定修正
        jump_corrections = {}  # idx -> correction state
        
        for i in range(1, n):
            prev = states[i - 1]
            curr = states[i]
            
            prev_id = STATE_ID.get(prev, 0)
            curr_id = STATE_ID.get(curr, 0)
            
            if prev_id == 0 or curr_id == 0:
                continue
            
            # 计算价格变化
            if closes is not None and closes[i - 1] > 0:
                price_change = (closes[i] - closes[i - 1]) / closes[i - 1]
            else:
                price_change = 0.0
            
            # 检查是否为合法转换
            valid_next = VALID_TRANSITIONS.get(prev, [])
            is_legal = curr in valid_next
            
            # 同状态延续（如 Wave1→Wave1, Wave2→Wave2）是合法的
            if curr == prev:
                continue  # 合法跳变，不修正
            
            if is_legal:
                continue  # 合法跳变，不修正
            
            # =============================================
            # 非法跳变，需要修正
            # =============================================
            
            correction = None
            
            # ---------- 序号倒退 ----------
            if curr_id < prev_id:
                # 规则：推动浪中序号倒退通常是错误
                # 调整浪中 WaveB→WaveA 是合法的（反弹失败）
                # WaveC→Wave1 是合法的（新周期开始）
                
                if prev == 'WaveC' and curr == 'Wave1':
                    # WaveC→Wave1: 合法（新周期）
                    continue
                
                # WaveC→Wave1: 合法（新周期开始）
                if prev == 'WaveC' and curr == 'Wave1':
                    continue  # 不修正
                
                # 推动浪中的倒退
                # 注意：Wave5→Wave1 可能合法（进入调整浪），需要检查价格反转
                # 其他倒退（如 Wave4→Wave3, Wave3→Wave2, Wave2→Wave1）通常是错误
                
                if prev == 'Wave5':
                    # Wave5→Wave1: 永远是 WaveA（推动浪结束后进入调整浪）
                    # Wave5→Wave1 不可能是新周期（那是 WaveC→Wave1 的规则）
                    correction = 'WaveA'
                elif prev == 'Wave4':
                    # Wave4→Wave3: 应该是 Wave5
                    correction = 'Wave5'
                elif prev == 'Wave3':
                    # Wave3→Wave2 或 Wave3→Wave1: 应该是 Wave4
                    correction = 'Wave4'
                elif prev == 'Wave2':
                    # Wave2→Wave1: 检查价格是否反转
                    is_reversal = price_change > 0
                    if not is_reversal:
                        # 非反转 → 错误，应该是 Wave3（继续下跌）
                        correction = 'Wave3'
                elif prev == 'WaveB':
                    # WaveB→WaveA: 错误（反弹失败后应该继续到 WaveC）
                    # WaveB→Wave1: 错误（B后面必须是C，不是1）
                    if curr == 'WaveA':
                        correction = 'WaveC'
                    elif curr == 'Wave1':
                        correction = 'unknown'  # WaveC 占位符
                elif prev == 'WaveA':
                    # WaveA→Wave1 是错误的（跳过 WaveB/WaveC）
                    correction = 'WaveB'
            
            # ---------- 序号跳跃（跳过中间波浪）----------
            else:  # curr_id > prev_id + 1
                # 例如：Wave3→Wave5（跳过 Wave4）
                # 例如：Wave5→Wave1（跳过调整浪）
                
                if prev == 'Wave5' and curr == 'Wave1':
                    # Wave5→Wave1: 非法（跳过调整浪 ABC）
                    correction = 'WaveA'
                elif prev == 'Wave5' and curr in ['Wave2', 'Wave3', 'Wave4']:
                    # Wave5→Wave2/3/4: 非法
                    correction = 'WaveA'
                elif prev == 'Wave4' and curr == 'Wave1':
                    # Wave4→Wave1: 非法（跳过 Wave5）
                    correction = 'Wave5'
                elif prev == 'Wave4' and curr == 'Wave2':
                    # Wave4→Wave2: 非法
                    correction = 'Wave5'
                elif prev == 'Wave4' and curr == 'Wave3':
                    # Wave4→Wave3: 非法
                    correction = 'Wave5'
                elif prev == 'Wave3' and curr in ['Wave1', 'Wave2']:
                    # Wave3→Wave1/2: 非法
                    correction = 'Wave4'
                elif prev == 'Wave3' and curr == 'Wave5':
                    # Wave3→Wave5: 非法（跳过 Wave4）
                    correction = 'Wave4'
                elif prev == 'Wave2' and curr == 'Wave1':
                    # Wave2→Wave1: 非法
                    correction = 'Wave3'
                elif prev == 'Wave2' and curr in ['Wave4', 'Wave5']:
                    # Wave2→Wave4/5: 非法
                    correction = 'Wave3'
                elif prev == 'Wave1' and curr == 'Wave5':
                    # Wave1→Wave5: 非法（跳过 Wave2/3/4）
                    # 需要看 Wave4 是否已经出现过
                    wave4_in_states = 'Wave4' in states[:i]
                    if wave4_in_states:
                        # Wave4 已出现，Wave5 合法，Wave1 是错误的
                        correction = 'WaveB'  # Wave1 应该是调整浪
                    else:
                        # Wave4 还没出现，Wave5 应该是 Wave4
                        correction = 'Wave4'
                elif prev == 'Wave1' and curr == 'Wave4':
                    # Wave1→Wave4: 非法
                    wave4_in_states = 'Wave4' in states[:i]
                    if wave4_in_states:
                        correction = 'WaveB'
                    else:
                        correction = 'Wave2'
                elif prev == 'Wave1' and curr == 'Wave3':
                    # Wave1→Wave3: 非法
                    wave4_in_states = 'Wave4' in states[:i]
                    if wave4_in_states:
                        correction = 'WaveA'
                    else:
                        correction = 'Wave2'
                # 调整浪中的跳跃
                elif prev == 'WaveA' and curr in ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5']:
                    # WaveA→推动浪: 非法
                    correction = 'WaveB'
                elif prev == 'WaveB' and curr in ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5']:
                    # WaveB→推动浪: 非法（B后面必须是C，不是1）
                    if curr != 'Wave1':
                        correction = 'WaveC'
                    else:
                        # WaveB→Wave1: 应该是 unknown（WaveC 占位符）
                        correction = 'unknown'
                elif prev == 'WaveC' and curr in ['Wave2', 'Wave3', 'Wave4', 'Wave5', 'WaveA']:
                    # WaveC→推动浪/调整浪: 非法
                    correction = 'Wave1'
            
            if correction:
                jump_corrections[i] = correction

        # 步骤2：应用修正
        for idx, correction in jump_corrections.items():
            if correction:
                corrected[idx] = correction
                logger.debug(f"[Backtrack] idx={idx}: {states[idx]}→{correction}")

        # 步骤3：传播修正
        # 当某个波浪被修正后，后续的"不合理"波浪应该被调整为期望的波浪
        for idx, correction in jump_corrections.items():
            if correction is None:
                continue
            
            # 找到下一个期望的波浪
            expected_next_list = VALID_TRANSITIONS.get(correction, [])
            if not expected_next_list:
                continue
            
            expected_next = expected_next_list[0]  # 取第一个合法后续
            
            # 检查 Wave4 是否已经在修正序列中出现过
            # 如果 Wave4 已经出现，Wave5 不应该再传播（Wave5 可能是合法的）
            wave4_appeared = 'Wave4' in corrected[:idx]
            wave5_appeared = 'Wave5' in corrected[:idx]
            
            # 向前查找，修正在"期望"位置的错误波浪
            for j in range(idx + 1, n):
                curr_state = corrected[j]
                curr_id = STATE_ID.get(curr_state, 0)
                
                # 如果当前状态序号小于期望的波浪序号，说明它可能是误标
                expected_id = STATE_ID.get(expected_next, 0)
                
                # 推动浪序号 1-5，调整浪 6-8
                # 修正规则：
                # WaveA 之后出现 Wave1/2/3/4/5 → 应该是 WaveB
                # WaveB 之后出现 Wave2/3/4/5 → 应该是 WaveC
                # Wave4 之后出现 Wave1/2/3 → 应该是 Wave5
                # Wave3 之后出现 Wave1/2 → 应该是 Wave4
                # Wave2 之后出现 Wave1 → 应该是 Wave3
                
                if correction == 'WaveA' and curr_state in ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5']:
                    corrected[j] = expected_next
                elif correction == 'WaveB' and curr_state in ['Wave2', 'Wave3', 'Wave4', 'Wave5']:
                    corrected[j] = expected_next
                elif correction == 'Wave5' and curr_state in ['Wave1', 'Wave2', 'Wave3']:
                    # Wave5 后面出现 Wave1/2/3 是非法的
                    # Wave5 应该过渡到 WaveA
                    # 但如果 Wave4 已经出现过，Wave5 可能本身就是正确的（只是被误标为 Wave1/2/3）
                    if not wave4_appeared:
                        # Wave4 还没出现，Wave5 是对 Wave4 的误标，修正
                        corrected[j] = expected_next
                    elif wave4_appeared and wave5_appeared:
                        # Wave4 和 Wave5 都出现过，Wave5 是合法的延续
                        # Wave1/2/3 是对 WaveA 的误标
                        if curr_state == 'Wave1':
                            corrected[j] = 'WaveA'
                        elif curr_state == 'Wave2':
                            corrected[j] = 'WaveA'
                        elif curr_state == 'Wave3':
                            corrected[j] = 'WaveA'
                    else:
                        # Wave4 出现过但 Wave5 还没出现（被误标为 Wave1/2/3）
                        if curr_state in ['Wave1', 'Wave2', 'Wave3', 'Wave4']:
                            corrected[j] = expected_next  # Wave5
                elif correction == 'Wave4' and curr_state in ['Wave1', 'Wave2']:
                    corrected[j] = expected_next
                elif correction == 'Wave3' and curr_state in ['Wave1', 'Wave2']:
                    # Wave3→Wave1 和 Wave3→Wave2 都是非法的
                    # Wave3 之后应该是 Wave4
                    corrected[j] = expected_next
                elif correction == 'WaveC' and curr_state in ['Wave2', 'Wave3', 'Wave4', 'Wave5', 'WaveA']:
                    corrected[j] = expected_next

        return corrected

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _normalize_state(state: str) -> str:
        """标准化波浪状态名称"""
        if pd.isna(state):
            return 'unknown'
        
        state = str(state).strip()
        
        # 映射表
        state_map = {
            'Wave1': 'Wave1', 'Wave2': 'Wave2', 'Wave3': 'Wave3',
            'Wave4': 'Wave4', 'Wave5': 'Wave5',
            'WaveA': 'WaveA', 'WaveB': 'WaveB', 'WaveC': 'WaveC',
            'w1_formed': 'Wave1', 'w2_formed': 'Wave2', 'w3_formed': 'Wave3',
            'w4_formed': 'Wave4', 'w5_formed': 'Wave5', 'w4_in_progress': 'Wave4',
            'initial': 'unknown', 'unknown': 'unknown', 'neutral': 'unknown',
        }
        
        return state_map.get(state, state)
