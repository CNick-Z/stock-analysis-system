"""
WaveChan V3 - 持续修正波浪识别系统
=====================================

核心设计：
- 每根新K线 → 可能形成新笔 → 重新扫描全部笔序列 → 修正所有浪型
- 不预判，走走出来之后确认
- 持久化笔序列 + 波浪状态到 .pkl

Phase 1.1: 浪终结信号检测
Phase 1.2: 买卖点信号生成
Phase 1.3: 止损体系
Phase 2.2: SymbolWaveCache 增强
Phase 3.1: BatchWaveBuilder 批量构建

依赖：czsc >= 0.10
"""

import pickle
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================
# 数据结构
# ======================

@dataclass
class WaveSnapshot:
    """某一时点的波浪状态快照"""
    date: str = ""
    state: str = "initial"
    direction: str = "neutral"  # up/down/neutral

    # 各浪高低点
    w1_start: Optional[float] = None
    w1_end: Optional[float] = None
    w2_end: Optional[float] = None
    w3_end: Optional[float] = None
    w4_end: Optional[float] = None
    w5_end: Optional[float] = None

    # 斐波那契位
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    fib_target: Optional[float] = None  # W5目标

    # 当前笔信息
    bi_count: int = 0
    bi_direction: str = "neutral"
    bi_high: Optional[float] = None
    bi_low: Optional[float] = None

    # 缠论信号（czsc）
    czsc_signal: str = ""
    w3_start: Optional[float] = None  # W3起点 = W2终点

    # Phase 1.1: 浪终结信号
    wave_end_signal: str = ""           # '' | 'W5_END' | 'W4_END' | 'C_END' | ...
    wave_end_confidence: float = 0.0    # 0.0~1.0

    # Phase 1.1: 分型确认
    last_fx_mark: str = ""              # 'D'(底分型) / 'G'(顶分型)
    last_fx_price: Optional[float] = None
    last_fx_date: str = ""

    # Phase 1.1: 推动浪终结信号详情
    w5_divergence: bool = False         # 第5浪量价背离
    w5_failed: bool = False             # 失败浪（5<3）
    w5_ending_diagonal: bool = False    # 终结楔形

    # Phase 1.1: 调整浪终结信号详情
    c_wave_5_struct: bool = False       # C浪5浪结构完成
    c_wave_fib_target: bool = False     # C浪 = A浪 × 1.618

    # Phase 1.3: 止损位
    stop_loss: Optional[float] = None  # 动态止损位
    stop_loss_type: str = ""            # 'wave_end' / 'pct'

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class BiRecord:
    """一笔的记录"""
    seq: int           # 笔序号
    direction: str     # up/down
    start_price: float
    end_price: float
    start_date: str
    end_date: str
    volume: float = 0.0  # 该笔总成交量（Phase 1.1 量价背离检测）


# ======================
# WaveCounterV3 核心
# ======================

class WaveCounterV3:
    """
    波浪计数器 V3 - 持续修正版

    每次喂入一个新笔(bi)，内部重新计算所有浪型状态。
    核心原则：等走出来确认，不预判。

    Phase 1.1: 浪终结信号检测（分型确认 + 量价背离 + 斐波那契目标）
    Phase 1.2: 买卖点信号生成
    Phase 1.3: 止损体系
    """

    def __init__(self):
        self.bis: List[BiRecord] = []
        self.state = "initial"
        self.snapshot = WaveSnapshot()
        # 缠论分型列表（从 CZSC 获取）
        self.fx_list: List[Any] = []

    def feed_bi(self, bi: BiRecord, fx_list: List[Any] = None) -> WaveSnapshot:
        """
        喂入一个新笔，触发重新计算

        Args:
            bi: 新完成的笔记录
            fx_list: CZSC 分型列表（用于浪终结确认）
        """
        self.bis.append(bi)
        if fx_list is not None:
            self.fx_list = fx_list
        self._recalc()
        return self.snapshot

    def feed_bis(self, bis: List[BiRecord], fx_list: List[Any] = None) -> WaveSnapshot:
        """
        批量喂入笔序列（用于初始化）
        """
        self.bis = list(bis)
        if fx_list is not None:
            self.fx_list = fx_list
        self._recalc()
        return self.snapshot

    def _recalc(self):
        """
        核心：从当前笔序列重新计算波浪（持续修正）
        从最新往回分析，应用波浪规则确认各浪
        """
        n = len(self.bis)
        if n < 3:
            self._set_state("initial")
            return

        # 从最后一笔往前分析
        last_bi = self.bis[-1]

        # 收集所有高低点（笔的端点）
        turning_points = self._get_turning_points()

        if len(turning_points) < 3:
            self._set_state("initial")
            return

        # 尝试确认波浪
        wave_result = self._identify_waves(turning_points)

        self.snapshot.date = last_bi.end_date
        self.snapshot.bi_count = n
        self.snapshot.bi_direction = last_bi.direction
        self.snapshot.bi_high = last_bi.end_price if last_bi.direction == "up" else last_bi.start_price
        self.snapshot.bi_low = last_bi.end_price if last_bi.direction == "down" else last_bi.start_price

        # 缠论信号
        self.snapshot.czsc_signal = self._detect_czsc_signal(wave_result)

        # Phase 1.1: 浪终结检测
        self._detect_wave_end(wave_result)

        # Phase 1.3: 止损位计算
        self._calc_stop_loss(wave_result)

        # Phase 1.1: 更新最新分型信息
        self._update_fx_info()

    def _update_fx_info(self):
        """从 fx_list 更新最新分型信息到 snapshot"""
        if not self.fx_list:
            return
        last_fx = self.fx_list[-1]
        mark_str = str(last_fx.mark)
        if mark_str == '底分型':
            self.snapshot.last_fx_mark = 'D'
        elif mark_str == '顶分型':
            self.snapshot.last_fx_mark = 'G'
        else:
            self.snapshot.last_fx_mark = mark_str
        self.snapshot.last_fx_price = float(last_fx.fx)
        self.snapshot.last_fx_date = last_fx.dt.strftime('%Y-%m-%d') if hasattr(last_fx.dt, 'strftime') else str(last_fx.dt)[:10]

    def _get_turning_points(self) -> List[Dict]:
        """
        从笔序列提取高低点列表
        [{'idx': 0, 'price': 10.5, 'type': 'low', 'bi_seq': 1},
         {'idx': 1, 'price': 12.3, 'type': 'high', 'bi_seq': 2}, ...]
        """
        points = []
        for i, bi in enumerate(self.bis):
            # 每笔的起点
            # 向下笔: 高点(start) → 低点(end)
            # 向上笔: 低点(start) → 高点(end)
            if bi.direction == 'down':
                start_type, end_type = 'high', 'low'
            else:
                start_type, end_type = 'low', 'high'

            points.append({
                'idx': i,
                'price': bi.start_price,
                'type': start_type,
                'bi_seq': bi.seq,
                'date': bi.start_date
            })
            points.append({
                'idx': i,
                'price': bi.end_price,
                'type': end_type,
                'bi_seq': bi.seq,
                'date': bi.end_date
            })

        # 合并相邻同向端点（取极值）
        merged = self._merge_adjacent(points)
        return merged

    def _merge_adjacent(self, points: List[Dict]) -> List[Dict]:
        """合并相邻的同向极点，保留极值"""
        if not points:
            return []

        result = [points[0]]
        for p in points[1:]:
            last = result[-1]
            if last['type'] == p['type']:
                # 同向：保留更极值的那个
                if (p['type'] == 'high' and p['price'] > last['price']) or \
                   (p['type'] == 'low' and p['price'] < last['price']):
                    result[-1] = p
            else:
                result.append(p)
        return result

    def _identify_waves(self, points: List[Dict]) -> Dict:
        """
        从转折点序列识别波浪
        入口函数，调用 _find_wave_structure
        """
        return self._find_wave_structure(points)

    def _find_wave_structure(self, points: List[Dict]) -> Dict:
        """
        从转折点序列寻找波浪结构，同时检测浪终结信号

        老板核心教导的完整算法：
        1. 第1个向下笔出现 → 记录为W1起点候选
        2. 第2个向下笔出现 → 检查是否跌破W1起点
           - 跌破 → W1起点下移，重新扫描
           - 没跌破 → W2确认！之前的向上笔合并为W1
        3. 新向上笔突破W1终点 → W3确认！
        4. 向上笔突破前高 → 确认是驱动浪，不是反弹

        Phase 1.1 增强：推动浪终结检测
        - 失败浪检测：W5终点 < W3终点
        - 量价背离：W5创新高但成交量萎缩（通过 BiRecord.volume 检测）
        """
        bis = self.bis
        n = len(bis)
        if n < 5:
            return {'state': 'initial', 'waves': {}}

        up_bis = [b for b in bis if b.direction == 'up']
        down_bis = [b for b in bis if b.direction == 'down']

        if len(up_bis) < 2 or len(down_bis) < 2:
            return {'state': 'initial', 'waves': {}}

        result = self._scan_waves(bis, up_bis, down_bis, broken_bi=None)
        state = result['state']
        waves = result['waves']

        self._set_state(state, waves)
        return result

    def _scan_waves(self, bis: List[BiRecord],
                    up_bis: List[BiRecord],
                    down_bis: List[BiRecord],
                    broken_bi: BiRecord = None) -> Dict:
        """
        扫描笔序列，按规则确认波浪

        规则（老板核心教导）：
        - W1起点 = 第一个向下笔的终点（反弹的起始点）
        - W1终点 = 之后第一个向上笔的高点
        - W2确认 = 向下笔没跌破W1起点（第一个向下笔的终点）
        - W3确认 = 向上笔突破W1终点
        - W4 = 向下笔没跌破W3起点
        - W5 = 向上笔突破W3高点
        """

        n = len(bis)
        if n < 5:
            return {'state': 'initial', 'waves': {}}

        # ============================================================
        # 核心原则：首位相接
        #
        # W1 = bi1.end → 第一个"起点=bi1.end"的向上笔的终点
        # W2 = W1.end → 第一个"起点=W1.end"的向下笔的终点（不能跌破W1起点）
        # W3 = W2.end → 追踪所有从W2.end开始的向上次浪，直到超过W1.end的那一波的终点
        #     重要：W3内部可以有次级回调（bi5→bi6→bi7），只要没跌破W2.end，W3就继续
        # W4 = W3.end → 第一个"起点=W3.end"的向下笔的终点（不能跌破W3起点）
        # W5 = W4.end → 第一个"起点=W4.end"的向上笔的终点（超过W3.end）
        # ============================================================

        def eq(p1: float, p2: float) -> bool:
            """首位相接判断：容差0.02元，处理浮点精度"""
            return abs(p1 - p2) < 0.02

        # ===== W1 = bi1.end → 第一个起点=W1起点的向上笔的终点 =====
        w1_start = down_bis[0].end_price
        w1_start_seq = down_bis[0].seq

        w1_end = None
        w1_end_seq = None
        for b in up_bis:
            if b.seq > w1_start_seq and eq(b.start_price, w1_start):
                w1_end = b.end_price
                w1_end_seq = b.seq
                break

        if w1_end is None:
            self._set_state('initial')
            return {'state': 'initial', 'waves': {}}

        waves = {
            'W1': {'start': w1_start, 'end': w1_end},
        }
        self.snapshot.w1_start = w1_start
        self.snapshot.w1_end = w1_end

        # ===== W2起点 = W1终点（首尾相接），W2终点 = 第一个起点=W2起点的向下笔的终点 =====
        w2_start = w1_end
        w2_end = None
        w2_end_seq = None

        # 跳过已检查过的向下笔（在broken_bi之后重新扫描，避免无限递归）
        checked_seqs = set()
        if broken_bi:
            checked_seqs.add(broken_bi.seq)

        for b in down_bis:
            if b.seq in checked_seqs:
                continue
            if b.seq > w1_end_seq and eq(b.start_price, w1_end):
                if b.end_price < w1_start:
                    # 跌破W1起点 → W1起点下移，跳过该笔继续找W2
                    checked_seqs.add(b.seq)
                    continue
                w2_end = b.end_price
                w2_end_seq = b.seq
                break

        if w2_end is None:
            state = 'w1_formed'
            self._set_state(state, waves)
            return {'state': state, 'waves': waves}

        waves['W2'] = {'start': w2_start, 'end': w2_end}
        self.snapshot.w2_end = w2_end
        state = 'w2_formed'

        # ===== W3起点 = W2终点（首尾相接）
        # W3 = 从W2_end开始的向上次浪 bi + 中间次级回调 + 最终突破W1_end的那一波
        # W3确认条件：某向上笔终点 > W1_end
        # W3破坏条件：某向下笔终点 < W2_end
        w3_start = w2_end
        w3_end = None
        w3_end_seq = None

        # 从W2_end之后找所有笔
        after_w2 = [b for b in bis if b.seq > w2_end_seq]
        if after_w2:
            # 找第一个起点=W2_end的向上笔作为W3开始
            first_up = next((b for b in after_w2
                           if b.direction == 'up' and eq(b.start_price, w2_end)), None)
            if first_up:
                # 从W3开始追踪所有笔
                w3_bis = [b for b in after_w2 if b.seq >= first_up.seq]
                w3_highest = first_up.end_price
                w3_highest_seq = first_up.seq

                for b2 in w3_bis:
                    if b2.seq == first_up.seq:
                        continue
                    if b2.direction == 'up':
                        if b2.end_price > w3_highest:
                            w3_highest = b2.end_price
                            w3_highest_seq = b2.seq
                        # 检查是否突破W1_end
                        if w3_highest > w1_end:
                            w3_end = w3_highest
                            w3_end_seq = w3_highest_seq
                            break
                    else:  # down bi
                        # W3内部的次级回调，只要没跌破W2_end就继续
                        if b2.end_price < w3_start:
                            # 跌破W2起点，W3破坏
                            break

        if w3_end is None:
            self._set_state(state, waves)
            return {'state': state, 'waves': waves}

        waves['W3'] = {'start': w3_start, 'end': w3_end}
        self.snapshot.w3_start = w3_start
        self.snapshot.w3_end = w3_end
        state = 'w3_formed'

        # 斐波那契回调位（W4支撑位）
        diff = w3_end - w3_start
        if diff > 0:
            self.snapshot.fib_382 = round(w3_end - diff * 0.382, 2)
            self.snapshot.fib_500 = round(w3_end - diff * 0.500, 2)
            self.snapshot.fib_618 = round(w3_end - diff * 0.618, 2)
            self.snapshot.fib_target = round(w3_end + diff * 1.618, 2)

        # ===== W4：追踪从W3终点之后所有向下笔，找最低点 =====
        # W4可能包含多个向下次浪+中间反弹，只要没跌破W3_start就继续
        # W4终点 = W4确认前那个最低点（第一个向上笔的起点）
        w4_start = w3_end
        w4_end = None
        w4_end_seq = None

        after_w3 = [b for b in bis if b.seq > w3_end_seq]
        w4_candidate_low = None
        w4_candidate_seq = None

        for b in after_w3:
            if b.direction == 'down':
                if eq(b.start_price, w4_start) or b.start_price < w4_start:
                    # 新的向下笔开始
                    if b.end_price < w3_start:
                        # 跌破W3起点 → W4破坏，趋势可能变了
                        break
                    # 更新W4最低点
                    if w4_candidate_low is None or b.end_price < w4_candidate_low:
                        w4_candidate_low = b.end_price
                        w4_candidate_seq = b.seq
            else:  # up bi
                # 第一个向上笔出现 → W4确认！
                # 起点应该等于W4最低点（首位相接）
                if w4_candidate_low is not None:
                    if eq(b.start_price, w4_candidate_low):
                        # W4确认！
                        w4_end = w4_candidate_low
                        w4_end_seq = w4_candidate_seq
                        break
                    elif abs(b.start_price - w4_candidate_low) < 0.02:
                        # 容差连接
                        w4_end = b.start_price
                        w4_end_seq = w4_candidate_seq
                        break
                    # start不连接，但已有W4最低点记录

        if w4_end:
            waves['W4'] = {'start': w4_start, 'end': w4_end}
            self.snapshot.w4_end = w4_end
            state = 'w4_formed'

        # ===== W5起点 = W4终点（首尾相接），W5终点 = 第一个起点=W5起点的向上笔的终点 =====
        if w4_end_seq:
            w5_start = w4_end
            for b in up_bis:
                if b.seq > w4_end_seq and eq(b.start_price, w4_end):
                    if b.end_price > w3_end:
                        waves['W5'] = {'start': w5_start, 'end': b.end_price}
                        self.snapshot.w5_end = b.end_price
                        state = 'w5_formed'
                        break

        last_bi = bis[-1]
        self.snapshot.direction = 'up' if last_bi.direction == 'up' else 'down'
        self._set_state(state, waves)
        return {'state': state, 'waves': waves}

    def _set_state(self, state: str, waves: Dict = None):
        """设置状态"""
        self.state = state
        self.snapshot.state = state

    def _detect_czsc_signal(self, wave_result: Dict) -> str:
        """
        根据波浪状态检测缠论信号
        这是一个简化版，完整版需要调用czsc库
        """
        state = wave_result.get('state', 'initial')
        waves = wave_result.get('waves', {})

        # 基于波浪位置判断缠论买点/卖点
        if state == 'w3_formed':
            # W3形成后，回调是买入机会
            return '二买机会'
        elif state == 'w4_formed':
            return '三买观察'
        elif state == 'w5_formed':
            return '一卖警告'
        elif state == 'initial':
            return '等待确认'

        return ''

    # =====================================================================
    # Phase 1.1: 浪终结信号检测
    # =====================================================================

    def _detect_wave_end(self, wave_result: Dict):
        """
        检测浪终结信号（Phase 1.1）

        推动浪终结检测：
        - W5量价背离：收盘创新高但成交量萎缩
        - 失败浪：W5终点 < W3终点
        - 终结楔形：可通过检测子浪重叠判断

        调整浪终结检测：
        - C浪5浪结构完成
        - C浪 = A浪 × 1.618 斐波那契目标

        缠论分型确认：
        - 底分型形成 = 可能的浪终结（底）
        - 顶分型形成 = 可能的浪终结（顶）
        """
        s = self.snapshot
        s.wave_end_signal = ''
        s.wave_end_confidence = 0.0

        # 重置终结信号详情
        s.w5_divergence = False
        s.w5_failed = False
        s.w5_ending_diagonal = False
        s.c_wave_5_struct = False
        s.c_wave_fib_target = False

        state = wave_result.get('state', 'initial')
        waves = wave_result.get('waves', {})

        # ----- 推动浪终结检测 -----
        if state == 'w5_formed' and 'W5' in waves and 'W3' in waves:
            w5_end = waves['W5']['end']
            w3_end = waves['W3']['end']
            w1_end = waves.get('W1', {}).get('end')

            # 1. 失败浪检测：W5 < W3（W5未能突破W3高点）
            if w5_end < w3_end:
                s.w5_failed = True
                s.wave_end_signal = 'W5_END'
                s.wave_end_confidence = 0.75
                return

            # 2. W5量价背离检测：收盘创新高但成交量萎缩
            #    W5高点 > W3高点，但W5成交量 < W3成交量
            if w5_end > w3_end:
                w5_vol = self._get_wave_volume('W5')
                w3_vol = self._get_wave_volume('W3')
                if w5_vol > 0 and w3_vol > 0 and w5_vol < w3_vol * 0.8:
                    s.w5_divergence = True
                    s.wave_end_signal = 'W5_END'
                    s.wave_end_confidence = max(s.wave_end_confidence, 0.70)

            # 3. 终结楔形检测：检测W5内部子浪是否有重叠（5-3-5-3-5结构）
            #    楔形特征：后一子浪的高点 < 前一子浪的高点（上升楔形）
            if self._detect_ending_diagonal('W5'):
                s.w5_ending_diagonal = True
                s.wave_end_signal = 'W5_END'
                s.wave_end_confidence = max(s.wave_end_confidence, 0.80)

        # ----- 调整浪终结检测 -----
        # 检测C浪终结（适用于熊市或大级别调整）
        # 条件：完整的5浪向下结构 + C浪达到A浪的1.618倍
        if state == 'w4_formed' and 'W4' in waves and 'W2' in waves:
            # 这里简化处理：检查当前向下笔是否是第5个子浪
            # 以及是否达到斐波那契目标
            last_bi = self.bis[-1] if self.bis else None
            if last_bi and last_bi.direction == 'down':
                c_struct_complete = self._detect_5_wave_down()
                if c_struct_complete:
                    s.c_wave_5_struct = True
                    s.wave_end_signal = 'C_END'
                    s.wave_end_confidence = max(s.wave_end_confidence, 0.65)

            # 检查C浪是否 = A浪 × 1.618
            if 'W1' in waves and 'W2' in waves:
                w1_end = waves['W1']['end']
                w2_end = waves['W2']['end']
                a_len = w2_end - w1_end  # A浪长度（假设A浪=W2）
                if a_len > 0 and last_bi:
                    c_target = w2_end - a_len * 1.618
                    # 如果当前低点接近C浪目标
                    if abs(last_bi.end_price - c_target) < c_target * 0.02:
                        s.c_wave_fib_target = True
                        s.wave_end_signal = 'C_END'
                        s.wave_end_confidence = max(s.wave_end_confidence, 0.70)

        # ----- W4终结检测 -----
        # W4终结 = 调整结束，可能开启W5或新的上升浪
        # 特征：W4内部完成abc三浪或更多次级结构 + 底分型形成
        if state == 'w4_formed' and s.last_fx_mark == 'D':
            # 底分型 + W4不破W1高点 → W4可能终结
            if 'W1' in waves and s.w4_end is not None:
                w1_end = waves['W1']['end']
                if s.w4_end > w1_end:  # W4低点没跌破W1终点
                    s.wave_end_signal = 'W4_END'
                    s.wave_end_confidence = 0.60

        # ----- W2终结检测 -----
        # W2终结 = 最佳买入机会
        # 特征：回撤50%-61.8% + 底分型 + 成交量萎缩
        if state == 'w2_formed' and 'W1' in waves and 'W2' in waves:
            w1_start = waves['W1']['start']
            w1_end = waves['W1']['end']
            w2_end = waves['W2']['end']
            w1_len = w1_end - w1_start

            if w1_len > 0:
                retracement = (w1_end - w2_end) / w1_len
                # 回撤比例50%-61.8%
                if 0.50 <= retracement <= 0.70:
                    # W2量能萎缩
                    w2_vol = self._get_wave_volume('W2')
                    w1_vol = self._get_wave_volume('W1')
                    if w2_vol > 0 and w1_vol > 0 and w2_vol < w1_vol * 0.6:
                        if s.last_fx_mark == 'D':
                            s.wave_end_signal = 'W2_END'
                            s.wave_end_confidence = 0.75

    def _get_wave_volume(self, wave_name: str) -> float:
        """
        获取某个浪的总成交量
        通过统计该浪包含的笔的 volume 累加
        """
        waves_map = {
            'W1': (self.snapshot.w1_start, self.snapshot.w1_end),
            'W2': (self.snapshot.w1_end, self.snapshot.w2_end),
            'W3': (self.snapshot.w2_end, self.snapshot.w3_end),
            'W4': (self.snapshot.w3_end, self.snapshot.w4_end),
            'W5': (self.snapshot.w4_end, self.snapshot.w5_end),
        }
        if wave_name not in waves_map:
            return 0.0

        start_p, end_p = waves_map[wave_name]
        if start_p is None or end_p is None:
            return 0.0

        # 找到该浪时间段内的所有笔，累加成交量
        total_vol = 0.0
        for bi in self.bis:
            # 简化：只检查笔的终点价格是否在浪的高点范围内
            # 实际应该根据笔的日期范围来判断
            if start_p <= bi.end_price <= end_p or end_p <= bi.end_price <= start_p:
                total_vol += bi.volume

        return total_vol

    def _detect_5_wave_down(self) -> bool:
        """
        检测是否完成了5浪向下的结构（用于C浪终结判断）
        通过检测最近的向下笔数量和结构来判断
        """
        # 简化实现：检查最近是否有5个连续的向下笔
        # 且最后一个向下笔形成了底分型
        recent_bis = self.bis[-10:] if len(self.bis) >= 10 else self.bis
        down_bis = [b for b in recent_bis if b.direction == 'down']

        # 需要至少5个向下笔
        if len(down_bis) < 5:
            return False

        # 检查最后几个向下笔是否是递减的（5浪特征）
        last_5 = down_bis[-5:]
        lows = [b.end_price for b in last_5]

        # 5浪向下：每个低点都创新低
        for i in range(len(lows) - 1):
            if lows[i+1] >= lows[i]:
                return False

        return True

    def _detect_ending_diagonal(self, wave_name: str) -> bool:
        """
        检测终结楔形（Ending Diagonal）

        终结楔形特征：
        - 5个子浪组成（5-3-5-3-5）
        - 上升楔形：每浪高点递减
        - 下降楔形：每浪低点点燃递增
        - 通道线收敛

        这里用简化方法：检测W5内部子浪的高点是否递减
        """
        # 找到W5对应的笔
        waves_map = {
            'W5': (self.snapshot.w4_end, self.snapshot.w5_end),
        }
        if wave_name not in waves_map:
            return False

        start_p, end_p = waves_map[wave_name]
        if start_p is None or end_p is None:
            return False

        # 获取W5区间内的所有向上笔
        w5_bis = [b for b in self.bis
                  if b.seq > 0 and
                  ((start_p <= b.start_price <= end_p) or (start_p <= b.end_price <= end_p)) and
                  b.direction == 'up']

        # 终结楔形需要至少3个向上子浪
        if len(w5_bis) < 3:
            return False

        # 检查上升高点是否逐浪降低（楔形特征）
        up_highs = [b.end_price for b in w5_bis]
        # 后浪高点 < 前浪高点 = 上升楔形（看跌）
        for i in range(len(up_highs) - 1):
            if up_highs[i+1] >= up_highs[i]:
                return False

        return True

    # =====================================================================
    # Phase 1.2: 买卖点信号生成
    # =====================================================================

    def get_buy_sell_signals(self) -> Dict:
        """
        生成买卖点信号（Phase 1.2）

        返回格式：
        {
            'signal': str,          # 信号类型
            'price': float,         # 建议价格
            'stop_loss': float,     # 止损位
            'reason': str,          # 信号理由
            'confidence': float,   # 置信度 0.0~1.0
        }

        信号类型：
        - W2_BUY: W2买信号（最佳买入点）
        - W4_BUY_ALERT: W4买预警（W4进行中）
        - W4_BUY_CONFIRMED: W4买确认（周线底分型）
        - W5_SELL: W5卖信号
        - C_BUY: C浪买信号（熊市）
        - HOLD: 持仓信号
        - NO_SIGNAL: 无信号
        """
        s = self.snapshot
        result = {
            'signal': 'NO_SIGNAL',
            'price': None,
            'stop_loss': None,
            'reason': '',
            'confidence': 0.0,
        }

        # 根据当前波浪状态生成信号
        state = s.state

        # ----- W5卖信号 -----
        if s.wave_end_signal == 'W5_END' and s.w5_end:
            result['signal'] = 'W5_SELL'
            result['price'] = s.w5_end
            # 止损：W5高点的103%（小幅止损）
            result['stop_loss'] = round(s.w5_end * 1.03, 2)
            result['reason'] = self._build_w5_sell_reason()
            result['confidence'] = s.wave_end_confidence
            return result

        # ----- W4买确认 -----
        # 条件：W4进行中 + 底分型形成 + 价格不破前低
        if s.wave_end_signal == 'W4_END' and s.w4_end:
            result['signal'] = 'W4_BUY_CONFIRMED'
            result['price'] = s.w4_end
            result['stop_loss'] = round(s.w4_end * 0.97, 2)  # 3%止损
            result['reason'] = f"W4调整结束，底分型确认，价格{s.w4_end:.2f}"
            result['confidence'] = s.wave_end_confidence
            return result

        # ----- W4买预警 -----
        if state == 'w4_formed' and s.w4_end:
            result['signal'] = 'W4_BUY_ALERT'
            result['price'] = s.bi_low  # 当前价（近似）
            result['stop_loss'] = round(s.w4_end * 0.97, 2)
            result['reason'] = f"W4调整中，关注{s.fib_382:.2f}/{s.fib_500:.2f}/{s.fib_618:.2f}支撑"
            result['confidence'] = 0.5
            return result

        # ----- W2买信号 -----
        if s.wave_end_signal == 'W2_END' and s.w2_end:
            result['signal'] = 'W2_BUY'
            result['price'] = s.w2_end
            # 止损：W2低点的103%
            result['stop_loss'] = round(s.w2_end * 0.97, 2)
            result['reason'] = f"W2回撤结束，底分型确认，价格{s.w2_end:.2f}"
            result['confidence'] = s.wave_end_confidence
            return result

        # ----- C浪买信号（熊市）-----
        if s.wave_end_signal == 'C_END' and s.c_wave_5_struct:
            result['signal'] = 'C_BUY'
            result['price'] = s.bi_low
            result['stop_loss'] = round(s.bi_low * 0.97, 2)
            result['reason'] = 'C浪5浪完成，熊市调整结束，可能反转'
            result['confidence'] = s.wave_end_confidence
            return result

        # ----- 无信号 -----
        result['signal'] = 'NO_SIGNAL'
        return result

    def _build_w5_sell_reason(self) -> str:
        """构建W5卖信号的理由字符串"""
        s = self.snapshot
        reasons = []
        if s.w5_failed:
            reasons.append('失败浪(W5<W3)')
        if s.w5_divergence:
            reasons.append('量价背离')
        if s.w5_ending_diagonal:
            reasons.append('终结楔形')
        if s.last_fx_mark == 'G':
            reasons.append('顶分型形成')
        return '+'.join(reasons) if reasons else 'W5上升动能衰竭'

    # =====================================================================
    # Phase 1.3: 止损体系
    # =====================================================================

    def _calc_stop_loss(self, wave_result: Dict):
        """
        计算动态止损位（Phase 1.3）

        止损条件：
        1. 跌破前低（W2/W4低点）
        2. 浮亏超过5%

        止损类型：
        - 'wave_end': 波浪前低止损
        - 'pct': 百分比止损（浮亏5%）
        """
        s = self.snapshot
        s.stop_loss = None
        s.stop_loss_type = ''

        state = s.state
        current_price = s.bi_high if s.bi_direction == 'up' else s.bi_low

        if current_price is None:
            return

        # W2形成时：止损 = W2低点的103%
        if state == 'w2_formed' and s.w2_end:
            sl = round(s.w2_end * 0.97, 2)
            s.stop_loss = sl
            s.stop_loss_type = 'wave_end'
            return

        # W3形成后（持仓中）：止损 = W4斐波那契支撑位
        if state == 'w3_formed' and s.fib_618:
            # 用W4可能回调到的最低位置作为止损
            sl = round(s.fib_618 * 0.97, 2)
            s.stop_loss = sl
            s.stop_loss_type = 'fib_support'
            return

        # W4进行中：止损 = W4低点（若跌破说明W4破坏）
        if state == 'w4_formed' and s.w4_end:
            sl = round(s.w4_end * 0.97, 2)
            s.stop_loss = sl
            s.stop_loss_type = 'wave_end'
            return

        # W5形成后：止损 = W5低点的103%
        if state == 'w5_formed' and s.w5_end:
            sl = round(s.w5_end * 0.97, 2)
            s.stop_loss = sl
            s.stop_loss_type = 'wave_end'
            return

    def get_stop_loss(self, entry_price: float) -> Optional[float]:
        """
        获取入场后的动态止损位

        Args:
            entry_price: 入场价格

        Returns:
            止损价，若无需止损则返回None
        """
        s = self.snapshot

        # 如果有明确的波浪止损位，优先使用
        if s.stop_loss:
            # 同时检查浮亏5%
            pct_sl = round(entry_price * 0.95, 2)
            if s.stop_loss_type == 'pct':
                return pct_sl
            # 取较近的止损位（更保守）
            return min(s.stop_loss, pct_sl)

        # 默认5%止损
        return round(entry_price * 0.95, 2)

    def get_snapshot(self) -> WaveSnapshot:
        """获取当前快照"""
        return self.snapshot

    def get_state_str(self) -> str:
        """获取状态描述"""
        s = self.snapshot
        if s.state == 'initial':
            return '等待形成W1'
        elif s.state == 'w1_formed':
            return f"W1已确认: {s.w1_start:.2f}→{s.w1_end:.2f}"
        elif s.state == 'w2_formed':
            return f"W2回调中: {s.w2_end:.2f}"
        elif s.state == 'w3_formed':
            fib = f" W4回调位:{s.fib_382:.2f}/{s.fib_500:.2f}/{s.fib_618:.2f}" if s.fib_382 else ''
            return f"W3已确认: {s.w3_end:.2f}{fib}"
        elif s.state == 'w4_formed':
            return f"W4调整中: {s.w4_end:.2f}"
        return s.state


# ======================
# SymbolWaveCache 增强 (Phase 2.2)
# ======================

class SymbolWaveCache:
    """
    单只股票的波浪数据缓存

    Phase 2.2 增强字段：
    - last_wave_end: 最后一浪终点时间
    - next_expected_wave: 下一个预期浪（如 W4）
    - signal_history: 买卖点信号历史
    """

    CACHE_VERSION = 2  # 版本升级，兼容旧缓存

    def __init__(self, symbol: str, cache_dir: str):
        self.symbol = symbol
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, f"{symbol}.pkl")
        self.counter = WaveCounterV3()
        self.completed_bis: List[BiRecord] = []
        self.features: List[Dict] = []
        self.last_date: Optional[str] = None

        # Phase 2.2: 增强字段
        self.last_wave_end: Optional[str] = None      # 最后一浪终点时间
        self.next_expected_wave: str = "W1"            # 下一个预期浪
        self.signal_history: List[Dict] = []          # 买卖点信号历史

        # 分型历史（Phase 1.1: 缠论分型确认）
        self.fx_history: List[Dict] = []              # [{date, mark, price}]

        self.meta = {
            'version': self.CACHE_VERSION,
            'symbol': symbol,
            'created': None,
            'last_updated': None,
            'total_bis': 0
        }

    def feed_bar(self, bar: dict) -> WaveSnapshot:
        """
        喂入一根日K线
        bar: {'date': '2026-03-01', 'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}
        内部累计High/Low形成笔
        """
        from czsc import CZSC, RawBar, Freq

        # 转换为自己需要的格式
        raw_bar = RawBar(
            symbol=self.symbol,
            dt=pd.to_datetime(bar['date']).to_pydatetime(),
            freq=Freq.D,
            open=float(bar['open']),
            high=float(bar['high']),
            low=float(bar['low']),
            close=float(bar['close']),
            vol=float(bar.get('volume', 0)),
            amount=0.0
        )

        # 累计到临时bars中
        if not hasattr(self, '_pending_bars'):
            self._pending_bars = []

        self._pending_bars.append(raw_bar)

        # CZSC需要至少9根K线才形成笔
        if len(self._pending_bars) < 9:
            return self.counter.get_snapshot()

        # 用CZSC检测是否有新笔形成
        c = CZSC(self._pending_bars)
        finished_bis = c.finished_bis
        fx_list = c.fx_list  # Phase 1.1: 获取分型列表

        # 对比之前的笔数量，看是否有新笔形成
        prev_count = len(self.completed_bis)
        new_count = len(finished_bis)

        if new_count > prev_count:
            # 有新笔形成！
            # 取新增的笔
            new_bis = finished_bis[prev_count:]

            for bi in new_bis:
                # 转换 CZSC 的笔 为 BiRecord
                direction = 'up' if bi.direction.value == '向上' else 'down'
                # 计算该笔的成交量（通过笔内K线累加）
                bi_volume = self._calc_bi_volume(bi, bar['date'])

                bi_record = BiRecord(
                    seq=len(self.completed_bis) + 1,
                    direction=direction,
                    start_price=float(bi.fx_a.fx),
                    end_price=float(bi.fx_b.fx),
                    start_date=bi.fx_a.dt.strftime('%Y-%m-%d'),
                    end_date=bi.fx_b.dt.strftime('%Y-%m-%d'),
                    volume=bi_volume
                )
                self.completed_bis.append(bi_record)
                # 喂入计数器（同时传入fx_list用于浪终结检测）
                self.counter.feed_bi(bi_record, fx_list=fx_list)

            self.meta['total_bis'] = len(self.completed_bis)

            # Phase 2.2: 更新最后一浪终点时间
            last_bi = self.completed_bis[-1]
            self.last_wave_end = last_bi.end_date

            # Phase 2.2: 更新下一个预期浪
            self._update_next_expected_wave()

        # Phase 1.1: 更新分型历史
        if fx_list and len(fx_list) > 0:
            latest_fx = fx_list[-1]
            fx_date = latest_fx.dt.strftime('%Y-%m-%d') if hasattr(latest_fx.dt, 'strftime') else str(latest_fx.dt)[:10]
            mark_str = str(latest_fx.mark)
            # 只记录新的分型
            if not self.fx_history or self.fx_history[-1]['date'] != fx_date:
                self.fx_history.append({
                    'date': fx_date,
                    'mark': 'D' if mark_str == '底分型' else 'G',
                    'price': float(latest_fx.fx),
                    'high': float(latest_fx.high),
                    'low': float(latest_fx.low),
                })

        # Phase 2.2: 检查是否有新信号
        self._check_and_record_signal()

        # 更新最后处理日期
        self.last_date = bar['date']

        # 生成当日特征快照
        snap = self.counter.get_snapshot()
        snap.date = bar['date']
        self._append_feature(snap)

        return snap

    def _calc_bi_volume(self, bi, current_date: str) -> float:
        """
        计算某笔的总成交量
        通过 CZSC 笔的 elements 属性获取笔内K线
        """
        try:
            if hasattr(bi, 'elements') and bi.elements:
                return sum(float(k.vol) for k in bi.elements if hasattr(k, 'vol'))
        except Exception:
            pass
        return 0.0

    def _update_next_expected_wave(self):
        """Phase 2.2: 根据当前状态更新下一个预期浪"""
        state = self.counter.state
        wave_map = {
            'initial': 'W1',
            'w1_formed': 'W2',
            'w2_formed': 'W3',
            'w3_formed': 'W4',
            'w4_formed': 'W5',
            'w5_formed': 'W1',  # 新周期开始
        }
        self.next_expected_wave = wave_map.get(state, 'W1')

    def _check_and_record_signal(self):
        """Phase 2.2: 检查是否有新信号并记录"""
        sig = self.counter.get_buy_sell_signals()
        if sig['signal'] == 'NO_SIGNAL':
            return

        # 检查是否与上一个信号相同（避免重复记录）
        if self.signal_history:
            last_sig = self.signal_history[-1]
            if last_sig['signal'] == sig['signal'] and last_sig['date'] == self.last_date:
                return

        # 记录新信号
        self.signal_history.append({
            'date': self.last_date,
            'signal': sig['signal'],
            'price': sig['price'],
            'stop_loss': sig['stop_loss'],
            'reason': sig['reason'],
            'confidence': sig['confidence'],
        })

        # 只保留最近100条信号
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

    def _append_feature(self, snap: WaveSnapshot):
        """追加特征到历史"""
        feat = snap.to_dict()
        # Phase 2.2: 添加增强字段
        feat['last_wave_end'] = self.last_wave_end
        feat['next_expected_wave'] = self.next_expected_wave
        self.features.append(feat)
        self.meta['last_updated'] = datetime.now().isoformat()

    def save(self):
        """保存到 pkl 文件"""
        os.makedirs(self.cache_dir, exist_ok=True)

        data = {
            'meta': self.meta,
            'bis': self.completed_bis,
            'features': self.features,
            'last_date': self.last_date,
            # Phase 2.2: 增强字段
            'last_wave_end': self.last_wave_end,
            'next_expected_wave': self.next_expected_wave,
            'signal_history': self.signal_history,
            'fx_history': self.fx_history,
        }

        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

        logger.debug(f"[{self.symbol}] 缓存已保存: {self.cache_path}")

    def load(self) -> bool:
        """从 pkl 文件加载，返回是否成功"""
        if not os.path.exists(self.cache_path):
            return False

        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)

            self.meta = data.get('meta', self.meta)
            self.completed_bis = data.get('bis', [])
            self.features = data.get('features', [])
            self.last_date = data.get('last_date')

            # Phase 2.2: 加载增强字段
            self.last_wave_end = data.get('last_wave_end')
            self.next_expected_wave = data.get('next_expected_wave', 'W1')
            self.signal_history = data.get('signal_history', [])
            self.fx_history = data.get('fx_history', [])

            # 重建计数器状态
            if self.completed_bis:
                # 尝试恢复fx_list（可能为空，因为旧缓存不含此字段）
                self.counter.feed_bis(self.completed_bis, fx_list=[])

            logger.info(f"[{self.symbol}] 加载缓存: {len(self.completed_bis)}笔, 最后日期:{self.last_date}")
            return True

        except Exception as e:
            logger.warning(f"[{self.symbol}] 缓存加载失败: {e}")
            return False

    def get_features_df(self) -> pd.DataFrame:
        """获取历史特征 DataFrame"""
        if not self.features:
            return pd.DataFrame()
        return pd.DataFrame(self.features)

    def get_signals_df(self) -> pd.DataFrame:
        """Phase 2.2: 获取信号历史 DataFrame"""
        if not self.signal_history:
            return pd.DataFrame()
        return pd.DataFrame(self.signal_history)

    def get_latest_signal(self) -> Optional[Dict]:
        """Phase 2.2: 获取最新信号"""
        if not self.signal_history:
            return None
        return self.signal_history[-1]

    def get_fx_df(self) -> pd.DataFrame:
        """Phase 1.1: 获取分型历史 DataFrame"""
        if not self.fx_history:
            return pd.DataFrame()
        return pd.DataFrame(self.fx_history)


# ======================
# BatchWaveBuilder (Phase 3.1)
# ======================

class BatchWaveBuilder:
    """
    批量构建全量波浪特征
    分批处理，不一次性加载所有数据（每次处理100只股票）
    """

    def __init__(self,
                 cache_dir: str,
                 data_dir: str = '/root/.openclaw/workspace/data'):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 加载全市场股票列表
        self.symbols = self._load_symbols()
        self._processed_count = 0
        self._batch_size = 100  # Phase 3.1: 每次处理100只股票

    def _load_symbols(self) -> List[str]:
        """从Parquet仓库加载所有股票代码"""
        import glob
        # 尝试多个可能的数据目录
        possible_dirs = [
            self.data_dir,
            '/data/warehouse',
            '/root/.openclaw/workspace/data',
        ]

        symbols = set()
        for data_dir in possible_dirs:
            if not os.path.exists(data_dir):
                continue
            years = glob.glob(os.path.join(data_dir, 'daily_data_year=*'))
            for year_dir in years:
                parquet_files = glob.glob(os.path.join(year_dir, '*.parquet'))
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf, columns=['symbol'])
                        symbols.update(df['symbol'].astype(str).unique())
                    except Exception:
                        continue

        if not symbols:
            logger.warning("未找到任何股票数据，将使用空列表")
        else:
            logger.info(f"共找到 {len(symbols)} 只股票")

        return sorted(list(symbols))

    def build_symbol(self, symbol: str, start_date: str = None, end_date: str = None) -> SymbolWaveCache:
        """
        为单只股票构建波浪数据
        1. 加载历史数据
        2. 逐K线喂入
        3. 返回缓存对象
        """
        cache = SymbolWaveCache(symbol, self.cache_dir)

        # 尝试加载已有缓存（增量模式）
        cache.load()

        # 确定日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 加载该股票的历史数据
        bars = self._load_symbol_data(symbol, start_date, end_date)

        if not bars:
            logger.warning(f"[{symbol}] 无数据")
            return cache

        # 逐K线喂入
        for bar in bars:
            cache.feed_bar(bar)

        cache.save()
        return cache

    def _load_symbol_data(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """从Parquet加载单只股票数据"""
        import glob

        all_bars = []

        # 尝试多个可能的数据目录
        possible_dirs = [
            self.data_dir,
            '/data/warehouse',
            '/root/.openclaw/workspace/data',
        ]

        for data_dir in possible_dirs:
            if not os.path.exists(data_dir):
                continue
            # 加载所有年份的数据
            year_dirs = sorted(glob.glob(os.path.join(data_dir, 'daily_data_year=*')))

            for year_dir in year_dirs:
                parquet_file = os.path.join(year_dir, 'data.parquet')
                if not os.path.exists(parquet_file):
                    continue

                try:
                    # 读取该股票的数据
                    df = pd.read_parquet(parquet_file)
                    df = df[df['symbol'].astype(str) == symbol].copy()

                    if df.empty:
                        continue

                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df = df.sort_values('date')

                    # 日期过滤
                    if start_date:
                        df = df[df['date'] >= start_date]
                    if end_date:
                        df = df[df['date'] <= end_date]

                    # 只保留最近300天（波浪分析不需要全量历史）
                    if len(df) > 300:
                        df = df.tail(300)

                    bars = df.to_dict('records')
                    all_bars.extend(bars)

                except Exception as e:
                    logger.warning(f"加载 {parquet_file} 失败: {e}")
                    continue

        # 按日期排序
        all_bars.sort(key=lambda x: x['date'])
        return all_bars

    def run_batch(self, symbols: List[str] = None,
                   start_date: str = None, end_date: str = None,
                   batch_size: int = 100,
                   progress_file: str = None) -> List[SymbolWaveCache]:
        """
        Phase 3.1: 批量构建

        Args:
            symbols: None 表示全市场
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 每批处理股票数量（默认100）
            progress_file: 进度文件路径

        Returns:
            所有缓存对象列表
        """
        if symbols is None:
            symbols = self.symbols

        total = len(symbols)
        results = []

        # 进度管理
        done_file = progress_file or os.path.join(self.cache_dir, '_progress.json')
        done_symbols = self._load_progress(done_file)

        logger.info(f"开始批量构建: {total} 只股票, 已完成: {len(done_symbols)}, 每批: {batch_size}")

        # Phase 3.1: 分批处理
        batch_count = 0
        for batch_start in range(0, len(symbols), batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_count += 1

            logger.info(f"处理第 {batch_count} 批: {len(batch_symbols)} 只股票 "
                        f"(总进度: {batch_start}/{total})")

            for i, symbol in enumerate(batch_symbols):
                if symbol in done_symbols:
                    continue

                try:
                    cache = self.build_symbol(symbol, start_date, end_date)
                    results.append(cache)

                    # 记录进度
                    done_symbols.add(symbol)
                    self._save_progress(done_file, done_symbols)
                    self._processed_count += 1

                    if (i + 1) % 10 == 0:
                        logger.info(f"批次内进度: {i+1}/{len(batch_symbols)}")

                except Exception as e:
                    logger.error(f"[{symbol}] 处理失败: {e}")
                    continue

            # 每批完成后记录进度
            logger.info(f"第 {batch_count} 批完成, 累计处理: {len(done_symbols)}/{total}")

        logger.info(f"批量构建完成: {len(results)} 只股票")
        return results

    def run_incremental(self, date: str, batch_size: int = 100) -> List[SymbolWaveCache]:
        """
        增量更新：只更新指定日期有数据的股票

        Args:
            date: 要更新的日期 (YYYY-MM-DD)
            batch_size: 每批处理数量

        Returns:
            更新过的缓存列表
        """
        import glob

        logger.info(f"增量更新: {date}, 每批: {batch_size}")

        # 找出该日期有数据的股票
        symbols_with_data = set()
        possible_dirs = [
            self.data_dir,
            '/data/warehouse',
            '/root/.openclaw/workspace/data',
        ]

        for data_dir in possible_dirs:
            if not os.path.exists(data_dir):
                continue
            year_dirs = glob.glob(os.path.join(data_dir, 'daily_data_year=*'))
            for year_dir in year_dirs:
                parquet_file = os.path.join(year_dir, 'data.parquet')
                if not os.path.exists(parquet_file):
                    continue
                try:
                    df = pd.read_parquet(parquet_file, columns=['symbol', 'date'])
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df = df[df['date'] == date]
                    symbols_with_data.update(df['symbol'].astype(str).unique())
                except Exception:
                    continue

        if not symbols_with_data:
            logger.info(f"{date} 无新数据")
            return []

        logger.info(f"{date} 有 {len(symbols_with_data)} 只股票有新数据")
        return self.run_batch(symbols=list(symbols_with_data), batch_size=batch_size)

    def _load_progress(self, progress_file: str) -> set:
        """加载进度"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('done', []))
            except Exception:
                return set()
        return set()

    def _save_progress(self, progress_file: str, done_symbols: set):
        """保存进度"""
        with open(progress_file, 'w') as f:
            json.dump({'done': list(done_symbols)}, f)


# ======================
# 工具函数
# ======================

def aggregate_daily_to_weekly(bars: List[dict]) -> List[dict]:
    """日线聚合为周线"""
    if not bars:
        return []

    df = pd.DataFrame(bars)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').apply(lambda x: x.start_time)

    weekly = df.groupby('week').agg({
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    weekly.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    weekly['date'] = weekly['date'].dt.strftime('%Y-%m-%d')
    return weekly.to_dict('records')


# ======================
# 测试
# ======================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    print("=== WaveChan V3 Phase 1 测试 ===")

    # ============================================================
    # 测试1: 波浪计数器核心 - 完整5浪测试
    # ============================================================
    print("\n--- WaveCounterV3 完整波浪测试 ---")

    # 构造有效的波浪序列：
    # W1: bi1(15→13) down, bi2(13→14.5) up = 13→14.5
    # W2: bi3(14.5→13.2) down (回撤50%) = 14.5→13.2
    # W3: bi4(13.2→16.0) up (突破w1_end=14.5) = 13.2→16.0
    # W4: bi5(16.0→14.5) down = 16.0→14.5
    # W5: bi6(14.5→17.0) up (量减=量价背离) = 14.5→17.0
    counter = WaveCounterV3()

    test_bis = [
        BiRecord(1, 'down', 15.0, 13.0, '2025-01-01', '2025-01-10', volume=100),
        BiRecord(2, 'up',   13.0, 14.5, '2025-01-10', '2025-01-20', volume=90),
        BiRecord(3, 'down', 14.5, 13.2, '2025-01-20', '2025-02-01', volume=45),  # W2 回撤50%, 量缩
        BiRecord(4, 'up',   13.2, 16.0, '2025-02-01', '2025-02-15', volume=130), # W3 突破14.5
        BiRecord(5, 'down', 16.0, 14.5, '2025-02-15', '2025-03-01', volume=70),  # W4
        BiRecord(6, 'up',   14.5, 17.0, '2025-03-01', '2025-03-15', volume=50),  # W5 量减
    ]

    for bi in test_bis:
        counter.feed_bi(bi)
        snap = counter.get_snapshot()
        print(f"笔{bi.seq} {bi.direction}: state={counter.state}, "
              f"W1E={snap.w1_end}, W2E={snap.w2_end}, W3E={snap.w3_end}, "
              f"W4E={snap.w4_end}, W5E={snap.w5_end}")

    print(f"\n最终状态: {counter.get_state_str()}")

    # ============================================================
    # 测试2: Phase 1.1 浪终结信号检测
    # ============================================================
    print("\n--- Phase 1.1: 浪终结信号检测 ---")
    snap = counter.get_snapshot()
    print(f"wave_end_signal: '{snap.wave_end_signal}'")
    print(f"wave_end_confidence: {snap.wave_end_confidence}")
    print(f"W5_divergence: {snap.w5_divergence}")
    print(f"W5_failed: {snap.w5_failed}")
    print(f"W5_ending_diagonal: {snap.w5_ending_diagonal}")
    print(f"last_fx_mark: '{snap.last_fx_mark}'")

    # ============================================================
    # 测试3: Phase 1.2 买卖点信号
    # ============================================================
    print("\n--- Phase 1.2: 买卖点信号 ---")
    sig = counter.get_buy_sell_signals()
    print(f"signal: {sig['signal']}")
    print(f"price: {sig['price']}")
    print(f"stop_loss: {sig['stop_loss']}")
    print(f"reason: {sig['reason']}")
    print(f"confidence: {sig['confidence']}")

    # ============================================================
    # 测试4: Phase 1.3 止损体系
    # ============================================================
    print("\n--- Phase 1.3: 止损体系 ---")
    if sig['price']:
        sl = counter.get_stop_loss(sig['price'])
        print(f"入场{sig['price']:.2f}后的止损位: {sl:.2f}")
    # 测试W2状态的止损
    counter_w2 = WaveCounterV3()
    w2_test_bis = [
        BiRecord(1, 'down', 15.0, 13.0, '2025-01-01', '2025-01-10', volume=100),
        BiRecord(2, 'up',   13.0, 14.5, '2025-01-10', '2025-01-20', volume=90),
        BiRecord(3, 'down', 14.5, 13.2, '2025-01-20', '2025-02-01', volume=45),
        BiRecord(4, 'up',   13.2, 14.8, '2025-02-01', '2025-02-10', volume=35),  # 未突破14.5
    ]
    for bi in w2_test_bis:
        counter_w2.feed_bi(bi)
    snap_w2 = counter_w2.get_snapshot()
    print(f"W2状态止损: state={counter_w2.state}, stop_loss={snap_w2.stop_loss}, type={snap_w2.stop_loss_type}")

    # ============================================================
    # 测试5: SymbolWaveCache 增强字段
    # ============================================================
    print("\n--- SymbolWaveCache 增强字段测试 ---")
    cache = SymbolWaveCache('TEST', '/tmp/wavechan_v3_test')
    print(f"缓存路径: {cache.cache_path}")
    print(f"last_wave_end: {cache.last_wave_end}")
    print(f"next_expected_wave: {cache.next_expected_wave}")
    print(f"signal_history: {cache.signal_history}")
    print(f"fx_history: {cache.fx_history}")

    # 模拟喂入一些bar测试
    import pandas as pd
    from czsc import CZSC, RawBar, Freq
    for i in range(30):
        bar = {
            'date': f'2025-01-{i+1:02d}',
            'open': 10.0 + i * 0.1,
            'high': 10.5 + i * 0.1,
            'low': 9.5 + i * 0.1,
            'close': 10.2 + i * 0.1,
            'volume': 1000000,
        }
        cache.feed_bar(bar)

    print(f"喂入30根K线后: {len(cache.completed_bis)}笔, state={cache.counter.state}")
    print(f"last_wave_end: {cache.last_wave_end}")
    print(f"next_expected_wave: {cache.next_expected_wave}")
    print(f"signal_count: {len(cache.signal_history)}")

    cache.save()
    print("已保存")

    # 重新加载
    cache2 = SymbolWaveCache('TEST', '/tmp/wavechan_v3_test')
    cache2.load()
    print(f"重新加载: {len(cache2.completed_bis)} 笔")
    print(f"重新加载 - last_wave_end: {cache2.last_wave_end}")
    print(f"重新加载 - next_expected_wave: {cache2.next_expected_wave}")
    print(f"重新加载 - signal_history count: {len(cache2.signal_history)}")

    # ============================================================
    # 测试6: BatchWaveBuilder (检查类可用)
    # ============================================================
    print("\n--- BatchWaveBuilder 测试 ---")
    builder = BatchWaveBuilder(cache_dir='/tmp/wavechan_batch_test')
    print(f"batch_size: {builder._batch_size}")
    print(f"symbols count: {len(builder.symbols)} (data dir may be empty)")

    print("\n=== 所有测试完成 ===")
