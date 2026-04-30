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

修复日志（Oracle 二审反馈）:
- P0-1: W4确认条件加强：要求向上笔突破幅度 > W4向下幅度的38.2%
- P0-2: W4_BUY_CONFIRMED止损改为fib_618（不再用w4_end×0.97）
- P1-1: W4_END检测增加向上笔有效突破验证 + W4幅度斐波那契比例检查
- P1-2: W5失败浪检测后标记并输出在reason中
- P2-2: 新增w4_in_progress状态，区分"W4进行中"和"W4已确认终结"
"""

import pickle
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from czsc import CZSC, RawBar, Freq

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

    # Phase 1.1: W4候选最低点（P2-2: 区分进行中vs已确认）
    w4_candidate_low: Optional[float] = None  # W4候选最低点（未确认）

    # Phase 1.3: 止损位
    stop_loss: Optional[float] = None  # 动态止损位
    stop_loss_type: str = ""            # 'wave_end' / 'pct' / 'fib618'

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
# 信号状态枚举
# ======================

class SignalStatus:
    """信号状态：假设 + 持续验证"""
    ALERT = "ALERT"       # 假设终结出现，需验证
    CONFIRMED = "CONFIRMED"  # 验证通过
    INVALID = "INVALID"     # 验证失败


@dataclass
class WaveSignal:
    """
    波浪信号 - 假设 + 验证模式

    核心原则：不是一次性确认，而是持续验证的假设
    - ALERT: 假设终结出现，需后续数据验证
    - CONFIRMED: 验证条件全部满足
    - INVALID: 验证条件失败

    验证条件遵循老板的"分型辅助判断规则"：
    - W4终点 → 底分型确认
    - 验证：价格不跌破W4起点（W4进行中的低点）
    """
    signal: str                    # 信号类型: W4_BUY, W2_BUY, etc.
    status: str                   # ALERT / CONFIRMED / INVALID
    price: float                  # 假设的浪终点价格
    stop_loss: float              # 止损位
    verify_conditions: List[str]  # 需要验证的条件列表
    verified_conditions: List[str]  # 已通过的验证条件
    reason: str                   # 信号理由
    confidence: float              # 置信度 0.0~1.0
    created_date: str              # 信号创建日期
    wave_type: str = ""           # 浪型: W1/W2/W3/W4/W5/A/B/C
    invalidated_reason: str = ""   # 失效原因（INVALID时填写）
    action: str = ""               # 操作指令："持仓待涨" / "止损出场" / "止盈离场" / "观望"
    exit_price: float = 0         # 实际出场价格（止损/止盈触发时记录）
    is_b_wave_rebound: bool = False  # V3新增：是否为B浪反弹（仅在bearish+W1<5时为True）
    position_size_ratio: float = 1.0  # V3新增：仓位比例 0.0~1.0，默认1.0表示满仓
    large_trend: str = "neutral"        # V3新增：大级别趋势 bullish/bearish/neutral

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


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

    修复（Oracle 二审反馈）:
    - P0-1: W4确认需要向上笔突破幅度 > W4向下幅度的38.2%
    - P0-2: W4_BUY_CONFIRMED止损改为fib_618
    - P1-1: W4_END检测增加向上笔突破验证 + W4幅度斐波那契检查
    - P1-2: W5失败浪检测后标记并输出在reason中
    - P2-2: 新增w4_in_progress状态
    """

    # W4确认阈值：向上笔幅度需 > W4向下幅度的38.2%
    W4_CONFIRM_MIN_RATIO = 0.382

    def __init__(self):
        self.bis: List[BiRecord] = []
        self.state = "initial"
        self.snapshot = WaveSnapshot()
        # 缠论分型列表（从 CZSC 获取）
        self.fx_list: List[Any] = []
        # 活跃信号列表（假设+验证模式）
        self.active_signals: List[WaveSignal] = []

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

        # 信号验证：检查活跃信号的验证条件是否满足
        self._verify_signals(wave_result)

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

    # ======================
    # V3 大级别趋势与仓位验证（2026-04-08同步）
    # ======================

    def _get_major_swing_points(self) -> List[Dict]:
        """
        获取大级别Swing拐点（只保留极值转折点）

        参考wave_simple.py的find_major_swing_points算法：
        - 从_turning_points()获取所有转折点
        - 只保留极值（running_high/running_low更新时才保留）
        - 高低交替
        """
        turning = self._get_turning_points()
        if not turning:
            return []

        all_points = []
        for i, p in enumerate(turning):
            all_points.append({
                'idx': i,
                'date': p['date'],
                'price': p['price'],
                'type': p['type'],
            })

        if not all_points:
            return []

        running_high = all_points[0]['price']
        running_low = all_points[0]['price']
        all_points[0]['extreme'] = all_points[0]['type']

        for p in all_points[1:]:
            if p['type'] == 'high':
                if p['price'] >= running_high:
                    running_high = p['price']
                    p['extreme'] = 'high'
                else:
                    p['extreme'] = None
            else:
                if p['price'] <= running_low:
                    running_low = p['price']
                    p['extreme'] = 'low'
                else:
                    p['extreme'] = None

        result = []
        for p in all_points:
            if not p.get('extreme'):
                continue
            if not result:
                result.append(p)
            elif result[-1]['extreme'] != p['extreme']:
                result.append(p)
            else:
                result[-1] = p

        if all_points[-1] not in result:
            result.append(all_points[-1])

        return result

    def _count_wave_segments(self, start_price: float, end_price: float,
                               direction: str) -> int:
        """
        统计从start_price到end_price区间内，指定方向笔的连续段数

        参考wave_simple.py的find_internal_segments：
        - 一个"段"由连续同向笔组成
        - UP方向的段 = 连续向上笔的终点
        - DOWN方向的段 = 连续向下笔的终点

        Args:
            start_price: 区间起始价（通常是一个浪的起点）
            end_price: 区间结束价（通常是一个浪的终点）
            direction: 'up' 或 'down'

        Returns:
            区间内指定方向的段数量
        """
        # 找在这个价格区间内的笔
        in_range = []
        for bi in self.bis:
            # 检查笔是否与区间相交
            if bi.start_price <= end_price and bi.end_price >= start_price:
                in_range.append(bi)

        # 统计连续同向笔段数
        segments = 0
        i = 0
        while i < len(in_range):
            # 跳过反向笔
            while i < len(in_range) and in_range[i].direction != direction:
                i += 1
            # 统计连续同向笔
            count = 0
            while i < len(in_range) and in_range[i].direction == direction:
                count += 1
                i += 1
            if count > 0:
                segments += 1

        return segments

    def _get_large_trend(self) -> str:
        """
        判断大级别趋势（用高低点同向变化，老板教的方法）
        
        优先从周线L1缓存读取trend，失败则用原方法逐根算
        """
        # 尝试从周线L1缓存读取
        try:
            from utils.wavechan_l1.reader import read_weekly_l1_trend
            cached_trend = read_weekly_l1_trend(self.symbol, self.year)
            if cached_trend:
                # 周线L1返回UP/DOWN/NEUTRAL → V3用bullish/bearish/neutral
                if cached_trend == 'UP':
                    return 'bullish'
                elif cached_trend == 'DOWN':
                    return 'bearish'
                else:
                    return 'neutral'
        except:
            pass
        
        # 缓存读取失败，用原方法逐根算
        turning_points = self._get_major_swing_points()

        if len(turning_points) < 4:
            return 'neutral'

        # 分离高点和低点
        highs = [p for p in turning_points if p['type'] == 'high']
        lows = [p for p in turning_points if p['type'] == 'low']

        if len(highs) < 2 or len(lows) < 2:
            return 'neutral'

        # 取最近的两个高点和两个低点
        recent_highs = highs[-2:]
        recent_lows = lows[-2:]

        # 判断高低点变化方向
        higher_high = recent_highs[-1]['price'] > recent_highs[-2]['price']
        higher_low = recent_lows[-1]['price'] > recent_lows[-2]['price']
        lower_high = recent_highs[-1]['price'] < recent_highs[-2]['price']
        lower_low = recent_lows[-1]['price'] < recent_lows[-2]['price']

        # 同向向上 = 上涨趋势
        if higher_high and higher_low:
            return 'bullish'
        # 同向向下 = 下跌趋势
        elif lower_high and lower_low:
            return 'bearish'
        else:
            return 'neutral'

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

        # V3: 验证波浪结构并计算仓位比例
        struct_check = self._validate_wave_structure(result)
        result.update(struct_check)

        return result

    def _validate_wave_structure(self, wave_result: Dict) -> Dict:
        """
        验证波浪结构合法性（V3新增，2026-04-08）

        结合wave_simple.py的算法：
        1. W1内部结构检查：推动浪W1内部须是5浪结构
        2. 大级别趋势检查：down_bis/up_bis比率判断趋势方向
        3. W2自身结构检查：调整浪W2内部须是3浪结构
        4. 三条铁律验证：W2不能跌破W1起点（已在现有代码中实现）

        V3 仓位调整（2026-04-08）：
        - bearish + W1<5浪 → B浪反弹，仓位10%
        - bearish + W1推动 → 下跌1浪，降仓10%
        - bullish + W1推动 → W2回调，仓位50-70%
        - 不再过滤B浪，而是打标签+降仓参与

        Returns:
            {
                'valid': bool,                    # 结构是否合法
                'filter_reason': str,            # 如果无效，原因是什么
                'w1_is_impulse': bool,           # W1是否为推动浪（5浪）
                'w2_is_correction': bool,        # W2是否为调整浪（3浪）
                'large_trend': str,              # 大级别趋势 bullish/bearish/neutral
                'w1_internal_segments': int,     # W1内部上涨子浪数
                'w2_internal_segments': int,     # W2内部下跌子浪数
                'is_b_wave_rebound': bool,       # V3新增：是否为B浪反弹
                'position_size_ratio': float,     # V3新增：仓位比例 0.3~1.0
            }
        """
        waves = wave_result.get('waves', {})
        if 'W1' not in waves or 'W2' not in waves:
            return {'valid': True, 'filter_reason': '', 'large_trend': 'neutral',
                    'w1_is_impulse': False, 'w2_is_correction': False,
                    'w1_internal_segments': 0, 'w2_internal_segments': 0,
                    'is_b_wave_rebound': False, 'position_size_ratio': 1.0}

        w1_start = waves['W1']['start']
        w1_end = waves['W1']['end']
        w2_end = waves['W2']['end']

        result = {
            'valid': True,
            'filter_reason': '',
            'large_trend': 'neutral',
            'w1_is_impulse': False,
            'w2_is_correction': False,
            'w1_internal_segments': 0,
            'w2_internal_segments': 0,
            'is_b_wave_rebound': False,    # V3新增
            'position_size_ratio': 1.0,     # V3新增
        }

        # === 1. 大级别趋势检查 ===
        large_trend = self._get_large_trend()
        result['large_trend'] = large_trend

        # === 2. W1内部结构检查 ===
        # 推动浪W1内部须有5个上涨子浪（1-2-3-4-5）
        w1_up_segments = self._count_wave_segments(w1_start, w1_end, 'up')
        result['w1_internal_segments'] = w1_up_segments
        result['w1_is_impulse'] = w1_up_segments >= 5

        # === 3. W2内部结构检查 ===
        # 调整浪W2内部是3浪（a-b-c）
        w2_down_segments = self._count_wave_segments(w1_end, w2_end, 'down')
        result['w2_internal_segments'] = w2_down_segments
        result['w2_is_correction'] = (2 <= w2_down_segments <= 4)

        # === V3 B浪判断逻辑（2026-04-08）===
        if large_trend == 'bearish':
            if not result['w1_is_impulse']:
                # 大级别下跌趋势中，W1<5浪 = B浪反弹
                result['is_b_wave_rebound'] = True
                # W2结构校验 - 必须是3浪结构才视为有效B浪
                if not (2 <= w2_down_segments <= 4):
                    result['filter_reason'] = (
                        f"大级别下跌趋势中，W1只有{w1_up_segments}个上涨子浪，"
                        f"疑似B浪但W2内部结构为{w2_down_segments}浪（非3浪），"
                        f"不满足B浪反弹条件"
                    )
                    result['valid'] = False
                    return result
                # B浪仓位统一降为10%
                result['position_size_ratio'] = 0.10
                result['filter_reason'] = (
                    f"大级别下跌趋势中，W1只有{w1_up_segments}个上涨子浪（非5浪推动），"
                    f"W2内部{w2_down_segments}浪结构确认，B浪反弹仓位{result['position_size_ratio']*100:.0f}%"
                )
                return result
            # bearish + W1推动 → 下跌1浪，降仓10%
            if result['w1_is_impulse']:
                result['is_b_wave_rebound'] = False
                result['position_size_ratio'] = 0.10
                result['filter_reason'] = (
                    f"大级别下跌趋势，W1有{w1_up_segments}个上涨子浪（5浪推动），"
                    f"判定为下跌1浪，降仓10%参与"
                )
                return result

        # === neutral趋势处理 ===
        if large_trend == 'neutral':
            if result['w1_is_impulse']:
                result['position_size_ratio'] = 0.40
                result['filter_reason'] = (
                    f"大级别趋势中性，W1有{w1_up_segments}个上涨子浪，"
                    f"判定为W2回调，仓位40%"
                )
            else:
                result['position_size_ratio'] = 0.50
                result['filter_reason'] = (
                    f"大级别趋势中性，W1只有{w1_up_segments}个子浪（非5浪），"
                    f"降低仓位至50%参与"
                )
            return result

        # === bullish + W1推动 → W2回调，仓位50-70% ===
        if large_trend == 'bullish' and result['w1_is_impulse']:
            w1_len = w1_end - w1_start
            if w1_len > 0:
                retracement = (w1_end - w2_end) / w1_len
                if retracement <= 0.50:
                    result['position_size_ratio'] = 0.70
                elif retracement <= 0.618:
                    result['position_size_ratio'] = 0.60
                else:
                    result['position_size_ratio'] = 0.50
            else:
                result['position_size_ratio'] = 0.60
            return result

        # === W1不是推动浪结构 → 降仓50% ===
        if not result['w1_is_impulse']:
            result['filter_reason'] = (
                f"W1内部只有{w1_up_segments}个子浪（非5浪），"
                f"降低仓位至50%参与"
            )
            result['position_size_ratio'] = 0.50
            return result

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
            """首位相接判断：自适应容差（兼顾日线和周线）
            - 绝对容差 0.02 元（小价格精度）
            - 相对容差 10%（大价格波动，如周线数据）
            取两者中更宽松的，确保首尾相接不会因浮点精度或周线跳空而失败
            """
            if p2 == 0:
                return abs(p1 - p2) < 0.02
            abs_diff = abs(p1 - p2)
            rel_tolerance = abs_diff / max(abs(p1), abs(p2), 0.01)
            return abs_diff < 0.02 or rel_tolerance < 0.10

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
        #
        # 修复（P0-1）: W4确认需要向上笔突破幅度 > W4向下幅度的38.2%
        # 修复（P2-2）: 新增w4_in_progress状态，区分进行中vs已确认
        w4_start = w3_end
        w4_end = None
        w4_end_seq = None
        w4_candidate_low = None
        w4_candidate_seq = None
        w4_down_length = 0.0  # W4向下幅度（用于幅度检查）

        after_w3 = [b for b in bis if b.seq > w3_end_seq]

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
                        # 向下笔的幅度应该是正值（高点-低点）
                        w4_down_length = b.start_price - b.end_price
            else:  # up bi
                # 第一个向上笔出现 → 检查是否满足W4确认条件
                # 修复（P0-1）: 要求向上笔幅度 > W4向下幅度的38.2%
                if w4_candidate_low is not None:
                    # 计算向上笔的幅度
                    up_amplitude = b.end_price - b.start_price

                    # 修复（P0-1）: W4确认需要向上笔突破一定幅度
                    # 判断条件：向上笔幅度 > W4向下幅度 × 38.2%
                    min_up_ratio = self.W4_CONFIRM_MIN_RATIO  # 0.382
                    required_min_up = w4_down_length * min_up_ratio

                    if up_amplitude >= required_min_up:
                        # 幅度满足要求，W4确认
                        if eq(b.start_price, w4_candidate_low):
                            w4_end = w4_candidate_low
                            w4_end_seq = w4_candidate_seq
                        elif abs(b.start_price - w4_candidate_low) < 0.02:
                            # 容差连接
                            w4_end = b.start_price
                            w4_end_seq = w4_candidate_seq
                        break
                    else:
                        # 幅度不足，W4未确认（进入w4_in_progress状态）
                        # 仅记录candidate_low，但不设置w4_end
                        self.snapshot.w4_candidate_low = w4_candidate_low
                        self._set_state('w4_in_progress', waves)
                        return {'state': 'w4_in_progress', 'waves': waves}

        # 如果找到有效的W4终点
        if w4_end:
            waves['W4'] = {'start': w4_start, 'end': w4_end}
            self.snapshot.w4_end = w4_end
            self.snapshot.w4_candidate_low = None  # 确认后清空candidate
            state = 'w4_formed'
        else:
            # W4candidate存在但未确认，设置w4_in_progress状态
            if w4_candidate_low is not None:
                self.snapshot.w4_candidate_low = w4_candidate_low
                self._set_state('w4_in_progress', waves)
                return {'state': 'w4_in_progress', 'waves': waves}

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
        # P0-1: 使用波浪结构判断趋势方向，而非最后一根笔的方向
        self.snapshot.direction = self._get_trend_direction()
        self._set_state(state, waves)
        return {'state': state, 'waves': waves}

    def _set_state(self, state: str, waves: Dict = None):
        """设置状态"""
        self.state = state
        self.snapshot.state = state

    def _get_trend_direction(self) -> str:
        """基于波浪结构判断真实趋势方向"""
        state = self.state
        if state in ['w3_formed', 'w4_formed', 'w4_in_progress']:
            return 'long'
        elif state == 'w5_formed':
            return 'neutral'
        return 'neutral'

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
        elif state == 'w4_formed' or state == 'w4_in_progress':
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
        last_bi = self.bis[-1] if self.bis else None

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
        # 新信号体系：假设 + 持续验证
        # 底分型出现 → W4终结 ALERT（假设）
        # 后续验证 → CONFIRMED / INVALID
        #
        # 修复（P1-1）: 增加向上笔有效突破验证 + W4幅度斐波那契比例检查
        # 特征：W4内部完成abc三浪或更多次级结构 + 底分型形成 + 向上笔突破验证

        # 检查是否已有活跃的W4_BUY信号（避免重复）
        existing_w4_signals = [sig for sig in self.active_signals
                               if sig.signal == 'W4_BUY' and sig.status == SignalStatus.ALERT]

        if state == 'w4_in_progress' and s.last_fx_mark == 'D' and s.w4_candidate_low:
            # 底分型出现在W4进行中 → 假设W4终结，发行ALERT
            if not existing_w4_signals:
                # 检查是否应该INVALID之前的W4信号
                # W4进行中时，新低是正常的（W4回调本身就是下降）
                # 只有当W4跌破W1高点时，才说明推动结构被破坏，应该INVALID
                for sig in self.active_signals:
                    if sig.signal == 'W4_BUY' and sig.status == SignalStatus.ALERT:
                        if 'W1' in waves and s.w4_candidate_low < waves['W1']['end']:
                            sig.status = SignalStatus.INVALID
                            self._determine_exit_action(
                                sig, s.w4_candidate_low,
                                f"W4跌破W1高点{waves['W1']['end']:.2f}，推动浪结构破坏"
                            )

                # 发行新的W4_BUY ALERT
                # 嵌套验证：只有nested_verified=True时才发行信号
                if self._nested_verify('W4'):
                    new_signal = WaveSignal(
                        signal='W4_BUY',
                        status=SignalStatus.ALERT,
                        price=s.w4_candidate_low,
                        stop_loss=round(s.fib_618, 2) if s.fib_618 else round(s.w4_candidate_low * 0.97, 2),
                        verify_conditions=['price_holds_w4_start', 'upward_break'],
                        verified_conditions=[],
                        reason=f"W4底分型确认，候选低点{s.w4_candidate_low:.2f}",
                        confidence=0.50,
                        created_date=self.bis[-1].end_date if self.bis else '',
                        wave_type='W4'
                    )
                    self.active_signals.append(new_signal)
                    s.wave_end_signal = 'W4_END'
                    s.wave_end_confidence = 0.50

        # ----- 改动2: W4_BUY 备选激活路径（更宽松条件）-----
        # 路径1: 深度回撤（接近W3起点，W4低点靠近W3起点）
        # 路径2: a-b-c三浪形态（W4内部完成abc调整结构）
        if state == 'w4_in_progress' and s.w4_candidate_low:
            if not existing_w4_signals:
                w3_end = waves.get('W3', {}).get('end')

                # 路径1: 深度回撤 - W4低点接近W3起点（距离<5%）
                if w3_end:
                    proximity_to_w3 = abs(s.w4_candidate_low - w3_end) / w3_end
                    if proximity_to_w3 < 0.05:
                        # 深度回撤确认W4终结
                        # 嵌套验证：只有nested_verified=True时才发行信号
                        if self._nested_verify('W4'):
                            new_signal = WaveSignal(
                                signal='W4_BUY',
                                status=SignalStatus.ALERT,
                                price=s.w4_candidate_low,
                                stop_loss=round(s.fib_618, 2) if s.fib_618 else round(s.w4_candidate_low * 0.97, 2),
                                verify_conditions=['price_holds_w4_start'],
                                verified_conditions=[],
                                reason=f"W4深度回撤接近W3起点({proximity_to_w3:.1%})，候选低点{s.w4_candidate_low:.2f}",
                                confidence=0.45,
                                created_date=self.bis[-1].end_date if self.bis else '',
                                wave_type='W4'
                            )
                            self.active_signals.append(new_signal)
                            if not s.wave_end_signal:
                                s.wave_end_signal = 'W4_END'
                                s.wave_end_confidence = 0.45

                # 路径2: a-b-c三浪形态（W4内部完成abc调整）
                # 判断标准：W4经历3次明显的高低点摆动
                if self._detect_abc_structure(waves):
                    # 嵌套验证：只有nested_verified=True时才发行信号
                    if self._nested_verify('W4'):
                        new_signal = WaveSignal(
                            signal='W4_BUY',
                            status=SignalStatus.ALERT,
                            price=s.w4_candidate_low,
                            stop_loss=round(s.fib_618, 2) if s.fib_618 else round(s.w4_candidate_low * 0.97, 2),
                            verify_conditions=['price_holds_w4_start', 'upward_break'],
                            verified_conditions=[],
                            reason=f"W4 a-b-c三浪完成，候选低点{s.w4_candidate_low:.2f}",
                            confidence=0.50,
                            created_date=self.bis[-1].end_date if self.bis else '',
                            wave_type='W4'
                        )
                        self.active_signals.append(new_signal)
                        if not s.wave_end_signal:
                            s.wave_end_signal = 'W4_END'
                            s.wave_end_confidence = 0.50

        # w4_formed状态的W4终结检测（旧逻辑保留，用于CONFIRMED）
        if state == 'w4_formed' and s.last_fx_mark == 'D':
            # 底分型 + W4不破W1高点 → W4可能终结
            if 'W1' in waves and s.w4_end is not None:
                w1_end = waves['W1']['end']
                w3_end = waves.get('W3', {}).get('end')

                # W4低点没跌破W1终点
                if s.w4_end > w1_end:
                    # 修复（P1-1）: 增加向上笔突破验证
                    # 检查是否有有效的向上笔突破W4终点
                    if self._verify_w4_upward_break(waves):
                        # 修复（P1-1）: 增加W4幅度斐波那契比例检查
                        # W4幅度应该在W3的38.2%以内才算正常回调
                        if w3_end:
                            w4_len = s.w4_end - w3_end
                            w3_len = w3_end - waves.get('W3', {}).get('start', w3_end)
                            if w3_len > 0:
                                w4_ratio = abs(w4_len) / w3_len
                                # W4幅度超过W3的61.8%可能是深幅回调，降低置信度
                                if w4_ratio <= 0.618:
                                    s.wave_end_signal = 'W4_END'
                                    s.wave_end_confidence = 0.70
                                elif w4_ratio <= 0.90:
                                    # 深幅回调但仍在合理范围
                                    s.wave_end_signal = 'W4_END'
                                    s.wave_end_confidence = 0.55
                                # 超过90%可能是趋势破坏，不发信号
                            else:
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
                if 0.40 <= retracement <= 0.70:
                    # W2量能萎缩
                    w2_vol = self._get_wave_volume('W2')
                    w1_vol = self._get_wave_volume('W1')
                    if w2_vol > 0 and w1_vol > 0 and w2_vol < w1_vol * 0.6:
                        if s.last_fx_mark == 'D':
                            s.wave_end_signal = 'W2_END'
                            s.wave_end_confidence = 0.75

                            # 新信号体系：W2底分型出现 → W2_BUY ALERT
                            existing_w2_signals = [sig for sig in self.active_signals
                                                   if sig.signal == 'W2_BUY' and sig.status == SignalStatus.ALERT]
                            if not existing_w2_signals:
                                # 嵌套验证：只有nested_verified=True时才发行信号
                                if self._nested_verify('W2'):
                                    new_signal = WaveSignal(
                                        signal='W2_BUY',
                                        status=SignalStatus.ALERT,
                                        price=w2_end,
                                        stop_loss=round(w2_end * 0.97, 2),
                                        verify_conditions=['price_holds_w2_low', 'upward_momentum'],
                                        verified_conditions=[],
                                        reason=f"W2回撤{w2_end:.2f}，底分型确认",
                                        confidence=0.65,
                                        created_date=last_bi.end_date if last_bi else '',
                                        wave_type='W2'
                                    )
                                    self.active_signals.append(new_signal)

    def _determine_exit_action(self, sig: WaveSignal, current_price: float, reason: str):
        """
        根据盈亏状态确定INVALID时的操作指令和出场价格

        - 盈利状态（current_price > sig.price）→ 止盈离场，以当前价格出场
        - 亏损/保本状态 → 止损出场，以止损价出场
        """
        if current_price > sig.price:
            sig.action = "止盈离场"
            sig.exit_price = current_price
        else:
            sig.action = "止损出场"
            sig.exit_price = sig.stop_loss if sig.stop_loss else current_price
        sig.invalidated_reason = reason

    def _verify_signals(self, wave_result: Dict):
        """
        验证活跃信号的验证条件

        核心原则：假设 + 持续验证
        - 每个新数据点都检查活跃信号的验证条件
        - 满足 → CONFIRMED
        - 失败 → INVALID
        - 新ALERT出现 → 可能替代旧的或新增
        """
        s = self.snapshot
        state = wave_result.get('state', 'initial')
        waves = wave_result.get('waves', {})

        if not self.active_signals:
            return

        last_bi = self.bis[-1] if self.bis else None
        if not last_bi:
            return

        current_price = last_bi.end_price
        current_low = last_bi.end_price if last_bi.direction == 'down' else last_bi.start_price

        for sig in self.active_signals:
            if sig.status != SignalStatus.ALERT:
                # 只验证ALERT状态的信号
                continue

            # 跳过刚创建的信号（在同一_bi循环中创建）
            # 这类信号在下一次_recalc时再验证
            if sig.created_date == (last_bi.end_date if last_bi else ''):
                continue

            # ===== W4_BUY 信号验证 =====
            if sig.signal == 'W4_BUY':
                # 验证条件1: 价格不跌破W4起点（W4候选低点）
                # W4起点 = W4开始回调的价格 = waves['W4']['start']
                # 如果W4还未确认（w4_in_progress），用W3终点作为W4起点
                if 'W4' in waves:
                    w4_start = waves['W4']['start']
                elif state == 'w4_in_progress' and 'W3' in waves:
                    w4_start = waves['W3']['end']
                else:
                    w4_start = None

                # 撤销条件：W4跌破W1高点（结构性破坏），而不是W4正常下跌
                # W4从高点正常回调下跌是预期行为，不应INVALID
                if 'W1' in waves and current_low < waves['W1']['end']:
                    sig.status = SignalStatus.INVALID
                    self._determine_exit_action(
                        sig, current_low,
                        f"W4跌破W1高点{waves['W1']['end']:.2f}，推动浪结构破坏"
                    )
                    continue

                # 验证条件2: 向上笔突破W4候选低点（确认W4终结）
                # 如果当前向上笔的终点 > sig.price，说明假设成立
                if last_bi.direction == 'up' and last_bi.end_price > sig.price:
                    if 'upward_break' not in sig.verified_conditions:
                        sig.verified_conditions.append('upward_break')

                # 所有验证条件满足 → CONFIRMED
                if len(sig.verified_conditions) >= len(sig.verify_conditions):
                    sig.status = SignalStatus.CONFIRMED
                    sig.action = "持仓待涨"
                    sig.confidence = 0.75

                # 如果价格继续下跌创出新低（低于候选低点），假设新的W4终点
                # 这种情况下，旧的ALERT失效，假设新的低点，发行新的ALERT
                if sig.wave_type == 'W4' and current_low < sig.price:
                    sig.status = SignalStatus.INVALID
                    self._determine_exit_action(
                        sig, current_low,
                        f"价格跌破假设终点{sig.price:.2f}，新低点{current_low:.2f}"
                    )

                    # 发行新的W4_BUY ALERT at new low
                    # 嵌套验证：只有nested_verified=True时才发行信号
                    if self._nested_verify('W4'):
                        new_signal = WaveSignal(
                            signal='W4_BUY',
                            status=SignalStatus.ALERT,
                            price=current_low,
                            stop_loss=round(s.fib_618, 2) if s.fib_618 else round(current_low * 0.97, 2),
                            verify_conditions=['price_holds_w4_start', 'upward_break'],
                            verified_conditions=[],
                            reason=f"W4新候选低点{current_low:.2f}",
                            confidence=0.40,
                            created_date=last_bi.end_date if last_bi else '',
                            wave_type='W4'
                        )
                        self.active_signals.append(new_signal)

            # ===== W2_BUY 信号验证 =====
            elif sig.signal == 'W2_BUY':
                # 验证条件: 价格不跌破W2低点（sig.price）
                if current_low < sig.price:
                    sig.status = SignalStatus.INVALID
                    self._determine_exit_action(
                        sig, current_low,
                        f"价格跌破W2低点{sig.price:.2f}"
                    )
                    continue

                # 价格在W2低点上方的向上笔 → 确认
                if last_bi.direction == 'up' and last_bi.end_price > sig.price:
                    sig.status = SignalStatus.CONFIRMED
                    sig.action = "持仓待涨"
                    sig.confidence = 0.80

    def _verify_w4_upward_break(self, waves: Dict) -> bool:
        """
        修复（P1-1）: 验证W4后是否有有效的向上笔突破

        W4终结确认需要：
        1. 出现向上笔突破W4终点（首位相接）
        2. 向上笔的幅度足够大（后续可以通过W4_CONFIRM_MIN_RATIO判断）

        Returns:
            True if there's a valid upward break after W4
        """
        if 'W4' not in waves or 'W3' not in waves:
            return False

        w4_end = waves['W4']['end']
        w4_end_seq = None

        # 找到W4终点对应的笔序号
        for i, bi in enumerate(self.bis):
            if bi.seq > 0 and abs(bi.end_price - w4_end) < 0.02:
                w4_end_seq = bi.seq
                break

        if w4_end_seq is None:
            return False

        # 检查W4之后是否有向上笔突破W4终点
        after_w4 = [b for b in self.bis if b.seq > w4_end_seq]
        for b in after_w4:
            if b.direction == 'up' and b.end_price > w4_end:
                # 找到有效向上突破
                return True

        return False

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

    def _nested_verify(self, wave_label: str) -> bool:
        """
        嵌套验证：通过wave_recognizer.label_wave_stage()检查波浪嵌套是否通过。
        用于在发行W2_BUY/W4_BUY ALERT信号前进行增强过滤。

        Args:
            wave_label: 'W2' 或 'W4'

        Returns:
            True: 嵌套验证通过，或label_wave_stage调用失败（容错）
            False: 嵌套验证未通过，应放弃该信号
        """
        try:
            import datetime
            from projects.stock_wave_recognition.wave_recognizer import label_wave_stage
            current_year = datetime.datetime.now().year
            label_result = label_wave_stage(self.symbol, current_year)
            for lw in label_result.get('labeled_waves', []):
                if lw.get('wave') == wave_label and lw.get('nested_verified', False):
                    return True
            return False
        except Exception:
            # 容错：识别失败时不过滤，避免因数据问题阻断正常信号
            return True

    def _detect_abc_structure(self, waves: Dict) -> bool:
        """
        检测W4内部是否完成a-b-c三浪调整结构（改动2备选路径）

        a-b-c三浪调整特征：
        - 向下a浪：第1个向下笔/段
        - 向上b浪：反弹过a浪起点或达到a浪的38.2%
        - 向下c浪：新低（或接近a浪低点）
        - 底分型确认

        这里用简化方法：检查最近是否经历了至少3次明显的高低点摆动
        """
        recent_bis = self.bis[-12:] if len(self.bis) >= 12 else self.bis
        if len(recent_bis) < 6:
            return False

        # 提取最近的高低点序列
        turning_points = []
        for b in recent_bis:
            if b.direction == 'down':
                turning_points.append(('low', b.end_price))
            else:
                turning_points.append(('high', b.end_price))

        # 需要至少3次完整的高低点摆动（a-b-c基础结构）
        if len(turning_points) < 6:
            return False

        # 检查是否是a-b-c结构特征：
        # - a浪：第1个低点
        # - b浪：随后的高点
        # - c浪：随后的低点（创新低或不创新低）
        lows = [p[1] for p in turning_points if p[0] == 'low']
        highs = [p[1] for p in turning_points if p[0] == 'high']

        if len(lows) >= 2 and len(highs) >= 1:
            # a浪低点和c浪低点（c应该接近或低于a）
            # b浪高点应该明显高于a浪低点
            a_low = lows[0]
            b_high = highs[0]
            if len(lows) >= 2:
                c_low = lows[1]
                # b浪反弹幅度至少是a浪的38.2%
                b_retrace = (b_high - a_low) / a_low
                # c浪不应该明显低于a浪（否则可能是推动浪，不是调整）
                c_drop = (a_low - c_low) / a_low
                if b_retrace >= 0.20 and c_drop >= -0.10:  # c低不低于a的10%以上
                    return True

        return False

    # =====================================================================
    # Phase 1.2: 买卖点信号生成
    # =====================================================================

    def get_buy_sell_signals(self) -> Dict:
        """
        生成买卖点信号（Phase 1.2）

        返回格式（新信号体系）：
        {
            'signal': str,              # 信号类型
            'status': str,              # ALERT / CONFIRMED / INVALID
            'price': float,             # 建议价格
            'stop_loss': float,         # 止损位
            'reason': str,              # 信号理由
            'confidence': float,        # 置信度 0.0~1.0
            'verify_conditions': list,  # 需要验证的条件
            'verified_conditions': list, # 已通过的验证
            'invalidated_reason': str,  # 失效原因（INVALID时）
            'wave_type': str,           # 浪型
        }

        信号状态（假设+验证模式）：
        - ALERT: 假设终结出现，需后续数据验证
        - CONFIRMED: 验证条件全部满足
        - INVALID: 验证条件失败
        """
        s = self.snapshot
        result = {
            'signal': 'NO_SIGNAL',
            'status': '',
            'price': None,
            'stop_loss': None,
            'reason': '',
            'confidence': 0.0,
            'verify_conditions': [],
            'verified_conditions': [],
            'invalidated_reason': '',
            'wave_type': '',
        }

        # 根据当前波浪状态生成信号
        state = s.state

        # 首先检查活跃信号（ALERT/CONFIRMED）
        if self.active_signals:
            # 找最新的有效信号（排除已完成的）
            active = [sig for sig in self.active_signals
                     if sig.status in (SignalStatus.ALERT, SignalStatus.CONFIRMED)]

            if active:
                # 返回最新的活跃信号
                latest = active[-1]
                result['signal'] = latest.signal
                result['status'] = latest.status
                result['price'] = latest.price
                result['stop_loss'] = latest.stop_loss
                result['reason'] = latest.reason
                result['confidence'] = latest.confidence
                result['verify_conditions'] = latest.verify_conditions
                result['verified_conditions'] = latest.verified_conditions
                result['invalidated_reason'] = latest.invalidated_reason
                result['wave_type'] = latest.wave_type
                return result
            else:
                # 有INVALID信号但没有新的ALERT，说明之前的假设已失败
                # 返回最新的INVALID信号作为参考
                latest_invalid = self.active_signals[-1]
                result['signal'] = latest_invalid.signal
                result['status'] = SignalStatus.INVALID
                result['price'] = latest_invalid.price
                result['stop_loss'] = latest_invalid.stop_loss
                result['reason'] = latest_invalid.invalidated_reason or latest_invalid.reason
                result['confidence'] = 0.0
                result['verify_conditions'] = latest_invalid.verify_conditions
                result['verified_conditions'] = latest_invalid.verified_conditions
                result['invalidated_reason'] = latest_invalid.invalidated_reason
                result['wave_type'] = latest_invalid.wave_type
                return result

        # ----- W5卖信号 -----
        if s.wave_end_signal == 'W5_END' and s.w5_end:
            result['signal'] = 'W5_SELL'
            result['status'] = SignalStatus.CONFIRMED
            result['price'] = s.w5_end
            # 止损：W5高点的103%（小幅止损）
            result['stop_loss'] = round(s.w5_end * 1.03, 2)
            result['reason'] = self._build_w5_sell_reason()
            result['confidence'] = s.wave_end_confidence
            result['verify_conditions'] = []
            result['verified_conditions'] = []
            result['wave_type'] = 'W5'
            return result

        # ----- W4买确认（旧逻辑保留，用于CONFIRMED情况）-----
        # 修复（P0-2）: 止损改为fib_618而非w4_end×0.97
        # 条件：W4终结信号确认 + 底分型形成
        if s.wave_end_signal == 'W4_END' and s.w4_end:
            result['signal'] = 'W4_BUY'
            result['status'] = SignalStatus.CONFIRMED
            result['price'] = s.w4_end
            # 修复（P0-2）: 使用fib_618作为止损位（更科学）
            if s.fib_618:
                result['stop_loss'] = round(s.fib_618, 2)
            else:
                result['stop_loss'] = round(s.w4_end * 0.97, 2)
            result['reason'] = f"W4调整结束，底分型确认，价格{s.w4_end:.2f}，止损{s.stop_loss:.2f}"
            result['confidence'] = s.wave_end_confidence
            result['verify_conditions'] = ['price_holds_w4_start']
            result['verified_conditions'] = ['price_holds_w4_start']
            result['wave_type'] = 'W4'
            return result

        # ----- W4买预警 -----
        # 修复（P2-2）: 同时处理w4_formed和w4_in_progress状态
        # 修复（P0-1）: W4_IN_PROGRESS时，如果有candidate_low也要能发出预警
        if state == 'w4_formed' and s.w4_end:
            result['signal'] = 'W4_BUY'
            result['status'] = SignalStatus.ALERT
            result['price'] = s.bi_low  # 当前价（近似）
            # 修复（P0-2）: W4进行中时，止损也用fib_618
            if s.fib_618:
                result['stop_loss'] = round(s.fib_618, 2)
            else:
                result['stop_loss'] = round(s.w4_end * 0.97, 2)
            result['reason'] = f"W4调整中，关注{s.fib_382:.2f}/{s.fib_500:.2f}/{s.fib_618:.2f}支撑"
            result['confidence'] = 0.5
            result['verify_conditions'] = ['price_holds_w4_start']
            result['verified_conditions'] = []
            result['wave_type'] = 'W4'
            return result
        elif state == 'w4_in_progress' and s.w4_candidate_low:
            # W4未确认（向上笔幅度不足），发出预警
            result['signal'] = 'W4_BUY'
            result['status'] = SignalStatus.ALERT
            result['price'] = s.w4_candidate_low  # 候选低点
            if s.fib_618:
                result['stop_loss'] = round(s.fib_618, 2)
            else:
                result['stop_loss'] = round(s.w4_candidate_low * 0.97, 2)
            result['reason'] = f"W4进行中(未确认)，候选低点{s.w4_candidate_low:.2f}，关注{s.fib_382:.2f}/{s.fib_500:.2f}/{s.fib_618:.2f}支撑"
            result['confidence'] = 0.35  # 降低置信度，因为W4未确认
            result['verify_conditions'] = ['price_holds_w4_start', 'upward_break']
            result['verified_conditions'] = []
            result['wave_type'] = 'W4'
            return result

        # ----- W2买信号 -----
        if s.wave_end_signal == 'W2_END' and s.w2_end:
            result['signal'] = 'W2_BUY'
            result['status'] = SignalStatus.ALERT
            result['price'] = s.w2_end
            # 止损：W2低点的103%
            result['stop_loss'] = round(s.w2_end * 0.97, 2)
            result['reason'] = f"W2回撤结束，底分型确认，价格{s.w2_end:.2f}"
            result['confidence'] = s.wave_end_confidence
            result['verify_conditions'] = ['price_holds_w2_low']
            result['verified_conditions'] = []
            result['wave_type'] = 'W2'
            return result

        # ----- C浪买信号（熊市）-----
        if s.wave_end_signal == 'C_END' and s.c_wave_5_struct:
            result['signal'] = 'C_BUY'
            result['status'] = SignalStatus.CONFIRMED
            result['price'] = s.bi_low
            result['stop_loss'] = round(s.bi_low * 0.97, 2)
            result['reason'] = 'C浪5浪完成，熊市调整结束，可能反转'
            result['confidence'] = s.wave_end_confidence
            result['verify_conditions'] = []
            result['verified_conditions'] = []
            result['wave_type'] = 'C'
            return result

        # ----- 无信号 -----
        result['signal'] = 'NO_SIGNAL'
        result['status'] = ''
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

        # W4进行中：修复（P0-2）使用fib_618作为止损（更科学）
        # 修复（P2-2）: 同时处理w4_formed和w4_in_progress状态
        if (state == 'w4_formed' or state == 'w4_in_progress') and s.fib_618:
            sl = round(s.fib_618, 2)
            s.stop_loss = sl
            s.stop_loss_type = 'fib618'
            return
        elif (state == 'w4_formed' or state == 'w4_in_progress') and s.w4_end:
            # fallback到w4_end×0.97
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
        elif s.state == 'w4_in_progress':
            return f"W4进行中(候选:{s.w4_candidate_low:.2f})" if s.w4_candidate_low else "W4进行中"
        return s.state


# ======================
# PositionManager (P0-4)
# ======================

class PositionManager:
    """持仓管理器 - P1风控增强"""

    def __init__(self):
        self.position = None
        # P1-3: 信号有效期窗口控制
        self.signal_window_days = 10  # 交易日窗口（自然日约14天）
        # P1-5: 最大持仓天数
        self.max_hold_days = 20  # 最大持仓天数（20交易日≈1个月，不做短线客）

    # P1-3: 信号有效期窗口控制
    def check_window_expired(self) -> bool:
        """
        检查信号是否过期

        逻辑：周线预兆后日线在 signal_window_days（10交易日）内算有效买点
        超过窗口期，无论盈亏都应考虑离场
        """
        if self.position is None:
            return False
        days_held = (datetime.now() - self.position["entry_time"]).days
        return days_held > self.signal_window_days

    # P1-6: 仓位管理
    def calc_position_size(self, account_balance: float,
                            risk_ratio: float = 0.02) -> float:
        """
        计算仓位

        基于固定风险比例计算买入股数：
        - 每笔交易风险敞口 = 账户余额 × risk_ratio（默认2%）
        - 股数 = 风险敞口 / 单股止损距离

        Args:
            account_balance: 账户余额
            risk_ratio: 风险比例，默认2%

        Returns:
            建议买入股数（整百）
        """
        if self.position is None:
            return 0
        entry_price = self.position["entry_price"]
        stop_loss = self.position["stop_loss"]

        if stop_loss >= entry_price:
            return 0  # 止损价设置错误

        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            return 0

        # 风险金额
        risk_amount = account_balance * risk_ratio
        # 可买入股数
        shares = risk_amount / stop_distance

        # V3: 乘以 position_size_ratio（波浪仓位比例）
        position_size_ratio = self.position.get("position_size_ratio", 1.0)
        shares = shares * position_size_ratio

        # 整百取整（A股最少100股）
        shares = int(shares // 100 * 100)
        return max(shares, 100)

    def entry_long(self, price: float, stop_loss: float,
                   verify_price: float,
                   position_size_ratio: float = 1.0):
        """买入 + 制定卖出计划"""
        self.position = {
            "entry_price": price,
            "stop_loss": stop_loss,
            "verify_price": verify_price,  # 必须突破的价格（假突破判断用）
            "entry_time": datetime.now(),
            "position_size_ratio": position_size_ratio,  # V3: 波浪仓位比例
        }

    # P1-4: 假突破应对
    # P1-5: 时间止损
    def verify_position(self, current_price: float,
                       daily_czsc: CZSC,
                       high_after_entry: float = None) -> str:
        """
        持续验证持仓状态

        验证优先级：
        1. 止损（跌破止损价）
        2. 假突破离场（突破后回落至verify_price以下）
        3. 时间到期离场（超窗口期/超最大持仓期）
        4. 正常持有

        Args:
            current_price: 当前价格
            daily_czsc: 日线CZSC对象（用于缠论顶分型止盈）
            high_after_entry: 入场后最高价（用于假突破判断）
        """
        if self.position is None:
            return "无持仓"

        # ===== 1. 止损（优先用信号自带的W2动态止损位，其次用固定百分比） =====
        entry_price = self.position["entry_price"]
        dyn_stop = self.position.get("stop_loss", 0)
        if dyn_stop and dyn_stop > 0 and dyn_stop < entry_price:
            stop_price = dyn_stop
        else:
            # 保险：dyn_stop 无效（>=买入价或为0），用固定8%止损
            stop_price = entry_price * 0.92
        if current_price < stop_price:
            return "止损出场"

        # ===== P1-4: 2. 假突破离场 =====
        # 价格短暂突破verify_price后又跌回来，说明是假突破
        if high_after_entry is not None:
            verify_price = self.position["verify_price"]
            entry_price = self.position["entry_price"]
            # 条件：入场后曾突破verify_price，且当前价格跌回verify_price以下
            # 注意：只有在verify_price > entry_price（真突破需上涨）时才判断
            if (verify_price > entry_price and
                    high_after_entry > verify_price and
                    current_price < verify_price):
                return "假突破离场"

        # ===== P1-5: 3. 时间止损 =====
        # 两个维度：
        # a) max_hold_days（20交易日）：最大持仓期，持仓太久必须走
        # b) signal_window_days（10交易日）：信号有效期（仅无持仓时有效）
        days_held = (datetime.now() - self.position["entry_time"]).days
        
        # 最大持仓期优先检查
        if days_held > self.max_hold_days:
            if current_price > self.position["entry_price"]:
                return "超最大持仓期，盈利离场"
            else:
                return "超最大持仓期，保本离场"
        
        # 信号窗口期（仅用于判断是否放弃等待入场）
        if self.check_window_expired():
            if current_price > self.position["entry_price"]:
                return "时间到期，盈利离场"
            else:
                return "时间到期，保本离场"

        # ===== 4. 顶分型止盈（原有逻辑）=====
        if daily_czsc and daily_czsc.has_top_fx():
            return "止盈离场"

        # ===== 5. 正常持有 =====
        return "继续持有"

    def get_exit_action(self, current_price: float,
                        entry_price: float) -> tuple:
        """判断出场类型和价格"""
        if self.position is None:
            return None, None

        if current_price > entry_price:
            return "止盈离场", current_price
        else:
            # P0-5: 修复变量引用错误，统一使用 self.position["stop_loss"]
            return "止损出场", self.position["stop_loss"]

    def has_position(self) -> bool:
        return self.position is not None


# ======================
# WaveEngine 双周期架构 (Phase 1)
# ======================

class WaveEngine:
    """
    WaveChan V3 双周期波浪引擎

    双周期架构：
    - 日线周期：精细笔识别，用于精确买卖点
    - 周线周期：判断大势趋势，用于方向确认

    P0-3 修复：self.daily_czsc = CZSC([]) 避免 feed_daily 报错
    """

    def __init__(self, symbol: str, cache_dir: str = None):
        self.symbol = symbol
        self.cache_dir = cache_dir or f"/tmp/wavechan_cache/{symbol}"
        os.makedirs(self.cache_dir, exist_ok=True)

        # P0-3 修复: 延迟初始化 CZSC（避免 CZSC([]) 空列表 panic）
        self.daily_czsc = None

        # 日线 + 周线波浪计数器
        self.daily_cache = SymbolWaveCache(symbol, self.cache_dir)
        self.weekly_cache = None  # 周线缓存，按需初始化

        # 持仓管理器 (P0-4)
        self.position_manager = PositionManager()

        # 当前状态
        self.last_date: Optional[str] = None
        self.last_snapshot: Optional[WaveSnapshot] = None

        # 周线聚合缓冲（用于累积日线→周线）
        self._pending_daily_bars: deque = deque(maxlen=500)   # 待聚合的日线 bars
        self._current_week: Optional[int] = None              # 当前日线所属周（isocalendar week）

        # 尝试加载缓存
        self.daily_cache.load()

    def feed_daily(self, bar: dict) -> WaveSnapshot:
        """
        喂入一根日K线，触发双周期重新计算

        Args:
            bar: {'date': '2026-03-01', 'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}

        Phase 2 (weekly aggregation):
          - 累积日线 bars 到 _pending_daily_bars
          - 当周结束（周五或月末）时，聚合为周线 OHLCV 并喂入 weekly_cache
          - weekly_cache 使用 CZSC 识别周线笔，进而计算周线方向
        """
        # 1. 更新日线 CZSC（用于分型判断）
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
        # P0-3 修复: 延迟初始化 CZSC（首根K线时）
        if self.daily_czsc is None:
            self.daily_czsc = CZSC([raw_bar])
        else:
            self.daily_czsc.update(raw_bar)

        # 2. 喂入日线波浪计数器
        snap = self.daily_cache.feed_bar(bar)
        self.last_date = bar['date']
        self.last_snapshot = snap

        # 3. 周线聚合：累积日线 → 周线 bar
        # ─────────────────────────────────────────────────────────────
        current_date = pd.to_datetime(bar['date'])
        current_week = current_date.isocalendar().week
        is_week_end = (current_date.dayofweek == 4)  # 周五
        is_month_end = (current_date.day >= 25 and current_date.day <= 31)

        # 累积日线 bar
        self._pending_daily_bars.append(bar)

        # 初始化周线缓存（延迟加载/创建）
        if self.weekly_cache is None:
            self.weekly_cache = SymbolWaveCache(f"{self.symbol}_W", self.cache_dir)
            self.weekly_cache.load()
            self._current_week = current_week

        # 检测周切换：当前 bar 是周五/月末 OR 进入了新的一周
        week_changed = (self._current_week is not None and current_week != self._current_week)

        if (is_week_end or is_month_end or week_changed) and len(self._pending_daily_bars) >= 5:
            try:
                # 取最近一周的 bars（避免跨月数据混淆）
                bars_to_aggregate = list(self._pending_daily_bars)
                weekly_bar = self._aggregate_weekly_bar(bars_to_aggregate)

                if weekly_bar:
                    # 喂入周线计数器（CZSC 识别周线笔 → WaveCounterV3 计算方向）
                    self.weekly_cache.feed_bar(weekly_bar, freq=Freq.W)
                    logger.debug(f"[{self.symbol}] 周线已更新: {weekly_bar['date']} "
                                f"OHLCV=({weekly_bar['open']:.2f},{weekly_bar['high']:.2f},"
                                f"{weekly_bar['low']:.2f},{weekly_bar['close']:.2f})")

                # 更新当前周标记
                self._current_week = current_week
                # 清空缓冲（保留当前 bar，因为它属于新一周）
                self._pending_daily_bars.clear()
                self._pending_daily_bars.append(bar)

            except Exception as e:
                logger.warning(f"[{self.symbol}] 周线聚合失败: {e}")

        return snap

    def _aggregate_weekly_bar(self, daily_bars: list) -> dict:
        """
        将日线 bars 聚合为一根周线 OHLCV bar

        Args:
            daily_bars: [{date, open, high, low, close, volume}, ...]

        Returns:
            weekly_bar: {date, open, high, low, close, volume}
            - date: 取最后一根日线的日期
            - open: 第一根日线的 open
            - high: max(high)
            - low: min(low)
            - close: 最后一根日线的 close
            - volume: sum(volume)
        """
        if not daily_bars or len(daily_bars) < 1:
            return None

        bars_df = pd.DataFrame(daily_bars)
        bars_df['date'] = pd.to_datetime(bars_df['date'])

        return {
            'date': bars_df['date'].max().strftime('%Y-%m-%d'),
            'open': float(bars_df['open'].iloc[0]),
            'high': float(bars_df['high'].max()),
            'low': float(bars_df['low'].min()),
            'close': float(bars_df['close'].iloc[-1]),
            'volume': float(bars_df['volume'].sum()) if 'volume' in bars_df.columns else 0.0,
        }

    def get_trend(self) -> str:
        """
        获取周线趋势方向（日线方向 + 周线确认）

        Returns:
            'long': 多头趋势
            'short': 空头趋势
            'neutral': 震荡/不确定
        """
        daily_dir = self.last_snapshot.direction if self.last_snapshot else 'neutral'
        weekly_dir = self.get_weekly_dir()

        # 双周期共振判断
        if weekly_dir == 'down':
            return 'short'  # 周线下降，日线也看短
        elif weekly_dir == 'up' and daily_dir in ('up', 'long'):
            return 'long'   # 周线+日线共振做多
        elif daily_dir in ('up', 'long'):
            return daily_dir
        else:
            return 'neutral'

    def get_weekly_dir(self) -> str:
        """
        获取周线趋势方向（用于方向过滤）

        【原则】：使用真正的周K线数据 + WaveCounterV3 波浪状态判断方向
        - 不使用滚动20日统计近似估算
        - 不简化比较两根笔的价格

        判断逻辑（优先级从高到低）：
        1. WaveCounterV3.state 为 w3_formed/w4_formed/w4_in_progress → 'up'
        2. WaveCounterV3.state 为 w5_formed → 结合最后笔方向判断
        3. 有 ≥3 根已完成周线笔：分析笔序列高低点结构（阶梯式抬升/下降）
        4. 只有1-2根周线笔：基于笔序列方向判断
        5. 无周线笔：返回 'neutral'

        Returns:
            'up': 周线上升趋势
            'down': 周线下降趋势
            'neutral': 周线震荡/不确定
        """
        if self.weekly_cache is None:
            return 'neutral'

        try:
            counter = self.weekly_cache.counter
            state = counter.state
            bis = self.weekly_cache.completed_bis

            # ── 优先使用 WaveCounterV3 波浪状态 ─────────────────────────
            if state in ('w3_formed', 'w4_formed', 'w4_in_progress'):
                return 'up'
            elif state == 'w5_formed':
                # W5已确认：结合最后笔方向判断
                if len(bis) >= 1:
                    last_bi = bis[-1]
                    if last_bi.direction == 'up':
                        return 'neutral'  # W5向上，震荡偏多
                    else:
                        return 'down'    # W5向下，已进入调整
                return 'neutral'

            # ── 备选：分析已完成笔序列的方向趋势 ─────────────────────
            # 对于周线数据，使用更鲁棒的判断方式：
            # 1. 看最近笔的方向 majority（去噪声）
            # 2. 结合整体高低点趋势（防震荡）
            # 3. 综合评分得出方向

            recent_bis = bis[-5:] if len(bis) >= 5 else bis  # 最近最多5笔

            # 1) 方向 majority vote
            up_count = sum(1 for b in recent_bis if b.direction == 'up')
            down_count = len(recent_bis) - up_count
            dir_score = up_count - down_count  # 正=偏多，负=偏空

            # 2) 整体高低点趋势（防止震荡误判）
            if len(recent_bis) >= 2:
                first_bi = recent_bis[0]
                last_bi = recent_bis[-1]
                first_range = (min(first_bi.start_price, first_bi.end_price),
                               max(first_bi.start_price, first_bi.end_price))
                last_range = (min(last_bi.start_price, last_bi.end_price),
                              max(last_bi.start_price, last_bi.end_price))

                # 整体上升：最新笔区间在最早笔区间之上
                overall_up = last_range[0] > first_range[0] and last_range[1] > first_range[1]
                overall_down = last_range[1] < first_range[0]  # 完全在下方=空头
            else:
                overall_up = overall_down = False

            # 综合判断
            if dir_score >= 2 or (dir_score >= 1 and overall_up):
                return 'up'
            elif dir_score <= -2 or (dir_score <= -1 and overall_down):
                return 'down'
            else:
                return 'neutral'

        except Exception:
            # 出错时降级，不阻塞交易
            return 'neutral'

    def get_signal(self) -> Dict:
        """
        获取当前信号（含周线方向过滤）

        周线过滤规则：
        - 周线 up / neutral：接受所有买点（W2/W4/C BUY）
        - 周线 down：拒绝 W2_BUY 和 W4_BUY（下跌趋势中的回调是陷阱）
        - C_BUY 作为熊市反转信号，无论周线方向均保留
        """
        signals = self.daily_cache.counter.get_buy_sell_signals()

        # 获取周线方向
        weekly_dir = self.get_weekly_dir()

        # 周线下降趋势：过滤 W2/W4 买点（保留 C_BUY 抄底信号）
        if weekly_dir == 'down':
            sig = signals.get('signal', '')
            if sig in ('W2_BUY', 'W4_BUY'):
                # 降级为 INVALID，记录原因
                signals = dict(signals)
                signals['status'] = 'INVALID'
                signals['invalidated_reason'] = (
                    f'周线下降趋势({weekly_dir})，拒绝{sig}信号'
                )
                signals['confidence'] = 0.0
                logger.info(f"[{self.symbol}] 周线下降，拒绝{sig}信号")

        return signals

    def get_state_str(self) -> str:
        """获取状态描述"""
        return self.daily_cache.counter.get_state_str()


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

    def feed_bar(self, bar: dict, freq: Freq = Freq.D) -> WaveSnapshot:
        """
        喂入一根K线（支持日线/周线）
        bar: {'date': '2026-03-01', 'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}
        freq: Freq.D（日线）或 Freq.W（周线），影响 CZSC 笔识别逻辑
        内部累计High/Low形成笔
        """
        # CZSC 已在模块顶部导入

        # 转换为自己需要的格式
        raw_bar = RawBar(
            symbol=self.symbol,
            dt=pd.to_datetime(bar['date']).to_pydatetime(),
            freq=freq,  # 支持日线/周线频率
            open=float(bar['open']),
            high=float(bar['high']),
            low=float(bar['low']),
            close=float(bar['close']),
            vol=float(bar.get('volume', 0)),
            amount=0.0
        )

        # 累计到临时bars中（用deque避免内存泄漏，最多保留500根）
        if not hasattr(self, '_pending_bars'):
            self._pending_bars = deque(maxlen=500)

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
        通过 CZSC 笔的 bars 属性获取笔内K线
        """
        try:
            # 优先使用 bars 属性（K线列表）
            if hasattr(bi, 'bars') and bi.bars:
                return sum(float(k.vol) for k in bi.bars if hasattr(k, 'vol'))
            # 备选：使用 power_volume 属性
            elif hasattr(bi, 'power_volume') and bi.power_volume:
                return float(bi.power_volume)
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
        possible_dirs = ['/root/.openclaw/workspace/data/warehouse']  # 优化：直接定位

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
        possible_dirs = ['/root/.openclaw/workspace/data/warehouse']  # 优化：直接定位

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
        possible_dirs = ['/root/.openclaw/workspace/data/warehouse']  # 优化：直接定位

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
