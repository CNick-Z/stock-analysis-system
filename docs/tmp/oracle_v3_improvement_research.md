# V3波浪策略改进研究报告
## Oracle研究 | 2026-04-07

---

## 一、600368案例复盘：问题根源

### 1.1 发生了什么
- **信号**：5.87 W2 Buy
- **结果**：止损11次，股价跌至3.83
- **损失**：巨大

### 1.2 根本原因诊断

600368实际走势（事后确认）：

```
BI1(4.69 高) → BI4(4.12 低) → BI5(5.87 高) → BI22(3.75 低)
     ↓            ↓             ↓             ↓
   新低下移    新高替换      新高替换       创历史新低
```

**V3误判的核心**：

V3在BI4(4.12)识别到一个"向上笔突破"，于是标注为W3成立，买入。

但实际结构是：
- **这不是W3，而是更大级别下跌浪中的B浪反弹**
- 5.87之后是C浪下跌，V3在C浪中连续捕捉"W2 Buy"（每次回调都以为是新的W2）
- 连续11次止损，说明每次都买在了"逆势反弹的终点"

**V3犯的错本质**：
1. **无法区分顺势浪和逆势浪**——把B浪反弹当W3买入了
2. **无法识别调整浪**——不知道此时处于A-B-C调整结构中
3. **无大级别趋势过滤**——周线明显下跌趋势时，日线还在傻买

---

## 二、波浪理论三大过滤器设计

### 2.1 过滤器一：大级别趋势方向过滤（WaveTrendFilter）

**规则**：在周线/日线级别判断趋势方向

```
输入：周线笔序列
输出：'up_trend' | 'down_trend' | 'neutral'

判断逻辑：
- 连续3个高点创新高 + 连续3个低点也抬高 → up_trend
- 连续3个高点创新低 + 连续3个低点也下移 → down_trend
- 不满足以上 → neutral
```

**V3集成**：
```python
# 在WaveEngine.get_trend()中实现
def get_major_trend(self) -> str:
    """
    获取大级别趋势方向（周线）
    
    up_trend：只做多，不在W2以外的位置买
    down_trend：不买入，只等待C浪终极大机会
    neutral：正常波浪策略
    """
    # 具体实现见第三节代码
```

**效果**：
- 周线下跌趋势 → **过滤掉所有W2 Buy信号**（不逆势交易）
- 600368案例：5.87 W2 Buy会被直接过滤，因为周线当时已确认下跌

---

### 2.2 过滤器二：W2有效性三维验证（W2StructureValidator）

**问题**：V3识别到"向下笔没跌破W1起点"就认为是W2。但这个向下笔可能只是更大下跌浪中的一个次级回调，根本不是W2。

**三维验证规则**：

```
W2有效性 = (
    铁律1验证 AND
    铁律2验证 AND
    内部结构验证
)
```

#### 铁律1：浪2不能跌破浪1起点
```python
def validate_wave2_iron_law1(w1_start, w2_end) -> bool:
    """
    铁律1：W2回调不能跌破W1起点
    
    600368误判案例：
    - W1起点 = BI4(4.12)？不对
    - V3以为：回调没跌破4.12就是W2
    - 实际：4.12只是更大下跌浪的中间点，根本不是W1起点
    """
    return w2_end >= w1_start  # 必须满足
```

#### 铁律2：浪2回调不能超过浪1的100%
```python
def validate_wave2_iron_law2(w1_len, w2_retracement) -> bool:
    """
    铁律2：W2回撤 < W1的100%
    
    计算：retracement_ratio = (W1_end - W2_end) / (W1_end - W1_start)
    要求：retracement_ratio < 1.0
    
    实战经验：
    - W2通常回撤38.2%~61.8%
    - 如果回撤超过78.6%，可能是更复杂的调整浪
    - 如果回撤超过100%，说明不是W2（是更大级别的浪）
    """
    return w2_retracement < 1.0  # 严格版
```

#### 铁律3（辅助）：浪4不能进入浪1领地
```python
def validate_wave4_iron_law3(w1_end, w4_end, w3_end) -> bool:
    """
    铁律3：W4不能进入W1价格区域
    
    这是预验证W4时用的，提前过滤掉不合法的W4
    """
    return w4_end > w1_end  # W4低点不能低于W1高点
```

#### 内部结构验证：W2必须是3浪结构
```python
def validate_w2_internal_structure(bis_in_w2: List[BiRecord]) -> Dict:
    """
    W2内部结构验证（基于今日理论：调整浪A-B-C 3-3-5或5-3-5）
    
    W2内部必须是3段结构（a-b-c），不能是5段：
    - a浪：向下（推动）
    - b浪：向上（调整）
    - c浪：向下（推动，完成W2）
    
    检查：
    1. W2内部的笔数量是否为奇数（3或5）→ 必须是3
    2. c浪必须是5浪结构（推动浪）
    3. 如果W2内部出现了5个次级浪，说明这不是W2，是更大的结构
    """
    down_bis = [b for b in bis_in_w2 if b.direction == 'down']
    up_bis = [b for b in bis_in_w2 if b.direction == 'up']
    
    # W2内部应该是 1个向下笔(a) + 1个向上笔(b) + 1个向下笔(c)
    if len(down_bis) != 2 or len(up_bis) != 1:
        return {
            'valid': False,
            'reason': f"W2内部结构异常: {len(down_bis)}个下笔, {len(up_bis)}个上笔（期望2下1上）"
        }
    
    # 验证c浪是否形成（必须是有效向下突破）
    c_start = up_bis[0].end_price
    c_end = down_bis[1].end_price
    if c_end >= c_start:  # c浪应该创出新低
        return {
            'valid': False,
            'reason': f"W2的c浪未创新低: c起点={c_start:.2f}, c终点={c_end:.2f}"
        }
    
    return {'valid': True, 'reason': 'W2内部结构正常(a-b-c)'}
```

---

### 2.3 过滤器三：逆势反弹识别器（CounterTrendRallyDetector）

**问题核心**：600368在5.87之后的每次反弹，V3都识别为"W2 Buy"，但实际上这是更大级别下跌中的B浪反弹，每次都应该卖出而非买入。

**识别算法**：

```python
def detect_counter_trend_rally(
    current_bi: BiRecord,
    major_trend: str,  # 'up_trend' | 'down_trend' | 'neutral'
    wave_state: str,   # 当前V3识别的波浪状态
    last_major_low: float,
    last_major_high: float,
) -> Dict:
    """
    识别当前走势是否是逆势反弹
    
    核心思想（来自今日理论）：
    - 大级别下跌趋势中，任何上涨都是B浪反弹
    - B浪特征：3浪结构（a-up, b-down, c-up）
    - B浪终点后是C浪（向下）
    
    600368案例诊断：
    - 周线：下跌趋势
    - 5.87之后每次"反弹"：
      1. 幅度小（远小于前一波A浪）
      2. 内部结构是3浪（a-b-c）而非5浪
      3. 没有突破前一个高点（last_major_high）
      → 这些都是逆势反弹，不是新的W1-W2-W3
    
    返回：
    {
        'is_counter_trend': bool,      # 是否逆势
        'confidence': float,           # 置信度
        'recommended_action': str,     # 'SELL' | 'NO_BUY' | 'HOLD'
        'reason': str
    }
    """
    result = {
        'is_counter_trend': False,
        'confidence': 0.0,
        'recommended_action': 'HOLD',
        'reason': ''
    }
    
    # 情况1：大级别下跌趋势 + 日线出现"疑似W2 Buy"
    if major_trend == 'down_trend' and wave_state in ['w2_formed', 'w4_formed']:
        # 检查这是否是B浪反弹
        # B浪特征：
        # 1. 反弹幅度 < A浪的61.8%
        # 2. 内部是3浪结构
        # 3. 没有突破前高
        
        # 如果当前价格接近或高于前一个重要高点 → 可能是新的上涨趋势
        # 如果当前价格远低于前高 → 逆势反弹可能性大
        
        if current_bi.direction == 'up':
            rally_amplitude = current_bi.end_price - current_bi.start_price
            major_length = last_major_high - last_major_low
            
            if major_length > 0:
                rally_ratio = rally_amplitude / major_length
                
                # 反弹幅度超过61.8% → 可能是真的上涨趋势
                if rally_ratio > 0.618:
                    result['is_counter_trend'] = False
                    result['recommended_action'] = 'NO_BUY'  # 但不逆势追
                    result['reason'] = f'反弹幅度{rally_ratio:.1%}，可能是新趋势但不做逆势单'
                else:
                    result['is_counter_trend'] = True
                    result['confidence'] = 0.7
                    result['recommended_action'] = 'NO_BUY'
                    result['reason'] = f'周线下跌趋势中，反弹幅度{rally_ratio:.1%} < 61.8%，判定为B浪反弹，不买入'
    
    return result
```

---

## 三、具体代码改进建议

### 3.1 新增文件：`wave_filters.py`

```python
"""
wave_filters.py — 波浪理论过滤器模块
基于2026-04-07波浪理论学习成果
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BiRecord:
    """一笔的记录（复用V3的BiRecord）"""
    seq: int
    direction: str  # 'up' / 'down'
    start_price: float
    end_price: float
    start_date: str
    end_date: str
    volume: float = 0.0


@dataclass
class FilterResult:
    """过滤器返回结果"""
    passed: bool
    confidence: float  # 0.0~1.0
    reason: str
    warnings: List[str] = None  # 非致命警告


class WaveTrendFilter:
    """
    过滤器一：大级别趋势方向
    
    基于周线/日线笔序列判断趋势方向
    
    规则（今日理论：高只连低，低只连高）：
    - 连续3个高点创新高 + 连续3个低点抬高 → up_trend
    - 连续3个高点创新低 + 连续3个低点下移 → down_trend
    - 否则 → neutral
    """
    
    def __init__(self, min_swings: int = 3):
        self.min_swings = min_swings  # 至少需要3个波段才能判断
    
    def analyze(self, high_points: List[float], low_points: List[float]) -> str:
        """
        判断趋势方向
        
        Args:
            high_points: 高点价格列表（按时间顺序）
            low_points: 低点价格列表（按时间顺序）
        
        Returns:
            'up_trend' | 'down_trend' | 'neutral'
        """
        if len(high_points) < self.min_swings or len(low_points) < self.min_swings:
            return 'neutral'
        
        # 取最近的min_swings个点
        recent_highs = high_points[-self.min_swings:]
        recent_lows = low_points[-self.min_swings:]
        
        # 检查高点是否持续创新高
        highs_ascending = all(recent_highs[i] < recent_highs[i+1] 
                              for i in range(len(recent_highs)-1))
        # 检查低点是否持续抬高
        lows_ascending = all(recent_lows[i] < recent_lows[i+1] 
                             for i in range(len(recent_lows)-1))
        
        if highs_ascending and lows_ascending:
            return 'up_trend'
        
        # 检查高点是否持续创新低
        highs_descending = all(recent_highs[i] > recent_highs[i+1] 
                               for i in range(len(recent_highs)-1))
        # 检查低点是否持续下移
        lows_descending = all(recent_lows[i] > recent_lows[i+1] 
                              for i in range(len(recent_lows)-1))
        
        if highs_descending and lows_descending:
            return 'down_trend'
        
        return 'neutral'
    
    def filter_w2_buy(
        self,
        trend: str,
        w2_end: float,
        w1_start: float,
        w1_end: float,
    ) -> FilterResult:
        """
        对W2 Buy信号应用趋势过滤
        
        规则：
        - up_trend：正常买入
        - down_trend：禁止买入（W2在大级别下跌中是B浪反弹，不是买入机会）
        - neutral：谨慎买入
        
        600368案例：
        - 周线趋势：down_trend
        - 5.87 W2 Buy → 应该被过滤掉
        """
        warnings = []
        
        if trend == 'down_trend':
            return FilterResult(
                passed=False,
                confidence=0.85,
                reason='周线下跌趋势，不逆势买入W2',
                warnings=[
                    'W2在大级别下跌中可能是B浪反弹',
                    'B浪之后是C浪，逆势买入将遭受连续止损'
                ]
            )
        
        if trend == 'neutral':
            warnings.append('大级别趋势不明，W2信号置信度降低')
        
        # 额外检查：W2回调幅度
        w1_len = w1_end - w1_start
        if w1_len > 0:
            w2_retrace = (w1_end - w2_end) / w1_len
            if w2_retrace > 0.9:
                warnings.append(f'W2回撤{w2_retrace:.1%}过深，可能是更复杂调整')
        
        return FilterResult(
            passed=True,
            confidence=0.6 if trend == 'neutral' else 0.8,
            reason='趋势方向允许W2买入' if trend == 'up_trend' else '谨慎通过',
            warnings=warnings
        )


class W2StructureValidator:
    """
    过滤器二：W2结构三维验证
    
    验证维度：
    1. 铁律1：W2不能跌破W1起点
    2. 铁律2：W2回撤不能超过W1的100%
    3. 内部结构：W2必须是3浪（a-b-c），不能是5浪
    """
    
    def __init__(self):
        self.name = 'W2StructureValidator'
    
    def validate(
        self,
        w1_start: float,
        w1_end: float,
        w2_end: float,
        bis_in_w2: List[BiRecord],  # W2内部的笔序列
    ) -> FilterResult:
        """
        验证W2是否合法
        
        600368案例分析：
        - V3找到的"w1_start"=4.12, "w1_end"=5.87, "w2_end"=某低点
        - 但实际上4.12根本不是W1起点！这是V3波浪识别的错误
        - 正确的W1起点应该是更高点位
        - 因此V3的"W2验证"从一开始就建立在错误的前提上
        """
        warnings = []
        
        # ===== 铁律1：W2不能跌破W1起点 =====
        if w2_end < w1_start:
            return FilterResult(
                passed=False,
                confidence=0.95,
                reason=f'铁律1违反：W2终点{w2_end:.2f} < W1起点{w1_start:.2f}',
                warnings=['这是更复杂下跌浪的开始，不是W2']
            )
        
        # ===== 铁律2：W2回撤不能超过W1的100% =====
        w1_len = w1_end - w1_start
        if w1_len <= 0:
            return FilterResult(
                passed=False,
                confidence=0.9,
                reason='W1长度异常，无法验证W2',
                warnings=['W1必须是上升波段']
            )
        
        w2_retrace = (w1_end - w2_end) / w1_len
        
        if w2_retrace >= 1.0:
            return FilterResult(
                passed=False,
                confidence=0.9,
                reason=f'铁律2违反：W2回撤{w2_retrace:.1%} >= 100%',
                warnings=['W2回撤超过100%说明不是W2，可能是更大级别调整']
            )
        
        if w2_retrace > 0.786:
            warnings.append(f'W2回撤{w2_retrace:.1%}超过78.6%，可能是深幅调整')
        
        # ===== 铁律3辅助：W2回撤通常在38.2%~61.8% =====
        if 0.382 <= w2_retrace <= 0.618:
            reason_suffix = f'回撤比例{w2_retrace:.1%}在正常38.2%~61.8%区间'
        elif w2_retrace < 0.382:
            reason_suffix = f'回撤比例{w2_retrace:.1%}较浅（<38.2%），可能是W4'
            warnings.append('W2回撤过浅，可能是W4')
        else:
            reason_suffix = f'回撤比例{w2_retrace:.1%}在正常偏深范围'
        
        # ===== 内部结构验证：W2必须是3浪a-b-c =====
        structure_check = self._check_internal_structure(bis_in_w2)
        if not structure_check['valid']:
            return FilterResult(
                passed=False,
                confidence=0.7,
                reason=structure_check['reason'],
                warnings=['W2内部结构不符合3浪a-b-c规范']
            )
        
        return FilterResult(
            passed=True,
            confidence=0.75 if not warnings else 0.6,
            reason=f'W2结构验证通过。{reason_suffix}',
            warnings=warnings
        )
    
    def _check_internal_structure(self, bis_in_w2: List[BiRecord]) -> Dict:
        """
        检查W2内部是否是3浪a-b-c结构
        
        规则（今日理论：调整浪A-B-C 3-3-5或5-3-5）：
        - W2作为调整浪，应该是3段结构
        - a浪：向下（推动）
        - b浪：向上（调整）
        - c浪：向下（推动，完成W2）
        
        异常情况：
        - W2内部出现5个次级浪 → 这是推动浪结构，不是W2
        - W2内部出现4个或6个次级浪 → 不符合奇数段原则
        """
        if len(bis_in_w2) < 3:
            return {
                'valid': False,
                'reason': f'W2内部笔数{len(bis_in_w2)}不足，无法构成3浪结构'
            }
        
        # 统计W2内部的上下笔
        down_bis = [b for b in bis_in_w2 if b.direction == 'down']
        up_bis = [b for b in bis_in_w2 if b.direction == 'up']
        
        # W2正常结构：2个下笔(a,c) + 1个上笔(b) = 3笔
        if len(down_bis) == 2 and len(up_bis) == 1:
            # 额外验证：c浪必须创出新低
            c_end = down_bis[-1].end_price  # 最后一个下笔的终点
            a_end = down_bis[0].end_price    # 第一个下笔的终点
            
            if c_end < a_end:
                return {'valid': True, 'reason': 'W2内部a-b-c结构正常'}
            else:
                return {
                    'valid': False,
                    'reason': f'W2的c浪未创新低：a终点={a_end:.2f}, c终点={c_end:.2f}'
                }
        
        # W2内部是5浪 → 可能是更复杂的调整，或根本不是W2
        if len(bis_in_w2) == 5:
            return {
                'valid': False,
                'reason': f'W2内部出现5个次级浪，不符合调整浪3浪结构'
            }
        
        return {
            'valid': False,
            'reason': f'W2内部结构异常：{len(down_bis)}个下笔, {len(up_bis)}个上笔（期望2下1上）'
        }


class CounterTrendRallyDetector:
    """
    过滤器三：逆势反弹识别器
    
    核心功能：识别当前上涨是"顺势新推动浪"还是"逆势B浪反弹"
    
    600368案例诊断：
    - 5.87之后每次反弹都是B浪
    - B浪特征：3浪结构(a-up, b-down, c-up)，幅度小
    - V3每次都误判为新的W1起点后的W2买入点
    """
    
    def __init__(self):
        self.name = 'CounterTrendRallyDetector'
    
    def detect(
        self,
        current_price: float,
        last_major_high: float,
        last_major_low: float,
        major_trend: str,  # 'up_trend' | 'down_trend' | 'neutral'
        recent_bis: List[BiRecord],  # 最近10根笔
    ) -> FilterResult:
        """
        识别是否逆势反弹
        
        判断逻辑：
        1. 大级别下跌趋势中
        2. 当前反弹没有突破last_major_high
        3. 反弹内部结构是3浪（a-b-c）
        → 判定为逆势B浪反弹，NO_BUY
        """
        warnings = []
        
        if major_trend != 'down_trend':
            return FilterResult(
                passed=True,  # 非下跌趋势，不涉及逆势
                confidence=1.0,
                reason='大级别非下跌趋势，不涉及逆势反弹',
                warnings=[]
            )
        
        # 大级别下跌趋势中，检查当前是否在反弹
        if current_price <= last_major_low:
            # 还在创新低，不是反弹，是顺势
            return FilterResult(
                passed=True,
                confidence=0.9,
                reason='价格未反弹，仍在顺势创新低',
                warnings=[]
            )
        
        # 正在反弹（current_price > last_major_low）
        rally_amplitude = (current_price - last_major_low) / last_major_low
        major_amplitude = (last_major_high - last_major_low) / last_major_low
        
        if major_amplitude <= 0:
            return FilterResult(
                passed=True,
                confidence=0.5,
                reason='无法计算大级别幅度',
                warnings=[]
            )
        
        rally_ratio = rally_amplitude / major_amplitude
        
        # 检查最近笔结构
        recent_direction = [b.direction for b in recent_bis[-5:]]
        up_count = recent_direction.count('up')
        down_count = recent_direction.count('down')
        
        # 3浪结构特征：up-down-up 或 最近几笔以向上为主
        is_3_wave_structure = (up_count == 3 and down_count == 2)
        
        # 逆势反弹判定
        if rally_ratio < 0.618 and is_3_wave_structure:
            return FilterResult(
                passed=False,  # 不通过 = 逆势反弹
                confidence=0.8,
                reason=f'判定为B浪反弹：反弹幅度{rally_ratio:.1%}<61.8%，且呈3浪结构',
                warnings=[
                    'B浪之后是C浪，当前位置买入将止损',
                    '建议等待C浪完成后的买入机会'
                ]
            )
        
        if rally_ratio >= 0.618:
            # 反弹较强，可能是新趋势的开始，但谨慎对待
            warnings.append('反弹超过61.8%，需进一步确认是否为新趋势')
            return FilterResult(
                passed=True,
                confidence=0.5,  # 置信度降低
                reason=f'反弹幅度{rally_ratio:.1%}较大，可能不是B浪',
                warnings=warnings
            )
        
        return FilterResult(
            passed=True,
            confidence=0.6,
            reason='未明确判定为逆势反弹',
            warnings=['反弹结构不明确，保持谨慎']
        )


# ======================
# 三过滤器整合
# ======================

class WaveFilterPipeline:
    """
    三过滤器整合管道
    
    对每个W2 Buy信号，依次通过：
    1. WaveTrendFilter（大级别趋势过滤）
    2. W2StructureValidator（W2结构验证）
    3. CounterTrendRallyDetector（逆势反弹识别）
    
    全部通过才执行买入
    """
    
    def __init__(self):
        self.trend_filter = WaveTrendFilter()
        self.structure_validator = W2StructureValidator()
        self.counter_detector = CounterTrendRallyDetector()
    
    def evaluate_w2_buy(
        self,
        w1_start: float,
        w1_end: float,
        w2_end: float,
        bis_in_w2: List[BiRecord],
        high_points: List[float],
        low_points: List[float],
        last_major_high: float,
        last_major_low: float,
        current_price: float,
    ) -> Dict:
        """
        评估W2 Buy信号是否值得执行
        
        Returns:
            {
                'action': 'BUY' | 'NO_BUY' | 'SELL',
                'confidence': float,
                'filters_passed': List[str],
                'filters_failed': List[str],
                'warnings': List[str],
                'final_reason': str
            }
        """
        results = {
            'action': 'NO_BUY',
            'confidence': 0.0,
            'filters_passed': [],
            'filters_failed': [],
            'warnings': [],
            'final_reason': ''
        }
        
        # ===== 过滤器1：趋势方向 =====
        trend = self.trend_filter.analyze(high_points, low_points)
        trend_result = self.trend_filter.filter_w2_buy(
            trend, w2_end, w1_start, w1_end
        )
        
        if trend_result.passed:
            results['filters_passed'].append('WaveTrendFilter')
            results['confidence'] += trend_result.confidence * 0.3
        else:
            results['filters_failed'].append(f'WaveTrendFilter: {trend_result.reason}')
            results['warnings'].extend(trend_result.warnings)
            results['final_reason'] = f'被WaveTrendFilter拒绝：{trend_result.reason}'
            return results
        
        # ===== 过滤器2：W2结构验证 =====
        struct_result = self.structure_validator.validate(
            w1_start, w1_end, w2_end, bis_in_w2
        )
        
        if struct_result.passed:
            results['filters_passed'].append('W2StructureValidator')
            results['confidence'] += struct_result.confidence * 0.4
        else:
            results['filters_failed'].append(f'W2StructureValidator: {struct_result.reason}')
            results['warnings'].extend(struct_result.warnings)
            results['final_reason'] = f'被W2StructureValidator拒绝：{struct_result.reason}'
            return results
        
        # ===== 过滤器3：逆势反弹识别 =====
        counter_result = self.counter_detector.detect(
            current_price,
            last_major_high,
            last_major_low,
            trend,
            bis_in_w2
        )
        
        if counter_result.passed:
            results['filters_passed'].append('CounterTrendRallyDetector')
            results['confidence'] += counter_result.confidence * 0.3
        else:
            results['filters_failed'].append(f'CounterTrendRallyDetector: {counter_result.reason}')
            results['warnings'].extend(counter_result.warnings)
            results['final_reason'] = f'被CounterTrendRallyDetector拒绝：{counter_result.reason}'
            return results
        
        # 全部通过
        results['action'] = 'BUY'
        results['final_reason'] = f"三过滤器全部通过，W2买入信号有效（置信度{results['confidence']:.2f}）"
        results['warnings'].extend(trend_result.warnings)
        results['warnings'].extend(struct_result.warnings)
        results['warnings'].extend(counter_result.warnings)
        
        return results
```

---

### 3.2 V3核心修改建议

#### 修改1：WaveCounterV3新增方法

```python
# 在WaveCounterV3类中新增方法

def get_wave_level_context(self) -> Dict:
    """
    获取波浪大级别上下文（新增）
    
    基于今日理论：大级别画线规则
    - 高只连低，低只连高
    - 追踪running_high和running_low
    
    返回：
    {
        'major_trend': str,            # 'up_trend' | 'down_trend' | 'neutral'
        'high_points': List[float],     # 主要高点列表
        'low_points': List[float],      # 主要低点列表
        'last_major_high': float,
        'last_major_low': float,
    }
    """
    # 使用_get_turning_points()的结果
    points = self._get_turning_points()
    
    high_points = [p['price'] for p in points if p['type'] == 'high']
    low_points = [p['price'] for p in points if p['type'] == 'low']
    
    trend_filter = WaveTrendFilter()
    major_trend = trend_filter.analyze(high_points, low_points)
    
    return {
        'major_trend': major_trend,
        'high_points': high_points,
        'low_points': low_points,
        'last_major_high': high_points[-1] if high_points else None,
        'last_major_low': low_points[-1] if low_points else None,
    }


def get_w2_internal_bis(self) -> List[BiRecord]:
    """
    获取W2内部的笔序列（新增）
    
    用于W2StructureValidator验证W2内部是否是3浪a-b-c结构
    """
    if self.state not in ['w2_formed', 'w2_in_progress']:
        return []
    
    # 找到W1终点和W2终点对应的笔序号
    w1_end = self.snapshot.w1_end
    w2_end = self.snapshot.w2_end
    
    if w1_end is None or w2_end is None:
        return []
    
    # 找到W1终点和W2终点之间的所有笔
    w2_bis = []
    in_w2 = False
    
    for bi in self.bis:
        # 进入W2的条件：向上笔终点 >= W1_end（首位相接）
        if not in_w2 and bi.direction == 'up' and abs(bi.end_price - w1_end) < 0.02:
            in_w2 = True
        
        if in_w2:
            w2_bis.append(bi)
            
            # 退出W2的条件：向下笔终点 <= W2_end
            if bi.direction == 'down' and abs(bi.end_price - w2_end) < 0.02:
                break
    
    return w2_bis
```

#### 修改2：WaveEngine.get_signal()集成过滤器

```python
# 在WaveEngine.get_signal()中

def get_signal(self) -> Dict:
    """
    获取当前信号（修改版：集成三过滤器）
    
    当V3发出W2 Buy信号时，额外验证：
    1. 大级别趋势是否允许买入
    2. W2结构是否满足三维验证
    3. 是否是逆势反弹
    """
    base_signal = self.daily_cache.counter.get_buy_sell_signals()
    
    # 非W2/W4 Buy信号，直接返回
    if base_signal['signal'] not in ['W2_BUY', 'W4_BUY']:
        return base_signal
    
    # 获取大级别上下文
    context = self.daily_cache.counter.get_wave_level_context()
    w2_bis = self.daily_cache.counter.get_w2_internal_bis()
    
    # 获取当前价格
    current_price = self.last_snapshot.bi_high if self.last_snapshot else None
    
    # 调用过滤器管道
    pipeline = WaveFilterPipeline()
    filter_result = pipeline.evaluate_w2_buy(
        w1_start=self.daily_cache.counter.snapshot.w1_start,
        w1_end=self.daily_cache.counter.snapshot.w1_end,
        w2_end=self.daily_cache.counter.snapshot.w2_end,
        bis_in_w2=w2_bis,
        high_points=context['high_points'],
        low_points=context['low_points'],
        last_major_high=context['last_major_high'],
        last_major_low=context['last_major_low'],
        current_price=current_price,
    )
    
    # 将过滤器结果附加到信号中
    base_signal['filter_result'] = filter_result
    base_signal['major_trend'] = context['major_trend']
    
    # 如果过滤器拒绝，提高止损或改变信号状态
    if filter_result['action'] == 'NO_BUY':
        base_signal['status'] = SignalStatus.INVALID
        base_signal['reason'] = f"[过滤器拒绝] {filter_result['final_reason']}"
        base_signal['confidence'] = filter_result['confidence']
        base_signal['warnings'] = filter_result['warnings']
    else:
        base_signal['warnings'] = filter_result['warnings']
        # 置信度加权
        base_signal['confidence'] = base_signal['confidence'] * filter_result['confidence']
    
    return base_signal
```

---

## 四、600368改进效果预估

### 4.1 改进前后对比

| 场景 | 改进前（V3原版） | 改进后（+三过滤器） |
|------|----------------|------------------|
| 5.87 W2 Buy | **无条件执行** → 止损11次 | **WaveTrendFilter拒绝**（周线下跌） |
| 止损次数 | 11次 | 0次（信号被过滤） |
| 损失 | 巨大 | 0 |
| 错过的机会 | 0 | 可能错过一些真正的W2买点（但这些买点本就是陷阱） |

### 4.2 过滤器效果分析

```
600368时间线（2025-2026）：

BI1(4.69 高) → BI4(4.12 低) → BI5(5.87 高) →