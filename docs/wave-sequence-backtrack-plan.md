# 波浪序列回溯修正算法 — 实现方案 v2.0
**准备者:** Forge
**版本:** v2.0（Oracle 2026-04-11 完全重写）
**状态:** ✅ 实现完成

---

## 变更摘要（v1.0 → v2.0）

| | v1.0 | v2.0 |
|--|------|------|
| **核心假设** | L2 标注有错，需修正 | L2 是子浪，大级别决定合法性 |
| **w3/w4/w5 → w1** | 非法跳变，需修正 | 可能是合法子序列重置 |
| **回溯逻辑** | 修正当前点 | 重新理解整段上下文 |
| **大级别** | 不考虑 | 先识别周线大级别 |
| **算法** | 状态机 | 三层架构 + Viterbi |

---

## 一、核心洞察（Oracle v2.0）

### 1.1 L2 标注的是子浪，不是错误

L2 cache 中的 `wave_state` 是**日线子浪**的当前位置。子浪嵌套在大级别波浪中，大级别推进时子浪会重新从1开始：

```
大级别 Wave4（调整，下跌）
└── 子浪序列
    ├── 子浪A1 → w1_formed  ← 子浪从1开始
    ├── 子浪A2 → w2_formed
    ├── ...
    └── 子浪A5 → w5_formed
         ↓
    子浪B1 → w1_formed  ← B浪反弹，子浪又从1开始！
```

### 1.2 假"跳变"的真实含义

| 表面跳变 | 实际真相 |
|---------|---------|
| `w3_formed → w1_formed` | 大浪3-3结束 → 大浪4-A开始（调整浪，子浪从1数） |
| `w5_formed → w2_formed` | 大浪5结束 → 大浪A-2（调整A浪的2回调） |
| `w1_formed → w3_formed` | 大浪1-1 → 大浪1-3（1浪内部，子浪加速） |

### 1.3 判据：价格方向是否反转

- **价格反转**（prev_trend × curr_change < 0）→ 合法，不修正
- **价格未反转** → 真正错误，触发 Viterbi 回溯

---

## 二、三层架构

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 大级别判定层（Weekly MACD + 波浪计数）          │
│  输出：large_degree ∈ {WAVE1/2/3/4/5_UP/DOWN, A/B/C}   │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 子浪合法性约束层                               │
│  输入：L2 raw_state, LARGE_WAVE                          │
│  输出：legal_transitions（根据大级别裁剪后的合法集）       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: 回溯重标注层（Viterbi + 子序列重置检测）        │
│  输出：wave_state_corrected, LARGE_WAVE_annotated         │
└─────────────────────────────────────────────────────────┘
```

---

## 三、实现代码

### 3.1 文件结构

```
strategies/wavechan/v3_l2_cache/
├── wave_backtrack_corrector.py   ← 新增：WaveBacktrackCorrector v2.0
└── wavechan_strategy.py          ← 修改：集成回溯修正
```

### 3.2 WaveBacktrackCorrector 类

**核心方法：**

| 方法 | 职责 |
|------|------|
| `correct(df)` | 公开 API：对整个 DataFrame 应用三层修正 |
| `_identify_weekly_wave_context()` | Layer 3：周线 MACD + 波浪计数 |
| `_is_legal_sub_transition()` | Layer 2：判定子浪转移是否合法 |
| `_identify_sub_sequence_resets()` | Layer 1：识别 w3/w4/w5 → w1 跳变 |
| `_apply_corrections()` | Layer 1：对 NEED_REVIEW 执行 Viterbi 回溯 |
| `_viterbi_backtrack()` | Viterbi 重打分算法 |

### 3.3 大级别上下文约束（Layer 2）

```python
LARGE_DEGREE_CONTEXT = {
    'WAVE3_UP': {
        'sub_must_follow': {
            'w1_formed': {'w2_formed'},
            'w2_formed': {'w3_formed'},
            'w3_formed': {'w3_formed', 'w4_formed'},  # 3浪延长
            'w4_formed': {'w4_formed', 'w5_formed'},
            'w5_formed': {'w1_formed'},   # 子5结束，大浪3继续
        },
        'sub_sequence_count': 1,
    },
    'WAVE4_DOWN': {
        'sub_direction': 'DOWN',
        'sub_must_follow': { ... },
        'sub_sequence_count': 2,   # A浪 + B浪（子序列重置最可能发生处）
    },
    # ... 其他大级别
}
```

### 3.4 子序列重置识别（Layer 1）

```python
def identify_sub_sequence_resets(states, prices, large_degree):
    """
    识别 w3/w4/w5 → w1 的跳变
    - 价格反转 → LEGAL_RESTART（合法，不修正）
    - 价格未反转 → NEED_REVIEW（触发 Viterbi 回溯）
    """
    for i in range(1, len(states)):
        curr = states[i]
        prev = states[i-1]
        if curr == 'w1_formed' and prev in ('w3_formed', 'w4_formed', 'w5_formed'):
            reversal = check_price_direction_reversal(prices, i)
            if reversal:
                return ('LEGAL_RESTART', prev)   # 不修正
            else:
                return ('NEED_REVIEW', prev)    # Viterbi 回溯
```

---

## 四、集成到 WaveChanStrategy

### 4.1 修改位置

```python
# wavechan_strategy.py

def prepare(self, dates, loader):
    # ... 现有代码 ...

    # ── v2.0 回溯修正（新增）──────────────────────────────
    self._apply_backtrack_correction(dates)
    # 结果：
    #   self._large_degree_cache[symbol] = large_degree
    #   self._corrected_wave_state_cache[symbol][date] = corrected_state

    # ── 向量化预计算周线方向（使用修正后状态）────────────
    self._precompute_weekly_dirs_vectorized(dates)
```

### 4.2 修改后的 impulse_state 映射（v2.0）

```python
def _wave_state_to_impulse_state(self, wave_state, large_degree=None):
    ws = str(wave_state)
    large = large_degree or 'UNKNOWN'

    base_mapping = {
        'w1_formed':       'W2_correction',
        'w2_formed':       'W3_in_progress',
        'w3_formed':       'W4_correction',
        'w4_formed':       'W5_in_progress',
        'w4_in_progress': 'W4_correction',
        'w5_formed':       'W5_done',
    }
    base = base_mapping.get(ws, 'W1_or_W2')

    # v2.0 大级别上下文调整
    if large in ('WAVE4_DOWN', 'WAVEA_DOWN', 'WAVEC_DOWN', 'WAVEB_UP'):
        # 大级别调整/反弹中：子浪推动 ≠ 大级别推动
        if ws == 'w2_formed':
            return 'W1_or_W2'  # 调整浪中的反弹，不买
        if ws == 'w4_formed':
            return 'W4_correction'  # 调整浪中的子浪4，不买

    return base
```

### 4.3 BUY 过滤效果

| wave_state | large_degree | impulse_state | BUY |
|------------|-------------|---------------|-----|
| w2_formed | WAVE3_UP | W3_in_progress | ✅ |
| w4_formed | WAVE3_UP | W5_in_progress | ✅ |
| w2_formed | WAVE4_DOWN | W1_or_W2 | ❌ |
| w4_formed | WAVE4_DOWN | W4_correction | ❌ |

---

## 五、性能

| 操作 | 复杂度 | 全量（6000只×250天）|
|------|--------|---------------------|
| 周线聚合 | O(N) | ~1s |
| MACD 计算 | O(N) | ~1s |
| 子序列检测 | O(N) | ~0.5s |
| Viterbi 回溯 | O(N×S²) per segment | ~2-3s（稀疏触发）|
| **总计** | | **< 5 秒** |

**内存峰值：**
- DataFrame：~300MB（6000×250×22列）
- 缓存 dicts：~30MB
- **合计 < 350MB**

---

## 六、测试验证

### 6.1 单元测试

```bash
# 1. 基础导入
python3 -c "from strategies.wavechan.v3_l2_cache.wave_backtrack_corrector import WaveBacktrackCorrector; print('OK')"

# 2. 子序列重置识别
# - w3_formed → w1_formed + 价格反转 → LEGAL_RESTART
# - w5_formed → w1_formed + 无反转 → NEED_REVIEW

# 3. Viterbi 输出验证
# - 输出与 segment_states 等长
# - 最后一个状态是概率最高的
```

### 6.2 回归测试

```bash
# 对比修正前后 filter_buy 结果
# 确保修正不破坏原有的 W3_in_progress / W5_in_progress 过滤
```

---

## 七、已实现的代码清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `wave_backtrack_corrector.py` | ✅ 完成 | WaveBacktrackCorrector 类（~600行） |
| `wavechan_strategy.py::prepare()` | ✅ 完成 | 调用 `_apply_backtrack_correction()` |
| `wavechan_strategy.py::_apply_backtrack_correction()` | ✅ 完成 | 回溯修正主入口 |
| `wavechan_strategy.py::_wave_state_to_impulse_state()` | ✅ 完成 | 支持 large_degree 参数 |
| `wavechan_strategy.py::get_weekly_impulse_state()` | ✅ 完成 | 使用修正后状态和缓存 |

---

## 八、关键算法细节

### 8.1 大级别识别（简化版，无需周线数据）

```python
def _identify_weekly_wave_context(self, sym_df):
    # 1. 取最近 60 天日线
    # 2. 计算 MACD histogram
    # 3. 找局部高低点
    # 4. 结合 hist 方向分类：
    #    - hist > 0 + n_highs>=3 → WAVE3_UP
    #    - hist > 0 + n_highs>=5 → WAVE5_UP
    #    - hist < 0 + n_lows>=4 → WAVEC_DOWN
    #    - hist < 0 + n_lows>=3 → WAVE4_DOWN
    #    - 否则 → UNKNOWN
```

### 8.2 Viterbi 转移矩阵（基于 LARGE_DEGREE_CONTEXT）

```python
# 大级别为 WAVE3_UP 时：
# w1 → w2 (0.8), w1 → w1 (0.2)  # 允许子序列重置
# w2 → w3 (0.8), w2 → w1 (0.2)
# w3 → w3 (0.4), w3 → w4 (0.4), w3 → w1 (0.2)  # 3浪延长+子序列重置
# w4 → w4 (0.4), w4 → w5 (0.4), w4 → w1 (0.2)
# w5 → w1 (0.8), w5 → w1 (0.2)
```

### 8.3 Viterbi 发射矩阵（基于价格方向）

```python
# 上涨日（close[i] > close[i-1]）：
#   w1/w3/w5 高概率（向上浪）
#   w2/w4 低概率（向下回调）
# 下跌日（close[i] < close[i-1]）：
#   w2/w4 高概率
#   w1/w3/w5 低概率
```
