# WaveChan V3 架构文档

> 全量缠论波浪特征数据库
> 创建时间：2026-03-27
> 核心特性：**持续修正机制**

---

## 🎯 设计目标

1. **全量覆盖** — A股全市场（约6000只）历史数据
2. **分批计算** — 不一次性加载全部数据，内存可控
3. **持续修正** — 每一根新K线进来，浪型可以持续修正
4. **增量更新** — 每日只需计算新数据，复用历史缓存

---

## 📁 数据目录

```
/data/warehouse/wavechan_v3/
├── symbol_cache/              # 每只股票一个 .pkl
│   ├── 600985.pkl
│   ├── 000001.pkl
│   └── ...
├── features_full.parquet       # 全量特征（按日期，全市场）
└── meta.json                   # 计算进度元信息
```

---

## 🔑 核心理念：持续修正

### 什么是持续修正？

在波浪理论中，我们**不是预判**下一浪怎么走，而是：
> "等走出来之后确认" — 每一根新笔（bi）形成后，可能需要对前面的浪型进行修正

### 修正场景举例

**场景A：W2不破W1起点（正常确认）**
```
笔1↓ → 笔2↑ → 笔3↓（没跌破笔1低点）→ W2确认！
之前的向上笔（笔2）合并为W1
笔4↑突破W1高点 → W3确认！
```

**场景B：W2跌破W1起点（趋势破坏，需要修正）**
```
笔1↓ → 笔2↑ → 笔3↓ → 笔4↑ → 笔5↓ → 笔6↓（跌破笔1起点！）
→ W1起点下移，W2重新扫描
```

**场景C：W3内部细分**
```
W3确认后，可能内部由次级别日线笔细分
回调不破W3起点 = W4确认
再向上 = W5
```

---

## 🌊 波浪状态机

```
                    ┌─────────────┐
                    │  initial   │  等待形成W1
                    └──────┬──────┘
                           │ 笔1-5完成，笔6向下没跌破W1起点
                           ▼
                    ┌─────────────┐
                    │ w1_formed   │  W1已确认，等待W2确认
                    └──────┬──────┘
                           │ 笔6没跌破W1起点 → W2确认！
                           ▼
                    ┌─────────────┐
                    │ w2_formed   │  W2已确认，等待W3
                    └──────┬──────┘
                           │ 笔7向上突破W1终点 → W3确认！
                           ▼
                    ┌─────────────┐
                    │ w3_formed   │  W3已确认，W4调整中
                    └──────┬──────┘
                           │ 笔8向下跌破W3起点 → W4确认！
                           ▼
                    ┌─────────────┐
                    │ w4_formed   │  W4调整中，等待W5
                    └──────┬──────┘
                           │ 笔10向上突破W3高点 → W5确认！
                           ▼
                    ┌─────────────┐
                    │ w5_formed   │  5浪完成，新的下跌开始
                    └──────┬──────┘
                           │ 新的向下笔...
                           ▼
                    ┌─────────────┐
                    │ 循环回 initial │  重新扫描新趋势
                    └─────────────┘
```

---

## 🔧 核心类设计

### 1. WaveCounterV3（波浪计数器）

```python
class WaveCounterV3:
    """
    波浪计数器 - 持续修正版
    
    每次喂入一个新笔(bi)，内部重新计算所有浪型高低点。
    """

    def __init__(self):
        self.bis = []           # 所有笔序列
        self.state = "initial"  # 当前状态
        self.waves = {
            'W1': {'start': None, 'end': None},
            'W2': {'end': None},
            'W3': {'end': None},
            'W4': {'end': None},
            'W5': {'end': None},
        }
        # 斐波那契回调位（动态计算）
        self.fib = {'382': None, '500': None, '618': None}
    
    def feed_bi(self, bi: RawBar) -> dict:
        """
        喂入一个新笔，自动重新计算所有波浪状态
        返回当前状态快照
        """
        self.bis.append(bi)
        self._recalc()
        return self.get_snapshot()
    
    def _recalc(self):
        """核心：从当前笔序列重新计算波浪（持续修正）"""
        # 1. 扫描所有笔，找所有高低点
        # 2. 从最新往回回溯，按规则确认W1/W2/W3/W4/W5
        # 3. 计算斐波那契位
        # 4. 检查是否有矛盾需要修正
```

### 2. SymbolWaveCache（单股票缓存）

```python
class SymbolWaveCache:
    """
    单只股票的波浪数据缓存
    持久化到 .pkl 文件
    """

    def __init__(self, symbol: str, cache_path: str):
        self.symbol = symbol
        self.cache_path = cache_path
        self.counter = WaveCounterV3()  # 波浪计数器
        self.last_date = None           # 最后处理日期
        self.completed_bis = []         # 已完成的笔序列
        self.current_pending_bars = []  # 未形成笔的K线
        self.feature_cache = []         # 特征历史

    def feed_bar(self, bar: RawBar) -> dict:
        """
        喂入一根K线：
        1. 先尝试形成新笔
        2. 如果新笔形成，喂入counter重新计算波浪
        3. 返回当前特征
        """
        # 内部逻辑：
        # - 累计High/Low形成新的笔
        # - 新笔形成时调用 counter.feed_bi(new_bi)
        # - 返回当前波浪状态快照

    def save(self):
        """保存到 pkl"""
        
    def load(self) -> bool:
        """从 pkl 加载"""
```

### 3. BatchWaveBuilder（批量构建器）

```python
class BatchWaveBuilder:
    """
    分批构建全量波浪特征
    不一次性加载所有数据，分批处理
    """

    def __init__(self, cache_dir: str, data_source: callable):
        self.cache_dir = cache_dir
        self.data_source = data_source  # 函数：fetch_data(symbols, date_range)
        self.batch_size = 50            # 每批50只股票
        self.symbols = get_all_symbols() # 从数据库获取全市场股票列表

    def run_full(self, date_range: tuple):
        """
        全量构建（一次性）
        按日期从旧到新，每批50只股票逐批计算
        """
        for batch in self.chunks(self.symbols, self.batch_size):
            # 1. 获取这批股票的历史数据
            # 2. 对每只股票逐日喂入K线
            # 3. 保存到 symbol_cache/*.pkl
            # 4. 记录进度
            pass

    def run_incremental(self, date: str):
        """
        增量更新（每日）
        只重算当天有数据的股票
        """
        changed_symbols = get_changed_symbols(date)
        for symbol in changed_symbols:
            cache = SymbolWaveCache(symbol).load()
            new_bar = get_bar(symbol, date)
            cache.feed_bar(new_bar).save()
```

---

## 📊 输出特征

每只股票每日输出：

| 字段 | 说明 |
|------|------|
| `date` | 日期 |
| `symbol` | 股票代码 |
| `wave_state` | 当前状态（initial/w1_formed/w2_formed/w3_formed/w4_formed/w5_formed） |
| `wave_direction` | up/down/neutral |
| `w1/w2/w3/w4/w5_start` | 各浪起点价格 |
| `w1/w2/w3/w4/w5_end` | 各浪终点价格 |
| `fib_382/500/618` | W4回调位 |
| `fib_target` | W5目标位 = W3 × 1.618 |
| `czsc_signal` | czsc库识别的一买/二买/三买/一卖/二卖 |
| `bi_count` | 当前笔序号 |
| `bi_direction` | 当前笔方向 up/down |
| `bi_high/low` | 当前笔高低点 |
| `last_updated` | 最后更新日期 |

---

## ⚙️ 计算流程（逐日）

```
新的一天 K线数据到达
    │
    ▼
加载该股票的 symbol_cache/*.pkl（如果没有则新建）
    │
    ▼
逐根K线喂入 feed_bar()
    │
    ├── 新笔形成？ → counter.feed_bi(new_bi) → 重新计算所有波浪
    │
    ▼
生成当日特征快照 → 追加到 feature_cache
    │
    ▼
保存 symbol_cache/*.pkl（增量更新，约秒级）
```

---

## 🚀 性能目标

| 场景 | 耗时 |
|------|------|
| 全量构建（6000只，约5年数据） | ~2-3小时（分批后台运行） |
| 单只股票全量重算 | ~5-10秒 |
| 每日增量更新（只算新K线） | < 30秒/全部股 |
| 内存占用 | 每批50只股票约 200MB |

---

## 🔄 与旧版 wavechan 的区别

| | 旧版 wavechan | V3 |
|---|---|---|
| 波浪识别 | Zigzag阈值 | CZSC笔序列 |
| 笔序列 | 无 | 完整持久化 |
| 持续修正 | 无 | ✅ 支持 |
| 缠论信号 | czsc买卖点 | czsc买卖点（不变） |
| 回测速度 | 快 | 中等（首次建缓存慢） |
| 增量更新 | 支持 | 支持 |

---

## 📝 待完成

- [ ] `WaveCounterV3` 核心计数器（含持续修正逻辑）
- [ ] `SymbolWaveCache` 单股票缓存类
- [ ] `BatchWaveBuilder` 批量构建器
- [ ] `wavechan_v3_signal.py` 每日信号脚本
- [ ] 对接 `backtester.py` 回测框架
- [ ] 对接 `auto_trade_executor.py` 实盘流程
