# 量化交易框架设计方案
> 状态：**✅ 已评审通过** | 日期：2026-03-31

---

## 一、目标

将回测/模拟盘系统重构为 **Strategy Pattern** 架构：
- **框架（Framework）**：公共部分，只写一次
- **策略（Strategy）**：各策略实现信号接口，插拔式切换

实现所有策略共用同一套回测引擎、数据加载、仓位管理、通知推送。

---

## 二、架构总览

```
┌─────────────────────────────────────────────────────────┐
│                      统一入口脚本                           │
│    backtest.py（回测）    /    simulate.py（模拟盘）        │
└────────────────────────┬────────────────────────────────┘
                         │  load_data()
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  DataProvider（数据层）                    │
│    fetch_daily(start, end) → DataFrame                  │
│    fetch_indicators(start, end) → DataFrame             │
│    fetch_money_flow(start, end) → DataFrame            │
│                                                          │
│    实现：ParquetDataProvider（历史）                     │
│         LiveDataProvider（实盘 - 待实现）                  │
└────────────────────────┬────────────────────────────────┘
                         │  merge → 完整日线数据
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 BaseFramework（框架层）                    │
│                                                          │
│  【回测引擎】                                            │
│    - run_backtest(start, end, strategy)                 │
│    - 逐日循环：先处理出场 → 再处理入场                     │
│    - 涨跌停过滤 / 滑点 / 手续费                           │
│                                                          │
│  【仓位管理】                                            │
│    - 资金分配（等权 or 按评分）                           │
│    - 最大持仓数限制                                       │
│    - 仓位调整（止损/止盈）                               │
│                                                          │
│  【状态持久化】                                           │
│    - save_state() / load_state()                         │
│    - 中断后可恢复继续运行                                 │
│                                                          │
│  【通知推送】                                             │
│    - QQ / 飞书推送交易信号                               │
└────────────────────────┬────────────────────────────────┘
                         │  调用策略3个接口
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Strategy Interface（策略接口）                │
│                                                          │
│   class Strategy:                                        │
│       name: str                                         │
│                                                          │
│       def prepare(self, dates: List[str])               │
│           '''可选：预加载数据、初始化模型'''               │
│                                                          │
│       def filter_buy(self, daily_df: DataFrame)        │
│           → DataFrame【候选股票】                         │
│                                                          │
│       def score(self, candidates: DataFrame)            │
│           → DataFrame【带 score 列，按序】               │
│                                                          │
│       def should_sell(self, row, pos, market)         │
│           → (bool 是否出场, str 原因)                    │
│                                                          │
│       def on_tick(self, row, pos)                      │
│           '''可选：每日回调，更新状态如连续天数'''         │
└────────────────────────┬────────────────────────────────┘
                         ▲
         ┌───────────────┼───────────────┐
         │               │               │
┌────────┴───┐  ┌───────┴────┐  ┌──────┴──────┐
│ V8ScoreStrategy │ WaveChanStrategy │ FutureStrategy │
│   (纯量化)     │  (波浪缠论)    │               │
└───────────────┘  └─────────────┘  └──────────────┘
```

---

## 三、框架层详解

### 3.1 数据加载（DataProvider）

```python
class DataProvider:
    """数据访问统一接口"""

    def fetch_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日线行情：date, symbol, open, high, low, close, volume, amount"""

    def fetch_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """技术指标：sma_5/10/20/55, rsi_14, macd_hist, cci_20, williams_r, ... """

    def fetch_money_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """资金流：money_flow_trend, money_flow_positive, ...（从 data_loader.py 计算）"""

    def fetch_signals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """预计算信号（如L2缓存），仅V3波浪策略需要"""
```

**ParquetDataProvider 实现：**
- 日线/指标：从 `/data/warehouse/` Parquet 仓库读取
- 资金流：调用 `data_loader.py` 的 `calculate_money_flow_indicators()`
- 信号：从 `/data/warehouse/wavechan/wavechan_cache/` L2缓存读取

### 3.2 回测引擎（run_backtest）

```
逐日循环（每个交易日）：
  ┌─────────────────────────────────────┐
  1. 更新持仓信息（最新价、连续状态）        │
  2. 调用 should_sell() 遍历持仓 → 处理出场│
  3. 若有空位 → filter_buy() → score()    │
  4. 分配新仓位（等权 or 按评分）           │
  5. 记录交易、更新资金                     │
  6. 发送通知（如有交易）                   │
  └─────────────────────────────────────┘
```

**关键设计点：**
- **先出场再入场**：避免资金冲突
- **next_open 模拟真实成交价**：避免未来函数
- **涨跌停不交易**：过滤 T+0 干扰

### 3.3 仓位管理

```python
class PositionManager:
    max_positions: int = 5          # 最大持仓数
    position_size: float = 0.20    # 每只仓位比例
    stop_loss_pct: float = 0.08   # 止损（也可由策略提供）
    take_profit_pct: float = 0.20 # 止盈

    def allocate(self, cash: float, scored_candidates: DataFrame) -> List[Order]:
        """按评分排序分配仓位，最高分优先"""

    def update(self, pos: dict, row: Series):
        """更新持仓状态（持仓天数、连续空头天数等）"""
```

### 3.4 状态持久化

```python
# 状态文件：JSON格式
{
  "last_date": "2026-03-30",
  "cash": 502000,
  "positions": {
    "600538": {
      "qty": 1000,
      "avg_cost": 12.50,
      "entry_date": "2026-03-10",
      "consecutive_bad_days": 0,
      "days_held": 15
    }
  },
  "trades": [...],
  "total_trades": 23,
  "winning_trades": 14
}
```

---

## 四、策略接口详解

### 4.1 filter_buy(daily_df: DataFrame) → DataFrame

**输入**：当日全市场数据（含所有指标）

**返回**：满足入场条件的候选股票 DataFrame，必须包含 `symbol` 列

**示例（V8）：**
```python
def filter_buy(self, daily_df):
    return daily_df[
        (daily_df['has_signal'] == True) &
        (daily_df['total_score'] >= 50) &
        (daily_df['wave_trend'].isin(['up', 'neutral', '']))
    ]
```

**示例（V3波浪）：**
```python
def filter_buy(self, daily_df):
    return daily_df[
        (daily_df['has_signal'] == True) &
        (daily_df['total_score'] >= 50) &
        (daily_df['wave_trend'].isin(['long', 'neutral', ''])) &
        (daily_df['signal_type'].isin(['W2_BUY', 'W4_BUY', 'C_BUY']))
    ]
```

### 4.2 score(candidates: DataFrame) → DataFrame

**输入**：`filter_buy` 的结果

**返回**：按 `score` 降序排列的 DataFrame（必须包含 `symbol` 和 `score` 列）

**约束**：框架取 `score` 最高的 N 只分配仓位

### 4.3 should_sell(row, pos, market) → (bool, str)

**输入**：
- `row`：当日行情（Series）
- `pos`：当前持仓信息（dict）
- `market`：市场状态（dict，含 `date`, `cash`, `total_value`）

**返回**：
- `bool`：是否出场
- `str`：出场原因（用于日志和统计）

**出场原因规范**：
| 原因 | 说明 |
|------|------|
| `STOP_LOSS` | 止损 |
| `TAKE_PROFIT` | 止盈 |
| `TIME_EXIT` | 时间止损 |
| `WAVE_SIGNAL` | 波浪出场信号 |
| `TREND_BREAK` | 趋势破坏 |

### 4.4 on_tick(row, pos)

**用途**：每个交易日调用一次，用于更新持仓的临时状态（如"连续多少天不符合条件"）

**示例**：
```python
def on_tick(self, row, pos):
    trend = row.get('wave_trend', '')
    if trend == 'down':
        pos['consecutive_bad_days'] = pos.get('consecutive_bad_days', 0) + 1
    else:
        pos['consecutive_bad_days'] = 0
    pos['days_held'] = pos.get('days_held', 0) + 1
```

---

## 五、策略实现清单

| 策略 | 文件 | 数据源 | 状态 |
|------|------|--------|------|
| V8ScoreStrategy | `strategies/score/v8/strategy.py` | Parquet + data_loader | ✅ 已有 |
| WaveChanStrategy | `strategies/wavechan/v3_l2_cache/wavechan_strategy.py` | L2缓存 + Parquet | ✅ 已有 |
| Future... | - | - | ❓ 待扩展 |

---

## 六、统一入口脚本

### 6.1 backtest.py

```bash
# 用法
python3 backtest.py --strategy v8 --start 2018-01-01 --end 2025-12-31
python3 backtest.py --strategy wavechan --start 2018-01-01 --end 2025-12-31
python3 backtest.py --strategy all --start 2018-01-01 --end 2025-12-31  # 所有策略横向对比
```

### 6.2 simulate.py

```bash
# 每日运行（自动读取上次状态继续）
python3 simulate.py --strategy v8

# 重置模拟盘
python3 simulate.py --strategy v8 --reset

# 查看状态
python3 simulate.py --strategy v8 --show-state
```

---

## 七、实施计划

### Phase 1：框架基础设施
- [✅ 完成] 重构 `DataProvider` 抽象层
- [✅ 完成] 实现 `BaseFramework.run_backtest()`
- [✅ 完成] 实现 `PositionManager` 仓位管理
- [✅ 完成] 实现状态持久化（save/load）

### Phase 2：策略迁移
- [✅ 完成] 迁移 V8ScoreStrategy 到新框架
- [✅ 完成] 迁移 WaveChanStrategy 到新框架
- [✅ 完成] 验证两组回测结果与旧版本一致

### Phase 3：统一入口
- [✅ 完成] 实现 `backtest.py` 统一入口
- [✅ 完成] 实现 `simulate.py` 统一入口
- [✅ 完成] 横向对比功能（`--strategy all`）

### Phase 4：清理
- [✅ 完成] 删除旧的独立回测文件（保留备份到 archive/）
- [✅ 完成] 更新文档
- [✅ 完成] 同步 learnings

---

## 八、待评审问题（已解决）

1. **V8 和 V3 的 `should_sell` 出场条件不同**：✅ 框架不统一，策略自行决定
2. **持仓天数的跟踪**：✅ 策略自行维护（框架通过 hasattr 调用）
3. **资金流计算**：✅ 数据层统一算好（`load_strategy_data` 包含 money_flow）
4. **实盘数据源**：⏳ 待实现（LiveDataProvider）

---

## 九、附录：关键文件

| 文件 | 职责 |
|------|------|
| `simulator/base_framework.py` | 框架核心 |
| `simulator/shared.py` | 公共模块（策略注册、数据加载） |
| `strategies/score/v8/strategy.py` | V8策略实现 |
| `strategies/wavechan/v3_l2_cache/wavechan_strategy.py` | V3波浪策略 |
| `utils/data_loader.py` | 数据查询模块 |
| `backtest.py` | 统一回测入口 |
| `simulate.py` | 统一模拟盘入口 |
| `archive/` | 旧文件归档（13个） |

---

## 十、改造实施顺序

### Step 1 ✅ 现状（已完成）
- [x] 方案设计文档
- [x] 快速重建补数据（2024跑完就全部完成）

### Step 2（下一步建议）
**先改造 V8 + 统一数据层**，理由：
1. V8 是当前主战策略，改造后可以直接验证效果
2. 数据层统一后，V3 改造更简单

### Step 3（后续）
- 迁移 V3 波浪策略
- 实现统一入口（backtest.py / simulate.py）
- 横向对比功能

---

## 十一、第一步具体计划

### 11.1 新建框架文件

| 文件 | 职责 | 优先级 |
|------|------|--------|
| `simulator/data_provider.py` | 数据访问统一接口 | ⭐⭐⭐ |
| `simulator/base_framework.py` | 回测引擎核心 | ⭐⭐⭐ |
| `backtest.py` | 统一回测入口 | ⭐⭐ |
| `simulate.py` | 统一模拟盘入口 | ⭐⭐ |

### 11.2 改造已有文件

| 文件 | 改动 | 优先级 |
|------|------|--------|
| `strategies/score/v8/strategy.py` | 实现 Strategy 接口（3个方法） | ⭐⭐⭐ |
| `strategies/wavechan/v3_l2_cache/wavechan_strategy.py` | 实现 Strategy 接口 | ⭐⭐ |

### 11.3 待删旧文件（备份后删）

| 文件 | 说明 |
|------|------|
| `v8_simulated_trading.py` | 并入 simulate.py |
| `v3_simulated_trading.py` | 并入 simulate.py |
| `backtest_score_v8.py` | 并入 backtest.py |
| `wavechan_v3_backtest.py` | 并入 backtest.py |

---

## 十二、数据层接口

`data_loader.py` 是统一数据查询模块，各策略按需调用。

```python
# data_loader.py — 统一数据查询（已有，改造）

class DataLoader:
    """数据访问统一入口"""

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)

    # ── 公共数据（所有策略共用）─────────────────────────
    def load_daily(self, years: List[int]) -> pd.DataFrame:
        """日线行情：open/high/low/close/volume/amount"""

    def load_indicators(self, years: List[int]) -> pd.DataFrame:
        """技术指标：sma/macd/rsi/cci/wr/kdj/boll"""

    def load_money_flow(self, years: List[int]) -> pd.DataFrame:
        """资金流向：XVL/LIJIN/LLJX/money_flow_trend 等（所有年份）"""

    def load_financial(self) -> pd.DataFrame:
        """财务基本面：PE/PB/ROE/营收增速等（forward-fill 年度数据）"""

    def load_basic_info(self) -> pd.DataFrame:
        """股票信息：total_shares/industry/listing_date"""

    # ── 策略私有数据（按需调用）─────────────────────────
    def load_wavechan_signals(self, start: str, end: str) -> pd.DataFrame:
        """波浪缠论信号：仅 V3 需要，从 L2 缓存读取"""
        # 按 date+symbol 关联返回

    # ── 派生数据（自动计算）────────────────────────────
    def derive_limit_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """涨跌停状态：limit_up / limit_down"""

    def calculate_market_breadth(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场宽度：全市场 MA55>SMA240 比例"""

    # ── 统一加载 ──────────────────────────────────────
    def load_strategy_data(self, years: List[int], add_money_flow: bool = True) -> pd.DataFrame:
        """
        策略数据加载主入口（V8 在用）
        合并：日线 + 技术指标 + 财务 + 股票信息 + 资金流 + 涨跌停 + 市场宽度
        """
        ...
```

---

## 十三、代码实现记录（2026-03-31）

### Step 1 ✅ `utils/data_loader.py` — 数据查询模块
- [x] `load_wavechan_signals(start, end)` — 波浪L2缓存读取
- [x] `calculate_market_breadth(df)` — 市场宽度计算

### Step 2 ✅ `simulator/base_framework.py` — 回测引擎核心
- [x] `BaseFramework` 类（~596行）
  - ✅ `Strategy` 接口定义（`filter_buy` / `score` / `should_sell` / `on_tick`）
  - ✅ `Strategy` 新增 `REQUIRED_COLUMNS` + `prepare()` 接口
  - ✅ `Position` / `Trade` 数据类
  - ✅ `run_backtest()` 回测方法
  - ✅ `run_simulate()` 模拟盘方法（含 dates 参数传入）
  - ✅ `_on_day()` 核心循环（先出场再入场）
  - ✅ `_process_sells()` / `_process_buys()` 交易处理
  - ✅ exec_price 逻辑：回测用 next_open，模拟盘 fallback 到 open/close
  - ✅ 涨跌停过滤、佣金、滑点
  - ✅ 状态持久化（`save_state()` / `load_state()` / `reset()`）
- [x] 验证：V8 / WaveChan 回测通过

### Step 3 ✅ 统一入口
- [x] `backtest.py` — 统一回测入口（`--strategy v8|wavechan|all`）
- [x] `simulate.py` — 统一模拟盘入口（`--show-state --reset --date`）
- [x] `simulator/shared.py` — 公共模块（策略注册表、load_strategy、波浪缓存加载）

### Step 4 ✅ 策略改造
- [x] V8ScoreStrategy — 实现 `REQUIRED_COLUMNS` + `prepare()` 接口
- [x] WaveChanStrategy — 实现 `REQUIRED_COLUMNS` + `prepare()` 接口
- [x] WaveChan `filter_buy` 加 `signal_status == 'confirmed'` 过滤（修复 look-ahead bias）

### Step 5 ✅ 归档
- [x] 13个旧文件归档到 `archive/`
- [x] `systems.yaml` 更新，移除归档文件引用

### Step 6 ✅ 架构评审（2026-03-31）
- [x] 多 Agent 评审完成，发现2个高优先级问题
- [x] 问题1（on_tick顺序）：确认不是 bug（隔日执行模型正确设计）
- [x] 问题2（DRY违规）：已提取 `simulator/shared.py`
- [x] 问题3/5：确认为设计决策，无问题

