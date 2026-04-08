## Oracle 代码评审

### 评审文件
- market_regime.py：⚠️ 整体合格，2处可优化（死代码、参数语义）
- base_framework.py：✅ 框架集成正确，无明显问题
- backtest.py：✅ 年度循环 + MarketRegimeFilter 集成正确
- simulate.py：⚠️ 存在重复初始化，但不影响功能

---

### 关键问题（P0）

**无 P0 级问题。**

---

### 建议修改（P1）

#### 1. market_regime.py — RSI 计算存在死代码（P1）

**位置**：`prepare()` 方法，约第 80-84 行

```python
# RSI14（取前30个交易日数据计算）
delta = close.diff()
gain = delta.clip(lower=0).rolling(30, min_periods=30).mean()   # ← 死代码
loss = (-delta.clip(upper=0)).clip(lower=0).rolling(30, min_periods=30).mean()  # ← 死代码
# 用 .iloc[-1] / .mean() 技巧实现 Wilders RSI
avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
```

**问题**：`gain` 和 `loss` 变量定义后从未被使用（实际使用的是后面的 `avg_gain` / `avg_loss` EWM 版本）。这是开发遗留的死代码，可能造成混淆。

**建议**：删除这两行无用代码。

**附注**：实际 RSI 计算（EWM 版）逻辑正确，Wilders 平滑（alpha=1/14）实现无误。

---

#### 2. market_regime.py — warmup_days 最小值过高（P1）

**位置**：`prepare()` 方法，约第 70 行

```python
warmup_days = max(self.confirm_days + 30, 60)
```

**问题**：
- 当 `confirm_days=2` 时，`warmup_days = max(32, 60) = 60`
- RSI14 实际用 EWM 计算，`min_periods=14`，只需 14 天即可输出有效值
- 固定取 60 天 warmup 可能导致回测数据窗口比必要的大 28 个交易日（每年 ~28 个交易日浪费）

**影响**：轻微。仅影响 prepare 加载的数据范围，不影响计算正确性。

**建议**：可考虑改为 `warmup_days = self.confirm_days + 30`，去掉 `max(..., 60)` 的兜底。

---

#### 3. simulate.py — MarketRegimeFilter 重复初始化（P1）

**位置**：`main()` 函数，约第 163-165 行

```python
# 初始化 MarketRegimeFilter（传入 framework）
mrf = MarketRegimeFilter()
mrf.prepare(target_date, target_date)
framework.market_regime_filter = mrf
```

**问题**：如果 `args.show_regime=True`，在第 149-150 行已经创建并 prepare 过一个 `MarketRegimeFilter` 实例用于展示大盘状态。随后进入正常模拟分支时，又重新创建了第二个实例。虽然功能上正确（后者覆盖前者），但语义上多余。

**建议**：复用 `args.show_regime` 分支中已创建的 `mrf`，避免重复 `prepare`。

---

### 详细评审

#### 信号判断逻辑

| 指标 | 实现 | 评估 |
|------|------|------|
| **RSI14** | EWM Wilders (alpha=1/14)，`min_periods=14` | ✅ 正确 |
| **MA5/10/20** | `rolling(n, min_periods=n).mean()` | ✅ 正确 |
| **MACD (12,26,9)** | 标准 EMA 差值，DIF-SIGNAL | ✅ 正确 |
| **MACD Histogram** | DIF - SIGNAL | ✅ 正确 |
| **BEAR 阈值** | `close < MA20 AND RSI14 < 40` | ✅ 符合 SPEC |
| **NEUTRAL 阈值** | `RSI14 < 50`（RSI预警）或 `MA5<MA10<MA20 AND MACD<0`（趋势确认） | ✅ 符合 SPEC |
| **BULL 退出** | `close > MA20 AND RSI14 > 50 AND MA5>MA10>MA20` | ✅ 符合 SPEC |

#### 连续确认机制（`confirm_days=2`）

`_consecutive_sum` 实现：

```
series  = [F, F, T, T, T, F, T, T]
result  = [0, 0, 1, 2, 3, 0, 1, 2]
```

**评估**：✅ 实现正确。
- 连续计数，只在条件为 False 时重置为 0
- 严格按"连续交易日"计数
- `bear_consec >= n` 触发 BEAR 状态，意味着**条件满足后第 n 天**才切换状态（设计合理，防抖）

#### 退出条件严格性

| 对比维度 | 进入 BEAR | 退出 BEAR → BULL |
|----------|-----------|-------------------|
| 收盘/MA20 | < MA20 ✅ | > MA20 ✅ |
| RSI14 | < 40 ✅ | > 50 ✅（更严格：50 > 40） |
| MA 排列 | 无要求 | MA5>MA10>MA20 ✅（新增要求） |

**评估**：✅ 退出条件比进入更严格，设计合理（防止频繁切换）。

#### 边界情况处理

| 场景 | 处理方式 | 评估 |
|------|----------|------|
| 查询日 = 周末/节假日 | 自动回溯到最近的前一个交易日 | ✅ 合理 |
| RSI 数据不足 30 天 | `min_periods=30` 保守等待，EWM 方式则从 `min_periods=14` 开始输出 | ⚠️ EWM 版本 `min_periods=14`，但 warmup 取 60 天，实际影响有限 |
| 数据区间早于起始日 | 抛出明确 ValueError | ✅ 合理 |

#### 与 SPEC v2.1 一致性

基于代码注释中声明的规则与实现对照：

| SPEC 声明 | 代码实现 | 一致性 |
|-----------|----------|--------|
| BEAR: 收盘<MA20 AND RSI<40 | `bear_cond = (~close_above_ma20) & rsi_below_40` | ✅ |
| NEUTRAL: RSI<50 | `neutral_rsi_cond = rsi_below_50` | ✅ |
| NEUTRAL: MA5<MA10<MA20 AND MACD<0 | `neutral_trend_cond` | ✅ |
| BULL: 收盘>MA20 AND RSI>50 AND MA5>MA10>MA20 | `bull_cond` | ✅ |
| 兜底：BULL | `else: return "BULL"` | ✅ |
| 仓位：BEAR=40%, NEUTRAL=80%, BULL=100% | `bear/neutral/bull_position` 参数 | ✅ |

**总体**：实现与 SPEC 一致。

#### 框架集成正确性

**base_framework.py `_process_buys()`**：
```python
effective_position_size = min(self.position_size, position_limit)
```
✅ 正确使用 `position_limit` 限制单只仓位。

**backtest.py**：
- `run_backtest_year_by_year` 中一次性 `prepare(start_date, end_date)` ✅
- 逐年 `_on_day` 循环中调用 `get_regime(date)`，数据已在 prepare 时预计算 ✅

**simulate.py**：
- `mrf.prepare(target_date, target_date)` ✅（单日场景，warmup 自动扩展）
- `framework.market_regime_filter = mrf` ✅

---

### 总体评价

**通过（需小修）**

MarketRegimeFilter 核心逻辑实现正确，信号判断、连续确认机制、退出条件严格性均符合设计规范。存在的 3 处 P1 问题均为次要优化项（死代码、warmup 参数、重复初始化），不影响功能正确性。建议修复 P1 后再合并。

---

*评审时间：2026-04-02*
*评审人：Oracle（研究员）*
