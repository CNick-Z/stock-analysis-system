## Forge 代码自检

### 评审文件

- **market_regime.py**：⚠️ — 核心逻辑正确，接口设计合理，但有 2 处代码冗余和 1 处潜在 NaN 问题
- **base_framework.py**：✅ — 框架集成正确，None 降级处理完善，逻辑清晰
- **backtest.py**：✅ — CLI 参数传递正确，逐年回测设计合理，集成无误
- **simulate.py**：✅ — MarketRegimeFilter 集成正确，--show-regime 流程合理

---

### 发现的 bug/问题

#### 1. market_regime.py — RSI 计算段存在**废弃变量**（代码冗余）

**位置**：`prepare()` 方法，约 RSI 计算部分

```python
# ⚠️ 废弃变量，以下两行计算后从未被使用
gain = delta.clip(lower=0).rolling(30, min_periods=30).mean()
loss = (-delta.clip(upper=0)).clip(lower=0).rolling(30, min_periods=30).mean()

# ✅ 实际生效的是这段 Wilders RSI（EWM 实现）
avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi14"] = (100 - 100 / (1 + rs)).clip(upper=100)
```

**影响**：无功能性影响（废弃变量），但影响代码可读性，建议删除。

---

#### 2. market_regime.py — RSI 计算中 `loss=0` 时**未完全防护 NaN**

**位置**：`rs = avg_gain / avg_loss.replace(0, np.nan)`

**场景**：当 `avg_loss = 0`（即过去14个交易日连续上涨）时：
- `rs = inf`，`1 + rs = inf`，`100 / inf = 0`
- `rsi14 = 100 - 0 = 100` ✅（结果正确，只是没有 NaN）

但若 `avg_gain = 0` 且 `avg_loss = 0`（价格完全不变），则 `0/0 → NaN`。

**严重程度**：低 — 实际数据中 CSI300 连续14天完全不涨不跌几乎不可能。现有 `clip(upper=100)` 提供了基本保护。

---

#### 3. market_regime.py — `get_regime()` 回溯逻辑**稍显冗余**

**位置**：`get_regime()` 内

```python
if row.empty:
    candidates = self._df[self._df["date"] <= dt]
    if candidates.empty:
        raise ValueError(...)
    # ⚠️ 回溯找到了，但后面又做了一次 == dt 的精确查询
    row = candidates
    date = str(candidates.iloc[-1]["date"].strftime("%Y-%m-%d"))
    dt = pd.to_datetime(date)        # 这里 dt 已经是字符串转回 datetime
    row = self._df[self._df["date"] == dt]  # 冗余查询：== dt 必然等于 candidates.iloc[-1]
```

**分析**：逻辑正确（结果等效），但有冗余查询。在 `candidates.iloc[-1]` 已知的情况下，可直接 `row = candidates.iloc[[-1]]` 避免再查一次。

**影响**：无功能性 bug（查询结果相同），性能微损。

---

#### 4. market_regime.py — `macd_hist < 0` 在 `neutral_trend_cond` 中的**语义矛盾**（设计审视，非 bug）

**位置**：`neutral_trend_cond = ma5_below_ma10 & ma10_below_ma20 & macd_hist_negative`

**观察**：当 MA5 < MA10 < MA20（空头排列）时，MACD 通常也是负值，`macd_hist < 0` 条件是**多余的**（隐含在均线空头排列里）。但这不是 bug，只是"防御性冗余"——即使均线空头排列，若出现反弹（MACD 柱由负转正），则 `neutral_trend_cond = False`，符合预期。

**结论**：设计合理，无需修改。

---

#### 5. base_framework.py — `on_tick` 兼容模式**参数签名不统一**

**位置**：`_on_day()` 方法

```python
try:
    self._strategy.on_tick(r, pos, market)
except TypeError:
    self._strategy.on_tick(r, pos)  # 兼容2参数策略
```

**观察**：Strategy 接口定义的是 `on_tick(row, pos, market)`（3参数），但框架内做了 try/except 兼容 2 参数版本。backtest.py 中的 `_on_day` 调用传入 `market["date"]`（3参数），simulate.py 同。

**分析**：这是有意为之的向前兼容设计，不影响功能。但建议在 Strategy 接口注释中注明"策略可选择实现 2 参数版本以兼容旧代码"。

---

### 已修复的问题

无（本次为首次自检，代码为初始实现版本，未有修复记录）。

---

### 总体评价：**通过**

| 文件 | 评级 | 说明 |
|------|------|------|
| market_regime.py | B | RSI 废弃变量需清理；NaN 防护可加强（非阻塞） |
| base_framework.py | A | 框架集成完善，None 降级正确 |
| backtest.py | A | CLI 参数正确，逐 year 回测设计合理 |
| simulate.py | A | MarketRegimeFilter 集成正确，--show-regime 流程清晰 |

**交付条件**：可合并，但建议在下个迭代中清理 market_regime.py 中的废弃变量（`gain` / `loss` rolling 计算），并考虑为 `get_regime()` 的回溯路径补充一个边界单元测试（传入周末/节假日日期，验证返回最近交易日）。
