## Trader 代码评审

### 评审文件
- market_regime.py：⚠️ 整体可用，核心逻辑正确，但有1个P0问题和1个P1冗余
- base_framework.py：✅ 仓位语义正确，框架设计合理
- backtest.py：✅ 回测入口正确，支持逐年回测避免OOM，逐年 reset trades 缓冲区的设计合理
- simulate.py：✅ 模拟盘入口正确，状态恢复机制完善

---

### 关键问题（P0）

#### P0-1：Fallback 默认 BULL（满仓）—— 高风险

**位置**：`market_regime.py` 第 129 行，`assign_regime()` 函数的 else 分支：

```python
def assign_regime(row):
    if row["bear_consec"] >= n:
        return "BEAR"
    elif row["neutral_rsi_consec"] >= n:
        return "NEUTRAL"
    elif row["neutral_trend_consec"] >= n:
        return "NEUTRAL"
    elif row["bull_consec"] >= n:
        return "BULL"
    else:
        return "BULL"  # ← 问题在这里
```

**问题**：当四种状态都不满足时（牛市确认条件不充分、熊市确认也不充分、震荡也不充分），系统 fallback 到 `BULL`（100%仓位）。

**实盘风险**：在市场从 BEAR/NEUTRAL 向 BULL 过渡的模糊期，所有条件都未达到连续确认，但系统以 100% 仓位运行。这是极其危险的。

**建议修改**：
```python
else:
    return "NEUTRAL"  # 模糊期默认降档到中性，用80%仓位
```
或至少记录一条 WARNING log。

---

### 建议修改（P1）

#### P1-1：RSI 计算存在死代码

**位置**：`market_regime.py` 第 82-85 行

```python
# RSI14（取前30个交易日数据计算）
delta = close.diff()
gain = delta.clip(lower=0).rolling(30, min_periods=30).mean()   # ← 声明但从未使用
loss = (-delta.clip(upper=0)).clip(lower=0).rolling(30, min_periods=30).mean()  # ← 声明但从未使用
# 用 .iloc[-1] / .mean() 技巧实现 Wilders RSI
avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
```

`gain` 和 `loss` 变量被计算但最终没有被使用（后面用的是 `avg_gain/avg_loss` 的 EWM 方式）。建议删除这两行以避免混淆。

---

#### P1-2：退出熊市条件偏严，可能导致踏空

**位置**：`market_regime.py` 第 40 行，BEAR→NEUTRAL/BULL 的退出条件：

```python
# BULL（退出熊市）：收盘 > MA20 AND RSI14 > 50 AND MA5 > MA10 > MA20
df["bull_cond"] = (
    df["close_above_ma20"] & df["rsi14_above_50"] & df["ma5_above_ma10"] & df["ma10_above_ma20"]
)
```

**问题**：
1. 需要同时满足 4 个条件（收盘>MA20 AND RSI>50 AND MA5>MA10 AND MA10>MA20），过于严格
2. 2日连续确认意味着从熊市最低点算起，至少需要 2 个交易日才能触发退出信号
3. 实盘中，MA 均线滞后严重，熊市反弹初期 MA5/MA10 往往还在 MA20 下方，会导致严重踏空

**建议**：
- 方案A：降低退出条件，只要求 `收盘>MA20 AND RSI>50`（去除 MA 均线排列要求），均线排列作为加分项
- 方案B：保留当前条件，但将连续确认天数从 2 降为 1（快速响应，但需要在框架层处理T+1）
- 方案C：增加"试探性加仓"逻辑——NEUTRAL 后即可用 60% 仓位，BULL 确认后用 100%

---

#### P1-3：三档仓位的实盘合理性

| 档位 | 当前参数 | 评价 |
|------|---------|------|
| BEAR | 40% | ✅ 偏保守但合理，熊市保留60%现金 |
| NEUTRAL | 80% | ✅ 可接受，保留20%现金应对突发 |
| BULL | 100% | ⚠️ 需确认框架层 `max_positions=5` 限制，实际不会超过5只×20%=100% |

**补充**：在 A 股环境中，BEAR=40% 意味着最多持有 2 只（40%÷20%=2只），这个数字偏少。建议熊市时允许更多只股票但单只仓位更小，或者直接确认 BEAR=40% 是正确的风控选择。

---

#### P1-4：仓位计算无"单笔打折"问题（验证通过）

**验证代码**（`base_framework.py` `_process_buys`）：

```python
effective_position_size = min(self.position_size, position_limit)
avail_cash = self.cash * effective_position_size
slots = self.max_positions - len(self.positions)
per_stock_cash = avail_cash / fill_count
```

- `self.position_size` = 每只股票分配仓位的默认比例（默认20%）
- `position_limit` = 账户级别的市场状态上限（40%/80%/100%）
- `min(20%, 40%)` = 20% ✅ —— 不会出现 `min(15%, 40%) = 15%` 的不合理问题
- 即使用户把 `position_size` 设为 15%，结果也是 `min(15%, 40%) = 15%`（用户的保守设置被尊重，也是合理的）

**结论**：仓位语义实现正确，没有语义错误。

---

#### P1-5：非交易日回溯处理

**位置**：`market_regime.py` 第 66-71 行，`get_regime()` 方法：

```python
if row.empty:
    candidates = self._df[self._df["date"] <= dt]
    if candidates.empty:
        raise ValueError(...)
    row = candidates
    date = str(self._df[self._df["date"] <= dt].iloc[-1]["date"].strftime("%Y-%m-%d"))
```

**评价**：✅ **正确**。实盘中非交易日没有交易，自然沿用最近一个交易日的大盘状态。在回测中 `_on_day` 只在真实交易日调用，此逻辑仅用于 `get_regime()` 直接查询或模拟盘单日运行，不会导致数据泄漏。

---

#### P1-6：连续确认能否有效过滤A股 T+1 假信号？

**分析**：
- 2日连续确认确实能过滤相当一部分"一日游"假信号 ✅
- 但 T+1 限制的影响不在市场状态判断层，而是在**持仓出场层**（`base_framework.py` `_process_sells` 检查 `next_limit_up/down`）——这部分处理是正确的
- 真正的问题是：**从熊市退出条件本身就过于严格**，2日确认反而加剧了踏空，而不是在防假信号

---

### 总体评价：需修改后通过

**通过标准**：修复 P0-1（Fallback 改为 NEUTRAL）后，可进入实盘模拟测试。

**保留意见**：P1-2（退出条件过严）建议尽快优化，否则会在牛市初期造成严重踏空，影响策略整体收益表现。

---

*评审人：Trader（交易员视角）*  
*评审时间：2026-04-02*
