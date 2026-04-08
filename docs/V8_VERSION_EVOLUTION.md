# V8 策略版本演进记录

> 创建：2026-03-30 20:03
> 负责人：Fairy
> 状态：整理中

---

## 一、Git 提交历史中的 V8 改动（2026-03-30 当日）

| Commit | 时间 | 改动内容 |
|--------|------|---------|
| 2c89275 | 13:57 | 📁 策略目录重构：创建 strategies/score/v4/ 和 v8/ 子目录 |
| b052ece | 16:37 | 备份v8（vol_ratio已移除，待调整为0.8）|
| 39c8ffc | 16:40 | 调整 vol_ratio 阈值 1.25 → 0.8 |
| 40051e8 | 16:45 | vol_ratio阈值1.25→0.8（IC/IR调优，8年IC=+0.0054）|
| 8a227d1 | 17:41 | **当前最新**：vol_ratio剔除改为低量加分（<0.71区间IC最优+0.0037）|

---

## 二、本地未提交的重要文件

| 文件 | 说明 | 与最新代码差异 |
|------|------|--------------|
| `backtest_score_v8.py` | 18年回测版本，**从未提交git** | 含 MA死叉修复，是最新可回测版本 |
| `backups/20260330/v8_strategy.py` | 09:08快照 | 未包含 vol_ratio 修复 |
| `backups/20260330/backtest_score_v8.py` | 09:08快照 | 同上 |
| `versions/score_strategy_v4_notrail.py` | V4无trail版本 | 最新baseline |

---

## 三、策略版本对照表

### Score 策略演进（versions/ 目录）

| 版本 | 文件 | 核心改动 | 收益 |
|------|------|---------|------|
| v1 | score_strategy_v1_original/baseline | 原始版 nlargest | — |
| v2 | score_strategy_v2_rsi | RSI 50-60过滤 | +76.71% |
| v3 | score_strategy_v3_trailing | trailing stop 15%保本 | — |
| v4 | score_strategy_v4_notrail | 移除死叉，趋势不破一直持有 | +148.84% |
| v5 | score_strategy_v5_rsi_price | RSI 54-58 + 股价3-15元过滤 | — |
| v6 | score_strategy_v6_lowprice | 仅加股价过滤3-15元 | — |
| v7 | （有bug，已废弃）| 错误替换权重 | -49% |
| v8 | backtest_score_v8.py（本地）| V4核心 + IC过滤 + MA出场 | +40.6%（8年）|

---

## 四、当前 V8 最新代码状态（2026-03-30 17:41）

**文件路径**：`strategies/score/v8/strategy.py`（已提交 git）
**Commit**：`8a227d1`

### 当前 vol_ratio 逻辑：
```python
# IC 增强过滤（硬剔除）
exclude_mask = (
    (df['vol_ratio'] > 0.8) |  # ⚠️ 已移除（commit 8a227d1 清理）
    ...
)

# IC 加分（新增）
ic_bonus = 0.0
if df['vol_ratio'] < 0.71:      # IC=+0.0037 最优区间
    ic_bonus += 0.05
```

### 待改参数（基于 IC/IR 分析结果）：
- **RSI 剔除**：<25 → <35（IC提升75%，待改）
- **WR 剔除**：<-95 → <-90（待改）
- **volume_condition**：维持 >1.2（已是相对最优）

---

## 五、版本定标建议

| 版本号 | 文件 | 状态 | 说明 |
|--------|------|------|------|
| V8.0 | backtest_score_v8.py（本地，未提交）| 基准 | 18年+40.6%，MA死叉修复 |
| V8.1 | strategies/score/v8/strategy.py（git 8a227d1）| 最新 | vol_ratio<0.71加分，RSI/WR待改 |

**建议**：以 V8.1 为基准，改完 RSI/WR 后标记为 V8.2。

---

## 六、问题：为何版本看起来"乱"

1. **一个项目里有多套 v8**：`backtest_score_v8.py` 和 `strategies/score/v8/strategy.py` 是两个不同文件
2. **未统一命名规则**：versions/ 里是 v1-v6，strategies/score/ 里是 v4/v8，容易混淆
3. **多次 commit 微调同一个参数**：vol_ratio 在 4 小时内改了 4 次

---

*整理：Fairy 2026-03-30*

---

## V8.3（commit a486c4c）
- **commit**: a486c4c
- **新增**: `data/fundamentals.py` - FundamentalProvider
- **改动**: `should_sell()` 出场逻辑改为"跌破MA20+均线死叉"
- **理由**: V4原始出场逻辑，减少假信号

## V8.3 评审后修复（commit fe27483）
- 字段验证: sma_5/sma_10/sma_20 100%非空 ✅
- 移除兼容性函数中过时的 entry_sma20_le_sma55 说明
- commit: fe27483

## V8.3 最终修正（commit 945c32e）
- 出场改为: 跌破MA20+vol_ratio<0.8（缩量=资金流出代理）
- 原因: 老板确认V4原版是"资金流向为负"，不是MA5<MA10
- commit: 945c32e

## V8.3 二次修正（commit ff03232）
- 出场改为: 跌破MA20+ money_flow_trend==False（V4原版）
- 原因: data_loader已计算money_flow_trend，无需vol_ratio代理
- commit: ff03232

---

## V8.3 评审报告（commit ff03232 评审，2026-03-31 凌晨）

### 评审执行
- **评审人**：Fairy
- **代码**：commit ff03232（strategies/score/v8/strategy.py）
- **回测脚本**：backtest_score_v8.py（已修改，接入 data_loader.py）

---

### 一、入场逻辑 ✅ 全部正确

| 条件 | 实现 | 评审 |
|------|------|------|
| growth_condition | `close >= open AND high <= open * 1.06` | ✅ |
| ma_condition | `sma_5 > sma_10 AND sma_10 < sma_20` | ✅ |
| volume_condition | `vol > vol.shift(1)*1.5 \| vol > vol_ma5*1.2` | ✅ |
| macd_condition | `macd < 0 AND macd > signal` | ✅ |
| jc_condition | SMA5连续2日升 AND SMA20升 AND gap<2% | ✅ |
| trend_condition | `sma_20 < sma_55 AND sma_55 > sma_240` | ✅ |
| rsi_filter | `rsi 50-60` | ✅ |
| price_filter | `close 3-15元` | ✅ |
| IC剔除 | RSI>70/<35, 换手>2.79%, WR<-90, CCI<-200 | ✅ |
| IC加分 | CCI<-100:+0.10, WR<-80:+0.05, 换手<0.42%:+0.05, vol_ratio<0.71:+0.05 | ✅ |

---

### 二、出场逻辑 should_sell()

```python
# 止损
if next_open < entry_price * (1 - 0.05): return True, "STOP_LOSS"

# 止盈
if next_open > entry_price * (1 + 0.15): return True, "TAKE_PROFIT"

# 方案A出场
if close < sma_20 and money_flow_trend == False: return True, "跌破MA20+资金流出"
```

**⚠️ 发现重大数据问题（已修复）**：

| 问题 | 说明 |
|------|------|
| `backtest_score_v8.py` 绕过了 `data_loader.py` | 直接读 Parquet，money_flow_trend 字段不存在 |
| `v8_simulated_trading.py` 同样问题 | 同样直接读 Parquet |
| `data_loader.py` 本身有完整的 money_flow_trend 计算 | LIJIN/GJJ/LLJX/主生量/量基线，基于 8 期/3 期 EMA |
| **结果** | V8.3 的"跌破MA20+资金流出"出场从未真正生效 |

**修复**：`backtest_score_v8.py` 的 `load_year_data()` 改用 `load_strategy_data(years=[year], add_money_flow=True)`
- `money_flow_trend` 非空率：100%
- True/False 分布：约各 50%

---

### 三、回测结果（money_flow_trend 生效后）

**回测配置**：初始资金 100万，持仓上限 5 只，单只仓位 20%，止损 5%，止盈 15%

| 年份 | 收益 | 买入 | 卖出 | 胜率 | 最大回撤 |
|------|------|------|------|------|----------|
| 2024 | **-6.67%** | 16 | 12 | 0% | — |
| 2025 | **+14.21%** | 29 | 25 | 48% | — |
| **合计** | **+6.59%** | 45 | 37 | — | — |

**最终资金**：114.21 万 vs 100 万初始

---

### 四、历史数据对比

| 版本 | 数据状态 | 2024-2025 收益 | 交易笔数 |
|------|---------|---------------|---------|
| V8（vol_ratio修复，commit 8a227d1） | money_flow_trend 缺失（错误） | **+124%** | 72 笔 |
| V8.3（commit ff03232） | money_flow_trend 生效（正确） | **+6.59%** | 45 笔 |

**结论**：之前 +124% 的回测是**假象**——出场条件 `money_flow_trend == False` 从未触发，只靠止损/止盈在交易。V8.3 双重出场条件生效后，大幅收紧了交易频率。

---

### 五、待老板决策

1. **出场条件是否过严？** 2024 年胜率 0%，全年几乎一直在止损。money_flow_trend 双杀条件可能太早打断持仓。
2. **是否改用方案B（vol_ratio < 0.8）** 作为资金流替代？
3. **是否需要资金流只在特定市场环境下生效？**

---

*评审记录：Fairy 2026-03-31 03:52*

---

## V8.3 深度分析报告：2024 年交易记录（2026-03-31 凌晨）

### 📊 2024 交易数据总览

| 项目 | 数值 |
|------|------|
| 总买入 | 16 笔 |
| 总卖出 | 12 笔 |
| 4 笔仍持仓 | 601939、601101、600299、601919、601058 |
| 胜率 | **0%**（0 止盈 / 12 卖出） |
| 总收益 | **-6.67%** |
| 卖出原因分布 | **止损 100%（12/12）** |

---

### 🔍 核心发现

#### 1. 出场原因：100% 止损，无一幸免

所有 12 笔卖出全部触发止损（-5% 到 -8.77%），**止盈 0 笔，资金流出场 0 笔**。

止损幅度分布：
- 8 笔：-5% 到 -6%
- 4 笔：-7% 到 -8.77%（300521、601163、600026）

**根因不是出场条件太松，而是入场质量差**——买完就跌，没有给止盈机会。

---

#### 2. 入场时机高度集中，错过关键月份

全年只在 4 个时间段买入：
- **1月2-3日**（5只）：开局全买在高点，之后 1 月中开始持续下跌
- **7月1日**（4只）：买在 7 月反弹顶部，8 月全线崩塌
- **7月11日**（2只）：同上
- **8月16日**（1只）：600026
- **9月24日**（1只）：601101 → 12月19日止损
- **12月**（4只）：部分仍持仓

**完全错过**：
- **4-5 月大反弹**（每月候选极少，全被 filter 过滤）
- **9月下旬暴涨**（止盈未能触发，因为资金流持续正向）
- **10月暴涨**（候选数 = 0，策略完全停摆）

---

#### 3. 2024 各月候选股票数量

| 月份 | 候选股数 | 关键过滤 |
|------|---------|---------|
| 1月 | 39 | 正常 |
| 4月 | 5 | MA收敛+Trend条件过滤后 |
| 5月 | 12 | 同上 |
| 7月 | 45 | 正常 |
| 8月 | 41 | 同上 |
| 9月 | 52 | 同上（但100%资金流正向，无法出场） |
| **10月** | **0** | **最严格：163只有MA+Trend → 38只价格 → 17只收涨 → 5只放量 → 0只MACD** |
| 12月 | 26 | 正常 |

---

#### 4. October = 策略停摆月（关键发现）

10 月是一年中最大涨幅的月份之一，但候选股票数为 **0**，原因如下：

```
MA收敛 + Trend 过滤：163 只股票满足
    ↓ + RSI 50-60：70 只
    ↓ + 价格 3-15 元：38 只
    ↓ + 收涨（close>=open 且 high<=open*1.06）：17 只
    ↓ + 放量（vol>1.5倍昨日 或 vol>1.2倍MA5）：5 只
    ↓ + MACD条件：0 只 ← 致命一击
```

**为什么 MACD 在 October 杀掉了所有候选？**

October 是一路大涨的月份，MACD 在 0 轴上方已经发散，不满足 `MACD < 0 且 > signal` 的条件。策略要求在 0 轴以下才考虑买入，但大涨的股票 MACD 早已在 0 轴上方。

→ **策略设计与趋势市场的根本矛盾**：要求 MACD 在 0 轴以下 = 要求在低位起涨，但实际上大涨的股票 MACD 已经远离 0 轴。

---

#### 5. 为什么 9月大涨没触发止盈？

9 月候选股有 52 只，但：
- **money_flow_trend = True 比例：100%**（52/52）
- 出场条件 `close < SMA20 AND money_flow_trend == False` → **完全无法触发**
- 9 月大涨，所有持仓都没有跌破 MA20，资金流也持续正向

所以 9 月大涨**完全没有被策略捕捉**——资金流正向保护了持仓，但没有止盈信号。

---

### 📋 结论汇总

| 问题 | 描述 | 严重程度 |
|------|------|---------|
| **MACD 0轴条件** | 趋势市场（10月大涨）时 MACD 在 0 轴上方，导致候选归零 | 🔴 严重 |
| **止损 5% 太紧** | 买完即跌，15% 止盈从未触发，止损先行 | 🔴 严重 |
| **MA收敛条件** | 只选收敛期股票，趋势市场自动失效 | 🟡 中等 |
| **收涨限制 6%** | 大涨日普遍 >6%，候选被过滤 | 🟡 中等 |
| **资金流出场** | 保护了 9 月持仓，但完全错过止盈 | 🟡 中等 |

---

### 💡 策略改进方向建议

1. **MACD 条件放宽**：考虑移除 `MACD < 0` 限制，或改为 `MACD 在 0 轴附近且 MACD > signal`
2. **止损放宽**：从 5% 改为 8%，给趋势市场更多喘息空间
3. **增加趋势市场候选**：当 MA 已多头排列（SMA5>SMA10>SMA20）时，调整选股条件
4. **止盈机制**：加入移动止盈或时间止盈，不依赖单一价格条件

---

*分析：Fairy 2026-03-31 06:46*

---

## v8.4（2026-04-01）— 架构重构 + Forge 评审通过

### 变更
1. **`prepare()` 缓存从 83 列缩减到 8 列**
   - 解决 3.6GB 服务器 OOM 问题
   - 内存：8列 × 1.2M行 ≈ 77MB

2. **`filter_buy` 接口改为 `(full_df, date)`**
   - 框架传完整 year df + 当前交易日
   - 内部查缓存 + 涨跌停过滤

3. **shift/rolling 全部分组化**
   - 所有 `g.shift()` / `g.rolling()` 均加 `groupby('symbol', sort=False)`
   - 消除跨股票数据污染

4. **方案A计数器隔离**
   - `pos.setdefault('_exit_n_days', 0)` 替代 `self._exit_n_days`
   - 多持仓互不干扰

### 评审
- Forge（dev-agent）评审通过 ✅
