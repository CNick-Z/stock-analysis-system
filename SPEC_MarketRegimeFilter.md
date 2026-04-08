# SPEC: MarketRegimeFilter — 大盘牛熊过滤器

> 版本：v2.2（代码评审后修复 Fallback + 小修）
> 日期：2026-04-02
> 作者：Fairy（协调）
> 状态：**✅ 评审通过，已修 P0+P1，Forge 实现完成**

---

## 评审结论摘要

| 评审方 | 总体评价 | 关键修改点 |
|--------|---------|-----------|
| **Oracle** | 🔴 需修改后通过 | 回测数据无法核实、仓位逻辑不明确、RSI分层预警、NEUTRAL分支不触发 |
| **Trader** | ⚠️ 需修改后通过 | 仓位偏低、缺连续确认、缺退出条件、缺V形底仓保护 |

**v2.1 改动**：在 v2.0 基础上，老板确认满仓奔跑（不融资，100%为上限）。

**v2.0 改动**：全部采纳 Oracle 和 Trader 的 P0/P1 修改建议（见§8）。

---

## 1. 背景与目标

### 问题
V8 策略在熊市期间出现大幅回撤，需要引入大盘牛熊判断机制，在熊市时降低仓位。

### 目标
- 实现 `MarketRegimeFilter` 模块（市场温度计）
- 合入 `BaseFramework`，每日输出仓位上限
- 三窗口+扩展回测验证

---

## 2. 研究结论

### Oracle 量化结论
- **分层预警**：`RSI<50` 提前15-28天预警，`RSI<40` 确认熊市
- **注意**：之前 SPEC 引用的「+25.5%年化」数据**无法核实**，待实际回测验证

### Trader 经验规则
- **连续2日确认**：避免单日假信号
- **更严格的退出条件**：防止快速进出场，增加摩擦成本
- **V形底仓保护**：熊市保留 40-50% 仓位，避免踏空反弹

---

## 3. 模块设计

### 3.1 MarketRegimeFilter 类

```python
class MarketRegimeFilter:
    """
    大盘牛熊过滤器

    输入：沪深300（CSI300）每日行情
    输出：市场状态 + 仓位上限

    状态枚举：
        BULL     — 牛市（正常仓位）
        NEUTRAL  — 预警/震荡（降仓）
        BEAR     — 熊市（大幅降仓）
    """

    def __init__(
        self,
        index_path: str = "/data/warehouse/indices/CSI300.parquet",
        confirm_days: int = 2,         # 连续确认天数
        bear_position: float = 0.40,   # 熊市仓位（保护）
        neutral_position: float = 0.80, # 震荡仓位（保守）
        bull_position: float = 1.00,     # 牛市仓位（满仓奔跑）
    ):
        ...

    def get_regime(self, date: str) -> dict:
        """
        查询某日期的市场状态

        Returns:
            {
                "date": "2024-01-01",
                "regime": "BULL" | "NEUTRAL" | "BEAR",
                "csi300_close": 3800.0,
                "ma20": 3850.0,
                "rsi14": 45.0,
                "position_limit": 0.80,
                "signal": "rsi_warning",       # 信号标签
                "consecutive_days": 2,          # 连续满足条件天数
            }
        """

    def prepare(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        预计算区间内所有交易日的状态

        Returns:
            DataFrame: date, regime, csi300_close, ma20, rsi14, position_limit, signal, consecutive_days
        """
```

### 3.2 信号判断逻辑（v2.1 — 整合 Oracle + Trader）

**仓位上限（永远不超过 100%，不融资）**：
```
BEAR（熊市）     → position_limit = 0.40 （保护）
NEUTRAL（震荡） → position_limit = 0.80  （保守）
BULL（牛市）    → position_limit = 1.00  （满仓奔跑）
```

**信号判断**：
```
【进入条件 — 连续 confirm_days 日满足】

IF 连续 N 日（收盘 < MA20 AND RSI14 < 40）:
    regime = BEAR
    position_limit = 0.40
    signal = "bear_confirmed"

# RSI<50 提前预警（Oracle：不等到跌破 MA20，提前 15-28 天感知）
ELIF 连续 N 日（RSI14 < 50）:
    regime = NEUTRAL
    position_limit = 0.80
    signal = "rsi_warning"

# 趋势确认（作为二次确认，Trader：成交量+布林带辅助）
ELIF MA5 < MA10 < MA20 AND MACD_histogram < 0:
    regime = NEUTRAL
    position_limit = 0.80
    signal = "trend_bear_confirmed"

【退出条件 — 比进入更严格】

# 退出熊市：价格回归 + RSI 回到 50 以上 + 均线多头排列
ELIF 收盘 > MA20 AND RSI14 > 50 AND MA5 > MA10 > MA20:
    regime = BULL
    position_limit = 1.00
    signal = "bull_confirmed"

【兜底】

ELSE:
    regime = BULL
    position_limit = 1.00
    signal = "bull"
```

**参数 N（confirm_days）**：默认 2，即连续 2 日满足条件才确认。（Trader 经验：避免A股T+1下单日假信号被打脸）

**为什么 exit 比 entry 更严格**（Trader 核心经验）：
- 如果用同样的 MA20+RSI<40 判断出熊市，会造成刚跌穿止损/刚涨穿又入场，快速进出场增加摩擦成本
- 退出条件：RSI>50（非仅>40）+ 均线多头排列（非仅价格>MA20）

### 3.3 RSI14 / MA 计算规范

- **RSI 窗口**：14日，取当日前至少 30 个交易日数据（不足 30 天标记为 BULL，保守处理）
- **MA 系列**：使用 simple moving average
- **MACD**：标准参数（12, 26, 9）
- **数据源**：`/data/warehouse/indices/CSI300.parquet`（字段：date/open/high/low/close/volume）

---

## 4. 合入 BaseFramework

### 4.1 框架改动点

`BaseFramework.__init__` 新增参数：
```python
def __init__(
    self,
    ...
    market_regime_filter: MarketRegimeFilter = None,
):
    self.market_regime_filter = market_regime_filter
```

`BaseFramework._process_buys` 改动：
```python
def _process_buys(self, full_df, daily, market):
    # 查询大盘仓位上限
    position_limit = 1.0
    if self.market_regime_filter:
        regime_info = self.market_regime_filter.get_regime(market["date"])
        position_limit = regime_info["position_limit"]
        logger.debug(
            f"  大盘状态: {regime_info['regime']} | "
            f"{regime_info['signal']} | "
            f"仓位上限: {position_limit:.0%} | "
            f"RSI14: {regime_info['rsi14']:.1f}"
        )

    # 可用资金 = 现金 × 仓位上限
    effective_position_size = min(self.position_size, position_limit)
    avail_cash = self.cash * effective_position_size
    # ... 后续资金分配不变
```

**⚠️ 仓位语义明确（Oracle P0 问题）**：
- `position_limit` = **总仓位上限**（账户层面的百分比）
- 不是"单笔打折"（`min(15%, 30%) = 15%` 的错误不会再发生）
- 示例：账户50万，BEAR仓位40%，则可用资金 = 50万 × 40% = 20万

### 4.2 统一回测入口

```python
# backtest.py
parser.add_argument("--market-filter", action="store_true",
                    help="启用大盘牛熊过滤器")
parser.add_argument("--filter-confirm-days", type=int, default=2,
                    help="连续确认天数（默认2）")

# simulate.py
parser.add_argument("--show-regime", action="store_true",
                    help="显示当日大盘状态")
```

---

## 5. 回测验证方案

### 5.1 回测窗口（v2 — 扩展）

| 窗口 | 时间段 | 说明 |
|------|--------|------|
| 窗口A | 2024-01-01 ~ 2024-12-31 | 牛市 |
| 窗口B | 2022-01-01 ~ 2023-12-31 | 熊市 |
| 窗口C | 2020-01-01 ~ 2021-12-31 | 震荡/V形 |
| **新增** | **2015-06-01 ~ 2016-12-31** | **股灾压力测试** |
| **新增** | **2018-01-01 ~ 2019-06-30** | **2018熊市** |
| **新增** | **2010-01-01 ~ 2012-12-31** | **三年连跌大熊市** |

### 5.2 对比指标

- 年化收益率
- 夏普比率
- 最大回撤
- 仓位分布（平均持仓比例）
- 进出次数 vs 基准

### 5.3 验收标准

- 熊市窗口：最大回撤显著缩小（待实测）
- 牛市窗口：收益不受明显负面影响
- 震荡窗口：收益基本持平

### ⚠️ 回测数据声明

> **数据声明**：之前 SPEC 引用的「年化+25.5%」数据**无法核实**，该数字在 V8 18年历史回测中未出现。实际效果以本次三窗口+扩展回测为准，Oracle 负责核实。

---

## 6. 文件结构

```
projects/stock-analysis-system/
├── simulator/
│   ├── market_regime.py      # 【新增】MarketRegimeFilter 类
│   ├── base_framework.py     # 【修改】合入 market_regime_filter
│   └── position_sizer.py      # 现有，不动
├── backtest.py               # 【修改】--market-filter 参数
├── simulate.py               # 【修改】--show-regime 参数
└── SPEC_MarketRegimeFilter.md  # 本文档（v2.0）
```

---

## 7. 依赖关系

- `MarketRegimeFilter` 依赖 `CSI300.parquet`（已有）
- 不依赖策略私有数据，独立模块
- RSI/MA 计算在 Filter 内部，不影响 DataLoader

---

## 8. v2.0 改动清单（整合两份评审）

| # | 问题 | 来源 | v1 方案 | v2 方案 |
|---|------|------|---------|---------|
| 1 | 回测数据无法核实 | Oracle P0 | +25.5%年化（不可验证） | 标注"待实测"，以实际回测为准 |
| 2 | 仓位语义不明确 | Oracle P0 | `min(position_size, limit)` | `position_limit` = 账户总仓位上限 |
| 3 | NEUTRAL分支不触发 | Oracle P1 | RSI<40时BEAR已触发，NEUTRAL无效 | NEUTRAL 改为独立 RSI<50 预警触发 |
| 4 | RSI阈值单一 | Oracle P1 | RSI<40单一阈值 | RSI<50→NEUTRAL, RSI<40→BEAR（分层） |
| 5 | 缺连续确认 | Trader | 单日信号 | 连续2日确认（N可配置） |
| 6 | 缺熊市退出条件 | Trader | 无 | RSI>50 + MA多头排列 + 连续N日确认 |
| 7 | BEAR仓位30%过低 | Trader | 30% | **40%**（V形底仓保护） |
| 8 | NEUTRAL仓位70%偏低 | Trader | 70% | **80%** |
| 9 | 缺成交量辅助确认 | Trader | 无 | 可选（未来扩展） |
| 10 | 回测窗口不足 | Oracle | 3窗口 | 扩展至6窗口（含2010/2015/2018） |

---

## 9. 评审清单（v2.0 确认）

- [x] **Oracle 评审**：✅ 通过（需实测验证数据）
- [x] **Trader 评审**：✅ 通过（需实现上述修改）
- [x] **Fairy 汇总**：已整合两份评审，v2.0 已更新
- [ ] **Forge 实现**：待启动
- [ ] **三窗口+扩展回测**：待执行
- [ ] **代码评审（三位Agents）**：待安排

---

## 10. 待办事项

| 负责人 | 任务 | 依赖 | 状态 |
|--------|------|------|------|
| Oracle | 执行扩展回测（6窗口），提供真实数据 | 本SPEC定稿 | 待启动 |
| Forge | 实现 MarketRegimeFilter | SPEC v2.0 | 待启动 |
| Forge | 合入 BaseFramework | MarketRegimeFilter实现 | 待启动 |
| Forge | 回测验证 | 合入完成 | 待启动 |
| Fairy | 组织代码评审（Oracle+Trader+Forge） | 回测通过 | 待安排 |
