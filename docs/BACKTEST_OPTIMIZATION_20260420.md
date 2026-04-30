# 回测框架性能优化备忘录

> 2026-04-20 Fairy @ Team Fairy
> 起因：老板要求优化回测效率和可视化输出

---

## 一、已确认的性能瓶颈

### 1.1 `load_strategy_data` 全年数据加载

| 数据 | 耗时 | 说明 |
|------|------|------|
| daily 数据 | 1.2s | 123万行 |
| technical 指标 | 1.8s | |
| **money_flow 计算** | **10.0s** ⚠️ | 最大瓶颈 |
| merge + 其他 | ~3s | |
| **总计** | **~31s** | 全年一次性 |

**性质**：一次性开销，全年数据加载后缓存在内存，后续每日模拟盘不重复加载。

### 1.2 money_flow 内部耗时拆解

`calculate_money_flow_indicators()` 内部各步骤：

| 步骤 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| PJJ ewm (alpha=0.9) | 4.36s | **0.26s** | **17x** ⚡ |
| GJJ ewm (span=8) | 0.72s | ~0.2s | ~3x |
| LLJX ewm (span=3) | 0.69s | ~0.2s | ~3x |
| 周量 rolling(5) | 0.79s | ~0.2s | ~4x |
| **money_flow 总计** | **10.0s** | **~3.0s** | **3x** ✅ |
| `rolling(5)` | 0.79s | 可优化 |

**根因**：PJJ 计算那步用了 `g['close'].transform(lambda s: (df.loc[s.index, ...]).ewm())`，每个 group 都对整列做 `df.loc` 索引，5000只股票 × 242天 = 120万次索引。

---

## 二、已实施的改动

### 2.1 V3 跳过 money_flow 计算 ✅

**发现**：V3 策略（WaveChanV3）代码中 0 处资金流引用，V8 用了 5 处。但 `backtest.py` 和 `simulate.py` 写死了 `add_money_flow=True`。

**改动**：
- `backtest.py` — V3 策略自动设置 `add_money_flow=False`
- `simulate.py` — 同上

**效果**：V3 回测/模拟盘加载省 10 秒（31s → ~21s）

**备份**：
```
backtest.py.bak.20260420_0850
simulate.py.bak.20260420_0850
```

---

## 三、佣金模型改动（今天同时做的）

### 3.1 A股分级税费 ✅

**发现**：从老板实盘交易记录反推的真实费率：

| 品种 | 费率 |
|------|------|
| ETF 买入/卖出 | 佣金 万1（无印花税） |
| 股票 买入 | 佣金 万1.1 |
| 股票 卖出 | 佣金 万1.1 + 印花税 万5 = 万6.1 |

**改动**：`simulator/base_framework.py`
- 删除单一 `commission_pct` 参数
- 新增分级常量：`STOCK_BUY_COMMISSION`、`STOCK_SELL_COMMISSION`、`STOCK_STAMP_TAX`、`ETF_COMMISSION`
- 新增方法：`_is_etf()`、`_buy_cost()`、`_sell_proceeds()`
- 持仓记录新增 `is_etf` 字段

**备份**：`base_framework.py.bak.20260420_0821`

**受影响文件**：`v3_weekly_filter_compare.py`（独立研究脚本，已修复为独立常量，不依赖框架）

---

## 四、已实施的优化

### 4.1 money_flow 结果缓存 ✅ 2026-04-20
**方案**：算完后存 parquet，下次直接读
```
/data/warehouse/money_flow/year=2024/data.parquet (139MB)
```
**改动**：`calculate_money_flow_indicators()` 增加缓存读写逻辑
**效果**：V8 每次回测省 ~10 秒（money_flow 计算被跳过）

### 4.2 PJJ/eewm/rolling 向量化优化 ✅ 2026-04-20
**原理**：pandas 2.x 支持 `groupby().ewm()` / `groupby().rolling()` 直接调用
**改动**：4 个 `transform(lambda)` 全部改为直接调用 + `.values`
- PJJ: `g['close'].transform(lambda s: (df.loc[...]...)` → `pjj_raw.groupby().ewm().values`
- GJJ/LLJX/周量：同上
**验证**：结果与原版完全一致（max diff < 1e-10）
**效果**：money_flow 10s → 3s，整体回测 46s → 39s

---

## 五、可视化方向（老板提到的另一个优化目标）

### 5.1 待实现
- Equity Curve 图（策略 vs 沪深300 benchmark）
- 回撤图
- Trade 详情表（入场/出场/盈亏）
- 月度收益热力图

### 5.2 工具选择建议
- `plotly` — 好看，交互式，可导出 HTML，适合报告
- 不建议 matplotlib（太丑）

---

## 六、框架调用关系（备忘）

```
auto_trade_executor.py  (定时调度)
         ↓ subprocess
simulate.py             (模拟盘入口)
         ↓ 直接调用
BaseFramework.run_simulate()

backtest.py              (回测入口)
         ↓ 直接调用
BaseFramework.run_backtest()

注意：run_simulate 和 run_backtest 是 BaseFramework 的两个独立方法，不互相调用。
```

---

## 七、安全备忘

### 7.1 Git 状态
- 最后一次提交：2026-04-16（4天前）
- 今天的改动**未提交**

### 7.2 备份文件
```
simulator/base_framework.py.bak.20260420_0821   ← 佣金分级改动后备份
backtest.py.bak.20260420_0850                  ← V3 money_flow 跳过改动后备份
simulate.py.bak.20260420_0850                  ← V3 money_flow 跳过改动后备份
utils/data_loader.py.bak.20260420_0924_pjj     ← PJJ优化改动后备份（最新）
```

### 7.3 回滚命令
```bash
# 回滚佣金改动
cp simulator/base_framework.py.bak.20260420_0821 simulator/base_framework.py

# 回滚 V3 money_flow 改动
cp backtest.py.bak.20260420_0850 backtest.py
cp simulate.py.bak.20260420_0850 simulate.py

# 回滚 PJJ/ewm/rolling 向量化优化（慎！会丢失缓存优化）
cp utils/data_loader.py.bak.20260420_0924_pjj utils/data_loader.py
```
