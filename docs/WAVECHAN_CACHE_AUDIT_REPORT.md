# 波浪缠论策略缓存系统代码审核报告

**审核日期**: 2026-04-10
**审核人**: Forge
**审核对象**: `scripts/rebuild_wavechan_l1.py` + `utils/wavechan_cache.py` + `strategies/wavechan_selector.py` + `strategies/wavechan_fast.py`

---

## 📊 执行摘要

波浪缓存重建速度慢的**根本原因**：V3 算法采用逐K线 CZSC 笔识别 + 单线程顺序处理，无并行化。

发现 **2 个 Bug** 和 **4 个严重性能问题**。

---

## 🔍 当前运行状态

| 指标 | 值 |
|------|-----|
| 进程 PID | 2834323 |
| 运行时间 | 2小时25分钟 |
| CPU占用 | 144%（单核跑满） |
| 当前进度 | 2022年，已完成 4750/5000 只 |
| 速度 | ~31只/分钟 |
| 预估剩余 | 10-15小时（2018-2025共8年） |

---

## 🐛 Bug 问题

### Bug 1: `rebuild_l2` 估算股票数量逻辑错误

**文件**: `utils/wavechan_cache.py` 第 434-442 行

```python
for yr in years:
    yr_data = self.l1l2.read_l2(yr, 1)  # 只检查1月！
    if not yr_data.empty:
        total_symbols = yr_data['symbol'].nunique()
        break
```

**问题**: 只检查 L2 的 1 月数据，如果 L2 是空的（如首次重建），`total_symbols` 会是 0，导致 fallback 到 `5400 * len(years)` 估算，后续 `progress_callback` 判断 `len(results) % 100 == 0` 永远不成立，**进度汇报失效**。

---

### Bug 2: L1/L2 目录空目录残留

**现象**: 以下目录存在但为空（26字节 = 空目录）：
- `l2_hot_year=2018_month=01~12`
- `l2_hot_year=2019_month=01~12`
- `l2_hot_year=2020_month=01~12`
- `l2_hot_year=2021_month=01~12`
- `l2_hot_year=2022_month=01~12`
- `l2_hot_year=2023_month=01~12`
- `l2_hot_year=2024_month=01~12`

**原因**: 脚本把数据先写入 `/tmp/wavechan_rebuild_YEAR/` 临时目录，最后才写入 L1。但处理顺序是先 2018-2021，导致这些 L2 空目录被"预留"但永远不会被填充。

**影响**: 目录污染，不影响功能（L1 才是冷数据存储，L2 热数据不在这里）。

---

## ⚡ 性能问题（重建慢的根本原因）

### 性能问题 1: 逐K线 CZSC 计算（最严重）

**文件**: `strategies/wavechan_selector.py` 第 360-456 行

```python
def _compute_symbol_scores(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    for i in range(n):  # n = 约250天/年
        bar = {...}
        snap = engine.feed_daily(bar)  # 每只股票每天调用一次 CZSC
```

**问题**:
- 对每只股票的每一天，逐根K线调用 `engine.feed_daily()`
- 每根K线触发 CZSC 笔识别 + WaveCounterV3 波浪计算
- 5000只股票 × 250天 = **125万次** CZSC 调用

**对比**: `wavechan_fast.py` 提供了 numba 加速的向量化版本，每20天计算一次（不是每天）。

---

### 性能问题 2: 单线程无并行处理

**文件**: `strategies/wavechan/v3_l2_cache/strategy.py` 第 2707-2726 行

```python
# BatchWaveBuilder.run_batch
for i, symbol in enumerate(batch_symbols):  # 顺序处理！
    cache = self.build_symbol(symbol, start_date, end_date)  # 单线程
```

**问题**:
- `batch_size=500` 只是概念，实际是顺序处理 500 个
- 没有使用 `multiprocessing.Pool` 或多进程
- 8核 CPU 只用了 1 核

---

### 性能问题 3: 重复读取重叠数据

**文件**: `utils/wavechan_cache.py` 第 451-455 行

```python
for month in range(1, 13):
    lookback_start = (dt - timedelta(days=150)).strftime("%Y-%m-%d")
    lookback_end = (dt.replace(day=28) + timedelta(days=4)).strftime("%Y-%m-%d")
    input_df = self.l1l2.read_range(lookback_start, lookback_end)
```

**问题**: 每月重建时重新读取 150 天 lookback，相邻月份数据 80% 以上重叠。

---

### 性能问题 4: 异常被静默吞掉

**文件**: `utils/wavechan_cache.py` 第 465-467 行

```python
except Exception as e:
    logger.debug(f"[WaveChanCache] {sym} 异常: {e}")  # debug 级别！
    continue
```

**问题**: 股票处理异常时只在 debug 日志显示，用户完全不知道，可能导致**部分股票信号缺失**。

---

## ⚠️ 重要：向量化 vs 逐K线计算的权衡

### 两种方案的本质差异

| 方案 | 计算方式 | 状态修正 | 准确性 |
|------|---------|---------|--------|
| 逐K线 | 每天喂入新K线，触发全局重算 | ✅ 支持 | ✅ 真实模拟 |
| 向量化 | 一次性输入全部数据，直接输出结果 | ❌ 不支持 | ⚠️ 待验证 |

### V3 算法的核心设计

V3 波浪理论**每天增量会对前一天的状态重新修订**：
```
day 1 → feed_bi → recalc → wave_state_t1
day 2 → feed_bi → recalc → wave_state_t2  ← 可能推翻 day1 的波浪
day 3 → feed_bi → recalc → wave_state_t3  ← 可能推翻 day1, day2 的波浪
```

### 向量化可能的问题

`wavechan_fast.py` 的向量化版本**可能**存在以下问题：

1. **"回头看"逻辑** — 如果 `_scan_waves` 算法需要"等后续走出来才能确认前面的浪"，向量化一次性计算会失去这个特性
2. **全局最优搜索** — V3 的浪型识别是全局搜索的，可能需要未来的信息来修正历史判断
3. **不确定性** — 需要实际测试对比才能确认

### 建议：先测试再优化

建议等当前重建完成后，做对比测试：
- 方式 A：逐K线计算（当前逻辑）
- 方式 B：向量化一次性计算
- 对比结果是否一致

---

## 📈 优化方案

### 方案 A：并行化（安全，可立即实施）

| 改动 | 预期加速 |
|------|---------|
| 用 `multiprocessing.Pool(8)` | **6-8倍** |

**优点**: 不改变算法逻辑，结果完全一致
**缺点**: 仍有逐K线计算开销

### 方案 B：向量化（风险未知，需测试验证）

| 改动 | 预期加速 |
|------|---------|
| 用 `wavechan_fast.py` 的 numba 版本 | **10-20倍** |

**优点**: 加速效果显著
**缺点**: 可能导致波浪状态失真
**前提**: 必须先做对比测试验证

### 方案 C：并行 + 向量化（收益最大，风险最大）

结合 A + B，预期 **50-100倍加速**。

---

## 📋 优化建议（按优先级）

| 优先级 | 问题 | 优化方案 | 预期收益 |
|--------|------|----------|---------|
| **P0** | Bug 1 | 修复 `total_symbols` 估算逻辑 | 进度显示正确 |
| **P1** | Bug 2 | 清理空 L2 目录残留 | 目录整洁 |
| **P1** | 性能问题 4 | 把 `logger.debug` 改为 `logger.warning` | 可观测性 |
| **P1** | 性能问题 3 | 缓存 lookback 数据，避免月份之间重复读取 | **2-3倍加速** |
| **P2** | 性能问题 2 | 添加 `multiprocessing.Pool` 并行处理 | **6-8倍加速** |
| **P2** | 性能问题 1 | 集成 `wavechan_fast.py` 的 numba 向量化 | **10-20倍加速** |

---

## 🛠️ 快速修复清单

1. [ ] 修复 `rebuild_l2` 中 `total_symbols` 估算逻辑
2. [ ] 把异常日志从 `debug` 改为 `warning`
3. [ ] 添加并行化支持（`multiprocessing.Pool`）
4. [ ] 清理空 L2 目录残留
5. [ ] 编写对比测试脚本，验证向量化是否可用

---

## 📁 相关文件

| 文件 | 用途 |
|------|------|
| `scripts/rebuild_wavechan_l1.py` | 重建入口脚本 |
| `utils/wavechan_cache.py` | 三层缓存管理器 |
| `strategies/wavechan_selector.py` | WaveChan 选股评分器 |
| `strategies/wavechan/v3_l2_cache/strategy.py` | BatchWaveBuilder 批量构建 |
| `strategies/wavechan_fast.py` | numba 加速版本（未被使用） |
| `strategies/wavechan_v3.py` | V3 波浪引擎核心 |
