# MarketRegimeFilter v2.5 — 三窗口回测结果

**执行时间**: 2026-04-02  
**回测引擎**: v8 策略  
**v2.5 修复**: `per_stock_cash = min(avail_cash / fill_count, cash * position_size * position_limit)`  
**position_limit**: BULL=100%, NEUTRAL=70%, BEAR=30%

---

## v2.5 三窗口回测结果

| 窗口 | 时间段 | 无过滤年化 | 无过滤MaxDD | v2.5有过滤年化 | v2.5有过滤MaxDD | 年化变化 | MaxDD改善 |
|------|--------|-----------|------------|--------------|----------------|---------|----------|
| **Window F** | 2010-01-01~2012-12-31 | **-12.81%** | -45.02% | **-9.75%** | -36.86% | +3.06pp | +8.16pp |
| **Window B** | 2022-01-01~2023-12-31 | **+9.64%** | -10.42% | **+7.47%** | -8.30% | -2.17pp | +2.12pp |
| **Window E** | 2018-01-01~2019-06-30 | **-5.76%** | -23.28% | **-5.58%** | -18.91% | +0.18pp | +4.37pp |

---

## 窗口 Regime 分布

| 窗口 | BULL占比 | NEUTRAL占比 | BEAR占比 | 适合验证 |
|------|---------|------------|---------|---------|
| Window F | 25.5% | 41.8% | **32.6%** | ✅ BEAR高 |
| Window B | 17.6% | **53.1%** | 29.3% | ✅ NEUTRAL高 |
| Window E | 28.3% | **47.6%** | 24.1% | ✅ 熊市代表 |

---

## 分析

### Window F（熊市主导，2010-2012）
- **结论：有效** ✅
- BEAR占比 32.6%，年化从 -12.81% 改善到 -9.75%（+3.06pp）
- MaxDD 从 -45.02% 改善到 -36.86%（+8.16pp）
- 这是 v2.5 效果最显著的窗口，position_limit 30% 在高BEAR期有效限制了亏损

### Window B（NEUTRAL主导，2022-2023）
- **结论：效果有限** ⚠️
- NEUTRAL占比 53.1%，但年化从 +9.64% 降至 +7.47%（-2.17pp）
- MaxDD 从 -10.42% 降至 -8.30%（+2.12pp）
- 收益下降明显，可能因为 NEUTRAL 期 position_limit 70% 限制了原本有效的仓位

### Window E（熊市+震荡，2018-01 ~ 2019-06）
- **结论：轻微有效** ✅
- 年化几乎不变（-5.76% → -5.58%，仅+0.18pp）
- MaxDD 从 -23.28% 改善到 -18.91%（+4.37pp）
- 在高BEAR期（24.1%）确实降低了回撤，但整体仍亏损

---

## 关键发现

1. **v2.5 position_limit 首次真正生效**（对比之前NEUTRAL/BEAR窗口无效的问题）
2. **BEAR期效果最显著**：Window F 的 MaxDD 改善 +8.16pp
3. **NEUTRAL期有代价**：Window B 年化下降 2.17pp，换来 MaxDD 改善 2.12pp
4. **胜率问题**：三个窗口胜率均仅 21-31%，StopLoss 驱动策略，MarketRegimeFilter 改善的是仓位管理，不是选股

---

## 对比 v2.4（上次无效版本）

> 上次回测中 Window B 和 Window E 无改善，v2.5 通过修复 `per_stock_cash = min(avail_cash/fill_count, cash * position_size * position_limit)` 公式，首次让 regime 限制真正叠加生效。

| 对比项 | v2.4 | v2.5 |
|--------|------|------|
| NEUTRAL position_limit | 未生效 | 14% 单股上限 ✅ |
| BEAR position_limit | 未生效 | 6% 单股上限 ✅ |
| Window B 年化改善 | ❌ 无改善 | ⚠️ 年化-2.17pp |
| Window E MaxDD改善 | ❌ 无改善 | ✅ +4.37pp |
| Window F MaxDD改善 | ✅ | ✅ +8.16pp |
