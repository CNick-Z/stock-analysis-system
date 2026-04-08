# 回测/模拟盘架构统一方案

> 日期：2026-03-30  
> 状态：规划中

---

## 现状问题

### 1. 策略代码重复
同一策略存在多个实现版本：

| 策略 | 文件A（主） | 文件B（重复） |
|------|------------|--------------|
| V8 | `strategies/score/v8/strategy.py`（类） | `v8_strategy.py`（函数式） |
| 波浪 | `strategies/wavechan_selector.py` | `wavechan_fast.py`、`wavechan_v3.py` |

### 2. 回测/模拟盘脚本各自内联策略逻辑
- `backtest_score_v8.py` → 内联 V8 逻辑，不走 `ScoreV8Strategy`
- `wavechan_18yr_baseline_v2.py` → 内联波浪逻辑，不走 `WaveChanSelector`
- 导致策略改动后，回测结果和模拟盘结果不一致

### 3. 性能问题
- `multi_simulator.py` 每日 `iterrows()` 构建价格映射（✅已修复）

---

## 统一架构

```
┌─────────────────────────────────────────────┐
│  统一数据层（Data Layer）                    │
│  Parquet → DuckDB → MultiSimulator          │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  统一引擎（Engine Layer）                    │
│  BasePortfolio + MultiSimulator             │
│  处理：止损/止盈/买入/卖出/市值计算           │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  统一策略接口（Strategy Interface）          │
│  ScoreV4Strategy / ScoreV8Strategy           │
│  WaveChanSelector / (Future: ComboStrategy) │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  统一入口（Entry Points）                    │
│  run_backtest.py / run_sim.py               │
│  支持：--strategy v4/v8/wavechan/combo      │
│  支持：--years 2018-2025 / --sim 2026       │
└─────────────────────────────────────────────┘
```

---

## 策略接口规范

每个策略类必须实现：

```python
class BaseStrategy(ABC):
    @abstractmethod
    def get_entry_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回满足入场条件的候选股票，带score"""
        pass

    @abstractmethod
    def should_sell(self, symbol: str, position: dict, row: pd.Series, market: dict) -> Tuple[bool, str]:
        """返回 (是否卖出, 原因)"""
        pass
```

---

## 待废弃文件清单

| 文件 | 废弃原因 | 确认人 |
|------|---------|--------|
| `v8_strategy.py` | 函数式旧版，与 ScoreV8Strategy 不同步 | 待确认 |
| `backtest_score_v8.py` | 内联V8逻辑，不走统一引擎 | 待确认 |
| `backtest_score_v7.py` | V7已废弃 | 待确认 |
| `wavechan_fast.py` | 与 WaveChanSelector 重复 | 待确认 |
| `wavechan_v3.py` | V3已废弃 | 待确认 |
| `wavechan_backtest.py` | 独立回测脚本 | 待确认 |
| `paper_trading_18years.py` | 独立18年脚本 | 待确认 |
| `paper_trading_v9.py` | V9已废弃 | 待确认 |

---

## 执行顺序

1. **T1**：V8 统一（改造 `backtest_score_v8.py`）
2. **T2**：波浪缠论统一（梳理 + 改造入口）
3. **T3**：创建统一入口脚本
4. **T4**：废弃文件清理 + 文档更新
