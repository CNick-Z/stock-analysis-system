# backtest.py & simulate.py 需求分析与设计文档
> 日期：2026-03-31 | 状态：📝 评审中

---

## 一、背景与目标

### 1.1 现状
- `simulator/base_framework.py` 已完成（约500行）
- V8 策略回测已通过飞书验证
- 旧回测/模拟盘文件散落（v8_simulated_trading.py / v3_simulated_trading.py / backtest_score_v8.py 等）

### 1.2 目标
- 实现统一的回测入口：`backtest.py`
- 实现统一的模拟盘入口：`simulate.py`
- 两个入口共用 `BaseFramework`，策略插拔切换
- 归档旧文件（备份到 `archive/`）

---

## 二、backtest.py 设计

### 2.1 功能
- **回测**：给定时间范围，逐日运行框架，输出完整回测报告
- **策略选择**：支持 `--strategy v8|wavechan|all`
- **时间范围**：`--start YYYY-MM-DD --end YYYY-MM-DD`

### 2.2 命令行接口
```bash
python3 backtest.py --strategy v8 --start 2018-01-01 --end 2025-12-31
python3 backtest.py --strategy wavechan --start 2024-01-01 --end 2026-03-31
python3 backtest.py --strategy all --start 2018-01-01 --end 2025-12-31
```

### 2.3 输出
- 控制台打印回测报告（最终资金、年化收益、最大回撤、夏普比率、交易统计）
- `--output-state FILE` 可选：保存最终状态 JSON

### 2.4 策略加载
| 策略名 | 策略类 | 数据源 |
|--------|--------|--------|
| `v8` | `V8ScoreStrategy` | Parquet + money_flow |
| `wavechan` | `WaveChanStrategy` | Parquet + wavechan L2缓存 |
| `all` | 两个都跑，横向对比 | 同上 |

### 2.5 关键实现
- 使用 `BaseFramework.run_backtest()` 作为核心
- 策略实例化在 `main()` 中完成，通过 `--strategy` 选择
- 汇总报告支持打印两个策略的横向对比

---

## 三、simulate.py 设计

### 3.1 功能
- **模拟盘**：每日运行，处理当日出场+入场，更新持仓状态
- **状态持久化**：中断后可 `--load-state` 恢复
- **状态查看**：`--show-state` 查看当前持仓和资金

### 3.2 命令行接口
```bash
# 每日运行（自动读取上次状态继续）
python3 simulate.py --strategy v8

# 指定日期运行
python3 simulate.py --strategy v8 --date 2026-03-31

# 重置模拟盘
python3 simulate.py --strategy v8 --reset

# 查看状态
python3 simulate.py --strategy v8 --show-state

# 查看交易记录
python3 simulate.py --strategy v8 --show-trades
```

### 3.3 状态文件
- 位置：`/tmp/simulate_{strategy}.json`（可通过 `--state-file` 自定义）
- 包含：cash / positions / trades / n_winning / n_total

### 3.4 通知
- 有交易时打印交易摘要（持仓变化）
- 收盘后打印持仓快照（持仓股、盈亏）

---

## 四、策略加载设计

### 4.1 策略注册表
```python
STRATEGY_REGISTRY = {
    "v8": {
        "class": "V8ScoreStrategy",
        "path": "strategies.score.v8.strategy",
    },
    "wavechan": {
        "class": "WaveChanStrategy",
        "path": "strategies.wavechan.v3_l2_cache.wavechan_strategy",
    },
}
```

### 4.2 动态加载
```python
def load_strategy(name: str) -> Strategy:
    cfg = STRATEGY_REGISTRY[name]
    module = __import__(cfg["path"], fromlist=[cfg["class"]])
    return getattr(module, cfg["class"])()
```

---

## 五、旧文件归档

### 5.1 归档清单
| 文件 | 目标位置 |
|------|----------|
| `v8_simulated_trading.py` | `archive/v8_simulated_trading.py.bak` |
| `v3_simulated_trading.py` | `archive/v3_simulated_trading.py.bak` |
| `backtest_score_v8.py` | `archive/backtest_score_v8.py.bak` |
| `backtest_score_v8b.py` | `archive/backtest_score_v8b.py.bak` |
| `backtest_score_v8_allyears.py` | `archive/backtest_score_v8_allyears.py.bak` |
| `backtest_score_v7.py` | `archive/backtest_score_v7.py.bak` |
| `backtester.py` | `archive/backtester.py.bak` |
| `backtest_bias_corrector.py` | `archive/backtest_bias_corrector.py.bak` |
| `wavechan_backtest.py` | `archive/wavechan_backtest.py.bak` |
| `wavechan_v3_backtest.py` | `archive/wavechan_v3_backtest.py.bak` |
| `wavechan_fast_backtest.py` | `archive/wavechan_fast_backtest.py.bak` |
| `wavechan_2018_backtest.py` | `archive/wavechan_2018_backtest.py.bak` |
| `wavechan_2026_backtest.py` | `archive/wavechan_2026_backtest.py.bak` |

### 5.2 归档脚本
```bash
mkdir -p archive
mv v8_simulated_trading.py v3_simulated_trading.py ... archive/
```

---

## 六、验收标准（2026-03-31 ✅ 全部通过）

1. ✅ `python3 backtest.py --strategy v8 --start 2024-01-01 --end 2025-12-31` 正常运行
2. ✅ `python3 simulate.py --strategy v8 --show-state` 正常运行
3. ✅ `python3 simulate.py --strategy v8 --reset` 能重置状态
4. ✅ 旧文件已归档到 `archive/`（13个文件）
5. ✅ 策略切换（`--strategy v8` / `--strategy wavechan`）正常工作

### 额外完成
- ✅ `simulator/shared.py` 公共模块提取（backtest.py/simulate.py 共用）
- ✅ WaveChan `signal_status == 'confirmed'` 过滤（修复 look-ahead bias）
- ✅ 日期格式统一（merge 前转 str）
- ✅ exec_price 逻辑正确（回测 next_open，模拟盘 fallback）

---

## 七、文件结构（完成后）

```
projects/stock-analysis-system/
├── backtest.py              ← 统一回测入口
├── simulate.py              ← 统一模拟盘入口
├── archive/                 ← 旧文件归档（13个）
│   ├── v8_simulated_trading.py.bak
│   ├── v3_simulated_trading.py.bak
│   └── ...
└── simulator/
    ├── base_framework.py    ← 框架核心
    └── shared.py            ← 公共模块（新增）
```

