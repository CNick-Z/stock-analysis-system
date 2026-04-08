# 策略目录索引

> 最后更新：2026-03-30

## 📁 目录结构

```
strategies/                    # 策略主目录（新版）
├── score/                    # Score 策略族 ⭐
│   ├── v1_baseline/         # 原始版本
│   ├── v4/                  # ⭐ 主力候选（18年+148.84%）
│   ├── v8/                  # ⭐ 主力（18年+80%）
│   └── archive/             # 废弃版本归档
├── wavechan/                # Wavechan 策略族
│   ├── v1_baseline/
│   ├── v3_l2_cache/
│   └── archive/
├── combo/                   # 组合策略
│   ├── score_wavechan_a/
│   └── archive/
└── core/                    # 共享核心模块

versions/                      # 历史版本存档（原始文件）
├── score_strategy_v1_baseline.py
├── score_strategy_v2_rsi.py
├── score_strategy_v3_trailing.py
├── score_strategy_v4_notrail.py   ← 18年+148.84%
├── score_strategy_v5_rsi_price.py
└── score_strategy_v6_lowprice.py
```

## 📊 策略状态总览

| 策略 | 版本 | 18年回测 | 最大回撤 | 状态 | 负责人 |
|------|------|---------|---------|------|-------|
| Score V4 | notrail | **+148.84%** | -32.81% | ⭐ 待确认上线 | Oracle |
| Score V8 | IC增强 | **+80%** | 待查 | ⭐ 主力候选 | Oracle/Forge |
| Score V4(内联) | paper内联 | ❌-50% | - | ⚠️ 错误实现 | 待修复 |
| Wavechan V3 | L2缓存 | 待查 | 待查 | 研究中 | Oracle |

## 🎯 生命周期阶段说明

```
Stage 1 研究 → Stage 2 回测 → Stage 3 模拟盘 → Stage 4 实盘 → Stage 5 监控
    ↓            ↓              ↓              ↓
  理论依据     18年>50%       运行2个月       仓位10%
               回撤<40%
```

## ⚠️ 重要澄清：V4 两个版本

| 版本 | 文件 | 18年回测 | 说明 |
|------|------|---------|------|
| **正确版本** | `versions/score_strategy_v4_notrail.py` | **+148.84%** | 独立策略文件，已迁移到 strategies/score/v4/ |
| **错误版本** | `paper_trading_18years.py` 的内联 score_v4() | **-50%** | 内联实现，逻辑与正确版本不同 |

**模拟盘需使用正确版本的 V4！**

## 📄 各策略文档

### Score V4（⭐ 待上线）
- 策略说明：[score/v4/STRATEGY.md](score/v4/STRATEGY.md)
- 原始代码：`versions/score_strategy_v4_notrail.py`

### Score V8（⭐ 主力候选）
- 策略说明：[score/v8/STRATEGY.md](score/v8/STRATEGY.md)
- 策略代码：`strategies/score/v8/strategy.py`

## 🔧 快速索引

| 用途 | 文件位置 |
|------|---------|
| 主回测引擎 | `backtester.py` |
| 模拟盘脚本 | `paper_trading_sim.py` |
| 18年回测 | `paper_trading_18years.py` |
| V8策略代码 | `strategies/score/v8/strategy.py` |
| 历史版本存档 | `versions/` |
| 数据仓库 | `/data/warehouse/` |

## 2026-04-02 V8 MarketRegimeFilter 参数更新

### 默认值变更
- `neutral_position`: 0.60 → **0.70**
- `bear_position`: 0.20 → **0.30**
- `confirm_days`: 2 → **1**

### 验证依据
离线模拟（v8_offline_sim.py）35组参数回测：
- 最优：neutral=0.70, bear=0.10, confirm=1 → 年化3.13%, MaxDD=-5.33%, 夏普+0.04
- 建议配置：neutral=0.70, bear=0.30, confirm=1 → 年化1.99%, MaxDD=-7.48%（稳健）

### 文件变更
- simulator/market_regime.py（默认值）
- backtest.py（新增 --filter-neutral --filter-bear 参数）
