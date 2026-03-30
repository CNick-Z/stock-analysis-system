# Score V4 — 无跟踪止损版（版本存档）

> 状态：⚠️ 需确认模拟盘对应版本

## 版本信息
- 版本号：v4 notrail
- 代码文件：`versions/score_strategy_v4_notrail.py`（419行）
- 已迁移至：`strategies/score/v4/strategy.py`
- 回测日期：2026-03-30

## ⚠️ 重要说明
此版本（v4 notrail）历史回测表现优秀，但需注意：
- 18年回测（2008-2025）：**+148.84%**
- 最大回撤：-32.81%
- ⚠️ paper_trading_18years.py 中的 v4 实现不同，结果为 -50%

## 核心逻辑
移除死叉过滤的原始评分体系。

## 回测表现

### 18年完整回测（2008-2025）
| 指标 | 数值 |
|------|------|
| 总收益 | **+148.84%** |
| 最大回撤 | -32.81% |
| 夏普比率 | 待查 |
| 交易次数 | 待查 |
| 胜率 | 待查 |

## 澄清：两个v4的区别

| 版本 | 来源 | 18年回测 |
|------|------|---------|
| **versions/score_strategy_v4_notrail.py** | 独立策略文件 | **+148.84%** ✅ |
| paper_trading_18years.py 的 score_v4() | 内联实现 | **-50%** ❌ |

## 文件索引
- 策略代码：`strategy.py`（已从 versions/ 迁移）
- 原始文件：`versions/score_strategy_v4_notrail.py`
- 回测脚本：`backtest.py`
- 模拟盘：`simulation.py`

## 当前状态
- [x] 历史版本归档完成（2026-03-30）
- [ ] 需确认 paper_trading 用的是哪个 v4
- [ ] 需将正确的 v4 对接到模拟盘
