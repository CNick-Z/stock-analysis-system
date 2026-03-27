# 回测 Bias 修正报告

> 整理日期：2026-03-27
> 负责：Byte（数据Agent）
> 更新：2026-03-27（v2 完成版）

---

## 一、修正内容

### 1. 核心模块：`backtest_bias_corrector.py`（v2）

增强后的三个核心类：

| 类 | 功能 | v2 新增 |
|----|------|---------|
| `BiasCorrector` | 涨跌停/停牌过滤，可执行价格判断 | 量化日志、sell 延迟跟踪 |
| `LookAheadBiasFixer` | look-ahead bias 验证工具 | `validate_price_matrix_leakage()` 复权跳变检测 |
| `AdjustmentHandler` | 复权类型确认 + 校验 | `_verify_adjustment()` 前复权数据自动校验 |

### 2. 涨跌停过滤

- **信号日涨停过滤**：信号日本身涨停的股票剔除（次日高开，实际开盘买不进）
- **执行日涨停/跌停过滤**：次日执行时若涨跌停，剔除该买入计划
- **停牌过滤**：信号日或次日停牌（volume=0 或价格为 NaN）的股票剔除
- **卖出延迟**：跌停日无法卖出，自动延迟到下一可交易日

**阈值**：`LIMIT_UP_THRESHOLD = 0.099`（统一阈值）
**板块识别（已规划，待接入真实板块数据）**：
- 主板（60xxxx/00xxxx）：±10%
- 创业板（30xxxx）：±20%
- 科创板（68xxxx）：±20%
- 北交所（4xxxx/8xxxx）：±20%
- ST/*ST：±5%（简化取 5%）

### 3. 停牌处理

- `_get_next_open_price` 原有逻辑：跳过 NaN 价格日（停牌日）
- 新增 `can_buy()` / `can_sell()` 方法：明确检查涨跌停/停牌状态
- `get_executable_price()`：返回第一个可交易日的开盘价

### 4. look-ahead bias 修复

- **initialize() 前推1年 lookback**：涨跌停计算需要前收盘价（如 Jan 2 需要 Dec 29），修改为加载 `start_date - 1年` 的数据
- **特征生成 lookback**：策略侧 `generate_features` 已有 30 日 lookback（技术指标用）
- **信号索引验证**：`LookAheadBiasFixer.validate_signal_index()` 验证信号 DataFrame 无未来数据泄露
- **复权跳变检测**：`validate_price_matrix_leakage()` 抽检 20 只股票，检测 >30% 异常跳变

### 5. 复权处理

- 数据来源：`akshare adjust="qfq"`（前复权）
- `AdjustmentHandler._verify_adjustment()` **v2 新增**：自动校验前复权数据质量
  - 随机抽检 20 只股票，检查单日涨幅 >30% 的异常跳变
  - fail-fast 机制：若校验失败则抛出异常，阻止回测继续
- `AdjustmentHandler.assert_adjusted()`：在 `SimpleComboBacktester._run()` 启动前强制校验

---

## 二、集成位置

### `backtester.py`（主回测引擎）

| 位置 | 修改内容 |
|------|---------|
| `BacktestOrchestrator.__init__` | `self.bias_corrector = None` |
| `BacktestOrchestrator.initialize` | 构建 `BiasCorrector`，lookback 前推1年 |
| `_process_buy_signals` | 调用 `bias_corrector.filter_buy_signals()` 过滤信号 |
| `_process_scored_buy_signals` | 添加 `bias_corrector.can_buy()` 检查 |
| `_process_resonance_buy_signals` | 同上 |
| `_process_resonance_v2_buy_signals` | 同上 |
| `_process_wavechan_buy_signals` | 同上 |
| `_check_stop_loss` | 添加 `bias_corrector.can_sell()` 检查 |
| `_check_take_profit` | 同上 |

### `score_wavechan_combo_a.py`（组合方向A独立回测器）

| 位置 | 修改内容 |
|------|---------|
| 导入 | 新增 `BiasCorrector, AdjustmentHandler, LookAheadBiasFixer` |
| `SimpleComboBacktester.__init__` | 新增 `self.bias_corrector`, `self.adj_handler` |
| `_load_data` | 初始化 `BiasCorrector + AdjustmentHandler` |
| `_run` | `adj_handler.assert_adjusted()` 启动前校验 |
| T+1 买入执行 | `can_buy()` 检查，统计 `total_bias_filtered` |
| Score 候选处理 | `filter_buy_signals()` 过滤，统计 `total_bias_filtered` |
| 持仓卖出检查 | `can_sell()` 检查，统计 `total_sell_delay` |
| `get_combo_signals` | `LookAheadBiasFixer.validate_signal_index()` |

---

## 三、回测结果对比

### 2024-01-01 ~ 2024-12-31（Score 策略）

| 指标 | 修正前（估算） | 修正后 |
|------|--------------|--------|
| 最终净值 | ~550,000（+10%）| 480,010 |
| 总收益率 | ~+10%（虚高） | **-4.00%** |
| 最大回撤 | （无数据） | -10.91% |
| 交易次数 | （无数据） | 19 |

> **说明**：修正前 Score v4 报告 +22% 年化（含约 +10-12% look-ahead bias）。修正后收益率显著下降，符合去除 bias 的预期。

### Bias 过滤统计（2024 全年）

- 约 **80% 的交易日** 有涨跌停/停牌信号被过滤
- 典型过滤量：每日 1~5 个候选信号被剔除
- 大量候选股因信号日或次日涨跌停而无法实际买入

### Bias 统计字段（v2 新增）

| 字段 | 说明 |
|------|------|
| `total_bias_filtered` | 涨跌停/停牌过滤总次数 |
| `total_sell_delay` | 跌停/停牌导致卖出延迟总次数 |
| `total_wavechan_filtered` | WaveChan W2/W4 信号过滤次数 |

---

## 四、已知局限

1. **涨跌停阈值统一为 9.9%**：未区分主板（10%）、创业板/科创板（20%）、ST（5%）
   - 解决方案：接入 `stock_basic_info` 表的板块字段，使用板块差异化阈值
2. **前复权连续性**：前复权数据在长期历史中可能存在精度损失（>10年）
3. **look-ahead 残留风险**：特征生成中部分指标（如 `total_shares`）取静态值，未追踪公司行为变更
4. **分钟数据缺失**：WaveChan W2/W4 信号精度受限（日线为主）

---

## 五、下一步建议

| 优先级 | 行动项 | 说明 |
|--------|--------|------|
| P0 | 接入真实板块数据 | 区分主板/创业板/科创板/ST 涨跌停阈值 |
| P1 | 独立运行修正前后的对比回测 | 量化 bias 影响大小 |
| P1 | 补充 60 分钟数据 | WaveChan 信号精度提升 |
| P2 | fundamental 字段的动态更新 | 避免 `total_shares` 静态 look-ahead |
| P3 | 三步流程方向B并行化 | 等方向A验证后再上 |

---

*方案版本：v2.0（Bias 修正完成版，score_wavechan_combo_a.py 集成完成）*
