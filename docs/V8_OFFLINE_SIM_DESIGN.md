# V8 熊市过滤离线模拟 — 方案设计

**文档版本**: v2.0（对照 backtest.py 核实后更新）  
**创建日期**: 2026-04-02  
**创建人**: Fairy  
**状态**: 🔄 待评审

---

## 1. 整体架构

```
xlsx 交易记录                              沪深300 parquet
     ↓                                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    离线模拟引擎                                │
│                                                              │
│  1. load_trades()        加载交易 + 数据校验                  │
│  2. MarketRegimeFilter   预计算 regime（全量，非逐日）          │
│  3. replay_trades()      按时间顺序重放，维护核心状态            │
│  4. compute_metrics()    绩效计算                            │
└──────────────────────────────────────────────────────────────┘
                         ↓
              绩效对比表 + 最优参数推荐
```

**核心原则**：直接复用 `MarketRegimeFilter` 类，不重写 regime 逻辑，确保100%一致。

### 1.1 核心状态（全程维护）

| 状态变量 | 类型 | 说明 |
|----------|------|------|
| `cash` | float | 当前可用资金 |
| `positions` | dict | 持仓：{symbol: {'qty': 股数, 'cost': 总成本}} |
| `equity_curve` | list[dict] | 每日组合市值 [{date, value}] |
| `trades_log` | list[dict] | 实际执行记录 [{date, symbol, type, qty, price}] |

**所有状态在重放过程中实时更新，买入/卖出/结算都会改变状态。**

---

## 2. 数据规格

### 2.1 交易记录（xlsx）

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | str (YYYY-MM-DD) | 交易日期 |
| `symbol` | str | 股票代码 |
| `type` | str (buy/sell) | 交易类型 |
| `execution_price` | float | 实际成交价（含滑点） |
| `quantity` | int | 成交数量（股） |
| `commission` | float | 佣金 |

### 2.2 指数数据（parquet）

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | str | 日期 |
| `close` | float | 收盘价 |
| `code` | str | 指数代码（SH.000300） |

路径：`/data/warehouse/indices/CSI300.parquet`

---

## 3. regime 判断规则

### 3.1 复用 `MarketRegimeFilter`

```python
from simulator.market_regime import MarketRegimeFilter

mrf = MarketRegimeFilter(
    index_path="/data/warehouse/indices/CSI300.parquet",
    confirm_days=1,              # 可配置
    neutral_position=0.70,      # 可配置
    bear_position=0.30,          # 可配置
    bull_position=1.00,         # 固定
    regime_persist_days=3,       # 状态锁，默认3天
)

# 预计算（warmup 自动处理，只需调用一次）
mrf.prepare(start_date, end_date)
```

### 3.2 原始 regime 判定条件

| 优先级 | 条件 | 结果 | 仓位 |
|--------|------|------|------|
| 1 | RSI<40 **且** 当日跌幅>2%（快速止损） | **立即 BEAR** | bear_position |
| 2 | close < MA20 **且** RSI < 40（连续 confirm_days 天） | BEAR | bear_position |
| 3 | RSI < 50（连续 confirm_days 天） | NEUTRAL（RSI预警） | neutral_position |
| 4 | MA5<MA10<MA20 **且** MACD直方图<0 | NEUTRAL（趋势确认） | neutral_position |
| 5 | close > MA20 **且** RSI > 50 **且** MA5>MA10>MA20 | BULL | 1.0 |
| — | 都不满足 | NEUTRAL（兜底） | neutral_position |

### 3.3 状态锁（regime_persist_days）

调用 `get_regime(date)` 时：
- 切换 regime 后必须保持 `regime_persist_days` 天（默认3）才真正切换
- **避免频繁抖动**

> ⚠️ **`get_regime()` 是状态ful的**：内部维护 `_last_regime` 和 `_consecutive_regime_days`，必须**按时间顺序逐日调用**，不能并行。

---

## 4. 交易重放逻辑

### 4.1 资金初始化

```python
INITIAL_CASH = 1_000_000   # 初始资金 100万（与回测一致）
POSITION_SIZE = 0.20        # V8 固定参数：单股上限 20% 总资金
```

### 4.2 买入流程

```python
def process_buy(date, symbol, execution_price, quantity, avail_cash, positions):
    # 1. 获取当天 regime（必须先调用 mrf.get_regime(date)）
    regime_info = regime_cache[date]  # 预存结果
    limit = regime_info['position_limit']
    
    # 2. 可投资金上限 = 可用现金 × position_limit
    max_invest = avail_cash * limit
    max_qty_by_money = int(max_invest / execution_price / 100) * 100  # 按手取整
    
    # 3. V8 单股上限 = 总资金 × 20%
    max_qty_by_possize = int(INITIAL_CASH * POSITION_SIZE / execution_price / 100) * 100
    
    # 4. 实际可买量（取较小值，按手）
    actual_qty = min(quantity, max_qty_by_money, max_qty_by_possize)
    actual_qty = (actual_qty // 100) * 100  # 确保整手
    
    # 5. 记录降仓事件（用于统计）
    if actual_qty < quantity:
        reduced_count += 1  # 统计降仓次数
    
    # 6. 扣除资金
    cost = actual_qty * execution_price
    commission = max(5.0, cost * 0.0003)  # 万三佣金，最低5元
    avail_cash -= (cost + commission)
    
    # 7. 记录持仓（成本法）
    if symbol in positions:
        old = positions[symbol]
        positions[symbol] = {
            'qty': old['qty'] + actual_qty,
            'cost': old['cost'] + cost + commission
        }
    else:
        positions[symbol] = {'qty': actual_qty, 'cost': cost + commission}
    
    return avail_cash, positions
```

### 4.3 卖出流程（受持仓限制）

**关键**：因为买入量被降仓限制了，卖出时也只能卖持仓量。

```python
def process_sell(date, symbol, execution_price, quantity, avail_cash, positions):
    # 实际可卖量 = min(记录量, 当前持仓量)
    actual_sell = min(quantity, positions.get(symbol, {}).get('qty', 0))
    
    if actual_sell == 0:
        return avail_cash, positions, 0.0  # 无持仓，跳过
    
    pos = positions[symbol]
    avg_cost = pos['cost'] / pos['qty']
    cost_of_sold = avg_cost * actual_sell
    commission = max(5.0, actual_sell * execution_price * 0.0003)
    
    realized_pnl = actual_sell * execution_price - cost_of_sold - commission
    
    # 更新持仓
    pos['qty'] -= actual_sell
    pos['cost'] -= cost_of_sold
    if pos['qty'] <= 0:
        del positions[symbol]
    
    avail_cash += actual_sell * execution_price - commission
    return avail_cash, positions, realized_pnl
```

### 4.4 每日价值计算

```python
# 交易日后（收盘时）计算组合价值
def calc_portfolio_value(date, avail_cash, positions, close_prices):
    pos_value = sum(
        pos['qty'] * close_prices.get(symbol, pos['cost'] / pos['qty'])
        for symbol, pos in positions.items()
    )
    return avail_cash + pos_value
```

---

## 5. 参数网格

| 参数 | 扫描值 | 说明 |
|------|--------|------|
| `neutral_position` | [0.50, 0.60, 0.70, 0.80] | 震荡市仓位上限 |
| `bear_position` | [0.10, 0.20, 0.30, 0.40] | 熊市仓位上限 |
| `confirm_days` | [1, 2] | 连续确认天数 |
| `regime_persist_days` | [3] | 固定为3（状态锁） |

**共 4×4×2 = 32 组**，加 3 组基准 = **35 组**

---

## 6. 基准对照

| 组别 | neutral | bear | confirm | 用途 |
|------|---------|------|---------|------|
| Baseline | 1.0 | 1.0 | 2 | 无过滤基准 |
| v2.5修复前 | 1.0 | 0.40 | 2 | 当前生产配置 |
| 联合建议 | 0.70 | 0.30 | 1 | Oracle+Trader建议 |

---

## 7. 样本划分

| 部分 | 时间段 | 长度 | 用途 |
|------|--------|------|------|
| **样本内** | 2010-01-01 ~ 2019-12-31 | 10年 | 参数调优 |
| **样本外** | 2020-01-01 ~ 2025-12-31 | 6年 | 独立验证 |

---

## 8. 绩效指标

```python
def compute_metrics(equity_curve, trades, INITIAL_CASH=1_000_000):
    # 年化收益
    values = equity_curve['value'].values
    total_return = (values[-1] - INITIAL_CASH) / INITIAL_CASH
    years = (len(values)) / 252  # 按交易日
    annualized = (1 + total_return) ** (1/years) - 1
    
    # 最大回撤
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_dd = drawdown.min()
    
    # 夏普比率（无风险利率 3%）
    returns = pd.Series(values).pct_change().dropna()
    daily_rf = 0.03 / 252
    sharpe = (returns.mean() - daily_rf) / returns.std() * sqrt(252)
    
    # 总交易次数
    total_trades = len(trades)
    
    # 平均持仓天数（按买入-卖出配对估算）
    avg_hold_days = compute_avg_hold_days(trades)
    
    return {
        'annualized': f"{annualized:.2%}",
        'max_drawdown': f"{max_dd:.2%}",
        'sharpe': f"{sharpe:.2f}",
        'total_trades': total_trades,
        'avg_hold_days': f"{avg_hold_days:.1f}",
        'reduced_count': reduced_count  # 降仓次数
    }
```

---

## 9. 代码结构

```
scripts/
  v8_offline_sim.py       # 主入口（1文件，包含所有逻辑）
  │
  ├── load_trades(xlsx_path)         # 加载 + 校验
  ├── load_index(parquet_path)       # 加载指数（用于 close price）
  ├── run_simulation(params, trades, index_df)  # 单组参数重放
  ├── compute_metrics()              # 绩效计算
  └── main()                          # 参数网格 + 输出报告
```

---

## 10. 输出文件

| 文件 | 内容 |
|------|------|
| `V8_OFFLINE_SIM_RESULTS.md` | 绩效对比表 + 最优推荐 |
| `V8_OFFLINE_SIM_DETAILS.csv` | 每组参数详细指标 |
| `V8_OFFLINE_SIM_equity.csv` | 资金曲线（方便后续画图） |

---

## 11. 数据校验

```python
def validate_trades(df):
    issues = []
    
    # 1. 空值检查（signal_info 列允许空）
    required = ['date', 'symbol', 'type', 'execution_price', 'quantity']
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"{col} 有 {null_count} 个空值")
    
    # 2. 正数检查
    if (df['execution_price'] <= 0).any():
        issues.append("存在 execution_price <= 0")
    if (df['quantity'] <= 0).any():
        issues.append("存在 quantity <= 0")
    
    # 3. 日期格式
    try:
        pd.to_datetime(df['date'])
    except:
        issues.append("日期格式错误")
    
    # 4. type 只能是 buy/sell
    invalid_types = set(df['type']) - {'buy', 'sell'}
    if invalid_types:
        issues.append(f"存在非法 type: {invalid_types}")
    
    if issues:
        print("⚠️ 数据问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 数据校验通过")
    
    return df
```

---

## 12. 已知差异说明（与回测不完全一致的地方）

| 项目 | 回测 | 离线模拟 | 影响 |
|------|------|----------|------|
| 每日收盘价 | 使用实际 close | 使用交易执行价（近似） | 极小 |
| 持仓期间现金流 | 不考虑 | 不考虑（Buy & Hold） | 一致 |
| regime warmup | 预热30天 | 自动由 prepare() 处理 | 一致 |
| slippage | 成交时记录 | 已含在 execution_price | 一致 |

---

**下一步**：方案评审通过后，由 Forge 开发代码。

---

## 13. 补充说明（老板提出）

**评审日期**：2026-04-02

**老板意见**：
> 交易记录已经有买卖点了，我们只是在买卖时结合趋势减少/增加仓位比例。最大的影响是有没有可能导致本来成交的交易因为限制无法成交。

**分析**：
- ✅ 止损/止盈已经在交易记录里体现，不需要另外处理
- ✅ 卖出行为完全不变
- ❌ ~~唯一影响：买入量~~ → **错误！买少了持仓就少，卖的也少**
- ✅ 实际影响：**买和卖都被持仓量限制**，盈亏绝对值都变小
- ❌ 快速止损触发BEAR是另一套机制，与离线模拟逻辑无关，删除

**简化结论**：
```
买入时：查当天 regime → position_limit 限制最大可买量
卖出时：直接从交易记录执行，不做任何额外判断
```
