# V8 出场逻辑优化方案

> 创建：2026-03-30 22:35
> 状态：待执行

---

## 一、问题根因

### 当前 V8.2 出场逻辑（2024-2025年表现）

| 出场方式 | 触发次数 | 说明 |
|---------|---------|------|
| MA死叉 | 39次 | 太多假信号 |
| 止损 | 14次 | 正常 |
| 止盈 | 2次 | 止盈15%很少触发 |
| 趋势破坏 | 0次 | 未触发 |

**问题：MA死叉太敏感，趋势还没真破就触发了，导致无法让利润奔跑。**

---

## 二、参考版本：V4 出场逻辑

**文件**：`versions/score_strategy_v4_notrail.py`
**收益**：+148.84%（18年）

### V4 出场条件（双重确认）：
```python
# 卖出条件：价格跌破MA20 **且** 资金流向为负
sell_condition = (
    (signals['close'] < signals['ma_20']) &
    (signals['money_flow_trend'] == False)
)
```

### V4 的核心思路：
- 不是"MA死叉就卖"，而是"价格真跌破MA20且资金流确认流出才卖"
- 双重确认大幅减少假信号
- 趋势不破就一直持有，让利润奔跑

---

## 三、优化方案：迁移 V4 出场逻辑到 V8

### 当前 V8 出场代码（strategy.py）：
```python
# 止损
if next_open < pos['avg_cost'] * (1 - self.stop_loss):
    return True, "止损"

# 止盈
if next_open > pos['avg_cost'] * (1 + self.take_profit):
    return True, "止盈"

# MA死叉（太敏感）
if row['sma_20'] > row['sma_55']:
    return True, "MA死叉"
```

### 优化后 V8 出场代码：
```python
# 止损（保留）
if next_open < pos['avg_cost'] * (1 - self.stop_loss):
    return True, "止损"

# 止盈（改为移动止盈）
# 持仓期间记录最高价，涨超15%后用 trailing stop
if self.trailing_stop and current_price > pos['entry_high'] * 1.15:
    # 从最高点回撤8%则止盈
    if current_price < pos['entry_high'] * 0.92:
        return True, "移动止盈"

# 趋势出场（迁移V4逻辑）
# 条件：价格跌破MA20 **且** 资金流向为负
if row['close'] < row['sma_20'] and row.get('money_flow_trend') == False:
    return True, "跌破MA20+资金流出"
```

---

## 四、数据字段确认

### money_flow_trend 字段检查

| 字段名 | 是否存在于当前数据 | 说明 |
|--------|-----------------|------|
| money_flow_trend | ❌ 不存在 | V4旧版数据特有 |
| money_flow_positive | ❌ 不存在 | V8原始回测里有但数据中没有 |

**需要替代方案**：可以用其他趋势确认指标替代 money_flow_trend：
- 方案A：用 MA5 < MA10 的"短期均线死叉"替代资金流
- 方案B：用成交量萎缩（vol_ratio < 0.8）替代资金流
- 方案C：简化版——只要价格跌破MA20就卖（不加资金流确认）

---

## 五、风险提示

1. **money_flow_trend 字段不存在**，需要设计替代逻辑
2. **改动出场逻辑需要重新回测验证**
3. **改动需要 Oracle 评审**
4. **需要 commit 记录版本**

---

## 六、执行计划

| 步骤 | 内容 | 负责 |
|------|------|------|
| 1 | 确认 money_flow_trend 替代方案 | Fairy |
| 2 | 改动 strategy.py 的 should_sell | Forge |
| 3 | 回测 2024-2025 验证效果 | — |
| 4 | Oracle 评审 | Oracle |
| 5 | commit 改动 | Forge |

---

## 七、版本记录

| 版本 | 改动 | 状态 |
|------|------|------|
| V8.2 | RSI/WR IC调优，当前版本 | ✅ 已冻结 |
| V8.3 | 出场逻辑优化 | ⬜ 待执行 |


---

## 八、money_flow_trend 替代方案（数据中没有该字段）

当前数据只有这些均线字段：`sma_5, sma_10, sma_20, sma_55, sma_240`

| 替代方案 | 逻辑 | 优缺点 |
|---------|------|--------|
| **方案A（推荐）** | 价格跌破MA20 **且** MA5<MA10（短期均线死叉）| 双重确认，逻辑清晰 |
| 方案B | 价格跌破MA20 **且** vol_ratio<0.8（缩量确认）| 用缩量替代资金流 |
| 方案C | 只用价格跌破MA20（不加确认）| 最简单，但可能有假信号 |

### 推荐方案A的代码：
```python
# 趋势出场（V4逻辑迁移，价格跌破MA20+短期均线死叉）
if row['close'] < row['sma_20'] and row['sma_5'] < row['sma_10']:
    return True, "跌破MA20+均线死叉"
```


---

## 九、公共数据接口设计（fundamentals provider）

### 现有数据
| 数据源 | 内容 | 路径 |
|--------|------|------|
| financial_summary.parquet | PE/PB/ROE/营收增长/净利润增长 | `/data/warehouse/` |
| stock_basic_info.parquet | 行业/上市日期/市值 | `/data/warehouse/` |
| stock_daily.db | 日线行情+技术指标 | `/data/warehouse/` |

### 设计目标
```python
# 统一的接口，回测和模拟盘都能用
fundamentals = FundamentalProvider()
fund = fundamentals.get(symbol, date)  # 获取某股票某日的基本面
# 返回: {'pe': 15.2, 'pb': 1.3, 'roe': 12.5, 'revenue_growth': 5.2, ...}
```

### 效率优化
- **预加载 + 缓存**：启动时一次性加载financial_summary，按 symbol+date 建索引
- **按需查询**：只查持仓股票的基本面，不全量扫描
- **内存映射**：避免重复读取大文件

### 实现方案
```python
class FundamentalProvider:
    def __init__(self, data_dir='/data/warehouse'):
        self.df = pd.read_parquet(f'{data_dir}/financial_summary.parquet')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.index = self.df.set_index(['symbol', 'date'])
    
    def get(self, symbol: str, date: str) -> dict:
        """获取某股票在某日期的最新基本面（不早于date）"""
        dt = pd.to_datetime(date)
        subset = self.index.loc[symbol]
        valid = subset[subset.index <= dt]  # 取最新一期
        if valid.empty:
            return {}
        row = valid.iloc[-1]
        return row.to_dict()
```


## 最终方案（commit ff03232）
- 资金流向: money_flow_trend == False（data_loader已提供）
- 不再使用vol_ratio代理
