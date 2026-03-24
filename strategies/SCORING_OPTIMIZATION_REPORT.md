# 评分系统优化深度研究报告

**研究员**: Oracle  
**日期**: 2026-03-24  
**baseline 回测**: +25.83%

---

## 一、排序方向确认（nsmallest vs nlargest）

### 现状代码
```python
top = daily.nsmallest(self.config['top_n'], 'total_score')  # 低分优先
```

### 问题诊断：发现严重矛盾

**评分公式**:
```
total = tech_score(正) * 0.4483 + flow_score(可正可负) * 0.2947 + market_heat(正) * 0.257
```

**各组件方向**:
| 组件 | 贡献方向 | 说明 |
|------|---------|------|
| tech_score | 始终 ≥ 0 | 满足条件越多越高 |
| market_heat | 始终 ≥ 0 | 成交量比越高越高 |
| flow_score | 可正可负 | `positive_flow: -0.0072` 竟是**负权重**！ |

**矛盾点**:
- 代码注释写"评分越低越好（0.8-1.0 是最佳区间）"
- 但实际评分结构：条件越多、成交量越大 → total_score **越高**
- 用 `nsmallest` 选低分 = **选中了条件差、成交量低、资金流向不明确的股票**

### 结论
**`nsmallest` 大概率是错误的**。正确做法应是 `nlargest`（高分优先）。

### 量化建议
```
将 nsmallest 改为 nlargest，预计提升 5-10% 收益率
```

---

## 二、权重分析

### 当前权重
```
technical:    44.83%
capital_flow: 29.47%
market_heat:  25.70%
```

### 问题 1: 资金流权重过低
- 资金流是**领先指标**，价格往往跟随资金
- 当前 29.47% 低于技术面的 44.83%
- 建议调整为：**资金流 40%，技术面 35%，市场热度 25%**

### 问题 2: 市场热度 25.7% 可能过高
- 市场热度只衡量"相对板块的成交量比"
- 在牛市中，所有股票都会被"热度"拉高，失去区分度
- 建议：加入时间衰减，或设置热度上限（如 3.0 倍）

### 问题 3: flow_score 中存在负权重 bug
```python
cap_weights = {
    'positive_flow': -0.0072,  # ← 资金流入竟然是减分？BUG！
    ...
}
```
资金流入反而扣分，这是明显错误。

### 量化建议
| 调整项 | 当前值 | 建议值 | 理由 |
|--------|--------|--------|------|
| capital_flow 权重 | 0.2947 | 0.35-0.40 | 资金流是领先指标 |
| technical 权重 | 0.4483 | 0.35 | 降低技术面主导 |
| market_heat 权重 | 0.257 | 0.20-0.25 | 防止牛市失真 |
| positive_flow 符号 | **-0.0072** | **+0.0072** | 资金流入应加分 |

---

## 三、阈值分析

### 当前阈值
```python
thresholds = {
    'growth_min': 0.5,   # 日涨幅 ≥ 0.5%
    'growth_max': 6.0,   # 日涨幅 ≤ 6.0%
    'angle_min': 30,     # 10日线角度 ≥ 30度
}
```

### growth_min = 0.5% 太宽松
- 0.5% 的涨幅几乎等同于"不跌"
- 建议提高到 **1.0-1.5%**（原版 EnhancedTDXStrategy 用的就是 1.0%）

### growth_max = 6.0% 合理
- 排除涨停（6%）和过暴涨股票
- 合理，但可以考虑缩小到 **5.5%**

### angle_min = 30 合理
- 均线角度衡量趋势强度
- 30 度是合理的启动阈值

### 补充缺失的阈值
```python
thresholds = {
    'growth_min': 1.0,      # 上调
    'growth_max': 5.5,      # 缩小
    'angle_min': 30,        # 保持
    'volume_min_ratio': 1.3,  # 新增：最小成交量倍数
    'rsi_overbought': 70,      # 新增：RSI 超买过滤
    'kdj_overbought': 85,      # 新增：KDJ 超买过滤
}
```

---

## 四、波浪缠论结合方案（重点）

### 已有 wave_chan_weekly_daily.py 分析

该策略包含三个核心类：
1. `WeeklyTrendAnalyzer` — 周线趋势判断（定方向）
2. `DailyEntryFinder` — 日线入场点（找买点）
3. `WaveChanWeeklyDailyStrategy` — 整合器

### 周线判断逻辑
```
趋势判断：MA10 > MA20 → 上升趋势
波浪阶段：浪1/浪3 → 可买入；浪5 → 不买入；浪2/浪4 → 观望
```

### 日线入场逻辑
```
一买：底分型 + 背驰 + 下跌末端
二买：底分型 + 回抽不创新低 + 中枢下沿
浪5卖点：顶分型 + 背驰
```

### 结合方案 1: 评分前置过滤器（推荐）

```python
def _filter_by_wave_chan(self, signals: pd.DataFrame, 
                          weekly_results: Dict) -> pd.DataFrame:
    """
    用波浪缠论过滤评分候选股票
    只在周线趋势向上时（浪1/浪3）才选股
    """
    if signals.empty:
        return signals
    
    filtered = []
    for symbol in signals['symbol'].unique():
        stock_signals = signals[signals['symbol'] == symbol]
        weekly = weekly_results.get(symbol, {})
        
        # 周线判断
        if not weekly.get('can_trade', False):
            # 周线趋势不明或浪5/下降趋势 → 跳过
            continue
        
        # 只在浪1/浪3操作
        wave_stage = weekly.get('wave_stage', 'unknown')
        if wave_stage not in ['浪 1', '浪 3']:
            continue
        
        filtered.append(stock_signals)
    
    if not filtered:
        return pd.DataFrame()
    
    return pd.concat(filtered)
```

### 结合方案 2: 评分增强（加入波浪因子）

```python
def _score_stocks_v2(self, signals: pd.DataFrame, 
                      weekly_results: Dict) -> pd.DataFrame:
    """增强版评分：加入波浪缠论因子"""
    
    # 在原有评分基础上，乘以波浪置信度
    for idx, row in signals.iterrows():
        symbol = row['symbol']
        weekly = weekly_results.get(symbol, {})
        
        wave_confidence = weekly.get('confidence', 0.5)
        wave_stage = weekly.get('wave_stage', 'unknown')
        
        # 浪3加权（更高确定性）
        wave_boost = 1.0
        if wave_stage == '浪 3':
            wave_boost = 1.2  # 浪3确定性更高
        elif wave_stage == '浪 1':
            wave_boost = 1.1  # 浪1次之
        
        # 应用波浪加成
        original_total = row['total_score']
        enhanced_total = original_total * wave_confidence * wave_boost
        
        signals.loc[idx, 'total_score'] = enhanced_total
        signals.loc[idx, 'wave_stage'] = wave_stage
        signals.loc[idx, 'wave_confidence'] = wave_confidence
    
    return signals
```

### 结合方案 3: 避免在下降浪中买入

```python
# 在 buy_condition 中加入
def get_signals(self, start_date: str, end_date: str):
    signals = self.generate_features(start_date, end_date)
    
    # 获取周线数据并分析
    weekly_data = self._get_weekly_data(start_date, end_date)
    weekly_analyzer = WeeklyTrendAnalyzer()
    weekly_results = {}
    for symbol in signals['symbol'].unique():
        sym_weekly = weekly_data.get(symbol, [])
        if len(sym_weekly) >= 20:
            result = weekly_analyzer.analyze(sym_weekly)
            weekly_results[symbol] = result
    
    # 【新增】波浪过滤：排除下降浪
    buy_condition = (
        signals['growth_condition'] &
        signals['ma_condition'] &
        signals['angle_condition'] &
        signals['volume_condition'] &
        signals['macd_condition'] &
        (signals['jc_condition'] | signals['macd_jc']) &
        (signals['ma_20'] < signals['ma_55']) &
        (signals['ma_55'] > signals['ma_240']) &
        signals['symbol'].apply(lambda s: weekly_results.get(s, {}).get('can_trade', False))  # 周线允许交易
    )
    
    buy_signals = signals[buy_condition].copy()
    buy_signals['signal_type'] = 'buy'
    
    # 评分时加入波浪因子
    buy_signals = self._score_stocks_v2(buy_signals, weekly_results)
    
    # ... 其余逻辑
```

---

## 五、止盈止损优化

### 当前配置
```python
trailing_config = {
    (0.0, 0.05): {'type': 'breakeven', 'pct': 0.003},   # 保本
    (0.05, 0.10): {'type': 'trailing', 'pct': 0.50},   # 50% 利润回吐
    (0.10, 0.20): {'type': 'trailing', 'pct': 0.30},   # 30% 利润回吐
    (0.20, 999.0): {'type': 'trailing', 'pct': 0.10},  # 10% 利润回吐
}
```

### 问题
- 固定百分比止盈无法适应不同股票的波动特性
- 没有结合波浪理论判断止盈时机

### 结合波浪理论的止盈方案

```python
def calculate_wave_based_take_profit(entry_price: float, 
                                      wave_stage: str,
                                      current_price: float) -> Dict:
    """
    结合波浪阶段计算动态止盈
    """
    profit_pct = (current_price - entry_price) / entry_price
    
    if wave_stage == '浪 3':
        # 浪3最肥美，不轻易止盈
        if profit_pct < 0.20:
            trailing_pct = 0.10  # 保留 90% 利润
        elif profit_pct < 0.40:
            trailing_pct = 0.20  # 保留 80% 利润
        else:
            trailing_pct = 0.30  # 保留 70% 利润
        stop_loss = entry_price * 1.05  # 浪3：移动止损到成本+5%
    
    elif wave_stage == '浪 5':
        # 浪5末端：分批止盈
        trailing_pct = 0.40  # 保留 60% 利润即可
        stop_loss = current_price * 0.90  # 快速锁定利润
    
    elif wave_stage in ['浪 1', '浪 2', '浪 4']:
        # 浪1/2/4：正常止盈
        trailing_pct = 0.30
        stop_loss = entry_price * 1.03
    
    else:
        # 未知阶段
        trailing_pct = 0.25
        stop_loss = entry_price * 1.02
    
    return {
        'trailing_pct': trailing_pct,
        'stop_loss': stop_loss,
        'wave_stage': wave_stage
    }
```

### 浪5末端的止盈策略

根据 wave_chan_weekly_daily.py 的逻辑：

```python
# 在持仓管理中增加：
if current_wave_stage == '浪 5':
    if weekly_divergence:
        # 背驰 → 清仓
        action = 'SELL_ALL'
        reason = '周线浪5背驰'
    else:
        # 无背驰 → 减半仓
        action = 'SELL_HALF'
        reason = '周线浪5无背驰，减半'
```

---

## 六、综合优化建议汇总

### 高优先级（立即修复）

| # | 问题 | 修复方案 | 预期收益变化 |
|---|------|---------|-------------|
| 1 | `nsmallest` 方向错误 | 改为 `nlargest` | +5~10% |
| 2 | `positive_flow: -0.0072` 符号错误 | 改为 `+0.0072` | +2~5% |
| 3 | growth_min 太低 (0.5%) | 提高到 1.0% | +1~3% |

### 中优先级（下次迭代）

| # | 优化项 | 方案 | 预期收益变化 |
|---|--------|------|-------------|
| 4 | 权重分配 | 资金流 40%，技术 35%，热度 20% | +2~5% |
| 5 | 波浪前置过滤 | 只在浪1/3操作 | +3~8% |
| 6 | 波浪止盈 | 结合浪3/浪5动态止盈 | +2~5% |

### 低优先级（可选优化）

| # | 优化项 | 方案 |
|---|--------|------|
| 7 | 热度上限 | 设置 max(market_heat, 3.0) |
| 8 | RSI/KDJ 超买过滤 | 超过阈值时降低权重 |

---

## 七、波浪缠论结合代码框架

```python
# strategies/score_strategy.py 新增

from .wave_chan_weekly_daily import (
    WaveChanWeeklyDailyStrategy,
    WeeklyTrendAnalyzer,
    daily_to_weekly
)

class ScoreStrategyV2(ScoreStrategy):
    """增强版评分策略：融合波浪缠论"""
    
    def __init__(self, db_path: str = None, config: dict = None):
        super().__init__(db_path, config)
        self.wave_analyzer = WeeklyTrendAnalyzer()
        self.entry_finder = DailyEntryFinder()
        self.wave_strategy = WaveChanWeeklyDailyStrategy()
    
    def _get_weekly_data_for_symbol(self, symbol: str, 
                                     end_date: str) -> List[Dict]:
        """获取单只股票的周线数据"""
        # 从日线数据聚合周线
        daily_data = self.db_manager.load_data(...)
        if len(daily_data) < 100:
            return []
        return daily_to_weekly(daily_data.to_dict('records'))
    
    def _pre_filter_by_wave(self, candidates: pd.DataFrame,
                             end_date: str) -> pd.DataFrame:
        """周线预过滤：排除下降浪"""
        if candidates.empty:
            return candidates
        
        symbols = candidates['symbol'].unique()
        tradeable = {}
        
        for symbol in symbols:
            weekly_data = self._get_weekly_data_for_symbol(symbol, end_date)
            if len(weekly_data) >= 20:
                result = self.wave_analyzer.analyze(weekly_data)
                tradeable[symbol] = result
        
        # 过滤：只在 can_trade=True 时操作
        mask = candidates['symbol'].map(
            lambda s: tradeable.get(s, {}).get('can_trade', False)
        )
        return candidates[mask].copy()
    
    def get_signals(self, start_date: str, end_date: str):
        """获取信号（融合波浪缠论版本）"""
        # 1. 生成特征
        signals = self.generate_features(start_date, end_date)
        
        # 2. 【新增】周线预过滤
        tradeable_signals = self._pre_filter_by_wave(signals, end_date)
        
        # 3. 买入条件
        buy_condition = (
            tradeable_signals['growth_condition'] &
            tradeable_signals['ma_condition'] &
            tradeable_signals['angle_condition'] &
            tradeable_signals['volume_condition'] &
            tradeable_signals['macd_condition'] &
            (tradeable_signals['jc_condition'] | tradeable_signals['macd_jc']) &
            (tradeable_signals['ma_20'] < tradeable_signals['ma_55']) &
            (tradeable_signals['ma_55'] > tradeable_signals['ma_240'])
        )
        
        buy_signals = tradeable_signals[buy_condition].copy()
        buy_signals['signal_type'] = 'buy'
        
        # 4. 评分（加入波浪加成）
        buy_signals = self._score_with_wave_boost(buy_signals, end_date)
        
        # 5. 排序选股
        if not buy_signals.empty:
            final_picks = []
            for date in buy_signals['date'].unique():
                daily = buy_signals[buy_signals['date'] == date]
                # 【修复】改为 nlargest
                top = daily.nlargest(self.config['top_n'], 'total_score')
                final_picks.append(top)
            selected_buy = pd.concat(final_picks) if final_picks else None
        else:
            selected_buy = None
        
        # ...卖出信号逻辑不变
```

---

## 八、预期优化效果

假设 baseline = +25.83%

| 优化项 | 单独效果 | 累计效果 |
|--------|---------|---------|
| 修复 nsmallest | +7% | +32.83% |
| 修复 positive_flow | +3% | +35.83% |
| 提高 growth_min | +2% | +37.83% |
| 权重再分配 | +3% | +40.83% |
| 波浪预过滤 | +5% | +45.83% |
| **综合优化后** | | **+40~50%** |

**注**：以上为估算，实际效果需回测验证。建议按优先级逐项回测，避免一次性全量修改。

---

## 九、下一步行动

1. **立即**：修复 `nsmallest` → `nlargest` 和 `positive_flow` 符号
2. **本周**：回测验证修改效果
3. **下周**：实现波浪预过滤模块
4. **下月**：调优权重和止盈策略

---

*报告完成*
