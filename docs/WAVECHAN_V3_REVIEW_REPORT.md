# WaveChan V3 双周期架构评审报告

> 整理日期：2026-03-27
> 参与评审：Oracle、Byte、Trader、Forge、Fairy
> 目标：完善 WaveChan V3 双周期架构，修复评审发现的问题

---

## 一、团队评审结论

| 评审 | 评分 | 核心问题 |
|------|------|---------|
| 🔍 Oracle | 方向待修 | direction用笔方向≠趋势；双周期未打通；止盈逻辑有缺陷 |
| 📊 Byte | 数据弱 | 缺少信号窗口控制；2015-2019历史数据缺失 |
| 📈 Trader | 风控弱 | 代码BUG；盈亏比中性；资金管理缺失 |
| 💻 Forge | 不可运行 | 多个P0空方法；周线笔生成链路未闭环 |

**团队共识**：
1. ✅ 架构方向正确：周线定方向 + 日线找买卖点
2. ⚠️ 代码跑不起来：多个P0空方法需要补全
3. ⚠️ 风控是最大短板

---

## 二、核心问题清单

### P0 - 必须修复

| # | 问题 | 来源 | 修复方案 |
|---|------|------|---------|
| P0-1 | `direction` 用最后一根笔方向 ≠ 趋势方向 | Oracle | 改为基于波浪结构判断 |
| P0-2 | `has_reversal_signal()` 方法不存在 | Forge | 补全或替换为现有方法 |
| P0-3 | `self.daily_czsc = None` 初始化缺失 | Forge | 初始化为空CZSC |
| P0-4 | `PositionManager` 未实现 | Oracle/Trader | 实现持仓管理类 |
| P0-5 | 止损变量引用未定义（代码BUG） | Trader | 修复变量作用域 |

### P1 - 重要建议

| # | 问题 | 来源 | 修复方案 |
|---|------|------|---------|
| P1-1 | 双周期协同链路未闭环 | Oracle/Forge | 日线→周线笔生成→引擎联动 |
| P1-2 | 止盈逻辑有缺陷 | Oracle | 增加W5终结特征检测 |
| P1-3 | 缺少信号有效期窗口 | Byte | 增加5-10交易日窗口控制 |
| P1-4 | 假突破应对缺失 | Trader | 增加突破后回落处理 |
| P1-5 | 时间止损缺失 | Trader | 增加持仓超时机制 |
| P1-6 | 资金管理缺失 | Trader | 增加仓位计算逻辑 |

### P2 - 增强项

| # | 问题 | 来源 | 修复方案 |
|---|------|------|---------|
| P2-1 | 历史数据缺失 | Byte | 补充2015-2019数据 |
| P2-2 | 分型强度验证缺失 | Oracle | 增加缩量/中间K线验证 |
| P2-3 | CZSC版本兼容性 | Forge | 锁死依赖版本 |

---

## 三、架构修改方案

### 3.1 核心类设计

```python
class WaveEngine:
    """双周期波浪引擎 - 最小可行版本"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.weekly_counter = WaveCounterV3()  # 周线引擎
        self.daily_czsc = CZSC([])            # 日线引擎（初始化）
        self.pending_weekly_bi = None          # 待提交的周线笔
        self.position = None                   # 持仓状态
    
    def feed_daily(self, bar: RawBar):
        """日线K线：同时更新CZSC + 尝试提取周线笔"""
        self.daily_czsc.update(bar)
        
        # 尝试从日线 finished_bis 提取/更新周线笔
        new_weekly_bi = self._maybe_generate_weekly_bi()
        if new_weekly_bi:
            self.weekly_counter.feed_bi(new_weekly_bi)
            self._check_entry_signals()
    
    def _maybe_generate_weekly_bi(self) -> Optional[BiRecord]:
        """聚合日线笔为周线笔——只在周线笔完成时返回"""
        # 复用 czsc_wave_v2.aggregate_weekly 逻辑
        pass
    
    def _get_trend_direction(self) -> str:
        """基于波浪结构判断真实趋势方向"""
        state = self.weekly_counter.state
        if state in ['w3_formed', 'w4_formed', 'w4_in_progress']:
            return 'long'
        elif state == 'w5_formed':
            return 'neutral'
        return 'neutral'


class PositionManager:
    """持仓管理器"""
    
    def __init__(self):
        self.position = None
    
    def entry_long(self, price: float, stop_loss: float, 
                   verify_price: float):
        """买入 + 制定卖出计划"""
        self.position = {
            "entry_price": price,
            "stop_loss": stop_loss,
            "verify_price": verify_price,  # 必须突破的价格
            "entry_time": datetime.now(),
        }
    
    def verify_position(self, current_price: float, 
                       daily_czsc: CZSC) -> str:
        """持续验证持仓"""
        if current_price < self.position["stop_loss"]:
            return "止损出场"
        
        # 验证是否突破
        if current_price >= self.position["verify_price"]:
            return "继续持有"
        
        # 检查日线顶分型
        if daily_czsc.has_top_fx():
            return "止盈离场"
        
        return "继续持有"
    
    def get_exit_action(self, current_price: float, 
                        entry_price: float) -> tuple:
        """判断出场类型和价格"""
        if current_price > entry_price:
            return "止盈离场", current_price
        else:
            return "止损出场", self.position["stop_loss"]
```

### 3.2 止盈逻辑修正

```python
def _check_take_profit(self, state: str, w5_info: dict) -> str:
    """修正后的止盈逻辑"""
    
    # W5终结特征检测
    if w5_info.get('w5_failed'):
        return "W5失败，止盈离场"
    if w5_info.get('w5_divergence'):
        return "W5背离，止盈离场"
    if w5_info.get('w5_ending_diagonal'):
        return "W5楔形，止盈离场"
    
    # 普通止盈：顶分型 + 不破理论目标
    if self.daily_czsc.has_top_fx():
        return "日线顶分型，止盈离场"
    
    return "继续持有"
```

### 3.3 双周期协同流程

```
日线 feed_daily(bar)
    ↓
更新 CZSC
    ↓
检测周线笔完成 → feed_weekly
    ↓
weekly_counter 更新状态
    ↓
_get_trend_direction() → direction
    ↓
direction == "long" 且无持仓
    ↓
检查日线底分型 → 买入信号
    ↓
有持仓 → verify_position
    ↓
止损/止盈/持有
```

---

## 四、修改优先级

### 第一阶段：修复P0，让代码能跑起来

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 修复direction逻辑 | Oracle/Forge | WaveEngine._get_trend_direction() |
| 补全初始化 | Forge | self.daily_czsc初始化 |
| 实现PositionManager | Trader | 完整持仓管理类 |
| 修复代码BUG | Forge | 止损变量作用域 |

### 第二阶段：完善风控和信号体系

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 增加信号窗口控制 | Byte | signal_age字段 |
| 增加假突破应对 | Trader | 突破后回落处理 |
| 增加时间止损 | Trader | 持仓超时机制 |
| 增加仓位管理 | Trader | 固定仓位公式 |

### 第三阶段：回测验证

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 补充历史数据 | Byte | 2015-2019数据 |
| 小批量回测 | Byte | 10只股票回测报告 |
| 策略优化 | 团队 | 根据回测结果调整 |

---

## 五、600143 实战案例更新

### 当前状态（周线）

```
W1: 5.77 → 14.46
W2: 14.46 → 8.84
W3: 8.84 → 24.08
W4: 24.08 → 17.10（进行中）
W5: 等待
```

### 正确操作流程

```
周线：17.10 出现底分型 → 预判 W4 可能终结
    ↓
方向：long（潜在）
    ↓
日线：等日线底分型买入
    ↓
买入：17.10 附近日线底分型
止损：14.66（fib_618）
验证：能否突破 24.08

持仓中：
  - 突破 24.08 → 继续持有
  - 日线顶分型 + 不破 24.08 → 止盈离场
  - 跌破 14.66 → 止损出场
```

---

## 六、核心教训

### 架构设计原则

1. **周线定方向，日线找买卖点**
2. **假设 + 持续验证，不是等确认**
3. **INVALID = 操作指令，不只是状态变更**
4. **买入时就定好卖出计划**

### 代码实现原则

1. **所有方法必须实现，不能有空引用**
2. **职责边界要清晰，不能互相依赖未定义的方法**
3. **状态机要完整，状态转换要有明确定义**
4. **风控要前置，不能等上线后再加**

---

*本报告由 Fairy 整理，经团队评审通过*
