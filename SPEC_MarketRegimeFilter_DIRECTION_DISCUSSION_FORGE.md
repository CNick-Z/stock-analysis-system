# MarketRegimeFilter — Forge 技术方向讨论

> **Forge 视角 | 2026-04-02**
> 基于 `SPEC_MarketRegimeFilter_BACKTEST_RESULTS_v2.md` 回测报告 + 代码审查

---

## Forge 意见

### 技术实现评估

#### 1. NEUTRAL position_limit 调参 — 框架完全兼容，改默认值即可

**结论：✅ 零风险改动**

代码层面，`MarketRegimeFilter` 的 `__init__` 签名如下：

```python
def __init__(
    self,
    index_path: str = "/data/warehouse/indices/CSI300.parquet",
    confirm_days: int = 2,
    bear_position: float = 0.40,
    neutral_position: float = 0.80,   # ← 改这个默认值
    bull_position: float = 1.00,
):
```

调用方（`base_framework.py` `_process_buys`）完全不关心这些具体值，只取 `regime_info["position_limit"]` 这个 float：

```python
position_limit = regime_info["position_limit"]   # 0.0 ~ 1.0 之间任何值
avail_cash = self.cash * position_limit
```

`avail_cash / fill_count` 和 `self.cash * self.position_size` 的 `min()` 逻辑对任何 `position_limit` 值都是数学正确的，不需要改一行框架代码。

**唯一注意**：如果回测脚本通过 CLI 参数（`--filter-neutral-position`）传入，新值必须透传到 `MarketRegimeFilter()` 构造函数。需要检查回测入口脚本是否原样传递了这些参数——如果 CLI 参数集没有暴露 `neutral_position`，加一个参数也是 5 分钟的事。

---

#### 2. confirm_days=1 改为 1 — RSI 计算不变，连续确认逻辑需要理解变更风险

**结论：⚠️ RSI 计算本身不需要改，但 filter 行为会有质的变化**

**RSI 计算**：`prepare()` 中 RSI14 用 Wilders EWM 算法，一次计算、静态存储在 DataFrame 中，`confirm_days` 只决定"连续多少天满足条件才触发 regime 切换"。改 `confirm_days` 不影响 RSI 数值本身。

**连续确认逻辑**：当前 `_consecutive_sum` 是**向前**计数（look-back），即今天 `bear_consec = N` 表示"从今天往前数连续 N 天都满足 BEAR 条件"。改为 `confirm_days=1` 后：

| 场景 | confirm_days=2 | confirm_days=1 |
|------|----------------|----------------|
| 单日反弹打断 | 反弹日 `bear_consec→0`，从次日起重新计数 | 同左 |
| 快速切换 | 需要等连续2天才切换 regime | 隔日立即切换 |
| 虚假信号敏感度 | 低（需要连续确认） | **高（单日满足即触发）** |

**风险点**：2010-2012 大熊市（Window F）能有效，核心原因是 BEAR 天占比高达 26.5%、且持续时间长（连续下跌）。如果用 `confirm_days=1`，在震荡市中单日 RSI<40 就可能误触发 BEAR，导致不必要的减仓。回测报告自己也说"filter-confirm-days=2 使部分下跌期未被确认为 BEAR"——改为 1 之后，这个"未被确认"的问题消失了，但**假信号风险同等地增加了**。

**建议**：如果要试 `confirm_days=1`，必须同时看 Window C（2020-2021 震荡市，BEAR 仅 3.7%）的表现，确认误触发是否显著增加亏损。如果懒，那就保持 2 不动。

---

#### 3. 放弃过滤器（保留模块默认不启用）— 最小改动 2 处

**结论：✅ 最小改动，框架天然支持**

框架设计时已经考虑了 filter 可选——`market_regime_filter=None` 时买入逻辑完全不调用过滤器。最小改动：

```python
# 改动 1：MarketRegimeFilter 默认值（simulator/market_regime.py 或回测入口）
#   任何地方都不动，只在实例化时 explicit 传 None
#   → 改动量：0 行（框架已默认 None）

# 改动 2：回测入口脚本
#   将 --market-filter 参数默认设为 False（或移除默认启用逻辑）
#   → 改动量：1-2 行（取决于入口脚本怎么写参数解析）
```

或者更懒的方式：什么都不改，只需要在回测命令里不加 `--market-filter` 参数即可。框架 `_process_buys` 检查 `if self.market_regime_filter:`，没有传就是 None，逻辑上等价于"过滤器不存在"。

**唯一需要决策**：如果老板决定"保留模块但永远不用"，则代码注释应该标注 `deprecated`，避免后续开发者误以为还需要维护这个 feature。

---

### 建议

#### 如果老板问我"到底有没有用"——我会说实话

回测数据已经说明了一切：
- **唯一有效窗口**：2010-2012 大熊市，BEAR 占比 26.5%，持续时间长 → 收益 +1.23%，MaxDD -1.03%
- **其他 5 个窗口**：NEUTRAL 占比 50-70%，`position_limit=1.0` 等于没过滤；或者止损已经触发在先，`position_limit` 改不了已亏的钱

这不是"参数没调好"的问题，是**机制层面的局限**：

1. **最大回撤往往发生在"买在高点"之后**：`position_limit` 控制的是"买多少"，但最大亏损是在最高点买了之后跌下来的那段时间。跌的时候已经满仓了，限制新买入对这段回撤毫无作用。
2. **NEUTRAL 是主要状态**：6 个窗口里 5 个 NEUTRAL 超过 50%——只要 NEUTRAL→1.0，这个过滤器 50%+ 的时间是废的。
3. **confirm_days=2 是双刃剑**：减少了假信号，但也让真正的下跌确认过慢，错过了减仓的最佳时机。

#### 我的个人建议（仅供参考）

**保留模块，不启用**。理由：
- 代码已经写好了，维护成本几乎为零
- `market_regime_filter=None` 是框架原生支持的，不需要任何改动
- 未来如果有大牛市或长期熊市（比如 2025 年之后出现新的长期下跌），可以随时启用
- 但不要花时间调参——回测数据已经证明调参的边际收益极低，ROI 不值得

**如果要启用，最低改动让老板看到效果**：
- `confirm_days=1`（快速响应）
- `neutral_position=0.6`（打破 NEUTRAL=1.0 的死区）
- `bear_position=0.2`（更激进的熊市减仓）

但我必须说明：**Window F 的改善是 +1.23% 收益，-1.03% MaxDD，其他 5 个窗口全是 0。** 这意味着在正常市场环境下，这个过滤器大概率什么都不改变。老板要不要为"可能在大熊市省 1% 的亏损"付出维护成本，这是商业决策，不是技术决策。

---

*Forge — 💻 代码即工艺，数据说话*
