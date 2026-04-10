# WaveChan L1 缓存与 V3 策略结合研究
**Oracle 配合研究 | 2026-04-10**

---

## 一、数据评估结果

### 1.1 L1 缓存（极值点 + 趋势）

**存储位置：** `/data/warehouse/wavechan_l1/extrema_year={year}/`

**数据结构（每 symbol-year 一个 Parquet 文件 ~9KB）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | string | 股票代码 |
| date | timestamp | 极值日期 |
| price | float64 | 极值价格 |
| type | string | 'high' 或 'low' |
| extrema_index | int32 | 序号（全局） |
| swing_high | float64 | 该极值前的最高价 |
| swing_low | float64 | 该极值前的最低价 |
| retrace_pct | float64 | 回撤百分比（=回撤幅度/ swing幅度) |
| is_major | bool | 是否为主要极值（创新高/新低） |

**各年数据量：**

| 年份 | 股票数 | 平均极值点/股 | 备注 |
|------|--------|---------------|------|
| 2021 | 4,039 | ~46 | 完整 |
| 2022 | 2,750 | ~83 | 股票池收缩 |
| 2023 | — | — | 无数据 |
| 2024 | — | — | 无数据 |
| 2025 | 5,291 | ~70 | 股票池扩大 |

**关键质量指标（2025年抽样200文件）：**
- `retrace_pct > 5%`：**74.4%** 的极值点通过5%阈值
- 平均回撤：6.23%（略高于阈值，说明过滤有效）
- 最大日期：2025-12-31（数据已覆盖到最新）
- `is_major` 分布：平均每股票约 6 个主要极值点/年

**存在的问题：**
1. **年份断层**：2023、2024 无 L1 数据（目录为空）
2. **股票池漂移**：2021→2022 股票重叠率仅 **62.8%**（1504只消失，218只新增）
3. **无趋势标注**：当前 L1 只存储原始极值点，**没有**趋势方向（up/down/neutral）列

---

### 1.2 L2 缓存（wave_signals_cache）

**存储位置：** `/data/warehouse/wavechan_signals_cache.parquet`（单文件，~4.16M行）

**覆盖范围：**
- 时间：2021-07-06 → 2025-12-31
- 股票数：5,400 只

**核心信号列：**

| 字段 | 说明 |
|------|------|
| has_signal | 是否有信号 |
| total_score | 综合评分（0-80） |
| signal_type | W2_BUY / W4_BUY / C_BUY / none |
| signal_status | confirmed / alert / invalid |
| wave_state | w1_formed / w2_formed / w3_formed / w4_formed / w5_formed / w4_in_progress / initial |
| wave_trend | long / neutral / down |
| stop_loss | 止损价 |
| rsi / macd_hist / divergence / volume_ratio / fractal | 技术指标 |

**信号质量统计：**

| signal_type | 数量 | 平均 total_score | 中位数 |
|-------------|------|-----------------|--------|
| C_BUY | 48,110 | **56.5** | 55.0 |
| W2_BUY | 439,518 | **34.4** | 35.0 |
| W4_BUY | 530,751 | **22.4** | 20.0 |
| none | 3,146,919 | — | — |

**signal_status 分布：**
- confirmed: 287,472（28.2%）← 高质量，可用于入场
- alert: 439,229（43.1%）← 观察级
- invalid: 291,678（28.6%）← 已过期/失败

**V3 入场阈值：** total_score >= 15 → 通过率 **93.8%**（包含大量 W4_BUY alert）

---

### 1.3 L1 vs L2 数据分工

```
L1 极值缓存                    L2 信号缓存
────────────────────           ──────────────────────────
极值点（~40-80/股/年）          每日信号（~250/股/年）
精度：周线级别                  精度：日线级别
用途：结构判断 / 支撑阻力       用途：精细买卖点
数据量：~10KB/股/年            数据量：~50KB/股/年
延迟：低（静态归档）            延迟：中等（热数据）
```

**结论：L1 和 L2 是垂直互补关系，不是替代关系。**

---

## 二、接口设计方案

### 2.1 L1 查询接口

```python
from utils.wavechan_l1.manager import WaveChanL1Manager

manager = WaveChanL1Manager()

# ── 查询单只股票的极值点（支持跨年）─────────────────────
extrema = manager.get_extrema(symbol="000001.SZ", start_date="2025-01-01", end_date="2025-12-31")
# Returns: DataFrame with columns [date, price, type, retrace_pct, is_major, ...]

# ── 查询极值点附近的价格区间（用于判断支撑/压力）─────────
nearby = manager.get_nearby_extrema(symbol="000001.SZ", date="2025-06-15", lookback=3)
# 用于判断：当前价格是否接近重要极值点

# ── 批量预加载（回测初始化时）───────────────────────────
manager.preload_year(2025, symbols=target_list)
```

### 2.2 L1 + L2 联合查询

```python
def get_wave_context(symbol: str, date: str, l1_manager, l2_df) -> dict:
    """
    获取某只股票在某日期的波浪上下文（结合L1极值 + L2信号）
    
    Returns:
        {
            'extrema_nearby': DataFrame,   # L1: 日期附近3个极值点
            'last_major_extrema': dict,     # L1: 最近的主要极值点
            'daily_signal': Series,         # L2: 当日信号
            'wave_state': str,              # L2: 当前波浪状态
            'near_extrema_ratio': float,    # 价格离最近极值的幅度%
        }
    """
    # Step 1: L2 查当日信号（毫秒级）
    daily = l2_df[(l2_df['symbol'] == symbol) & (l2_df['date'] == date)]
    
    # Step 2: L1 查附近极值点（毫秒级，本地文件）
    nearby = l1_manager.get_nearby_extrema(symbol, date, lookback=3)
    
    # Step 3: 计算价格偏离度
    if len(nearby) > 0:
        last_ext = nearby.iloc[-1]
        price = daily['close'].values[0] if len(daily) else None
        if price:
            near_ratio = abs(price - last_ext['price']) / last_ext['price']
        else:
            near_ratio = None
    else:
        near_ratio = None
    
    return {
        'daily_signal': daily.iloc[0] if len(daily) else None,
        'extrema_nearby': nearby,
        'near_extrema_ratio': near_ratio,
    }
```

### 2.3 快速加载接口（回测初始化）

```python
class L1FastLoader:
    """
    L1 缓存快速加载器 - 用于回测初始化
    
    设计目标：
    - 全量加载 5000 只股票 1 年极值点：< 5 秒
    - 单股查询：< 1ms
    """
    
    def __init__(self, base_path="/data/warehouse/wavechan_l1"):
        self.base_path = Path(base_path)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._index: Dict[str, str] = {}  # symbol → file_path
    
    def load_year(self, year: int, symbols: List[str] = None) -> None:
        """预加载整年数据（推荐在回测开始前调用）"""
        year_path = self.base_path / f"extrema_year={year}"
        if not year_path.exists():
            raise ValueError(f"L1 year {year} not found")
        
        all_files = list(year_path.glob("*.parquet"))
        
        if symbols:
            targets = {s.replace('.SZ','').replace('.SH','') for s in symbols}
            files = [f for f in all_files if f.stem in targets]
        else:
            files = all_files
        
        # 并行读取（n_jobs=8）
        with ProcessPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(pd.read_parquet, str(f)): f.stem for f in files}
            for fut in as_completed(futures):
                sym = futures[fut]
                self._cache[sym] = fut.result()
                self._index[sym] = str(futures[fut])
        
        logger.info(f"L1: loaded {len(self._cache)} symbols for {year}")
    
    def query(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """查询某股票的极值点（支持日期过滤）"""
        sym_key = symbol.replace('.SZ','').replace('.SH','')
        if sym_key not in self._cache:
            return pd.DataFrame()
        
        df = self._cache[sym_key]
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        return df
    
    def get_near_extrema(self, symbol: str, price: float, direction: str = 'up', 
                         lookback: int = 5) -> pd.DataFrame:
        """
        查找最近极值点（用于判断突破/回踩）
        direction='up': 查最近的高点（潜在压力位）
        direction='down': 查最近的地点（潜在支撑位）
        """
        df = self.query(symbol)
        if df.empty:
            return pd.DataFrame()
        
        if direction == 'up':
            candidates = df[df['type'] == 'high'].tail(lookback)
        else:
            candidates = df[df['type'] == 'low'].tail(lookback)
        
        return candidates
```

---

## 三、V3 策略使用 L1 缓存的流程模拟

### 3.1 当前 V3 策略数据流（无 L1）

```
filter_buy(daily_df)
    ↓
L2 cache 查询：has_signal=True & total_score >= 15
    ↓
周线方向过滤（use_weekly_filter=True）
    ↓
候选股票列表
```

### 3.2 改进后 V3 策略数据流（加入 L1）

```
                    ┌──────────────────────────────────┐
                    │  每日选股流程（改进后）           │
                    └──────────────────────────────────┘

Step 1: L2 候选筛选（毫秒级，内存过滤）
    daily_df[L2] → has_signal=True & total_score >= 15
    → candidates_1 (~2400/日)

Step 2: L1 极值上下文过滤（毫秒级，本地Parquet）
    For each candidate:
        获取当日信号(signal_type, wave_state)
        查询 L1 极值点（最近3个）
        IF signal_type == 'W2_BUY':
            # W2 回调需要靠近前期低点
            nearest_low = L1.get_nearest_extrema(cand, direction='down')
            IF price < nearest_low['price'] * 0.95:
                REJECT（已跌破支撑，不入场）
        IF signal_type == 'W4_BUY':
            # W4 买入需要回调到重要支撑
            nearest_major = L1.get_major_extrema(cand, direction='down')
            IF price > nearest_major['price'] * 1.03:
                REJECT（未到支撑位，不追高）
    → candidates_2 (~500-800/日)

Step 3: L1 趋势结构辅助评分
    For each candidate:
        trend_score = 0
        # 检查前3个极值点的趋势方向
        recent_ext = L1.get_recent_extrema(cand, n=5)
        IF all recent_ext[-3:] are 'high' AND trend=='down':
            trend_score -= 10  # 下降趋势中的W4_BUY降低评分
        IF recent_ext[-1]['is_major'] == True AND type=='low':
            trend_score += 15  # 创新低后的反弹，W4_BUY加分
    → candidates_3 ( scored & ranked )

Step 4: 周线方向过滤（保持现有逻辑）
    IF use_weekly_filter:
        weekly_dir = get_weekly_direction(symbol)
        IF weekly_dir != 'up':
            REJECT
    → final candidates
```

### 3.3 出场判断中的 L1 应用

```python
def should_sell_with_l1(row, pos, market, l1_manager) -> Tuple[bool, str]:
    """
    出场判断（加入 L1 极值支持位）
    
    额外逻辑：
    - 如果持仓期间价格跌破 L1 近期主要低点（is_major=True, type='low'）
      且不是W5末期 → 提前出场（防止C浪破坏）
    """
    should_exit, reason = existing_should_sell(row, pos, market)
    
    if not should_exit:
        # 检查是否跌破近期主要支撑
        symbol = row['symbol']
        entry_price = pos['entry_price']
        
        # 获取持仓期间出现的新低
        new_extrema = l1_manager.query_since(symbol, since=pos['entry_date'])
        major_lows = new_extrema[
            (new_extrema['type'] == 'low') & 
            (new_extrema['is_major'] == True)
        ]
        
        for _, ext in major_lows.iterrows():
            # 如果出现创新低（相对入场后），且当前价格跌破该低点
            if ext['price'] < entry_price and row['close'] < ext['price']:
                return True, f"L1_MAJOR_BREAK:{ext['date']}:{ext['price']}"
    
    return should_exit, reason
```

---

## 四、性能预期

### 4.1 数据规模

| 层级 | 数据量 | 单文件大小 | 全量加载耗时（估算） |
|------|--------|-----------|---------------------|
| L1 2025 | 5,291 股票 × 70点 ≈ 37万行 | ~9KB/股 | **3-5 秒**（并行8核） |
| L2 全量 | 416万行 × 24列 ≈ 800MB | 1文件 | **5-10 秒**（内存映射） |

### 4.2 查询延迟

| 操作 | L1 查询 | L2 查询 |
|------|--------|---------|
| 单股单日极值点 | **< 1ms**（内存） | N/A |
| 单股N日极值窗口 | **< 5ms**（内存） | N/A |
| 单股单日信号 | N/A | **< 10ms**（索引） |
| 全市场候选过滤（2400股） | **< 200ms** | **< 500ms** |

### 4.3 每日增量回测预期

```
回测区间：2025-01-01 → 2025-04-10（约70个交易日）
股票池：5000只

当前（仅L2）：
  初始化加载：5-10秒
  每日 filter_buy：~500ms（含L2索引过滤）
  总回测耗时：~60秒

改进后（L1+L2）：
  初始化加载：8-15秒（含L1预热）
  每日 filter_buy（含L1上下文）：~800ms
  总回测耗时：~90秒

增加：~30秒（+50%），但选股质量显著提升
```

---

## 五、实现建议

### 5.1 数据层优先级

**P0（立即可做）：**
1. 补充 L1 2023、2024 年极值数据（当前目录为空）
2. 在 `wavechan_l1_design.py` 中补充 **趋势方向列**（up/down/neutral），这是当前最大的数据缺口
3. 将 L1 查询接入 `filter_buy` 前置过滤（先用 L2 粗筛，再用 L1 极值上下文过滤）

**P1（短期）：**
1. 实现 `L1FastLoader` 类，支持回测开始前的全量预加载
2. 在 `should_sell` 中增加 L1 major_low 跌破检测
3. 将 L1 极值点 is_major + retrace_pct 作为结构评分因子

**P2（中期）：**
1. 将 L1 数据预聚合为**支撑阻力位表**（symbol → date → price → 强度分），用于快速查询
2. 实现 L1 + L2 联合索引（symbol-date 双索引，避免 L2 全表扫描）
3. 将 L1 的 major extrema 标注到 L2 的 wave_signals_cache（merge 后持久化）

### 5.2 架构改动最小化方案

```
改动点1：在 WaveChanStrategy.__init__ 增加 l1_manager 参数
改动点2：在 prepare() 中调用 l1_manager.preload_year()
改动点3：在 filter_buy() 循环中加入 L1 上下文判断（10行以内）
改动点4：在 should_sell() 中增加 L1 major_low 检查（5行以内）
```

### 5.3 L1 缺失年份处理方案

```python
def get_extrema_fallback(symbol, year, l1_manager):
    """L1 数据缺失时的 fallback"""
    if year in [2023, 2024]:
        # 使用 L2 的 fractal 列作为极值替代
        # fractal 列包含 '底分型'/'顶分型'，可提取日线级别极值
        return l2_df[l2_df['symbol'] == symbol & l2_df['fractal'].notna()]
    return l1_manager.get_extrema(symbol, year=year)
```

---

## 六、总结

| 维度 | 评估 |
|------|------|
| L1 数据完整性 | ⚠️ 中等（2023-2024 缺失，趋势列缺失） |
| L1 数据质量 | ✅ 良好（74.4% 通过5%阈值，is_major 标注有效） |
| L1 与 V3 互补性 | ✅ 高（L1 提供结构判断，L2 提供精细信号） |
| 集成改造成本 | ✅ 低（架构改动 < 50 行代码） |
| 预期性能影响 | 可接受（回测总时长 +30-50%） |
| 最大价值点 | W2/W4 买入前增加极值支撑位过滤，减少假信号 |

**核心建议：优先补充 L1 缺失的 2023-2024 年数据，并在 `filter_buy` 中增加"价格是否回踩到 L1 极值支撑位"的前置判断，可有效过滤 W4_BUY 在高位追入的假信号。**
