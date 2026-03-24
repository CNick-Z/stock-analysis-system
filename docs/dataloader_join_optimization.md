# DataLoader JOIN 预计算指标优化方案

> 创建时间：2026-03-23
> 负责人：Fairy
> 状态：进行中

---

## 📋 问题分析

### 现状
- `utils/strategy.py` 的 `_fetch_precalculated_data()` 每次调用 `get_signals()` 都会：
  1. 从 Parquet 加载 `technical_indicators` 数据
  2. 从 Parquet 加载 `daily_data` 数据
  3. 合并（JOIN）两个数据集
- 回测循环中每天都会调用 `get_signals()`，导致大量重复 IO 和 JOIN

### 优化点
- **重复 IO**：同一批 Parquet 文件被重复读取
- **重复 JOIN**：同一批数据被重复合并

---

## 🎯 优化目标

减少 80%+ 重复 IO 和 JOIN 操作，提升回测速度 3-5 倍

---

## 💡 优化方案

### 核心思路：缓存预加载 + 内存 JOIN

**原流程**：
```
for 每天 in 回测期:
    get_signals(每天)  # 每次都重新加载Parquet + JOIN
```

**优化后**：
```
一次性加载所有预计算指标到内存
for 每天 in 回测期:
    get_signals(每天)  # 直接从内存读取，不重复IO
```

### 具体实现

#### 1. 新增 `PreloadedDataManager` 类
```python
class PreloadedDataManager:
    """
    预加载数据管理器
    
    在回测开始前一次性加载所有技术指标和行情数据到内存，
    后续直接内存访问，避免重复 IO 和 JOIN
    """
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._tech_df = None      # 技术指标 DataFrame
        self._price_df = None     # 行情 DataFrame
        self._info_df = None      # 股票信息 DataFrame
        self._merged_cache = {}    # 合并后数据缓存
        
    def preload(self):
        """一次性加载所有数据"""
        # 1. 扩展日期范围（技术指标需要前置数据计算角度）
        extended_start = pd.to_datetime(self.start_date) - pd.Timedelta(days=30)
        
        # 2. 批量加载所有年份的 Parquet（合并为单个 DataFrame）
        self._tech_df = self._load_technical_indicators(extended_start, self.end_date)
        self._price_df = self._load_daily_data(extended_start, self.end_date)
        self._info_df = self._load_stock_info()
        
        # 3. 预先计算合并结果（所有日期的完整数据）
        self._merged_cache = self._precompute_merged_data()
        
    def get_data(self, date: str) -> pd.DataFrame:
        """获取指定日期的数据（直接从缓存）"""
        return self._merged_cache.get(date, pd.DataFrame())
    
    def get_range_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日期范围内的数据"""
        mask = (self._merged_cache['date'] >= start_date) & \
               (self._merged_cache['date'] <= end_date)
        return self._merged_cache[mask]
```

#### 2. 修改 `EnhancedTDXStrategy` 使用缓存

**修改前**：
```python
class EnhancedTDXStrategy:
    def __init__(self, db_path):
        self.db_manager = DatabaseIntegrator(db_path)
        
    def get_signals(self, start_date, end_date):
        # 每次都重新加载
        raw_data = self._fetch_precalculated_data(start_date, end_date)
        # ... 生成信号
```

**修改后**：
```python
class EnhancedTDXStrategy:
    def __init__(self, db_path, use_cache=True):
        self.db_manager = DatabaseIntegrator(db_path)
        self.use_cache = use_cache
        self._data_cache = None
        
    def enable_cache(self, start_date: str, end_date: str):
        """启用数据缓存（在回测开始前调用）"""
        if self.use_cache:
            self._data_cache = PreloadedDataManager(start_date, end_date)
            self._data_cache.preload()
            
    def _fetch_precalculated_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        # 如果有缓存，直接从缓存获取
        if self._data_cache is not None:
            return self._data_cache.get_range_data(start_date, end_date)
        
        # 如果没有缓存，走原来的逻辑（兼容单次调用）
        return self._original_fetch_logic(start_date, end_date)
```

#### 3. 修改 `BacktestOrchestrator` 初始化缓存

```python
class BacktestOrchestrator:
    def run_backtest(self, start_date, end_date, ...):
        # 回测开始前，启用数据缓存
        self.strategy.enable_cache(start_date, end_date)
        
        # 后续的 get_signals() 调用会自动使用缓存
        for date in trading_dates:
            signals = self.strategy.get_signals(date, date)
            # ... 执行交易逻辑
```

---

## 📊 性能对比预估

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Parquet 读取次数 | N 次（N=交易日数） | 1 次 | 减少 99% |
| JOIN 次数 | N 次 | 0 次 | 减少 100% |
| 预估回测时间 | 30-40 分钟 | 5-10 分钟 | 3-5x |

---

## 🔧 实施步骤

1. [ ] 创建 `PreloadedDataManager` 类
2. [ ] 修改 `EnhancedTDXStrategy._fetch_precalculated_data()` 支持缓存模式
3. [ ] 修改 `BacktestOrchestrator.run_backtest()` 初始化缓存
4. [ ] 单元测试验证正确性
5. [ ] 性能测试验证提升效果

---

## ⚠️ 注意事项

1. **内存占用**：预加载会增加内存使用，预估 8-16GB（所有年份数据）
2. **日期扩展**：技术指标计算需要前置数据，start_date 需要向前扩展至少 30 天
3. **兼容性**：保留原逻辑，缓存模式仅在回测时启用

---

## 📁 关键文件

- 策略文件：`/projects/stock-analysis-system/utils/strategy.py`
- 回测引擎：`/projects/stock-analysis-system/backtester.py`
- 数据仓库：`/data/warehouse/`
- 技术指标：`technical_indicators_year=*/data.parquet`
