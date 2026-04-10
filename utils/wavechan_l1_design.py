"""
WaveChan L1 缓存 - 新方案详细设计
===================================
目标：周线数据 + 周线笔 + 大浪标签（W1-W5）
存储：/data/warehouse/wavechan_l1/  按年+symbol 分区
数据源：/root/.openclaw/workspace/data/warehouse/daily_data_year={yr}
"""

# ============================================================
# 一、数据结构定义（Parquet Schema）
# ============================================================

# --------------------------------------------------
# 1. weekly_klines - 周线K线
# 文件路径：weekly_klines_year={year}/{symbol}.parquet
# 每只股票每年一个文件
# --------------------------------------------------
WEEKLY_KLINES_SCHEMA = {
    "date": "timestamp[ns]",      # 周线日期（每周最后一个交易日）
    "symbol": "string",           # 股票代码
    "open": "float64",            # 周开盘价
    "high": "float64",            # 周最高价
    "low": "float64",             # 周最低价
    "close": "float64",           # 周收盘价
    "volume": "float64",          # 周成交量
    "amount": "float64",          # 周成交额
    "change_pct": "float64",      # 周涨跌百分比
    "turnover_rate": "float64",   # 周换手率
    "upper_shadow": "float64",    # 上影线长度
    "lower_shadow": "float64",    # 下影线长度
    "body_size": "float64",       # K线实体大小
}

# 示例数据：
# date=2021-01-08, symbol=600368, open=10.0, high=10.8, low=9.5, close=10.5, volume=xxx, amount=xxx, ...

# --------------------------------------------------
# 2. weekly_bi - 周线笔
# 文件路径：weekly_bi_year={year}/{symbol}.parquet
# 每只股票每年一个文件
# --------------------------------------------------
WEEKLY_BI_SCHEMA = {
    "symbol": "string",            # 股票代码
    "bi_index": "int32",           # 笔序号（从0开始，全局唯一于该symbol）
    "start_date": "timestamp[ns]", # 笔起始日期
    "end_date": "timestamp[ns]",   # 笔结束日期
    "start_price": "float64",      # 笔起始价格
    "end_price": "float64",        # 笔结束价格
    "direction": "string",         # 'up' 或 'down'
    "power": "float64",            # 笔力度（涨跌幅绝对值）
    "length_weeks": "int32",       # 笔持续周数
    "is_major_high": "bool",       # 是否为主要高点（创新高）
    "is_major_low": "bool",        # 是否为主要低点（创新低）
    "wave_label": "string",        # 大浪标签：W1/W2/W3/W4/W5/None（后续标注）
}

# 示例数据：
# symbol=600368, bi_index=0, start_date=2021-01-08, end_date=2021-03-15,
# start_price=9.5, end_price=11.2, direction=up, power=17.9, length_weeks=9,
# is_major_high=False, is_major_low=False, wave_label=None

# --------------------------------------------------
# 3. wave_labels - 大浪标注点
# 文件路径：wave_labels_year={year}/{symbol}.parquet
# 只存储主要拐点（创新高/创新低的笔终点）
# --------------------------------------------------
WAVE_LABELS_SCHEMA = {
    "symbol": "string",             # 股票代码
    "wave_label": "string",          # W1/W2/W3/W4/W5
    "date": "timestamp[ns]",         # 拐点日期
    "price": "float64",             # 拐点价格
    "bi_index": "int32",            # 对应的周线笔序号
    "trend_type": "string",          # uptrend/downtrend/oscillation（标注时的判断）
}

# 示例数据：
# symbol=600368, wave_label=W1, date=2021-03-15, price=11.2, bi_index=1, trend_type=uptrend
# symbol=600368, wave_label=W2, date=2021-05-10, price=9.8, bi_index=2, trend_type=oscillation

# --------------------------------------------------
# 4. symbols_index - 股票索引（每年一个文件，记录该年涉及哪些symbol）
# --------------------------------------------------
SYMBOLS_INDEX_SCHEMA = {
    "symbol": "string",
    "min_date": "timestamp[ns]",  # 该股票在该年数据的最早日期
    "max_date": "timestamp[ns]",  # 该股票在该年数据的最晚日期
    "klines_count": "int32",     # 周线K线条数
    "bis_count": "int32",        # 周线笔数量
    "waves_count": "int32",      # 标注的大浪数量
}


# ============================================================
# 二、代码模块设计
# ============================================================

"""
目录结构：
/root/.openclaw/workspace/projects/stock-analysis-system/utils/wavechan_l1/
    __init__.py
    aggregator.py      # 周线K线聚合
    bi_recognizer.py   # CZSC 周线笔识别
    wave_labeler.py     # 大浪标注算法
    reader.py           # 读取接口
    manager.py          # 主管理器（增删改查）
    cli.py              # CLI 入口
    _path.py            # 路径工具函数
"""

# --------------------------------------------------
# 模块 1: _path.py - 路径工具
# --------------------------------------------------
"""
Base: /data/warehouse/wavechan_l1/

weekly_klines_year={year}/{symbol}.parquet
weekly_bi_year={year}/{symbol}.parquet
wave_labels_year={year}/{symbol}.parquet
symbols_index_year={year}.parquet

函数：
- get_weekly_klines_path(symbol, year) -> Path
- get_weekly_bi_path(symbol, year) -> Path
- get_wave_labels_path(symbol, year) -> Path
- get_symbols_index_path(year) -> Path
- list_symbols_in_year(year) -> List[str]
- list_available_years() -> List[int]
- ensure_base_dirs() -> None
"""

# --------------------------------------------------
# 模块 2: aggregator.py - 周线K线聚合
# --------------------------------------------------
"""
核心函数：aggregate_daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame

算法：
1. daily_df 必须包含: date(str), symbol, open, high, low, close, volume, amount
2. date 转换为 timestamp[ns]
3. 按 symbol + date.groupby(
       date=resample='W',  # 每周最后一个交易日
       open='first', high='max', low='min', close='last',
       volume='sum', amount='sum'
   )
4. change_pct = (close - prev_close) / prev_close * 100
5. turnover_rate 需要总股本数据，估算为 volume / 流通股本（可用当日成交量估算）
6. upper_shadow = high - max(open, close)
7. lower_shadow = min(open, close) - low
8. body_size = abs(close - open)

注意事项：
- 周线的"星期"概念：中国A股一周是周一到周五
- resample 策略：'W-FRI' 每周最后一个交易日（周五）
- 需要 lookback 一周数据用于计算第一根周K的 open（如果该周只有一天则用日的 open）
- 如果某周数据不足3天（少于3个交易日），跳过该周（数据不完整）

批量聚合：aggregate_year_daily(year: int, symbols: List[str]) -> None
- 读取 /root/.openclaw/workspace/data/warehouse/daily_data_year={year}
- 按 symbol 分批处理
- 输出到 /data/warehouse/wavechan_l1/weekly_klines_year={year}/
"""

# --------------------------------------------------
# 模块 3: bi_recognizer.py - CZSC 周线笔识别
# --------------------------------------------------
"""
核心函数：recognize_weekly_bi(klines_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]

输入：周线K线 DataFrame（from aggregator）
输出：
  1. bi_df: 周线笔 DataFrame（WEEKLY_BI_SCHEMA）
  2. remaining_bars: 未完成笔的最后一根K线（用于后续续算）

算法：
1. 构建 RawBar 列表（周线，freq=Freq.W）
2. CZSC(bars, max_bi_num=500)
3. 遍历 c.finished_bis，构建 bi_df
4. 关键字段：
   - bi_index: 全局序号（需要从多年数据连续编号）
   - is_major_high: 该笔终点是否为当时的历史最高
   - is_major_low: 该笔终点是否为当时的历史最低
   - power: abs(end_price - start_price) / start_price * 100
   - length_weeks: 周数

多年数据处理：
- 需要 lookback 至少20根历史周K（用于 CZSC 初始化笔）
- 处理跨年边界：年的第一根周K需要继承上一年的最后一根K线
- bi_index 需要跨年连续（symbol 维度）

批量处理：recognize_year_bi(year: int, symbols: List[str], lookback_years: int = 3) -> None
- 需要读取多年数据来识别年的边界笔
- 输出到 /data/warehouse/wavechan_l1/weekly_bi_year={year}/
"""

# --------------------------------------------------
# 模块 4: wave_labeler.py - 大浪标注算法
# --------------------------------------------------
"""
核心函数：label_waves(bi_df: pd.DataFrame) -> pd.DataFrame

"高低点同向变化"原则：
  - 高点抬高 + 低点抬高 = 上涨（W1/W3/W5）
  - 高点降低 + 低点降低 = 下跌
  - 否则 = 震荡（W2/W4）

算法（找主要拐点 + 标注 W1-W5）：

1. 过滤 is_major_high=True OR is_major_low=True 的笔
   → 得到主要拐点序列 major_points = [(date, price, type), ...]
   其中 type ∈ {high, low}

2. 从 major_points 中找 W1-W5：
   - W1: major_points[0] 的方向决定起始趋势
   - W2: major_points[1]，震荡（回撤超过 W1 幅度的50%？）
   - W3: major_points[2]，方向与 W1 相同
   - W4: major_points[3]，震荡
   - W5: major_points[4]，方向与 W1 相同
   - ...后续更多点继续标注 W1-W5 循环

3. 简化标注规则（奇数段原则）：
   - 从 major_points[0] 开始，每3个点组成一组：
     [W1_start, W2_low/high, W3_end] = 一段完整趋势
   - 连接主要高低点标注 W1-W5
   - 如果只有 W1/W2 未完成 W3，等待数据补齐再标注

4. trend_type 判断（用"同向变化"）：
   - 遍历 major_points，相邻两点：
     - 如果后一个高点 > 前一个高点 AND 后一个低点 > 前一个低点 → uptrend
     - 如果后一个高点 < 前一个高点 AND 后一个低点 < 前一个低点 → downtrend
     - 否则 → oscillation

大浪标注状态：
- wave_label 初始为 None
- 当 W3 完成时，W1/W2/W3 可以确定
- 当 W5 完成时，W4/W5 可以确定
- 可以分批标注（先标注已完成的浪）

输出：wave_labels DataFrame（WAVE_LABELS_SCHEMA）

批量处理：label_year_waves(year: int, symbols: List[str]) -> None
- 需要读取多年 BI 数据（至少包含所有 major_points）
- 输出到 /data/warehouse/wavechan_l1/wave_labels_year={year}/
"""

# --------------------------------------------------
# 模块 5: reader.py - 读取接口
# --------------------------------------------------
"""
# 读取某只股票的周线K线（跨多年）
read_weekly_klines(symbol: str, start_year: int, end_year: int) -> pd.DataFrame

# 读取某只股票的周线笔（跨多年）
read_weekly_bi(symbol: str, start_year: int, end_year: int) -> pd.DataFrame

# 读取某只股票的大浪标注
read_wave_labels(symbol: str, start_year: int, end_year: int) -> pd.DataFrame

# 读取某年某只股票的完整 L1 数据（合并三张表）
read_symbol_l1(symbol: str, year: int) -> dict:
    {
        'klines': DataFrame,
        'bi': DataFrame,
        'labels': DataFrame,
    }

# 高效读取：利用 Parquet 的 partition 剪裁
# symbol 作为文件名，不需要 filter，直接拼路径读
"""

# --------------------------------------------------
# 模块 6: manager.py - 主管理器
# --------------------------------------------------
"""
class WaveChanL1Manager:
    base_path: Path = /data/warehouse/wavechan_l1/

    # ------ 构建流程 ------
    def build_year(
        self,
        year: int,
        symbols: List[str] = None,  # None = 全部
        n_jobs: int = 8,             # 并行worker数
        verbose: bool = True,
    ) -> None:
        '''
        完整构建某年的 L1 数据（klines → bi → labels）
        流程：
        1. aggregator.aggregate_year_daily(year, symbols)
        2. bi_recognizer.recognize_year_bi(year, symbols, lookback_years=max(3, year-2018))
        3. wave_labeler.label_year_waves(year, symbols)
        4. 更新 symbols_index
        '''
        # 分批并行：symbol 维度并行（每只股票独立）
        # 少量股票用串行，大量股票用 ProcessPoolExecutor

    def rebuild_year(
        self,
        year: int,
        symbols: List[str] = None,
    ) -> None:
        '''重建某年的 L1（删除旧数据，重新构建）'''
        # 先删目录，再 build_year

    # ------ 增量更新 ------
    def daily_increment(
        self,
        date: str,      # YYYY-MM-DD
        daily_df: pd.DataFrame,
    ) -> None:
        '''
        每日增量：当日数据进入日线库后，更新 L1
        1. 找到 date 所在的 year
        2. 更新 weekly_klines（重算当周K线）
        3. 更新 weekly_bi（续算 CZSC）
        4. 更新 wave_labels（追加新标注）
        '''
        # 当周K线 = 找到 date 所属周（周一到周五），重新聚合
        # CZSC.update() 续算
        # 标注：如果形成新的 major point，追加 wave_label

    # ------ 查询 ------
    def get(self, symbol: str, year: int) -> dict:
        '''读取单只股票单年的完整 L1 数据'''

    def query(
        self,
        symbols: List[str] = None,
        years: List[int] = None,
        data_type: str = 'all',  # 'klines' | 'bi' | 'labels' | 'all'
    ) -> dict:
        '''批量查询'''

    def get_major_points(
        self,
        symbol: str,
        years: List[int] = None,  # None = 全部
    ) -> pd.DataFrame:
        '''获取某股票的所有主要拐点（用于画大级别浪）'''

    def status(self) -> dict:
        '''返回 L1 状态（各年数据量、文件大小）'''
"""

# --------------------------------------------------
# 模块 7: cli.py - CLI 入口
# --------------------------------------------------
"""
用法：
  python -m wavechan_l1 build --year 2024 --symbols 600368,000001
  python -m wavechan_l1 build --year 2024 --all
  python -m wavechan_l1 rebuild --year 2024
  python -m wavechan_l1 status
  python -m wavechan_l1 read --symbol 600368 --year 2024
"""


# ============================================================
# 三、核心算法逻辑详解
# ============================================================

# --------------------------------------------------
# 3.1 周线K线聚合算法
# --------------------------------------------------
"""
输入：daily_df (date, symbol, open, high, low, close, volume, amount)

伪代码：
```
def aggregate_daily_to_weekly(daily_df):
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # 中国A股日历：周一~周五为一周
    # 每周最后一个交易日作为周线日期
    daily_df['week'] = daily_df['date'].dt.to_period('W-FRI')
    
    weekly = daily_df.groupby(['symbol', 'week']).agg(
        date=('date', 'max'),           # 当周最后一个交易日
        open=('open', 'first'),         # 周一开盘
        high=('high', 'max'),           # 周最高
        low=('low', 'min'),             # 周最低
        close=('close', 'last'),       # 周五收盘
        volume=('volume', 'sum'),       # 周总成交量
        amount=('amount', 'sum'),      # 周总成交额
    ).reset_index(drop=True)
    
    # 计算衍生字段
    weekly = weekly.sort_values(['symbol', 'date'])
    weekly['change_pct'] = weekly.groupby('symbol')['close'].pct_change() * 100
    weekly['upper_shadow'] = weekly['high'] - weekly[['open', 'close']].max(axis=1)
    weekly['lower_shadow'] = weekly[['open', 'close']].min(axis=1) - weekly['low']
    weekly['body_size'] = (weekly['close'] - weekly['open']).abs()
    
    return weekly
```

注意：
- 每年第一根周K的前一周 change_pct 为 NaN（没有前一周数据），保留即可
- 如果某周少于3个交易日（节假日等原因），该周数据保留但不标注
- 中国节假日处理：groupby('week') 时自然处理（没有数据的周不出现在结果中）
"""

# --------------------------------------------------
# 3.2 周线笔识别算法
# --------------------------------------------------
"""
输入：klines_df (周线K线，多年数据)

关键点：
1. CZSC 需要至少 20 根历史K线才能识别第一笔
2. 跨年边界：第一年的最后几根K线需要保留用于第二年的初始化
3. bi_index 跨年连续

伪代码：
```
def recognize_weekly_bi(klines_df):
    bars = [RawBar(symbol=s, dt=row.date, freq=Freq.W,
                   open=row.open, high=row.high, low=row.low,
                   close=row.close, vol=row.volume, amount=row.amount)
            for _, row in klines_df.iterrows()]
    
    c = CZSC(bars, max_bi_num=500)
    
    # 计算历史高低点（用于判断 is_major_high/low）
    all_prices = klines_df['close'].tolist()
    
    bi_list = []
    for i, bi in enumerate(c.finished_bis):
        end_price = bi.fx_b.fx
        end_idx = ...  # 在 klines_df 中的位置
        
        # 判断是否创新高：end_price > 之前所有收盘价
        is_major_high = end_price == max(all_prices[:end_idx+1])
        # 判断是否创新低
        is_major_low = end_price == min(all_prices[:end_idx+1])
        
        bi_list.append({
            'bi_index': global_idx,
            'end_date': bi.fx_b.dt,
            'end_price': end_price,
            'start_date': bi.fx_a.dt,
            'start_price': bi.fx_a.fx,
            'direction': 'up' if bi.direction == Direction.Up else 'down',
            'is_major_high': is_major_high,
            'is_major_low': is_major_low,
        })
    
    return pd.DataFrame(bi_list)
```

关于 is_major_high / is_major_low 的准确理解：
- 周线笔的终点（fx_b）如果在当时创出了历史最高价 → is_major_high = True
- 周线笔的终点如果在当时创出了历史最低价 → is_major_low = True
- 同一个终点不可能同时是 major_high 和 major_low（互斥）
- 如果既不是创新高也不是创新低 → 这是一个"次级折返"（W2/W4）
"""

# --------------------------------------------------
# 3.3 大浪标注算法（核心）
# --------------------------------------------------
"""
输入：bi_df (周线笔DataFrame，包含 is_major_high, is_major_low)

算法：

Step 1: 提取主要拐点
```
# 只保留主要拐点
major = bi_df[(bi_df.is_major_high == True) | (bi_df.is_major_low == True)].copy()
major = major.sort_values('end_date').reset_index(drop=True)
# 同时记录是 high 还是 low
```

Step 2: 判断趋势（用同向变化原则）
```
def judge_trend(prev_point, curr_point):
    # prev_point, curr_point 都是 {'date', 'price', 'is_high'}
    if curr_point['is_high'] and prev_point['is_high']:
        if curr_point['price'] > prev_point['price']:
            return 'uptrend'   # 高点抬高
        else:
            return 'downtrend' # 高点降低
    elif not curr_point['is_high'] and not prev_point['is_high']:
        if curr_point['price'] > prev_point['price']:
            return 'uptrend'   # 低点抬高
        else:
            return 'downtrend' # 低点降低
    else:
        return 'oscillation'   # 高低点方向不一致
```

Step 3: 标注 W1-W5（奇数段原则）
```
# 简化标注：major_points 依次编号
# major_points[0] → W1起始
# major_points[1] → W2（震荡点）
# major_points[2] → W3
# major_points[3] → W4（震荡点）
# major_points[4] → W5
# ...

# 更精确的标注：只有同向变化才继续趋势，否则为震荡
labels = []
i = 0
wave_num = 1  # W1
while i < len(major_points) - 1:
    curr = major_points[i]
    next_p = major_points[i + 1]
    
    trend = judge_trend(curr, next_p)
    
    if wave_num in [1, 3, 5]:
        expected_trend = 'uptrend' if curr['is_high'] else 'downtrend'
    else:  # W2, W4
        expected_trend = 'oscillation'
    
    if trend == expected_trend or trend in ['uptrend', 'downtrend']:
        labels.append({'wave': f'W{wave_num}', 'point': curr})
        wave_num += 1
        if wave_num > 5:
            wave_num = 1  # 循环
    else:
        labels.append({'wave': f'W{wave_num}', 'point': curr})
        wave_num += 1
    
    i += 1
```

Step 4: 分批标注完成状态
```
# W1, W2, W3 标注时机：
# - 找到 W3 的终点（下一个 major point）后，W1/W2/W3 可以确定
# - 找到 W5 的终点后，W4/W5 可以确定

# 未完成的浪标注为 W1_forming, W2_forming 等
```

输出：wave_labels DataFrame
"""

# --------------------------------------------------
# 3.4 完整处理流程
# --------------------------------------------------
"""
Year = 2024

Step 1: 准备多年日线数据（用于 lookback）
  - 读取 2021-01-01 到 2024-12-31 的日线数据
  - 总计约 4 × 250 = 1000 交易日 × 4000 股票 = 400万行

Step 2: 周线K线聚合
  - 按 symbol + 周聚合
  - 输出：2021-2024 每年约 50 周 × N 只股票

Step 3: 周线笔识别
  - 对每只股票用 CZSC(Freq.W) 识别周线笔
  - 需要至少 20 根周K（≈5个月）才能产生第一笔
  - 跨年数据确保边界笔不断

Step 4: 大浪标注
  - 从多年 BI 数据中找到所有 major points
  - 标注 W1-W5
  - 按年+symbol 拆分写入

Step 5: 写入 Parquet
  - 每个 symbol → 一个 Parquet 文件
  - 按年分区（year={year}）
  - 使用 PyArrow，snappy 压缩
"""


# ============================================================
# 四、性能预估
# ============================================================

"""
场景：构建 2018-2026 共 8 年的 L1 数据，约 5000 只股票

1. 数据规模估算
   - 日线数据：8年 × 250天 × 5000股票 = 1000万行
   - 周线K线：8年 × 50周 × 5000股票 = 200万行
   - 周线笔：每只股票约 50-100 根笔（取决于走势）
   - 大浪标注：每只股票约 10-30 个点

2. 耗时预估（单进程）
   - 日线读取：~5分钟（1000万行 Parquet）
   - 周线聚合：~3分钟
   - CZSC 周线笔识别：~30秒/只股票 × 5000 = 约 40小时（！！）
   
   → 必须并行化

3. 并行化方案
   - Symbol 维度并行（每只股票独立）
   - 8核 CPU → 8 workers
   - 5000 / 8 ≈ 625 只/worker
   - 每只股票 ~1分钟（读取 + 聚合 + CZSC + 标注）
   - 总耗时：625 × 1分钟 / 8核 ≈ 78分钟 ≈ 1.3小时

4. 内存预估
   - 单只股票日线：250天 × 8年 = 2000行 × 12列 ≈ 200KB
   - 单只股票周线：50周 × 8年 = 400行 × 12列 ≈ 40KB
   - CZSC 对象：~1MB/只股票（500根 bars）
   - 并行 worker × 8 → 峰值 ~8MB + CZSC开销 ≈ 几十 MB
   - 完全在内存容许范围内

5. 存储预估
   - 周线K线：200万行 × 100字节 ≈ 200MB（压缩后）
   - 周线笔：50万行 × 150字节 ≈ 75MB
   - 大浪标注：15万行 × 100字节 ≈ 15MB
   - 总计：~300MB / 全量 8年数据

6. 优化建议
   - 日线数据只读一次，不要重复读取
   - 用 pyarrow.dataset 批量读取（比 pandas.read_parquet 快）
   - CZSC max_bi_num 不要太大（500够了）
   - is_major_high/low 计算：用 running_max/running_min 避免 O(n²)
   - Parquet 写入用 pyarrow 直接写（不转 pandas 再写）
"""


# ============================================================
# 五、目录结构
# ============================================================

DIRECTORY_STRUCTURE = """
/data/warehouse/wavechan_l1/          # L1 根目录
├── weekly_klines_year=2018/
│   ├── 600368.parquet                # 每只股票一个文件
│   ├── 600519.parquet
│   └── ...
├── weekly_klines_year=2019/
│   └── ...
├── ...
├── weekly_klines_year=2026/
│   └── ...
├── weekly_bi_year=2018/
│   └── {symbol}.parquet
├── ...
├── wave_labels_year=2018/
│   └── {symbol}.parquet
├── ...
├── symbols_index_year=2018.parquet    # 该年股票索引
├── symbols_index_year=2019.parquet
└── ...
"""
