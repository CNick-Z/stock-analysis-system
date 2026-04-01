# 数据库说明文档

> 所有 Parquet 文件统一存储在 `/data/warehouse/` 目录下

---

## 一、股票日线数据

### 1.1 目录结构

```
/data/warehouse/
├── daily/                    # 日线行情（逐年）
│   ├── daily_2020.parquet
│   ├── daily_2021.parquet
│   ├── ...
│   └── daily_2026.parquet
├── technical/                # 技术指标（逐年）
│   ├── technical_indicators_2020.parquet
│   ├── technical_indicators_2021.parquet
│   └── ...
├── money_flow/               # 资金流指标（逐年）
│   ├── money_flow_indicators_2020.parquet
│   └── ...
└── indices/                  # 关键指数数据（增量更新）
    ├── CSI300.parquet         # 沪深300
    ├── SSE.parquet           # 上证指数
    └── GEM.parquet           # 创业板指
```

### 1.2 股票日线数据（daily）

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 交易日期，格式 YYYY-MM-DD |
| `symbol` | string | 股票代码，如 000001 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `volume` | float | 成交量（股） |
| `amount` | float | 成交额（元） |
| `turnover_rate` | float | 换手率 |
| `amplitude` | float | 振幅（%） |
| `change_pct` | float | 涨跌幅（%） |

**来源**：通达信增量 CSV `/opt/tdx_increment_*.csv`
**更新**：每日收盘后由 `auto_trade_executor.py` 追加

### 1.3 技术指标数据（technical）

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 交易日期 |
| `symbol` | string | 股票代码 |
| `sma_5/10/20/55/240` | float | 简单移动均线 |
| `macd/macd_signal/macd_histogram` | float | MACD 指标 |
| `rsi_14` | float | RSI（14日） |
| `kdj_k/kdj_d/kdj_j` | float | KDJ 指标 |
| `cci_20` | float | CCI 指标 |
| `williams_r` | float | 威廉指标 |
| `bb_upper/middle/lower` | float | 布林带 |
| `vol_ma5` | float | 成交量5日均线 |
| `boll_pos` | float | 布林带位置 |

**来源**：`data_loader.py` 从 daily 数据计算生成

### 1.4 资金流数据（money_flow）

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 交易日期 |
| `symbol` | string | 股票代码 |
| `XVL` | float | 主动卖量 |
| `LIJIN` | float | 主力净流入 |
| `ZLL` | float | 主力占比 |
| `money_flow_trend` | bool | 资金流趋势（正向=1） |
| `money_flow_increasing` | bool | 资金流入增加 |
| `vol_ratio` | float | 量比 |

**重建**：如需重建，执行：
```bash
python3 scripts/rebuild_money_flow.py --years 2024 2025 2026
```

---

## 二、指数数据（indices）

> 数据源：baostock（免费，无需 Token）  
> 更新频率：每个交易日 16:05 后增量更新（cron 自动）  
> 存储格式：Parquet，路径 `/data/warehouse/indices/`

### 2.1 指数列表

| 文件名 | 指数 | 代码 | 起始日期 | 说明 |
|--------|------|------|----------|------|
| `CSI300.parquet` | 沪深300 | `sh.000300` | 2006-01-04 | A股主流量化基准 |
| `SSE.parquet` | 上证指数 | `sh.000001` | 2006-01-04 | 最常用大盘指数 |
| `GEM.parquet` | 创业板指 | `sz.399006` | 2010-06-01 | 成长股代表 |

### 2.2 指数数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 交易日期 |
| `code` | string | 指数代码 |
| `open` | float | 开盘点位 |
| `high` | float | 最高点位 |
| `low` | float | 最低点位 |
| `close` | float | 收盘点位 |
| `volume` | float | 成交量（股） |
| `amount` | float | 成交额（元） |

### 2.3 指数数据用途

- **大盘牛熊判断**（MA20 + RSI14 组合）
- **仓位管理前置过滤**（MarketRegimeFilter）
- **择时信号**（均线多头/空头排列）

典型用法：
```python
import pandas as pd
df = pd.read_parquet("/data/warehouse/indices/CSI300.parquet")
df["ma20"] = df["close"].rolling(20).mean()
df["rsi14"] = compute_rsi(df["close"], 14)
```

### 2.4 增量更新机制

- **更新时间**：每个交易日 16:05（A股收盘后）
- **更新方式**：只抓取最新日期之后的数据，避免重复
- **日志文件**：`/var/log/index_data_update.log`
- **手动触发**：
  ```bash
  python3 scripts/pull_index_data.py        # 完整重建
  bash scripts/update_index_data.sh        # 增量更新
  ```

---

## 三、数据加载接口

### 3.1 统一数据入口

```python
from utils.data_loader import load_strategy_data

# 加载多年数据（含技术指标+资金流）
df = load_strategy_data(years=[2024, 2025])

# 参数：
#   years: 年份列表，如 [2024, 2025]
#   add_money_flow: 是否加载资金流（默认 True）
```

### 3.2 指数数据加载

```python
import pandas as pd
csi300 = pd.read_parquet("/data/warehouse/indices/CSI300.parquet")
```

---

## 四、数据校验

```bash
# 校验指数数据完整性
python3 -c "
import pandas as pd
for f, n in [('CSI300','沪深300'),('SSE','上证'),('GEM','创业板')]:
    df = pd.read_parquet(f'/data/warehouse/indices/{f}.parquet')
    print(f'{n}: {len(df)}行, {df[\"date\"].min()} ~ {df[\"date\"].max()}')
"
```

---

## 五、已知限制

- **科创50**：baostock 暂不支持 `sh.000688` 历史数据
- **指数不复权**：目前存储为不复权数据，如有复权需求需修改 `pull_index_data.py` 的 `adjustflag` 参数
