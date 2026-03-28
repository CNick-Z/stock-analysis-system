# WaveChan 缓存优化方案

**日期**: 2026-03-28
**状态**: ✅ **已实施完成**

## 背景

- 当前缓存是每日快照（416万行），换算法需7小时全量重建
- 目标：换算法分钟级，日常增量秒级

## 方案：三层缓存架构

```
L1. 历史归档层（Parquet，按年分区）
   存：2021-2024年数据，永不重建

L2. 热数据层（Parquet，按月分区）
   存：2025-2026年数据，换算法只重建当年
   大小：约50MB（2025年×12个月）

L3. 参数缓存层（SQLite，LRU）
   存：(算法版本+参数hash) → 回测结果
   淘汰：LRU，保留最近100组
```

## 收益对比

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 换算法回测 | 7小时全量重建 | 只重建L2（1-2小时） |
| 换参数回测 | 7小时全量 | 命中L3：毫秒级 / 未命中：分钟级 |
| 每日增量 | 需重建全量 | 只append到L2当月（约5秒） |

## 实施结果

### L1 历史归档（2021-2024）

| 年份 | 行数 | 大小 |
|------|------|------|
| 2021 | 379,471 | 16.5 MB |
| 2022 | 825,957 | 34.9 MB |
| 2023 | 949,469 | 39.9 MB |
| 2024 | 1,012,669 | 42.2 MB |

### L2 热数据（2025，按月分区）

| 月份 | 行数 | 大小 |
|------|------|------|
| 01 | 84,695 | 4.2 MB |
| 02 | 69,542 | 3.5 MB |
| 03 | 81,644 | 4.1 MB |
| 04 | 97,895 | 4.8 MB |
| 05 | 68,083 | 3.4 MB |
| 06 | 81,750 | 4.2 MB |
| 07 | 82,761 | 4.2 MB |
| 08 | 77,207 | 4.0 MB |
| 09 | 93,581 | 4.8 MB |
| 10 | 73,335 | 3.8 MB |
| 11 | 84,282 | 4.4 MB |
| 12 | 102,957 | 5.3 MB |
| **合计** | **997,732** | **50.6 MB** |

### L3 参数缓存

- 条目数：支持最多 100 组 LRU
- 版本：v3.0

## 文件清单

### 核心实现

| 文件 | 说明 |
|------|------|
| `utils/wavechan_cache.py` | 三层缓存管理器（主模块） |
| `wavechan_cache_migrate.py` | 缓存分区迁移脚本 |
| `wavechan_optimizer.py` | 优化器（已适配三层缓存） |

### 缓存路径

| 层级 | 路径 |
|------|------|
| L1 | `/root/.openclaw/workspace/data/warehouse/wavechan_cache/l1_cold_year={year}/data.parquet` |
| L2 | `/root/.openclaw/workspace/data/warehouse/wavechan_cache/l2_hot_year={year}_month={mm}/data.parquet` |
| L3 | `/root/.openclaw/workspace/data/warehouse/wavechan_l3_cache.db` |

## 使用方法

### 查看缓存状态

```bash
python3 wavechan_optimizer.py --cache-status
```

### 运行优化（启用L3）

```bash
python3 wavechan_optimizer.py --trials 100 --use-l3
```

### 每日增量更新

```python
from utils.wavechan_cache import WaveChanCacheManager

cm = WaveChanCacheManager()
# df 是当日新计算出的信号数据（包含 date, symbol 等字段）
cm.daily_increment("2026-03-28", df)
```

### 重建L2（换算法时）

```python
from utils.wavechan_cache import WaveChanCacheManager
from strategies.wavechan_selector import WaveChanSelector

cm = WaveChanCacheManager()
selector = WaveChanSelector()

# 只重建 2025-2026
cm.rebuild_l2([2025, 2026], params={...}, selector=selector)
```

### 迁移旧缓存

```bash
# Dry run
python3 wavechan_cache_migrate.py --dry-run

# 执行迁移
python3 wavechan_cache_migrate.py
```

## 下一步（Phase 2 可选）

1. L3 参数缓存增强：支持更多回测结果类型
2. L2 并行重建：多进程加速算法切换
3. 监控告警：缓存重建失败时通知
