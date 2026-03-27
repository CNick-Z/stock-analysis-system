# WaveChan V3 开发任务状态

> 更新日期：2026-03-27
> 负责人：Forge (dev-agent)

---

## Phase 1.1: 浪终结信号检测 ✅ 已完成

| 功能 | 状态 | 说明 |
|------|------|------|
| 推动浪终结检测 | ✅ 完成 | `_detect_wave_end()` 方法 |
| - W5量价背离 | ✅ 完成 | `w5_divergence` 标志 |
| - 失败浪检测 | ✅ 完成 | `w5_failed` 标志 |
| - 终结楔形检测 | ✅ 完成 | `_detect_ending_diagonal()` |
| 调整浪终结检测 | ✅ 完成 | |
| - C浪5浪结构完成 | ✅ 完成 | `c_wave_5_struct` |
| - C浪=Alun×1.618 | ✅ 完成 | `c_wave_fib_target` |
| 缠论分型确认 | ✅ 完成 | `_update_fx_info()` 使用 CZSC fx_list |

**实现文件**：`strategies/wavechan_v3.py`

**核心方法**：
- `WaveCounterV3._detect_wave_end(wave_result)` - 浪终结检测主方法
- `WaveCounterV3._get_wave_volume(wave_name)` - 获取浪的成交量
- `WaveCounterV3._detect_5_wave_down()` - C浪5浪检测
- `WaveCounterV3._detect_ending_diagonal(wave_name)` - 终结楔形检测
- `WaveCounterV3._update_fx_info()` - 分型信息更新

---

## Phase 1.2: 买卖点信号生成 ✅ 已完成

| 信号类型 | 状态 | 说明 |
|----------|------|------|
| W2_BUY | ✅ 完成 | W2回撤50-70%+底分型+量缩 |
| W4_BUY_ALERT | ✅ 完成 | W4进行中+向上笔 |
| W4_BUY_CONFIRMED | ✅ 完成 | W4终结+底分型 |
| W5_SELL | ✅ 完成 | W5终结信号 |
| C_BUY | ✅ 完成 | C浪买信号（熊市） |

**返回格式**：
```python
{
    'signal': str,      # 'W2_BUY' | 'W4_BUY_ALERT' | 'W4_BUY_CONFIRMED' | 'W5_SELL' | 'C_BUY' | 'NO_SIGNAL'
    'price': float,    # 建议价格
    'stop_loss': float,# 止损位
    'reason': str,     # 信号理由
    'confidence': float # 置信度 0.0~1.0
}
```

---

## Phase 1.3: 止损体系 ✅ 已完成

| 功能 | 状态 | 说明 |
|------|------|------|
| 动态止损位计算 | ✅ 完成 | `_calc_stop_loss()` |
| 波浪前低止损 | ✅ 完成 | `stop_loss_type='wave_end'` |
| 斐波那契支撑止损 | ✅ 完成 | `stop_loss_type='fib_support'` |
| 浮亏5%止损 | ✅ 完成 | 默认止损 |
| `get_stop_loss(entry_price)` | ✅ 完成 | 外部调用接口 |

---

## Phase 2.2: SymbolWaveCache 增强 ✅ 已完成

| 增强字段 | 状态 | 说明 |
|----------|------|------|
| `last_wave_end` | ✅ 完成 | 最后一浪终点时间 |
| `next_expected_wave` | ✅ 完成 | 下一个预期浪(W1/W2/W3/W4/W5) |
| `signal_history` | ✅ 完成 | 买卖点信号历史列表 |
| `fx_history` | ✅ 完成 | 缠论分型历史 |
| `get_signals_df()` | ✅ 完成 | 信号历史DataFrame |
| `get_latest_signal()` | ✅ 完成 | 获取最新信号 |
| `get_fx_df()` | ✅ 完成 | 分型历史DataFrame |
| `_check_and_record_signal()` | ✅ 完成 | 信号检查与记录 |
| `_update_next_expected_wave()` | ✅ 完成 | 预期浪更新 |
| 缓存版本升级 | ✅ 完成 | CACHE_VERSION = 2 |

---

## Phase 3.1: BatchWaveBuilder ✅ 已完成

| 功能 | 状态 | 说明 |
|------|------|------|
| `BatchWaveBuilder` 类 | ✅ 完成 | 在 wavechan_v3.py 中 |
| `batch_wave_builder.py` 脚本 | ✅ 完成 | 独立脚本 |
| 分批处理 | ✅ 完成 | 默认每批100只股票 |
| 进度管理 | ✅ 完成 | JSON进度文件 |
| 增量更新 | ✅ 完成 | `run_incremental()` |
| 指定股票构建 | ✅ 完成 | `run_symbols()` |
| 全量构建 | ✅ 完成 | `run_full()` |

**脚本用法**：
```bash
# 全量构建
python batch_wave_builder.py --full

# 增量更新今日
python batch_wave_builder.py --incremental

# 指定股票
python batch_wave_builder.py --symbols 600985 000001

# 查看进度
python batch_wave_builder.py --progress
```

---

## 其他改进

| 项目 | 说明 |
|------|------|
| Bug修复: `_get_turning_points` | 修正了向下笔的高/低价错误赋值 |
| Bug修复: `_scan_waves` 递归 | 修复了W1起点下移时的无限递归问题 |
| `BiRecord.volume` | 新增成交量字段用于量价背离检测 |
| 缓存结构升级 | v1 → v2，支持增强字段 |

---

## 待后续处理

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 多周期协同 (Phase 2.1) | P1 | 周线+日线+60分钟协同 |
| 历史案例验证 | P1 | 600143、上证指数等验证 |
| 对接 backtester.py | P2 | 集成到回测框架 |
| 对接实盘流程 | P2 | auto_trade_executor.py |

---

## 技术说明

### Bug修复记录

1. **`_get_turning_points` 高低赋值错误**：
   - 原代码对向下笔: start_type='low', end_type='high'（错误）
   - 修复后: start_type='high', end_type='low'（正确）
   - 影响: 所有波浪计数结果

2. **`_scan_waves` 无限递归**：
   - 原代码: 发现打破W1起点的笔时递归调用，但传入相同参数导致无限循环
   - 修复: 跳过已确认破坏W1起点的笔，用 `checked_seqs` 集合记录

### 注意事项

- CZSC `fx_list` 分型标记: `fx.mark == fx.mark.D` (底分型) 或 `fx.mark == fx.mark.G` (顶分型)
- `str(fx.mark)` 返回 '底分型' 或 '顶分型'
- `BiRecord` 新增 `volume` 字段用于量价背离检测
- `feed_bi(bi, fx_list=None)` 新增可选参数传入分型列表
