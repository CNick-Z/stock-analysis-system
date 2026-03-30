#!/usr/bin/env python3
"""
Score 策略 v8 — IC/IR 增强版（v6 核心 + IC 过滤增强）

【设计原则】
v6 核心完全不动！只在买入前加 IC 过滤条件

【v6 核心保留】 ✅
- growth_condition: 涨幅条件
- ma_condition: MA5 > MA10, MA10 < MA20
- angle_condition: 均线角度
- volume_condition: 成交量放大
- macd_condition: MACD 负值区
- jc_condition: 金叉条件
- MA20 < MA55, MA55 > MA240: 趋势条件
- 低价股: 3-15元
- 评分排序选 top_n

【IC 增强 — 买入前过滤】 ✅
- ✅ CCI 超卖(CCI < -100): 必须满足（IC最强因子）
- ✅ 低换手(turnover < 2.79%): 必须满足（IC剔除高换手陷阱）
- ✅ 低量比(vol_ratio < 1.25): 必须满足（IC剔除放量陷阱）
- ✅ RSI 极值排除: >70 或 <25 剔除（IC显示极值反转差）
- ✅ WR 极度超卖排除: WR < -95 剔除（极度超卖反转差）

【IC 增强 — 评分加分】 ✅
- CCI < -100 加分: 权重 0.10（最强 IC 因子额外加权）
- WR < -80 加分: 权重 0.05
- 低换手加分: turnover < 0.42% 加分 0.05

【移除 v6 中 IC 负向的加权】 ⚠️
- v6: vol_ratio 越大加分越多 → v8: 改为 vol_ratio > 1.25 直接剔除
- v6: rsi_oversold 权重 0.0597 → v8: RSI < 70 才给分，但加 IC 极值剔除

【关键不同】
v7 的错误：用 IC 反转信号（CCI超卖）替换趋势信号
v8 的正确：在 v6 趋势框架上，用 IC 因子做"增强过滤"
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# 注意：此类继承自 v6，不修改 v6 任何逻辑
# 只在 get_signals 层面加 IC 过滤
