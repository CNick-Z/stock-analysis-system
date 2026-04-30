#!/usr/bin/env python3
"""
2025年回测 - 使用 V4 增强配置
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/.openclaw/workspace/projects/stock-analysis-system")
sys.path.insert(0, str(PROJECT_ROOT)

# 导入 V4 配置
from strategies.wavechan.v3_l2_cache.wavechan_config_v4 import WaveChanConfigV4

# V4 配置 - 全部开启
V4_CONFIG = WaveChanConfigV4(
    enable_b_trap_alert=True,
    use_extended_fib=True,
    use_tight_w2_range=True,
    use_fib_382_stop=True,
    enable_w5_divergence_penalty=True,
)

print("=== 2025年回测 (V4增强) ===")
print(f"V4 CONFIG: {V4_CONFIG}")
print()

# 导入策略数据
from utils.data_loader import load_strategy_data
import pandas as pd

# 加载 2025 年数据
print("加载 2025 年数据...")
df = load_strategy_data(years=[2025])
print(f"总行数: {len(df)}")

# 测试 V4 函数
from strategies.wavechan.v3_l2_cache.wavechan_v4 import compute_extended_fib, detect_b_trap, calc_w2_range

# 示例：测试斐波那契多档
fib = compute_extended_fib(10.0, 12.0, 11.0)
print(f"\n斐波那契多档测试: {fib}")

# 测试 W2 区间
r = calc_w2_range(use_tight=V4_CONFIG.use_tight_w2_range)
print(f"W2区间: {r}")

# 测试 B 浪检测
trap = detect_b_trap('w1_formed', w1_is_impulse=False, w1_internal_segments=3, volume_trend='shrinking', config_enabled=V4_CONFIG.enable_b_trap_alert)
print(f"B浪陷阱: {trap}")

print("\n✅ V4配置测试完成")