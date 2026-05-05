"""
统一路径配置 - 所有数据路径常量在此定义

Usage:
    from utils.paths import DAILILY_DATA_ROOT, WAVECHAN_ROOT
"""
from pathlib import Path

# ============================================================
# 主数据仓库（日线、指标、财务、股票信息）
# ============================================================
DAILY_DATA_ROOT = Path("/root/.openclaw/workspace/data/warehouse")

# ============================================================
# 波浪缠论缓存（L1极值 / L2信号 / L3参数 / 铁律验证）
# ============================================================
WAVECHAN_ROOT = Path("/data/warehouse/wavechan")
WAVECHAN_L1_ROOT = WAVECHAN_ROOT / "wavechan_l1"        # L1 极值（按年+symbol分区）
WAVECHAN_L2_ROOT = WAVECHAN_ROOT / "wavechan_cache"     # L2 信号（按年月分区，2025+）
WAVECHAN_L3_DB = WAVECHAN_ROOT / "wavechan_l3_cache.db" # L3 SQLite 参数缓存
WAVECHAN_WEEKLY = WAVECHAN_ROOT / "wavechan_weekly"      # 周线数据

# ============================================================
# 资金流数据
# ============================================================
MONEY_FLOW_ROOT = Path("/data/warehouse/money_flow")

# ============================================================
# 指数数据
# ============================================================
INDICES_ROOT = Path("/data/warehouse/indices")

# ============================================================
# 缠论信号快照（每日计算结果）
# ============================================================
CHANLUN_CACHE = Path("/data/warehouse/chanlun/cache")