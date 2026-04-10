"""
WaveChan L1 缓存
=================
局部显著极值点（ZigZag）+ 趋势判断（UP/DOWN/NEUTRAL）

存储结构：
    /data/warehouse/wavechan_l1/extrema_year={year}/{symbol}.parquet
    /data/warehouse/wavechan_l1/trend_year={year}/{symbol}.parquet

快速开始：
    from utils.wavechan_l1 import WaveChanL1Manager

    mgr = WaveChanL1Manager()

    # 构建 2024 年 L1
    mgr.build_year(2024)

    # 查询极值点
    extrema = mgr.get_extrema('600368', 2024)
    print(extrema)

    # 查询趋势序列
    trend_df = mgr.get_trend('600368', 2024)
    print(trend_df)

    # 当前趋势
    current = mgr.get_current_trend('600368', 2024)
    print(current)  # 'UP' / 'DOWN' / 'NEUTRAL'

    # 状态
    print(mgr.status())
"""

from .manager import WaveChanL1Manager
from . import zigzag
from . import trend
from . import _path

__all__ = [
    "WaveChanL1Manager",
    "zigzag",
    "trend",
    "_path",
]
