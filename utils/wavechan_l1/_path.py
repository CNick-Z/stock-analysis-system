"""
WaveChan L1 - 路径配置
"""

from pathlib import Path
from typing import List

# ============================================================
# 根目录
# ============================================================
from utils.paths import WAVECHAN_ROOT, DAILY_DATA_ROOT

# ============================================================
# 根目录
# ============================================================
BASE_PATH = WAVECHAN_ROOT / "wavechan_l1"

# 原始日线数据根目录
DAILY_DATA_ROOT = DAILY_DATA_ROOT

# ============================================================
# 存储路径
# ============================================================

def extrema_path(symbol: str, year: int) -> Path:
    """极值点数据路径: extrema_year={year}/{symbol}.parquet"""
    return BASE_PATH / f"extrema_year={year}" / f"{symbol}.parquet"


def trend_path(symbol: str, year: int) -> Path:
    """趋势标签数据路径: trend_year={year}/{symbol}.parquet"""
    return BASE_PATH / f"trend_year={year}" / f"{symbol}.parquet"


def daily_data_path(year: int) -> Path:
    """日线数据根目录: daily_data_year={year}"""
    return DAILY_DATA_ROOT / f"daily_data_year={year}"


# ============================================================
# 目录操作
# ============================================================

def ensure_dirs(year: int) -> None:
    """确保某年的所有目录存在"""
    (BASE_PATH / f"extrema_year={year}").mkdir(parents=True, exist_ok=True)
    (BASE_PATH / f"trend_year={year}").mkdir(parents=True, exist_ok=True)


def list_symbols_in_year(year: int, data_type: str = "extrema") -> List[str]:
    """
    列出某年数据目录下的所有股票

    参数：
        data_type: 'extrema' | 'trend'
    """
    if data_type == "extrema":
        data_dir = BASE_PATH / f"extrema_year={year}"
    else:
        data_dir = BASE_PATH / f"trend_year={year}"

    if not data_dir.exists():
        return []
    return [p.stem for p in data_dir.glob("*.parquet")]


def list_available_years(data_type: str = "extrema") -> List[int]:
    """
    列出已有数据的年份

    参数：
        data_type: 'extrema' | 'trend'
    """
    if data_type == "extrema":
        pattern = "extrema_year=*"
    else:
        pattern = "trend_year=*"

    years = []
    for p in BASE_PATH.glob(pattern):
        try:
            yr = int(p.name.split("=")[1])
            years.append(yr)
        except (IndexError, ValueError):
            pass
    return sorted(years)
