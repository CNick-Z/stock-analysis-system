"""
Parquet 数据库集成模块 - 替代 SQLite

提供与 DatabaseIntegrator 相同的接口，使用 Parquet 数据仓库
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Parquet 数据仓库路径
WAREHOUSE_PATH = Path("/root/.openclaw/workspace/data/warehouse")


class ParquetDatabaseIntegrator:
    """
    Parquet 数据仓库集成器
    
    提供与原 SQLite DatabaseIntegrator 相同的数据接口
    """
    
    def __init__(self, db_path: str = None):
        """
        初始化 Parquet 数据库集成器
        
        Args:
            db_path: 保留参数（兼容性），实际使用 WAREHOUSE_PATH，不再用于数据库连接
        """
        self.db_path = db_path  # 仅保留用于兼容性，不实际使用
        self.warehouse_path = WAREHOUSE_PATH
        self._ensure_warehouse_exists()
    
    def _ensure_warehouse_exists(self):
        """确保数据仓库存在"""
        if not self.warehouse_path.exists():
            raise FileNotFoundError(f"数据仓库不存在: {self.warehouse_path}")
        
        # 检查必要的目录
        daily_data_dir = self.warehouse_path / "daily_data_year=2024"
        if not daily_data_dir.exists():
            raise FileNotFoundError(f"日线数据目录不存在: {daily_data_dir}")
        
        logger.info(f"✅ Parquet 数据仓库初始化成功: {self.warehouse_path}")
    
    def fetch_trading_dates_and_price_matrix(self, start_date: str, end_date: str) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
        """
        一次性查询并生成交易日期和价格矩阵
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            trading_dates: 交易日期索引
            price_matrix: 价格矩阵，index=date, columns=(symbol, 'open'/'close')
        """
        logger.info(f"加载日线数据: {start_date} ~ {end_date}")
        
        # 计算需要加载的年份
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        # 读取所有需要的年份数据
        dfs = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.warehouse_path / f"daily_data_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
        
        if not dfs:
            raise ValueError(f"没有找到 {start_year}~{end_year} 年的数据")
        
        data = pd.concat(dfs, ignore_index=True)
        
        # 过滤日期范围
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        if data.empty:
            logger.warning(f"指定日期范围内没有数据: {start_date} ~ {end_date}")
            return pd.DatetimeIndex([]), pd.DataFrame()
        
        # 生成交易日期索引
        trading_dates = pd.to_datetime(data['date']).sort_values().unique()
        
        # 生成价格矩阵 (date x symbol x field)
        # pivot 后 columns 是 MultiIndex: (field, symbol)
        # v2: 增加 amount, volume 字段支持资金流计算
        price_matrix = data.pivot_table(
            index='date', 
            columns='symbol', 
            values=['open', 'close', 'amount', 'volume']
        )
        
        # 不需要重新组织，保持原格式 (field, symbol)
        
        logger.info(f"加载了 {len(trading_dates)} 个交易日, {len(price_matrix.columns)//2} 只股票")
        
        return trading_dates, price_matrix
    
    def fetch_daily_data(self, start_date: str, end_date: str, columns: List[str] = None) -> pd.DataFrame:
        """
        获取日线数据（兼容原接口）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要获取的列名
        """
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        dfs = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.warehouse_path / f"daily_data_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        data = pd.concat(dfs, ignore_index=True)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        if columns:
            available_cols = [c for c in columns if c in data.columns]
            data = data[available_cols]
        
        return data
    
    def fetch_technical_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取技术指标数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        dfs = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.warehouse_path / f"technical_indicators_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        data = pd.concat(dfs, ignore_index=True)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        return data
    
    def get_stock_basic_info(self) -> pd.DataFrame:
        """获取股票基本信息"""
        info_file = self.warehouse_path / "stock_basic_info.parquet"
        if not info_file.exists():
            return pd.DataFrame()
        
        return pd.read_parquet(info_file)
    
    def get_latest_data_date(self) -> str:
        """获取最新数据日期"""
        # 查找最新的年份目录
        max_year = 2025
        for year in range(2025, 2000, -1):
            parquet_file = self.warehouse_path / f"daily_data_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                return df['date'].max()
        
        return None
    
    def load_data(self, table_class=None, filter_conditions=None, columns=None, distinct_column=None, limit=None):
        """
        兼容原接口的 load_data 方法
        
        Args:
            table_class: 表类型（用于识别是日线还是技术指标）
            filter_conditions: 过滤条件 {'date': {'$between': [start, end]}}
            columns: 需要获取的列名
            distinct_column: 去重列
            limit: 限制数量
        """
        table_class_str = str(table_class) if table_class else ''
        
        # 根据 table_class 判断数据类型
        if 'TechnicalIndicators' in table_class_str:
            # 加载技术指标
            return self._load_technical_indicators(filter_conditions, columns)
        elif 'StockBasicInfo' in table_class_str:
            # 加载股票基本信息
            return self._load_stock_basic_info(filter_conditions, columns)
        else:
            # 加载日线数据
            return self._load_daily_data(filter_conditions, columns)
    
    def _load_stock_basic_info(self, filter_conditions, columns):
        """加载股票基本信息"""
        info_file = self.warehouse_path / "stock_basic_info.parquet"
        if not info_file.exists():
            logger.warning(f"股票基本信息文件不存在: {info_file}")
            return pd.DataFrame()
        
        data = pd.read_parquet(info_file)
        
        # 过滤 ST 股票
        if 'name' in data.columns:
            data = data[~data['name'].str.contains('ST', na=False)]
        
        if columns:
            available_cols = [c for c in columns if c in data.columns]
            data = data[available_cols]
        
        return data
    
    def _load_daily_data(self, filter_conditions, columns):
        """加载日线数据"""
        if filter_conditions and 'date' in filter_conditions:
            date_range = filter_conditions['date']
            if '$between' in date_range:
                start_date, end_date = date_range['$between']
            else:
                start_date, end_date = date_range.get('$gte', '2000-01-01'), date_range.get('$lte', '2025-12-31')
        else:
            start_date, end_date = '2000-01-01', '2025-12-31'
        
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        dfs = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.warehouse_path / f"daily_data_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        data = pd.concat(dfs, ignore_index=True)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        if columns:
            available_cols = [c for c in columns if c in data.columns]
            data = data[available_cols]
        
        return data
    
    def _load_technical_indicators(self, filter_conditions, columns):
        """加载技术指标数据"""
        if filter_conditions and 'date' in filter_conditions:
            date_range = filter_conditions['date']
            if '$between' in date_range:
                start_date, end_date = date_range['$between']
            else:
                start_date, end_date = date_range.get('$gte', '2000-01-01'), date_range.get('$lte', '2025-12-31')
        else:
            start_date, end_date = '2000-01-01', '2025-12-31'
        
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        dfs = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.warehouse_path / f"technical_indicators_year={year}" / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        data = pd.concat(dfs, ignore_index=True)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # Parquet 技术指标使用 sma_X 格式，策略需要 ma_X 格式
        # 添加兼容列
        rename_map = {}
        for col in data.columns:
            if col.startswith('sma_'):
                rename_map[col] = col.replace('sma_', 'ma_')
        data = data.rename(columns=rename_map)
        
        if columns:
            available_cols = [c for c in columns if c in data.columns]
            # 同时保留可能的 sma_ 列（会被映射）
            available_cols.extend([c for c in data.columns if c.startswith('sma_')])
            data = data[list(set(available_cols))]
        
        return data


# 兼容性别名
DatabaseIntegrator = ParquetDatabaseIntegrator
