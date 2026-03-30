#!/usr/bin/env python3
"""基本面数据提供者 - 回测和模拟盘共用"""
import pandas as pd
from typing import Optional

class FundamentalProvider:
    """提供股票基本面数据的统一接口（启动时加载，按需查询）"""
    
    def __init__(self, data_dir: str = '/root/.openclaw/workspace/data/warehouse'):
        self.df = pd.read_parquet(f'{data_dir}/financial_summary.parquet')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.index = self.df.set_index(['symbol', 'date'])
        # 预加载 stock_basic_info
        self.basic = pd.read_parquet(f'{data_dir}/stock_basic_info.parquet')
        self.basic_index = self.basic.set_index('symbol')
    
    def get(self, symbol: str, date: str) -> dict:
        """获取某股票在某日期的最新基本面（不早于date）"""
        dt = pd.to_datetime(date)
        try:
            subset = self.index.loc[symbol]
            valid = subset[subset.index <= dt]
            if valid.empty:
                return {}
            return valid.iloc[-1].to_dict()
        except KeyError:
            return {}
    
    def get_basic(self, symbol: str) -> dict:
        """获取股票基本信息（行业、市值等）"""
        try:
            return self.basic_index.loc[symbol].to_dict()
        except KeyError:
            return {}
    
    def has_data(self, symbol: str, date: str) -> bool:
        """检查某股票某日期是否有基本面数据"""
        return len(self.get(symbol, date)) > 0
