# data_fetcher.py
import akshare as ak
import pandas as pd
import sqlite3
from functools import lru_cache

class DataFetcher:
    @lru_cache(maxsize=100)
    def __init__(self):
        self.db_path = "data/stock_data.db"
        
    def fetch_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="hfq"
        )
        df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        }, inplace=True)
        return df

    def save_to_db(self, df, table_name):
        """存储数据到SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.close()

    def get_index_constituents(self, index_code='000300'):
        """获取指数成分股（沪深300）"""
        return ak.index_stock_cons_sina(index_code)