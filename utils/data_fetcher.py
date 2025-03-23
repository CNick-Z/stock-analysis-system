import akshare as ak
import pandas as pd
from utils.db_operations import DatabaseManager, DailyDataBase
import concurrent.futures
import os
from functools import lru_cache
import logging
import time

# 配置日志
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

class DataFetcher:
    def __init__(self,db_url):
        self.db_manager = DatabaseManager(db_url=db_url)
        self.db_manager.ensure_tables_exist()
        self.existing_data = {}  # 用于存储已存在的数据

    @staticmethod
    def format_columns(df):
        """格式化列名以匹配数据库字段"""
        column_mapping = {
            '日期': 'date',
            '股票代码': 'symbol',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }
        df.rename(columns=column_mapping, inplace=True)
        return df
    @lru_cache(maxsize=100)
    def fetch_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            # 检查 DataFrame 是否为空
            if df.empty:
                logging.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            # 确保 date 列是 datetime 类型
            df['日期'] = pd.to_datetime(df['日期'])
            # 格式化列名
            df = self.format_columns(df)
            # 确保数据按日期排序
            df.sort_values('date', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def load_existing_data(self, start_date, end_date, table_name="daily_data"):
        """使用load_data方法加载已存在数据"""
        start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        # 构建过滤条件
        filter_conditions = {
            'date': {
                '$between': [start_date, end_date]
            }
        }
        # 指定需要加载的列
        columns = ['symbol', 'date']
        # 使用load_data方法获取数据
        existing_data = self.db_manager.load_data(
            table_class=DailyDataBase,
            filter_conditions=filter_conditions,
            columns=columns
        )
        # 转换为字典格式
        existing_data_dict = existing_data.groupby('symbol')['date'].apply(set).to_dict()
        
        return existing_data_dict

    def process_symbol(self, symbol, start_date, end_date, table_name):
        """处理单个股票的逻辑"""
        logging.info(f"Processing {symbol}...")
        existing_dates = self.existing_data.get(symbol, set())
        df = self.fetch_daily_data(symbol, start_date, end_date)
        
        # 如果没有数据，跳过
        if df.empty:
            logging.info(f"No data available for {symbol}, skipping...")
            return
        
        # 确保 'date' 列是 datetime 类型
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        new_data = df[~df['date'].isin(existing_dates)]
        
        if not new_data.empty:
            # 深拷贝以避免 SettingWithCopyWarning
            new_data = new_data.copy()
            # 确保 symbol 字段存在
            new_data.loc[:, 'symbol'] = symbol
            # 保存到数据库
            # 确保动态表存在
            years = pd.to_datetime(df['date']).dt.year.unique()
            for year in years:
                self.db_manager.ensure_dynamic_table_exists(DailyDataBase, year)
            result=self.db_manager.bulk_insert(DailyDataBase, new_data)
            if result==1:
                logging.info(f"All data for {symbol} saved successfully.")
        else:
            logging.info(f"All data for {symbol} already exists in the database.")

    def fetch_and_save_all_data(self, start_date, end_date, table_name="daily_data"):
        """使用多线程下载所有股票的数据"""
        logging.info("Fetching all stock and fund symbols...")
        stock_list = ak.stock_zh_a_spot_em()
        #symbols = ~stock_list['代码'].str.startswith("8").tolist()
        symbols = stock_list[~stock_list['代码'].str.startswith('8','4')]['代码'].tolist()
        # 一次性加载所有已存在的数据
        self.existing_data = self.load_existing_data(start_date, end_date, table_name)
        
        # 使用多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.process_symbol, symbol, start_date, end_date, table_name)
                for symbol in symbols
            ]
            # 捕获任何可能的异常
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing symbol: {e}")

# 示例用法
if __name__ == "__main__":
    # 使用 Copy-on-Write 模式
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db"  
    fetcher = DataFetcher(db_url)
    start_date = "20140101"
    end_date = "20141231"
    fetcher.fetch_and_save_all_data(start_date, end_date)