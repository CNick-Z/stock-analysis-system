import akshare as ak
import pandas as pd
import sqlite3
import os
from functools import lru_cache
import concurrent.futures
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataFetcher:
    def __init__(self):
        self.db_path = "./db/stock_data.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.ensure_table_exists()
        self.existing_data = {}  # 用于存储已存在的数据

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
                #logging.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            # 确保 date 列是 datetime 类型
            df['日期'] = pd.to_datetime(df['日期'])
            # 重命名列名以匹配数据库字段
            df.rename(columns={
                '日期': 'date',
                '股票代码': 'symbol',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            }, inplace=True)
            # 确保数据按日期排序
            df.sort_values('date', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def save_to_db(self, df, symbol, table_name, conn=None, max_attempts=3):
        """存储数据到SQLite数据库"""
        if df.empty:
            return
        
        # 确保 symbol 字段存在
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        df.reset_index(drop=True, inplace=True)
        
        # 确保 date 和 symbol 列是字符串类型
        df.loc[:, 'symbol'] = df['symbol'].astype(str)
        
        # 生成插入语句
        columns = df.columns.tolist()
        placeholders = ', '.join(['?'] * len(columns))
        sql = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        data = df.values.tolist()
        
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            use_external_conn = False
        else:
            use_external_conn = True
        
        attempts = 0
        success = False
        while attempts < max_attempts:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.cursor()
                cursor.executemany(sql, data)
                if not use_external_conn:
                    conn.commit()
                success = True
                break
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    attempts += 1
                    time.sleep(2)
                else:
                    logging.error(f"Database operational error: {e}")
                    raise
            except sqlite3.Error as e:
                logging.error(f"Database error: {e}")
                logging.error(f"Failed to execute SQL: {sql}")
                logging.error(f"Data: {data}")
            finally:
                if not use_external_conn:
                    conn.close()
        
        if not success:
            logging.error(f"Failed to save data for {symbol} after {max_attempts} attempts")

    def ensure_table_exists(self, table_name="daily_data"):
        """确保表存在"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                symbol TEXT,
                processed BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (date, symbol)
            )
        """)
        conn.commit()
        conn.close()

    def load_existing_data(self, start_date, end_date, table_name="daily_data"):
        """一次性加载所有已存在的数据到内存"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        existing_data = pd.read_sql_query(
            f"SELECT date, symbol FROM {table_name} WHERE date >= ? AND date <= ?",
            conn,
            params=[start_date, end_date],
            parse_dates=['date']  # 将 date 列解析为 datetime 类型
        )
        conn.close()
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
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except Exception as e:
                logging.error(f"Failed to convert date column to datetime for {symbol}: {e}")
                return
        
        # 将日期格式转换为字符串
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        new_data = df[~df['date'].isin(existing_dates)]
        
        if not new_data.empty:
            # 深拷贝以避免 SettingWithCopyWarning
            new_data = new_data.copy()
            # 确保 symbol 字段存在
            new_data.loc[:, 'symbol'] = symbol
            # 保存到数据库
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            try:
                self.save_to_db(new_data, symbol, table_name, conn)
                conn.commit()
                logging.info(f"All data for {symbol} save sessfully.")
            except Exception as e:
                logging.error(f"Failed to save data for {symbol}: {e}")
                logging.info(f"the data is {new_data} .")
            conn.close()
        else:
            logging.info(f"All data for {symbol} already exists in the database.")

    def fetch_and_save_all_data(self, start_date, end_date, table_name="daily_data"):
        """使用多线程下载所有股票的数据"""
        logging.info("Fetching all stock and fund symbols...")
        stock_list = ak.stock_zh_a_spot_em()
        symbols = stock_list['代码'].tolist()
        
        # 一次性加载所有已存在的数据
        self.existing_data = self.load_existing_data(start_date, end_date, table_name)
        
        # 使用多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
    fetcher = DataFetcher()
    start_date = "20240101"
    end_date = "20250223"
    table_name = "daily_data"
    fetcher.fetch_and_save_all_data(start_date, end_date, table_name)