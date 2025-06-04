import pandas as pd
from utils.db_operations import DatabaseManager, DailyDataBase
import concurrent.futures
import os
from functools import lru_cache
import logging
from pathlib import Path
# 移除 pytdx 导入
# from pytdx.reader import TdxDailyBarReader
# 导入自定义的 TdxDayReader
from utils.tdx_reader import TdxDayReader  # 确保两个文件在同一目录

# 配置日志
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

class DataFetcher:
    def __init__(self, db_url):
        self.db_manager = DatabaseManager(db_url=db_url)
        self.db_manager.ensure_tables_exist()
        self.existing_data = {}  # 用于存储已存在的数据
        # 设置通达信数据目录（可根据需要修改）
        self.tdx_base_path = "E:/tdx/vipdoc"
        # 初始化自定义的TdxDayReader
        self.tdx_reader = TdxDayReader(root_dir=self.tdx_base_path)

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
    
    def get_tdx_file_path(self, symbol):
        """根据股票代码获取通达信数据文件路径"""
        # 确定市场前缀
        if symbol.startswith('6'):
            market = 'sh'
        elif symbol.startswith(('0', '3')):
            market = 'sz'
        elif symbol.startswith('4') or symbol.startswith('8'):
            market = 'bj'
        else:
            market = 'sh'  # 默认上海市场
        
        # 构建文件路径
        file_path = os.path.join(
            self.tdx_base_path, 
            market, 
            'lday', 
            f"{market}{symbol}.day"
        )
        
        if not os.path.exists(file_path):
            logging.warning(f"通达信数据文件不存在: {file_path}")
            return None
        
        return file_path

    @lru_cache(maxsize=100)
    def fetch_daily_data(self, symbol, start_date, end_date):
        """从本地通达信获取日线数据（使用自定义TdxDayReader）"""
        try:
            # 使用自定义的TdxDayReader读取数据
            df = self.tdx_reader.read_by_code(stock_code=symbol)
            
            # 检查DataFrame是否为空
            if df.empty:
                logging.warning(f"本地无数据: {symbol}")
                return pd.DataFrame()
            
            # 确保date列是datetime类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 添加股票代码列
            df['symbol'] = symbol
            
            # 添加其他必要字段（使用默认值）
            # 注意：自定义读取器已经包含amount字段（成交额），不需要再计算
            df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100  # 计算振幅
            df['change_pct'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100  # 计算涨跌幅
            df['change_amount'] = df['close'] - df['close'].shift(1)  # 计算涨跌额
            df['turnover_rate'] = 0  # 换手率（本地数据无此字段，设为0）
            
            # 重命名列以匹配format_columns的期望
            df.rename(columns={
                'date': '日期',
                'symbol': '股票代码',
                'open': '开盘',
                'high': '最高',
                'low': '最低',
                'close': '收盘',
                'vol': '成交量',  # 注意：自定义读取器使用'vol'作为成交量列名
                'amount': '成交额',
                'amplitude': '振幅',
                'change_pct': '涨跌幅',
                'change_amount': '涨跌额',
                'turnover_rate': '换手率'
            }, inplace=True)
            
            # 过滤日期范围
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df = df[(df['日期'] >= start_date_dt) & (df['日期'] <= end_date_dt)]
            
            # 确保数据按日期排序
            df.sort_values('日期', inplace=True)
            
            return df
        except Exception as e:
            logging.error(f"读取本地数据错误 {symbol}: {e}")
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
        if existing_data.empty:
            return {}
        existing_data_dict = existing_data.groupby('symbol')['date'].apply(set).to_dict()
        
        return existing_data_dict

    def process_symbol(self, symbol, start_date, end_date, table_name):
        """处理单个股票的逻辑"""
        logging.info(f"处理 {symbol}...")
        existing_dates = self.existing_data.get(symbol, set())
        df = self.fetch_daily_data(symbol, start_date, end_date)
        
        # 如果没有数据，跳过
        if df.empty:
            logging.info(f"{symbol} 无数据, 跳过...")
            return
        
        # 格式化列名
        df = self.format_columns(df)
        
        # 确保 'date' 列是字符串格式
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
            result = self.db_manager.bulk_insert(DailyDataBase, new_data)
            if result == 1:
                logging.info(f"{symbol} 数据保存成功.")
        else:
            logging.info(f"{symbol} 数据已存在.")

    def fetch_and_save_all_data(self, start_date, end_date, table_name="daily_data"):
        """使用多线程下载所有股票的数据"""
        logging.info("获取所有股票代码...")
        
        # 直接从通达信目录获取所有股票代码
        symbols = self.get_all_symbols_from_tdx()
        
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
                    logging.error(f"处理错误: {e}")

    def get_all_symbols_from_tdx(self):
        """
        从通达信目录获取所有股票代码，并过滤掉非A股数据
        
        过滤规则：
        沪市（sh）: 只保留以6开头的代码（包括600、601、603、605、688等A股）
        深市（sz）: 只保留以0或3开头的代码（主板000-004开头，创业板300-301开头）
        
        排除以下类型：
        - 基金（沪市5开头，深市1开头）
        - 债券（沪市11开头，深市10开头）
        - 北交所股票（43、83、87、88开头）
        - 其他非A股品种
        """
        symbols = []
        markets = ['sh', 'sz']
        # A股股票代码前缀规则
        valid_prefixes = {
        'sh': ['6'],       # 沪市A股：6开头（包括600、601、603、605、688等）
        'sz': ['0', '3']    # 深市A股：0开头（主板）和3开头（创业板）
        }
        
        for market in markets:
            market_path = os.path.join(self.tdx_base_path, market, 'lday')
            if os.path.exists(market_path):
                for file in Path(market_path).glob('*.day'):
                    # 提取股票代码（移除市场前缀）
                    symbol = file.stem[2:]
                    
                    # 根据市场过滤有效A股代码
                    if symbol[0] in valid_prefixes[market]:
                        symbols.append(symbol)
        
        logging.info(f"找到 {len(symbols)} 个股票代码")
        return symbols

# 示例用法
if __name__ == "__main__":
    # 使用 Copy-on-Write 模式
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db"  
    fetcher = DataFetcher(db_url)
    start_date = "20140101"
    end_date = "20141231"
    fetcher.fetch_and_save_all_data(start_date, end_date)