#DataProcessor.py
import pandas as pd
from ta.momentum import rsi, stoch, stoch_signal,williams_r
from ta.trend import cci,macd, macd_signal, macd_diff
import talib
import logging
from utils.db_operations import *
from contextlib import contextmanager
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TechnicalIndicatorCalculator:
    def __init__(self,db_url):
        self.db_manager = DatabaseManager(db_url=db_url)
        self.db_manager.ensure_tables_exist()

    def load_unprocessed_stocks(self):
        """加载所有包含未处理的股票列表"""
        filter_conditions = {"processed": False}
        df = self.db_manager.load_data(DailyDataBase, filter_conditions=filter_conditions,distinct_column="symbol" )
        return df

    def load_full_stock_data(self,symbol,date):
        """加载指定股票的历史数据"""
        start_date = datetime.strptime(date, "%Y-%m-%d")+timedelta(days=-365)
        start_date = start_date.strftime("%Y-%m-%d")
        filter_conditions = {"symbol": symbol,
                            'date': {
                            '$gte': start_date
                            }
                        }
        df = self.db_manager.load_data(DailyDataBase, filter_conditions=filter_conditions)
        return df
    def calculate_indicators(self, df):
        """计算各种技术指标"""
        # 计算均线
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_55'] = df['close'].rolling(window=55).mean()
        df['sma_240'] = df['close'].rolling(window=240).mean()

        # 计算 MACD
        df['macd'] = macd(close=df['close'], window_slow=26, window_fast=12)
        df['macd_signal'] = macd_signal(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_histogram'] = macd_diff(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        
        # 计算 RSI（14 天）
        df['rsi_14'] = rsi(df['close'], window=14)
        
        # 计算 KDJ（9 天）
        df['k_stoch'] = stoch(df['high'], df['low'], df['close'], window=9)
        df['d_stoch'] = stoch_signal(df['high'], df['low'], df['close'], window=9, smooth_window=3)
        df['kdj_j'] = 3 * df['k_stoch'] - 2 * df['d_stoch']
        df.rename(columns={'k_stoch': 'kdj_k', 'd_stoch': 'kdj_d'}, inplace=True)
        
        # 计算 CCI（20 天）
        df['cci_20'] = cci(df['high'], df['low'], df['close'], window=20)
        
        # 计算威廉指标（14 天）
        df['williams_r'] = williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # 计算布林带（20 天）
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # 计算成交量均线
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()

        # 提取需要的列
        indicators = df[['date', 'symbol', 'sma_5', 'sma_10', 'sma_20', 'sma_55', 'sma_240', 'vol_ma5',
                         'macd', 'macd_signal', 'macd_histogram', 
                         'rsi_14', 'kdj_k', 'kdj_d', 'kdj_j', 'cci_20', 'williams_r',
                         'bb_upper', 'bb_middle', 'bb_lower']].copy()
        return indicators

    def save_to_database(self, indicators):
        """将技术指标保存到 technical_indicators 表"""
        self.db_manager.bulk_insert(TechnicalIndicatorsBase, indicators)

    def mark_as_processed(self, data):
        """标记数据为已处理"""
        # 确保 data 是 DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a Pandas DataFrame")
        # 转换为字典列表
        data_dict = data.to_dict(orient='records')
        update_fields = {'processed': 1}
        filter_fields = ['date', 'symbol']
        self.db_manager.bulk_update(DailyDataBase, data_dict, update_fields, filter_fields)

    def process_by_stock(self,date):
        """按股票逐个处理模式"""
        symbols = self.load_unprocessed_stocks()
        for symbol in symbols['symbol']:
            logging.info(f"Processing {symbol}...")
            # 加载完整历史数据
            full_data = self.load_full_stock_data(symbol,date)            
            # 计算技术指标
            indicators = self.calculate_indicators(full_data)            
            # 仅保留未处理日期的指标
            unprocessed_mask = full_data['processed'] == 0
            new_indicators = indicators[unprocessed_mask]
            if not new_indicators.empty:
                self.save_to_database(new_indicators)
                self.mark_as_processed(full_data[unprocessed_mask])
    

# 示例用法
if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db"
    calculator = TechnicalIndicatorCalculator(db_url=db_url)
    calculator.process_by_stock('2025-01-01')