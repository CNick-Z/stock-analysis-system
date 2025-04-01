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

    def load_all_data(self,date):
        """加载指定股票的历史数据"""
        start_date = datetime.strptime(date, "%Y%m%d")+timedelta(days=-365)
        start_date = start_date.strftime("%Y-%m-%d")
        filter_conditions = {
                            'date': {
                            '$gte': start_date
                            }
                        }
        df = self.db_manager.load_data(DailyDataBase, filter_conditions=filter_conditions)
        return df
    def calculate_indicators(self, df):
        """使用TA-Lib优化计算各种技术指标"""
        # 确保数据按日期升序排列
        df = df.sort_values('date')
        
        # 计算均线 - 使用TA-Lib
        df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_55'] = talib.SMA(df['close'], timeperiod=55)
        df['sma_240'] = talib.SMA(df['close'], timeperiod=240)
        
        # 计算MACD - 使用TA-Lib
        df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 计算RSI - 使用TA-Lib
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        
        # 计算KDJ - 使用TA-Lib的STOCH
        df['kdj_k'], df['kdj_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0)
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 计算CCI - 使用TA-Lib
        df['cci_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # 计算威廉指标 - 使用TA-Lib的WILLR
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 计算布林带 - 使用TA-Lib
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 计算成交量均线 - 使用TA-Lib
        df['vol_ma5'] = talib.SMA(df['volume'], timeperiod=5)
        
        # 提取需要的列
        indicators = df[['date', 'symbol', 'sma_5', 'sma_10', 'sma_20', 'sma_55', 'sma_240', 'vol_ma5',
                        'macd', 'macd_signal', 'macd_histogram', 
                        'rsi_14', 'kdj_k', 'kdj_d', 'kdj_j', 'cci_20', 'williams_r',
                        'bb_upper', 'bb_middle', 'bb_lower','processed']].copy()
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

    def process_all_data(self, date):
        """批量处理所有数据的优化版本"""
        # 1. 一次性加载所有需要的数据
        logging.info("Loading all historical data...")
        all_data = self.load_all_data(date)
        
        if all_data.empty:
            logging.info("No data found.")
            return
            
        # 2. 按股票分组计算所有技术指标
        logging.info("Calculating indicators for all stocks...")
        indicators_list = []
        for symbol, group in all_data.groupby('symbol'):
            indicators = self.calculate_indicators(group)
            indicators_list.append(indicators)
        
        all_indicators = pd.concat(indicators_list)
        
        # 3. 筛选出未处理的数据
        unprocessed_indicators = all_indicators[all_indicators['processed']==0]
        
        if not unprocessed_indicators.empty:
            # 4. 批量保存技术指标
            logging.info(f"Saving {len(unprocessed_indicators)} indicators to database...")
            self.save_to_database(unprocessed_indicators)
            
            # 5. 批量标记为已处理
            logging.info("Marking data as processed...")
            unprocessed_data = all_data[all_data['processed'] == 0]
            self.mark_as_processed(unprocessed_data)
        
        logging.info("Processing completed. Total processed: %d", len(unprocessed_indicators))
    

# 示例用法
if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db"
    calculator = TechnicalIndicatorCalculator(db_url=db_url)
    calculator.process_all_data('20250101')