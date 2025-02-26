#DataProcessor.py
import sqlite3
import pandas as pd
from ta.momentum import rsi, stoch, stoch_signal,williams_r
from ta.trend import cci,macd, macd_signal, macd_diff
import talib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TechnicalIndicatorCalculator:
    def __init__(self):
        self.db_path = "./db/stock_data.db"
        self.ensure_tables_exist()

    def ensure_tables_exist(self):
        """确保数据库表存在并包含 processed 字段"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
       # 修改 technical_indicators 表结构以包含新指标
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                date TEXT,
                symbol TEXT,
                sma_5 REAL,
                sma_10 REAL,
                sma_20 REAL,
                sma_55 REAL,
                sma_240 REAL,
                vol_ma5 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                rsi_14 REAL,         -- 新增 RSI 指标
                kdj_k REAL,          -- 新增 KDJ 指标
                kdj_d REAL,
                kdj_j REAL,
                cci_20 REAL,         -- 新增 CCI 指标
                williams_r REAL,     -- 新增 Williams%R 指标
                bb_upper REAL,       -- 新增布林带上轨
                bb_middle REAL,      -- 新增布林带中轨
                bb_lower REAL,       -- 新增布林带下轨
                PRIMARY KEY (date, symbol)
            );
        """)

        conn.commit()
        conn.close()

    def load_unprocessed_data(self, batch_size=5000):
        """加载未处理的数据"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE processed = 0 LIMIT ?",
            conn,
            params=(batch_size,),
            parse_dates=['date']
        )
        conn.close()
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
        indicators['date'] = indicators['date'].dt.strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)
        conn.close()

    def mark_as_processed(self, data):
        """标记数据为已处理"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            for row in data.itertuples(index=False):
                #logging.info(f"Marking as processed: date={row.date}, symbol={row.symbol}")
                formatted_date = row.date.strftime('%Y-%m-%d') if not pd.isna(row.date) else None
                symbol = row.symbol if not pd.isna(row.symbol) else None
                if formatted_date and symbol:
                    cursor.execute(
                        "UPDATE daily_data SET processed = 1 WHERE date = ? AND symbol = ?",
                        (formatted_date, symbol)
                    )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error marking data as processed: {e}")
        finally:
            cursor.close()
            conn.close()

    def process_unprocessed_data(self, batch_size=5000):
        """分批处理未处理的数据"""
        batch_number = 0
        while True:
            batch_number += 1
            logging.info(f"Processing batch {batch_number}...")

            # 加载未处理的数据
            df = self.load_unprocessed_data(batch_size=batch_size)
            if df.empty:
                logging.info("No more unprocessed data.")
                break

            # 计算指标
            indicators = self.calculate_indicators(df)

            # 保存到数据库
            self.save_to_database(indicators)

            # 标记数据为已处理
            self.mark_as_processed(df)

# 示例用法
if __name__ == "__main__":
    db_path = "./db/stock_data.db"
    calculator = TechnicalIndicatorCalculator()
    calculator.process_unprocessed_data(batch_size=5000)