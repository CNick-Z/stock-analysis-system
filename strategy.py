# strategy.py
import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import linregress
from typing import Dict, Tuple
from datetime import datetime

class BaseStrategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
        

class TDXStrategy(BaseStrategy):
    def calculate_technical_indicators(self):
        super().calculate_technical_indicators()
        
        # 计算更多均线

        self.data['MA10'] = self.data['close'].rolling(10).mean()       
        self.data['MA55'] = self.data['close'].rolling(55).mean()
        self.data['MA240'] = self.data['close'].rolling(240).mean()
        
        # 使用滑动窗口计算均线斜率（优化后）
        def calculate_angle(series, window=5):
            def _slope(x):
                return linregress(np.arange(len(x)), x).slope
            return series.rolling(window).apply(_slope).pipe(np.degrees)
        
        self.data['angle_MA10'] = calculate_angle(self.data['MA10'], window=5)
        self.data['angle_MA20'] = calculate_angle(self.data['MA20'], window=5)
        self.data['angle_MA55'] = calculate_angle(self.data['MA55'], window=5)
        self.data['angle_MA240'] = calculate_angle(self.data['MA240'], window=5)
        
        # 成交量均线
        self.data['VOL_MA5'] = self.data['volume'].rolling(5).mean()
        
    def generate_signals(self):
        self.calculate_technical_indicators()
        
        # 拆分条件变量（提升可读性）
        ma_condition = (
            (self.data['MA5'] > self.data['MA10']) &
            (self.data['MA10'] < self.data['MA20'])
        )
        
        angle_condition = (
            (self.data['angle_MA10'] > 0) &
            (self.data['angle_MA240'].between(-20, 20)) &
            (self.data['angle_MA20'].between(-40, 35))
        )
        
        volume_condition = (
            (self.data['volume'] > self.data['volume'].shift(1) * 1.5) |
            (self.data['volume'] > self.data['VOL_MA5'] * 1.2)
        )
        
        # 综合买入条件
        self.signals['enter_long'] = (
            ma_condition &
            angle_condition &
            volume_condition &
            (self.data['MACD'] < 0) &
            (self.data['MACD'] > self.data['MACDsignal'])
        )
        
        # 卖出条件修正括号
        self.signals['exit_long'] = (
            (self.data['MA20'] > self.data['MA5']) &
            (
                (self.data['MACDsignal'] > self.data['MACD']) |
                (
                    (self.data['close'] < self.data['MA10']) & 
                    (self.data['volume'] < self.data['volume'].shift(1) * 0.8)
                )
            )
        )
        
        return self.signals

class EnhancedTDXStrategy:
    def __init__(self, 
                 config: Dict = {
                     'ma_windows': (5, 10, 20, 55, 240),
                     'angle_window': 5,
                     'volume_ma_window': 5,
                     'macd_params': (12, 26, 9)
                 }):
        """
        深度优化后的策略类，保留原有核心逻辑
        
        参数说明：
        - ma_windows: 预计算的均线周期
        - angle_window: 均线角度计算窗口
        - volume_ma_window: 成交量均线窗口
        - macd_params: MACD参数（short, long, signal）
        """
        self.config = config
        self.db_path = "./db/stock_data.db"
        
        # 预校验参数
        self._validate_config()

    def _validate_config(self):
        """参数有效性验证"""
        if not all(isinstance(x, int) and x > 0 for x in self.config['ma_windows']):
            raise ValueError("均线周期必须为正整数")
        if len(self.config['macd_params']) != 3:
            raise ValueError("MACD参数需要三个值(short, long, signal)")

    def _fetch_precalculated_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据库获取预计算的技术指标数据
        返回包含以下字段的DataFrame：
        [date, symbol, ma_5, ma_10, ma_20, ma_55, ma_240, macd, macd_signal, close, volume]
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = f"""
                SELECT 
                    ti.date,
                    ti.symbol,
                    ti.sma_{self.config['ma_windows'][0]} as ma_5,
                    ti.sma_{self.config['ma_windows'][1]} as ma_10,
                    ti.sma_{self.config['ma_windows'][2]} as ma_20,
                    ti.sma_{self.config['ma_windows'][3]} as ma_55,
                    ti.sma_{self.config['ma_windows'][4]} as ma_240,
                    ti.vol_ma5,
                    ti.macd,
                    ti.macd_signal,
                    ti.macd_histogram,
                    dd.close,
                    dd.volume
                FROM technical_indicators ti
                JOIN daily_data dd 
                    ON ti.date = dd.date 
                    AND ti.symbol = dd.symbol
                WHERE ti.date BETWEEN '{start_date}' AND '{end_date}'
            """
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values(['symbol', 'date']).reset_index(drop=True)
        finally:
            conn.close()

    def _calculate_ma_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线角度（基于预计算的MA值）
        保留原有角度计算逻辑
        """
        def calculate_angle(series, window):
            return series.rolling(window).apply(
                lambda x: np.degrees(linregress(np.arange(len(x)), x).slope),
                raw=False
            )
        
        # 计算各均线角度
        for ma in ['ma_10', 'ma_20', 'ma_55', 'ma_240']:
            df[f'angle_{ma}'] = calculate_angle(df[ma], self.config['angle_window'])
        
        return df


    def _generate_core_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成核心交易信号（保持原有信号逻辑不变）
        """
        # 均线条件
        df['ma_condition'] = (
            (df['ma_5'] > df['ma_10']) & 
            (df['ma_10'] < df['ma_20'])
        )
        
        # 角度条件（优化边界检查）
        df['angle_condition'] = (
            (df['angle_ma_10'] > 0) &
            df['angle_ma_240'].between(-20, 20) &
            df['angle_ma_20'].between(-40, 35)
        )
        
        # 成交量条件（保留原有逻辑）
        df['volume_condition'] = (
            (df['volume'] > df['volume'].shift(1) * 1.5) |
            (df['volume'] > df['vol_ma5'] * 1.2)
        )
        
        # MACD条件（使用预计算值）
        df['macd_condition'] = (
            (df['macd'] < 0) & 
            (df['macd'] > df['macd_signal'])
        )
        
        return df

    def generate_signals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成完整信号（优化执行流程）
        """
        # 数据获取
        raw_data = self._fetch_precalculated_data(start_date, end_date)
        
        # 特征工程
        with_angles = self._calculate_ma_angles(raw_data)
        
        # 信号生成
        signals = self._generate_core_signals(with_angles)
        
        # 综合信号
        signals['enter_long'] = (
            signals['ma_condition'] &
            signals['angle_condition'] &
            signals['volume_condition'] &
            signals['macd_condition']
        )
        
        # 卖出信号（优化逻辑结构）
        signals['exit_long'] = (
            (signals['ma_20'] > signals['ma_5']) &
            (
                (signals['macd_signal'] > signals['macd']) |
                (
                    (signals['close'] < signals['ma_10']) & 
                    (signals['volume'] < signals['volume'].shift(1) * 0.8)
                )
            )
        )
        filtered_signals = signals[
            (signals['enter_long']) |  # 有买入信号
            (signals['exit_long'])     # 有卖出信号
            ].copy()
        
        return filtered_signals[['date', 'symbol', 'enter_long', 'exit_long', 
                        'ma_5', 'ma_10', 'ma_20', 'angle_ma_10', 
                        'vol_ma5', 'macd','macd_signal','volume']]

    def get_trading_advice(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易建议（新增方法）
        """
        return (
            signals.groupby('symbol')
            .apply(lambda x: x[x['enter_long']].iloc[-1] if any(x['enter_long']) else pd.Series())
            .reset_index(drop=True)
            .sort_values('ma_5', ascending=False)
            .head(5)
        )

# 使用示例
if __name__ == "__main__":
    strategy = EnhancedTDXStrategy()
    
    # 获取最近30天信号
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
    
    signals = strategy.generate_signals(start_date, end_date)
    top_picks = strategy.get_trading_advice(signals)
    
    print("近期推荐标的：")
    print(top_picks[['symbol', 'ma_5', 'angle_ma_10', 'macd']])