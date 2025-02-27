# strategy.py
import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import linregress
from typing import Dict, Tuple
from datetime import datetime
from db_operations import DatabaseManager, DailyData, TechnicalIndicators
 

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
        self.db_url = "sqlite:///c:/db/stock_data.db"
        self.db_manager = DatabaseManager(self.db_url)
        self.db_manager.ensure_tables_exist()
        
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
        使用 load_data 方法获取预计算的技术指标数据
        返回包含以下字段的DataFrame：
        [date, symbol, ma_5, ma_10, ma_20, ma_55, ma_240, macd, macd_signal, close, volume]
        """
        # 加载技术指标数据
        tech_columns = {
            'date': 'date',
            'symbol': 'symbol',
            f'sma_{self.config["ma_windows"][0]}': 'ma_5',
            f'sma_{self.config["ma_windows"][1]}': 'ma_10',
            f'sma_{self.config["ma_windows"][2]}': 'ma_20',
            f'sma_{self.config["ma_windows"][3]}': 'ma_55',
            f'sma_{self.config["ma_windows"][4]}': 'ma_240',
            'vol_ma5': 'vol_ma5',
            'macd': 'macd',
            'macd_signal': 'macd_signal',
            'macd_histogram': 'macd_histogram',
            'rsi_14': 'rsi_14',
            'kdj_k': 'kdj_k',
            'kdj_d': 'kdj_d',
            'kdj_j': 'kdj_j',
            'cci_20': 'cci_20',
            'williams_r': 'williams_r',
            'bb_upper': 'bb_upper',
            'bb_middle': 'bb_middle',
            'bb_lower': 'bb_lower'
        }
        
        # 加载技术指标数据
        tech_df = self.db_manager.load_data(
            table=TechnicalIndicators,
            filter_conditions={
                'date': [start_date, end_date]  # 需要扩展过滤逻辑
            }
        ).rename(columns=tech_columns)
        
        # 加载行情数据
        price_df = self.db_manager.load_data(
            table=DailyData,
            filter_conditions={
                'date': [start_date, end_date]  # 需要扩展过滤逻辑
            },
            distinct_column=None,
            limit=None
        )[['date', 'symbol', 'close', 'volume']]
        
        # 合并数据集
        merged_df = pd.merge(
            tech_df,
            price_df,
            on=['date', 'symbol'],
            how='inner'
        )
        
        # 后处理
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        return merged_df.sort_values(['symbol', 'date']).reset_index(drop=True)

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
        
        # 新增超买超卖条件
        df['rsi_overbought'] = df['rsi_14'] > 70
        df['rsi_oversold'] = df['rsi_14'] < 30
        
        df['kdj_overbought'] = (df['kdj_k'] > 80) & (df['kdj_d'] > 80)
        df['kdj_oversold'] = (df['kdj_k'] < 20) & (df['kdj_d'] < 20)
        
        df['cci_overbought'] = df['cci_20'] > 100
        df['cci_oversold'] = df['cci_20'] < -100
        
        df['williams_overbought'] = df['williams_r'] > -20
        df['williams_oversold'] = df['williams_r'] < -80
        
        df['bb_upper_break'] = df['close'] > df['bb_upper']
        df['bb_lower_break'] = df['close'] < df['bb_lower']

        return df

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成完整信号（优化执行流程）
        """
        # 数据获取
        raw_data = self._fetch_precalculated_data(start_date, end_date)
        
        # 特征工程
        with_angles = self._calculate_ma_angles(raw_data)
        
        # 信号生成
        signals = self._generate_core_signals(with_angles)

        return signals
    def get_buy_signals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取买点信号"""
        signals = self.generate_features(start_date, end_date)    
        # 综合买入条件
        buy_condition = (
            signals['ma_condition'] &
            signals['angle_condition'] &
            signals['volume_condition'] &
            signals['macd_condition'] 
        )
        
        return signals[buy_condition][[
            'date', 'symbol', 'ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'vol_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14', 
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close'
        ]].assign(signal_type='buy')
        
    def get_sell_signals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取卖点信号"""
        signals = self.generate_features(start_date, end_date)
        
        # 综合卖出条件
        sell_condition = (
            (signals['ma_20'] > signals['ma_5']) &  # 短期均线下穿长期
            (
                (signals['macd_signal'] > signals['macd']) |  # MACD死叉
                (
                    (signals['close'] < signals['ma_10']) & 
                    (signals['volume'] < signals['volume'].shift(1) * 0.8)
                )
            ) |
            signals['rsi_overbought'] |    # RSI超买
            signals['kdj_overbought'] |     # KDJ超买
            signals['bb_upper_break']      # 突破布林上轨（超买信号）
        )
        
        return signals[sell_condition][[
            'date', 'symbol', 'ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'vol_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14',
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close'
        ]].assign(signal_type='sell')    
        


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
