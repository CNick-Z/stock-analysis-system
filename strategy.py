# strategy.py
import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, Tuple
from datetime import datetime
from db_operations import *
 
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
            'vol_ma5': 'volume_ma5',
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
                'date': {'$between':[start_date, end_date]}  # 需要扩展过滤逻辑
            }
        ).rename(columns=tech_columns)
        
        # 加载行情数据
        price_df = self.db_manager.load_data(
            table=DailyData,
            filter_conditions={
                'date': {'$between':[start_date, end_date]}  # 需要扩展过滤逻辑
            },
            columns=['date', 'symbol','high','low','volume','amount','open','close'],
            distinct_column=None,
            limit=None
        )
        
        #加载基础数据
        info_df = self.db_manager.load_data(
            table=StockBasicInfo,
            columns=['symbol', 'name', 'total_shares', 'industry']
        )

        # 合并数据集
        merged_df = pd.merge(
            tech_df,
            price_df,
            on=['date', 'symbol'],
            how='inner'
        )

        merged_df = pd.merge(
            merged_df,
            info_df,
            on=['symbol'],
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

    def _calculate_xvl(self,row):
        if row['close'] > row['open']:
            # 阳线
            flow_in = row['QJJ'] * (row['high'] - row['low'])
            flow_out = -row['QJJ'] * (row['high'] - row['close'] + row['open'] - row['low'])
        elif row['close'] < row['open']:
            # 阴线
            flow_in = row['QJJ'] * (row['high'] - row['open'] + row['close'] - row['low'])
            flow_out = -row['QJJ'] * (row['high'] - row['low'])
        else:
            # 平线
            flow_in = row['volume'] / 2
            flow_out = -row['volume'] / 2

        return flow_in + flow_out


    def _calculate_money_flow_indicators(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        计算资金流向相关指标
        """
        # 1. 计算金（单位成交量的资金量）
        df['金'] = df['amount'] / df['volume']

        # 2. 计算PJJ（加权平均价格）
        df['PJJ'] = (df['high'] + df['low'] + df['close'] * 2) / 4
        df['PJJ'] = df['PJJ'].ewm(alpha=0.9, adjust=False).mean()

        # 3. 计算JJ（PJJ的3期EMA的前一日值）
        df['JJ'] = df['PJJ'].ewm(span=3, adjust=False).mean().shift(1)

        # 4. 计算QJJ（单位价格波动范围的成交量）
        df['QJJ'] = df['volume'] / ((df['high'] - df['low']) * 2 - abs(df['close'] - df['open']))

        # 5. 计算XVL（资金流向）
        df['XVL'] = df.apply(self._calculate_xvl, axis=1)

        # 6. 计算ZLL（成交量占比）
        df['ZLL'] = df['volume'] / df['total_shares']

        # 7. 计算LIJIN1（限制ZLL的最大值为10）
        df['LIJIN1'] = df['ZLL'].apply(lambda x: min(x, 10))

        # 8. 计算LIJIN（资金流向强度）
        df['LIJIN'] = (df['XVL'] / 20) / 1.15

        # 9. 计算主生量（综合资金流向）
        df['主生量'] = df['LIJIN'] * 0.55 + df['LIJIN'].shift(1) * 0.33 + df['LIJIN'].shift(2) * 0.22

        # 10. 计算GJJ（8期EMA的主生量）
        df['GJJ'] = df['主生量'].ewm(span=8, adjust=False).mean()

        # 11. 计算LLJX（3期EMA的主生量）
        df['LLJX'] = df['主生量'].ewm(span=3, adjust=False).mean()

        # 12. 计算资金量（LLJX）
        df['资金量'] = df['LLJX']

        # 13. 计算量基线（GJJ）
        df['量基线'] = df['GJJ']

        # 14. 计算量增幅（LLJX的变化率）
        df['ZJLL'] = df['LLJX'].shift(1)
        df['QZJJ'] = ((df['LLJX'] - df['ZJLL']) / df['ZJLL']) * 100
        df['量增幅'] = df.apply(
            lambda row: abs(row['QZJJ']) if row['LLJX'] > 0 and row['ZJLL'] < 0 else
            -row['QZJJ'] if row['LLJX'] < 0 and row['ZJLL'] < 0 and row['LLJX'] < row['ZJLL'] else
            row['QZJJ'],
            axis=1
        )

        # 15. 计算力度（LIJIN的缩放值）
        df['力度'] = df['LIJIN'] / 1000

        # 16. 计算周量（过去5日的LLJX总和）
        df['周量'] = df['LLJX'].rolling(window=5).sum()

        # 17. 计算周增幅（周量的变化率）
        df['BB'] = df['周量'].shift(1)
        df['ZQZJJ'] = ((df['周量'] - df['BB']) / df['BB']) * 100
        df['周增幅'] = df.apply(
            lambda row: abs(row['ZQZJJ']) if row['周量'] > 0 and row['BB'] < 0 else
            -row['ZQZJJ'] if row['周量'] < 0 and row['BB'] < 0 and row['周量'] < row['BB'] else
            row['ZQZJJ'],
            axis=1
        )

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
            (df['volume'] > df['volume_ma5'] * 1.2)
        )
        
        # MACD条件（使用预计算值）
        df['macd_condition'] = (
            (df['macd'] < 0) & 
            (df['macd'] > df['macd_signal']) 
        )

        #MACDjc
        df['macd_jc']=(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) < df['macd_signal'].shift(1))
        
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

        # 计算资金流向相关指标
        df = self._calculate_money_flow_indicators(df)
        
        # 新增资金流向信号
        df['money_flow_positive'] = df['资金量'] > 0  # 资金流入
        df['money_flow_increasing'] = df['量增幅'] > 0  # 资金流入加速
        df['money_flow_trend'] = df['主生量'] > df['量基线']  # 短期资金流向趋势向上
        df['money_flow_weekly'] = df['周量'] > 0  # 周资金流入
        df['money_flow_weekly_increasing'] = df['周增幅'] > 0  # 周资金流入加速

        # 补充 JC 条件
        df['jc_condition'] = (
            (df['ma_5'] > df['ma_5'].shift(1)) &  # MA5 上升
            (df['ma_20'] > df['ma_20'].shift(1)) &  # MA20 上升
            (abs(df['ma_5'] - df['ma_20']) / df['ma_20'] < 0.02)  # MA5 和 MA20 的差值占 MA20 的比例小于 2%
        )
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
            signals['macd_condition'] &
            (signals['jc_condition']|signals['macd_jc']) &  # JC 条件
            (signals['ma_20'] < signals['ma_55']) &  # MA20 < MA55
            (signals['ma_55'] > signals['ma_240'])  # MA55 > MA240
        )
        
        return signals[buy_condition][[
            'date', 'symbol', 'ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'volume_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14','macd_jc',
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close','money_flow_positive',
            'money_flow_increasing','money_flow_trend','money_flow_weekly','money_flow_weekly_increasing','量增幅','量基线','主生量'
        ]].assign(signal_type='buy')
        
    def get_sell_signals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取卖点信号"""
        signals = self.generate_features(start_date, end_date)
        
        # 综合卖出条件
        sell_condition = (
            (signals['ma_20'] > signals['ma_5']) &  # 短期均线下穿长期
            (
                (signals['macd'] > signals['macd_signal']) & (signals['macd'].shift(1) < signals['macd_signal'].shift(1)) |  # MACD死叉
                (
                    (signals['close'] < signals['ma_10']) & 
                    (signals['volume'] < signals['volume'].shift(1) * 0.8)
                )
            )
           # signals['rsi_overbought'] |    # RSI超买
           # signals['kdj_overbought'] |     # KDJ超买
           # signals['bb_upper_break']      # 突破布林上轨（超买信号）
        )
        
        return signals[sell_condition][[
            'date', 'symbol', 'ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'volume_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14','macd_jc',
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close','money_flow_positive',
            'money_flow_increasing','money_flow_trend','money_flow_weekly','money_flow_weekly_increasing','量增幅','量基线','主生量'
        ]].assign(signal_type='sell')    
