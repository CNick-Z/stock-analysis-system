# strategy.py
import pandas as pd
import talib
import numpy as np

class BaseStrategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
        
    def calculate_technical_indicators(self):
        """计算技术指标"""
        # 示例：计算均线
        self.data['MA5'] = self.data['close'].rolling(5).mean()
        self.data['MA20'] = self.data['close'].rolling(20).mean()
        
        # 计算MACD
        self.data['MACD'], self.data['MACDsignal'], _ = talib.MACD(
            self.data['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
    def generate_signals(self):
        """生成交易信号（需子类实现）"""
        raise NotImplementedError

class MAStrategy(BaseStrategy):
    """均线策略示例"""
    def generate_signals(self):
        self.calculate_technical_indicators()
        
        # 金叉信号
        self.signals['golden_cross'] = (
            (self.data['MA5'] > self.data['MA20']) &
            (self.data['MA5'].shift(1) <= self.data['MA20'].shift(1))
        )
        
        # 死叉信号
        self.signals['death_cross'] = (
            (self.data['MA5'] < self.data['MA20']) &
            (self.data['MA5'].shift(1) >= self.data['MA20'].shift(1))
        )
        return self.signals

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