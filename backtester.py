# backtester.py
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals
        self.positions = pd.DataFrame(index=data.index)
        self.returns = pd.DataFrame(index=data.index)
        
    def run_backtest(self, initial_capital=100000):
        """执行回测"""
        # 初始化持仓
        self.positions['holdings'] = 0
        self.positions['cash'] = initial_capital
        self.positions['total'] = initial_capital
        
        # 生成交易信号
        buy_signals = self.signals['golden_cross']
        sell_signals = self.signals['death_cross']
        
        # 模拟交易
        position = 0
        for i in range(1, len(self.data)):
            if buy_signals[i] and position == 0:
                # 全仓买入
                position = self.positions['cash'][i-1] / self.data['close'][i]
                self.positions.loc[i, 'holdings'] = position
                self.positions.loc[i, 'cash'] = 0
                
            elif sell_signals[i] and position > 0:
                # 全仓卖出
                self.positions.loc[i, 'cash'] = position * self.data['close'][i]
                self.positions.loc[i, 'holdings'] = 0
                position = 0
                
            else:
                # 维持仓位
                self.positions.loc[i] = self.positions.loc[i-1]
                
            # 计算收益
            self.positions['total'] = self.positions['cash'] + \
                                     self.positions['holdings'] * self.data['close']
            self.returns['daily_return'] = self.positions['total'].pct_change()
            
        return self._generate_report()
    
    def _generate_report(self):
        """生成回测报告"""
        report = {}
        total_return = self.positions['total'][-1]/self.positions['total'][0]-1
        report['累计收益'] = total_return
        
        max_drawdown = (self.positions['total'].cummax() - self.positions['total']).max()
        report['最大回撤'] = max_drawdown
        
        # 计算年化收益
        years = len(self.data)/252
        report['年化收益'] = (1 + total_return)**(1/years) - 1
        
        return report