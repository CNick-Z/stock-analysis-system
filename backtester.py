# backtest_integrated.py
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

class DatabaseIntegrator:
    """数据库集成模块"""
    def __init__(self, db_path='.db/stock_data.db'):
        self.conn = sqlite3.connect(db_path)
        
    def fetch_trading_dates(self) -> pd.DatetimeIndex:
        """获取所有交易日历"""
        dates = pd.read_sql("SELECT DISTINCT date FROM daily_data ORDER BY date", self.conn)
        return pd.to_datetime(dates['date']).sort_values()

    def load_price_matrix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载价格矩阵（宽表格式）"""
        query = f"""
        SELECT date, symbol, open 
        FROM daily_data 
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        df = pd.read_sql(query, self.conn)
        return df.pivot(index='date', columns='symbol', values='open')

    def get_daily_features(self, date: datetime) -> pd.DataFrame:
        """获取当日选股特征数据"""
        query = f"""
        SELECT * 
        FROM daily_data 
        WHERE date = '{date.strftime('%Y-%m-%d')}'
        """
        return pd.read_sql(query, self.conn)

class TradingSimulator:
    """交易模拟引擎"""
    def __init__(self, initial_capital=5e5, commission=0.0003):
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: {'qty': int, 'cost': float}}
            'history': []
        }
        self.commission_rate = commission
        self.position_limit = 5
        
    def execute_order(self, order_type: str, symbol: str, price: float, date: datetime, quantity: int):
        """执行订单并更新持仓"""
        if order_type == 'buy':
            return self._execute_buy(symbol, price, date, quantity)
        elif order_type == 'sell':
            return self._execute_sell(symbol, price, date, quantity)
        
    def _execute_buy(self, symbol, price, date, quantity):
        # 计算交易成本
        commission = price * quantity * self.commission_rate
        total_cost = price * quantity + commission
        
        if total_cost > self.portfolio['cash']:
            return False
            
        # 更新持仓
        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol]['qty'] += quantity
            self.portfolio['positions'][symbol]['cost'] += total_cost
        else:
            self.portfolio['positions'][symbol] = {
                'qty': quantity,
                'cost': total_cost,
                'entry_date': date
            }
            
        # 更新现金
        self.portfolio['cash'] -= total_cost
        
        # 记录交易
        self.portfolio['history'].append({
            'date': date,
            'symbol': symbol,
            'type': 'buy',
            'price': price,
            'quantity': quantity,
            'commission': commission
        })
        return True

    def _execute_sell(self, symbol, price, date, quantity):
        position = self.portfolio['positions'].get(symbol)
        if not position or position['qty'] < quantity:
            return False
            
        # 计算交易金额
        proceeds = price * quantity
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        # 更新现金和持仓
        self.portfolio['cash'] += net_proceeds
        position['qty'] -= quantity
        if position['qty'] == 0:
            del self.portfolio['positions'][symbol]
            
        # 记录交易
        self.portfolio['history'].append({
            'date': date,
            'symbol': symbol,
            'type': 'sell',
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'pnl': net_proceeds - (position['cost'] * quantity / (quantity + position['qty']))
        })
        return True

class BacktestOrchestrator:
    """回测总控模块"""
    def __init__(self, selector, db_path='.db/stock_data.db'):
        self.db = DatabaseIntegrator(db_path)
        self.selector = selector
        self.price_matrix = None
        self.trading_dates = None
        
    def initialize(self, start_date, end_date):
        """初始化回测环境"""
        self.trading_dates = self.db.fetch_trading_dates()
        self.price_matrix = self.db.load_price_matrix(start_date, end_date)
        
    def run(self, start_date, end_date):
        """运行完整回测流程"""
        self.initialize(start_date, end_date)
        simulator = TradingSimulator()
        
        # 获取回测日期范围
        dates = self.trading_dates[
            (self.trading_dates >= pd.to_datetime(start_date)) & 
            (self.trading_dates <= pd.to_datetime(end_date))
        ]
        
        for date in tqdm(dates, desc="回测进度"):
            # 生成选股信号
            features = self.db.get_daily_features(date)
            signals = self.selector.generate_signals(features)
            
            # 处理卖出信号
            self._process_sell_signals(date, signals, simulator)
            
            # 处理买入信号
            self._process_buy_signals(date, signals, simulator)
            
            # 检查止损
            self._check_stop_loss(date, simulator)
            
            # 记录每日净值
            self._record_daily_value(date, simulator)
            
        return self._generate_report(simulator)
    
    def _process_sell_signals(self, date, signals, simulator):
        """处理卖出信号"""
        sell_signals = signals[signals['signal_type'] == 'sell']
        for _, row in sell_signals.iterrows():
            if row['symbol'] in simulator.portfolio['positions']:
                next_open = self._get_next_open_price(date, row['symbol'])
                if next_open:
                    simulator.execute_order('sell', row['symbol'], next_open, date, 
                                           simulator.portfolio['positions'][row['symbol']]['qty'])
    
    def _process_buy_signals(self, date, signals, simulator):
        """处理买入信号"""
        buy_candidates = signals[signals['signal_type'] == 'buy']
        buy_candidates = buy_candidates.nlargest(5, 'score')
        
        available_slots = self.position_limit - len(simulator.portfolio['positions'])
        buy_candidates = buy_candidates.head(available_slots)
        
        for _, row in buy_candidates.iterrows():
            next_open = self._get_next_open_price(date, row['symbol'])
            if next_open:
                max_afford = simulator.portfolio['cash'] // (next_open * (1 + self.commission_rate))
                if max_afford > 0:
                    simulator.execute_order('buy', row['symbol'], next_open, date, max_afford)
    
    def _get_next_open_price(self, date, symbol):
        """获取次日开盘价"""
        next_date = date + timedelta(days=1)
        try:
            return self.price_matrix.loc[next_date, symbol]
        except KeyError:
            return None
    
    def _check_stop_loss(self, date, simulator):
        """执行止损检查"""
        for symbol, pos in list(simulator.portfolio['positions'].items()):
            current_price = self.price_matrix.loc[date, symbol]
            cost_basis = pos['cost'] / pos['qty']
            if current_price <= cost_basis * 0.85:
                next_open = self._get_next_open_price(date, symbol)
                if next_open:
                    simulator.execute_order('sell', symbol, next_open, date, pos['qty'])
    
    def _record_daily_value(self, date, simulator):
        """记录每日组合净值"""
        position_value = sum(
            self.price_matrix.loc[date, symbol] * pos['qty'] 
            for symbol, pos in simulator.portfolio['positions'].items()
            if symbol in self.price_matrix.columns
        )
        total_value = simulator.portfolio['cash'] + position_value
        simulator.portfolio['history'].append({'date': date, 'value': total_value})
    
    def _generate_report(self, simulator):
        """生成详细报告"""
        df = pd.DataFrame(simulator.portfolio['history']).set_index('date')
        df['returns'] = df['value'].pct_change()
        
        plt.figure(figsize=(12,6))
        df['value'].plot(title='Portfolio Value Curve')
        plt.savefig('portfolio_curve.png')
        
        trades = pd.DataFrame(simulator.portfolio['history'])
        return {
            'summary': {
                'final_value': df['value'].iloc[-1],
                'total_return': df['value'].iloc[-1] / 5e5 - 1,
                'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
                'turnover': trades.groupby('symbol').size().mean()
            },
            'trades': trades
        }

# 使用示例
if __name__ == "__main__":
    from stock_selector import StockSelector  # 假设已有选股器
    
    # 初始化选股策略
    selector = StockSelector(
        momentum_window=20,
        volume_threshold=1e6
    )
    
    # 运行回测
    orchestrator = BacktestOrchestrator(selector)
    report = orchestrator.run(start_date='2020-01-01', end_date='2023-12-31')
    
    print("回测结果摘要:")
    print(f"最终净值: {report['summary']['final_value']:,.2f}")
    print(f"总收益率: {report['summary']['total_return']:.2%}")
    print(f"最大回撤: {report['summary']['max_drawdown']:.2%}")