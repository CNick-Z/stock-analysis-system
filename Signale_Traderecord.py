# Signale_Traderecord.py
from db_operations import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester import DynamicPositionManager,TradingSimulator
from stock_report import StockReport
from strategy import EnhancedTDXStrategy


class SignalTraderecord:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_manager = DatabaseManager(f"sqlite:///{db_url}")
        self.db_manager.ensure_tables_exist()
        self.PositionStatus={}
        self.strategy=EnhancedTDXStrategy()
        self.buy_signals=pd.DataFrame()
        self.sell_signals=pd.DataFrame()
        self.holding_stocks=[]
        self.commission_rate=0.00011

    


    def _generate_daily_signals(self):
        date =date.today().strftime("%Y-%m-%d")
        buy_signals,sell_signals = strategy.get_signals(date,date)
        self.buy_signals =   buy_signals[buy_signals.index == date]
        self.sell_signals =   sell_signals[sell_signals.index == date]

    def get_unsell_stock(self):
        table=PositionDetail
        column=['date','symbol','price','quantity','commission']
        unsell_stocks=self.db_manager.load_data(table=PositionDetail,columns=column,filter_conditions={'sell_date':None})
        self.holding_stocks=unsell_stocks

    def record_trade(self,data,operation_type='sell'):
        db_data=[]        
        if operation_type == 'buy':
            for item in data:
                date=item[0]
                symbol=item[1]
                price=item[2]
                quantity=item[3]
                commission = price * quantity * self.commission_rate
                pnl = 0
                sell_date = None
                sell_price = None
                db_data.append({'date': date,
                        'symbol': symbol,
                        'price': price,
                        'quantity': quantity,
                        'commission': commission,
                        'pnl': pnl,
                        'sell_date': sell_date,
                        'sell_price': sell_price
                    })
            data=pd.DataFrame(db_data)
            self.db_manager.bulk_insert(
            table=PositionDetail,
            data=data     
        )
        else:
            for item in data:
                date=item[0]
                symbol=item[1]
                price=item[2]
                quantity=item[3]
                commission = price * quantity * (self.commission_rate*2+0.0005)
                #id=self.holding_stocks.loc[self.holding_stocks['symbol']==symbol,'id'].iloc[0]
                buy_price=self.holding_stocks.loc[self.holding_stocks['symbol']==symbol,'price'].iloc[0]
                pnl = (price - buy_price) * quantity-commission
                sell_date = date
                buy_day=self.holding_stocks.loc[self.holding_stocks['symbol']==symbol,'date'].iloc[0]
                db_data.append({
                        'date': buy_day,
                        'symbol': symbol,
                        'price': buy_price,
                        'quantity': quantity,
                        'commission': commission,
                        'pnl': pnl,
                        'sell_date': sell_date,
                        'sell_price':price
                    })
            filter_fields=['date','symbol']
            self.db_manager.bulk_update(
            table=PositionDetail,
            data=db_data,
            update_fields=['sell_date','pnl','sell_price','commission'],
            filter_fields=filter_fields
            )
        
    

    def run(self):
        self.PositionStatus=self.db_manager.load_data(table=PositionStatus).apply(set).to_dict()
        self._generate_daily_signals()


'''
simulator = TradingSimulator(
            initial_capital=1e5,
            commission=0.0003,
            position_limit=5
            )
portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: {'qty': int, 'cost': float}}
            'history': []
        }

def _execute_sell(self, symbol, price, date, quantity,signal_info):
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
            'pnl': net_proceeds - (position['cost'] * quantity / (quantity + position['qty'])),
            'signal_info': signal_info  # 新增字段，记录交易信号信息
        })
        print(f"\n{date} 卖出 {symbol}，价格: {price}，数量: {quantity}，佣金: {commission}")
        return True

def _execute_buy(self, symbol, price, date, quantity,signal_info):
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
            'commission': commission,
            'signal_info': signal_info  # 新增字段，记录交易信号信息
        })
        print(f"\n{date} 买入 {symbol}，价格: {price}，数量: {quantity}，佣金: {commission}")
        return True

db_manager = DatabaseManager(db_url)
db_manager.ensure_tables_exist()
db_manager.create_additional_indexes()

def update_PositionDetail():
'''
if __name__ == '__main__':
    recorder=SignalTraderecord('c:/db/stock_data.db')    
    buylist=[('2025-02-21','300041',9.03,1200),
            ('2025-02-24','601311',8.48,1200),
            ('2025-02-24','603100',21.3,500),
            ('2025-02-26','688308',21.41,500),
            ('2025-03-05','600764',27.41,400),
            ('2025-03-07','600893',38.28,300),
            ('2025-03-12','600798',3.08,3500),
            ('2025-03-12','603301',21.52,600)]
    #selllist=[('2025-03-14','603301',21.41,600)]
    recorder.record_trade(buylist,'buy')
    recorder.get_unsell_stock()
    #recorder.record_trade(selllist)
    

