# Signale_Traderecord.py
from db_operations import *
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date
from stock_report import StockReport
from strategy import EnhancedTDXStrategy





class SignalTraderecord:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_manager = DatabaseManager(f"sqlite:///{db_url}")
        self.db_manager.ensure_tables_exist()
        self.PositionStatus={}
        self.strategy=EnhancedTDXStrategy(db_url)
        self.buy_signals=pd.DataFrame()
        self.sell_signals=pd.DataFrame()
        self.holding_stocks=[]
        self.commission_rate=0.00011
        self.price_matrix=None
        self.position_manager = self.DynamicPositionAdjuster(
            initial_position=0.5, position_levels=[0.3, 0.5, 0.8, 1], window_size=3, db_manager=self.db_manager
        )
    class DynamicPositionAdjuster:
        def __init__(self, initial_position=0.5, position_levels=[0.3, 0.5, 0.8, 1], window_size=3, db_manager=None):
            self.initial_position = initial_position
            self.position_levels = position_levels
            self.window_size = window_size
            self.db_manager = db_manager
        
        def get_previous_total_assets(self, date):
            start_date = (pd.to_datetime(date) - pd.Timedelta(days=self.window_size)).strftime("%Y-%m-%d")
            table = PositionStatus
            column = ['date', 'total_assets']
            filter_conditions={
                'date': {'$between':[start_date, date]} 
            }
            data = self.db_manager.load_data(table=table, columns=column, filter_conditions=filter_conditions)
            return data.set_index('date')['total_assets'].tolist()

        def update_position(self, current_assets, date):
            previous_values = self.get_previous_total_assets(date)
            if previous_values and len(previous_values) >= self.window_size:
                returns = np.array(previous_values[1:]) / np.array(previous_values[:-1]) - 1
                multi_day_return = np.prod(1 + returns) - 1

                if multi_day_return > 0.1:
                    new_position = min(self.position_levels[-1], self.initial_position + 0.2)
                elif multi_day_return < -0.1:
                    new_position = max(self.position_levels[0], self.initial_position - 0.2)
                else:
                    new_position = self.initial_position
            else:
                new_position = self.initial_position
            return new_position

    

    def _fetch_price_matrix(self, date: str):
        filter_conditions = {'date': date}
        # 查询数据        
        df = self.db_manager.load_data(DailyData, filter_conditions, columns=['date', 'symbol','close'])
        # 生成价格矩阵
        price_matrix = df.pivot_table(index='date', columns='symbol', values='close')
        self.price_matrix=price_matrix



    def _generate_daily_signals(self,date):
        buy_signals,sell_signals = self.strategy.get_signals(date,date)
        if buy_signals is not None:
            self.buy_signals = buy_signals[buy_signals.index == date]
        if sell_signals is not None:
            self.sell_signals = sell_signals[sell_signals.index == date]

    def _get_unsell_stock(self):
        table=PositionDetail
        column=['date','symbol','price','quantity','commission']
        unsell_stocks=self.db_manager.load_data(table=table,columns=column,filter_conditions={'sell_date':None})
        self.holding_stocks=unsell_stocks

    def _get_PositionStatus(self,date):
        table=PositionStatus
        column=['date','total_assets','stock_value','cash','position_ratio','available_position']
        position_status=self.db_manager.load_data(table=table,columns=column,filter_conditions={'date':date})
        if position_status.empty:
            position_status={'date':date,'total_assets':0,'stock_value':0,'cash':100000,'position_ratio':0,'available_position':10,'db_data':'no'}
        else:
            position_status=position_status.to_dict(orient='records')[0]
            position_status['db_data']='yes'
        self.PositionStatus=position_status
    

    def _set_PositionStatus(self,date):
        self._get_unsell_stock()
        stock_value=0
        for index,row in self.holding_stocks.iterrows():
            symbol = row['symbol']
            quantity = row['quantity']
            # 查询价格矩阵中的价格
            if symbol in self.price_matrix.columns:
                price = self.price_matrix[symbol].values[0]
            else:
                price = row['price']
            stock_value+=quantity*price
        table=PositionStatus
        db_data=[{'date':date,
                'stock_value':stock_value,
                'cash':self.PositionStatus['cash'],
                'position_ratio':self.PositionStatus['position_ratio'],
                'available_position':self.PositionStatus['available_position'],
                'total_assets':stock_value+self.PositionStatus['cash']
                }]
        if  self.PositionStatus['db_data']=='no':            
            data=pd.DataFrame(db_data)
            self.db_manager.bulk_insert(
            table=table,
            data=data     
            )
        else:
            self.db_manager.bulk_update(
            table=table,
            data=db_data,
            update_fields=['stock_value','cash','position_ratio','available_position','total_assets'],
            filter_fields=['date']
            )
            


    def record_trade(self,data,operation_type='sell'):
        self._get_unsell_stock()
        db_data=[]        
        if operation_type == 'buy':
            for item in data:
                date=item[0]
                symbol=item[1]
                price=item[2]
                quantity=item[3]
                commission = price * quantity * self.commission_rate
                self.PositionStatus['cash']-=commission+price * quantity
                self.PositionStatus['available_position']-=1
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
                buy_price=self.holding_stocks.loc[self.holding_stocks['symbol']==symbol,'price'].iloc[0]
                pnl = (price - buy_price) * quantity-commission
                sell_date = date
                buy_day=self.holding_stocks.loc[self.holding_stocks['symbol']==symbol,'date'].iloc[0]
                self.PositionStatus['cash']+=price * quantity-price * quantity * (self.commission_rate+0.0005)
                self.PositionStatus['available_position']+=1
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
    def _process_buy_signals(self, date):
        current_position = self.position_manager.update_position(self.PositionStatus['total_assets'], date)
        available_slots = self.PositionStatus['available_position']
        cash = self.PositionStatus['cash']
        if available_slots <= 0:
            return []
        if self.buy_signals.empty:
            return []
        scored_signals = self.buy_signals[self.buy_signals.index == date]
        if scored_signals.empty:
            return []
        holding_symbols = self.holding_stocks['symbol'].to_list()
        buy_advice = []
        for _, row in scored_signals.iterrows():
            if available_slots > 0:
                if row['symbol'] in holding_symbols:
                    continue
                total_account_value = self.PositionStatus['total_assets']
                if (total_account_value - cash) / total_account_value > current_position:
                    continue
                max_investment_cash = total_account_value * current_position // self.position_limit
                max_investment = cash // available_slots
                max_investment = min(max_investment, max_investment_cash)
                if row['symbol'] in self.price_matrix.columns:
                    current_price = self.price_matrix[row['symbol']].values[0]
                else:
                    continue
                max_afford = max_investment // (current_price * (1 + self.commission_rate))
                max_afford = (max_afford // 100) * 100
                if max_afford > 0:
                    symbol_data = scored_signals[scored_signals['symbol'] == row['symbol']]
                    signal_info = StockReport.generate_report(symbol_data)
                    buy_advice.append(
                        f"建议买入 {row['symbol']}，可买入数量: {max_afford}，信号评分报告: {signal_info}")
                    available_slots -= 1
                    cash -= max_investment
            else:
                break
        return buy_advice

    def _process_sell_signals(self, date):
        sell_advice = []
        if self.sell_signals.empty:
            return sell_advice
        scored_signals = self.sell_signals[self.sell_signals.index == date]
        if scored_signals.empty:
            return sell_advice
        holding_symbols = self.holding_stocks['symbol'].to_list()
        for _, row in scored_signals.iterrows():
            if row['symbol'] in holding_symbols:
                sell_advice.append(f"建议卖出 {row['symbol']}")
        return sell_advice
        
    def generate_trading_advice(self, date):
        self._get_PositionStatus(date)
        self._get_unsell_stock()
        self._generate_daily_signals(date)
        buy_advice = self._process_buy_signals(date)
        sell_advice = self._process_sell_signals(date)
        advice_text = f"日期: {date}\n"
        if buy_advice:
            advice_text += "买入建议:\n"
            for advice in buy_advice:
                advice_text += f" - {advice}\n"
        if sell_advice:
            advice_text += "卖出建议:\n"
            for advice in sell_advice:
                advice_text += f" - {advice}\n"
        if not buy_advice and not sell_advice:
            advice_text += "今日无交易建议。"
        return advice_text

    def run(self,buy_list,sell_list,date):
        #self._get_PositionStatus(date)
        #self._generate_daily_signals()
        #self.record_trade(buylist,'buy')
        #self.record_trade(selllist)
        #self._fetch_price_matrix(date)
        #self._set_PositionStatus(date)
        advice=self.generate_trading_advice(date)
        print(advice)

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
    date =date.today().strftime("%Y-%m-%d")
    buylist=[('2025-02-21','300041',9.03,1200),
            ('2025-02-24','601311',8.48,1200),
            ('2025-02-24','603100',21.3,500),
            ('2025-02-26','688308',21.41,500),
            ('2025-03-05','600764',27.41,400),
            ('2025-03-07','600893',38.28,300),
            ('2025-03-12','603301',21.52,600)]
    selllist=[('2025-03-14','603301',21.41,600)]
    recorder.run(buylist,selllist,'2025-03-02')
    

