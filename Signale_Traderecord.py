# Signale_Traderecord.py
from db_operations import *
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date
from stock_report import StockReport
from strategy import EnhancedTDXStrategy
from get_notion_database_info import NotionDatabaseManager


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
        self.position_limit=10
        self.report = StockReport()
        self.position_manager = self.DynamicPositionAdjuster(
            initial_position=0.5, position_levels=[0.3, 0.5, 0.8, 1], window_size=3, db_manager=self.db_manager
        )
    class DynamicPositionAdjuster:
        def __init__(self, initial_position=0.5, position_levels=[0.3, 0.5, 0.8, 1], window_size=3, db_manager=None):
            self.initial_position = initial_position
            self.position_levels = position_levels
            self.window_size = window_size
            self.db_manager = db_manager
            self.previous_values = []
            self.consecutive_up_periods = 0
            self.consecutive_down_periods = 0
            self.current_position = self.initial_position
            self.multi_day_return = 0.0
        
        def get_previous_total_assets(self, date):
            start_date = (pd.to_datetime(date) - pd.Timedelta(days=self.window_size)).strftime("%Y-%m-%d")
            table = PositionStatus
            column = ['date', 'total_assets']
            filter_conditions={
                'date': {'$between':[start_date, date]} 
            }
            data = self.db_manager.load_data(table_class=table, columns=column, filter_conditions=filter_conditions)
            return data.set_index('date')['total_assets'].tolist()

        def update_position(self, current_assets, date):
            self.previous_values = self.get_previous_total_assets(date)
            self.previous_values.append(current_assets)
            if len(self.previous_values) > self.window_size:
                del self.previous_values[0]
            if len(self.previous_values)==self.window_size:
                returns = np.array(self.previous_values[1:]) / np.array(self.previous_values[:-1]) - 1
                self.multi_day_return = np.prod(1 + returns) - 1

                if self.multi_day_return > 0:
                    self.consecutive_up_periods += 1
                    self.consecutive_down_periods = 0
                elif self.multi_day_return < 0:
                    self.consecutive_up_periods = 0
                    self.consecutive_down_periods += 1
                else:
                    pass
                # 根据连续涨跌周期数调整仓位
                if self.consecutive_up_periods >= 2:  # 这里假设一个窗口期作为一个周期
                    # 连续上涨一个窗口期，判断是否为温和上涨
                    if self.multi_day_return < 0.1:  # 假设温和上涨的阈值为10%
                        self._increase_position()
                    # 迅速上涨，仓位不变
                elif self.consecutive_down_periods >= 2:
                    # 连续下跌一个窗口期，降低仓位
                    self._decrease_position()
                # 根据多日收益率的大小进一步调整仓位
                self._adjust_position_by_return()
            return self.current_position

        def _increase_position(self):
            new_position = min(self.position_levels[-1], self.initial_position + 0.2)
            self.current_position = new_position

        def _decrease_position(self):
            new_position = max(self.position_levels[0], self.initial_position - 0.2)
            self.current_position = new_position

        def _adjust_position_by_return(self):
            if self.multi_day_return > 0.1:
                self._increase_position()
            elif self.multi_day_return < -0.1:
                self._decrease_position()

    

    def _fetch_price_matrix(self, date: str):
        filter_conditions = {'date': date}
        # 查询数据        
        df = self.db_manager.load_data(DailyDataBase, filter_conditions, columns=['date', 'symbol','close'])
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
        column=['date','symbol','price','newprice','highprice','quantity','commission']
        unsell_stocks=self.db_manager.load_data(table_class=table,columns=column,filter_conditions={'sell_date':None})
        self.holding_stocks=unsell_stocks

    def _get_PositionStatus(self,date):
        table=PositionStatus
        #获取数据库中最后更新仓位记录的日期
        lastdate=self.db_manager.load_data(table_class=table,columns=['date'],order_by=[{'column':'date','direction':'desc'}],limit=1)
        if lastdate.empty:
            position_status={'date':date,'total_assets':0,'stock_value':0,'cash':150000,'position_ratio':0.5,'available_position':10,'db_data':'no'}
        else:
            column=['date','total_assets','stock_value','cash','position_ratio','available_position']
            position_status=self.db_manager.load_data(table_class=table,columns=column,filter_conditions={'date':lastdate['date'].iloc[0]})
            position_status=position_status.to_dict(orient='records')[0]
            position_status['db_data']='yes'            
        self.PositionStatus=position_status
    

    def _set_PositionStatus(self,date):

        self._get_unsell_stock()
        stock_value=0
        for index,row in self.holding_stocks.iterrows():
            buydate=row['date']
            symbol = row['symbol']
            quantity = row['quantity']
            highprice = row['highprice']
            # 查询价格矩阵中的价格
            if symbol in self.price_matrix.columns:
                price = self.price_matrix[symbol].values[0]
                #更新持仓股票最新价格和最高价格
                if price>highprice:
                    highprice=price
                update_fields=['newprice','highprice',]
                self.db_manager.bulk_update(
                    table=PositionDetail,
                    data=[{'symbol':symbol,'date':buydate,'newprice':price,'highprice':highprice}],
                    update_fields=['newprice','highprice'],
                    filter_fields={'symbol':symbol,'date':buydate}
                )
            else:
                price = row['price']
            stock_value+=quantity*price
        table=PositionStatus
        total_assets=stock_value+self.PositionStatus['cash']
        position_ratio=stock_value/total_assets
        db_data=[{'date':date,
                'stock_value':stock_value,
                'cash':self.PositionStatus['cash'],
                'position_ratio':position_ratio,
                'available_position':self.PositionStatus['available_position'],
                'total_assets':total_assets
                }]
        if  self.PositionStatus['db_data']=='no' or date != self.PositionStatus['date']:

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
        self.position_manager.update_position(self.PositionStatus['total_assets'],date)
        
            


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
                        'newprice': price,
                        'highprice': price,
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
                        'newprice': price,
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
                    signal_info = self.report.generate_report(symbol_data)
                    buy_advice.append(
                        f"建议买入 {row['symbol']}，可买入数量: {max_afford}，信号评分报告: {signal_info}")
                    cash -= max_investment
            else:
                break
        return buy_advice
    def _check_stop_loss(self):
        """执行止损检查"""
        stoploss_advice=[]
        self._get_unsell_stock()
        for index,row in self.holding_stocks.iterrows():
            buydate=row['date']
            symbol = row['symbol']
            quantity = row['quantity']
            highprice = row['highprice']
            newprice=row['newprice']
            price=row['price']  
            # 成本止损
            if newprice <= price * 0.9:
                stoploss_advice.append(f"{row['symbol']}价格跌至成本价的90%以下,止损卖出。 ")
            return stoploss_advice


    def _process_sell_signals(self, date):
        sell_advice = []
        if self.sell_signals.empty:
            return sell_advice
        scored_signals = self.sell_signals[self.sell_signals.index == date]
        if scored_signals.empty:
            return sell_advice
        holding_symbols = self.holding_stocks['symbol'].to_list()
        for _, row in scored_signals.iterrows():
            if row['symbol'] in holding_symbols and (self.holding_stocks.loc[self.holding_stocks['symbol']==row['symbol'],'date']!=date).item():
                sell_advice.append(f"{row['symbol']}出现卖出信号，建议卖出.")
        return sell_advice
        
    def generate_trading_advice(self, date):
        # 获取当前仓位状态
        self._get_PositionStatus(date)
        self._generate_daily_signals(date)
        self._fetch_price_matrix(date)

        # 更新仓位
        current_position = self.position_manager.update_position(self.PositionStatus['total_assets'], date)
        current_position_ratio = self.PositionStatus['position_ratio']
        available_position = self.PositionStatus['available_position']

        # 生成买入和卖出建议
        buy_advice = self._process_buy_signals(date)
        sell_advice = self._process_sell_signals(date)
        stoploss_advice = self._check_stop_loss()

        # 仓位控制建议
        position_control_advice = []
        if current_position_ratio < current_position:
            position_control_advice.append(f"当前仓位比例为 {current_position_ratio:.2f}，低于目标仓位 {current_position:.2f}。建议适当增加仓位。")
        elif current_position_ratio > current_position:
            position_control_advice.append(f"当前仓位比例为 {current_position_ratio:.2f}，高于目标仓位 {current_position:.2f}。建议适当减少仓位。")
        else:
            position_control_advice.append(f"当前仓位比例为 {current_position_ratio:.2f}，符合目标仓位 {current_position:.2f}。无需调整仓位。")

        # 如果可用仓位不足，提醒用户
        if available_position <= 0:
            position_control_advice.append("当前持股数量已达上限，无法进行新的买入操作。")
        else:
            position_control_advice.append(f"当前剩余持股数量 {available_position}，可以进行新的买入操作。")

        # 汇总建议文本
        advice_text = f"日期: {date}\n"
        if buy_advice:
            advice_text += "买入建议:\n"
            for advice in buy_advice:
                advice_text += f" - {advice}\n"
        if sell_advice:
            advice_text += "卖出建议:\n"
            for advice in sell_advice:
                advice_text += f" - {advice}\n"
        if stoploss_advice:
            advice_text += "止损建议:\n"
            for advice in stoploss_advice:
                advice_text += f" - {advice}\n"
        if position_control_advice:
            advice_text += "仓位控制建议:\n"
            for advice in position_control_advice:
                advice_text += f" - {advice}\n"
        if not buy_advice and not sell_advice and not position_control_advice:
            advice_text += "今日无交易建议。\n"

        return advice_text

    def run(self,buy_list,sell_list,date):
        self._get_PositionStatus(date)
        if buy_list:
            self.record_trade(buy_list,'buy')
        if sell_list:
            self.record_trade(sell_list)
        self._fetch_price_matrix(date)
        self._set_PositionStatus(date)
        advice=self.generate_trading_advice(date)
        print(advice)

    def get_dates_list(self,start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        dates_list = []

        current_date = start_date
        while current_date <= end_date:
            dates_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        return dates_list


if __name__ == '__main__':
    recorder=SignalTraderecord('c:/db/stock_data.db')
    date = datetime.today()+timedelta(days=-1)
    date = datetime.strftime(date,'%Y%m%d')
    notion=NotionDatabaseManager()
    '''
    buydic={
    '2025-02-21':[('300041',9.03,1200)],
    '2025-02-24':[('601311',8.48,1200),('603100',21.3,500)],
    '2025-02-26':[('688308',21.41,500)],
    '2025-03-05':[('600764',27.41,400)],
    '2025-03-07':[('600893',38.28,300)],
    '2025-03-12':[('603301',21.52,600),('600798',3.08,3500)],
    '2025-03-17':[('601860',2.8,4000)],
    '2025-03-18':[('688337',39.1,300),('603099',34.19,300),('301286',30.63,400)]
    }
    selldic={
    '2025-03-14':[('600798',3.13,3500)],
    '2025-03-18':[('600764',31.63,400),('300041',9.46,1200),('603100',21.61,500),('600893',38.1,300)]
    }
    '''
    datelist=recorder.get_dates_list('2025-02-06','2025-03-19')
    for date in datelist:
        buylist_day,selllist_day = notion.query_notion_database(date)
        recorder.run(buylist_day,selllist_day,date)
    '''
    buylist_day=[]
    selllist_day=[]
    if buydic is not {}:
        if date in buydic.keys():
            for item in buydic[date]:
                buylist=list(item)
                buylist.insert(0,date)
                buylist_day.append(tuple(buylist))
    if selldic is not {}:
        if date in selldic.keys():
            for item in selldic[date]:
                selllist=list(item)
                selllist.insert(0,date)
                selllist_day.append(tuple(selllist))
    recorder.run(buylist_day,selllist_day,date)
    '''
    
    

