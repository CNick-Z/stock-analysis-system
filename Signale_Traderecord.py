# Signale_Traderecord.py
from utils.db_operations import *
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date
from utils.stock_report import StockReport,StockScorer
from utils.strategy import EnhancedTDXStrategy
from utils.get_notion_database_info import NotionDatabaseManager


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
        self.Scorer= StockScorer()
        self.Report = StockReport(self.Scorer)
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
                #self._increase_position()
                self._decrease_position()
            elif self.multi_day_return < -0.1:
                #self._decrease_position()
                self._increase_position()

    

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
        column=['date','symbol','price','newprice','highprice','quantity','commission','pnl']
        unsell_stocks=self.db_manager.load_data(table_class=table,columns=column,filter_conditions={'quantity':{'$gt':0}})
        self.holding_stocks=unsell_stocks

    def _get_PositionStatus(self,date):
        table=PositionStatus
        #获取数据库中最后更新仓位记录的日期
        lastdate=self.db_manager.load_data(table_class=table,columns=['date'],order_by=[{'column':'date','direction':'desc'}],limit=1)
        if lastdate.empty:
            position_status={'date':date,'total_assets':0,'stock_value':0,'cash':200000,'position_ratio':0.5,'available_position':10,'db_data':'no'}
        else:
            column=['date','total_assets','stock_value','cash','position_ratio','available_position']
            position_status=self.db_manager.load_data(table_class=table,columns=column,filter_conditions={'date':lastdate['date'].iloc[0]})
            position_status=position_status.to_dict(orient='records')[0]
            position_status['db_data']='yes'            
        self.PositionStatus=position_status
    

    def _set_PositionStatus(self,date):

        self._get_unsell_stock()
        stock_value=0
        notion_update_dic = {}
        price_update = []
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
                price_update.append({'symbol':symbol,'date':buydate,'newprice':price,'highprice':highprice})
                notion_update_dic[symbol]=price
            else:
                price = row['price']
            stock_value+=quantity*price
        if price_update:
            update_fields=['newprice','highprice']
            self.db_manager.bulk_update(
                table=PositionDetail,
                data=price_update,
                update_fields=update_fields,
                filter_fields={'symbol':symbol,'date':buydate}
            )
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
        return notion_update_dic
        
            


    def record_trade(self,data,operation_type='sell'):
        self._get_unsell_stock()
        db_data=[]
        update_data=[]       
        if operation_type == 'buy':
            # Group buy transactions by symbol to consolidate same-day purchases
            buy_transactions = {}
            for item in data:
                date=item[0]
                symbol=item[1]
                price=item[2]
                quantity=item[3]
                if symbol not in buy_transactions:
                    buy_transactions[symbol] = {
                        'date': date,
                        'symbol': symbol,
                        'total_quantity': quantity,
                        'total_cost': price * quantity,
                        'max_price': price,        # 记录最高交易价(用于highprice)
                        'last_price': price,       # 记录最后一笔价格(用于newprice)
                        'commission': price * quantity * self.commission_rate
                    }
                else:
                    buy_transactions[symbol]['total_quantity'] += quantity
                    buy_transactions[symbol]['total_cost'] += price * quantity
                    buy_transactions[symbol]['max_price'] = max(buy_transactions[symbol]['max_price'], price)
                    buy_transactions[symbol]['last_price'] = price  # 更新为最后一笔价格
                    buy_transactions[symbol]['commission'] += price * quantity * self.commission_rate
            # Process consolidated transactions
            for symbol, trans in buy_transactions.items():
                date = trans['date']
                avg_price = trans['total_cost'] / trans['total_quantity']  # 加权平均价
                quantity = trans['total_quantity']
                commission = trans['commission']
                highprice = trans['max_price']      # 使用单笔最高价作为highprice
                newprice = trans['last_price']      # 使用最后一笔价格作为newprice
                
                self.PositionStatus['cash'] -= commission + trans['total_cost']
                
                
                # Check if stock is already held
                if not self.holding_stocks.empty:
                    existing_position = self.holding_stocks[
                        (self.holding_stocks['symbol'] == symbol) 
                    ]
                else:
                    existing_position = pd.DataFrame()
                    
                if len(existing_position) > 0:
                    # Existing position, update record
                    existing = existing_position.iloc[0]
                    new_quantity = existing['quantity'] + quantity
                    new_avg_price = (existing['price'] * existing['quantity'] + trans['total_cost']) / new_quantity
                    new_highprice = max(existing['highprice'], highprice)
                    
                    update_data.append({
                        'date': existing['date'],
                        'symbol': symbol,
                        'quantity': new_quantity,
                        'price': new_avg_price,    # 更新为新的加权平均价
                        'newprice': newprice,      # 使用最后一笔价格
                        'highprice': new_highprice, # 更新最高价
                        'commission': existing['commission'] + commission,
                        'pnl': 0,
                        'sell_date': None,
                        'sell_price': None
                    })
                else:
                    # New position, create record
                    self.PositionStatus['available_position'] -= 1
                    db_data.append({
                        'date': date,
                        'symbol': symbol,
                        'price': avg_price,       # 使用加权平均价
                        'newprice': newprice,      # 使用最后一笔价格
                        'highprice': highprice,    # 使用单笔最高价
                        'quantity': quantity,
                        'commission': commission,
                        'pnl': 0,
                        'sell_date': None,
                        'sell_price': None
                    })
            if db_data:
                data = pd.DataFrame(db_data)
                self.db_manager.bulk_insert(
                    table=PositionDetail,
                    data=data     
                )
            if update_data:
                self.db_manager.bulk_update(
                        table=PositionDetail,
                        data=update_data,
                        update_fields=['quantity', 'price', 'newprice', 'highprice', 'commission'],
                        filter_fields={
                            'date': existing['date'],
                            'symbol': symbol,
                            'sell_date': None
                        }
                    )

        else:
            # 第一步：按symbol分组合并当日卖出交易
            sell_transactions = {}
            for item in data:
                date = item[0]
                symbol = item[1]
                price = item[2]
                quantity = item[3]
                
                if symbol not in sell_transactions:
                    sell_transactions[symbol] = {
                        'total_quantity': 0,
                        'total_value': 0,
                        'avg_price': 0,
                        'commission': 0,
                        'last_price': price
                    }
                
                # 累加卖出数量和金额
                sell_transactions[symbol]['total_quantity'] += quantity
                sell_transactions[symbol]['total_value'] += price * quantity
                sell_transactions[symbol]['commission'] += price * quantity * (self.commission_rate*2 + 0.0005)
                sell_transactions[symbol]['last_price'] = price  # 记录最后一次卖出价
                sell_transactions[symbol]['date'] = date
            # 第二步：处理每个股票的卖出
            for symbol, trans in sell_transactions.items():
                # 计算平均卖出价
                trans['avg_price'] = trans['total_value'] / trans['total_quantity']
                
                # 查找对应持仓
                position = self.holding_stocks[
                    (self.holding_stocks['symbol'] == symbol) & 
                    (self.holding_stocks['quantity'] > 0)
                ]
                if position.empty:
                    continue
                position = position.iloc[0]
                buy_price = position['price']
                remaining_qty = position['quantity'] - trans['total_quantity']
                
                # 计算本次总盈亏（基于平均卖出价）
                pnl = (trans['avg_price'] - buy_price) * trans['total_quantity'] - trans['commission']
                
                # 更新现金和可用仓位
                self.PositionStatus['cash'] += trans['total_value'] - trans['commission']
                if remaining_qty == 0: #卖完增加一个股票持仓位
                    self.PositionStatus['available_position'] += 1
                 # 准备更新数据（PNL累加！）
                update_data.append({
                    'date': position['date'],  # 买入日期
                    'symbol': symbol,
                    'price': buy_price,
                    'quantity': remaining_qty,
                    'newprice': trans['last_price'],  # 最后一次卖出价
                    'sell_date': trans['date'],       # 卖出日期
                    'sell_price': trans['avg_price'], # 平均卖出价
                    'commission': position['commission'] + trans['commission'],
                    'pnl': position['pnl'] + pnl      # PNL累加
                })
            filter_fields=['date','symbol']
            self.db_manager.bulk_update(
            table=PositionDetail,
            data=update_data,
            update_fields=['sell_date','pnl','quantity','newprice','sell_price','commission'],
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
                    signal_info = self.Report.generate_report(symbol_data)
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
            newprice=row['newprice']
            price=row['price']  
            # 成本止损
            if newprice <= price * 0.9:
                stoploss_advice.append(f"{row['symbol']}价格跌至成本价的90%以下,建议止损卖出。 ")
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
        notion_update_dic = self._set_PositionStatus(date)
        advice=self.generate_trading_advice(date)
        advice_recordfile='./backtestresult/advice_record.txt'
        # 获取当前时间作为时间戳
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 准备要写入的内容
        advice_content = (
            f"{'*' * 40}\n"  # 分隔符
            f"记录时间: {current_time}\n"  # 时间戳
            f"{date}选股建议\n"
            f"{advice}\n"
            f"{'*' * 40}\n"  # 分隔符
        )
        with open(advice_recordfile, 'a') as f:
            f.write(advice_content)
        return advice,notion_update_dic

    def get_dates_list(self,start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        dates_list = []

        current_date = start_date
        while current_date <= end_date:
            dates_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        return dates_list

    def get_trading_data(self, start_date, end_date):
        filter_conditions = {
                    'date': {
                        '$between': [start_date, end_date]
                    }
                }
                # 查询数据
        df = self.db_manager.load_data(DailyDataBase, filter_conditions, columns=['date'])
        if df.empty:
            return []
        # 生成交易日期索引
        trading_dates = pd.to_datetime(df['date']).sort_values().unique()
        return trading_dates


if __name__ == '__main__':
    recorder=SignalTraderecord('c:/db/stock_data.db')
    date = datetime.today()
    date = datetime.strftime(date,'%Y-%m-%d')
    notion=NotionDatabaseManager()
    #trading_dates=recorder.get_trading_data(date,date)

    trading_dates=recorder.get_trading_data('2025-04-25','2025-04-25')
    for date in trading_dates:
        print(date)
        buylist_day,selllist_day = notion.query_notion_database(datetime.strftime(date,'%Y-%m-%d'))
        advice,notion_update_dic = recorder.run(buylist_day,selllist_day,datetime.strftime(date,'%Y-%m-%d'))
        notion.update_task_database(datetime.strftime(date,'%Y-%m-%d'),advice)
        notion.update_stock_database(notion_update_dic)

    

