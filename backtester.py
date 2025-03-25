# backtester.py
import os,time
from utils.db_operations import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from datetime import datetime, timedelta
from utils.stock_report import  *
from utils.strategy import StockScorer,EnhancedTDXStrategy

class DynamicPositionManager:
    """动态仓位管理模块"""
    def __init__(self, initial_position=0.5, position_levels=[0.3,0.5,0.8,1], window_size=5):
        self.current_position = initial_position  # 当前仓位
        self.position_levels = position_levels    # 仓位档次
        self.consecutive_up_periods = 0            # 连续上涨周期数
        self.consecutive_down_periods = 0          # 连续下跌周期数
        self.previous_values = []                  # 前几日组合价值，用于计算多日收益率
        self.window_size = window_size            # 多日收益率窗口大小
        self.position_index = position_levels.index(initial_position) if initial_position in position_levels else 1
        self.multi_day_return = 0                  # 多日收益率

    def update_position(self, current_value):
        """根据当前组合价值更新仓位"""
        self.previous_values.append(current_value)
        if len(self.previous_values) > self.window_size:
            self.previous_values=[]  # 保持窗口大小
        
        if len(self.previous_values) >= self.window_size:
            # 计算多日收益率
            start_value = self.previous_values[0]
            end_value = self.previous_values[-1]
            self.multi_day_return = (end_value - start_value) / start_value
            
            # 判断涨跌情况
            if self.multi_day_return > 0:
                self.consecutive_up_periods += 1
                self.consecutive_down_periods = 0
            elif self.multi_day_return < 0:
                self.consecutive_down_periods += 1
                self.consecutive_up_periods = 0
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
    #def _decrease_position(self):
        """增加仓位"""
        if self.position_index < len(self.position_levels) - 1:
            self.position_index += 1
            self.current_position = self.position_levels[self.position_index]
            print(f"仓位提升至: {self.current_position*100}%")
    
    def _decrease_position(self):
    #def _increase_position(self):
        """减少仓位"""
        if self.position_index > 0:
            self.position_index -= 1
            self.current_position = self.position_levels[self.position_index]
            print(f"仓位降低至: {self.current_position*100}%")
    
    def _adjust_position_by_return(self):
        """根据多日收益率调整仓位"""
        # 如果多日收益率超过某个阈值，增加仓位
        if self.multi_day_return > 0.20:  # 假设10%的收益率作为增加仓位的阈值
            self._increase_position()
        # 如果多日收益率低于某个阈值，减少仓位
        elif self.multi_day_return < -0.05:  # 假设-5%的收益率作为减少仓位的阈值
            self._decrease_position()

class DatabaseIntegrator:
    """数据库集成模块"""
    def __init__(self, db_path):
        self.db_manager = DatabaseManager(f"sqlite:///{db_path}")
        self.db_manager.ensure_tables_exist()

    def fetch_trading_dates_and_price_matrix(self, start_date: str, end_date: str) -> tuple:
        """一次性查询并生成交易日期和价格矩阵"""
        filter_conditions = {
            'date': {
                '$between': [start_date, end_date]
            }
        }
        # 查询数据
        df = self.db_manager.load_data(DailyDataBase, filter_conditions, columns=['date', 'symbol', 'open','close'])

        # 生成交易日期索引
        trading_dates = pd.to_datetime(df['date']).sort_values().unique()

        # 生成价格矩阵
        price_matrix = df.pivot_table(index='date', columns='symbol', values=['open','close'])

        return trading_dates, price_matrix


class TradingSimulator:
    """交易模拟引擎"""
    def __init__(self, initial_capital=5e5, commission=0.0003,position_limit=5):
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: {'qty': int, 'cost': float, 'latest_price': float}}
            'history': []
        }
        self.commission_rate = commission
        self.position_limit = position_limit  # 替换原来的硬编码
        self.buy_signals = {}
        self.sell_signals = {}
        
    def execute_order(self, order_type: str, symbol: str, price: float, date: datetime, quantity: int,signal_info,signal=None):
        """执行订单并更新持仓"""
        if order_type == 'buy':
            return self._execute_buy(symbol, price, date, quantity,signal_info,signal)
        elif order_type == 'sell':
            return self._execute_sell(symbol, price, date, quantity,signal_info)
        
    def _execute_buy(self, symbol, price, date, quantity,signal_info,signal):
        # 计算交易成本
        commission = price * quantity * self.commission_rate
        total_cost = price * quantity + commission
        if total_cost > self.portfolio['cash']:
            return False
            
        # 更新持仓
        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol]['qty'] += quantity
            self.portfolio['positions'][symbol]['cost'] += total_cost
            self.portfolio['positions'][symbol]['latest_price'] = price
        else:
            self.portfolio['positions'][symbol] = {
                'qty': quantity,
                'cost': total_cost,
                'entry_date': date,
                'latest_price': price  # 初始化最新价格为买入价格
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
            'signal_info': signal_info,  # 新增字段，记录交易信号信息
            'signal':signal
        })
        print(f"\n{date} 买入 {symbol}，价格: {price}，数量: {quantity}，佣金: {commission}")
        return True

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

class BacktestOrchestrator:
    """回测总控模块"""
    def __init__(self, db_path='c:/db/stock_data.db', live_plot=False,position_limit=5, commission_rate=0.0003):
        self.position_limit = position_limit  # 新增参数
        self.position_limit_base = position_limit  # 基础持仓限制
        self.db = DatabaseIntegrator(db_path)
        self.strategy = EnhancedTDXStrategy(db_path)
        self.Scorer= StockScorer()
        self.Report = StockReport(self.Scorer)
        self.price_matrix = None
        self.trading_dates = None
        self.live_plot = live_plot
        self.commission_rate = commission_rate 
        self.trailing_stop_ratio = 0.05
        self.position_manager = DynamicPositionManager(initial_position=0.5, position_levels=[0.3,0.5,0.8,1], window_size=3)
        self.plot_save_path = './backtestresult/'
        if self.live_plot:
            plt.ion()  # 启用交互模式
            # 创建网格布局，左侧显示持仓信息，右侧显示净值曲线
            self.fig = plt.figure(figsize=(12, 6))
            gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 7])  # 左右分布比例 1:3
            
            # 右侧显示净值曲线
            self.ax_value = self.fig.add_subplot(gs[0, 1])
            self.dates = []
            self.values = []
            (self.value_line,) = self.ax_value.plot([], [])
            self.ax_value.set_title("Live Portfolio Value")
            self.ax_value.set_xlabel("Date")
            self.ax_value.set_ylabel("Value")
            self.ax_value.grid(True)
            
            # 左侧显示持仓信息
            self.ax_info = self.fig.add_subplot(gs[0, 0])
            self.ax_info.axis('off')  # 关闭坐标轴
            import atexit
            atexit.register(self.save_plot)
            
    def save_plot(self):
        """保存当前图表到指定路径"""
        if hasattr(self, 'fig') and self.fig:
            now=str(time.time()).split('.')[0]
            filename = f"strategy_performance_{now}.png"
            full_path = os.path.join(self.plot_save_path, filename)
            self.fig.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close(self.fig)
            print(f"策略图表已保存至：{full_path}")

    def _info_callback(self, msg, data=None):
        """用于实时显示回测信息"""
        # 清空之前的文本信息
        self.ax_info.clear()
        self.ax_info.axis('off')  # 确保关闭坐标轴

        # 显示当前日期、净值和现金
        current_date = data['current_date']
        current_value = data['current_value']
        current_cash = data['cash']
    
        
        # 计算文本的垂直位置
        text_y = 0.95  # 初始垂直位置
        line_spacing = 0.05  # 行间距
        
        # 显示当前日期
        self.ax_info.text(
            0.05, text_y,
            f"Date: {current_date}",
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment='top'
        )
        text_y -= line_spacing  # 调整下一行的垂直位置
        
        # 显示组合价值
        self.ax_info.text(
            0.05, text_y,
            f"Portfolio Value: {current_value:.2f}",
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment='top'
        )
        text_y -= line_spacing  # 调整下一行的垂直位置
        
        # 显示现金
        self.ax_info.text(
            0.05, text_y,
            f"Portfolio Cash: {current_cash:.2f}",
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment='top'
        )
        text_y -= line_spacing  # 调整下一行的垂直位置
        
        # 显示持仓信息
        positions = data['positions']
        if positions:
            for i, (symbol, pos) in enumerate(positions.items()):
                self.ax_info.text(
                    0.05, text_y,
                    f"{symbol}: QTY={pos['qty']}, Cost={pos['cost']:.2f},Value={pos['latest_price']*pos['qty']:.2f}",
                    transform=self.ax_info.transAxes,
                    fontsize=10,
                    verticalalignment='top'
                )
                text_y -= line_spacing  # 调整下一行的垂直位置
    def initialize(self, start_date, end_date):
        """初始化回测环境"""
        print("初始化回测环境,获取交易日历和价格矩阵...")
        self.trading_dates, self.price_matrix = self.db.fetch_trading_dates_and_price_matrix(start_date, end_date)

        

    def date_by_year(self,start_date, end_date):
        # 将字符串日期转换为 datetime 对象
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)        
        # 生成按年分段的日期范围
        year_ranges = []
        current_start = start_date
        while current_start <= end_date:
            current_end = pd.to_datetime(f"{current_start.year}-12-31")
            if current_end > end_date:
                current_end = end_date
            year_ranges.append((current_start, current_end))
            current_start = pd.to_datetime(f"{current_start.year + 1}-01-01")
        return year_ranges

    def run(self, start_date, end_date):
        """运行完整回测流程"""
        self.initialize(start_date, end_date)
        simulator = TradingSimulator(
            initial_capital=5e5,
            commission=0.0003,
            position_limit=self.position_limit 
            )
        
        years=self.date_by_year(start_date, end_date)
        for year_start, year_end in years:
            # 获取回测日期范围
            dates = self.trading_dates[
            (self.trading_dates >= pd.to_datetime(year_start)) & 
            (self.trading_dates < pd.to_datetime(year_end))]
            year_buy_signals,year_sell_signals=self.strategy.get_signals(start_date,end_date)
            self._mian_process(dates,simulator,year_buy_signals,year_sell_signals)
        return self._generate_report(simulator)
        
    def _mian_process(self,dates,simulator,year_buy_signals,year_sell_signals):
        current_year=dates[0].year
        for date in tqdm(dates, desc=f"{current_year}回测进度"):
            date=date.strftime("%Y-%m-%d")
            #处理卖出计划
            if simulator.sell_signals:
                if date in simulator.sell_signals:
                    for info in simulator.sell_signals[date]:
                        simulator.execute_order('sell', info[0],info[1],info[2],info[3],info[4])
                    del simulator.sell_signals[date]
            #处理买入计划
            if simulator.buy_signals:
                if date in simulator.buy_signals:
                    for info in simulator.buy_signals[date]:
                        simulator.execute_order('buy', info[0],info[1],info[2],info[3],info[4],info[5])
                    del simulator.buy_signals[date]
            # 记录每日净值
            self._record_daily_value(date, simulator)            
            # 更新实时图表
            if self.live_plot:
                self._update_live_plot(simulator)            
            # 检查止损
            self._check_stop_loss(date, simulator)
            # 处理卖出信号
            self._process_sell_signals(date,simulator,year_sell_signals)           
            # 处理买入信号
            self._process_buy_signals(date,simulator,year_buy_signals)
            

    def _process_sell_signals(self, date, simulator,signals):
        """处理卖出信号"""
        # 只对当前持仓股票生成卖出信号
        holding_symbols = list(simulator.portfolio['positions'].keys())
        if not holding_symbols:
            return
        # 获取当前持仓股票的卖出信号
        sell_signals = signals[signals.index == date]
        if sell_signals.empty:
            return
        sell_signals = sell_signals[sell_signals['symbol'].isin(holding_symbols)]
        # 筛选出盈利的持仓股票
        profitable_sell_signals = []
        for symbol, pos in list(simulator.portfolio['positions'].items()):
            cost_per_share = pos['cost'] / pos['qty']
            if pos['latest_price'] > cost_per_share: 
                profitable_sell_signals.append(symbol)
        
        # 只对盈利的持仓股票处理卖出信号
        for _, row in sell_signals.iterrows():
            if row['symbol'] in profitable_sell_signals:
                try:
                    next_open_price,next_open_day = self._get_next_open_price(date, row['symbol'])
                    if next_open_day:
                        symbol_data = sell_signals[sell_signals['symbol'] == row['symbol']]
                        signal_info = {
                            'type': 'sell_signal',
                            'reason': f"出现卖出信号"
                        }
                        if next_open_day not in simulator.sell_signals:
                            simulator.sell_signals[next_open_day] = []
                        simulator.sell_signals[next_open_day].append([row['symbol'],next_open_price,next_open_day,simulator.portfolio['positions'][row['symbol']]['qty'],signal_info])
                        #simulator.execute_order('sell', row['symbol'], next_open_price, next_open_day, 
                        #                        simulator.portfolio['positions'][row['symbol']]['qty'])
                except:
                    continue
    def _process_buy_signals(self, date,  simulator,signals):
        """处理买入信号"""
        current_position = self.position_manager.current_position

        available_slots = self.position_limit - len(simulator.portfolio['positions'])
        if available_slots <= 0: 
            return
        
        # 获取买入信号
        if signals is None: 
            return
        scored_signals = signals[signals.index == date]
        if scored_signals.empty or scored_signals is None: 
            return
        holding_symbols = list(simulator.portfolio['positions'].keys())
        
        for _, row in scored_signals.iterrows():
            if available_slots>0:
                if row['symbol'] in holding_symbols: 
                    continue
                # 计算单支股票最大可投入资金（不超过总仓位10%）
                try:
                    next_open_price,next_open_day=self._get_next_open_price(date, row['symbol'])
                    next_open=self._get_next_open_day(date)
                    if next_open != next_open_day:
                        continue
                    # 计算总账户价值                    
                    total_account_value = self._total_value(date, simulator)
                    if (total_account_value-simulator.portfolio['cash'])/total_account_value>current_position:
                        continue
                    max_investment_cash = total_account_value*current_position//self.position_limit
                    max_investment = simulator.portfolio['cash']//available_slots
                    max_investment = min(max_investment, max_investment_cash)
                    max_afford = max_investment // (next_open_price * (1 + self.commission_rate))
                    max_afford = (max_afford // 100) * 100  # 这里进行按100取整操作
                    if max_afford > 0:
                        # 获取信号信息
                        symbol_data = scored_signals[scored_signals['symbol'] == row['symbol']]
                        signal_info = self.Report.generate_report(symbol_data)
                        if next_open_day not in simulator.buy_signals:
                            simulator.buy_signals[next_open_day] = []
                        simulator.buy_signals[next_open_day].append([row['symbol'],next_open_price,next_open_day,max_afford,signal_info,symbol_data.to_dict(orient='records')[0]])
                        #simulator.execute_order('buy', row['symbol'], next_open_price, next_open_day, max_afford)
                        available_slots-=1
                except:
                    continue
            else:
                break

    def _get_next_open_price(self, date, symbol):
        """获取下一个有效开盘价"""
        next_date = self._get_next_open_day(date)
        not_find_price = True
        while not_find_price:
            next_price = self.price_matrix.loc[next_date, ('open',symbol)]
            if pd.isna(next_price):
                next_date = self._get_next_open_day(next_date)
                if next_date is None:
                    return None,None
            else:
                not_find_price = False
        return next_price,next_date

    def _get_next_open_day(self,date):
        # 寻找下一个有效交易日
        if isinstance(date, str):
            # 如果是字符串，尝试转换为日期格式
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            date_obj = date
        next_date = date_obj + timedelta(days=1)

        while next_date.strftime('%Y-%m-%d') not in self.price_matrix.index:
            next_date += timedelta(days=1)
            if next_date > datetime.strptime(self.price_matrix.index[-1], '%Y-%m-%d').date():  # 修复类型不匹配问题
                return None
        return next_date.strftime('%Y-%m-%d')

    def _get_previous_open_day(self, date,symbol):
        """获取股票的前一个有效交易日"""
        if isinstance(date, str):
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            date_obj = date
        previous_date = date_obj - timedelta(days=1)
         # 获取价格矩阵的最早日期
        price_matrix_start_date = datetime.strptime(self.price_matrix.index[0], "%Y-%m-%d").date()

        while previous_date >= price_matrix_start_date:
            try:
                current_price = self.price_matrix.loc[previous_date.strftime("%Y-%m-%d"), symbol]
                if not pd.isna(current_price):
                    return previous_date.strftime("%Y-%m-%d")
            except:
                pass
            previous_date -= timedelta(days=1)
        return None

    def _check_stop_loss(self, date, simulator):
        """执行止损检查"""
        if isinstance(date, str):
            # 如果是字符串，尝试转换为日期格式
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            date_obj = date
        for symbol, pos in list(simulator.portfolio['positions'].items()):            
            if  pos['entry_date']== date_obj.strftime('%Y-%m-%d'):
                continue
            current_price = self.price_matrix.loc[date_obj.strftime('%Y-%m-%d'), ('close',symbol)]
            # 如果获取到价格，则更新 pos['latest_price']
            if current_price == current_price:
                pos['latest_price'] = current_price
                cost_basis = pos['cost'] / pos['qty']
                # 成本止损
                if current_price <= cost_basis * 0.9:
                    try:
                        next_open_price,next_open_day = self._get_next_open_price(date_obj.strftime('%Y-%m-%d'), symbol)
                        if next_open_price:
                            # 记录止损信息
                            signal_info = {
                                'type': 'stop_loss',
                                'reason': f"价格跌至成本价的90%以下，触发止损。"
                            }
                            if next_open_day not in simulator.sell_signals:
                                simulator.sell_signals[next_open_day] = []
                            simulator.sell_signals[next_open_day].append([symbol,next_open_price,next_open_day,simulator.portfolio['positions'][symbol]['qty'],signal_info])
                            #simulator.execute_order('sell', symbol, next_open_price, next_open_day, pos['qty'])
                    except:
                        continue
            '''
            # 跟踪止损
            if 'highest_price' not in pos:
                pos['highest_price'] = current_price
            else:
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                elif current_price <= pos['highest_price'] * (1 - self.trailing_stop_ratio):
                    try:
                        next_open_price,next_open_day = self._get_next_open_price(date_obj.strftime('%Y-%m-%d'), symbol)
                        if next_open_price:
                            # 记录跟踪止损信息
                            signal_info = {
                                'type': 'trailing_stop',
                                'reason': f"价格回撤达到跟踪止损比例，触发止损。"
                            }
                            if next_open_day not in simulator.sell_signals:
                                simulator.sell_signals[next_open_day] = []
                            simulator.sell_signals[next_open_day].append([symbol, next_open_price, next_open_day,simulator.portfolio['positions'][symbol]['qty'], signal_info])
                    except:
                        continue
            '''
            
           

    def _record_daily_value(self, date, simulator):
        total_value = self._total_value(date,simulator)
        simulator.portfolio['history'].append({'date': date, 'value': total_value})
         # 更新仓位管理
        current_position = self.position_manager.update_position(total_value)
        #simulator.position_limit = int(current_position * self.position_limit_base)

    def _total_value(self, date, simulator):
        """计算当前组合价值"""
        position_value = 0.0
        for symbol, pos in simulator.portfolio['positions'].items():
            position_value += pos['latest_price'] * pos['qty']  # 使用最新价格计算市值
        total_value = simulator.portfolio['cash'] + position_value
        return total_value
    
    def _generate_report(self, simulator):
        """生成详细报告"""
        df = pd.DataFrame(simulator.portfolio['history']).set_index('date')
        df['returns'] = df['value'].pct_change()
        trades = pd.DataFrame(simulator.portfolio['history'])
        
        now=str(time.time()).split('.')[0]
        trade_filename = f'./backtestresult/trades_report_{now}.xlsx'
        with pd.ExcelWriter(trade_filename) as writer:
            # 保存交易记录
            trades.to_excel(writer, sheet_name='Transactions', index=False)
            
            # 保存回测报告摘要
            summary_df = pd.DataFrame([{
                'Final Value': df['value'].iloc[-1],
                'Total Return': df['value'].iloc[-1] / 5e5 - 1,
                'Max Drawdown': (df['value'] / df['value'].cummax() - 1).min(),
                'Turnover': trades.groupby('symbol').size().mean()
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        return {
            'summary': {
                'final_value': df['value'].iloc[-1],
                'total_return': df['value'].iloc[-1] / 5e5 - 1,
                'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
                'turnover': trades.groupby('symbol').size().mean()
            },
            'trades': trades
        }

    def _update_live_plot(self, simulator):
        """更新实时图表"""
        if not simulator.portfolio['history'][-1]:
            return
        latest_value = simulator.portfolio['history'][-1]['value']
        date_str = simulator.portfolio['history'][-1]['date']
        latest_cash = simulator.portfolio['cash']
        
        # 将日期字符串转换为日期对象
        if isinstance(date_str, str):
            latest_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            latest_date = date_str
        
        self.dates.append(latest_date)
        self.values.append(latest_value)
        
        # 更新净值曲线数据
        self.value_line.set_data(self.dates, self.values)
        self.ax_value.relim()
        self.ax_value.autoscale_view()
        
        # 自动调整日期格式
        self.ax_value.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.fig.autofmt_xdate()
        
        # 更新持仓信息显示
        self._info_callback(
            msg="Update",
            data={
                "current_date": latest_date,
                "current_value": latest_value,
                "cash": latest_cash,
                "positions": simulator.portfolio["positions"]
            }
        )
        # 调整布局，避免左侧文字覆盖右侧图表
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# 使用示例
if __name__ == "__main__":  
    # 运行回测
    orchestrator = BacktestOrchestrator(live_plot=True)
    report = orchestrator.run(start_date='2019-01-01', end_date='2024-12-31')    
    print("回测结果摘要:")
    print(f"最终净值: {report['summary']['final_value']:,.2f}")
    print(f"总收益率: {report['summary']['total_return']:.2%}")
    print(f"最大回撤: {report['summary']['max_drawdown']:.2%}")