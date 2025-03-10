# backtester.py
from db_operations import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from datetime import datetime, timedelta
from stock_report import  *
from strategy import *

class DatabaseIntegrator:
    """数据库集成模块"""
    def __init__(self, db_path):
        self.db_manager = DatabaseManager(f"sqlite:///{db_path}")
        self.db_manager.ensure_tables_exist()
        
    def fetch_trading_dates(self) -> pd.DatetimeIndex:
        """获取所有交易日历"""
        dates = self.db_manager.load_data(DailyData, {}, distinct_column='date')
        return pd.to_datetime(dates['date']).sort_values()

    def load_price_matrix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载价格矩阵（宽表格式）"""
        filter_conditions = {
            'date': {
                '$between': [start_date, end_date]
            }
        }
        df = self.db_manager.load_data(DailyData, filter_conditions, columns=['date', 'symbol', 'open'])
        return df.pivot(index='date', columns='symbol', values='open')


class TradingSimulator:
    """交易模拟引擎"""
    def __init__(self, initial_capital=5e5, commission=0.0003,position_limit=5):
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: {'qty': int, 'cost': float}}
            'history': []
        }
        self.commission_rate = commission
        self.position_limit = position_limit  # 替换原来的硬编码
        self.buy_signals = {}
        self.sell_signals = {}
        
    def execute_order(self, order_type: str, symbol: str, price: float, date: datetime, quantity: int,signal_info):
        """执行订单并更新持仓"""
        if order_type == 'buy':
            return self._execute_buy(symbol, price, date, quantity,signal_info)
        elif order_type == 'sell':
            return self._execute_sell(symbol, price, date, quantity,signal_info)
        
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
        print(f"{date} 买入 {symbol}，价格: {price}，数量: {quantity}，佣金: {commission}")
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
        print(f"{date} 卖出 {symbol}，价格: {price}，数量: {quantity}，佣金: {commission}")
        return True

class BacktestOrchestrator:
    """回测总控模块"""
    def __init__(self, db_path='c:/db/stock_data.db', live_plot=False,position_limit=10, commission_rate=0.0003):
        self.position_limit = position_limit  # 新增参数
        self.db = DatabaseIntegrator(db_path)
        self.strategy = EnhancedTDXStrategy()
        self.Report = StockReport()
        self.Scorer = StockScorer()
        self.price_matrix = None
        self.trading_dates = None
        self.live_plot = live_plot
        self.commission_rate = commission_rate 
        self.trailing_stop_ratio = 0.1
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
                    f"{symbol}: QTY={pos['qty']}, Cost={pos['cost']:.2f}",
                    transform=self.ax_info.transAxes,
                    fontsize=10,
                    verticalalignment='top'
                )
                text_y -= line_spacing  # 调整下一行的垂直位置
    def initialize(self, start_date, end_date):
        """初始化回测环境"""
        self.trading_dates = self.db.fetch_trading_dates()
        self.price_matrix = self.db.load_price_matrix(start_date, end_date)
        

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
        
        # 获取回测日期范围
        dates = self.trading_dates[
            (self.trading_dates >= pd.to_datetime(start_date)) & 
            (self.trading_dates < pd.to_datetime(end_date))
        ]
        all_buy_signals = pd.DataFrame()
        all_sell_signals = pd.DataFrame()
        years=self.date_by_year(start_date, end_date)
        for year_start, year_end in years:
            year_buy_signals,year_sell_signals=self.strategy.get_signals(start_date,end_date)
            all_buy_signals=pd.concat([all_buy_signals,year_buy_signals])
            all_sell_signals=pd.concat([all_sell_signals,year_sell_signals])

        for date in tqdm(dates, desc="回测进度"):
            date=date.strftime("%Y-%m-%d")
            print('\n'+date)
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
                        simulator.execute_order('buy', info[0],info[1],info[2],info[3],info[4])
                    del simulator.buy_signals[date]
                # 处理卖出信号
            #print("处理卖出信号")
            self._process_sell_signals(date,simulator,all_sell_signals)
            # 检查止损
            #print("检查止损")
            self._check_stop_loss(date, simulator)            
            # 处理买入信号
            #print("处理买入信号")
            self._process_buy_signals(date,simulator,all_buy_signals)          
            # 记录每日净值
            self._record_daily_value(date, simulator)            
            # 更新实时图表
            if self.live_plot:
                self._update_live_plot(simulator)            
        return self._generate_report(simulator)

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
            current_price = self.price_matrix.loc[date, symbol]
            cost_per_share = pos['cost'] / pos['qty']
            if current_price > cost_per_share:  # 筛选出盈利的持仓
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
        available_slots = self.position_limit - len(simulator.portfolio['positions'])
        if available_slots <= 0: 
            return
        
        # 获取买入信号
        scored_signals = signals[signals.index == date]
        if scored_signals.empty: 
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
                    max_investment_cash = total_account_value//self.position_limit
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
                        simulator.buy_signals[next_open_day].append([row['symbol'],next_open_price,next_open_day,max_afford,signal_info])
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
            next_price = self.price_matrix.loc[next_date, symbol]
            if pd.isna(next_price):
                next_date = self._get_next_open_day(next_date)
                if next_date is None:
                    return None,None
            else:
                not_find_price = False
        return self.price_matrix.loc[next_date, symbol],next_date

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
            current_price = self.price_matrix.loc[date_obj.strftime('%Y-%m-%d'), symbol] 
            cost_basis = pos['cost'] / pos['qty']
            if  simulator.portfolio['positions'][symbol]['entry_date']== date_obj.strftime('%Y-%m-%d'):
                continue
            # 成本止损
            if current_price <= cost_basis * 0.85:
                try:
                    next_open_price,next_open_day = self._get_next_open_price(date_obj.strftime('%Y-%m-%d'), symbol)
                    if next_open_price:
                        # 记录止损信息
                        signal_info = {
                            'type': 'stop_loss',
                            'reason': f"价格跌至成本价的85%以下，触发止损。成本价：{cost_basis}, 当前价格：{current_price}"
                        }
                        if next_open_day not in simulator.sell_signals:
                            simulator.sell_signals[next_open_day] = []
                        simulator.sell_signals[next_open_day].append([row['symbol'],next_open_price,next_open_day,simulator.portfolio['positions'][row['symbol']]['qty'],signal_info])
                        #simulator.execute_order('sell', symbol, next_open_price, next_open_day, pos['qty'])
                except:
                    continue
            
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
                                'reason': f"价格回撤达到跟踪止损比例，触发止损。最高价：{pos['highest_price']}, 当前价格：{current_price}"
                            }
                            if next_open_day not in simulator.sell_signals:
                                simulator.sell_signals[next_open_day] = []
                            simulator.sell_signals[next_open_day].append([symbol, next_open_price, next_open_day,simulator.portfolio['positions'][symbol]['qty'], signal_info])
                    except:
                        continue
            
           

    def _record_daily_value(self, date, simulator):
        total_value = self._total_value(date,simulator)
        simulator.portfolio['history'].append({'date': date, 'value': total_value})

    def _total_value(self, date, simulator):
        """计算当前组合价值"""
        position_value = 0.0
        for symbol, pos in simulator.portfolio['positions'].items():
            try:
                current_price = self.price_matrix.loc[date, symbol]
                if pd.isna(current_price):
                    # 如果当前价格为 NaN，尝试使用前一交易日的价格
                    previous_date = self._get_previous_open_day(date,symbol)
                    if previous_date:
                        current_price = self.price_matrix.loc[previous_date, symbol]
                    else:
                        continue
            except KeyError:
                # 如果日期或股票符号不在价格矩阵中，跳过该股票的计算
                continue
            position_value += current_price * pos['qty']
        total_value = simulator.portfolio['cash'] + position_value
        return total_value
    
    def _generate_report(self, simulator):
        """生成详细报告"""
        df = pd.DataFrame(simulator.portfolio['history']).set_index('date')
        df['returns'] = df['value'].pct_change()
        
        plt.figure(figsize=(12,6))
        df['value'].plot(title='Portfolio Value Curve')
        plt.savefig('portfolio_curve.png')
        
        trades = pd.DataFrame(simulator.portfolio['history'])
        trade_filename = 'trades_report.xlsx'
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
    report = orchestrator.run(start_date='2016-03-13', end_date='2018-03-21')    
    print("回测结果摘要:")
    print(f"最终净值: {report['summary']['final_value']:,.2f}")
    print(f"总收益率: {report['summary']['total_return']:.2%}")
    print(f"最大回撤: {report['summary']['max_drawdown']:.2%}")