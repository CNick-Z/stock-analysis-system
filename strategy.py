# strategy.py
import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, Tuple
from datetime import datetime
from db_operations import *
from tqdm import tqdm
import talib as ta

class StockScorer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'weights': {
                'technical': 0.20,  # 调整权重
                'capital_flow': 0.35,  # 新增资金流向权重
                'fundamental': 0.25,
                'market_heat': 0.2  # 调整市场热度权重
            },
             # 新增资金流子项权重配置
            'capital_flow_weights': {
                'positive_flow': 0.2,    # 资金流入
                'flow_increasing': 0.25,  # 流入加速
                'trend_strength': 0.3,    # 趋势强度
                'weekly_flow': 0.15,      # 周级别流入
                'weekly_increasing': 0.1  # 周流入加速
            },
            'fundamental_metrics': ['pe_ratio', 'roe', 'profit_growth'],
            'heat_window': 30  # 市场热度计算窗口
        }

    def _get_fundamental_score(self, symbol: str) -> float:
        """获取财务指标评分（带缓存机制）"""
        '''
        try:
            # 从数据库获取预存财务数据
            conn = sqlite3.connect('./db/stock_data.db')
            query = f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                latest = df.iloc[-1]
                return sum(
                    latest[metric] * self.config['fundamental_weights'][metric]
                    for metric in self.config['fundamental_metrics']
                )
                
            # 实时获取作为后备
            df = ak.stock_financial_report_sina(symbol=symbol, indicator="主要指标")
            return df['net_profit'].iloc[-1] / df['revenue'].iloc[-1]
        except Exception as e:
            print(f"财务数据获取失败 {symbol}: {str(e)}")
            return 0
        '''
        return 0

    def _calculate_technical_score(self, row: pd.Series) -> float:
        """技术指标评分（基于策略信号）"""
        tech_score = 0
        # 均线系统评分
        tech_score += (row['ma_5'] > row['ma_20']) * 0.25
        tech_score += (row['angle_ma_10'] > 30) * 0.15
        tech_score += (row['macd'] > row['macd_signal']) * 0.30
        
        # 成交量动能
        volume_score = min(row['volume'] / row['volume_ma5'], 3)  # 限制最大3倍
        tech_score += volume_score * 0.3

        # 超买超卖评分（新增）
        tech_score += (row['rsi_14'] < 70) * 0.1  # 未超买加分
        tech_score += (row['kdj_k'] < 80) * 0.1   # KDJ 未超买加分
        tech_score += (row['cci_20'] < 100) * 0.1  # CCI 未超买加分
        tech_score += (row['close'] < row['bb_upper']) * 0.1  # 布林带未超买加分
        
        return tech_score

    def _calculate_capital_flow(self, row: pd.Series) -> float:
        """基于策略生成的新资金流信号进行评分"""
        flow_score = 0
        
        # 资金流入基础分
        if row['money_flow_positive']:
            flow_score += self.config['capital_flow_weights']['positive_flow'] * 1.2  # 正值强化
            
        # 流入加速
        if row['money_flow_increasing']:
            flow_score += self.config['capital_flow_weights']['flow_increasing'] * 1.0
            
        # 趋势强度
        if row['money_flow_trend']:
            trend_strength = min(row['主生量'] / row['量基线'], 2.0)  # 限制最大2倍
            flow_score += self.config['capital_flow_weights']['trend_strength'] * trend_strength
            
        # 周级别资金流
        if row['money_flow_weekly']:
            flow_score += self.config['capital_flow_weights']['weekly_flow'] * 1.0
            
        # 周流入加速
        if row['money_flow_weekly_increasing']:
            flow_score += self.config['capital_flow_weights']['weekly_increasing'] * 1.5  # 加速给予更高权重
            
        # 量增幅强化
        if row['量增幅'] > 10:  # 显著增长
            flow_score *= 1.2
        elif row['量增幅'] < -5:  # 显著减少
            flow_score *= 0.8
            
        return min(flow_score, 1.0)  # 限制最大1分

    

    def score_daily_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """每日信号综合评分"""
        for _, row in tqdm(signals.iterrows(), total=len(signals), desc="每日评分"):                
            # 计算各维度评分
            technical_score = self._calculate_technical_score(row)
            capital_flow_score = self._calculate_capital_flow(row)
            fundamental_score = self._get_fundamental_score(row['symbol'])
            market_heat_score = row['market_heat']
            # 加权总分
            weights = self.config['weights']
            total_score = (
                technical_score * weights['technical'] +
                capital_flow_score * weights['capital_flow'] +
                fundamental_score * weights['fundamental'] +
                market_heat_score * weights['market_heat']
            )
            
            # 将评分结果添加到行中
            signals.loc[row.name, 'technical'] = technical_score
            signals.loc[row.name, 'capital_flow'] = capital_flow_score
            signals.loc[row.name, 'fundamental'] = fundamental_score
            signals.loc[row.name, 'market_heat'] = market_heat_score
            signals.loc[row.name, 'total_score'] = total_score
        
        return signals

    def select_top_stocks(self, signals: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """选股主流程"""
        # 评分
        scored_df = self.score_daily_signals(signals)
        if scored_df.empty: return None
        scored_df = scored_df[scored_df['total_score'] > 1]
        if scored_df.empty:
            return None
        else:
            # 每日TopN选择
            final_picks = []
            for date in scored_df['date'].unique():
                daily = scored_df[scored_df['date'] == date]
                top = daily.nlargest(top_n, 'total_score')
                final_picks.append(top)
                
            return pd.concat(final_picks)
class EnhancedTDXStrategy:
    def __init__(self, 
                db_path='c:/db/stock_data.db',
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
        self.db_manager = DatabaseManager(f"sqlite:///{db_path}")
        self.db_manager.ensure_tables_exist()
        self.Scorer = StockScorer()
        
        # 预校验参数
        self._validate_config()

    def _validate_config(self):
        """参数有效性验证"""
        if not all(isinstance(x, int) and x > 0 for x in self.config['ma_windows']):
            raise ValueError("均线周期必须为正整数")
        if len(self.config['macd_params']) != 3:
            raise ValueError("MACD参数需要三个值(short, long, signal)")

    def _fetch_precalculated_data(self, start_date: str,end_date:str) -> pd.DataFrame:
        """
        获取数据，用于计算技术指标和生成信号
        参数：
        - start_date: 日期字符串（如 '2024-01-01'）
        - end_date: 日期字符串（如 '2024-01-01'）
        返回：
        包含
        返回包含以下字段的DataFrame：
        [date, symbol, ma_5, ma_10, ma_20, ma_55, ma_240, macd, macd_signal, close, volume]
        """
        extended_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=7)
        extended_start_date = extended_start_date.strftime("%Y-%m-%d")
        end_date = end_date
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
        print("加载技术指标数据...")
        tech_df = self.db_manager.load_data(
            table=TechnicalIndicators,
            filter_conditions={
                'date': {'$between':[extended_start_date, end_date]}  # 需要扩展过滤逻辑
            }
        ).rename(columns=tech_columns)
        
        # 加载行情数据
        print("加载行情数据...")
        price_df = self.db_manager.load_data(
            table=DailyData,
            filter_conditions={
                'date': {'$between':[extended_start_date, end_date]}  # 需要扩展过滤逻辑
            },
            columns=['date', 'symbol','high','low','volume','amount','open','close'],
            distinct_column=None,
            limit=None
        )
        
        #加载基础数据
        print("加载基础数据...")
        info_df = self.db_manager.load_data(
            table=StockBasicInfo,
            filter_conditions={
                'name': {'$not_like': 'ST'}  # 需要扩展过滤逻辑
            },
            columns=['symbol', 'name', 'total_shares', 'industry']
        )

        # 合并数据集
        print("合并数据集...")
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

    

    def _calculate_angle(self,series, window):
        """
        根据选股公式计算角度：
        角度MA := ATAN((MA / REF(MA, 2) - 1) * 100) * 180 / π
        """
        # 计算前两期的 MA 值
        ref_ma = series.shift(window)
        
        # 计算比值变化率
        ratio_change = (series / ref_ma - 1) * 100
        
        # 计算角度
        angle = np.degrees(np.arctan(ratio_change))
        
        return angle

    def _calculate_ma_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线角度（与选股公式一致）
        """
        # 定义需要计算角度的 MA 列
        ma_columns = ['ma_10', 'ma_20', 'ma_55', 'ma_240']
        
        for ma in ma_columns:
            df[f'angle_{ma}'] = self._calculate_angle(df[ma], window=2)        
        return df


    def _calculate_xvl(self, df: pd.DataFrame) -> pd.Series:
        """
        向量化计算 XVL（资金流向）。
        """
        # 阳线、阴线和平线的条件
        is_positive = df['close'] > df['open']
        is_negative = df['close'] < df['open']
        is_flat = df['close'] == df['open']

        # 阳线的计算
        flow_in_positive = df['QJJ'] * (df['high'] - df['low'])
        flow_out_positive = -df['QJJ'] * (df['high'] - df['close'] + df['open'] - df['low'])

        # 阴线的计算
        flow_in_negative = df['QJJ'] * (df['high'] - df['open'] + df['close'] - df['low'])
        flow_out_negative = -df['QJJ'] * (df['high'] - df['low'])

        # 平线的计算
        flow_in_flat = df['volume'] / 2
        flow_out_flat = -df['volume'] / 2

        # 使用条件选择计算结果
        flow_in = (
            flow_in_positive.where(is_positive, 
            flow_in_negative.where(is_negative, flow_in_flat))
        )
        flow_out = (
            flow_out_positive.where(is_positive, 
            flow_out_negative.where(is_negative, flow_out_flat))
        )

        # 返回资金流向的总和
        return flow_in + flow_out

    def _calculate_multi_period_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多周期MACD共振指标
        参数:
            df: 包含日线数据的DataFrame
        返回:
            增加周线、月线MACD信号的DataFrame
        """
        # 生成周线数据（5交易日）
        weekly_df = df.resample('W-Mon', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        # 计算周线MACD
        weekly_macd, weekly_signal, _ = ta.MACD(
            weekly_df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        weekly_df['weekly_macd'] = weekly_macd
        weekly_df['weekly_signal'] = weekly_signal
        
        # 生成月线数据（20交易日）
        monthly_df = df.resample('M', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        # 计算月线MACD
        monthly_macd, monthly_signal, _ = ta.MACD(
            monthly_df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        monthly_df['monthly_macd'] = monthly_macd
        monthly_df['monthly_signal'] = monthly_signal
        
        # 合并多周期数据
        df = df.merge(
            weekly_df[['weekly_macd', 'weekly_signal']], 
            left_on='date', 
            right_index=True, 
            how='left'
        )
        df = df.merge(
            monthly_df[['monthly_macd', 'monthly_signal']], 
            left_on='date', 
            right_index=True, 
            how='left'
        )
        
        # 填充缺失值（向前填充）
        df[['weekly_macd', 'weekly_signal', 'monthly_macd', 'monthly_signal']] = \
            df[['weekly_macd', 'weekly_signal', 'monthly_macd', 'monthly_signal']].ffill()
        
        return df

    def _calculate_money_flow_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # 假设 _calculate_xvl 是一个向量化的函数，否则需要改写为向量化形式
        df['XVL'] = self._calculate_xvl(df)  # 确保 _calculate_xvl 支持向量化

        # 6. 计算ZLL（成交量占比）
        df['ZLL'] = df['volume'] / df['total_shares']

        # 7. 计算LIJIN1（限制ZLL的最大值为10）
        df['LIJIN1'] = df['ZLL'].clip(upper=10)

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
        df['量增幅'] = df['QZJJ'].where(
            (df['LLJX'] > 0) & (df['ZJLL'] < 0),
            df['QZJJ'].where(
                (df['LLJX'] < 0) & (df['ZJLL'] < 0) & (df['LLJX'] < df['ZJLL']),
                -df['QZJJ']
            )
        ).abs()

        # 15. 计算力度（LIJIN的缩放值）
        df['力度'] = df['LIJIN'] / 1000

        # 16. 计算周量（过去5日的LLJX总和）
        df['周量'] = df['LLJX'].rolling(window=5).sum()

        # 17. 计算周增幅（周量的变化率）
        df['BB'] = df['周量'].shift(1)
        df['ZQZJJ'] = ((df['周量'] - df['BB']) / df['BB']) * 100
        df['周增幅'] = df['ZQZJJ'].where(
            (df['周量'] > 0) & (df['BB'] < 0),
            df['ZQZJJ'].where(
                (df['周量'] < 0) & (df['BB'] < 0) & (df['周量'] < df['BB']),
                -df['ZQZJJ']
            )
        ).abs()

        return df

    '''
    def _calculate_volume_price_resonance(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            计算量价共振三重滤波顶底背离指标
            """
            # 计算量能饱和度
            df['hhv_amount'] = df['volume'].rolling(window=20).max()
            df['hhv_close'] = df['close'].rolling(window=20).max()
            df['volume_saturation'] = (df['volume'] / df['close']) / (df['hhv_amount'] / df['hhv_close']) * 100
            df['volume_saturation'] = df['volume_saturation'].apply(lambda x: 100 if x > 100 else x)
            
            # 计算支撑和压力线
            df['support'] = df['low'].rolling(window=30).min().rolling(window=2).mean()
            df['resistance'] = df['high'].rolling(window=30).max().rolling(window=2).mean()
            
            
            # 计算动量滤波
            df['波段'] = df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
            df['趋势线'] = df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # 计算顶底背离
            df['RSV'] = (df['close'] - df['low'].rolling(9).min()) / (df['high'].rolling(9).max() - df['low'].rolling(9).min()) * 100
            df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
            df['D'] = df['K'].ewm(com=2, adjust=False).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            
            df['顶背离'] = (df['close'] > df['close'].shift(1)) & (df['K'] < df['K'].shift(1)) & (df['D'] < df['D'].shift(1))
            df['底背离'] = (df['close'] < df['close'].shift(1)) & (df['K'] > df['K'].shift(1)) & (df['D'] > df['D'].shift(1))
            
            return df
    '''

    def _generate_core_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成核心交易信号（保持原有信号逻辑不变）
        """
        #涨幅条件
        #df['growth'] = (df['close'] - df['open']) / df['open'] * 100
        #df['growth'] = df['growth'].round(1)
        df['growth'] = (df['close']>=df['open']*1.03) & (df['high']<=df['open']*1.05)

        # 计算多周期MACD
        print('计算多周期MACD...')
        df = self._calculate_multi_period_macd(df)

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

        #df['macd_daily_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) < df['macd_signal'].shift(1))
        df['macd_weekly_jc'] = (df['weekly_macd'] > df['weekly_signal']) & (df['weekly_macd'].shift(1) < df['weekly_signal'].shift(1))
        df['macd_monthly_jc'] = (df['monthly_macd'] > df['monthly_signal']) & (df['monthly_macd'].shift(1) < df['monthly_signal'].shift(1))
        
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
        print('生成资金流向指标...')
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
    def _calculate_market_heat(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场热度，并将结果整合到原始数据中。
        市场热度基于股票的成交量计算，按日计算每个股票的市场热度。
        """
        # 确保数据中包含必要的列
        required_columns = ['date', 'symbol', 'volume', 'industry']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"数据中缺少必要的列：{required_columns}")
        
        # 按日期和行业分组，计算每个日期每个行业的平均成交量
        data['market_heat'] = data.groupby(['date', 'industry'])['volume'].transform(lambda x: x / x.mean())
        
        # 返回包含市场热度的完整数据集
        return data

    def generate_features(self, start_date: str,end_date:str) -> pd.DataFrame:
        """
        生成完整信号（优化执行流程）
        """
        # 数据获取
        print("正在获取数据...")
        raw_data = self._fetch_precalculated_data(start_date,end_date)
        
        # 特征工程
        print("正在生成特征...")
        with_angles = self._calculate_ma_angles(raw_data)
        
        # 信号生成
        print("正在生成信号...")
        signals = self._generate_core_signals(with_angles)

        # 计算市场热度
        print("正在计算市场热度...")
        market_heat = self._calculate_market_heat(signals)

        # 计量价共振三重滤波顶底背离指标
        #volume_price_resonance = self._calculate_volume_price_resonance(raw_data)

        return market_heat
    def get_signals(self, start_date: str,end_date:str) -> pd.DataFrame:        
        """获取买点信号"""
        print("计算买点信号...")
        signals = self.generate_features(start_date,end_date)
        # 综合买入条件
        buy_condition = (
            signals['growth'] &
            signals['ma_condition'] &
            signals['angle_condition'] &
            signals['volume_condition'] &
            signals['macd_condition'] &
            (signals['jc_condition']|signals['macd_jc']) &  # JC 条件
            (signals['ma_20'] < signals['ma_55']) &  # MA20 < MA55
            (signals['ma_55'] > signals['ma_240'])  # MA55 > MA240
            #(signals['close'] > signals['support'])  # 价格在支撑线上方
            #(signals['底背离'])  # 底背离信号
        )

        buy_signals= signals[buy_condition][[
            'date', 'symbol', 'name',  'industry','ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'volume_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14','macd_jc',
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close','money_flow_positive',
            'money_flow_increasing','money_flow_trend','money_flow_weekly','money_flow_weekly_increasing','量增幅','量基线','主生量','growth','market_heat'
        ]].assign(signal_type='buy')

        selected_buy_signals = self.Scorer.select_top_stocks(buy_signals,5)

        if selected_buy_signals is not None:
            selected_buy_signals=selected_buy_signals.set_index('date')       
        # 综合卖出条件
        print("计算卖点信号...")
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

        profitable_sell_condition = (           
                (signals['close'] < signals['ma_20']) &  # 价格跌破20日均线
                (signals['money_flow_trend'] == False)  # 资金流向趋势转弱

        )


        # 合并卖出条件
        combined_sell_condition =  sell_condition |profitable_sell_condition
        
        sell_signals= signals[combined_sell_condition][[
            'date', 'symbol',  'name',  'industry','ma_5', 'ma_10', 'ma_20', 'angle_ma_10',
            'volume_ma5', 'macd', 'macd_signal', 'volume', 'rsi_14','macd_jc',
            'kdj_k', 'kdj_d', 'cci_20', 'williams_r', 'bb_upper',
            'bb_middle', 'bb_lower','close','money_flow_positive',
            'money_flow_increasing','money_flow_trend','money_flow_weekly','money_flow_weekly_increasing','量增幅','量基线','主生量','growth','market_heat'
        ]].assign(signal_type='sell')
        
        #scored_sell_signals = self.Scorer.score_daily_signals(sell_signals)
        return [selected_buy_signals, sell_signals.set_index('date')]
