# strategies/score_strategy.py
# 旧评分策略插件（继承 EnhancedTDXStrategy 的核心逻辑）

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class ScoreStrategy(BaseStrategy):
    """
    评分策略插件

    使用综合评分系统选股：
    - 技术面评分（均线、MACD、成交量、RSI、KDJ等）
    - 资金流评分
    - 市场热度

    保持原有 EnhancedTDXStrategy 的核心逻辑不变，
    适配插件架构接口。
    """

    def __init__(self, db_path: str = None, config: dict = None):
        # 默认配置
        default_config = {
            'ma_windows': (5, 10, 20, 55, 240),
            'angle_window': 5,
            'volume_ma_window': 5,
            'macd_params': (12, 26, 9),
            'buy_conditions': {
                'growth_threshold': 1.00,
                'max_upper_shadow': 1.06,
                'ma_diff_threshold': 0.02
            },
            # 选股参数
            'top_n': 5,  # 每日选股数量
        }
        self.config = {**default_config, **(config or {})}

        super().__init__(db_path, self.config)

        # 初始化内部组件
        from utils.parquet_db import ParquetDatabaseIntegrator
        from utils.db_operations import StockBasicInfo

        self.db_manager = ParquetDatabaseIntegrator(db_path)
        self.stock_info_cache = None

    def _get_stock_basic_info(self):
        """获取股票基础信息（带缓存）"""
        if self.stock_info_cache is None:
            from utils.db_operations import StockBasicInfo
            self.stock_info_cache = self.db_manager.load_data(
                table_class=StockBasicInfo,
                filter_conditions={'name': {'$not_like': 'ST'}},
                columns=['symbol', 'name', 'total_shares', 'industry']
            )
        return self.stock_info_cache

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """生成特征数据"""
        from utils.stock_report import StockReport
        from scipy.stats import linregress
        import talib as ta

        # 扩展日期范围（技术指标需要前置数据）
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=30)
        extended_start_str = extended_start.strftime("%Y-%m-%d")

        # 加载技术指标
        tech_columns = {
            'date': 'date', 'symbol': 'symbol',
            f"sma_{self.config['ma_windows'][0]}": 'ma_5',
            f"sma_{self.config['ma_windows'][1]}": 'ma_10',
            f"sma_{self.config['ma_windows'][2]}": 'ma_20',
            f"sma_{self.config['ma_windows'][3]}": 'ma_55',
            f"sma_{self.config['ma_windows'][4]}": 'ma_240',
            'vol_ma5': 'volume_ma5',
            'macd': 'macd', 'macd_signal': 'macd_signal',
            'macd_histogram': 'macd_histogram',
            'rsi_14': 'rsi_14',
            'kdj_k': 'kdj_k', 'kdj_d': 'kdj_d', 'kdj_j': 'kdj_j',
            'cci_20': 'cci_20',
            'williams_r': 'williams_r',
            'bb_upper': 'bb_upper', 'bb_middle': 'bb_middle', 'bb_lower': 'bb_lower'
        }

        tech_df = self.db_manager.load_data(
            table_class=type('TechnicalIndicatorsBase', (), {}),
            filter_conditions={'date': {'$between': [extended_start_str, end_date]}}
        )
        if hasattr(tech_df, 'rename'):
            tech_df = tech_df.rename(columns=tech_columns)

        price_df = self.db_manager.load_data(
            table_class=type('DailyDataBase', (), {}),
            filter_conditions={'date': {'$between': [extended_start_str, end_date]}},
            columns=['date', 'symbol', 'high', 'low', 'volume', 'amount', 'open', 'close']
        )

        info_df = self._get_stock_basic_info()

        # 合并
        df = pd.merge(tech_df, price_df, on=['date', 'symbol'], how='inner')
        df = pd.merge(df, info_df, on=['symbol'], how='inner')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 计算均线角度
        df = self._calculate_ma_angles(df)

        # 计算核心信号
        df = self._generate_core_signals(df)

        # 计算市场热度
        df = self._calculate_market_heat(df)

        return df

    def _calculate_ma_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线角度"""
        ma_columns = {'ma_10': 'ma_5', 'ma_20': 'ma_10', 'ma_55': 'ma_20', 'ma_240': 'ma_55'}
        for long_ma, short_ma in ma_columns.items():
            df[long_ma] = df[long_ma].fillna(df[short_ma])
            ref_ma = df[long_ma].shift(2)
            ratio_change = (df[long_ma] / ref_ma - 1) * 100
            df[f'angle_{long_ma}'] = np.degrees(np.arctan(ratio_change))
        return df

    def _generate_core_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成核心交易信号"""
        buy_cfg = self.config['buy_conditions']

        # 涨幅条件
        df['growth'] = (df['close'] - df['open']) / df['open'] * 100
        df['growth'] = df['growth'].round(1)
        df['growth_condition'] = (
            (df['close'] >= df['open'] * buy_cfg['growth_threshold']) &
            (df['high'] <= df['open'] * buy_cfg['max_upper_shadow'])
        )

        # 均线条件
        df['ma_condition'] = (
            (df['ma_5'] > df['ma_10']) &
            (df['ma_10'] < df['ma_20'])
        )

        # 角度条件
        df['angle_condition'] = (
            (df['angle_ma_10'] > 0) &
            df['angle_ma_240'].between(-20, 20) &
            df['angle_ma_20'].between(-40, 35)
        )

        # 成交量条件
        df['volume_condition'] = (
            (df['volume'] > df['volume'].shift(1) * 1.5) |
            (df['volume'] > df['volume_ma5'] * 1.2)
        )

        # MACD条件
        df['macd_condition'] = (
            (df['macd'] < 0) &
            (df['macd'] > df['macd_signal'])
        )

        # MACD金叉
        df['macd_jc'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) < df['macd_signal'].shift(1))
        )

        # JC条件
        df['jc_condition'] = (
            (df['ma_5'] > df['ma_5'].shift(1)) &
            (df['ma_20'] > df['ma_20'].shift(1)) &
            (abs(df['ma_5'] - df['ma_20']) / df['ma_20'] < buy_cfg['ma_diff_threshold'])
        )

        # 超买超卖
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
        df['bollinger_condition'] = df['close'] < df['bb_upper']

        # 资金流向
        df = self._calculate_money_flow_indicators(df)
        df['money_flow_positive'] = df['资金量'] > 0
        df['money_flow_increasing'] = df['量增幅'] > 0
        df['money_flow_trend'] = df['主生量'] > df['量基线']
        df['money_flow_weekly'] = df['周量'] > 0
        df['money_flow_weekly_increasing'] = df['周增幅'] > 0

        return df

    def _calculate_money_flow_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资金流向指标"""
        df['金'] = df['amount'] / df['volume']
        df['PJJ'] = (df['high'] + df['low'] + df['close'] * 2) / 4
        df['PJJ'] = df['PJJ'].ewm(alpha=0.9, adjust=False).mean()
        df['JJ'] = df['PJJ'].ewm(span=3, adjust=False).mean().shift(1)
        df['QJJ'] = df['volume'] / ((df['high'] - df['low']) * 2 - abs(df['close'] - df['open']))

        # 向量化XVL
        is_pos = df['close'] > df['open']
        is_neg = df['close'] < df['open']
        is_flat = df['close'] == df['open']
        flow_in_pos = df['QJJ'] * (df['high'] - df['low'])
        flow_out_pos = -df['QJJ'] * (df['high'] - df['close'] + df['open'] - df['low'])
        flow_in_neg = df['QJJ'] * (df['high'] - df['open'] + df['close'] - df['low'])
        flow_out_neg = -df['QJJ'] * (df['high'] - df['low'])
        flow_in_flat = df['volume'] / 2
        flow_out_flat = -df['volume'] / 2

        flow_in = flow_in_pos.where(is_pos, flow_in_neg.where(is_neg, flow_in_flat))
        flow_out = flow_out_pos.where(is_pos, flow_out_neg.where(is_neg, flow_out_flat))
        df['XVL'] = flow_in + flow_out

        df['ZLL'] = df['volume'] / df['total_shares']
        df['LIJIN1'] = df['ZLL'].clip(upper=10)
        df['LIJIN'] = (df['XVL'] / 20) / 1.15
        df['主生量'] = df['LIJIN'] * 0.55 + df['LIJIN'].shift(1) * 0.33 + df['LIJIN'].shift(2) * 0.22
        df['GJJ'] = df['主生量'].ewm(span=8, adjust=False).mean()
        df['LLJX'] = df['主生量'].ewm(span=3, adjust=False).mean()
        df['资金量'] = df['LLJX']
        df['量基线'] = df['GJJ']
        df['ZJLL'] = df['LLJX'].shift(1)
        df['QZJJ'] = ((df['LLJX'] - df['ZJLL']) / df['ZJLL']) * 100
        df['量增幅'] = df['QZJJ'].where(
            (df['LLJX'] > 0) & (df['ZJLL'] < 0),
            df['QZJJ'].where(
                (df['LLJX'] < 0) & (df['ZJLL'] < 0) & (df['LLJX'] < df['ZJLL']),
                -df['QZJJ']
            )
        ).abs()

        df['力度'] = df['LIJIN'] / 1000
        df['周量'] = df['LLJX'].rolling(window=5).sum()
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

    def _calculate_market_heat(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场热度"""
        mean_volumes = df.groupby(['date', 'industry'])['volume'].mean().reset_index()
        mean_volumes.columns = ['date', 'industry', 'mean_volume']
        df = df.reset_index(drop=True)
        mean_volumes = mean_volumes.reset_index(drop=True)
        df = df.merge(mean_volumes, on=['date', 'industry'], how='left')
        df['market_heat'] = df['volume'] / df['mean_volume']
        return df

    def _score_stocks(self, signals: pd.DataFrame) -> pd.DataFrame:
        """综合评分"""
        weights = {
            "technical": 0.4483,
            "capital_flow": 0.2947,
            "market_heat": 0.257,
            "fundamental": 0.0
        }
        tech_weights = {
            "ma_condition": 0.1622, "angle_condition": 0.0854,
            "macd_condition": 0.1366, "volume_score": 0.1704,
            "rsi_oversold": 0.0597, "kdj_oversold": 0.0597,
            "cci_oversold": 0.0597, "bollinger_condition": 0.1191,
            "macd_jc": 0.0873, "price_growth": 0.06
        }
        thresholds = {
            'volume_gain_threshold': 1.3, 'volume_loss_threshold': -0.65,
            'growth_min': 0.5, 'growth_max': 6.0, 'angle_min': 30,
        }
        cap_weights = {
            'positive_flow': -0.0072, 'flow_increasing': 0.0108,
            'trend_strength': 0.0147, 'weekly_flow': 0.0072,
            'weekly_increasing': 0.0036, 'volume_gain_ratio': 0.1524,
            'volume_baseline': 0.0437, 'primary_volume': 0.0159,
            'weekly_growth': 0.2438, 'volume_gain_multiplier': 0.3434,
            'volume_loss_multiplier': 0.1717
        }

        for _, row in signals.iterrows():
            tech_score = 0
            growth = row.get('growth', 0)
            if thresholds['growth_min'] <= growth <= thresholds['growth_max']:
                tech_score += tech_weights['price_growth']
            elif growth > 0:
                tech_score += tech_weights['price_growth'] * 0.5

            tech_score += int(row.get('ma_condition', False)) * tech_weights['ma_condition']
            tech_score += int(row.get('angle_ma_10', 0) > thresholds['angle_min']) * tech_weights['angle_condition']
            tech_score += int(row.get('macd_condition', False)) * tech_weights['macd_condition']
            vol_ratio = min(row.get('volume', 0) / max(row.get('volume_ma5', 1), 1), 3)
            tech_score += vol_ratio * tech_weights['volume_score']
            tech_score += int(row.get('rsi_14', 100) < 70) * tech_weights['rsi_oversold']
            tech_score += int(row.get('kdj_k', 100) < 80) * tech_weights['kdj_oversold']
            tech_score += int(row.get('cci_20', 100) < 100) * tech_weights['cci_oversold']
            tech_score += int(row.get('close', 0) < row.get('bb_upper', float('inf'))) * tech_weights['bollinger_condition']
            tech_score += int(row.get('macd_jc', False)) * tech_weights['macd_jc']

            flow_score = 0
            if row.get('money_flow_positive'):
                flow_score += cap_weights['positive_flow'] * 1.4
            if row.get('money_flow_increasing'):
                flow_score += cap_weights['flow_increasing']
            if row.get('money_flow_trend'):
                flow_score += cap_weights['trend_strength'] * min(
                    row.get('主生量', 0) / max(row.get('量基线', 1), 1e-6), 2.0
                )
            if row.get('money_flow_weekly'):
                flow_score += cap_weights['weekly_flow']
            if row.get('money_flow_weekly_increasing'):
                flow_score += cap_weights['weekly_increasing'] * 1.3

            vol_gain = row.get('量增幅', 0)
            if vol_gain > thresholds['volume_gain_threshold']:
                flow_score *= cap_weights['volume_gain_multiplier']
            elif vol_gain < thresholds['volume_loss_threshold']:
                flow_score *= cap_weights['volume_loss_multiplier']

            flow_score += min(vol_gain, 5) * cap_weights['volume_gain_ratio']
            flow_score += int(row.get('量基线', 0) > 0) * cap_weights['volume_baseline']
            flow_score += min(row.get('主生量', 0) / max(row.get('量基线', 1), 1e-6), 2.0) * cap_weights['primary_volume']
            flow_score += min(row.get('周增幅', 0), 5) * cap_weights['weekly_growth']

            market_heat = row.get('market_heat', 0)

            total = (
                tech_score * weights['technical'] +
                flow_score * weights['capital_flow'] +
                market_heat * weights['market_heat']
            )

            signals.loc[row.name, 'technical'] = tech_score
            signals.loc[row.name, 'capital_flow'] = flow_score
            signals.loc[row.name, 'market_heat'] = market_heat
            signals.loc[row.name, 'total_score'] = total

        return signals

    def get_signals(self, start_date: str, end_date: str):
        """获取买卖信号"""
        logger.info(f"[ScoreStrategy] 生成信号 {start_date} ~ {end_date}")

        signals = self.generate_features(start_date, end_date)

        # 买入条件
        buy_condition = (
            signals['growth_condition'] &
            signals['ma_condition'] &
            signals['angle_condition'] &
            signals['volume_condition'] &
            signals['macd_condition'] &
            (signals['jc_condition'] | signals['macd_jc']) &
            (signals['ma_20'] < signals['ma_55']) &
            (signals['ma_55'] > signals['ma_240'])
        )

        buy_signals = signals[buy_condition].copy()
        buy_signals['signal_type'] = 'buy'

        # 评分选股
        buy_signals = self._score_stocks(buy_signals)
        if buy_signals.empty:
            selected_buy = None
        else:
            final_picks = []
            for date in buy_signals['date'].unique():
                daily = buy_signals[buy_signals['date'] == date]
                top = daily.nlargest(self.config['top_n'], 'total_score')
                final_picks.append(top)
            selected_buy = pd.concat(final_picks) if final_picks else None
            if selected_buy is not None:
                selected_buy = selected_buy.set_index('date')

        # 卖出条件
        sell_condition = (
            (signals['ma_20'] > signals['ma_5']) &
            (
                ((signals['macd'] > signals['macd_signal']) & (signals['macd'].shift(1) < signals['macd_signal'].shift(1))) |
                ((signals['close'] < signals['ma_10']) & (signals['volume'] < signals['volume'].shift(1) * 0.8))
            )
        ) | (
            (signals['close'] < signals['ma_20']) &
            (signals['money_flow_trend'] == False)
        )

        sell_signals = signals[sell_condition].copy()
        sell_signals['signal_type'] = 'sell'
        if not sell_signals.empty:
            sell_signals = sell_signals.set_index('date')

        return [selected_buy, sell_signals]

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        评分策略的买入信号筛选（保持原有逻辑）
        此方法由backtester调用，用于进一步筛选和处理买入信号
        """
        # 评分策略不在这儿处理额外逻辑，直接返回候选
        return []
