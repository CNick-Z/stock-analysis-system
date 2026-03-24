# strategies/resonance_strategy.py
# 8指标共振策略插件

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class ResonanceStrategy(BaseStrategy):
    """
    8指标共振策略插件

    8个指标全部满足才买入：
    1. MACD > MACD_SIGNAL（金叉）
    2. KDJ K > D（K在D上方）
    3. RSI < 70（未超买）
    4. LWR (williams_r) > -20（未超卖）
    5. BBI > close（BBI多空线上）
    6. MTM > MTM_MA（动量多头）
    7. MA5 > MA20（均线多头）
    8. 成交量 > vol_ma5（放量）

    仓位分级：
    - 一级信号（满足4-5个指标）：20%仓位
    - 二级信号（满足6-7个指标）：40%仓位
    - 三级信号（满足8个指标）：60%仓位
    """

    def __init__(self, db_path: str = None, config: dict = None):
        default_config = {
            'ma_windows': (5, 10, 20, 55, 240),
            # Phase 2: BBI/MTM参数
            'bbi_ma_periods': (3, 6, 12, 24),  # BBI计算的均线周期
            'mtm_period': 12,                    # MTM计算周期（N日）
            'mtm_ma_period': 6,                  # MTM均线周期
            # 仓位配置
            'signal_position_config': {
                1: 0.20,   # 一级信号：20%仓位
                2: 0.40,   # 二级信号：40%仓位
                3: 0.60,   # 三级信号：60%仓位
            },
            # 选股参数
            'top_n': 5,
        }
        self.config = {**default_config, **(config or {})}

        super().__init__(db_path, self.config)

        # 初始化数据库管理器
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
        """生成特征数据（包含BBI和MTM）"""
        # 扩展日期范围（需要前置数据计算BBI/MTM）
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=60)
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

        # Phase 2: 计算BBI和MTM
        logger.info("[ResonanceStrategy] 计算BBI/MTM指标...")
        df = self._calculate_bbi_mtm(df)

        # 生成核心信号
        df = self._generate_core_signals(df)

        # 计算市场热度
        df = self._calculate_market_heat(df)

        return df

    def _calculate_ma_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线角度"""
        ma_map = {'ma_10': 'ma_5', 'ma_20': 'ma_10', 'ma_55': 'ma_20', 'ma_240': 'ma_55'}
        for long_ma, short_ma in ma_map.items():
            df[long_ma] = df[long_ma].fillna(df[short_ma])
            ref_ma = df[long_ma].shift(2)
            ratio_change = (df[long_ma] / ref_ma - 1) * 100
            df[f'angle_{long_ma}'] = np.degrees(np.arctan(ratio_change))
        return df

    def _calculate_bbi_mtm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: 计算BBI多空线和MTM动量指标

        BBI = (MA3 + MA6 + MA12 + MA24) / 4
        MTM = close - close(N日前)，MTM_MA = MTM的N日均线
        """
        cfg = self.config
        ma3_p, ma6_p, ma12_p, ma24_p = cfg['bbi_ma_periods']
        mtm_period = cfg['mtm_period']
        mtm_ma_period = cfg['mtm_ma_period']

        def calc_per_stock(group):
            g = group.sort_values('date').copy()
            # BBI = (MA3 + MA6 + MA12 + MA24) / 4
            g['ma3'] = g['close'].rolling(ma3_p).mean()
            g['ma6'] = g['close'].rolling(ma6_p).mean()
            g['ma12'] = g['close'].rolling(ma12_p).mean()
            g['ma24'] = g['close'].rolling(ma24_p).mean()
            g['bbi'] = (g['ma3'] + g['ma6'] + g['ma12'] + g['ma24']) / 4

            # MTM = close - close(N日前)
            g['mtm'] = g['close'] - g['close'].shift(mtm_period)
            # MTM_MA = MTM的N日均线
            g['mtm_ma'] = g['mtm'].rolling(mtm_ma_period).mean()
            return g

        df = df.groupby('symbol', group_keys=False).apply(calc_per_stock)
        return df

    def _generate_core_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成核心交易信号"""
        buy_cfg = {'growth_threshold': 1.00, 'max_upper_shadow': 1.06, 'ma_diff_threshold': 0.02}

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
        df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
        df['macd_jc'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) < df['macd_signal'].shift(1))

        # JC条件
        df['jc_condition'] = (
            (df['ma_5'] > df['ma_5'].shift(1)) &
            (df['ma_20'] > df['ma_20'].shift(1)) &
            (abs(df['ma_5'] - df['ma_20']) / df['ma_20'] < buy_cfg['ma_diff_threshold'])
        )

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

        is_pos = df['close'] > df['open']
        is_neg = df['close'] < df['open']
        flow_in = (df['QJJ'] * (df['high'] - df['low'])).where(is_pos,
            (df['QJJ'] * (df['high'] - df['open'] + df['close'] - df['low'])).where(is_neg, df['volume'] / 2))
        flow_out = (-df['QJJ'] * (df['high'] - df['close'] + df['open'] - df['low'])).where(is_pos,
            (-df['QJJ'] * (df['high'] - df['low'])).where(is_neg, -df['volume'] / 2))
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

    def _is_multi_indicator_bullish(self, row: pd.Series) -> Tuple[bool, int, int]:
        """
        Phase 2: 8指标共振多头检查

        8个指标:
        1. MACD > MACD_SIGNAL（金叉）
        2. KDJ K > D（K在D上方）
        3. RSI < 70（未超买）
        4. LWR > -20（未超卖，数值越大越超卖，>-20表示未超卖）
        5. BBI > close（BBI多空线上）
        6. MTM > MTM_MA（动量多头）
        7. MA5 > MA20（均线多头）
        8. 成交量 > vol_ma5（放量）

        返回: (是否满足4+指标, 满足指标数量, 信号等级)
        """
        count = 0

        # 1. MACD金叉
        if row['macd'] > row['macd_signal']:
            count += 1

        # 2. KDJ K > D
        if row['kdj_k'] > row['kdj_d']:
            count += 1

        # 3. RSI < 70（未超买）
        if row['rsi_14'] < 70:
            count += 1

        # 4. LWR > -20（未超卖）
        if row['williams_r'] > -20:
            count += 1

        # 5. BBI > close（BBI多空线上）
        if row['bbi'] > row['close']:
            count += 1

        # 6. MTM > MTM_MA（动量多头）
        if row['mtm'] > row['mtm_ma']:
            count += 1

        # 7. MA5 > MA20（均线多头）
        if row['ma_5'] > row['ma_20']:
            count += 1

        # 8. 成交量 > volume_ma5（放量）
        if row['volume_ma5'] and row['volume'] > row['volume_ma5']:
            count += 1

        # 信号强度分级
        if count >= 8:
            signal_level = 3  # 三级：60%仓位
        elif count >= 6:
            signal_level = 2  # 二级：40%仓位
        elif count >= 4:
            signal_level = 1  # 一级：20%仓位
        else:
            signal_level = 0  # 不买入

        return count >= 4, count, signal_level

    def get_signals(self, start_date: str, end_date: str):
        """
        获取买卖信号

        共振策略的买入信号由 backtester 的 _process_resonance_buy_signals 处理
        这里返回所有满足基础条件的候选信号
        """
        logger.info(f"[ResonanceStrategy] 生成信号 {start_date} ~ {end_date}")

        signals = self.generate_features(start_date, end_date)

        # 基础买入条件（与评分策略相同）
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

        # 共振策略不做评分排序，而是把所有候选返回给 backtester 处理
        if not buy_signals.empty:
            buy_signals = buy_signals.set_index('date')

        # 卖出条件（与评分策略相同）
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

        return [buy_signals, sell_signals]

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        共振策略的买入信号筛选

        参数:
            date: 当前日期
            candidates: 候选股票 DataFrame
            **kwargs: 包含持仓信息、资金等

        返回:
            满足共振条件的股票列表，每项包含信号等级和满足的指标数
        """
        results = []
        for _, row in candidates.iterrows():
            is_bullish, count, level = self._is_multi_indicator_bullish(row)
            if is_bullish:
                results.append({
                    'symbol': row['symbol'],
                    'indicator_count': count,
                    'signal_level': level,
                    'row': row
                })
        return results

    def get_position_ratio(self, signal_level: int) -> float:
        """根据信号等级获取仓位比例"""
        return self.config['signal_position_config'].get(signal_level, 0.20)
