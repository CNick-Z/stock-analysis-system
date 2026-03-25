# strategies/qlib_strategy.py
# Qlib增强策略 - 不影响现有框架
# 引入Alpha158因子和机器学习预测增强

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class QlibStrategy(BaseStrategy):
    """
    Qlib增强版策略
    
    继承BaseStrategy接口，同时引入qlib核心思想：
    1. Alpha158因子计算（qlib核心因子库）
    2. 特征重要性分析（类似qlib的模型解释）
    3. 与现有策略信号对齐输出
    
    不修改任何现有策略，完全独立运行。
    """

    def __init__(self, db_path: str = None, config: dict = None):
        # 默认配置
        default_config = {
            'ma_windows': (5, 10, 20, 55, 240),
            'macd_params': (12, 26, 9),
            # Qlib增强配置
            'alpha158_enabled': True,
            'ml_model_enabled': False,  # 未来可开启
            'top_n': 5,
            # qlib风格参数
            'rank_window': 5,  # CS_Rank窗口
            'corr_window': 10,  # 相关性计算窗口
            'tsmean_window': 5,  # 移动平均窗口
        }
        self.config = {**default_config, **(config or {})}

        super().__init__(db_path, self.config)

        # 初始化数据管理器
        from utils.parquet_db import ParquetDatabaseIntegrator
        self.db_manager = ParquetDatabaseIntegrator(db_path)
        self.stock_info_cache = None
        
        # Alpha158因子缓存
        self.alpha_cache = {}

    def _get_stock_basic_info(self):
        """获取股票基础信息"""
        if self.stock_info_cache is None:
            from utils.db_operations import StockBasicInfo
            self.stock_info_cache = self.db_manager.load_data(
                table_class=StockBasicInfo,
                filter_conditions={'name': {'$not_like': 'ST'}},
                columns=['symbol', 'name', 'total_shares', 'industry']
            )
        return self.stock_info_cache

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成特征数据，包含qlib Alpha158因子
        
        Alpha158因子是从qlib的158个预定义因子中精选的实用因子，
        涵盖量价关系、均线排列、资金流向等多个维度。
        """
        # 扩展日期范围
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=60)
        extended_start_str = extended_start.strftime("%Y-%m-%d")

        # 加载基础数据
        tech_df = self._load_technical_data(extended_start_str, end_date)
        price_df = self._load_price_data(extended_start_str, end_date)
        info_df = self._get_stock_basic_info()

        # 合并数据
        df = pd.merge(tech_df, price_df, on=['date', 'symbol'], how='inner')
        df = pd.merge(df, info_df, on=['symbol'], how='inner')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 计算qlib风格Alpha因子
        if self.config.get('alpha158_enabled', True):
            df = self._calculate_alpha_factors(df)

        return df

    def _calculate_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Alpha158核心因子
        
        这些因子源自qlib的Alpha158因子库，经过挑选适合A股的实用因子。
        分为以下类别：
        1. 量价相关 (CS_Rank, Corr)
        2. 均线相关 (MA, MA_Diff, MA_Ratio)
        3. 波动率 (STD, VAR)
        4. 资金流 (MoneyFlow)
        5. 动量 (Return, Momentum)
        """
        logger.info("计算Alpha158核心因子...")
        
        # 1. 量价相关性因子 (CS_Rank - Cross Section Rank)
        df = self._add_corr_factors(df)
        
        # 2. 均线相关因子
        df = self._add_ma_factors(df)
        
        # 3. 波动率因子
        df = self._add_volatility_factors(df)
        
        # 4. 动量因子
        df = self._add_momentum_factors(df)
        
        # 5. 资金流因子
        df = self._add_money_flow_factors(df)
        
        # 6. 成交量因子
        df = self._add_volume_factors(df)
        
        logger.info(f"Alpha因子计算完成，共 {len([c for c in df.columns if c.startswith('alpha_')])} 个因子")
        return df

    def _add_corr_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """量价相关性因子"""
        window = self.config.get('corr_window', 10)
        
        # Alpha001: Rank(Corr(Rank(Open), Rank(Volume), 10))
        df['rank_open'] = df.groupby('date')['open'].rank(pct=True)
        df['rank_volume'] = df.groupby('date')['volume'].rank(pct=True)
        df['alpha_001'] = df.groupby('symbol')['rank_open'].transform(
            lambda x: x.rolling(window, min_periods=1).corr(df.loc[x.index, 'rank_volume'])
        )
        df['alpha_001'] = df.groupby('date')['alpha_001'].rank(pct=True)
        
        # Alpha002: Rank(Corr(Rank(High), Rank(Volume), 10))
        df['rank_high'] = df.groupby('date')['high'].rank(pct=True)
        df['alpha_002'] = df.groupby('symbol')['rank_high'].transform(
            lambda x: x.rolling(window, min_periods=1).corr(df.loc[x.index, 'rank_volume'])
        )
        df['alpha_002'] = df.groupby('date')['alpha_002'].rank(pct=True)
        
        # 清理临时列
        df.drop(['rank_open', 'rank_volume', 'rank_high'], axis=1, inplace=True)
        
        return df

    def _add_ma_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """均线相关因子"""
        windows = [5, 10, 20, 60]
        
        # MA_Diff_10_20: MA10与MA20的差值标准化
        if 'sma_10' in df.columns and 'sma_20' in df.columns:
            df['ma_diff_10_20'] = (df['sma_10'] - df['sma_20']) / df['sma_20']
            df['alpha_ma_diff'] = df.groupby('date')['ma_diff_10_20'].rank(pct=True)
        
        # MA排列因子: 多均线多头排列程度
        if all(f'sma_{w}' in df.columns for w in [5, 10, 20, 55]):
            df['ma_bull_count'] = (
                (df['sma_5'] > df['sma_10']).astype(int) +
                (df['sma_10'] > df['sma_20']).astype(int) +
                (df['sma_20'] > df['sma_55']).astype(int)
            )
            df['alpha_ma排列'] = df['ma_bull_count'] / 3
        
        # 均线角度 (类似qlib的TMQA_Taobao)
        for w in [10, 20]:
            col = f'sma_{w}'
            if col in df.columns:
                ma_pct_change = df.groupby('symbol')[col].pct_change(periods=2)
                df[f'alpha_angle_{w}'] = np.arctan(ma_pct_change.fillna(0).clip(-0.5, 0.5)) * 180 / np.pi
        
        return df

    def _add_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率因子"""
        windows = [10, 20, 30]
        
        # 收益率标准差 (类似Alpha043)
        for w in windows:
            ret = df.groupby('symbol')['close'].pct_change()
            df[f'alpha_volatility_{w}'] = df.groupby('symbol')[ret.name].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        
        # 价格波动率与成交量波动率的比
        if 'alpha_volatility_20' in df.columns:
            vol_std = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20, min_periods=1).std()
            )
            df['alpha_vol_ratio'] = df['alpha_volatility_20'] / (vol_std / df['volume'] + 1e-10)
        
        return df

    def _add_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量因子"""
        periods = [5, 10, 20, 60]
        
        # 收益率动量
        for p in periods:
            ret = df.groupby('symbol')['close'].pct_change(periods=p)
            df[f'alpha_return_{p}d'] = ret
        
        # 累计收益率排名
        for p in [5, 20]:
            if f'alpha_return_{p}d' in df.columns:
                df[f'alpha_momentum_{p}'] = df.groupby('date')[f'alpha_return_{p}d'].rank(pct=True)
        
        # RSI动量 (类似Alpha009)
        if 'rsi_14' in df.columns:
            rsi_change = df.groupby('symbol')['rsi_14'].diff()
            df['alpha_rsi_momentum'] = df.groupby('symbol')['rsi_14'].transform(
                lambda x: x.rolling(10, min_periods=1).apply(lambda y: y.diff().sum(), raw=False)
            )
        
        return df

    def _add_money_flow_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """资金流因子"""
        # 如果有资金流向数据
        if '资金量' in df.columns:
            # 资金流排名
            df['alpha_money_flow'] = df.groupby('date')['资金量'].rank(pct=True)
            
            # 资金流趋势
            money_ma5 = df.groupby('symbol')['资金量'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            df['alpha_money_trend'] = (df['资金量'] - money_ma5) / (money_ma5 + 1e-10)
        
        # 成交量与均量比值 (简化版资金流)
        if 'vol_ma5' in df.columns and 'volume' in df.columns:
            df['alpha_volume_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
            df['alpha_volume_ratio_rank'] = df.groupby('date')['alpha_volume_ratio'].rank(pct=True)
        
        return df

    def _add_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量因子"""
        # 量比
        if 'vol_ma5' in df.columns and 'volume' in df.columns:
            df['alpha_vol_ratio_raw'] = df['volume'] / (df['vol_ma5'] + 1e-10)
        
        # 成交量增长率
        vol_change = df.groupby('symbol')['volume'].pct_change(periods=5)
        df['alpha_vol_change_5d'] = vol_change
        
        # 成交量与价格趋势背离
        if 'alpha_return_5d' in df.columns:
            df['alpha_vol_price_div'] = df['alpha_vol_change_5d'] - df['alpha_return_5d'] * 10
        
        return df

    def _load_technical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载技术指标数据"""
        from utils.db_operations import TechnicalIndicators
        
        return self.db_manager.load_data(
            table_class=TechnicalIndicators,
            filter_conditions={'date': {'$between': [start_date, end_date]}}
        )

    def _load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载价格数据"""
        from utils.db_operations import DailyData
        
        return self.db_manager.load_data(
            table_class=DailyData,
            filter_conditions={'date': {'$between': [start_date, end_date]}},
            columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
        )

    def get_signals(self, start_date: str, end_date: str):
        """
        获取买卖信号
        
        返回格式: [buy_signals, sell_signals]
        - buy_signals: DataFrame with columns [date, symbol, score, ...]
        - sell_signals: DataFrame with columns [date, symbol, score, ...]
        """
        # 生成特征
        df = self.generate_features(start_date, end_date)
        
        # 生成买入信号
        buy_signals = self._generate_buy_signals(df)
        
        # 生成卖出信号 (使用现有策略逻辑)
        sell_signals = self._generate_sell_signals(df)
        
        return [buy_signals, sell_signals]

    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成买入信号 - 使用Alpha因子增强"""
        # 过滤ST
        df = df[~df['name'].str.contains('ST', na=False)]
        
        # 基础条件: MA5 > MA10 且 MA10 < MA20
        if all(f'sma_{w}' in df.columns for w in [5, 10, 20]):
            ma_condition = (
                (df['sma_5'] > df['sma_10']) &
                (df['sma_10'] < df['sma_20'])
            )
        else:
            ma_condition = pd.Series(True, index=df.index)
        
        # MACD条件
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_condition = (
                (df['macd'] < 0) &
                (df['macd'] > df['macd_signal'])
            )
        else:
            macd_condition = pd.Series(True, index=df.index)
        
        # 应用基础条件
        candidates = df[ma_condition & macd_condition].copy()
        
        if candidates.empty:
            return pd.DataFrame()
        
        # ========== Alpha因子评分 ==========
        alpha_score_cols = []
        
        # 1. 均线排列得分
        if 'alpha_ma排列' in candidates.columns:
            candidates['score_ma'] = candidates['alpha_ma排列'] * 0.15
            alpha_score_cols.append('score_ma')
        
        # 2. 动量得分
        if 'alpha_momentum_20' in candidates.columns:
            candidates['score_momentum'] = candidates['alpha_momentum_20'] * 0.15
            alpha_score_cols.append('score_momentum')
        
        # 3. 资金流得分
        if 'alpha_volume_ratio_rank' in candidates.columns:
            candidates['score_volume'] = candidates['alpha_volume_ratio_rank'] * 0.1
            alpha_score_cols.append('score_volume')
        
        # 4. 相关性得分
        if 'alpha_001' in candidates.columns:
            candidates['score_corr'] = candidates['alpha_001'] * 0.1
            alpha_score_cols.append('score_corr')
        
        # 计算总分
        if alpha_score_cols:
            candidates['alpha_total'] = candidates[alpha_score_cols].sum(axis=1)
        else:
            candidates['alpha_total'] = 0
        
        # 综合评分 (与技术指标结合)
        candidates['total_score'] = candidates.get('score', 0) + candidates['alpha_total']
        
        # 按日期选出top N
        top_n = self.config.get('top_n', 5)
        result = candidates.groupby('date').apply(
            lambda x: x.nlargest(top_n, 'total_score')
        ).reset_index(drop=True)
        
        result['signal_type'] = 'buy'
        
        return result

    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成卖出信号
        
        使用v4策略逻辑:
        - 跌破MA20 + 资金流出 → 卖出
        """
        # 基础条件: 价格在MA20下方
        if 'sma_20' in df.columns and 'close' in df.columns:
            sell_condition = df['close'] < df['sma_20']
        else:
            sell_condition = pd.Series(False, index=df.index)
        
        # 资金流为负
        if '资金量' in df.columns:
            money_negative = df['资金量'] < 0
        elif 'alpha_volume_ratio' in df.columns:
            money_negative = df['alpha_volume_ratio'] < 1
        else:
            money_negative = pd.Series(False, index=df.index)
        
        # 卖出信号
        sell_signals = df[sell_condition & money_negative].copy()
        sell_signals['signal_type'] = 'sell'
        
        return sell_signals

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        获取买入信号 (接口方法)
        
        返回格式: [symbol, price, date, quantity, signal_info, extra_data]
        """
        signals = []
        
        if candidates.empty:
            return signals
        
        # 评分排序
        candidates = candidates.sort_values('total_score', ascending=False)
        
        top_n = self.config.get('top_n', 5)
        top_stocks = candidates.head(top_n)
        
        for _, row in top_stocks.iterrows():
            signals.append([
                row['symbol'],
                row['close'],
                date,
                100,  # quantity placeholder
                {'type': 'qlib_alpha', 'score': row.get('total_score', 0)},
                row.to_dict()
            ])
        
        return signals

    def get_feature_importance(self, date: str) -> pd.DataFrame:
        """
        获取因子重要性 (类似qlib的feature importance)
        
        用于分析哪些Alpha因子对预测最有效
        """
        alpha_cols = [c for c in self.alpha_cache.keys() if c.startswith('alpha_')]
        
        importance = []
        for col in alpha_cols:
            importance.append({
                'factor': col,
                'description': self._factor_descriptions.get(col, 'Unknown'),
                'enabled': self.config.get(col, True)
            })
        
        return pd.DataFrame(importance)

    # 因子描述 (用于可解释性)
    _factor_descriptions = {
        'alpha_001': '量价相关性 (Open-Volume, 10日)',
        'alpha_002': '量价相关性 (High-Volume, 10日)',
        'alpha_ma_diff': '均线差异度 (MA10-MA20)',
        'alpha_ma排列': '均线多头排列程度',
        'alpha_volatility_10': '波动率 (10日)',
        'alpha_volatility_20': '波动率 (20日)',
        'alpha_momentum_5': '动量 (5日收益率排名)',
        'alpha_momentum_20': '动量 (20日收益率排名)',
        'alpha_volume_ratio': '量比',
        'alpha_money_flow': '资金流排名',
        'alpha_money_trend': '资金流趋势',
    }

    def __repr__(self):
        return f"<QlibStrategy alpha158={self.config.get('alpha158_enabled')} ml={self.config.get('ml_model_enabled')}>"
