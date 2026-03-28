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
            # 股票池大小限制（解决内存问题）
            'stock_universe_size': 500,  # 只保留成交额最高的前N只股票
        }
        self.config = {**default_config, **(config or {})}

        super().__init__(db_path, self.config)

        # 初始化数据管理器
        from utils.parquet_db import ParquetDatabaseIntegrator
        self.db_manager = ParquetDatabaseIntegrator(db_path)
        self.stock_info_cache = None
        
        # Alpha158因子缓存（滑动窗口优化）
        # 缓存整个回测周期的因子数据，避免逐年重复计算
        self._factor_cache = None  # DataFrame: 全量因子数据
        self._factor_cache_period = None  # (start_date, end_date): 缓存的数据范围
        self._factor_cache_loaded = False  # 是否已加载全量数据

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

    def precompute_all_factors(self, start_date: str, end_date: str) -> None:
        """
        预计算整个回测周期的Alpha158因子（滑动窗口优化核心）
        
        只在回测开始前调用一次，计算并缓存所有因子。
        后续 get_signals() 调用直接从缓存读取，不再重复计算。
        
        Args:
            start_date: 回测开始日期 (e.g., '2015-01-01')
            end_date: 回测结束日期 (e.g., '2024-12-31')
        """
        logger.info(f"🔄 预计算Alpha158因子: {start_date} ~ {end_date}")
        
        # 扩展日期范围（需要lookback数据计算因子）
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=120)
        extended_end = end_date
        extended_start_str = extended_start.strftime("%Y-%m-%d")
        
        # 加载全量数据
        tech_df = self._load_technical_data(extended_start_str, extended_end)
        price_df = self._load_price_data(extended_start_str, extended_end)
        info_df = self._get_stock_basic_info()
        
        # ========== 股票池限制：只保留成交额最高的前N只 ==========
        stock_universe_size = self.config.get('stock_universe_size', 500)
        if stock_universe_size and 'amount' in price_df.columns:
            total_before = price_df['symbol'].nunique()
            top_symbols = (
                price_df.groupby('symbol')['amount']
                .sum()
                .nlargest(stock_universe_size)
                .index.tolist()
            )
            price_df = price_df[price_df['symbol'].isin(top_symbols)]
            tech_df = tech_df[tech_df['symbol'].isin(top_symbols)]
            info_df = info_df[info_df['symbol'].isin(top_symbols)]
            logger.info(f"📦 股票池限制: {total_before} → {len(top_symbols)} 只（按成交额）")
        # ==========================================================
        
        # 合并数据
        df = pd.merge(tech_df, price_df, on=['date', 'symbol'], how='inner')
        df = pd.merge(df, info_df, on=['symbol'], how='inner')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)  # 重置index，避免alpha计算index不匹配
        
        logger.info(f"📊 预计算：加载了 {len(df):,} 行数据 ({df['date'].min().date()} ~ {df['date'].max().date()})")
        
        # 计算Alpha158因子（全量一次性计算）
        if self.config.get('alpha158_enabled', True):
            df = self._calculate_alpha_factors(df)
        
        # 缓存结果
        self._factor_cache = df
        self._factor_cache_period = (start_date, end_date)
        self._factor_cache_loaded = True
        
        logger.info(f"✅ Alpha158因子预计算完成，缓存 {len(df):,} 行，共 {df['date'].nunique()} 个交易日")
        logger.info(f"   缓存有效期: {start_date} ~ {end_date}")

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成特征数据，包含qlib Alpha158因子
        
        Alpha158因子是从qlib的158个预定义因子中精选的实用因子，
        涵盖量价关系、均线排列、资金流向等多个维度。
        
        优化：如果已预计算因子缓存，直接从缓存slice，不再重复计算。
        """
        # 如果已有全量缓存，直接slice返回（滑动窗口优化）
        if self._factor_cache_loaded and self._factor_cache is not None:
            cached_start, cached_end = self._factor_cache_period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 扩展范围需要lookback数据（用于计算当日因子），但返回时过滤
            lookback_start = start_dt - pd.Timedelta(days=60)
            
            # 从缓存中slice（扩展lookback范围）
            df = self._factor_cache[
                (self._factor_cache['date'] >= lookback_start) &
                (self._factor_cache['date'] <= end_dt)
            ].copy()
            
            logger.info(f"📦 因子缓存命中: {start_date} ~ {end_date} (获取 {len(df):,} 行)")
            return df
        
        # 没有缓存fallback：扩展日期范围
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=60)
        extended_start_str = extended_start.strftime("%Y-%m-%d")

        # 加载基础数据
        tech_df = self._load_technical_data(extended_start_str, end_date)
        price_df = self._load_price_data(extended_start_str, end_date)
        info_df = self._get_stock_basic_info()

        # ========== 股票池限制（同precompute_all_factors） ==========
        stock_universe_size = self.config.get('stock_universe_size', 500)
        if stock_universe_size and 'amount' in price_df.columns:
            top_symbols = (
                price_df.groupby('symbol')['amount']
                .sum()
                .nlargest(stock_universe_size)
                .index.tolist()
            )
            price_df = price_df[price_df['symbol'].isin(top_symbols)]
            tech_df = tech_df[tech_df['symbol'].isin(top_symbols)]
            info_df = info_df[info_df['symbol'].isin(top_symbols)]
        # ==========================================================

        # 合并数据
        df = pd.merge(tech_df, price_df, on=['date', 'symbol'], how='inner')
        df = pd.merge(df, info_df, on=['symbol'], how='inner')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)  # 重置index，避免alpha计算index不匹配

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
        """量价相关性因子 - 向量化优化（修复版）"""
        window = self.config.get('corr_window', 10)
        
        # 先计算截面排名
        df['rank_open'] = df.groupby('date')['open'].rank(pct=True)
        df['rank_volume'] = df.groupby('date')['volume'].rank(pct=True)
        df['rank_high'] = df.groupby('date')['high'].rank(pct=True)
        
        # 用 groupby+transform+rolling 计算滚动相关性（避免复杂index问题）
        def rolling_corr_col(series1, series2, window):
            """对两个同index的Series计算滚动相关系数"""
            return series1.rolling(window, min_periods=1).corr(series2)
        
        df['alpha_001'] = df.groupby('symbol')['rank_open'].transform(
            lambda x: rolling_corr_col(x, df.loc[x.index, 'rank_volume'], window))
        df['alpha_002'] = df.groupby('symbol')['rank_high'].transform(
            lambda x: rolling_corr_col(x, df.loc[x.index, 'rank_volume'], window))
        
        # 最后截面排名
        df['alpha_001'] = df.groupby('date')['alpha_001'].rank(pct=True)
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
            df[f'alpha_volatility_{w}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(w, min_periods=1).std()
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
        
        # RSI动量 (类似Alpha009) - 向量化优化
        if 'rsi_14' in df.columns:
            # y.diff().sum() 在窗口内 = y[-1] - y[0]，可以直接用rolling计算
            rsi_diff = df.groupby('symbol')['rsi_14'].diff()
            df['alpha_rsi_momentum'] = df.groupby('symbol')['rsi_14'].transform(
                lambda x: x.diff().rolling(10, min_periods=1).sum()
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
        from utils.db_operations import TechnicalIndicatorsBase
        
        df = self.db_manager.load_data(
            table_class=TechnicalIndicatorsBase,
            filter_conditions={'date': {'$between': [start_date, end_date]}}
        )
        
        # 列名映射：Parquet用 ma_5，策略用 sma_5
        rename_map = {
            'ma_5': 'sma_5',
            'ma_10': 'sma_10',
            'ma_20': 'sma_20',
            'ma_55': 'sma_55',
            'ma_240': 'sma_240',
            'vol_ma5': 'vol_ma5',
        }
        df = df.rename(columns=rename_map)
        return df

    def _load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载价格数据"""
        from utils.db_operations import DailyDataBase
        
        return self.db_manager.load_data(
            table_class=DailyDataBase,
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
        
        # 设置date为索引（与score策略保持一致）
        if not buy_signals.empty and 'date' in buy_signals.columns:
            buy_signals = buy_signals.set_index('date')
        if not sell_signals.empty and 'date' in sell_signals.columns:
            sell_signals = sell_signals.set_index('date')
        
        # 过滤：只返回start_date到end_date期间的信号（排除lookback期间）
        buy_signals = buy_signals[(buy_signals.index >= start_date) & (buy_signals.index <= end_date)]
        sell_signals = sell_signals[(sell_signals.index >= start_date) & (sell_signals.index <= end_date)]
        
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
            lambda x: x.nlargest(top_n, 'total_score'), include_groups=False
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
