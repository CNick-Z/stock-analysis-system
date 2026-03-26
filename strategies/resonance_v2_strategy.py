# resonance_v2_strategy.py
# 共振策略 v2 — 优化版本
# 优化方向：MACD修复 + RSI过滤 + 市场广度 + 加权评分
# 继承自 ResonanceStrategy，保持原有框架兼容

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from .resonance_strategy import ResonanceStrategy

logger = logging.getLogger(__name__)


class ResonanceV2Strategy(ResonanceStrategy):
    """
    共振策略 v2（优化版）

    基于《市场共振框架》文章的核心理念：
    "不要只交易你盯着的那张图，要交易推动整个市场运动的力量。"

    =============================================
    v1 的5个核心问题（已修复）
    =============================================
    1. MACD条件反向 — 修复：移除 < 0 限制，与score策略一致
    2. 缺少RSI过滤器  — 修复：增加 RSI 50~60 过滤
    3. 8指标平等称重  — 修复：改为加权评分制
    4. 缺少市场方向   — 修复：BBI作为市场多空分界线
    5. 缺少板块共振   — 修复：市场广度过滤

    =============================================
    8个指标（加权评分）
    =============================================
    1. MACD金叉     — 权重 0.20（趋势核心）
    2. MA5 > MA20   — 权重 0.15（均线多头）
    3. 成交量放量   — 权重 0.15（动量确认）
    4. BBI > Price  — 权重 0.15（多空分界）
    5. KDJ K > D    — 权重 0.10（辅助趋势）
    6. MTM > MTM_MA — 权重 0.10（动量加速）
    7. RSI < 70     — 权重 0.08（未超买）
    8. LWR > -20    — 权重 0.07（未超卖）

    =============================================
    市场广度过滤（VWAP/BBI代替）
    =============================================
    - 市场广度 = 当日 close > BBI 的股票占比
    - breadth > 60%：正常交易（市场强势）
    - breadth 30%~60%：只接受中高分信号（加权分 ≥ 0.50）
    - breadth < 30%：放弃所有买入（市场太弱）
    """

    def __init__(self, db_path: str = None, config: dict = None):
        # 扩展默认配置，增加v2特有参数
        default_config = {
            # 继承父类配置
            'ma_windows': (5, 10, 20, 55, 240),
            'bbi_ma_periods': (3, 6, 12, 24),
            'mtm_period': 12,
            'mtm_ma_period': 6,
            'signal_position_config': {
                1: 0.20,   # 一级信号：20%仓位
                2: 0.40,   # 二级信号：40%仓位
                3: 0.60,   # 三级信号：60%仓位
            },
            'top_n': 5,

            # ===== v2 新增配置 =====

            # RSI 过滤（与score策略一致）
            'rsi_filter_enabled': True,
            'rsi_min': 50,
            'rsi_max': 60,

            # 市场广度配置
            'breadth_enabled': True,
            'breadth_strong': 0.60,   # >60% 强势市场
            'breadth_weak': 0.30,     # <30% 放弃信号
            'breadth_medium_min_score': 0.50,  # 中等市场最低分数要求

            # 加权评分配置（8个指标权重）
            'indicator_weights': {
                'macd_cross': 0.20,     # MACD金叉
                'ma_bullish': 0.15,     # MA5 > MA20
                'volume_surge': 0.15,   # 成交量放量
                'bbi_above': 0.15,      # BBI > Price
                'kdj_bullish': 0.10,    # KDJ K > D
                'mtm_bullish': 0.10,    # MTM > MTM_MA
                'rsi_not_overbought': 0.08,  # RSI < 70
                'lwr_not_oversold': 0.07,    # LWR > -20
            },

            # 买入前置条件：价格必须在 BBI 上方（市场共振的核心）
            'bbi_direction_filter': True,
        }
        self.config = {**default_config, **(config or {})}

        # 调用父类构造函数（会设置 db_manager 等）
        super().__init__(db_path, self.config)

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成特征数据 — v2版本

        相比v1新增：
        - RSI过滤器字段
        - BBI方向字段（close > bbi）
        - 市场广度字段（按日期计算）
        """
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

        # 计算BBI和MTM
        logger.info("[ResonanceV2Strategy] 计算BBI/MTM指标...")
        df = self._calculate_bbi_mtm(df)

        # ===== v2: 生成核心信号 =====
        df = self._generate_core_signals(df)

        # 计算市场热度
        df = self._calculate_market_heat(df)

        # ===== v2: 市场广度计算（向量化，一次完成）=====
        # 不依赖滚动窗口，纯向量化计算
        df = self._calculate_market_breadth(df)

        return df

    def _calculate_market_breadth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        v2: 计算市场广度（向量化，性能友好）

        breadth = 当日 close > BBI 的股票占比
        逻辑与VWAP类似：BBI是日均价线，价格在BBI上方代表多头控制市场

        计算方式：
        1. 每日每只股票的 bbi_above = close > bbi（bool）
        2. 按日期 groupby，求 mean() = 多头股占比
        3. merge 回原表

        全程向量化，不逐股循环，5000+股票性能无忧。
        """
        if not self.config.get('breadth_enabled', True):
            # 未启用时填充默认值
            df['market_breadth'] = 0.5
            df['bbi_above'] = df['close'] > df['bbi']
            return df

        # 第一步：计算每只股票是否处于BBI多头（close > bbi）
        df['bbi_above'] = df['close'] > df['bbi']

        # 第二步：按日期计算市场广度（向量化 groupby.mean()）
        # mean() 对 bool 列求平均 = True占比 = 多头股占比
        daily_breadth = (
            df.groupby('date')['bbi_above']
            .mean()
            .reset_index()
            .rename(columns={'bbi_above': 'market_breadth_raw'})
        )
        # 避免除零（全市场无BBI数据时）
        daily_breadth['market_breadth'] = daily_breadth['market_breadth_raw'].fillna(0.5)

        # 第三步：merge回原表（每个股票每天一条记录）
        df = df.merge(daily_breadth[['date', 'market_breadth']], on='date', how='left')

        # 填充缺失值（交易日无数据时用0.5）
        df['market_breadth'] = df['market_breadth'].fillna(0.5)

        # 释放临时列
        df.drop(columns=['market_breadth_raw'], errors='ignore', inplace=True)

        logger.debug(f"[ResonanceV2] 市场广度范围: {df['market_breadth'].min():.2%} ~ {df['market_breadth'].max():.2%}")
        return df

    def _generate_core_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成核心交易信号 — v2版本

        相比v1的改进：
        1. MACD条件：移除 < 0 限制（v1的反向逻辑）
        2. RSI过滤：增加 RSI 50~60 条件
        3. BBI方向：增加 close > bbi 条件（市场多空分界）
        """
        buy_cfg = self.config.get('buy_conditions', {
            'growth_threshold': 1.00,
            'max_upper_shadow': 1.06,
            'ma_diff_threshold': 0.02
        })

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

        # ===== v2修复：MACD条件 =====
        # v1错误：(df['macd'] < 0) & (df['macd'] > df['macd_signal'])
        # v2正确：去掉 < 0 限制，只保留金叉逻辑，与score策略一致
        df['macd_condition'] = (df['macd'] > df['macd_signal'])

        # MACD金叉检测
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

        # ===== v2新增：RSI过滤条件 =====
        if self.config.get('rsi_filter_enabled', True):
            rsi_min = self.config.get('rsi_min', 50)
            rsi_max = self.config.get('rsi_max', 60)
            df['rsi_filter_condition'] = (
                (df['rsi_14'] >= rsi_min) & (df['rsi_14'] <= rsi_max)
            )
        else:
            df['rsi_filter_condition'] = True  # 未启用时全部通过

        # ===== v2新增：BBI方向条件 =====
        # 价格在BBI上方 = 多头控制市场（市场共振核心前置条件）
        if self.config.get('bbi_direction_filter', True):
            df['bbi_direction_condition'] = df['close'] > df['bbi']
        else:
            df['bbi_direction_condition'] = True

        # 资金流向
        df = self._calculate_money_flow_indicators(df)
        df['money_flow_positive'] = df['资金量'] > 0
        df['money_flow_increasing'] = df['量增幅'] > 0
        df['money_flow_trend'] = df['主生量'] > df['量基线']
        df['money_flow_weekly'] = df['周量'] > 0
        df['money_flow_weekly_increasing'] = df['周增幅'] > 0

        return df

    def _score_multi_indicator(self, row: pd.Series) -> Tuple[float, int, int]:
        """
        v2: 8指标加权评分（替代v1的简单计数）

        加权评分公式：
        score = Σ(指标i通过 × 权重i)

        返回: (加权分数, 通过指标数, 信号等级)
        """
        w = self.config['indicator_weights']
        score = 0.0
        count = 0

        # 1. MACD金叉（权重 0.20）
        if row['macd'] > row['macd_signal']:
            score += w['macd_cross']
            count += 1

        # 2. MA5 > MA20 均线多头（权重 0.15）
        if row['ma_5'] > row['ma_20']:
            score += w['ma_bullish']
            count += 1

        # 3. 成交量放量（权重 0.15）
        if row['volume_ma5'] and row['volume'] > row['volume_ma5']:
            score += w['volume_surge']
            count += 1

        # 4. BBI > Price 多空线上（权重 0.15）
        if row['bbi'] > row['close']:
            score += w['bbi_above']
            count += 1

        # 5. KDJ K > D（权重 0.10）
        if row['kdj_k'] > row['kdj_d']:
            score += w['kdj_bullish']
            count += 1

        # 6. MTM > MTM_MA 动量多头（权重 0.10）
        if row['mtm'] > row['mtm_ma']:
            score += w['mtm_bullish']
            count += 1

        # 7. RSI < 70 未超买（权重 0.08）
        if row['rsi_14'] < 70:
            score += w['rsi_not_overbought']
            count += 1

        # 8. LWR > -20 未超卖（权重 0.07）
        if row['williams_r'] > -20:
            score += w['lwr_not_oversold']
            count += 1

        # 信号等级（基于加权分数，不是指标数量）
        # 高分档：≥0.70，三级60%仓位
        # 中分档：≥0.50，二级40%仓位
        # 低分档：≥0.35，一级20%仓位
        if score >= 0.70:
            level = 3
        elif score >= 0.50:
            level = 2
        elif score >= 0.35:
            level = 1
        else:
            level = 0

        return score, count, level

    def _is_multi_indicator_bullish(self, row: pd.Series) -> Tuple[bool, int, int]:
        """
        v2: 多指标多头检查（返回加权分数 + 信号等级）

        返回: (是否通过, 加权分数, 信号等级)
        """
        score, count, level = self._score_multi_indicator(row)
        # 通过门槛：加权分数 ≥ 0.35（即至少4个指标通过，每个权重约0.09）
        return score >= 0.35, score, level

    def get_signals(self, start_date: str, end_date: str):
        """
        获取买卖信号 — v2版本

        相比v1的买入条件增加：
        1. RSI 50~60 过滤
        2. BBI方向过滤（close > bbi）
        """
        logger.info(f"[ResonanceV2Strategy] 生成信号 {start_date} ~ {end_date}")

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

        # ===== v2新增：RSI过滤 =====
        if self.config.get('rsi_filter_enabled', True):
            buy_condition = buy_condition & signals['rsi_filter_condition']

        # ===== v2新增：BBI方向过滤 =====
        if self.config.get('bbi_direction_filter', True):
            buy_condition = buy_condition & signals['bbi_direction_condition']

        buy_signals = signals[buy_condition].copy()
        buy_signals['signal_type'] = 'buy'

        # 共振v2不做评分排序，而是把所有候选返回给 backtester 处理
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
        v2: 买入信号筛选（带市场广度过滤）

        参数:
            date: 当前日期
            candidates: 候选股票 DataFrame
            **kwargs: 包含持仓信息、资金、市场广度等

        市场广度过滤逻辑：
        - breadth > 60%：正常交易
        - breadth 30%~60%：只接受加权分数 ≥ 0.50 的信号
        - breadth < 30%：放弃所有买入信号
        """
        results = []
        breadth = kwargs.get('market_breadth', 0.5)  # 默认0.5，不过滤

        cfg = self.config
        breadth_strong = cfg.get('breadth_strong', 0.60)
        breadth_weak = cfg.get('breadth_weak', 0.30)
        min_score_medium = cfg.get('breadth_medium_min_score', 0.50)

        for _, row in candidates.iterrows():
            is_bullish, score, level = self._is_multi_indicator_bullish(row)

            if not is_bullish:
                continue

            # ===== 市场广度过滤 =====
            if breadth < breadth_weak:
                # 市场太弱，放弃所有信号
                logger.debug(f"[ResonanceV2] 日期={date} breadth={breadth:.2%} < {breadth_weak:.2%}，放弃买入 {row['symbol']}")
                continue

            if breadth < breadth_strong:
                # 中等市场，只接受高分信号
                if score < min_score_medium:
                    logger.debug(f"[ResonanceV2] 日期={date} breadth={breadth:.2%} 中等市场，分数={score:.2f} < {min_score_medium:.2f}，放弃 {row['symbol']}")
                    continue

            results.append({
                'symbol': row['symbol'],
                'weighted_score': score,
                'indicator_count': self._count_indicators(row),
                'signal_level': level,
                'market_breadth': breadth,
                'row': row
            })

        # 按加权分数降序排序
        results.sort(key=lambda x: x['weighted_score'], reverse=True)

        return results

    def _count_indicators(self, row: pd.Series) -> int:
        """计算通过指标的简单计数（用于日志和调试）"""
        count = 0
        if row['macd'] > row['macd_signal']:
            count += 1
        if row['ma_5'] > row['ma_20']:
            count += 1
        if row['volume_ma5'] and row['volume'] > row['volume_ma5']:
            count += 1
        if row['bbi'] > row['close']:
            count += 1
        if row['kdj_k'] > row['kdj_d']:
            count += 1
        if row['mtm'] > row['mtm_ma']:
            count += 1
        if row['rsi_14'] < 70:
            count += 1
        if row['williams_r'] > -20:
            count += 1
        return count

    def get_position_ratio(self, signal_level: int) -> float:
        """根据信号等级获取仓位比例（与v1相同）"""
        return self.config['signal_position_config'].get(signal_level, 0.20)
