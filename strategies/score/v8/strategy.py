#!/usr/bin/env python3
"""
V8 IC增强版策略 — ScoreV8Strategy
====================================
v6核心 + IC增强过滤 + 极简缓存，实现标准策略接口。

接口说明
───────────
prepare(dates, full_df):
  对全量 year df 按股票分组执行 shift/rolling（正确性），
  缓存 8 列到 self._ic_cache{date → DataFrame}。

filter_buy(full_df, date):
  框架传完整 year df + 当前交易日。
  从缓存取当日候选，应用涨跌停过滤后返回。

score(df):
  对 filter_buy 返回的候选评分排序。

should_sell(row, pos, market):
  止损 / 止盈 / 方案A出场（跌MA20+资金流出，连续3天）。

评分公式
───────────
v6_score = (
  _ma_cond      * 0.1622 +
  _sma10_change * 加权系数   +
  _macd_cond    * 0.1366 +
  _vol_cond     * 0.1704 +
  (rsi_14<70)  * 0.0597 +
  (kdj_k<80)   * 0.0597 +
  (cci_20<100) * 0.0597 +
  (close<bb_upper) * 0.1191 +
  _macd_jc     * 0.0873 +
  growth.between(0.5,6.0) * 0.06
)
score = v6_score + ic_bonus
"""

import logging
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# 工具函数
# =============================================================================

def _is_buy_row(
    row: pd.Series,
    exclude: bool,
    ma_cond: bool,
    vol_cond: bool,
    macd_cond: bool,
    macd_jc: bool,
    jc_cond: bool,
    trend_cond: bool,
    rsi_f: bool,
    price_f: bool,
) -> bool:
    """判断单行是否满足全部入场条件（供 _score_row_v8 调用）"""
    if exclude:
        return False
    if not (row['close'] >= row['open'] and row['high'] <= row['open'] * 1.06):
        return False
    if not (ma_cond and vol_cond and macd_cond and trend_cond and rsi_f and price_f):
        return False
    if not (jc_cond or macd_jc):
        return False
    return True


def _score_row_v8(row: pd.Series) -> float:
    """
    对单行计算V8评分（用于模拟盘逐行评分）
    与 DataFrame 批量计算结果完全一致
    """
    score = 0.0

    # 先做条件过滤（与 filter_buy 一致）
    if row.get('_exclude', False):
        return 0.0
    if not (row['close'] >= row['open'] and row['high'] <= row['open'] * 1.06):
        return 0.0
    for cond_col in ('_ma_cond', '_vol_cond', '_macd_cond', '_trend_cond', '_rsi_f', '_price_f'):
        if not row.get(cond_col, False):
            return 0.0
    if not (row.get('_jc_cond', False) or row.get('_macd_jc', False)):
        return 0.0

    # V6基础分
    if row.get('_ma_cond', False):
        score += 0.1622

    # 均线角度
    sma10_change = row.get('_sma10_change', 0)
    if not pd.isna(sma10_change) and sma10_change > 0:
        score += min(sma10_change, 0.0854)

    # MACD
    if row.get('_macd_cond', False):
        score += 0.1366
    if row.get('_macd_jc', False):
        score += 0.0873

    # 成交量
    vr = min(row.get('vol_ratio', 0), 3)
    if row.get('_vol_cond', False):
        score += vr * 0.1704

    # 超买超卖
    if row.get('rsi_14', 100) < 70:
        score += 0.0597
    if row.get('kdj_k', 100) < 80:
        score += 0.0597
    if row.get('cci_20', 100) < 100:
        score += 0.0597

    # BOLL
    if row.get('close', 0) < row.get('bb_upper', float('inf')):
        score += 0.1191

    # 涨幅
    g = row.get('growth', 0)
    if 0.5 <= g <= 6.0:
        score += 0.06
    elif g > 0:
        score += 0.03

    # IC加分
    score += row.get('ic_bonus', 0.0)

    return score


# =============================================================================
# 兼容层：保留原有模块级全局函数，供旧脚本使用
# 新代码应使用 ScoreV8Strategy 类
# =============================================================================

def compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：计算所有选股条件（批量 DataFrame）"""
    df = df.copy()
    df['growth'] = (df['close'] - df['open']) / df['open'] * 100

    # 均线
    df['ma_condition'] = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])

    # 成交量（⚠️ 未 groupby，仅兼容旧脚本使用）
    df['volume_condition'] = (df['volume'] > df['volume'].shift(1) * 1.5) | \
                             (df['volume'] > df['vol_ma5'] * 1.2)

    # MACD（⚠️ 未 groupby）
    df['macd_condition'] = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
    df['macd_jc'] = (df['macd'] > df['macd_signal']) & \
                     (df['macd'].shift(1) <= df['macd_signal'].shift(1))

    # 金叉条件（⚠️ 未 groupby）
    df['jc_condition'] = (
        (df['sma_5'] > df['sma_5'].shift(1)) &
        (df['sma_5'].shift(1) > df['sma_5'].shift(2)) &
        (df['sma_20'] > df['sma_20'].shift(1)) &
        (abs(df['sma_5'] - df['sma_20']) / df['sma_20'] < 0.02)
    )

    # 趋势
    df['trend_condition'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])

    # RSI / 价格
    df['rsi_filter'] = (df['rsi_14'] >= 50) & (df['rsi_14'] <= 60)
    df['price_filter'] = (df['close'] >= 3) & (df['close'] <= 15)

    return df


def apply_ic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：IC增强过滤"""
    df = df.copy()

    if 'turnover_rate_y' in df.columns:
        df['turnover_rate'] = df['turnover_rate_y']
    elif 'turnover_rate_x' in df.columns:
        df['turnover_rate'] = df['turnover_rate_x']

    df['ic_exclude'] = (
        (df['rsi_14'] > 70) | (df['rsi_14'] < 35) |
        (df['turnover_rate'] > 2.79) |
        (df['williams_r'] < -90) |
        (df['cci_20'] < -200)
    )

    df['ic_buy'] = (
        (~df['ic_exclude']) &
        df.get('ma_condition', pd.Series(True, index=df.index)) &
        df.get('volume_condition', pd.Series(True, index=df.index)) &
        df.get('macd_condition', pd.Series(True, index=df.index)) &
        (df.get('jc_condition', pd.Series(False, index=df.index)) |
         df.get('macd_jc', pd.Series(False, index=df.index))) &
        df.get('trend_condition', pd.Series(True, index=df.index)) &
        df.get('rsi_filter', pd.Series(True, index=df.index)) &
        df.get('price_filter', pd.Series(True, index=df.index))
    )

    df['ic_bonus'] = 0.0
    df.loc[df['cci_20'] < -100, 'ic_bonus'] += 0.10
    df.loc[df['williams_r'] < -80, 'ic_bonus'] += 0.05
    df.loc[df['turnover_rate'] < 0.42, 'ic_bonus'] += 0.05
    df.loc[df['vol_ratio'] < 0.71, 'ic_bonus'] += 0.05

    return df


def compute_v6_score(df: pd.DataFrame) -> pd.DataFrame:
    """兼容层：V6评分"""
    df = df.copy()
    df['_sma10_change'] = (
        (df['sma_10'] / df['sma_10'].shift(1) - 1).abs().clip(0, 0.1) * 10 * 0.0854
    )

    df['v6_score'] = (
        df.get('ma_condition', pd.Series(0.0, index=df.index)).astype(float) * 0.1622 +
        df['_sma10_change'].fillna(0) +
        df.get('macd_condition', pd.Series(0.0, index=df.index)).astype(float) * 0.1366 +
        df.get('volume_condition', pd.Series(0.0, index=df.index)).astype(float) * 0.1704 +
        (df['rsi_14'] < 70).astype(float) * 0.0597 +
        (df['kdj_k'] < 80).astype(float) * 0.0597 +
        (df['cci_20'] < 100).astype(float) * 0.0597 +
        (df['close'] < df['bb_upper']).astype(float) * 0.1191 +
        df.get('macd_jc', pd.Series(0.0, index=df.index)).astype(float) * 0.0873 +
        df['growth'].between(0.5, 6.0).astype(float) * 0.06
    )
    return df


# =============================================================================
# 策略类
# =============================================================================

class ScoreV8Strategy:
    """
    V8 IC增强版策略类

    【入场条件】
      - 阳线：收涨且涨幅 <= 6%
      - ma_condition: SMA5 > SMA10 且 SMA10 < SMA20
      - volume_condition: 放量1.5倍 或 超过5日均量1.2倍
      - macd_condition: MACD < 0 且 > signal
      - jc_condition: SMA5连续2日上升 AND SMA20当日上升 AND |SMA5-SMA20|/SMA20 < 2%
        （或 macd_jc：MACD金叉）
      - trend_condition: SMA20 < SMA55 且 SMA55 > SMA240
      - rsi_filter: RSI 在 50-60 之间
      - price_filter: 价格 3-15 元

    【IC增强过滤 - 剔除】
      - RSI > 70 或 < 35
      - 换手率 > 2.79%
      - WR < -90
      - CCI < -200

    【IC增强过滤 - 加分】
      - CCI < -100: +0.10
      - WR < -80:   +0.05
      - 换手率 < 0.42%: +0.05
      - vol_ratio < 0.71: +0.05

    【出场规则】
      - 止损: 5%
      - 止盈: 15%
      - 方案A出场: 收盘价跌破SMA20 且 money_flow_trend == False，连续3天才触发
        （计数器由策略内部 on_tick() 维护）

    【持仓管理】
      - 最多持仓: 5只
      - 单只仓位上限: 20%
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.take_profit = self.config.get('take_profit', 0.15)
        self.max_positions = self.config.get('max_positions', 5)
        self.position_size = self.config.get('position_size', 0.20)
        self.rsi_filter_min = self.config.get('rsi_filter_min', 50)
        self.rsi_filter_max = self.config.get('rsi_filter_max', 60)

        self.name = "V8ScoreStrategy"
        self.REQUIRED_COLUMNS: list = []

        # ── 极简缓存 {date → DataFrame(index=symbol))} ──────────────
        # 仅缓存 score() 直接需要的 8 列：
        #   _ic_buy, ic_bonus, _sma10_change,
        #   _ma_cond, _macd_cond, _vol_cond, _macd_jc, growth
        # 其他条件在场内直接用原始列算
        self._ic_cache: dict = {}

    # ------------------------------------------------------------------
    # prepare: 全量 year df 按股票分组预计算条件，缓存 8 列
    # ------------------------------------------------------------------
    def prepare(self, dates: list, full_df: pd.DataFrame):
        """
        预计算并缓存全场每个日期的选股标记。

        ⚠️ 只缓存 score() 直接使用的 8 列，其余条件在场内当场算。
        ⚠️ 所有 shift/rolling 必须按 symbol 分组，否则跨股票污染。

        内存估算：8列 × 1.2M行 × 8字节 ≈ 77MB
        """
        if full_df is None or full_df.empty:
            logger.warning("V8 prepare: full_df 为空")
            return

        logger.info("V8 prepare: 开始预计算（按股票分组 shift/rolling）...")
        df = full_df.copy()
        g = df.groupby('symbol', sort=False)

        # ── 0. 修复 turnover_rate（如 parquet 全为0则从 total_shares 计算）──
        # 正确公式：换手率 = 成交量(股) / 总股本(股) * 100
        # 注意：不能用 volume*100*close/amount，因为 amount=volume*close（该数据集的 amount 就是成交额）
        if 'turnover_rate_y' in df.columns:
            df['turnover_rate'] = df['turnover_rate_y']
        elif 'turnover_rate_x' in df.columns:
            df['turnover_rate'] = df['turnover_rate_x']
        # 用 total_shares 正确计算换手率
        if df['turnover_rate'].sum() == 0 and 'total_shares' in df.columns:
            df['turnover_rate'] = (
                df['volume'] / df['total_shares'].replace(0, float('nan')) * 100
            ).fillna(0).clip(0, 30)
            logger.info("  turnover_rate 已从 total_shares 重建（parquet 原值全为0）")
        elif df['turnover_rate'].sum() == 0:
            # 如果没有 total_shares，换手率设为 0（不做排除）
            df['turnover_rate'] = 0.0
            logger.info("  turnover_rate 设为 0（无 total_shares 数据）")

        # ── 1. 原始条件列（中间结果，不缓存，只给后续步骤用）──
        df['growth'] = (df['close'] - df['open']) / df['open'] * 100

        # 剔除
        df['_exclude'] = (
            (df['rsi_14'] > 70) | (df['rsi_14'] < 35) |
            (df['turnover_rate'] > 2.79) |
            (df['williams_r'] < -90) |
            (df['cci_20'] < -200)
        )

        # 买入条件（全部用 groupby shift，正确）
        df['_ma_cond']    = (df['sma_5'] > df['sma_10']) & (df['sma_10'] < df['sma_20'])
        df['_vol_cond']   = (df['volume'] > g['volume'].shift(1) * 1.5) | \
                             (df['volume'] > df['vol_ma5'] * 1.2)
        df['_macd_cond']  = (df['macd'] < 0) & (df['macd'] > df['macd_signal'])
        df['_macd_jc']   = (df['macd'] > df['macd_signal']) & \
                             (g['macd'].shift(1) <= g['macd_signal'].shift(1))
        df['_jc_cond']   = (
            (g['sma_5'].shift(1) > g['sma_5'].shift(2)) &
            (g['sma_20'].shift(1) > g['sma_20'].shift(2)) &
            (abs(g['sma_5'].shift(1) - g['sma_20'].shift(1)) / g['sma_20'].shift(1) < 0.02)
        )
        df['_trend_cond'] = (df['sma_20'] < df['sma_55']) & (df['sma_55'] > df['sma_240'])
        df['_rsi_f']      = (df['rsi_14'] >= self.rsi_filter_min) & (df['rsi_14'] <= self.rsi_filter_max)
        df['_price_f']    = (df['close'] >= 3) & (df['close'] <= 15)

        # ── 2. 合并买入标记 ─────────────────────────────────────────
        df['_ic_buy'] = (
            (~df['_exclude']) &
            (df['close'] >= df['open']) &
            (df['high'] <= df['open'] * 1.06) &
            df['_ma_cond'] &
            df['_vol_cond'] &
            df['_macd_cond'] &
            (df['_jc_cond'] | df['_macd_jc']) &
            df['_trend_cond'] &
            df['_rsi_f'] &
            df['_price_f']
        )

        # ── 3. IC 加分 ───────────────────────────────────────────────
        df['ic_bonus'] = 0.0
        df.loc[df['cci_20'] < -100, 'ic_bonus'] += 0.10
        df.loc[df['williams_r'] < -80, 'ic_bonus'] += 0.05
        df.loc[df['turnover_rate'] < 0.42, 'ic_bonus'] += 0.05
        df.loc[df['vol_ratio'] < 0.71, 'ic_bonus'] += 0.05

        # ── 4. _sma10_change（评分用均线变化率）──
        df['_sma10_change'] = (
            (df['sma_10'] / g['sma_10'].shift(1) - 1).abs().clip(0, 0.1) * 10 * 0.0854
        )

        # ── 5. 构建极简缓存（仅 8 列）────────────────────────────────
        # 注意：growth 已在上方计算好，直接缓存
        cache_cols = [
            'symbol', 'date',
            '_ic_buy', 'ic_bonus', '_sma10_change',
            '_ma_cond', '_macd_cond', '_vol_cond', '_macd_jc',
            'growth',
        ]
        cache_df = df[cache_cols].copy()

        # 释放中间 df 内存
        del df

        for date, grp in cache_df.groupby('date'):
            self._ic_cache[date] = grp.set_index('symbol')

        logger.info(
            f"V8 prepare: 预计算完成，{len(self._ic_cache)} 个交易日，"
            f"~{len(cache_df) // max(len(self._ic_cache), 1)} 股/日，"
            f"缓存 8 列 ✓"
        )
        del cache_df

    # ------------------------------------------------------------------
    # filter_buy: 查缓存返回当日候选（含全部条件列供 score() 使用）
    # ------------------------------------------------------------------
    def filter_buy(self, full_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        过滤候选股票

        Args:
            full_df: 完整 year 日线数据
            date: 当前交易日

        Returns:
            过滤后的 DataFrame（已通过全部入场条件，含 score() 需要的全部列）
        """
        if date not in self._ic_cache:
            return pd.DataFrame()

        cached = self._ic_cache[date]   # DataFrame, index=symbol

        # ── 1. 涨跌停过滤（在场内对原始数据做，不依赖缓存）──
        daily = full_df[full_df['date'] == date].copy()
        if daily.empty:
            return daily

        if 'limit_up' in daily.columns:
            daily = daily[daily['limit_up'] != True]
        if 'limit_down' in daily.columns:
            daily = daily[daily['limit_down'] != True]

        # ── 2. 取涨跌停过滤后的 symbol，与缓存 merge──
        syms = set(daily['symbol'])
        cached_syms = cached.index.intersection(syms)
        if not len(cached_syms):
            return pd.DataFrame()

        # 缓存 merge（缓存已按 symbol 建索引）
        daily = daily[daily['symbol'].isin(cached_syms)].copy()
        cache_cols = ['_ic_buy', 'ic_bonus', '_sma10_change',
                      '_ma_cond', '_macd_cond', '_vol_cond', '_macd_jc', 'growth']
        available = [c for c in cache_cols if c in cached.columns]
        if available:
            # 删除已存在的同名列（避免 join 时 overlap）
            daily = daily.drop(columns=[c for c in available if c in daily.columns])
            daily = daily.set_index('symbol')
            daily = daily.join(cached.loc[cached_syms, available], how='left')
            daily = daily.reset_index()   # 恢复 symbol 列
        else:
            # 缓存为空时，直接从 daily 的同名列读取
            daily = daily.set_index('symbol')

        # ── 3. IC 买入过滤 ──────────────────────────────────────────
        result = daily[daily['_ic_buy'] == True]

        # ── 4. score() 需要的其他条件列（直接取场内原始列，不重复算）──
        #   _jc_cond / _trend_cond / _rsi_f / _price_f / _exclude
        #   在这里补上（从 full_df 补，避免 join 时丢失）
        if result.empty:
            return result

        result = result.copy()
        # 重新算一遍（简单比较列，不涉及 shift，直接用当日数据即可）
        g = full_df[full_df['date'] == date].groupby('symbol', sort=False)
        date_df = full_df[full_df['date'] == date].set_index('symbol')
        for col in ('_jc_cond', '_trend_cond', '_rsi_f', '_price_f', '_exclude'):
            if col not in result.columns and col in date_df.columns:
                result[col] = result['symbol'].map(date_df[col])

        return result

    # ------------------------------------------------------------------
    # score: 评分排序
    # ------------------------------------------------------------------
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评分排序（在 filter_buy 基础上评分）

        ⚠️ 依赖 filter_buy 返回的列：
               _sma10_change, _ma_cond, _macd_cond, _vol_cond,
               _macd_jc, growth, ic_bonus
        """
        if df.empty:
            return df

        df = df.copy()
        df['_sma10_change'] = df['_sma10_change'].fillna(0)

        df['v6_score'] = (
            df['_ma_cond'].astype(float) * 0.1622 +
            df['_sma10_change'].fillna(0) +
            df['_macd_cond'].astype(float) * 0.1366 +
            df['_vol_cond'].astype(float) * 0.1704 +
            (df['rsi_14'] < 70).astype(float) * 0.0597 +
            (df['kdj_k'] < 80).astype(float) * 0.0597 +
            (df['cci_20'] < 100).astype(float) * 0.0597 +
            (df['close'] < df['bb_upper']).astype(float) * 0.1191 +
            df['_macd_jc'].astype(float) * 0.0873 +
            df['growth'].between(0.5, 6.0).astype(float) * 0.06
        )
        df['score'] = df['v6_score'] + df['ic_bonus'].fillna(0)
        df = df.sort_values('score', ascending=False)
        return df

    # ------------------------------------------------------------------
    # should_sell: 出场判断
    # ------------------------------------------------------------------
    def should_sell(
        self,
        row: pd.Series,
        pos: dict,
        market: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        判断是否应该出场

        优先级：止损 > 止盈 > 方案A出场
          1. 止损: next_open < 入场价 * (1 - stop_loss)
          2. 止盈: next_open > 入场价 * (1 + take_profit)
          3. 方案A: 跌破MA20 + 资金流出，连续3天
        """
        price = row.get('next_open') or row.get('close')
        if pd.isna(price):
            return False, ""

        entry = pos.get('entry_price', 0)
        if entry <= 0:
            return False, ""

        # 止损
        if price < entry * (1 - self.stop_loss):
            return True, f"STOP_LOSS @ {price:.2f}"

        # 止盈
        if price > entry * (1 + self.take_profit):
            return True, f"TAKE_PROFIT @ {price:.2f}"

        # 方案A：跌破MA20 + 资金流出，连续3天
        ma20 = row.get('sma_20')
        mf_trend = row.get('money_flow_trend', True)  # 默认True（不触发）

        if not pd.isna(ma20) and price < ma20 and mf_trend is False:
            n_days = pos.setdefault('_exit_n_days', 0) + 1
            pos['_exit_n_days'] = n_days
            if n_days >= 3:
                return True, f"EXIT_MA20 @ {price:.2f}"
        else:
            pos['_exit_n_days'] = 0

        return False, ""

    # ------------------------------------------------------------------
    # on_tick: 策略轮 tick（计数器更新）
    # ------------------------------------------------------------------
    def on_tick(self, row: pd.Series, pos: dict, market: Optional[dict] = None):
        """支持2参数和3参数调用"""
        pass
