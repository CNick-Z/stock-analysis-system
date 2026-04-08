#!/usr/bin/env python3
"""
统一数据加载模块 — 所有策略共用
=====================================
提供标准化的数据加载流程：
  1. 日线数据（OHLCV）
  2. 技术指标
  3. 财务基本面
  4. 股票基本信息（总股本等）
  5. 资金流指标（统一计算）

用法:
    from utils.data_loader import load_strategy_data
    df = load_strategy_data([2024, 2025])
"""

import warnings
import pandas as pd
import numpy as np
import logging

# pandas 2.x groupby.apply 兼容性警告（不影响结果）
warnings.filterwarnings('ignore',
    message='DataFrameGroupBy\.apply operated on the grouping columns',
    category=FutureWarning)
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path('/root/.openclaw/workspace/data/warehouse')

# ============================================================
# 基础数据加载
# ============================================================

def load_daily_data(years: List[int]) -> pd.DataFrame:
    """加载日线数据（OHLCV）"""
    frames = []
    for y in years:
        path = DATA_DIR / f'daily_data_year={y}' / 'data.parquet'
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            logger.warning(f"日线数据不存在: {path}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # ── 防重复：每个 (symbol, date) 只保留一条 ──────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['symbol', 'date'])
    if len(df) < before:
        logger.warning(f"日线数据有 {before - len(df):,} 条重复(symbol,date)已去重")
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    logger.info(f"日线数据: {len(df):,}行, {df['symbol'].nunique()}只股票, {df['date'].min()}~{df['date'].max()}")
    return df


def load_technical_indicators(years: List[int]) -> pd.DataFrame:
    """加载技术指标数据"""
    frames = []
    for y in years:
        path = DATA_DIR / f'technical_indicators_year={y}' / 'data.parquet'
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            logger.warning(f"技术指标不存在: {path}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # ── 防重复：每个 (symbol, date) 只保留一条 ──────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['symbol', 'date'])
    if len(df) < before:
        logger.warning(f"技术指标有 {before - len(df):,} 条重复(symbol,date)已去重")
    logger.info(f"技术指标: {len(df):,}行, {df['symbol'].nunique()}只股票")
    return df


def load_financial_summary(years: List[int]) -> pd.DataFrame:
    """
    加载财务摘要数据（PE/PB/ROE/净利润增速等）
    注意：财务数据是季度更新的，取最近一期可用数据
    """
    path = DATA_DIR / 'financial_summary.parquet'
    if not path.exists():
        logger.warning("财务摘要数据不存在")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[df['year'].isin(years)]
    # 取每年最新一期财务数据（季度末）
    df = df.sort_values('date').groupby(['symbol', 'year']).last().reset_index()
    # ── 防重复：groupby 后检查 (symbol, year) 是否唯一 ─────────────
    n_dup = df.duplicated(subset=['symbol', 'year']).sum()
    if n_dup:
        logger.warning(f"财务数据有 {n_dup} 条重复(symbol,year)，已通过groupby.last保留最新一条")
    logger.info(f"财务摘要: {len(df):,}行, {df['symbol'].nunique()}只股票")
    return df


def load_stock_basic_info() -> pd.DataFrame:
    """加载股票基本信息（总股本、行业等）"""
    path = DATA_DIR / 'stock_basic_info.parquet'
    if not path.exists():
        logger.warning("股票基本信息不存在")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"股票基本信息: {len(df)}只股票")
    return df


# ============================================================
# 统一数据合并
# ============================================================

def load_strategy_data(
    years: List[int],
    add_money_flow: bool = True,
) -> pd.DataFrame:
    """
    统一数据加载 — 所有策略共用入口

    合并顺序：
        daily → technical → financial → basic_info
        → money_flow (可选)

    Args:
        years: 年份列表，如 [2024, 2025]
        add_money_flow: 是否计算资金流指标（默认True）

    Returns:
        合并后的 DataFrame，按 (symbol, date) 排序
    """
    # 1. 加载各数据源
    daily = load_daily_data(years)
    tech = load_technical_indicators(years)
    financial = load_financial_summary(years)
    basic = load_stock_basic_info()

    if daily.empty:
        raise ValueError("日线数据为空，请检查数据路径")

    # 2. 合并日线 + 技术指标
    df = pd.merge(daily, tech, on=['date', 'symbol'], how='left')
    # ── 防重复：merge 后若出现 (symbol, date) 重复说明数据源有问题 ──
    n_before = len(df)
    df = df.drop_duplicates(subset=['symbol', 'date'])
    if len(df) < n_before:
        logger.warning(f"合并后有 {n_before - len(df):,} 条重复(symbol,date)已去重")
    logger.info(f"合并技术指标后: {len(df):,}行")

    # 3. 合并财务数据（取最近一期，forward-fill到每日）
    if not financial.empty:
        # 财务数据按年度组织，join时需要对齐年份
        df['year'] = pd.to_datetime(df['date']).dt.year
        # 财务表里的 date_y 是季度末日期，对我们没用，合并前删掉避免冲突
        fin_cols_available = [c for c in financial.columns
                              if c not in ('date', 'year', 'symbol')]
        financial_sub = financial[['symbol', 'year'] + fin_cols_available]
        df = pd.merge(df, financial_sub, on=['symbol', 'year'], how='left')
        # 财务字段 forward-fill（季度数据在下一期之前保持不变）
        df[fin_cols_available] = df.groupby('symbol')[fin_cols_available].ffill()
        df.drop(columns=['year'], inplace=True)
        logger.info(f"合并财务数据后: {len(df):,}行")

    # 4. 合并股票基本信息（静态数据，直接left join）
    if not basic.empty:
        # total_shares 用于资金流计算，必须有
        if 'total_shares' in basic.columns:
            basic = basic[['symbol', 'total_shares', 'industry', 'listing_date']]
            df = pd.merge(df, basic, on='symbol', how='left')
        logger.info(f"合并基本信息后: {len(df):,}行")

    # 5. 计算派生日线字段
    df = _add_derived_daily_fields(df)

    # 6. 计算资金流指标
    if add_money_flow:
        df = calculate_money_flow_indicators(df)

    # 7. 最终清理
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # 8. 打印数据概况
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_null = null_pct[null_pct > 20].head(5)
    if not high_null.empty:
        logger.warning(f"以下字段空值率>20%: {high_null.to_dict()}")

    logger.info(f"最终数据集: {len(df):,}行, {df['symbol'].nunique()}只股票, "
                f"{df['date'].min()}~{df['date'].max()}")

    return df


def _add_derived_daily_fields(df: pd.DataFrame) -> pd.DataFrame:
    """添加派生日线字段"""
    # 涨跌幅（%）
    if 'change_pct' not in df.columns and 'close' in df.columns:
        df['change_pct'] = df.groupby('symbol')['close'].pct_change() * 100

    # 复权价格标记（统一使用前复权数据，此处仅做兼容）
    # amount 字段（成交额，元）
    if 'amount' not in df.columns:
        df['amount'] = df['volume'] * (df.get('close', 0) if 'close' in df.columns else 0)

    # vol_ratio 量比
    if 'vol_ma5' in df.columns and 'volume' in df.columns:
        df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)

    # boll_pos
    if all(c in df.columns for c in ['sma_5', 'bb_lower', 'bb_upper']):
        df['boll_pos'] = ((df['sma_5'] - df['bb_lower']) /
                          (df['bb_upper'] - df['bb_lower'] + 1e-10)).clip(0, 1)

    # next_open（用于模拟盘次日开盘买入）
    if 'next_open' not in df.columns:
        df['next_open'] = df.groupby('symbol')['open'].shift(-1)

    return df


# ============================================================
# 资金流指标计算（修正版）
# ============================================================

def calculate_money_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算资金流向相关指标（全程向量化版，基于 OHLCV）

    输入字段依赖：
        open, high, low, close, volume, amount, total_shares

    修复说明（v2 向量化版）：
        1. QJJ 分母保护（涨跌停分母为0的情况）
        2. 全程 groupby().transform() 向量化，无 Python 循环
        3. XVL sign correction：使用 QJJ 绝对值防止符号反转
        4. ewm/rolling/shift 全部用 transform 在组内执行

    产出字段：
        XVL, LIJIN, LLJX, 主生量, 量基线, 量增幅, 周量, 周增幅
        money_flow_positive, money_flow_trend, money_flow_increasing,
        money_flow_weekly, money_flow_weekly_increasing
    """
    import numpy.fft as _np_fft  # noqa: suppress unused

    df = df.copy()

    # 检查必需字段
    required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"资金流计算缺少字段: {missing}，跳过")
        return df

    has_shares = 'total_shares' in df.columns

    g = df.groupby('symbol')

    # ---- 1. 成交额/成交量 = 均价（元）----
    df['金'] = df['amount'] / df['volume'].replace(0, np.nan)

    # ---- 2. PJJ 加权平均价（组内 ewm）----
    df['PJJ'] = g['close'].transform(
        lambda s: ((df.loc[s.index, 'high'] + df.loc[s.index, 'low'] + df.loc[s.index, 'close'] * 2) / 4)
                  .ewm(alpha=0.9, adjust=False).mean()
    )

    # ---- 3. QJJ 单位价格波动范围的成交量（分母保护）----
    denom = (df['high'] - df['low']) * 2 - abs(df['close'] - df['open'])
    df['QJJ'] = df['volume'] / denom.replace(0, np.nan)

    # ---- 4. XVL 资金流向（纯向量化，跨股票无污染）----
    abs_qjj = df['QJJ'].abs()
    is_pos = df['close'] > df['open']
    is_neg = df['close'] < df['open']
    is_flat = df['close'] == df['open']

    flow_in = np.where(is_pos, abs_qjj * (df['high'] - df['low']), 0.0)
    flow_in = np.where(is_neg,
                       abs_qjj * (df['high'] - df['open'] + df['close'] - df['low']),
                       flow_in)
    flow_in = np.where(is_flat, df['volume'] / 2, flow_in)

    flow_out = np.where(is_pos,
                        -abs_qjj * (df['high'] - df['close'] + df['open'] - df['low']),
                        0.0)
    flow_out = np.where(is_neg, -abs_qjj * (df['high'] - df['low']), flow_out)
    flow_out = np.where(is_flat, -df['volume'] / 2, flow_out)

    df['XVL'] = flow_in + flow_out

    # ---- 5. ZLL 成交量占比----
    if has_shares:
        df['ZLL'] = df['volume'] / df['total_shares'].replace(0, np.nan)
    else:
        df['ZLL'] = np.nan

    # ---- 6. LIJIN1 限制ZLL上限为10----
    df['LIJIN1'] = df['ZLL'].clip(upper=10) if 'ZLL' in df.columns else np.nan

    # ---- 7. LIJIN 资金流向强度----
    df['LIJIN'] = (df['XVL'] / 20) / 1.15

    # ---- 8. 主生量 = LIJIN 加权移动平均（组内 shift + 向量加权）----
    liijin_prev1 = g['LIJIN'].shift(1).fillna(0)
    liijin_prev2 = g['LIJIN'].shift(2).fillna(0)
    df['主生量'] = df['LIJIN'] * 0.55 + liijin_prev1 * 0.33 + liijin_prev2 * 0.22

    # ---- 9. GJJ 8期EMA主生量（量基线，组内 ewm）----
    df['GJJ'] = g['主生量'].transform(lambda s: s.ewm(span=8, adjust=False).mean())

    # ---- 10. LLJX 3期EMA主生量----
    df['LLJX'] = g['主生量'].transform(lambda s: s.ewm(span=3, adjust=False).mean())

    # ---- 11. 资金量 = LLJX----
    df['资金量'] = df['LLJX']

    # ---- 12. 量基线 = GJJ----
    df['量基线'] = df['GJJ']

    # ---- 13. 量增幅 LLJX变化率（带符号分类）----
    prev_lljx = g['LLJX'].shift(1)
    qzjj = ((df['LLJX'] - prev_lljx) / prev_lljx.replace(0, np.nan)) * 100

    cond1 = (df['LLJX'] > 0) & (prev_lljx < 0)
    cond2 = (df['LLJX'] < 0) & (prev_lljx < 0) & (df['LLJX'] < prev_lljx)
    df['量增幅'] = np.where(cond1, qzjj,
                            np.where(cond2, -qzjj.abs(), qzjj.abs()))

    # ---- 14. 力度----
    df['力度'] = df['LIJIN'] / 1000

    # ---- 15. 周量 过去5日LLJX总和（组内 rolling）----
    df['周量'] = g['LLJX'].transform(lambda s: s.rolling(window=5, min_periods=1).sum())

    # ---- 16. 周增幅----
    prev_week = g['周量'].shift(1)
    df['周增幅'] = ((df['周量'] - prev_week) / prev_week.replace(0, np.nan) * 100).abs()

    # ---- 17. 二元信号----
    df['money_flow_positive'] = df['资金量'] > 0
    df['money_flow_increasing'] = df['量增幅'] > 0
    df['money_flow_trend'] = df['主生量'] > df['量基线']   # 趋势向上
    df['money_flow_weekly'] = df['周量'] > 0
    df['money_flow_weekly_increasing'] = df['周增幅'] > 0

    logger.info(f"资金流指标计算完成: {len(df):,}行, "
                f"money_flow_positive={df['money_flow_positive'].sum():,}, "
                f"money_flow_trend={df['money_flow_trend'].sum():,}")

    return df


# ============================================================
# 波浪缠论信号加载（V3 专用）
# ============================================================

WAVECHAN_L2_ROOT = Path("/data/warehouse/wavechan/wavechan_cache")


def load_wavechan_signals(start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载波浪缠论 L2 信号数据（仅 V3 需要）

    从 l2_hot_year=YYYY_month=MM/data.parquet 读取，
    按 date+symbol 关联返回。

    返回字段：
        date, symbol, has_signal, signal_type, signal_status,
        total_score, signal_score, structure_score, momentum_score, chan_score,
        wave_trend, wave_state, wave_retracement,
        stop_loss, close, open, high, low, volume

    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD

    Returns:
        DataFrame，按 (date, symbol) 排序
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_dfs = []
    current = start

    while current <= end:
        year = current.year
        month = current.strftime("%m")
        cache_path = WAVECHAN_L2_ROOT / f"l2_hot_year={year}_month={month}" / "data.parquet"

        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # 日期列统一转为字符串再比较，避免类型不一致
                df["date"] = df["date"].astype(str)
                mask = (df["date"] >= start_date) & (df["date"] <= end_date)
                all_dfs.append(df[mask])
            except Exception as e:
                logger.debug(f"读取 {cache_path} 失败: {e}")

        current += timedelta(days=32)  # 跳到下个月
        current = datetime(current.year, current.month, 1)

    if not all_dfs:
        logger.warning(f"波浪L2缓存: {start_date}~{end_date} 无数据")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # 过滤有信号的记录
    if "has_signal" in result.columns:
        result = result[result["has_signal"] == True]

    logger.info(f"波浪L2信号: {len(result):,} 条, {result['symbol'].nunique()} 只股票, "
                f"{result['date'].min()}~{result['date'].max()}")

    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


# ============================================================
# 市场宽度指标（仓位调节用）
# ============================================================

def calculate_market_breadth(indicators_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场宽度指标：每日全市场 MA55 > SMA240 的股票比例

    用于仓位调节：
        breadth >= 80% → 牛市，满仓
        breadth <= 20% → 熊市，降仓
        中间 → 震荡市，标准仓位

    Args:
        indicators_df: 技术指标数据（包含 sma_55, sma_240, date）

    Returns:
        DataFrame: date, breadth_pct（全市场MA55>MA240比例）
    """
    if 'sma_55' not in indicators_df.columns or 'sma_240' not in indicators_df.columns:
        logger.warning("技术指标缺少 sma_55/sma_240，无法计算市场宽度")
        return pd.DataFrame()

    # 每日满足 MA55 > MA240 的股票比例
    broad = indicators_df.copy()
    broad['above'] = (broad['sma_55'] > broad['sma_240']).astype(float)
    # 过滤掉当天所有股票SMA都无效的日期
    breadth = (broad.dropna(subset=['above'])
                .groupby('date')['above']
                .mean()
                .reset_index()
                .rename(columns={'above': 'breadth_pct'}))
    breadth['breadth_pct'] = (breadth['breadth_pct'] * 100).round(2)
    breadth = breadth.dropna(subset=['breadth_pct'])

    logger.info(f"市场宽度: {len(breadth)} 个交易日, "
                f"范围 {breadth['breadth_pct'].min():.1f}%~{breadth['breadth_pct'].max():.1f}%")

    return breadth[['date', 'breadth_pct']]


# ============================================================
# 涨跌停状态（已整合到 _add_derived_daily_fields）
# ============================================================


# ============================================================
# 旧版兼容别名
# ============================================================

def prepare_data(years: List[int]) -> pd.DataFrame:
    """prepare_data = load_strategy_data（别名，向后兼容）"""
    return load_strategy_data(years)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    df = load_strategy_data([2024])
    print(f"\n最终数据: {df.shape}")
    print(f"列: {list(df.columns)}")
