#!/usr/bin/env python3
"""
L2 Cache 快速重建脚本
========================

策略：用 wavechan_fast.py 一次性计算所有股票所有年份的波浪特征，
然后映射到 L2 Cache 信号格式，比逐月逐股票用 WaveEngine 快 ~16倍。

L2 Schema 字段：
  date, symbol,
  signal_score(40%), structure_score(30%), momentum_score(20%), chan_score(10%),
  total_score, has_signal, signal_type, signal_status,
  wave_state, wave_trend, wave_retracement,
  rsi, macd_hist, divergence, volume_ratio, fractal,
  stop_loss, close, open, high, low, volume

用法：
  python3 rebuild_l2_fast.py                    # 重建全部年份（2018-2024）
  python3 rebuild_l2_fast.py --year 2024          # 只重建某年
"""

import os, sys, time, warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

# ============================================================
# 配置
# ============================================================
DATA_WAREHOUSE = "/root/.openclaw/workspace/data/warehouse"
L2_CACHE_ROOT = "/data/warehouse/wavechan/wavechan_cache"
L2_COLUMNS = [
    'date', 'symbol',
    'signal_score', 'structure_score', 'momentum_score', 'chan_score',
    'total_score', 'has_signal', 'signal_type', 'signal_status',
    'wave_state', 'wave_trend', 'wave_retracement',
    'rsi', 'macd_hist', 'divergence', 'volume_ratio', 'fractal',
    'stop_loss', 'close', 'open', 'high', 'low', 'volume'
]
WAVE_CONFIG = {
    'wave_threshold_pct': 0.025,
    'decline_threshold': -0.15,
    'consolidation_threshold': 0.05,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
}

# 信号评分映射
SIGNAL_SCORE_MAP = {
    'C_BUY_confirmed': 40,
    'W2_BUY_confirmed': 35,
    'W2_BUY_alert': 25,
    'W4_BUY_confirmed': 25,
    'W4_BUY_alert': 15,
}
STRUCTURE_SCORE_MAP = {
    'W2_shallow': 20,
    'W2_optimal': 25,
    'W2_deep': 15,
}
MOMENTUM_SCORE_MAP = {
    'positive': 15,
    'divergence': 5,
}
CHAN_SCORE_MAP = {
    'first_buy': 10,
    'second_buy': 8,
    'third_buy': 5,
}


# ============================================================
# 技术指标计算（辅助）
# ============================================================

def _compute_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gains).rolling(period).mean()
    avg_loss = pd.Series(losses).rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values


def _compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow
    signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    return macd - signal_line


# ============================================================
# 信号映射
# ============================================================

def _map_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 fast 版输出映射为 L2 Cache 信号格式
    """
    df = df.copy()

    # ---- RSI ----
    if 'rsi' not in df.columns:
        df['rsi'] = _compute_rsi(df['close'].values)

    # ---- MACD Hist ----
    if 'macd_hist' not in df.columns:
        df['macd_hist'] = _compute_macd_hist(df['close'].values)

    # ---- volume_ratio ----
    if 'volume_ratio' not in df.columns:
        df['vol_ma5'] = df['volume'].rolling(5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma5'].replace(0, 1)
        df.drop(columns=['vol_ma5'], inplace=True)

    # ---- wave_trend 格式统一：fast版用 up/down/neutral，V3策略期望 long/neutral ----
    # 映射：up → long（上升趋势），down → down（下降趋势），neutral → neutral
    df['wave_trend'] = df['wave_trend'].map({
        'up': 'long',
        'down': 'down',
        'neutral': 'neutral',
        '': 'neutral'
    }).fillna('neutral')

    # ---- wave_stage 格式统一：fast版用 "Wave2"，V3策略期望 "w2_formed" ----
    if 'wave_stage' in df.columns:
        stage_map = {
            'Wave1': 'w1_formed', 'Wave2': 'w2_formed',
            'Wave3': 'w3_formed', 'Wave4': 'w4_formed',
            'Wave5': 'w5_formed', 'unknown': 'unknown',
            '': 'unknown'
        }
        df['wave_stage'] = df['wave_stage'].map(lambda x: stage_map.get(x, x)).fillna('unknown')

    # ---- wave_retracement: 从 wave prices 估算 ----
    if 'wave_retracement' not in df.columns:
        retr = np.full(len(df), np.nan)
        cond = (df['wave2_price'] > 0) & (df['wave2_start'] > 0) & (df['wave1_price'] > 0) & (df['wave1_start'] > 0)
        w1_range = df.loc[cond, 'wave1_price'] - df.loc[cond, 'wave1_start']
        w1_range = w1_range.replace(0, np.nan)
        w2_drop = df.loc[cond, 'wave2_start'] - df.loc[cond, 'wave2_price']
        retr[cond.values] = (w2_drop / w1_range).fillna(np.nan).values
        df['wave_retracement'] = retr

    # ---- 确定 signal_type ----
    c_buy_mask = df['chan_first_buy'].astype(bool) & (df['fractal'] == '底分型')
    w2_buy_mask = df['chan_second_buy'].astype(bool)
    w4_buy_mask = (df['wave_stage'] == 'w4_formed') & (df['fractal'] == '底分型')
    w4_buy_mask = w4_buy_mask & ~c_buy_mask & ~w2_buy_mask

    df['signal_type'] = 'none'
    df.loc[c_buy_mask, 'signal_type'] = 'C_BUY'
    df.loc[w2_buy_mask, 'signal_type'] = 'W2_BUY'
    df.loc[w4_buy_mask, 'signal_type'] = 'W4_BUY'

    df['signal_status'] = 'confirmed'
    df['has_signal'] = df['signal_type'] != 'none'

    # ---- signal_score (40%) ----
    df['signal_score'] = 0.0
    df.loc[c_buy_mask, 'signal_score'] = SIGNAL_SCORE_MAP['C_BUY_confirmed']
    df.loc[w2_buy_mask, 'signal_score'] = SIGNAL_SCORE_MAP['W2_BUY_confirmed']
    df.loc[w4_buy_mask, 'signal_score'] = SIGNAL_SCORE_MAP['W4_BUY_confirmed']

    # ---- structure_score (30%) ----
    df['structure_score'] = 0.0
    retr = df['wave_retracement'].values
    valid_retr = ~np.isnan(retr)
    w2_sig = df['signal_type'] == 'W2_BUY'
    if w2_sig.any():
        r = retr.copy()
        # W2浅(0.382-0.5)
        mask = w2_sig.values & valid_retr & (r > 0.382) & (r < 0.5)
        df.loc[mask, 'structure_score'] = STRUCTURE_SCORE_MAP['W2_shallow']
        # W2最佳(0.5-0.618)
        mask = w2_sig.values & valid_retr & (r >= 0.5) & (r <= 0.618)
        df.loc[mask, 'structure_score'] = STRUCTURE_SCORE_MAP['W2_optimal']
        # W2深(>0.618)
        mask = w2_sig.values & valid_retr & (r > 0.618)
        df.loc[mask, 'structure_score'] = STRUCTURE_SCORE_MAP['W2_deep']

    # ---- momentum_score (20%) ----
    df['momentum_score'] = 0.0
    pos_mask = df['wave_trend'] == 'up'
    div_mask = df['divergence'].astype(bool)
    df.loc[pos_mask, 'momentum_score'] = MOMENTUM_SCORE_MAP['positive']
    df.loc[div_mask, 'momentum_score'] = MOMENTUM_SCORE_MAP.get('divergence', 0)

    # ---- chan_score (10%) ----
    df['chan_score'] = 0.0
    df.loc[c_buy_mask, 'chan_score'] = CHAN_SCORE_MAP['first_buy']
    df.loc[w2_buy_mask, 'chan_score'] = CHAN_SCORE_MAP['second_buy']

    # ---- total_score ----
    df['total_score'] = (
        df['signal_score'] * 0.4 +
        df['structure_score'] * 0.3 +
        df['momentum_score'] * 0.2 +
        df['chan_score'] * 0.1
    )

    # ---- wave_state ----
    df['wave_state'] = df.get('wave_stage', 'unknown')

    return df


# ============================================================
# 主函数
# ============================================================

def rebuild_l2_for_year(year: int, db: ParquetDatabaseIntegrator) -> int:
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"重建 {year} 年 L2 Cache")
    print(f"{'='*50}")

    # lookback 3个月
    lookback_start = f"{year-1}-10-01"
    end_date = f"{year}-12-31"

    print(f"加载数据: {lookback_start} ~ {end_date}")
    try:
        year_df = db.fetch_daily_data(lookback_start, end_date)
    except Exception as e:
        print(f"加载失败: {e}")
        return 0

    if 'name' in year_df.columns:
        year_df = year_df[~year_df['name'].str.contains('ST', na=False)]
    year_df = year_df.dropna(subset=['close', 'volume'])

    print(f"数据: {len(year_df):,} 行, {year_df['symbol'].nunique()} 只股票")

    print(f"运行 wavechan_fast ...")
    t1 = time.time()
    features = compute_wavechan_features_fast(year_df, WAVE_CONFIG)
    print(f"  → {time.time()-t1:.0f}秒")

    features = features[features['date'].str.startswith(str(year))].copy()
    print(f"  → {year}年共 {len(features):,} 条")

    print(f"映射信号 ...")
    mapped = _map_signals(features)

    out_cols = [c for c in L2_COLUMNS if c in mapped.columns]
    result = mapped[out_cols].copy()

    # 按月分文件写出（跳过已存在的月份）
    elapsed = time.time() - t0
    result['_month'] = pd.to_datetime(result['date']).dt.strftime('%m')
    written = 0
    for month, g in result.groupby('_month'):
        month_dir = Path(L2_CACHE_ROOT) / f"l2_hot_year={year}_month={month}"
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / "data.parquet"
        if out_path.exists():
            print(f"  {year}-{month}: 已存在，跳过")
            continue
        g.drop(columns=['_month']).to_parquet(out_path, index=False)
        print(f"  {year}-{month}: {len(g):,} 条 → {out_path}")
        written += len(g)

    print(f"   总用时: {elapsed:.0f}秒")
    return written


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=None, help='只重建某年')
    args = parser.parse_args()

    print(f"WaveChan L2 Cache 快速重建")
    print(f"数据源: {DATA_WAREHOUSE}")
    print(f"输出: {L2_CACHE_ROOT}")

    db = ParquetDatabaseIntegrator(DATA_WAREHOUSE)
    years = [args.year] if args.year else [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    total_records = 0
    for year in years:
        n = rebuild_l2_for_year(year, db)
        total_records += n

    print(f"\n{'='*50}")
    print(f"全部完成！共写出 {total_records:,} 条记录")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
