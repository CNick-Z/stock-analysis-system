#!/usr/bin/env python3
"""
大盘技术性熊市/下跌趋势识别 - 回测研究
研究目标：哪些技术指标组合能有效识别熊市？
"""
import pandas as pd
import numpy as np
import baostock as bs
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

# ============================================================
# 1. 数据获取
# ============================================================
def get_index_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取指数日线数据"""
    bs.login()
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    bs.logout()
    
    df = pd.DataFrame(data, columns=rs.fields)
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return df

# ============================================================
# 2. 技术指标计算
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = df.copy()
    
    # ---- 移动平均线 ----
    for w in [5, 10, 20, 60]:
        df[f'ma{w}'] = df['close'].rolling(w, min_periods=1).mean()
    
    # ---- 均线排列状态 ----
    # 空头排列：ma5 < ma10 < ma20 < ma60
    df['ma_bearish_align'] = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma60'])
    # 多头排列
    df['ma_bullish_align'] = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma60'])
    
    # ---- 均线死叉（金叉）----
    df['ma5_under_ma20'] = (df['ma5'] < df['ma20']) & (df['ma5'].shift(1) >= df['ma20'].shift(1))
    df['ma20_under_ma60'] = (df['ma20'] < df['ma60']) & (df['ma20'].shift(1) >= df['ma60'].shift(1))
    df['ma5_above_ma20'] = (df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))
    df['ma20_above_ma60'] = (df['ma20'] > df['ma60']) & (df['ma20'].shift(1) <= df['ma60'].shift(1))
    
    # ---- MACD ----
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bearish'] = df['macd'] < 0
    df['macd_hist_declining'] = df['macd_hist'] < df['macd_hist'].shift(1)
    
    # MACD连续N日低于零轴
    df['macd_below_zero_days'] = df['macd_bearish'].groupby(
        (~df['macd_bearish']).cumsum()
    ).cumcount() * df['macd_bearish']
    
    # ---- RSI ----
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_bearish'] = df['rsi'] < 50
    df['rsi_oversold'] = df['rsi'] < 30
    
    # ---- 均线角度（MA20角度）----
    df['ma20_change_pct'] = df['ma20'].pct_change(periods=5)
    df['ma20_angle'] = np.degrees(np.arctan(df['ma20_change_pct'] * 10))  # 5日变化率转角度
    df['ma20_falling'] = df['ma20_angle'] < 0
    
    # ---- 波动率（ATR简化版：用5日高低差）----
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14, min_periods=1).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR占价格百分比
    df['atr_rising'] = df['atr_pct'] > df['atr_pct'].shift(1)
    
    # ---- 连续下跌天数 ----
    down_streak = (df['close'] < df['close'].shift(1)).groupby(
        (~(df['close'] < df['close'].shift(1))).cumsum()
    ).cumcount()
    df['down_streak'] = down_streak * (df['close'] < df['close'].shift(1))
    
    # ---- 均线跌破/站上 ----
    df['price_under_ma20'] = df['close'] < df['ma20']
    df['price_under_ma60'] = df['close'] < df['ma60']
    df['price_under_ma20_days'] = df['price_under_ma20'].groupby(
        (~df['price_under_ma20']).cumsum()
    ).cumcount() * df['price_under_ma20']
    df['price_under_ma60_days'] = df['price_under_ma60'].groupby(
        (~df['price_under_ma60']).cumsum()
    ).cumcount() * df['price_under_ma60']
    
    # ---- 均线空头排列天数 ----
    df['ma_bearish_days'] = df['ma_bearish_align'].groupby(
        (~df['ma_bearish_align']).cumsum()
    ).cumcount() * df['ma_bearish_align']
    
    # ---- 布林带 ----
    df['bb_mid'] = df['close'].rolling(20, min_periods=1).mean()
    df['bb_std'] = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    
    # ---- 综合下跌信号评分 ----
    df['bear_score'] = (
        df['ma_bearish_align'].astype(int) * 1 +
        (df['ma20_falling']).astype(int) * 1 +
        df['macd_bearish'].astype(int) * 1 +
        df['rsi_bearish'].astype(int) * 1 +
        (df['price_under_ma20_days'] >= 3).astype(int) * 1 +
        (df['price_under_ma60_days'] >= 5).astype(int) * 1 +
        df['ma20_under_ma60'].astype(int) * 2 +
        (df['atr_pct'] > df['atr_pct'].rolling(20).mean()).astype(int) * 1
    )
    
    return df

# ============================================================
# 3. 熊市定义（Ground Truth）
# ============================================================
# 已知熊市区间（中国A股）：
# 2007-10-16 到 2008-10-28 (大盘跌约70%)
# 2009-08-04 到 2010-07-02 (跌约25%)  
# 2011-04-18 到 2012-01-06 (跌约30%)
# 2015-06-12 到 2016-01-27 (跌约50%)
# 2018-02-06 到 2019-01-04 (跌约35%)
# 2021-02-18 到 2022-10-31 (跌约40%)
# 2023-05-09 到 2024-02-05 (跌约25%)

BEAR_PERIODS = [
    ('2007-10-16', '2008-10-28'),
    ('2009-08-04', '2010-07-02'),
    ('2011-04-18', '2012-01-06'),
    ('2015-06-12', '2016-01-27'),
    ('2018-02-06', '2019-01-04'),
    ('2021-02-18', '2022-10-31'),
    ('2023-05-09', '2024-02-05'),
]

def is_bear(date, bear_periods=BEAR_PERIODS):
    for start, end in bear_periods:
        if start <= str(date)[:10] <= end:
            return True
    return False

# ============================================================
# 4. 回测框架：测试单个指标
# ============================================================
def evaluate_indicator(df: pd.DataFrame, condition_col: str, lookback: int = 0, 
                       name: str = None, pred_series: pd.Series = None) -> dict:
    """
    评估单个指标的表现
    condition_col: 指标列名（布尔型）
    lookback: 信号必须持续多少天才确认
    pred_series: 预设预测序列（用于复合指标）
    """
    name = name or condition_col
    
    if pred_series is not None:
        pred_bear = pred_series.fillna(False)
    elif lookback > 0:
        # 需要连续N天满足条件
        valid = (df[condition_col].rolling(lookback, min_periods=lookback).min() == True)
        pred_bear = valid.fillna(False)
    else:
        pred_bear = df[condition_col].fillna(False)
    
    # 生成预测
    actual_bear = df['is_bear']
    
    # 构建混淆矩阵
    tp = (pred_bear & actual_bear).sum()
    tn = (~pred_bear & ~actual_bear).sum()
    fp = (pred_bear & ~actual_bear).sum()
    fn = (~pred_bear & actual_bear).sum()
    
    total = len(df)
    n_bear = actual_bear.sum()
    n_bull = total - n_bear
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / n_bear if n_bear > 0 else 0
    false_alarm_rate = fp / n_bull if n_bull > 0 else 0
    
    return {
        'name': name,
        'lookback': lookback,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'total_days': total,
        'bear_days': int(n_bear),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'false_alarm_rate': round(false_alarm_rate, 4),
    }

# ============================================================
# 5. 主回测
# ============================================================
def run_backtest():
    print("=" * 70)
    print("大盘技术性熊市识别 - 回测研究")
    print("=" * 70)
    
    # 获取数据（用沪深300：sh.000300）
    print("\n[1] 获取沪深300历史数据...")
    df = get_index_data('sh.000300', '2005-01-01', '2024-12-31')
    print(f"    数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}, 共 {len(df)} 个交易日")
    
    # 计算指标
    print("\n[2] 计算技术指标...")
    df = compute_indicators(df)
    
    # 标注真实熊市
    df['is_bear'] = df['date'].apply(is_bear)
    print(f"    真实熊市天数: {df['is_bear'].sum()} / {len(df)}")
    
    # ---- 6. 测试各个单一指标 ----
    print("\n[3] 测试单一指标...")
    
    indicators = [
        ('ma_bearish_align', 0, '均线空头排列(MA5<MA10<MA20<MA60)'),
        ('ma_bearish_align', 3, '均线空头排列(持续3天)'),
        ('ma_bearish_align', 5, '均线空头排列(持续5天)'),
        ('ma20_under_ma60', 0, 'MA20死叉MA60'),
        ('ma5_under_ma20', 0, 'MA5死叉MA20'),
        ('macd_bearish', 0, 'MACD<0'),
        ('macd_bearish', 5, 'MACD<0(持续5天)'),
        ('macd_hist_declining', 3, 'MACD柱状图连续3日下降'),
        ('rsi_bearish', 0, 'RSI<50'),
        ('rsi_bearish', 3, 'RSI<50(持续3天)'),
        ('price_under_ma20', 5, '价格<MA20(持续5天)'),
        ('price_under_ma60', 5, '价格<MA60(持续5天)'),
        ('ma20_falling', 3, 'MA20角度<0(持续3天)'),
        ('ma20_falling', 5, 'MA20角度<0(持续5天)'),
        ('atr_rising', 0, 'ATR上升(波动率扩大)'),
        ('down_streak', 5, '连续下跌5天'),
    ]
    
    results = []
    for col, lb, desc in indicators:
        if col in df.columns:
            r = evaluate_indicator(df, col, lb, desc)
            results.append(r)
            print(f"  {desc:40s} | 准确率:{r['accuracy']:.2%} | 熊市捕捉:{r['recall']:.2%} | 虚警率:{r['false_alarm_rate']:.2%}")
    
    # ---- 7. 测试复合指标 ----
    print("\n[4] 测试复合指标组合...")
    
    combo_results = []
    
    combos = [
        ('MA20死叉 + MACD<0', df['ma20_under_ma60'] & df['macd_bearish']),
        ('均线空头排列 + RSI<50', df['ma_bearish_align'] & df['rsi_bearish']),
        ('均线空头排列 + MACD<0 + RSI<50', df['ma_bearish_align'] & df['macd_bearish'] & df['rsi_bearish']),
        ('MA20死叉 + MACD<0 + RSI<50', df['ma20_under_ma60'] & df['macd_bearish'] & df['rsi_bearish']),
        ('均线空头排列(5天) + ATR上升', (df['ma_bearish_align'].rolling(5, min_periods=5).min() == True) & df['atr_rising']),
        ('综合下跌评分≥4分', df['bear_score'] >= 4),
        ('综合下跌评分≥5分', df['bear_score'] >= 5),
        ('综合下跌评分≥6分', df['bear_score'] >= 6),
        ('均线空头排列(3天) + MACD<0', (df['ma_bearish_align'].rolling(3, min_periods=3).min() == True) & df['macd_bearish']),
    ]
    
    for name, pred in combos:
        r = evaluate_indicator(df, None, 0, name, pred_series=pred.fillna(False))
        combo_results.append(r)
    
    for r in combo_results:
        print(f"  {r['name']:45s} | 准确率:{r['accuracy']:.2%} | 熊市捕捉:{r['recall']:.2%} | 虚警率:{r['false_alarm_rate']:.2%}")
    
    # ---- 8. 领先滞后分析 ----
    print("\n[5] 关键问题：信号能提前多久预警？")
    for start, end in BEAR_PERIODS:
        bear_start = pd.to_datetime(start)
        df_bear = df[df['date'] >= bear_start - pd.Timedelta(days=30)]
        df_bear = df_bear[df_bear['date'] <= bear_start + pd.Timedelta(days=10)]
        
        if len(df_bear) == 0:
            continue
        
        # 找到各指标首次在窗口内发出的信号
        signals_found = {}
        for col in ['ma_bearish_align', 'ma20_under_ma60', 'macd_bearish', 'rsi_bearish']:
            if col in df_bear.columns:
                signal_day = df_bear[df_bear[col]].head(1)
                if len(signal_day) > 0:
                    lead = (bear_start - signal_day['date'].iloc[0]).days
                    signals_found[col] = lead
        
        print(f"  {start} 熊市起点 | 信号提前量: {signals_found}")
    
    # ---- 9. 虚警分析 ----
    print("\n[6] 虚警案例分析（假阳性最多的指标）：")
    
    # 分析 bear_score >= 4 时哪些虚假信号最多
    combo_pred = df['bear_score'] >= 4
    fp_days = df[combo_pred & ~df['is_bear']]
    if len(fp_days) > 0:
        fp_dates = fp_days['date'].dt.date.value_counts().head(5)
        print(f"  bear_score≥4 虚警日期 Top5:")
        for d, cnt in fp_dates.items():
            print(f"    {d}: 出现 {cnt} 次记录")
    
    return df, results, combo_results

if __name__ == '__main__':
    df, single_results, combo_results = run_backtest()
    
    # 保存结果
    results_df = pd.DataFrame(single_results + combo_results)
    results_df.to_csv('/tmp/bear_market_results.csv', index=False)
    print("\n结果已保存到 /tmp/bear_market_results.csv")
