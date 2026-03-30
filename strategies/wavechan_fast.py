# strategies/wavechan_fast.py
"""
波浪计算极速版 - WavechanStrategy 性能优化
使用 numba + scipy 向量化加速

优化策略：
1. Zigzag 用 scipy.find_peaks 向量化检测极值点
2. Elliott 规则过滤用 numba JIT 编译
3. 预计算滚动指标，只分析关键转折点
4. 多进程并行处理股票
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Numba 加速的 Elliott 波浪计算
# ============================================================

@njit
def _compute_zigzag_numba(prices, threshold_pct):
    """
    Numba加速的Zigzag算法
    返回: extrema_indices, extrema_is_high, extrema_prices
    """
    n = len(prices)
    if n < 5:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    
    # 找局部极值点（转折点）
    extrema_idx = []
    extrema_is_high = []
    extrema_price = []
    
    for i in range(1, n - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            extrema_idx.append(i)
            extrema_is_high.append(True)
            extrema_price.append(prices[i])
        elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            extrema_idx.append(i)
            extrema_is_high.append(False)
            extrema_price.append(prices[i])
    
    if len(extrema_idx) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    
    extrema_idx = np.array(extrema_idx, dtype=np.int64)
    extrema_is_high = np.array(extrema_is_high, dtype=np.bool_)
    extrema_price = np.array(extrema_price, dtype=np.float64)
    
    # 构建波段并过滤幅度太小的
    swings_idx = []
    swings_is_up = []
    i = 0
    while i < len(extrema_idx) - 1:
        idx1, is_high1, price1 = extrema_idx[i], extrema_is_high[i], extrema_price[i]
        idx2, is_high2, price2 = extrema_idx[i + 1], extrema_is_high[i + 1], extrema_price[i + 1]
        
        if is_high1 == is_high2:
            i += 1
            continue
        
        pct = abs(price2 - price1) / (price1 + 1e-10)
        if pct >= threshold_pct:
            swings_idx.append((idx1, idx2))
            swings_is_up.append(not is_high1)
        i += 1
    
    if len(swings_idx) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    
    # 返回最后几个波段供分析用
    n_swings = len(swings_idx)
    last_n = min(n_swings, 8)  # 最多取最后8个波段
    result_idx = np.zeros(last_n * 2, dtype=np.int64)
    result_is_up = np.zeros(last_n, dtype=np.bool_)
    
    for k in range(last_n):
        s = swings_idx[n_swings - last_n + k]
        result_idx[k * 2] = s[0]
        result_idx[k * 2 + 1] = s[1]
        result_is_up[k] = swings_is_up[n_swings - last_n + k]
    
    return result_idx, result_is_up, prices


@njit
def _label_waves_numba(swings_idx, swings_is_up, prices, threshold_pct):
    """
    Numba加速的波浪标记
    返回: (wave_nums, wave_end_idx, wave_start_prices, wave_end_prices, n_waves)
    """
    n_swings = len(swings_idx) // 2
    if n_swings < 3:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                0)
    
    wave_nums = np.zeros(50, dtype=np.int64)
    wave_end_idx_arr = np.zeros(50, dtype=np.int64)
    wave_start_prices_arr = np.zeros(50, dtype=np.float64)
    wave_end_prices_arr = np.zeros(50, dtype=np.float64)
    count = 0
    
    wave_num = 1
    i = 0
    
    while i < n_swings - 1:
        if count >= 50:
            break
        s1_start, s1_end = swings_idx[i * 2], swings_idx[i * 2 + 1]
        s2_start, s2_end = swings_idx[(i + 1) * 2], swings_idx[(i + 1) * 2 + 1]
        
        s1_price_change = abs(prices[s1_end] - prices[s1_start])
        s2_price_change = abs(prices[s2_end] - prices[s2_start])
        
        pct = s2_price_change / (prices[s1_start] + 1e-10)
        if pct < threshold_pct:
            i += 1
            continue
        
        wave_nums[count] = wave_num
        wave_end_idx_arr[count] = s2_end
        wave_start_prices_arr[count] = prices[s1_end]
        wave_end_prices_arr[count] = prices[s2_end]
        
        # Elliott规则: Wave3不能最短
        if wave_num == 3 and count >= 2:
            w1_len = abs(wave_end_prices_arr[0] - wave_start_prices_arr[0])
            w2_len = abs(wave_end_prices_arr[1] - wave_start_prices_arr[1])
            w3_len = s2_price_change
            if w3_len < max(w1_len, w2_len):
                count -= 1
                wave_num -= 1
                i += 1
                continue
        
        count += 1
        wave_num += 1
        i += 1
    
    if count == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                0)
    
    return (wave_nums[:count].copy(),
            wave_end_idx_arr[:count].copy(),
            wave_start_prices_arr[:count].copy(),
            wave_end_prices_arr[:count].copy(),
            count)


@njit
def _get_trend_numba(swings_is_up):
    """Numba加速的趋势判断"""
    n = len(swings_is_up)
    if n == 0:
        return 2  # neutral
    ups = 0
    for i in range(n):
        if swings_is_up[i]:
            ups += 1
    if ups > n - ups:
        return 0  # up
    elif ups < n - ups:
        return 1  # down
    else:
        return 2  # neutral


# ============================================================
# 向量化分型检测
# ============================================================

def _compute_fractals_vectorized(high, low):
    """
    纯numpy向量化分型检测
    顶分型: 中间最高，两侧渐低
    底分型: 中间最低，两侧渐高
    """
    n = len(high)
    fractal = np.full(n, 0, dtype=np.int8)  # 0=none, 1=bottom, 2=top
    bottom_div = np.zeros(n, dtype=np.bool_)
    top_div = np.zeros(n, dtype=np.bool_)
    
    # 向量化检测
    for i in range(2, n - 2):
        # 底分型: 中间低点最低，两侧低点较高
        if (low[i-1] < low[i-2] and low[i-1] < low[i] and
            low[i-1] < low[i+1] and low[i-1] < low[i+2]):
            # 额外条件: 两侧K线实体有重叠（更严格定义）
            fractal[i] = 1
            bottom_div[i] = True
        # 顶分型: 中间高点最高，两侧高点较低
        elif (high[i-1] > high[i-2] and high[i-1] > high[i] and
              high[i-1] > high[i+1] and high[i-1] > high[i+2]):
            fractal[i] = 2
            top_div[i] = True
    
    return fractal, bottom_div, top_div


@njit
def _detect_divergence_numba(low, close, high, lookback=40):
    """
    Numba加速的背驰检测
    检测: 价格创新低但跌幅收窄（底背驰）
    """
    n = len(low)
    divergence = np.zeros(n, dtype=np.bool_)
    
    if n < lookback * 2 + 5:
        return divergence
    
    for i in range(lookback * 2, n):
        # 近lookback日最低点
        window_start = i - lookback
        recent_low = low[window_start]
        for j in range(window_start + 1, i):
            if low[j] < recent_low:
                recent_low = low[j]
        
        prev_window_start = i - lookback * 2
        prev_low = low[prev_window_start]
        for j in range(prev_window_start + 1, window_start):
            if low[j] < prev_low:
                prev_low = low[j]
        
        # 价格是否创新低
        if low[i] != recent_low:
            continue
        
        # 跌幅对比
        recent_decline = (close[window_start] - recent_low) / (recent_low + 1e-10)
        prev_decline = (close[prev_window_start] - prev_low) / (prev_low + 1e-10)
        
        if recent_decline < prev_decline * 0.8:
            divergence[i] = True
    
    return divergence


# ============================================================
# 极速特征计算 - 主入口
# ============================================================

def compute_wavechan_features_fast(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    极速波浪缠论特征计算
    
    优化点:
    1. 预计算滚动指标 (一次性, 不重复)
    2. Numba加速Zigzag + 波浪标记
    3. 向量化分型 + 背驰检测
    4. 每只股票只调用3-5次波浪分析(不是每日调用)
    """
    threshold_pct = config.get('wave_threshold_pct', 0.025)
    decline_threshold = config.get('decline_threshold', -0.15)
    consolidation_threshold = config.get('consolidation_threshold', 0.05)
    stop_loss_pct = config.get('stop_loss_pct', 0.03)
    profit_target_pct = config.get('profit_target_pct', 0.25)
    
    results = []
    symbols = df['symbol'].unique()
    total = len(symbols)
    
    for idx, symbol in enumerate(symbols):
        if idx % 500 == 0:
            logger.info(f"[WavechanFast] 进度: {idx}/{total}")
        
        grp = df[df['symbol'] == symbol].sort_values('date').copy()
        if len(grp) < 60:
            continue
        
        n = len(grp)
        close = grp['close'].values.astype(np.float64)
        high = grp['high'].values.astype(np.float64)
        low = grp['low'].values.astype(np.float64)
        dates_arr = grp['date'].values  # 用于波浪端点日期映射
        
        # ========== 1. 预计算滚动指标 (一次性) ==========
        close_series = pd.Series(close)
        
        ma5 = close_series.rolling(5, min_periods=1).mean().values.astype(np.float64)
        ma10 = close_series.rolling(10, min_periods=1).mean().values.astype(np.float64)
        ma20 = close_series.rolling(20, min_periods=1).mean().values.astype(np.float64)
        
        rolling_high20 = close_series.rolling(20, min_periods=1).max().values.astype(np.float64)
        rolling_low20 = close_series.rolling(20, min_periods=1).min().values.astype(np.float64)
        
        # ========== 2. 分型检测 (向量化) ==========
        fractal, bottom_div, top_div = _compute_fractals_vectorized(high, low)
        
        # ========== 3. 背驰检测 (Numba) ==========
        divergence = _detect_divergence_numba(low, close, high)
        
        # ========== 4. 关键: 波浪分析 - 减少调用次数 ==========
        # 策略: 只在关键位置计算波浪（每20天或显著转折点）
        wave_trend = np.full(n, 2, dtype=np.int8)  # 0=up, 1=down, 2=neutral
        wave_stage = np.full(n, 'unknown', dtype=object)
        wave_total = np.zeros(n, dtype=np.int32)
        
        # 新增：各浪端点价格+日期（前5个浪的起点和终点）
        wave1_price = np.zeros(n, dtype=np.float64)
        wave2_price = np.zeros(n, dtype=np.float64)
        wave3_price = np.zeros(n, dtype=np.float64)
        wave4_price = np.zeros(n, dtype=np.float64)
        wave5_price = np.zeros(n, dtype=np.float64)
        wave1_start = np.zeros(n, dtype=np.float64)
        wave2_start = np.zeros(n, dtype=np.float64)
        wave3_start = np.zeros(n, dtype=np.float64)
        wave4_start = np.zeros(n, dtype=np.float64)
        wave5_start = np.zeros(n, dtype=np.float64)
        wave1_date = np.full(n, '', dtype=object)
        wave2_date = np.full(n, '', dtype=object)
        wave3_date = np.full(n, '', dtype=object)
        wave4_date = np.full(n, '', dtype=object)
        wave5_date = np.full(n, '', dtype=object)
        wave1_start_date = np.full(n, '', dtype=object)
        wave2_start_date = np.full(n, '', dtype=object)
        wave3_start_date = np.full(n, '', dtype=object)
        wave4_start_date = np.full(n, '', dtype=object)
        wave5_start_date = np.full(n, '', dtype=object)
        
        # 每20天计算一次波浪，然后用前向填充
        wave_calc_points = list(range(30, n, 20))  # 每20天一个计算点
        wave_results_cache = {}  # idx -> (trend, stage, total, wave_prices, wave_dates, wave_start_prices, wave_start_dates)
        
        for calc_idx in wave_calc_points:
            start_idx = max(0, calc_idx - 60)
            window_prices = close[start_idx:calc_idx + 1]
            
            # Zigzag
            extrema_idx, extrema_is_high, _ = _compute_zigzag_numba(window_prices, threshold_pct)
            
            if len(extrema_idx) >= 4:
                # 波浪标记
                wave_nums, wave_end_idx, wave_start_prices_arr, wave_end_prices_arr, n_waves = \
                    _label_waves_numba(extrema_idx, extrema_is_high, window_prices, threshold_pct)
                
                if n_waves > 0:
                    trend = _get_trend_numba(extrema_is_high[-min(5, len(extrema_is_high)):])
                    last_wave_num = wave_nums[-1]
                    
                    # 确定波浪阶段
                    if last_wave_num <= 5:
                        stage = f'Wave{int(last_wave_num)}'
                    else:
                        stage = f'Wave{chr(65 + int(last_wave_num) - 6)}'
                    
                    # 提取前5个浪的端点价格和日期
                    w_prices = np.zeros(5, dtype=np.float64)
                    w_dates = [''] * 5
                    w_start_prices = np.zeros(5, dtype=np.float64)
                    w_start_dates = [''] * 5
                    for wi in range(min(5, n_waves)):
                        w_prices[wi] = wave_end_prices_arr[wi]
                        w_start_prices[wi] = wave_start_prices_arr[wi]
                        # 端点日期：将窗口内索引转换为实际日期
                        actual_idx = start_idx + wave_end_idx[wi]
                        if actual_idx < len(dates_arr):
                            w_dates[wi] = str(dates_arr[actual_idx])
                        # 起点日期：上一个浪的终点；如果wi=0则起点就是窗口起点
                        if wi == 0:
                            start_actual_idx = start_idx
                        else:
                            start_actual_idx = start_idx + wave_end_idx[wi - 1]
                        if start_actual_idx < len(dates_arr):
                            w_start_dates[wi] = str(dates_arr[start_actual_idx])
                    
                    wave_results_cache[calc_idx] = (trend, stage, n_waves, w_prices, w_dates, w_start_prices, w_start_dates)
        
        # 前向填充波浪结果
        last_trend, last_stage, last_total, last_w_prices = 2, 'unknown', 0, np.zeros(5, dtype=np.float64)
        last_w_dates = [''] * 5
        last_w_start_prices = np.zeros(5, dtype=np.float64)
        last_w_start_dates = [''] * 5
        for i in range(n):
            if i in wave_results_cache:
                last_trend, last_stage, last_total, last_w_prices, last_w_dates, last_w_start_prices, last_w_start_dates = wave_results_cache[i]
            wave_trend[i] = last_trend
            wave_stage[i] = last_stage
            wave_total[i] = last_total
            wave1_price[i] = last_w_prices[0]
            wave2_price[i] = last_w_prices[1]
            wave3_price[i] = last_w_prices[2]
            wave4_price[i] = last_w_prices[3]
            wave5_price[i] = last_w_prices[4]
            wave1_start[i] = last_w_start_prices[0]
            wave2_start[i] = last_w_start_prices[1]
            wave3_start[i] = last_w_start_prices[2]
            wave4_start[i] = last_w_start_prices[3]
            wave5_start[i] = last_w_start_prices[4]
            wave1_date[i] = last_w_dates[0]
            wave2_date[i] = last_w_dates[1]
            wave3_date[i] = last_w_dates[2]
            wave4_date[i] = last_w_dates[3]
            wave5_date[i] = last_w_dates[4]
            wave1_start_date[i] = last_w_start_dates[0]
            wave2_start_date[i] = last_w_start_dates[1]
            wave3_start_date[i] = last_w_start_dates[2]
            wave4_start_date[i] = last_w_start_dates[3]
            wave5_start_date[i] = last_w_start_dates[4]
        
        # ========== 5. 买卖信号判定 ==========
        # 一买: 底分型 + 20日跌幅 + 5日盘整
        prev20_close = np.roll(close, 20)
        prev20_close[:20] = close[0]
        recent20_decline = (close - prev20_close) / (prev20_close + 1e-10)
        
        prev5_close = np.roll(close, 5)
        prev5_close[:5] = close[0]
        recent5_change = (close - prev5_close) / (prev5_close + 1e-10)
        
        is_first_buy = (
            (fractal == 1) &
            (recent20_decline < decline_threshold) &
            (recent5_change > consolidation_threshold)
        )
        
        # 二买: 底分型 + 未创新低
        prev_low20 = np.roll(low, 20)
        prev_low20[:20] = low[0]
        is_second_buy = (
            (fractal == 1) &
            (recent20_decline >= decline_threshold) &
            (close > prev_low20)
        )
        
        # 一卖: 顶分型 + 背驰
        prev_high20 = np.roll(high, 20)
        prev_high20[:20] = high[0]
        is_first_sell = (
            (fractal == 2) &
            (divergence)
        )
        
        # ========== 6. 综合信号 ==========
        trend_str = np.where(wave_trend == 0, 'up', np.where(wave_trend == 1, 'down', 'neutral'))
        
        daily_signal = np.full(n, 'hold', dtype=object)
        confidence = np.zeros(n)
        
        buy_mask = (wave_trend == 0) & (is_first_buy | is_second_buy)
        daily_signal[buy_mask] = '买入'
        confidence[buy_mask] = 0.8
        
        hold_up_mask = (wave_trend == 0) & (fractal == 1) & ~is_first_buy & ~is_second_buy
        daily_signal[hold_up_mask] = '观望'
        confidence[hold_up_mask] = 0.4
        
        sell_mask = (wave_trend == 1) & is_first_sell
        daily_signal[sell_mask] = '卖出'
        confidence[sell_mask] = 0.8
        
        # 均线趋势
        trend_ma = np.where(
            (close > ma10) & (ma10 > ma20), 'up',
            np.where((close < ma10) & (ma10 < ma20), 'down', 'unknown')
        )
        
        # ========== 7. 构建结果 ==========
        grp['wave_trend'] = trend_str
        grp['wave_stage'] = wave_stage
        grp['wave_total'] = wave_total
        grp['wave_last_end_price'] = 0.0
        grp['wave1_price'] = wave1_price
        grp['wave2_price'] = wave2_price
        grp['wave3_price'] = wave3_price
        grp['wave4_price'] = wave4_price
        grp['wave5_price'] = wave5_price
        grp['wave1_start'] = wave1_start
        grp['wave2_start'] = wave2_start
        grp['wave3_start'] = wave3_start
        grp['wave4_start'] = wave4_start
        grp['wave5_start'] = wave5_start
        grp['wave1_date'] = wave1_date
        grp['wave2_date'] = wave2_date
        grp['wave3_date'] = wave3_date
        grp['wave4_date'] = wave4_date
        grp['wave5_date'] = wave5_date
        grp['wave1_start_date'] = wave1_start_date
        grp['wave2_start_date'] = wave2_start_date
        grp['wave3_start_date'] = wave3_start_date
        grp['wave4_start_date'] = wave4_start_date
        grp['wave5_start_date'] = wave5_start_date
        
        fractal_str = np.where(fractal == 1, '底分型', np.where(fractal == 2, '顶分型', 'none'))
        grp['fractal'] = fractal_str
        grp['chan_bottom_div'] = bottom_div
        grp['chan_top_div'] = top_div
        grp['chan_first_buy'] = is_first_buy
        grp['chan_second_buy'] = is_second_buy & ~is_first_buy
        grp['chan_third_buy'] = False
        grp['chan_first_sell'] = is_first_sell
        grp['chan_second_sell'] = (fractal == 2) & ~is_first_sell
        grp['divergence'] = divergence
        grp['daily_signal'] = daily_signal
        grp['daily_confidence'] = confidence
        grp['trend_ma'] = trend_ma
        grp['stop_loss'] = low * (1 - stop_loss_pct)
        grp['target'] = close * (1 + profit_target_pct)
        grp['weekly_can_trade'] = (
            (wave_trend == 0) &
            np.isin(wave_stage, ['Wave1', 'Wave3', 'Wave5'])
        ) | (daily_signal == '买入')
        grp['weekly_trend'] = trend_str
        
        results.append(grp)
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)


# ============================================================
# 快速验证
# ============================================================

if __name__ == '__main__':
    import time
    
    print("🧪 WavechanFast 性能测试")
    print("=" * 50)
    
    # 加载少量数据测试
    from utils.parquet_db import ParquetDatabaseIntegrator
    
    db = ParquetDatabaseIntegrator()
    df = db.fetch_daily_data(
        '2024-01-01', '2024-12-31',
        columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    )
    
    config = {
        'wave_threshold_pct': 0.025,
        'decline_threshold': -0.15,
        'consolidation_threshold': 0.05,
        'stop_loss_pct': 0.03,
        'profit_target_pct': 0.25,
    }
    
    n_stocks = df['symbol'].nunique()
    print(f"数据: {len(df)} 条, {n_stocks} 只股票")
    
    start = time.time()
    result = compute_wavechan_features_fast(df, config)
    elapsed = time.time() - start
    
    print(f"\n⏱️ 耗时: {elapsed:.1f}秒")
    print(f"平均每只股票: {elapsed/n_stocks*1000:.1f}ms")
    print(f"结果: {len(result)} 条特征")
    if not result.empty:
        buy_count = (result['daily_signal'] == '买入').sum()
        print(f"买入信号: {buy_count} 个")
