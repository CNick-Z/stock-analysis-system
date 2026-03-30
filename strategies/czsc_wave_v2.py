"""
WaveChan v2: CZSC波浪识别系统

核心思路:
- 周线确定趋势（日线聚合为周线，用CZSC提取笔）
- 日线寻找买点（回调到位 + 底分型）
- 斐波那契预判点位

依赖: czsc
"""

import pandas as pd
import numpy as np
from czsc import CZSC, RawBar, Freq
from typing import Tuple, List, Dict, Optional


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """将日线数据聚合为周线"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').apply(lambda x: x.start_time)
    
    weekly = df.groupby('week').agg({
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    weekly.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    return weekly.sort_values('date').reset_index(drop=True)


def get_czsc_bis(bars: List[RawBar]) -> List:
    """获取CZSC笔列表"""
    if len(bars) < 9:
        return []
    c = CZSC(bars)
    return c.finished_bis


def analyze_weekly_trend(df_daily: pd.DataFrame, symbol: str) -> Dict:
    """
    周线趋势分析
    
    Returns:
        {
            'trend': 'up'/'down'/'neutral',
            'bis': [...],  # 周线笔列表
            'last_bi_direction': 'up'/'down',
            'wave_structure': 'W?调整中/W?进行中'
        }
    """
    # 聚合为周线
    df_sym = df_daily[df_daily['symbol'] == symbol].copy()
    if len(df_sym) < 20:
        return {'trend': 'neutral', 'bis': [], 'last_bi_direction': None, 'wave_structure': '数据不足'}
    
    weekly = aggregate_weekly(df_sym)
    
    # 转为RawBar
    bars = []
    for _, row in weekly.iterrows():
        bar = RawBar(
            symbol=str(row['symbol']),
            dt=pd.to_datetime(row['date']).to_pydatetime(),
            freq=Freq.D,  # 数据已是周线，用D即可
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            vol=float(row['volume']),
            amount=0.0
        )
        bars.append(bar)
    
    # CZSC提取笔
    bis = get_czsc_bis(bars)
    
    if len(bis) < 3:
        return {'trend': 'neutral', 'bis': bis, 'last_bi_direction': None, 'wave_structure': '笔不足'}
    
    # 当前价格（最后一根周K线收盘价）
    current_price = bars[-1].close if bars else 0
    
    # 最后一个笔分析
    last_bi = bis[-1]
    last_direction = 'up' if last_bi.direction.value == '向上' else 'down'
    bounce_pct = 0  # 默认值
    
    # 判断趋势
    # 规则1：如果最后一笔向上，整体高点逐步抬高 → 上升趋势
    # 规则2：如果最后一笔向下，但之后价格大幅反弹 → 可能转向上
    recent_directions = [1 if bi.direction.value == '向上' else -1 for bi in bis[-5:]]
    up_count = sum(1 for d in recent_directions if d == 1)
    
    # 简化判断：看最后一笔的方向
    if last_bi.direction.value == '向上':
        trend = 'up'
        wave_structure = f'上升中({len(bis)}笔)'
    elif last_bi.direction.value == '向下':
        # 最后一笔向下，看是否有反弹
        bounce_pct = (current_price - last_bi.fx_b.fx) / last_bi.fx_b.fx
        if bounce_pct > 0.05:  # 反弹超过5%
            trend = 'up'
            wave_structure = f'反弹中({len(bis)}笔, 已弹{bounce_pct*100:.0f}%)'
        elif bounce_pct > 0:
            trend = 'neutral'  # 小幅反弹
            wave_structure = f'调整中({len(bis)}笔)'
        else:
            trend = 'down'
            wave_structure = f'下降中({len(bis)}笔)'
    else:
        trend = 'neutral'
        wave_structure = f'震荡中({len(bis)}笔)'
    
    return {
        'trend': trend,
        'bis': bis,
        'last_bi_direction': last_direction,
        'wave_structure': wave_structure,
        'current_price': current_price,
        'last_bi_end_price': last_bi.fx_b.fx,
        'bounce_pct': bounce_pct if last_bi.direction.value == '向下' else 0,
        'weekly_data': weekly
    }


def calc_fib_retracement(start: float, end: float, level: float) -> float:
    """
    计算斐波那契回调位
    
    Args:
        start: 起点价格
        end: 终点价格
        level: 回调比例 (0.382, 0.500, 0.618, 0.786)
    
    Returns:
        回调位价格
    """
    diff = end - start
    if diff > 0:
        # 上涨后的回调
        return end - diff * level
    else:
        # 下跌后的反弹
        return end - diff * level


def calc_fib_extension(start: float, end: float, level: float) -> float:
    """
    计算斐波那契扩展位
    
    Args:
        start: 起点价格
        end: 终点价格  
        level: 扩展比例 (1.000, 1.618, 2.618)
    
    Returns:
        扩展位价格
    """
    diff = end - start
    return end + diff * level


def get_fib_levels(wave_low: float, wave_high: float) -> Dict:
    """
    获取斐波那契关键位
    
    Args:
        wave_low: 浪低点
        wave_high: 浪高点
    
    Returns:
        回调位和目标位字典
    """
    diff = wave_high - wave_low
    
    return {
        'retracement_382': wave_high - diff * 0.382,
        'retracement_500': wave_high - diff * 0.500,
        'retracement_618': wave_high - diff * 0.618,
        'retracement_786': wave_high - diff * 0.786,
        'extension_100': wave_high + diff * 1.000,
        'extension_1618': wave_high + diff * 1.618,
        'wave_length': diff
    }


def find_buy_signals(df_daily: pd.DataFrame, symbol: str, weekly_trend: Dict) -> List[Dict]:
    """
    寻找日线买入信号
    
    条件:
    1. 周线趋势向上
    2. 价格回调到斐波那契位
    3. 出现底分型
    """
    if weekly_trend['trend'] != 'up':
        return []
    
    df_sym = df_daily[df_daily['symbol'] == symbol].sort_values('date').tail(60).copy()
    if len(df_sym) < 20:
        return []
    
    signals = []
    
    # 获取最近的高点和低点
    recent_high = df_sym['high'].max()
    recent_low = df_sym['low'].min()
    
    # 计算Fibonacci位
    fib_levels = get_fib_levels(recent_low, recent_high)
    
    # 简单底分型识别
    for i in range(1, len(df_sym) - 1):
        row_prev = df_sym.iloc[i-1]
        row_curr = df_sym.iloc[i]
        row_next = df_sym.iloc[i+1]
        
        # 底分型: 中间K线低点最低
        if (row_curr['low'] < row_prev['low'] and 
            row_curr['low'] < row_next['low']):
            
            price = row_curr['low']
            
            # 检查是否在Fibonacci位附近
            near_382 = abs(price - fib_levels['retracement_382']) / price < 0.02
            near_500 = abs(price - fib_levels['retracement_500']) / price < 0.02
            near_618 = abs(price - fib_levels['retracement_618']) / price < 0.02
            
            if near_618:
                signals.append({
                    'date': row_curr['date'],
                    'price': price,
                    'type': 'buy',
                    'fib_level': '0.618',
                    'confidence': 0.8
                })
            elif near_500:
                signals.append({
                    'date': row_curr['date'],
                    'price': price,
                    'type': 'buy',
                    'fib_level': '0.500',
                    'confidence': 0.6
                })
            elif near_382:
                signals.append({
                    'date': row_curr['date'],
                    'price': price,
                    'type': 'buy',
                    'fib_level': '0.382',
                    'confidence': 0.4
                })
    
    return signals


def analyze_stock(df_daily: pd.DataFrame, symbol: str) -> Dict:
    """
    综合分析一只股票
    
    Returns:
        {
            'symbol': str,
            'weekly_trend': str,  # up/down/neutral
            'fib_levels': {...},
            'signals': [...],
            'summary': str
        }
    """
    # 周线趋势
    weekly = analyze_weekly_trend(df_daily, symbol)
    
    result = {
        'symbol': symbol,
        'weekly_trend': weekly['trend'],
        'wave_structure': weekly['wave_structure'],
        'fib_levels': {},
        'signals': [],
        'summary': ''
    }
    
    if weekly['trend'] == 'neutral' or len(weekly.get('bis', [])) < 3:
        result['summary'] = '趋势不明'
        return result
    
    # 获取斐波那契位
    bis = weekly['bis']
    if len(bis) >= 2:
        last_bi = bis[-1]
        prev_bi = bis[-2]
        
        if last_bi.direction.value == '向下':
            # 向下笔: 从高点跌到低点
            # 反弹位从低到高计算
            wave_low = last_bi.fx_b.fx  # 终点(低点) = 11.02
            wave_high = last_bi.fx_a.fx  # 起点(高点) = 13.89
        else:
            wave_low = last_bi.fx_a.fx
            wave_high = last_bi.fx_b.fx
        
        result['fib_levels'] = get_fib_levels(wave_low, wave_high)
    
    # 寻找买入信号
    result['signals'] = find_buy_signals(df_daily, symbol, weekly)
    
    # 总结
    if weekly['trend'] == 'up':
        if result['signals']:
            result['summary'] = f"周线上升, 回调买入信号"
        else:
            result['summary'] = f"周线上升, 无买入信号"
    elif weekly['trend'] == 'down':
        result['summary'] = "周线下降, 观望"
    else:
        result['summary'] = "趋势不明"
    
    return result


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    # 加载数据
    db_2025 = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/daily_data_year=2025/data.parquet')
    db_2026 = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/daily_data_year=2026/data.parquet')
    df = pd.concat([db_2025, db_2026])
    
    # 测试600985
    result = analyze_stock(df, '600985')
    print(f"=== 600985 分析结果 ===")
    print(f"周线趋势: {result['weekly_trend']}")
    print(f"波浪结构: {result['wave_structure']}")
    print(f"总结: {result['summary']}")
    
    if result['fib_levels']:
        print(f"\n斐波那契位:")
        for k, v in result['fib_levels'].items():
            if k != 'wave_length':
                print(f"  {k}: {v:.2f}")
    
    if result['signals']:
        print(f"\n买入信号: {len(result['signals'])}个")
        for s in result['signals'][:3]:
            print(f"  {s['date']} @{s['price']:.2f} ({s['fib_level']})")
