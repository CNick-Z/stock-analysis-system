"""
波浪重划分核心模块 v2.0
基于CZSC笔分析 + 艾略特波浪规则

核心规则（2026-04-07 与老板确认）：
1. 连接规则：高→低→高→低交替，使用最近极端点
2. 笔最小定义：5根不完全包裹的K线（缠论规则， CZSC已实现）
3. 下跌 = a-b-c，上涨 = 1-2-3-4-5
4. 重划分：2跌破1起点 → 前面是C浪延伸，不是1-2
5. 1买 = C浪终点，2买 = 1浪回调终点，3买 = 3浪回调终点
6. 周线C浪判断：不需要区分a/c/2/4，只等"终结反转"买入机会
"""

from czsc import RawBar, Freq, CZSC, Direction
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import time


class WaveDir(Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class BiPoint:
    """笔的一个端点"""
    dt: Any
    price: float
    role: str  # 'H' or 'L'


@dataclass
class WaveSegment:
    """波浪段"""
    start_dt: Any
    end_dt: Any
    start_price: float
    end_price: float
    direction: str  # 'up' or 'down'
    wave_label: str = ""
    wave_degree: str = "primary"
    
    @property
    def length_pct(self) -> float:
        if self.direction == 'up':
            return (self.end_price - self.start_price) / self.start_price * 100
        else:
            return (self.start_price - self.end_price) / self.start_price * 100


@dataclass
class DailySignal:
    """每日信号"""
    date: Any
    symbol: str
    weekly_dir: WaveDir
    daily_dir: WaveDir  # 基于最近2笔的方向
    wave_label: str  # 当前所在浪的编号
    buy_type: Optional[str] = None  # 'BUY1', 'BUY2', 'BUY3'
    buy_price: float = 0.0
    stop_loss: float = 0.0
    last_low: float = 0.0  # 用于止损判断
    last_high: float = 0.0


@dataclass
class StockWaveResult:
    """完整股票波浪分析结果"""
    symbol: str
    weekly_dir: WaveDir
    weekly_bars: int
    daily_signals: List[DailySignal]
    meta: Dict


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """日线聚合成周线"""
    df = daily_df.copy()
    df['week'] = df['date'].dt.to_period('W')
    weekly = df.groupby('week').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    weekly['date'] = weekly['week'].apply(lambda x: x.start_time)
    return weekly


def get_weekly_direction(weekly_bars: List[RawBar]) -> WaveDir:
    """用CZSC周线笔判断周线方向"""
    if len(weekly_bars) < 10:
        return WaveDir.NEUTRAL
    
    try:
        c = CZSC(weekly_bars, max_bi_num=100)
        bis = c.finished_bis
        if len(bis) < 2:
            return WaveDir.NEUTRAL
        
        # 看最近2-3笔的方向
        recent = bis[-3:]
        up_count = sum(1 for b in recent if b.direction == Direction.Up)
        down_count = sum(1 for b in recent if b.direction == Direction.Down)
        
        if up_count > down_count:
            return WaveDir.UP
        elif down_count > up_count:
            return WaveDir.DOWN
        return WaveDir.NEUTRAL
    except:
        return WaveDir.NEUTRAL


def build_wave_segments_from_bis(bis: List) -> List[WaveSegment]:
    """
    从CZSC笔列表构建波浪段
    
    CZSC笔已经过缠论笔规则验证（至少5根不完全包裹K线）
    每笔有: direction(向上/向下), sdt, edt, fx_a(FX), fx_b(FX)
    FX对象有: mark(D/G), low(最低价), high(最高价)
    """
    segments = []
    for bi in bis:
        # CZSC direction 是中文：'向上' 或 '向下'
        if bi.direction == Direction.Up:
            # 向上笔: fx_a是底分型(D), fx_b是顶分型(G)
            start_price = bi.fx_a.low   # 笔开始的价格（低点）
            end_price = bi.fx_b.high   # 笔结束的价格（高点）
        else:  # '向下'
            # 向下笔: fx_a是顶分型(G), fx_b是底分型(D)
            start_price = bi.fx_a.high  # 笔开始的价格（高点）
            end_price = bi.fx_b.low    # 笔结束的价格（低点）
        
        segments.append(WaveSegment(
            start_dt=bi.sdt,
            end_dt=bi.edt,
            start_price=float(start_price),
            end_price=float(end_price),
            direction='up' if bi.direction == Direction.Up else 'down',
        ))
    
    return segments


def label_impulse_waves(segments: List[WaveSegment], 
                         trend: WaveDir) -> List[WaveSegment]:
    """
    给波浪段标注艾略特波浪编号
    
    规则（2026-04-07确认）：
    - 下降趋势(整体创新低)：推动浪 = a/c，调整 = b
    - 上升趋势(整体创新高)：推动浪 = 1/3/5，调整 = 2/4
    """
    if not segments:
        return segments
    
    if trend == WaveDir.DOWN:
        # 下降趋势：按a-b-c标注
        # 假设 segments 已经按时间排列
        # 推动段(下降) = a或c，调整段(上升) = b
        label_map = {0: 'a', 1: 'b', 2: 'c'}
        for i, seg in enumerate(segments):
            if seg.direction == 'down':
                seg.wave_label = label_map.get(i % 3, f'c{(i//3)+1}')
            else:
                seg.wave_label = 'b'
    else:
        # 上升趋势：按1-2-3-4-5标注
        wave_nums = ['1', '2', '3', '4', '5']
        for i, seg in enumerate(segments):
            seg.wave_label = wave_nums[i % 5]
    
    return segments


def detect_buy_from_segments(segments: List[WaveSegment],
                              trend: WaveDir) -> Tuple[Optional[str], float, float]:
    """
    从波浪段序列中检测买入信号
    
    1买 = C浪终点（下降趋势最后一波的终点）
    2买 = 1浪终点，回调不破起点
    3买 = 3浪终点，回调不破1浪高点
    
    Returns: (buy_type, buy_price, stop_loss) or (None, 0, 0)
    """
    if not segments:
        return None, 0.0, 0.0
    
    if trend == WaveDir.DOWN:
        # 下降趋势：找C浪终点（最后一波下降的终点）
        # C浪 = 连续下降段的最后一波
        # 从后往前找连续下降段
        for i in range(len(segments) - 1, -1, -1):
            if segments[i].direction == 'down':
                # 找到一个下降段，它的终点就是C浪终点 = 1买
                return ('BUY1', segments[i].end_price, segments[i].end_price)
        return None, 0.0, 0.0
    
    else:
        # 上升趋势：找2买和3买
        # 2买：回调段(2)不破1浪起点
        # 3买：回调段(4)不破1浪高点
        pass  # 待实现
    
    return None, 0.0, 0.0


def analyze_stock(symbol: str,
                 daily_df: pd.DataFrame,
                 min_bi_num: int = 200) -> StockWaveResult:
    """
    完整的单只股票波浪分析（周线+日线）
    
    Args:
        symbol: 股票代码
        daily_df: 日线数据
        min_bi_num: 最大笔数量（控制分析长度）
    
    Returns:
        StockWaveResult
    """
    df = daily_df.sort_values('date').reset_index(drop=True)
    
    if len(df) < 30:
        return StockWaveResult(
            symbol=symbol,
            weekly_dir=WaveDir.NEUTRAL,
            weekly_bars=0,
            daily_signals=[],
            meta={'error': '数据不足'}
        )
    
    # === 构建日线RawBar ===
    daily_bars = []
    for _, r in df.iterrows():
        daily_bars.append(RawBar(
            symbol=symbol,
            dt=r['date'].to_pydatetime() if isinstance(r['date'], pd.Timestamp) else r['date'],
            freq=Freq.D,
            open=float(r['open']), high=float(r['high']),
            low=float(r['low']), close=float(r['close']),
            vol=float(r['volume']), amount=0.0
        ))
    
    # === 周线分析 ===
    weekly_df = aggregate_to_weekly(df)
    weekly_bars = []
    for _, r in weekly_df.iterrows():
        weekly_bars.append(RawBar(
            symbol=symbol,
            dt=r['date'].to_pydatetime(),
            freq=Freq.W,
            open=float(r['open']), high=float(r['high']),
            low=float(r['low']), close=float(r['close']),
            vol=float(r['volume']), amount=0.0
        ))
    
    weekly_dir = get_weekly_direction(weekly_bars)
    
    # === 日线CZSC分析 ===
    c = CZSC(daily_bars, max_bi_num=min_bi_num)
    bis = c.finished_bis
    fx_list = c.fx_list
    
    # === 构建波浪段 ===
    segments = build_wave_segments_from_bis(bis)
    
    # === 标注波浪 ===
    # 判断整体趋势：用第一笔和最后一笔的关系
    if len(bis) >= 2:
        first_bi = bis[0]
        last_bi = bis[-1]
        # 如果最后一笔的终点比第一笔起点低，整体是下降
        if last_bi.fx_b < first_bi.fx_a:
            overall_trend = WaveDir.DOWN
        else:
            overall_trend = WaveDir.UP
    else:
        overall_trend = WaveDir.NEUTRAL
    
    segments = label_impulse_waves(segments, overall_trend)
    
    # === 检测买入信号 ===
    buy_type, buy_price, stop_loss = detect_buy_from_segments(segments, overall_trend)
    
    # === 生成每日信号 ===
    signals = []
    df_dates = df['date'].values
    
    # 创建日期到segment的映射
    seg_by_end_date = {seg.end_dt: seg for seg in segments}
    
    # 找到当前最近的segment
    for i, row in df.iterrows():
        dt = row['date']
        if isinstance(dt, pd.Timestamp):
            dt_py = dt.to_pydatetime()
        else:
            dt_py = dt
        
        # 找到包含当前日期的segment
        current_seg = None
        for seg in segments:
            if seg.start_dt <= dt_py <= seg.end_dt:
                current_seg = seg
                break
        
        # 判断日线方向：最近2笔的方向
        if len(bis) >= 2:
            recent_2 = bis[-2:]
            if all(b.direction == Direction.Up for b in recent_2):
                daily_dir = WaveDir.UP
            elif all(b.direction == Direction.Down for b in recent_2):
                daily_dir = WaveDir.DOWN
            else:
                daily_dir = WaveDir.NEUTRAL
        else:
            daily_dir = WaveDir.NEUTRAL
        
        signal = DailySignal(
            date=dt,
            symbol=symbol,
            weekly_dir=weekly_dir,
            daily_dir=daily_dir,
            wave_label=current_seg.wave_label if current_seg else '',
            last_low=float(row['low']),
            last_high=float(row['high']),
        )
        
        signals.append(signal)
    
    return StockWaveResult(
        symbol=symbol,
        weekly_dir=weekly_dir,
        weekly_bars=len(weekly_bars),
        daily_signals=signals,
        meta={
            'daily_bars': len(df),
            'bis_count': len(bis),
            'segments_count': len(segments),
            'overall_trend': overall_trend.value,
            'buy_type': buy_type,
            'buy_price': buy_price,
            'stop_loss': stop_loss,
        }
    )


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    print("=== 波浪分析模块 v2.0 测试 ===")
    
    # 测试300391
    sym = '300391'
    df = pd.read_parquet('/data/warehouse/wavechan/wavechan_cache/l1_cold_year=2024/data.parquet',
                         filters=[('symbol', '==', sym)])
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n测试: {sym}, {len(df)} days")
    result = analyze_stock(sym, df)
    
    print(f"周线方向: {result.weekly_dir.value}")
    print(f"日线笔数: {result.meta['bis_count']}")
    print(f"整体趋势: {result.meta['overall_trend']}")
    print(f"买入信号: {result.meta['buy_type']} @ {result.meta['buy_price']:.2f}")
    print(f"止损: {result.meta['stop_loss']:.2f}")
    print(f"\n波浪段:")
    segments = []
    for i, row in df.iterrows():
        bars = [RawBar(symbol=sym, dt=r['date'].to_pydatetime(), freq=Freq.D,
                       open=float(r['open']), high=float(r['high']),
                       low=float(r['low']), close=float(r['close']),
                       vol=float(r['volume']), amount=0.0)
                for _, r in df.iterrows()]
        c = CZSC(bars, max_bi_num=200)
        for bi in c.finished_bis:
            print(f"  {bi.direction} {bi.sdt.date()}~{bi.edt.date()} {bi.fx_a:.2f}->{bi.fx_b:.2f}")
        break
