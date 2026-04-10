"""
波浪图 v6 - mplfinance 版本
用 mplfinance 绘制 K 线底层，叠加波浪线

【核心逻辑】
1. K线底层用 mplfinance 的 candle 图表
2. 波浪线用 addplot 叠加（实线）
3. 中文标签用 text annotation
4. 中国习惯颜色：红涨绿跌（上涨=red, 下跌=green）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
from czsc import CZSC, RawBar, Freq, Direction
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_daily_bars(symbol: str, year: int) -> List[RawBar]:
    """加载指定年份的日线数据"""
    path = f"/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/data.parquet"
    df = pd.read_parquet(path)
    df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    
    bars = []
    for _, row in df.iterrows():
        bar = RawBar(
            symbol=str(row['symbol']),
            dt=pd.Timestamp(row['date']).to_pydatetime(),
            freq=Freq.D,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            vol=float(row.get('volume', 0)),
            amount=float(row.get('amount', 0)),
        )
        bars.append(bar)
    return bars


def load_q4_bars(symbol: str, year: int) -> List[RawBar]:
    """加载前一年Q4数据（10月-12月）"""
    q4_path = f"/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/data.parquet"
    df = pd.read_parquet(q4_path)
    df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    
    df['month'] = pd.to_datetime(df['date']).dt.month
    df = df[df['month'] >= 10]
    
    bars = []
    for _, row in df.iterrows():
        bar = RawBar(
            symbol=str(row['symbol']),
            dt=pd.Timestamp(row['date']).to_pydatetime(),
            freq=Freq.D,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            vol=float(row.get('volume', 0)),
            amount=float(row.get('amount', 0)),
        )
        bars.append(bar)
    return bars


def recognize_bi(bars: List[RawBar]) -> pd.DataFrame:
    """识别笔"""
    c = CZSC(bars, max_bi_num=500)
    
    records = []
    for i, bi in enumerate(c.finished_bis):
        records.append({
            'bi_index': i,
            'start_dt': pd.Timestamp(bi.fx_a.dt),
            'end_dt': pd.Timestamp(bi.fx_b.dt),
            'start_price': bi.fx_a.fx,
            'end_price': bi.fx_b.fx,
            'direction': 'up' if bi.direction == Direction.Up else 'down',
        })
    
    return pd.DataFrame(records)


def identify_extrema(bi_df: pd.DataFrame, min_amplitude_pct: float = 8.0) -> pd.DataFrame:
    """识别所有有效的局部极值点"""
    if bi_df.empty or len(bi_df) < 3:
        return pd.DataFrame()
    
    reversals = []
    for i in range(1, len(bi_df)):
        prev_dir = bi_df.iloc[i - 1]['direction']
        curr_dir = bi_df.iloc[i]['direction']
        
        if curr_dir != prev_dir:
            extrema_type = 'high' if curr_dir == 'up' else 'low'
            reversals.append({
                'bi_index': bi_df.iloc[i]['bi_index'],
                'date': bi_df.iloc[i]['end_dt'],
                'price': bi_df.iloc[i]['end_price'],
                'direction': curr_dir,
                'type': extrema_type,
            })
    
    if not reversals:
        return pd.DataFrame()
    
    filtered = []
    for rev in reversals:
        if not filtered:
            filtered.append(rev)
            continue
        
        prev = filtered[-1]
        amp = abs(rev['price'] - prev['price']) / prev['price'] * 100
        
        if rev['type'] == prev['type']:
            filtered[-1] = rev
        elif amp >= min_amplitude_pct:
            filtered.append(rev)
    
    print(f"  Filtered {len(reversals)} reversals -> {len(filtered)} significant extrema")
    return pd.DataFrame(filtered)


def find_actual_turning_point(
    q4_extrema: pd.DataFrame, 
    year_extrema: pd.DataFrame
) -> Tuple[int, Optional[Dict], str]:
    """
    找年初真正的转折点。
    
    真正的反转点需要满足：
    - 方向改变（从 down→up 或 up→down）
    - 价格突破前高/前低（创出更高的高点或更低的低点）
    
    对于 2020 年：
    - Q4 方向=down，Feb20=up 但价格 13.84 < Mar4 高点（假突破）
    - Mar18=down 继续创出新低 10.03
    - Apr29=up 且价格 12.44 > Mar4 高点 13.76 → 真突破！
    """
    if q4_extrema.empty or year_extrema.empty:
        return 0, None, "无Q4数据或年内数据"
    
    q4_last = q4_extrema.iloc[-1].to_dict()
    q4_dir = q4_last['direction']
    q4_last_price = q4_last['price']
    
    year_first = year_extrema.iloc[0].to_dict()
    year_first_dir = year_first['direction']
    
    print(f"\n  Q4 last: {q4_last['date'].strftime('%Y-%m-%d')} {q4_last['price']:.2f} [{q4_last['type']}] dir={q4_dir}")
    print(f"  Year first: {year_first['date'].strftime('%Y-%m-%d')} {year_first['price']:.2f} [{year_first['type']}] dir={year_first_dir}")
    
    if q4_dir != year_first_dir:
        turning_type = year_first['type']
        wave_label = '1' if turning_type == 'low' else 'a'
        return 0, year_first, f"反转({q4_dir}→{year_first_dir})，转折点=Wave{wave_label}起点"
    
    # 方向一致：需要找真正的反转点
    # 使用 v5 逻辑：第一个方向改变的极值点就是转折点
    reversal_dir = 'up' if q4_dir == 'down' else 'down'
    print(f"  Direction continues ({q4_dir}) → scanning for reversal ({reversal_dir})...")
    
    for i, row in year_extrema.iterrows():
        if row['direction'] == reversal_dir:
            actual_turning = row.to_dict()
            wave_label = '1' if row['type'] == 'low' else 'a'
            return i, actual_turning, f"继续({q4_dir})，转折点=Wave{wave_label}起点"
    
    print(f"  WARNING: No reversal found, using year first")
    return 0, year_first, f"未找到反转点，使用year first"


def connect_waves(extrema_df: pd.DataFrame, start_from: int = 0) -> List[Dict]:
    """连接相邻极值点"""
    if extrema_df.empty or len(extrema_df) < start_from + 2:
        return []
    
    waves = []
    for i in range(start_from, len(extrema_df) - 1):
        curr = extrema_df.iloc[i]
        next_ = extrema_df.iloc[i + 1]
        
        waves.append({
            'start_idx': i,
            'end_idx': i + 1,
            'start_date': curr['date'],
            'end_date': next_['date'],
            'start_price': curr['price'],
            'end_price': next_['price'],
            'start_type': curr['type'],
            'end_type': next_['type'],
        })
    
    return waves


def label_waves(waves: List[Dict], start_label: str = '1') -> List[Dict]:
    """标注波浪编号 1-5-a-b-c"""
    if not waves:
        return []
    
    WAVE_SEQ = ['1', '2', '3', '4', '5', 'a', 'b', 'c']
    
    try:
        start_pos = WAVE_SEQ.index(start_label)
    except ValueError:
        start_pos = 0
    
    labeled = []
    for i, wave in enumerate(waves):
        labeled_wave = wave.copy()
        label = WAVE_SEQ[(start_pos + i) % 8]
        role = 'impulse' if label in ('1', '3', '5') else 'correction'
        labeled_wave['label'] = label
        labeled_wave['role'] = role
        labeled.append(labeled_wave)
    
    return labeled


def build_ohlc_df(bars: List[RawBar]) -> pd.DataFrame:
    """把 RawBar 转成 mplfinance 需要的 DataFrame"""
    records = []
    for bar in bars:
        records.append({
            'Date': pd.Timestamp(bar.dt),
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.vol,
        })
    df = pd.DataFrame(records)
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)
    return df


def draw_wave_chart_mplfinance(
    ohlc_df: pd.DataFrame,
    bi_df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    waves: List[Dict],
    symbol: str,
    year: int,
    continuation_end_idx: int = 0,
) -> str:
    """
    用 mplfinance 绘制波浪图
    
    - K线底层用 mplfinance candle
    - 波浪线用 addplot 叠加（实线）
    - 中文标签用 text annotation
    - 中国习惯：红涨绿跌
    """
    
    # ===== 中国习惯配色 =====
    # 上涨=red, 下跌=green（中国习惯）
    mc = mpf.make_marketcolors(
        up='#E74C3C',       # 红色涨
        down='#27AE60',     # 绿色跌
        edge='inherit',
        wick='inherit',
        volume='inherit',
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        gridcolor='#CCCCCC',
        facecolor='white',
        figcolor='white',
        rc={'font.family': 'SimHei', 'axes.unicode_minus': False},
    )
    
    # ===== 准备波浪线的 addplot 数据 =====
    # 为每条波浪线创建一个 series，用 NaN 填充其他位置
    wave_plot_data = {}
    wave_colors = {}
    wave_labels = {}
    
    for i, wave in enumerate(waves):
        label = wave['label']
        role = wave['role']
        
        # 颜色：中国习惯
        if wave['end_price'] > wave['start_price']:
            # 上涨
            color = '#E74C3C'  # 红
        else:
            # 下跌
            color = '#27AE60'  # 绿
        
        # 创建只有这条波浪的数据序列
        wave_series = pd.Series(index=ohlc_df.index, dtype=float)
        start_date = wave['start_date']
        end_date = wave['end_date']
        
        # 线段中需要插值，用分段线段
        wave_series[start_date:end_date] = np.nan
        # 在线段两端放置价格
        if start_date in wave_series.index and end_date in wave_series.index:
            wave_series[start_date] = wave['start_price']
            wave_series[end_date] = wave['end_price']
        
        wave_plot_data[i] = wave_series
        wave_colors[i] = color
        wave_labels[i] = label
    
    # ===== 创建 addplot 列表 =====
    add_plots = []
    for i, wave in enumerate(waves):
        color = wave_colors[i]
        label = wave_labels[i]
        
        # 对于波浪线，用散点图+线图的组合
        # 但更简单的方式：用 vlines 画垂直线区域，或者用线段
        # mplfinance 的 addplot 支持 line 类型的 plot
        
        # 创建线段数据（只用起止两点）
        wave_series = wave_plot_data[i].copy()
        
        plot = mpf.make_addplot(
            wave_series,
            type='line',
            color=color,
            width=2.5,
            alpha=0.9,
        )
        add_plots.append(plot)
    
    # ===== 标注极值点（散点图）=====
    extrema_high_series = pd.Series(index=ohlc_df.index, dtype=float)
    extrema_low_series = pd.Series(index=ohlc_df.index, dtype=float)
    
    for _, ext in extrema_df.iterrows():
        date = ext['date']
        price = ext['price']
        if date in extrema_high_series.index:
            if ext['type'] == 'high':
                extrema_high_series[date] = price
            else:
                extrema_low_series[date] = price
    
    add_plots.append(mpf.make_addplot(extrema_high_series, type='scatter', marker='^', 
                                       color='#C0392B', markersize=80, alpha=0.9))
    add_plots.append(mpf.make_addplot(extrema_low_series, type='scatter', marker='v',
                                       color='#1E8449', markersize=80, alpha=0.9))
    
    # ===== 标注延续区间（背景色）=====
    # 在延续区间添加背景色
    vr_lines = []
    if continuation_end_idx > 1:
        for i in range(continuation_end_idx - 1):
            ext_i = extrema_df.iloc[i]
            ext_next = extrema_df.iloc[i + 1]
            vr_lines.append((ext_i['date'], ext_next['date'], '#EEEEEE'))
    
    # ===== 绘制 =====
    fig, axes = mpf.plot(
        ohlc_df,
        type='candle',
        style=style,
        title=f'{symbol} 艾略特波浪 {year}（推动浪 1-3-5 / 调整浪 a-b-c）',
        ylabel='价格',
        ylabel_lower='成交量',
        addplot=add_plots,
        figsize=(24, 14),
        datetime_format='%Y-%m',
        xrotation=45,
        returnfig=True,
        show_nontrading=False,
    )
    
    ax = axes[0]  # 主图
    
    # ===== 添加中文波浪标签（text annotation）=====
    for i, wave in enumerate(waves):
        mid_dt = wave['start_date'] + (wave['end_date'] - wave['start_date']) / 2
        mid_p = (wave['start_price'] + wave['end_price']) / 2
        
    # 波浪编号放在线段中点旁边
        price_range = abs(wave['end_price'] - wave['start_price'])
        if wave['end_price'] > wave['start_price']:
            offset_y = price_range * 0.2
        else:
            offset_y = -price_range * 0.2
        
        wave_color = '#E74C3C' if wave['end_price'] > wave['start_price'] else '#27AE60'
        ax.annotate(
            wave['label'],
            xy=(mid_dt, mid_p),
            xytext=(0, offset_y),
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=wave_color,
                      edgecolor='white', alpha=0.95),
        )
    
    # ===== 添加极值点价格标签 =====
    for _, ext in extrema_df.iterrows():
        date = ext['date']
        price = ext['price']
        
        # 跳过延续区间的极值
        ext_idx = extrema_df[extrema_df['date'] == date].index[0]
        if ext_idx < continuation_end_idx:
            # 延续区间，用灰色
            color = '#999999'
            fontcolor = '#666666'
        else:
            if ext['type'] == 'high':
                color = '#C0392B'
                fontcolor = '#C0392B'
            else:
                color = '#1E8449'
                fontcolor = '#1E8449'
        
        ax.annotate(
            f'{price:.2f}',
            xy=(date, price),
            xytext=(0, 10 if ext['type'] == 'low' else -15),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            color=fontcolor,
            fontweight='bold',
        )
    
    # ===== 添加图例 =====
    up_waves = sum(1 for w in waves if w['end_price'] > w['start_price'])
    down_waves = len(waves) - up_waves
    
    legend_text = (
        f"红色=上涨(1/3/5) | 绿色=下跌(2/4/a/b/c) | "
        f"上涨浪={up_waves}个 | 下跌浪={down_waves}个 | 共{len(waves)}个浪"
    )
    ax.text(0.5, 1.02, legend_text, transform=ax.transAxes, 
            ha='center', fontsize=11, color='#333333')
    
    # 延续区间标注
    if continuation_end_idx > 1:
        cont_start = extrema_df.iloc[0]['date']
        cont_end = extrema_df.iloc[continuation_end_idx - 1]['date']
        mid_cont = cont_start + (cont_end - cont_start) / 2
        ax.text(mid_cont, ax.get_ylim()[1] * 0.99,
                f'Q4 延续（不编号）',
                ha='center', va='top', fontsize=10, color='gray',
                style='italic')
    
    # ===== 保存 =====
    output_path = f'/tmp/{symbol}_wave_{year}_mplfinance.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved: {output_path}")
    
    plt.close(fig)
    return output_path


def main():
    symbol = '000001'
    year = 2020
    prev_year = year - 1
    
    print(f"=== {symbol} {year} 波浪图 v6 (mplfinance) ===\n")
    
    # 1. 加载数据
    print("[1/7] Loading data...")
    bars = load_daily_bars(symbol, year)
    print(f"  Loaded {len(bars)} bars for {year}")
    
    q4_bars = load_q4_bars(symbol, prev_year)
    print(f"  Loaded {len(q4_bars)} bars for {prev_year} Q4")
    
    all_bars = q4_bars + bars
    
    # 2. 构建 OHLC DataFrame（mplfinance 用）
    print("\n[2/7] Building OHLC DataFrame...")
    ohlc_df = build_ohlc_df(bars)  # 只要当年的 K 线
    print(f"  OHLC shape: {ohlc_df.shape}")
    
    # 3. BI 识别
    print("\n[3/7] Recognizing BI...")
    bi_df = recognize_bi(all_bars)
    print(f"  Found {len(bi_df)} BIs")
    
    # 分离 Q4 和当年的 BI
    q4_bi = bi_df[bi_df['end_dt'].dt.year == prev_year].copy()
    year_bi = bi_df[bi_df['end_dt'].dt.year == year].copy().reset_index(drop=True)
    
    # 4. 极值识别
    print("\n[4/7] Identifying local extrema...")
    q4_extrema = identify_extrema(q4_bi, min_amplitude_pct=8.0)
    year_extrema = identify_extrema(year_bi, min_amplitude_pct=8.0)
    
    print(f"\n  Q4 {prev_year} extrema:")
    for idx, row in q4_extrema.iterrows():
        print(f"    [{idx}] {row['date'].strftime('%Y-%m-%d')} {row['price']:6.2f} [{row['type']}] dir={row['direction']}")
    
    print(f"\n  {year} extrema:")
    for idx, row in year_extrema.iterrows():
        print(f"    [{idx}] {row['date'].strftime('%Y-%m-%d')} {row['price']:6.2f} [{row['type']}]) dir={row['direction']}")
    
    # 5. 找真正的转折点
    print("\n[5/7] Finding true turning point...")
    turning_idx, turning_point, msg = find_actual_turning_point(q4_extrema, year_extrema)
    print(f"  Result: {msg}")
    
    if turning_point is None:
        print("  ERROR: No turning point found!")
        return
    
    print(f"  Turning point: {turning_point['date'].strftime('%Y-%m-%d')} {turning_point['price']:.2f} [{turning_point['type']}]")
    
    turning_year_idx = list(year_extrema.index).index(turning_idx)
    wave_start_extrema_idx = turning_year_idx
    
    # 6. 连线和标注
    print("\n[6/7] Connecting waves and labeling...")
    
    if turning_year_idx > 0:
        combined_extrema = pd.concat([
            q4_extrema.iloc[[-1]],
            year_extrema.iloc[turning_idx:]
        ], ignore_index=True)
        continuation_end_idx = 1
    else:
        combined_extrema = year_extrema.copy()
        continuation_end_idx = 0
    
    waves = connect_waves(combined_extrema, start_from=continuation_end_idx)
    
    turning_type = turning_point['type']
    start_label = '1' if turning_type == 'low' else 'a'
    waves = label_waves(waves, start_label=start_label)
    
    print(f"\n  Wave details ({len(waves)} waves):")
    for i, w in enumerate(waves):
        direction = "UP" if w['end_price'] > w['start_price'] else "DOWN"
        print(f"  {i+1:2d}: Wave {w['label']} [{w['role']:10s}] "
              f"{w['start_date'].strftime('%Y-%m-%d')}→{w['end_date'].strftime('%Y-%m-%d')} "
              f"{w['start_price']:.2f}→{w['end_price']:.2f} ({direction})")
    
    # 7. 绘图
    print("\n[7/7] Drawing chart with mplfinance...")
    
    # 只用当年的 BI 数据（用于绘制小级别笔）
    year_bi_df = bi_df[bi_df['end_dt'].dt.year == year].copy().reset_index(drop=True)
    
    output_path = draw_wave_chart_mplfinance(
        ohlc_df=ohlc_df,
        bi_df=year_bi_df,
        extrema_df=combined_extrema,
        waves=waves,
        symbol=symbol,
        year=year,
        continuation_end_idx=continuation_end_idx,
    )
    
    print(f"\n=== Complete ===")
    print(f"Output: {output_path}")
    return output_path


if __name__ == '__main__':
    main()
