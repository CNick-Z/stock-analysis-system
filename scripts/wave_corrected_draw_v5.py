"""
波浪画法修正 v5 - 2026-04-10

【核心逻辑】年初趋势延续判断

问题：2020年初算法错误地把2月2日的临时反弹当新浪起点，
     实际上3月18日的10.03才是真正的转折点 = 1浪起点。

修正逻辑：
1. 判断年初趋势是否延续：
   - 加载前一年Q4数据，找到Q4最后一个极值点
   - 找到当年第一个极值点
   - 如果方向一致 → 继续扫描，找到真正的反转点（趋势改变的那个极值）
   - 如果方向反转 → 转折点算新浪起点

2. 转折点识别：
   - 向下→向上：第一个UP极值点是真正的1浪起点
   - 向上→向下：第一个DOWN极值点是真正的a浪起点

000001 2020年正确标注：
- 年初 → 3月18日低点10.03：延续2019年Q4下跌，不编号
- 3月18日 10.03 = 1浪起点（真正的反转点）
- 之后按1-2-3-4-5-a-b-c标注
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from czsc import CZSC, RawBar, Freq, Direction
from typing import List, Dict, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
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
    """加载前一年Q4数据（10月-12月），用于判断年初趋势是否延续"""
    q4_path = f"/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/data.parquet"
    df = pd.read_parquet(q4_path)
    df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    
    # 只取Q4（10月-12月）
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
    """
    识别所有有效的局部极值点。
    """
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
    
    核心逻辑：
    - Q4 last direction vs year first direction：
      - 反转 → year_first 本身就是转折点（1或a）
      - 一致 → 需要继续扫描，找到真正的反转点（趋势改变的那个极值）
    
    对于2020：
    - Q4 direction = down，year first direction = down（一致）
    - 扫描：Feb2=down(继续), Feb20=up(但不是反转), Mar1=down(继续), 
            Mar4=up(继续), Mar18=down(继续！创出新低)
    - 需要进一步检查Mar18之后是否有UP反转
    - Mar18=low，之后Apr29=high（从下转上！）→ 这就是真正的反转点
    
    Returns:
        (turning_idx, turning_point, message)
        turning_idx: 在year_extrema中的位置索引
        turning_point: 转折点信息
        message: 说明
    """
    if q4_extrema.empty or year_extrema.empty:
        return 0, None, "无Q4数据或年内数据"
    
    q4_last = q4_extrema.iloc[-1].to_dict()
    q4_dir = q4_last['direction']  # 'up' or 'down'
    
    year_first = year_extrema.iloc[0].to_dict()
    year_first_dir = year_first['direction']
    
    print(f"\n  Q4 last: {q4_last['date'].strftime('%Y-%m-%d')} {q4_last['price']:.2f} [{q4_last['type']}] dir={q4_dir}")
    print(f"  Year first: {year_first['date'].strftime('%Y-%m-%d')} {year_first['price']:.2f} [{year_first['type']}] dir={year_first_dir}")
    
    if q4_dir != year_first_dir:
        # 方向反转：year_first就是转折点
        turning_type = year_first['type']
        wave_label = '1' if turning_type == 'low' else 'a'
        return 0, year_first, f"反转({q4_dir}→{year_first_dir})，转折点=Wave{wave_label}起点"
    
    # 方向一致：需要扫描找到真正的反转点
    # 反转的意思是：当前极值之后，趋势改变了方向
    # 比如一直down，找到第一个UP极值（从down变up）
    # 或者一直up，找到第一个DOWN极值（从up变down）
    
    # 反转方向 = Q4方向的反方向
    reversal_dir = 'up' if q4_dir == 'down' else 'down'
    
    print(f"  Direction continues ({q4_dir}) → scanning for reversal ({reversal_dir})...")
    
    # 扫描找反转点
    for i, row in year_extrema.iterrows():
        if row['direction'] == reversal_dir:
            # 找到反转极值！但需要确认它真的是反转，不是短暂反弹
            # 检查：如果这个极值之后，趋势真的反了吗？
            # 即：下一个极值的方向又变回来了（说明这是反转）
            # 
            # 但我们没有"下一个"，因为我们只有反转点的信息
            # 
            # 更实用的判断：这个反转极值本身的幅度够大吗？
            # 如果是低点反转：价格应该比之前的所有低点都低（新低）？
            # 如果是高点反转：价格应该比之前的所有高点都高（新高）？
            #
            # 对于2020年：
            # - Mar18=low，方向down（不是反转，继续）
            # - Apr29=high，方向up（反转！）
            
            # 找到了！但需要验证：
            # 如果 reversal_dir='up'，那这个极值应该是高点
            # 如果 reversal_dir='down'，那这个极值应该是低点
            # 实际上，极值的方向就是它的类型（low=up方向的反转点，high=down方向的反转点）
            actual_turning = row.to_dict()
            
            # 额外验证：如果Q4和year_first同向（都是down），
            # 我们要找的"反转"应该比year_first更往同一方向走
            # 比如Feb2=low(继续down), Mar18=low(继续down创出新低)
            # 那么真正的反转是Mar18之后的第一个UP极值
            
            # 已经在正确的反转方向上
            print(f"  Found reversal candidate: {row['date'].strftime('%Y-%m-%d')} {row['price']:.2f} [{row['type']}] dir={row['direction']}")
            
            # 但等等！这个反转点本身的方向可能不对
            # 比如Q4=down, year_first=down（继续）
            # 我们扫描找direction=up的第一个极值
            # Mar18=low, direction=down → 跳过（不是up）
            # Apr29=high, direction=up → 这是反转点！
            
            return i, actual_turning, f"继续({q4_dir})，转折点=Wave{'1' if row['type']=='low' else 'a'}起点"
    
    # 没找到反转点（不应该发生）
    print(f"  WARNING: No reversal found, using year first")
    return 0, year_first, f"未找到反转点，使用year first"


def connect_waves(extrema_df: pd.DataFrame, start_from: int = 0) -> List[Dict]:
    """连接相邻极值点，从指定索引开始。"""
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
    """标注波浪编号 1-5-a-b-c。"""
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


def determine_trend(waves: List[Dict]) -> str:
    """根据波浪的整体方向判断主趋势"""
    up_count = sum(1 for w in waves if w['end_price'] > w['start_price'])
    down_count = len(waves) - up_count
    return 'up' if up_count > down_count else 'down'


def draw_chart(bi_df: pd.DataFrame, extrema_df: pd.DataFrame, 
               waves: List[Dict], symbol: str, year: int,
               continuation_end_idx: int = 0) -> str:
    """绘制波浪图"""
    
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Layer 1: 虚线 - 小级别笔（CZSC）
    for _, row in bi_df.iterrows():
        color = '#E74C3C' if row['direction'] == 'up' else '#3498DB'
        ax.plot([row['start_dt'], row['end_dt']], 
               [row['start_price'], row['end_price']], 
               color=color, linewidth=0.8, alpha=0.25, linestyle=':')
    
    # 标注延续区间（灰色背景）
    if continuation_end_idx > 0:
        for i in range(continuation_end_idx - 1):
            ext_i = extrema_df.iloc[i]
            ext_next = extrema_df.iloc[i + 1]
            ax.axvspan(ext_i['date'], ext_next['date'], 
                       alpha=0.08, color='gray')
        # 标注"延续"文字
        cont_start = extrema_df.iloc[0]['date']
        cont_end = extrema_df.iloc[continuation_end_idx - 1]['date']
        mid_cont = cont_start + (cont_end - cont_start) / 2
        ax.text(mid_cont, ax.get_ylim()[1] * 0.99, 
                f'Q4-{year} Continuation\n(no wave label)',
                ha='center', va='top', fontsize=10, color='gray',
                style='italic')
    
    # Layer 2: 实线 - 大级别波浪
    for wave in waves:
        color = '#E74C3C' if wave['end_price'] > wave['start_price'] else '#3498DB'
        lw = 2.5
        
        ax.plot([wave['start_date'], wave['end_date']],
               [wave['start_price'], wave['end_price']],
               color=color, linewidth=lw, linestyle='-', alpha=0.9)
        
        # 波浪编号放在线段中点旁边
        mid_dt = wave['start_date'] + (wave['end_date'] - wave['start_date']) / 2
        mid_p = (wave['start_price'] + wave['end_price']) / 2
        # 计算线段的水平偏移量，让标签偏离线段
        price_range = abs(wave['end_price'] - wave['start_price'])
        offset = price_range * 0.15 if wave['end_price'] > wave['start_price'] else -price_range * 0.15
        
        ax.annotate(wave['label'], (mid_dt, mid_p),
                   textcoords="offset points", xytext=(0, offset),
                   ha='center', fontsize=13, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                            edgecolor='white', alpha=0.9))
    
    # 标注极值点
    for i, (_, ext) in enumerate(extrema_df.iterrows()):
        if i < continuation_end_idx:
            color = '#BBBBBB'
            marker = '^' if ext['type'] == 'high' else 'v'
            alpha = 0.4
        else:
            color = '#E74C3C' if ext['type'] == 'high' else '#3498DB'
            marker = '^' if ext['type'] == 'high' else 'v'
            alpha = 0.9
        
        ax.scatter(ext['date'], ext['price'], color=color, s=100, 
                  marker=marker, zorder=6, edgecolors='white', linewidths=1.5, alpha=alpha)
        
        label_text = f"{ext['price']:.2f}"
        if i < continuation_end_idx:
            label_text += " (cont)"
        ax.annotate(label_text, 
                   (ext['date'], ext['price']),
                   textcoords="offset points", xytext=(0, 8),
                   ha='center', fontsize=8, color='#555555' if i < continuation_end_idx else '#333333',
                   fontweight='bold')
    
    # 图表装饰
    up_waves = sum(1 for w in waves if w['end_price'] > w['start_price'])
    down_waves = len(waves) - up_waves
    
    ax.set_title(f'{symbol} Elliott Wave {year} (v5 - Year Boundary + True Turning Point)\n'
                 f'Red=UP | Blue=DOWN | Up={up_waves} Down={down_waves} | {len(waves)} waves | '
                 f'Gray area = Q4 continuation (no label)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        plt.Line2D([0], [0], color='#E74C3C', linewidth=0.8, linestyle=':', alpha=0.4, label='UP pen (CZSC)'),
        plt.Line2D([0], [0], color='#3498DB', linewidth=0.8, linestyle=':', alpha=0.4, label='DOWN pen (CZSC)'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2.5, label='UP wave (1/3/5)'),
        plt.Line2D([0], [0], color='#3498DB', linewidth=2.5, label='DOWN wave (2/4/a/b/c)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C', markersize=10, label='Local High'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#3498DB', markersize=10, label='Local Low'),
        plt.Line2D([0], [0], color='#BBBBBB', linewidth=2, alpha=0.4, label='Continuation (no label)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 0.95, y_max * 1.05)
    
    plt.tight_layout()
    
    output_path = f'/tmp/{symbol}_wave_{year}_v5.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")
    
    return output_path


def main():
    symbol = '000001'
    year = 2020
    prev_year = year - 1
    
    print(f"=== {symbol} {year} Wave Analysis v5 ===")
    print(f"New Logic: Year boundary continuation + true turning point detection\n")
    
    # 1. 加载数据
    print("[1/6] Loading data...")
    bars = load_daily_bars(symbol, year)
    print(f"  Loaded {len(bars)} bars for {year}")
    
    q4_bars = load_q4_bars(symbol, prev_year)
    print(f"  Loaded {len(q4_bars)} bars for {prev_year} Q4")
    
    # 合并Q4+当年数据用于BI识别
    all_bars = q4_bars + bars
    
    # 2. BI识别
    print("\n[2/6] Recognizing BI...")
    bi_df = recognize_bi(all_bars)
    print(f"  Found {len(bi_df)} BIs")
    
    # 分离Q4和当年的BI
    q4_bi = bi_df[bi_df['end_dt'].dt.year == prev_year].copy()
    year_bi = bi_df[bi_df['end_dt'].dt.year == year].copy().reset_index(drop=True)
    
    # 3. 极值识别
    print("\n[3/6] Identifying local extrema...")
    q4_extrema = identify_extrema(q4_bi, min_amplitude_pct=8.0)
    year_extrema = identify_extrema(year_bi, min_amplitude_pct=8.0)
    
    print(f"\n  Q4 {prev_year} extrema:")
    for idx, row in q4_extrema.iterrows():
        print(f"    [{idx}] {row['date'].strftime('%Y-%m-%d')} {row['price']:6.2f} [{row['type']}] dir={row['direction']}")
    
    print(f"\n  {year} extrema:")
    for idx, row in year_extrema.iterrows():
        print(f"    [{idx}] {row['date'].strftime('%Y-%m-%d')} {row['price']:6.2f} [{row['type']}] dir={row['direction']}")
    
    # 4. 找真正的转折点
    print("\n[4/6] Finding true turning point...")
    turning_idx, turning_point, msg = find_actual_turning_point(q4_extrema, year_extrema)
    print(f"  Result: {msg}")
    
    if turning_point is None:
        print("  ERROR: No turning point found!")
        return
    
    print(f"  Turning point: {turning_point['date'].strftime('%Y-%m-%d')} {turning_point['price']:.2f} [{turning_point['type']}]")
    
    # 转折点之后的极值索引
    turning_year_idx = list(year_extrema.index).index(turning_idx)
    wave_start_extrema_idx = turning_year_idx
    
    print(f"  Wave labeling starts at year extrema position {wave_start_extrema_idx}")
    
    # 5. 连线和标注
    print("\n[5/6] Connecting waves and labeling...")
    
    # 从转折点开始构建波浪序列
    # 如果转折点不在第一个，需要加入Q4最后一个极值作为起点
    if turning_year_idx > 0:
        # Q4最后一个极值 + 转折点及之后的极值
        combined_extrema = pd.concat([
            q4_extrema.iloc[[-1]],
            year_extrema.iloc[turning_idx:]
        ], ignore_index=True)
        # 延续结束索引 = 1（第0个是Q4，第1个是转折点）
        continuation_end_idx = 1
        print(f"  Combined: Q4 last + year from pos {turning_idx}")
    else:
        combined_extrema = year_extrema.copy()
        continuation_end_idx = 0
        print(f"  Using year extrema from start")
    
    print(f"  Combined extrema count: {len(combined_extrema)}")
    print(f"  Continuation ends at idx: {continuation_end_idx}")
    
    waves = connect_waves(combined_extrema, start_from=continuation_end_idx)
    print(f"  Created {len(waves)} wave segments")
    
    # 确定起始标签
    turning_type = turning_point['type']
    start_label = '1' if turning_type == 'low' else 'a'
    print(f"  Starting label: Wave {start_label} (turning type={turning_type})")
    
    waves = label_waves(waves, start_label=start_label)
    main_trend = determine_trend(waves)
    print(f"  Main trend: {main_trend.upper()}")
    
    print("\n  Wave details:")
    for i, w in enumerate(waves):
        direction = "UP" if w['end_price'] > w['start_price'] else "DOWN"
        print(f"  {i+1:2d}: Wave {w['label']} [{w['role']:10s}] {w['start_type']}→{w['end_type']} "
              f"{w['start_date'].strftime('%Y-%m-%d')}→{w['end_date'].strftime('%Y-%m-%d')} "
              f"{w['start_price']:.2f}→{w['end_price']:.2f} ({direction})")
    
    # 6. 绘图
    print("\n[6/6] Drawing chart...")
    
    year_bi_df = bi_df[bi_df['end_dt'].dt.year == year].copy().reset_index(drop=True)
    output_path = draw_chart(year_bi_df, combined_extrema, waves, symbol, year,
                            continuation_end_idx=continuation_end_idx)
    
    print(f"\n=== Complete ===")
    return output_path


if __name__ == '__main__':
    main()
