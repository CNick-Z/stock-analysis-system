"""
波浪画法修正 v4 - 2026-04-10

【最终确认规则】

1. 连线（铁律）：
   - 高转折点 → 下一个低转折点 → 下一个高转折点 → ...
   - 永远交替，高只连低，低只连高

2. 标注 1-5-a-b-c：
   - 1, 3, 5 = 推动浪（和主趋势同向）
   - 2, 4, a, b, c = 调整浪（和主趋势反向）

3. 识别局部极值：
   - 每个显著的高点/低点极值都要标注
   - 当前后有相反方向的极值 → 有效

4. 年初年末：自然连续，无特殊处理

5. 两层图：
   - 虚线：小级别笔（CZSC）
   - 实线：大级别波浪（局部高低点交替连接，标注1-5-a-b-c）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from czsc import CZSC, RawBar, Freq, Direction
from typing import List, Dict

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_daily_bars(symbol: str, year: int) -> List[RawBar]:
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
    
    规则：当前笔终点相对于前后笔，如果是一个局部极值（有反向的高点/低点）→ 标注
    
    算法：
    1. 每个笔终点都是候选
    2. 方向反转点 = 有效极值
    3. 只保留幅度超过阈值的极值
    4. 如果相邻极值同类型（同是high或同是low），保留更大的，删除较小的
    """
    if bi_df.empty or len(bi_df) < 3:
        return pd.DataFrame()
    
    # 收集所有方向反转点
    reversals = []
    for i in range(1, len(bi_df)):
        prev_dir = bi_df.iloc[i - 1]['direction']
        curr_dir = bi_df.iloc[i]['direction']
        
        if curr_dir != prev_dir:
            # UP笔终点 = 局部高点
            # DOWN笔终点 = 局部低点
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
    
    # 过滤：只保留显著的反转
    filtered = []
    for rev in reversals:
        if not filtered:
            filtered.append(rev)
            continue
        
        prev = filtered[-1]
        amp = abs(rev['price'] - prev['price']) / prev['price'] * 100
        
        if rev['type'] == prev['type']:
            # 同类型（同是high或同是low）：保留价格变化更大的那个
            # 替换前一个
            filtered[-1] = rev
        elif amp >= min_amplitude_pct:
            # 不同类型且幅度足够
            filtered.append(rev)
        # else: 不同类型但幅度不够，跳过
    
    print(f"  Filtered {len(reversals)} reversals -> {len(filtered)} significant extrema")
    return pd.DataFrame(filtered)


def connect_waves(extrema_df: pd.DataFrame) -> List[Dict]:
    """
    连接相邻极值点。
    
    铁律：高只连低，低只连高，永远交替。
    """
    if extrema_df.empty or len(extrema_df) < 2:
        return []
    
    waves = []
    for i in range(len(extrema_df) - 1):
        curr = extrema_df.iloc[i]
        next_ = extrema_df.iloc[i + 1]
        
        # 永远是交替连接
        wave_type = 'correction'  # 默认调整
        
        waves.append({
            'start_idx': i,
            'end_idx': i + 1,
            'start_date': curr['date'],
            'end_date': next_['date'],
            'start_price': curr['price'],
            'end_price': next_['price'],
            'start_type': curr['type'],
            'end_type': next_['type'],
            'type': wave_type,
        })
    
    return waves


def label_waves(waves: List[Dict], main_trend: str = 'up') -> List[Dict]:
    """
    标注波浪编号 1-5-a-b-c。
    
    规则：
    - 1, 3, 5 = 推动浪（和主趋势同向）
    - 2, 4, a, b, c = 调整浪（和主趋势反向）
    - 推动浪：根据波浪方向（上升=推动，下降=调整）
    """
    if not waves:
        return []
    
    # 波浪序列
    WAVE_SEQ = ['1', '2', '3', '4', '5', 'a', 'b', 'c']
    
    labeled = []
    pos = 0  # 在WAVE_SEQ中的位置
    
    for i, wave in enumerate(waves):
        labeled_wave = wave.copy()
        
        label = WAVE_SEQ[pos % 8]
        
        # 判断是推动浪还是调整浪
        if label in ('1', '3', '5'):
            role = 'impulse'
        else:
            role = 'correction'
        
        labeled_wave['label'] = label
        labeled_wave['role'] = role
        labeled.append(labeled_wave)
        
        pos += 1
    
    return labeled


def determine_trend(waves: List[Dict]) -> str:
    """
    根据波浪的整体方向判断主趋势。
    如果上升浪多于下降浪，则主趋势向上。
    """
    up_count = 0
    down_count = 0
    
    for wave in waves:
        if wave['end_price'] > wave['start_price']:
            up_count += 1
        else:
            down_count += 1
    
    return 'up' if up_count > down_count else 'down'


def draw_chart(bi_df: pd.DataFrame, extrema_df: pd.DataFrame, 
               waves: List[Dict], symbol: str, year: int) -> str:
    """绘制波浪图：两层（虚线=笔，实线=波浪）"""
    
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # ========== Layer 1: 虚线 - 小级别笔（CZSC）==========
    for _, row in bi_df.iterrows():
        color = '#E74C3C' if row['direction'] == 'up' else '#3498DB'
        ax.plot([row['start_dt'], row['end_dt']], 
               [row['start_price'], row['end_price']], 
               color=color, linewidth=0.8, alpha=0.25, linestyle=':')
    
    # ========== Layer 2: 实线 - 大级别波浪（交替连接）==========
    for wave in waves:
        # 根据波浪方向着色
        if wave['end_price'] > wave['start_price']:
            color = '#E74C3C'  # 红色向上
            lw = 2.5
        else:
            color = '#3498DB'  # 蓝色向下
            lw = 2.5
        
        ax.plot([wave['start_date'], wave['end_date']],
               [wave['start_price'], wave['end_price']],
               color=color, linewidth=lw, linestyle='-', alpha=0.9)
        
        # 标注浪编号（在波浪中点上方）
        mid_dt = wave['start_date'] + (wave['end_date'] - wave['start_date']) / 2
        mid_p = (wave['start_price'] + wave['end_price']) / 2
        offset = 25 if wave['end_price'] > wave['start_price'] else -25
        
        ax.annotate(wave['label'], (mid_dt, mid_p),
                   textcoords="offset points", xytext=(0, offset),
                   ha='center', fontsize=12, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                            edgecolor='white', alpha=0.9))
    
    # ========== 标注极值点 ==========
    for _, ext in extrema_df.iterrows():
        if ext['type'] == 'high':
            color = '#E74C3C'
            marker = '^'
        else:
            color = '#3498DB'
            marker = 'v'
        
        ax.scatter(ext['date'], ext['price'], color=color, s=100, 
                  marker=marker, zorder=6, edgecolors='white', linewidths=1.5)
        
        # 标注价格
        ax.annotate(f"{ext['price']:.2f}", 
                   (ext['date'], ext['price']),
                   textcoords="offset points", xytext=(0, 8),
                   ha='center', fontsize=8, color='#333333',
                   fontweight='bold')
    
    # ========== 图表装饰 ==========
    # 统计
    up_waves = sum(1 for w in waves if w['end_price'] > w['start_price'])
    down_waves = len(waves) - up_waves
    
    ax.set_title(f'{symbol} Elliott Wave {year}\n'
                 f'Red=UP impulse waves | Blue=DOWN correction waves | '
                 f'Up={up_waves} Down={down_waves} | {len(waves)} waves total',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color='#E74C3C', linewidth=0.8, linestyle=':', alpha=0.5, label='UP pen (CZSC)'),
        plt.Line2D([0], [0], color='#3498DB', linewidth=0.8, linestyle=':', alpha=0.5, label='DOWN pen (CZSC)'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2.5, label='UP wave (1/3/5)'),
        plt.Line2D([0], [0], color='#3498DB', linewidth=2.5, label='DOWN wave (2/4/a/b/c)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C', markersize=10, label='Local High'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#3498DB', markersize=10, label='Local Low'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Y轴范围留点空间
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 0.95, y_max * 1.05)
    
    plt.tight_layout()
    
    output_path = f'/tmp/{symbol}_wave_{year}_v4.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")
    
    return output_path


def main():
    symbol = '000001'
    year = 2020
    
    print(f"=== {symbol} {year} Wave Analysis v4 ===")
    print(f"Rules: H→L→H→L alternating | 1,3,5=UP | 2,4,a,b,c=DOWN\n")
    
    # 1. 加载数据
    print("[1/5] Loading data...")
    bars = load_daily_bars(symbol, year)
    print(f"  Loaded {len(bars)} bars")
    
    # 2. BI识别
    print("\n[2/5] Recognizing BI...")
    bi_df = recognize_bi(bars)
    print(f"  Found {len(bi_df)} BIs")
    
    # 3. 极值识别
    print("\n[3/5] Identifying local extrema...")
    extrema_df = identify_extrema(bi_df, min_amplitude_pct=8.0)
    
    if extrema_df.empty:
        print("  ERROR: No extrema found!")
        return
    
    print("\n  Extrema list:")
    for i, row in extrema_df.iterrows():
        print(f"  {i:2d}: {row['date'].strftime('%Y-%m-%d')} {row['price']:6.2f} [{row['type']}]")
    
    # 4. 连线
    print("\n[4/5] Connecting waves (H→L→H→L alternating)...")
    waves = connect_waves(extrema_df)
    print(f"  Created {len(waves)} wave segments")
    
    # 5. 判断主趋势
    main_trend = determine_trend(waves)
    print(f"  Main trend: {main_trend.upper()} (more UP waves than DOWN)")
    
    # 6. 标注
    waves = label_waves(waves, main_trend)
    
    print("\n  Wave details:")
    for i, w in enumerate(waves):
        direction = "UP" if w['end_price'] > w['start_price'] else "DOWN"
        print(f"  {i+1:2d}: Wave {w['label']} [{w['role']:10s}] {w['start_type']}→{w['end_type']} "
              f"{w['start_date'].strftime('%Y-%m-%d')}→{w['end_date'].strftime('%Y-%m-%d')} "
              f"{w['start_price']:.2f}→{w['end_price']:.2f} ({direction})")
    
    # 7. 绘图
    print("\n[5/5] Drawing chart...")
    output_path = draw_chart(bi_df, extrema_df, waves, symbol, year)
    
    print(f"\n=== Complete ===")
    return output_path


if __name__ == '__main__':
    main()
