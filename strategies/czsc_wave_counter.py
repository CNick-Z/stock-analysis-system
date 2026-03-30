"""
波浪计数器 v2 - 清晰修正逻辑

老板的核心教导：
1. 笔1-5 = W1上升（合并）
2. 笔6下跌没跌破W1起点 → W2确认！
3. 之后笔7向上突破W1高点 → W3确认！

关键规则：
- W2不能跌破W1起点
- W3/W5是驱动浪（向上突破前高）
- 需要等待确认：不是预判，是走出来之后确认
"""

import pandas as pd
from czsc import CZSC, RawBar, Freq
from typing import Tuple, List, Optional


class WaveCounter:
    """波浪计数器 - 带修正"""
    
    def __init__(self):
        self.bis = []
        self.w1_start = None   # W1起点
        self.w1_end = None     # W1终点
        self.w2_end = None     # W2终点  
        self.w3_end = None     # W3终点
        self.w4_end = None     # W4终点
        self.state = "initial"
    
    def feed(self, bi) -> str:
        """喂入新笔，返回当前状态"""
        self.bis.append(bi)
        self._update()
        return self.state
    
    def _update(self):
        """根据当前笔序列更新波浪"""
        if len(self.bis) < 3:
            self.state = "initial"
            return
        
        # 简化逻辑：从后往前分析
        last_bi = self.bis[-1]
        
        if self.state == "initial":
            # 等待形成W1
            if len(self.bis) >= 3:
                self._try_confirm_w1()
        
        elif self.state == "w1_formed":
            # W1已确认，等待W2确认
            self._try_confirm_w2()
        
        elif self.state == "w2_formed":
            # W2已确认，等待W3确认
            self._try_confirm_w3()
        
        elif self.state == "w3_formed":
            # W3已确认，等待W4
            if last_bi.direction.value == '向下':
                self.w4_end = last_bi.fx_b.fx
                self.state = "w4_formed"
        
        elif self.state == "w4_formed":
            # W4中，等待W5
            if last_bi.direction.value == '向上':
                self.state = "w5_in_progress"
    
    def _try_confirm_w1(self):
        """尝试确认W1"""
        # W1应该是一个向上笔序列后的向下笔
        # 检查最后一个向下笔是否跌破前面所有低点
        
        # 找最后一个向下笔
        last_down = None
        for i in range(len(self.bis) - 1, -1, -1):
            if self.bis[i].direction.value == '向下':
                last_down = self.bis[i]
                break
        
        if last_down is None:
            return
        
        # 找前面所有低点中的最低点作为W1起点
        w1_start = float('inf')
        for i in range(len(self.bis)):
            bi = self.bis[i]
            low = bi.fx_b.fx if bi.direction.value == '向下' else bi.fx_a.fx
            if low < w1_start:
                w1_start = low
        
        # 检查是否跌破W1起点
        if last_down.fx_b.fx < w1_start:
            # 跌破！重新设定W1起点
            w1_start = last_down.fx_a.fx  # 用这个向下笔的起点作为新的W1起点参考
        
        # W1终点是最后一个向上笔的高点（在最后一个向下笔之前）
        w1_end = 0
        for i in range(len(self.bis) - 1, -1, -1):
            if self.bis[i].direction.value == '向上':
                w1_end = self.bis[i].fx_b.fx
                break
        
        if w1_end > 0:
            self.w1_start = w1_start
            self.w1_end = w1_end
            self.state = "w1_formed"
    
    def _try_confirm_w2(self):
        """尝试确认W2"""
        # W2确认条件：向下笔没有跌破W1起点
        last_bi = self.bis[-1]
        
        if last_bi.direction.value != '向下':
            return
        
        # 检查是否跌破W1起点
        if self.w1_start and last_bi.fx_b.fx < self.w1_start:
            # 跌破！可能不是W2，重新扫描
            self._reset()
            self.state = "initial"
            return
        
        # 没跌破W1起点，确认W2
        self.w2_end = last_bi.fx_b.fx
        self.state = "w2_formed"
    
    def _try_confirm_w3(self):
        """尝试确认W3"""
        # W3确认条件：向上笔突破W1终点
        last_bi = self.bis[-1]
        
        if last_bi.direction.value != '向上':
            return
        
        # 检查是否突破W1终点
        if self.w1_end and last_bi.fx_b.fx > self.w1_end:
            # 突破！W3确认
            self.w3_end = last_bi.fx_b.fx
            self.state = "w3_formed"
    
    def _reset(self):
        """重置"""
        self.w1_start = None
        self.w1_end = None
        self.w2_end = None
        self.w3_end = None
        self.w4_end = None
        self.state = "initial"
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'state': self.state,
            'w1': (self.w1_start, self.w1_end) if self.w1_start else None,
            'w2': self.w2_end,
            'w3': self.w3_end,
            'w4': self.w4_end,
        }
    
    def get_description(self) -> str:
        """获取描述"""
        if self.state == "initial":
            return "W1形成中"
        elif self.state == "w1_formed":
            return f"W1: {self.w1_start:.2f}→{self.w1_end:.2f}, W2确认中"
        elif self.state == "w2_formed":
            return f"W1: {self.w1_start:.2f}→{self.w1_end:.2f}, W2: {self.w2_end:.2f}, W3确认中"
        elif self.state == "w3_formed":
            return f"W1: {self.w1_start:.2f}→{self.w1_end:.2f}, W2: {self.w2_end:.2f}, W3: {self.w3_end:.2f}, W4调整中"
        elif self.state == "w4_formed":
            return f"W4调整中, 低点{self.w4_end:.2f}"
        elif self.state == "w5_in_progress":
            return "W5进行中"
        return self.state


def test_600985():
    """测试600985"""
    # 加载数据
    db_2025 = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/daily_data_year=2025/data.parquet')
    db_2026 = pd.read_parquet('/root/.openclaw/workspace/data/warehouse/daily_data_year=2026/data.parquet')
    df = pd.concat([db_2025, db_2026])
    
    # 聚合周线
    df_sym = df[df['symbol'] == '600985'].copy()
    df_sym['date'] = pd.to_datetime(df_sym['date'])
    df_sym['week'] = df_sym['date'].dt.to_period('W').apply(lambda x: x.start_time)
    weekly = df_sym.groupby('week').agg({
        'symbol': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).reset_index()
    weekly.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    weekly = weekly.sort_values('date').reset_index(drop=True)
    
    # CZSC
    bars = []
    for _, row in weekly.iterrows():
        bar = RawBar(
            symbol=str(row['symbol']),
            dt=pd.to_datetime(row['date']).to_pydatetime(),
            freq=Freq.D,
            open=float(row['open']), high=float(row['high']),
            low=float(row['low']), close=float(row['close']),
            vol=float(row['volume']), amount=0.0
        )
        bars.append(bar)
    
    c = CZSC(bars)
    bis = c.finished_bis
    
    print("=== 周线笔 ===")
    for i, bi in enumerate(bis):
        d = '↑' if bi.direction.value == '向上' else '↓'
        print(f"笔{i+1} {d} {bi.fx_a.dt.strftime('%Y-%m-%d')}~{bi.fx_b.dt.strftime('%Y-%m-%d')}: {bi.fx_a.fx:.2f}→{bi.fx_b.fx:.2f}")
    
    print("\n=== 波浪计数过程 ===")
    wc = WaveCounter()
    for i, bi in enumerate(bis):
        state = wc.feed(bi)
        d = '↑' if bi.direction.value == '向上' else '↓'
        print(f"笔{i+1} {d}: {wc.get_description()}")
    
    print("\n=== 最终结果 ===")
    result = wc.get_state()
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_600985()
