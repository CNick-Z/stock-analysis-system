#!/usr/bin/env python3
"""
Score策略 + 启动期识别器 2020-2025年回测
修复多年份预计算问题
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import timedelta
from strategies.score_strategy import ScoreStrategy
from backtester import BacktestOrchestrator

# ============================================================
# 启动期识别器配置
# ============================================================
STARTUP_FILTER = {
    'max_recent_return': 0.15,    # 近5日涨幅不超过15%
    'ma_cross_max_days': 3,        # MA5上穿MA20不超过3天
    'volume_ratio_min': 1.5,       # 量比 > 1.5
}


class StartupFilterStrategy(ScoreStrategy):
    """
    Score策略 + 启动期股票识别器（多年份版）
    按年份分别预计算MA交叉，支持多年连续回测
    """
    
    _gc_cache = {}  # {(year, symbol): set of dates}
    _data_cache = {}  # {year: DataFrame}
    _current_year = None
    
    @classmethod
    def reset(cls):
        """重置所有缓存"""
        cls._gc_cache = {}
        cls._data_cache = {}
        cls._current_year = None
    
    def _ensure_year_data(self, year: int):
        """确保指定年份的数据已加载"""
        if year == self._current_year and year in self._data_cache:
            return
        
        pq_path = f'/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/'
        
        if not os.path.exists(pq_path):
            print(f"[StartupFilter] 无数据: {pq_path}")
            self._data_cache[year] = None
            self._gc_cache[year] = {}
            self._current_year = year
            return
        
        files = [f for f in os.listdir(pq_path) if f.endswith('.parquet')]
        if not files:
            self._data_cache[year] = None
            self._gc_cache[year] = {}
            self._current_year = year
            return
        
        print(f"[StartupFilter] 加载 {year} 年数据...")
        full = pd.read_parquet(os.path.join(pq_path, files[0]))
        full['date'] = full['date'].astype(str).str[:10]
        full = full.sort_values(['symbol', 'date'])
        
        # 计算MA5和MA20
        for w in [5, 20]:
            full[f'ma_{w}'] = full.groupby('symbol')['close'].transform(
                lambda x: x.rolling(w, min_periods=w).mean()
            )
        
        self._data_cache[year] = full
        
        # 预计算金叉日期
        gc_days = {}
        gc_count = 0
        for sym, grp in full.groupby('symbol'):
            grp = grp.sort_values('date')
            ma5 = grp['ma_5'].values
            ma20 = grp['ma_20'].values
            dates = grp['date'].values
            
            for i in range(1, len(grp)):
                if not (np.isnan(ma5[i-1]) or np.isnan(ma20[i-1]) or np.isnan(ma5[i]) or np.isnan(ma20[i])):
                    if ma5[i-1] <= ma20[i-1] and ma5[i] > ma20[i]:
                        if sym not in gc_days:
                            gc_days[sym] = set()
                        gc_days[sym].add(dates[i])
                        gc_count += 1
        
        self._gc_cache[year] = gc_days
        self._current_year = year
        print(f"[StartupFilter] {year} 年: 计算了 {gc_count} 个金叉点")
    
    def _precompute_for_range(self, start_date: str, end_date: str):
        """预计算日期范围内所有年份的数据"""
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        for year in range(start_year, end_year + 1):
            self._ensure_year_data(year)
    
    def get_signals(self, start_date, end_date):
        # 预计算所有涉及年份的数据
        self._precompute_for_range(start_date, end_date)
        
        # 调用父类生成原始信号
        buy_signals, sell_signals = super().get_signals(start_date, end_date)
        
        if buy_signals is None or buy_signals.empty:
            return buy_signals, sell_signals
        
        # 对候选股票应用启动期过滤
        all_dates = sorted(buy_signals.index.unique())
        filtered_buy = []
        
        for date in all_dates:
            date_str = str(date)[:10]
            year = int(date_str[:4])
            daily = buy_signals[buy_signals.index == date].copy()
            
            startup = self._get_startup_symbols(date_str, daily, year)
            if startup:
                daily = daily[daily['symbol'].isin(startup)]
            
            if len(daily) > 0:
                filtered_buy.append(daily)
        
        if filtered_buy:
            filtered_signals = pd.concat(filtered_buy)
        else:
            filtered_signals = buy_signals.iloc[:0].copy()
        
        total_orig = len(buy_signals)
        total_filt = len(filtered_signals)
        print(f"[StartupFilter] {start_date}~{end_date}: 原始 {total_orig} → 启动期 {total_filt} (过滤 {total_orig - total_filt})")
        
        return filtered_signals, sell_signals
    
    def _get_startup_symbols(self, date_str, candidate_stocks, year):
        """判断候选股票中哪些是启动期"""
        if year not in self._data_cache or self._data_cache[year] is None:
            return set()
        
        full = self._data_cache[year]
        gc_days = self._gc_cache.get(year, {})
        
        candidates = set(candidate_stocks['symbol'].tolist())
        result = []
        
        date = pd.to_datetime(date_str)
        
        for sym in candidates:
            if sym not in gc_days:
                continue
            
            symbol_gc_dates = gc_days[sym]
            
            # 检查近N天内是否有金叉
            cross_in_window = False
            for d in range(1, STARTUP_FILTER['ma_cross_max_days'] + 1):
                check_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
                if check_date in symbol_gc_dates:
                    cross_in_window = True
                    break
            if not cross_in_window:
                continue
            
            # 获取该股票当日数据
            sym_data = full[
                (full['symbol'] == sym) &
                (full['date'] <= date_str)
            ].sort_values('date')
            
            if len(sym_data) < 6:
                continue
            
            prices = sym_data.tail(6)['close'].values
            vol = sym_data.tail(6)['volume'].values
            
            # 近5日涨幅
            ret_5d = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            if ret_5d > STARTUP_FILTER['max_recent_return']:
                continue
            
            # 量比
            vol_avg = np.mean(vol[:-1]) if len(vol) > 1 else vol[0]
            vol_ratio = vol[-1] / vol_avg if vol_avg > 0 else 0
            if vol_ratio < STARTUP_FILTER['volume_ratio_min']:
                continue
            
            result.append(sym)
        
        return set(result)


if __name__ == '__main__':
    config = {
        'top_n': 5,
        'weights': {'technical': 0.4483, 'capital_flow': 0.2947, 'market_heat': 0.257},
        'tech_weights': {
            'price_growth': 0.1704, 'ma_condition': 0.1704, 'angle_condition': 0.1704,
            'macd_condition': 0.1366, 'macd_jc': 0.1366, 'rsi_oversold': 0.0853,
            'kdj_oversold': 0.0853, 'cci_oversold': 0.0682, 'bollinger_condition': 0.0682,
            'volume_score': 0.1704,
        }
    }
    
    import time
    
    print("=" * 60)
    print("Score策略 + 启动期识别器 2020-2025年回测")
    print("=" * 60)
    print(f"启动期条件: MA5上穿MA20≤{STARTUP_FILTER['ma_cross_max_days']}天 | "
          f"近5日涨幅<{STARTUP_FILTER['max_recent_return']:.0%} | 量比>{STARTUP_FILTER['volume_ratio_min']}")
    print()
    
    # 原始策略
    print("🚀 跑原始Score策略 2020-2025...")
    t0 = time.time()
    StartupFilterStrategy.reset()
    strat = ScoreStrategy(config=config)
    orch = BacktestOrchestrator(db_path=None, live_plot=False, position_limit=5, strategy_name='score')
    orch.strategy = strat
    orch.strategy.db_manager = orch.db._parquet
    r1 = orch.run('2020-01-01', '2025-12-31')
    t1 = time.time()
    s1 = r1['summary']
    print(f"\n📊 原始Score (2020-2025):")
    print(f"   收益率: {s1['total_return']:.2%}  |  最大回撤: {s1['max_drawdown']:.2%}  |  净值: {s1['final_value']:,.0f}")
    print(f"   交易次数: {s1.get('total_trades', 'N/A')}")
    print(f"   耗时: {t1-t0:.1f}秒")
    
    # 启动期策略
    print("\n" + "=" * 60)
    print("🚀 跑Score + 启动期识别器 2020-2025...")
    StartupFilterStrategy.reset()
    t0 = time.time()
    strat2 = StartupFilterStrategy(config=config)
    orch2 = BacktestOrchestrator(db_path=None, live_plot=False, position_limit=5, strategy_name='score')
    orch2.strategy = strat2
    orch2.strategy.db_manager = orch2.db._parquet
    r2 = orch2.run('2020-01-01', '2025-12-31')
    t1 = time.time()
    s2 = r2['summary']
    print(f"\n📊 Score + 启动期识别器 (2020-2025):")
    print(f"   收益率: {s2['total_return']:.2%}  |  最大回撤: {s2['max_drawdown']:.2%}  |  净值: {s2['final_value']:,.0f}")
    print(f"   交易次数: {s2.get('total_trades', 'N/A')}")
    print(f"   耗时: {t1-t0:.1f}秒")
    
    # 对比
    print("\n" + "=" * 60)
    print("📈 对比结果:")
    ret_diff = s2['total_return'] - s1['total_return']
    dd_diff = s2['max_drawdown'] - s1['max_drawdown']
    print(f"   收益率变化: {ret_diff:+.2%}")
    print(f"   最大回撤变化: {dd_diff:+.2%}")
    if ret_diff > 0:
        print("   ✅ 启动期识别器有效！")
    else:
        print("   ❌ 启动期识别器无效")