# strategies/score_wavechan_combo.py
"""
Score + WaveChan V3 组合策略

方向A：Score 选强势股 → WaveChan 过滤找回调买点

逻辑：
1. ScoreStrategy 每日选 Top N 强势股（主升浪）
2. WaveChan 检查是否处于 W2/W4 回调买点
3. 同时满足 → 买入信号

接口：
- get_combo_signals(start_date, end_date) → (buy_signals, sell_signals)
- filter_by_wavechan(candidates, date) → filtered_candidates
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from .base import BaseStrategy
from .score_strategy import ScoreStrategy
from .wavechan_v3 import WaveEngine, WaveSnapshot

logger = logging.getLogger(__name__)


class ScoreWaveChanComboStrategy(BaseStrategy):
    """
    Score + WaveChan 组合策略
    
    顺势而为：Score 选出主升浪股票，WaveChan 找回调买点
    """

    def __init__(self, db_path: str = None, config: dict = None):
        default_config = {
            # Score 配置
            'top_n': 5,
            'score_weights': None,
            
            # WaveChan 过滤配置
            'wavechan_buy_signals': ['W2_BUY', 'W4_BUY_CONFIRMED', 'W4_BUY_ALERT', 'C_BUY'],
            'wavechan_cache_dir': '/tmp/score_wavechan_combo',
            'warm_cache': True,  # 是否预热缓存
            
            # 仓位配置
            'position_size': 0.3,
            'stop_loss_pct': 0.05,
            'profit_target_pct': 0.20,
            'max_hold_days': 20,
        }
        self.config = {**default_config, **(config or {})}
        
        super().__init__(db_path, self.config)
        
        # 初始化 Score 策略
        self.score_strategy = ScoreStrategy(db_path=db_path, config={
            'top_n': self.config['top_n'],
            **(self.config.get('score_config') or {})
        })
        
        # WaveChan 引擎缓存（symbol → WaveEngine）
        self._wave_engine_cache: Dict[str, WaveEngine] = {}
        self._wave_cache_dir = self.config['wavechan_cache_dir']
        
        import os
        os.makedirs(self._wave_cache_dir, exist_ok=True)

    def _get_wave_engine(self, symbol: str) -> WaveEngine:
        """获取或创建 WaveChan 引擎"""
        if symbol not in self._wave_engine_cache:
            cache_dir = f"{self._wave_cache_dir}/{symbol}"
            import os
            os.makedirs(cache_dir, exist_ok=True)
            self._wave_engine_cache[symbol] = WaveEngine(
                symbol=symbol,
                cache_dir=cache_dir
            )
        return self._wave_engine_cache[symbol]

    def _check_wavechan_signal(self, symbol: str, date: str, daily_data: dict) -> Optional[dict]:
        """
        检查 WaveChan 信号
        
        Args:
            symbol: 股票代码
            date: 日期
            daily_data: 当日数据 {open, high, low, close, volume}
            
        Returns:
            信号信息 dict 或 None
        """
        engine = self._get_wave_engine(symbol)
        
        # 喂数据
        snapshot = engine.feed_daily({
            'date': date,
            'open': daily_data['open'],
            'high': daily_data['high'],
            'low': daily_data['low'],
            'close': daily_data['close'],
            'volume': daily_data.get('volume', 0),
        })
        
        # 获取信号
        sig_dict = engine.get_signal()
        signal = sig_dict.get('signal', 'NO_SIGNAL')
        
        # 检查是否是我们想要的买点信号
        allowed_signals = self.config['wavechan_buy_signals']
        if signal in allowed_signals:
            return {
                'signal': signal,
                'status': sig_dict.get('status', ''),
                'price': sig_dict.get('price', daily_data['close']),
                'stop_loss': sig_dict.get('stop_loss'),
                'reason': sig_dict.get('reason', ''),
                'snapshot': snapshot,
            }
        
        return None

    def filter_by_wavechan(self, candidates: pd.DataFrame, date: str, daily_data_map: dict) -> pd.DataFrame:
        """
        用 WaveChan 过滤候选股
        
        Args:
            candidates: Score 选出的候选股 DataFrame
            date: 当前日期
            daily_data_map: {symbol: {open, high, low, close, volume}}
            
        Returns:
            同时满足 Score + WaveChan 条件的股票
        """
        if candidates.empty:
            return candidates
        
        filtered = []
        
        for _, row in candidates.iterrows():
            symbol = row['symbol']
            
            # 检查是否有当日数据
            if symbol not in daily_data_map:
                logger.debug(f"{symbol} 无当日数据，跳过")
                continue
            
            daily_data = daily_data_map[symbol]
            
            # 检查 WaveChan 信号
            wave_signal = self._check_wavechan_signal(symbol, date, daily_data)
            
            if wave_signal:
                # 合并 Score 信号 + WaveChan 信号
                row = row.copy()
                row['wave_signal'] = wave_signal['signal']
                row['wave_status'] = wave_signal['status']
                row['wave_reason'] = wave_signal['reason']
                row['wave_stop_loss'] = wave_signal['stop_loss']
                filtered.append(row)
                logger.info(f"✅ {symbol} Score+WaveChan 双确认: {wave_signal['signal']} | {wave_signal['reason']}")
            else:
                logger.debug(f"{symbol} WaveChan 无买点信号，跳过")
        
        if filtered:
            return pd.DataFrame(filtered)
        return pd.DataFrame()

    def get_combo_signals(self, start_date: str, end_date: str, daily_data_map: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取组合策略信号
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            daily_data_map: {symbol: {date: {open, high, low, close, volume}}}
            
        Returns:
            (buy_signals, sell_signals)
        """
        logger.info(f"[ScoreWaveChanCombo] 生成组合信号 {start_date} ~ {end_date}")
        
        # 1. 获取 Score 原始信号
        score_signals = self.score_strategy.get_signals(start_date, end_date)
        buy_candidates, sell_signals = score_signals
        
        if buy_candidates is None or buy_candidates.empty:
            logger.info("[ScoreWaveChanCombo] 无 Score 买入候选")
            return pd.DataFrame(), sell_signals
        
        # 2. 用 WaveChan 过滤
        # 按日期处理
        all_filtered = []
        dates = buy_candidates.index.unique()
        
        for date in dates:
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            daily_candidates = buy_candidates.loc[date]
            if isinstance(daily_candidates, pd.Series):
                daily_candidates = daily_candidates.to_frame().T
            
            # 构建当日 data map
            date_data_map = {}
            if daily_data_map and date_str in daily_data_map:
                date_data_map = daily_data_map[date_str]
            
            # WaveChan 过滤
            filtered = self.filter_by_wavechan(daily_candidates, date_str, date_data_map)
            
            if not filtered.empty:
                all_filtered.append(filtered)
        
        if all_filtered:
            final_buy = pd.concat(all_filtered)
        else:
            final_buy = pd.DataFrame()
        
        # 3. 卖出信号沿用 Score 逻辑
        return final_buy, sell_signals

    def get_signals(self, start_date: str, end_date: str):
        """
        实现 BaseStrategy 接口
        """
        return self.get_combo_signals(start_date, end_date)

    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        实现 BaseStrategy 接口
        组合策略本身不生成特征，委托给 ScoreStrategy
        """
        return self.score_strategy.generate_features(start_date, end_date)

    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        实现 BaseStrategy 接口
        组合策略的买入信号由 WaveChan 过滤决定
        """
        # 返回空列表，实际买入逻辑在 get_combo_signals 中处理
        return []

    def warm_cache(self, symbols: List[str], start_date: str, end_date: str, daily_data_map: dict):
        """
        预热 WaveChan 缓存
        
        提前计算好波浪状态，避免回测时重复计算
        """
        logger.info(f"[ScoreWaveChanCombo] 预热缓存 {len(symbols)} 只股票")
        
        for symbol in symbols:
            engine = self._get_wave_engine(symbol)
            
            # 获取该股的数据范围
            if daily_data_map:
                for date_str, data in daily_data_map.items():
                    if symbol in data:
                        bar_data = data[symbol]
                        engine.feed_daily({
                            'date': date_str,
                            'open': bar_data['open'],
                            'high': bar_data['high'],
                            'low': bar_data['low'],
                            'close': bar_data['close'],
                            'volume': bar_data.get('volume', 0),
                        })
        
        logger.info(f"[ScoreWaveChanCombo] 缓存预热完成")


# ============================================================
# 辅助函数：独立回测用
# ============================================================

def load_daily_data_for_combo(symbols: List[str], start_date: str, end_date: str) -> dict:
    """
    加载回测所需的日线数据
    
    Returns:
        {
            '2025-01-02': {
                '600036': {open, high, low, close, volume},
                ...
            },
            ...
        }
    """
    import os
    from utils.parquet_db import ParquetDatabaseIntegrator
    
    db_path = os.environ.get('PARQUET_DB_PATH', '/root/.openclaw/workspace/data/warehouse')
    
    # 加载所有数据
    all_data = []
    for year in range(2020, 2027):
        path = f"{db_path}/daily_data_year={year}/"
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"加载 {year} 年数据失败: {e}")
    
    if not all_data:
        return {}
    
    data = pd.concat(all_data)
    
    # 过滤 symbols
    if symbols:
        data = data[data['symbol'].isin(symbols)]
    
    # 构建 date → {symbol: data} 的映射
    result = {}
    for date, group in data.groupby('date'):
        date_str = date if isinstance(date, str) else pd.to_datetime(date).strftime('%Y-%m-%d')
        result[date_str] = {}
        for _, row in group.iterrows():
            result[date_str][row['symbol']] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if pd.notna(row.get('volume')) else 0,
            }
    
    return result
