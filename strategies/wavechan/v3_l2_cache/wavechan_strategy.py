# strategies/wavechan/v3_l2_cache/wavechan_strategy.py
# =============================================================
"""
WaveChan V3 策略 - 统一框架适配层
=============================================================

实现 BasePortfolio 所需的 3 个标准接口：
  - filter_buy(daily_df)  → 候选股票
  - score(candidates)     → 评分排序
  - should_sell(row, pos, market) → 出场判断（含连续3天出场逻辑）

入场信号：Plan A双入口
  波浪入口: W2_BUY(score≥50+RSI<52) / W4_BUY(score≥30+RSI<50)
  缠论入口: V1_BUY/V2_BUY(confidence≥0.7) 独立通过
出场逻辑：止损 / 止盈 / 时间止损 / 波浪信号出场 / 连续3天 wave_trend==down
禁止：C_BUY / w5_formed状态 / RSI>60
出场逻辑：止损 / 止盈 / 时间止损 / 波浪信号出场 / 连续3天 wave_trend==down
"""
# =============================================================

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.wavechan.v3_l2_cache.strategy import (
    WaveEngine, SymbolWaveCache, WaveCounterV3, BiRecord,
    aggregate_daily_to_weekly, WaveSnapshot,
)
from strategies.wavechan.v3_l2_cache.wave_backtrack_corrector import (
    WaveBacktrackCorrector,
)

logger = logging.getLogger(__name__)

# ================================================================
# W1 确认状态跟踪
# ================================================================

@dataclass
class W1TrackingState:
    """跟踪C_BUY入场后的W1确认状态"""
    entry_date: str
    entry_price: float
    # c_low: float  # C浪低点（已废弃，未使用）
    confirmed: bool = False
    failed: bool = False
    confirm_deadline: str = None  # 5个交易日后
    daily_lows: list = field(default_factory=list)  # 记录每日最低价

# ================================================================
# 配置
# ================================================================
L2_CACHE_DIR = "/data/warehouse/wavechan/wavechan_cache"
THRESHOLD = 50          # 买入阈值（研究结论：score≥50时20日胜率52.6%均幅+2.33%，score<20时46.5%均幅+0.75%）
STOP_LOSS_PCT = 0.08     # 8% 止损
TAKE_PROFIT_PCT = 0.20  # 20% 止盈
MAX_HOLD_DAYS = 20      # 最大持仓天数
MAX_POSITIONS = 5        # 最大持仓数


# ================================================================
# WaveChanStrategy - 统一框架适配层
# ================================================================

class WaveChanStrategy:
    """
    WaveChan V3 策略（统一框架版）

    实现 BasePortfolio 所需的 3 个标准接口：
      filter_buy(daily_df) → 返回有买入信号的候选股票 DataFrame
      score(candidates)   → 返回评分排序后的 DataFrame（含 score 列）
      should_sell(row, pos, market) → (是否出场, 出场原因)

    出场优先级：止损 > 止盈 > 时间止损 > 波浪信号出场 > 连续3天空头
    """

    def __init__(
        self,
        l2_cache_dir: str = L2_CACHE_DIR,
        threshold: int = THRESHOLD,
        stop_loss_pct: float = 0.05,   # 5%止损（旧V3盈利参数）
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_hold_days: int = MAX_HOLD_DAYS,
        max_positions: int = MAX_POSITIONS,
        position_size: float = 0.20,
        use_weekly_filter: bool = True,  # 是否启用周线过滤
        stock_list: list = None,        # 限定股票列表（None=全部）
        # 基本面过滤参数
        min_market_cap: float = 0,     # 最小市值(亿)，0=不过滤
        min_pe: float = 0,             # 最小PE，0=不过滤
        min_listing_days: int = 0,     # 最短上市天数，0=不过滤
        iron_laws_strict: bool = False,  # 是否要求铁律验证通过（wave_recognizer all_verified）
    ):
        self.l2_cache_dir = l2_cache_dir
        self.threshold = threshold
        self.stop_loss_pct = stop_loss_pct  # 5%（旧V3参数）
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days
        self.max_positions = max_positions
        self.position_size = position_size
        self.use_weekly_filter = use_weekly_filter
        self._stock_list = set(stock_list) if stock_list else None
        # 基本面过滤
        self.min_market_cap = min_market_cap
        self.min_pe = min_pe
        self.min_listing_days = min_listing_days
        self._fundamental_filter_enabled = min_market_cap > 0 or min_pe > 0 or min_listing_days > 0
        self.iron_laws_strict = iron_laws_strict

        # 引擎缓存（symbol → WaveEngine）
        self._engines: dict = {}

        # 周线方向缓存（symbol → {date: direction}）
        # 在 prepare() 时预计算，避免在 filter_buy 中重复计算
        self._weekly_dir_cache: Dict[str, Dict[str, str]] = defaultdict(dict)
        # 周线波浪阶段缓存（symbol → {date: impulse_state}）
        # impulse_state: W3_in_progress | W5_in_progress | W4_in_progress | W1_or_W2 | W5_done
        self._weekly_state_cache: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._weekly_counter_cache: Dict[str, WaveCounterV3] = {}

        # ── 回溯修正 v2.0 ─────────────────────────────────────────────
        # 大级别缓存（symbol → large_degree）
        self._large_degree_cache: Dict[str, str] = {}
        # 修正后 wave_state 缓存（symbol → {date: corrected_state}）
        self._corrected_wave_state_cache: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._backtrack_enabled: bool = True   # 可通过参数关闭

    def _write_l1_signals_to_cache(self, df: pd.DataFrame, current_date: str):
        """
        将 L1 fallback 计算的信号写回 L2 缓存目录，格式：
        /data/warehouse/wavechan/wavechan_cache/l2_hot_year=YYYY_month=MM/data.parquet

        这样做的好处：第一次用 L1 fallback 计算后，下次再跑相同月份就有现成的 L2 数据了。
        """
        import pandas as pd
        from pathlib import Path

        sig_rows = df[df.get('has_signal', False) == True]
        if sig_rows.empty:
            return

        year = current_date[:4]
        month = current_date[5:7]
        cache_path = Path(f'/data/warehouse/wavechan/wavechan_cache/l2_hot_year={year}_month={month}/data.parquet')

        # 读取现有数据（如果有）
        if cache_path.exists():
            try:
                existing = pd.read_parquet(cache_path)
                # 删除当前日期的旧数据
                existing = existing[existing['date'] != current_date]
                # 合并新数据
                sig_to_write = sig_rows[['date', 'symbol', 'has_signal', 'signal_type',
                                          'signal_status', 'total_score', 'wave_trend',
                                          'wave_state', 'stop_loss']].copy()
                sig_to_write = sig_to_write.astype({
                    'has_signal': 'bool',
                    'total_score': 'float64',
                    'stop_loss': 'float64',
                })
                result = pd.concat([existing, sig_to_write], ignore_index=True)
            except Exception:
                # 读取失败则只写新数据
                result = sig_rows[['date', 'symbol', 'has_signal', 'signal_type',
                                    'signal_status', 'total_score', 'wave_trend',
                                    'wave_state', 'stop_loss']].copy()
        else:
            # 不存在则创建目录
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result = sig_rows[['date', 'symbol', 'has_signal', 'signal_type',
                               'signal_status', 'total_score', 'wave_trend',
                               'wave_state', 'stop_loss']].copy()

        try:
            result.to_parquet(cache_path, index=False)
            logger.info(f"[L1Fallback→L2] {current_date} 写入 {len(sig_rows)} 条信号到缓存 {cache_path.name}")
        except Exception as e:
            logger.warning(f"[L1Fallback→L2] {current_date} 写入缓存失败: {e}")

        # 策略数据接口
        self.name = "WaveChanStrategy"
        # 波浪缠论信号列，需要从 L2 cache 加载
        self.REQUIRED_COLUMNS = [
            'has_signal', 'total_score', 'signal_type', 'signal_status',
            'wave_trend', 'wave_state', 'stop_loss',
        ]

    def prepare(self, dates: list, loader):
        """
        加载波浪缠论 L2 cache 数据，并预计算周线方向

        Args:
            dates: 交易日列表 [start_date, ..., end_date]
            loader: DataFrame（当年完整数据，用于计算周线方向）
        """
        self._extra_data = {}
        from utils.data_loader import load_wavechan_signals
        if len(dates) < 2:
            dates = [dates[0], dates[0]]
        new_cache = load_wavechan_signals(dates[0], dates[-1])
        if not hasattr(self, '_wave_cache') or self._wave_cache.empty:
            self._wave_cache = new_cache
        elif not new_cache.empty:
            # 追加合并（只追加有数据的月份，空月份不覆盖已有数据）
            self._wave_cache = pd.concat([self._wave_cache, new_cache], ignore_index=True)
            self._wave_cache = self._wave_cache.drop_duplicates(subset=['date', 'symbol'], keep='last')

        # ── 缠论信号缓存加载（P1接入，2026-04-16）──────────────────────
        # 从预计算的 Parquet 加载（scripts/compute_chanlun_batch.py 生成）
        # Plan A: (wave信号 AND score≥阈值) OR (缠论V1/V2信号 AND confidence≥0.7)
        self._chanlun_cache: pd.DataFrame = pd.DataFrame()
        CHANLUN_CACHE_DIR = Path('/data/warehouse/chanlun/cache')
        if dates and CHANLUN_CACHE_DIR.exists():
            try:
                # 加载最近日期的缓存（当日或前一交易日）
                cache_files = sorted(CHANLUN_CACHE_DIR.glob('chanlun_*.parquet'), reverse=True)
                if cache_files:
                    latest_cache = cache_files[0]
                    self._chanlun_cache = pd.read_parquet(latest_cache)
                    n_high_conf = (self._chanlun_cache['confidence'] >= 0.7).sum() if not self._chanlun_cache.empty else 0
                    logger.info(f"[ChanlunCache] 已加载: {latest_cache.name}, "
                                f"总信号{len(self._chanlun_cache)}只, 高置信≥0.7: {n_high_conf}只")
                else:
                    logger.info("[ChanlunCache] 无缓存文件，跳过缠论信号")
            except Exception as e:
                logger.warning(f"[ChanlunCache] 加载失败: {e}")

        # ── L1 Fallback 预计算信号加载 ───────────────────────────────
        # 从预计算的 Parquet 加载（scripts/build_l1_signals_2026.py 生成）
        # 比逐股票实时计算快 100 倍
        # ── 保存完整历史数据并预计算周线方向（向量化，O(n)）─────────────
        if loader is not None and hasattr(loader, 'empty') and not loader.empty:
            self._full_df = loader.copy()
        else:
            self._full_df = pd.DataFrame()

        # ── 基本面数据加载（在 _full_df 之后）────────────────────────
        if self._fundamental_filter_enabled:
            self._load_fundamental_data(dates)

        # ── v2.0 回溯修正 ───────────────────────────────────────────
        # 将 wave_state 从 wave_cache 合并到 _full_df，然后修正
        self._apply_backtrack_correction(dates)

        # ── 向量化预计算所有 symbol-date 的周线方向 ────────────────────
        self._precompute_weekly_dirs_vectorized(dates)

        return self._wave_cache

    def _load_fundamental_data(self, dates: list):
        """加载基本面数据并构建有效股票集合"""
        import pandas as pd
        from pathlib import Path
        
        DATA_DIR = Path('/root/.openclaw/workspace/data/warehouse')
        
        # 加载股票基本信息
        basic = pd.read_parquet(DATA_DIR / 'stock_basic_info.parquet')
        basic['listing_date'] = pd.to_datetime(basic['listing_date'])
        
        # 计算上市天数
        end_date = pd.Timestamp(dates[-1]) if dates else pd.Timestamp('2025-12-31')
        basic['listing_days'] = (end_date - basic['listing_date']).dt.days
        
        # 计算市值(亿) - 使用最新收盘价 × 总股本
        # 先获取每只股票的最新收盘价
        if hasattr(self, '_full_df') and not self._full_df.empty:
            latest_prices = self._full_df.groupby('symbol')['close'].last().reset_index()
            basic = basic.merge(latest_prices, on='symbol', how='left')
            basic['market_cap_yi'] = basic['total_shares'] * basic['close'] / 1e8
        else:
            basic['market_cap_yi'] = 0
        
        # 加载财务数据(PE)
        fin = pd.read_parquet(DATA_DIR / 'financial_summary.parquet')
        # 取最新的PE
        latest_fin = fin.sort_values('date').groupby('symbol').last().reset_index()
        basic = basic.merge(latest_fin[['symbol', 'pe_ratio']], on='symbol', how='left')
        
        # 构建有效股票集合
        mask = pd.Series(True, index=basic.index)
        if self.min_market_cap > 0:
            mask &= basic['market_cap_yi'].fillna(0) >= self.min_market_cap
        if self.min_pe > 0:
            mask &= basic['pe_ratio'].fillna(0) >= self.min_pe
        if self.min_listing_days > 0:
            mask &= basic['listing_days'].fillna(0) >= self.min_listing_days
        
        self._valid_symbols = set(basic[mask]['symbol'])
        logger.info(f"[FundamentalFilter] 市值>{self.min_market_cap}亿 PE>{self.min_pe} 上市>{self.min_listing_days}天 → {len(self._valid_symbols)} 只股票")

    def _apply_backtrack_correction(self, dates: list):
        """
        v2.0 回溯修正主入口。

        将 wave_state 合并到 _full_df，然后对每只股票执行三层架构修正：
          Layer 3: 周线大级别判定
          Layer 2: 子浪合法性约束
          Layer 1: Viterbi 回溯重标注

        修正结果存入 self._corrected_wave_state_cache 和 self._large_degree_cache。
        """
        if not self._backtrack_enabled:
            logger.info("[Backtrack] 已禁用，跳过修正")
            return

        if self._full_df.empty:
            logger.warning("[Backtrack] _full_df 为空，跳过修正")
            return

        # 检查 wave_state 列是否存在
        if 'wave_state' not in self._full_df.columns:
            logger.warning("[Backtrack] _full_df 缺少 wave_state 列，跳过修正")
            return

        # 只取有信号的数据进行修正（减少计算量）
        # wave_cache 中 has_signal==True 的行才是实际使用的
        if not self._wave_cache.empty and 'wave_state' in self._wave_cache.columns:
            # 取 date 范围内的 wave_cache 数据
            start_date = dates[0] if dates else None
            end_date = dates[-1] if dates else None
            wave_df = self._wave_cache.copy()
            if start_date and end_date:
                wave_df = wave_df[
                    (wave_df['date'] >= start_date) &
                    (wave_df['date'] <= end_date)
                ]
            # 合并 wave_state 到 _full_df（有信号的行）
            wave_merge = wave_df[['date', 'symbol', 'wave_state']].drop_duplicates(
                ['date', 'symbol']
            )
            full_df = self._full_df.merge(
                wave_merge, on=['date', 'symbol'], how='left', suffixes=('', '_ws')
            )
            if 'wave_state_ws' in full_df.columns:
                full_df['wave_state'] = full_df['wave_state_ws'].fillna(full_df['wave_state'])
                full_df.drop(columns=['wave_state_ws'], inplace=True)
        else:
            full_df = self._full_df.copy()

        # 过滤必要的列
        required_cols = ['date', 'symbol', 'wave_state', 'close']
        for col in ['high', 'low']:
            if col in full_df.columns:
                required_cols.append(col)

        available = [c for c in required_cols if c in full_df.columns]
        work_df = full_df[available].dropna(subset=['date', 'symbol'])

        if work_df.empty:
            logger.warning("[Backtrack] 无可用数据，跳过修正")
            return

        # 执行回溯修正
        corrector = WaveBacktrackCorrector(reversal_threshold=-0.02, lookback=20)
        corrected_df = corrector.correct(work_df)

        # 填入缓存
        self._large_degree_cache.clear()
        self._corrected_wave_state_cache.clear()

        for symbol, grp in corrected_df.groupby('symbol'):
            # 大级别（v1.0 corrector 不提供，设为 UNKNOWN）
            large_deg = grp['large_degree'].iloc[0] if 'large_degree' in grp.columns and len(grp) > 0 else 'UNKNOWN'
            self._large_degree_cache[symbol] = large_deg
            # 修正后 wave_state
            self._corrected_wave_state_cache[symbol] = dict(
                zip(grp['date'].astype(str), grp['wave_state_corrected'])
            )

        n = len(self._corrected_wave_state_cache)
        logger.info(f"[Backtrack v1.0] 修正完成: {n} 只股票的大级别和波浪状态已缓存")

    def _precompute_weekly_dirs_vectorized(self, dates: list):
        """
        向量化预计算所有 symbol-date 的周线方向

        使用 20 日滚动窗口：
          bullish_count = sum(wave_state in {w2_formed, w4_formed}) over rolling 20 days  # w2=W3推动, w4=W5推动
          correction_count = sum(wave_state in {w3_formed, w4_in_progress}) over rolling 20 days  # w3=W4调整, w4_in_progress=W4调整
          weekly_dir = 'bullish' if bullish_count > correction_count else 'neutral'

        存入 self._weekly_dir_cache[symbol][date_str] = direction
        """
        if self._full_df.empty or 'wave_state' not in self._full_df.columns:
            return

        df = self._full_df.copy()
        # 确保按 symbol-date 排序
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 为每个 symbol 计算 rolling counts
        # bullish = w2_formed(W3推动) or w4_formed(W5推动)
        df['_is_bullish'] = df['wave_state'].isin(['w2_formed', 'w4_formed']).astype(int)
        # correction = w3_formed(W4调整) or w4_in_progress(W4调整中)
        df['_is_correction'] = df['wave_state'].isin(['w3_formed', 'w4_in_progress']).astype(int)

        # 按 symbol 分组，滚动 20 日
        n = 20
        df['_bullish_sum'] = df.groupby('symbol')['_is_bullish'].transform(
            lambda x: x.rolling(window=n, min_periods=10).sum()
        )
        df['_correction_sum'] = df.groupby('symbol')['_is_correction'].transform(
            lambda x: x.rolling(window=n, min_periods=10).sum()
        )
        df['_weekly_dir'] = np.where(
            df['_bullish_sum'] > df['_correction_sum'],
            'bullish',
            'neutral'
        )

        # 存入缓存
        self._weekly_dir_cache.clear()
        for sym, grp in df.groupby('symbol'):
            self._weekly_dir_cache[sym] = dict(zip(grp['date'].astype(str), grp['_weekly_dir']))

        logger.info(f"[WeeklyFilter] 周线方向预计算完成: {len(self._weekly_dir_cache)} 只股票")

        # ── 同时预计算周线波浪阶段 ────────────────────────────────────
        # v2.0: 使用修正后的 wave_state 和大级别上下文
        self._weekly_state_cache.clear()
        for sym, grp in df.groupby('symbol'):
            state_map = {}
            large_deg = self._large_degree_cache.get(sym, 'UNKNOWN')
            corrected = self._corrected_wave_state_cache.get(sym, {})
            for _, row in grp.iterrows():
                d = str(row['date'])
                # 优先使用修正后的 wave_state，否则用原始
                ws = corrected.get(d, str(row.get('wave_state', 'initial')))
                state_map[d] = self._wave_state_to_impulse_state(ws, large_deg)
            self._weekly_state_cache[sym] = state_map

        logger.info(f"[WeeklyFilter] 周线波浪阶段预计算完成: {len(self._weekly_state_cache)} 只股票")

    # ------------------------------------------------------------------
    # 周线波浪阶段推断
    # ------------------------------------------------------------------

    # 推动浪（冲动）阶段：允许 BUY
    # W3_in_progress = w2_formed（W2完成，当前W3推动）
    # W5_in_progress = w4_formed（W4完成，当前W5推动）
    ALLOWED_WAVE_STATES = {'W3_in_progress', 'W5_in_progress'}

    def _wave_state_to_impulse_state(self, wave_state: str, large_degree: str = None) -> str:
        """
        将 wave_state 映射为周线波浪阶段（impulse_state）

        v2.0 改进（Oracle 2026-04-11）：
          - 引入大级别上下文（large_degree）
          - 大级别决定子浪转移的合法性
          - w3/w4/w5 → w1 在特定大级别下是合法的（子序列重置），不再强制修正

        大级别 → BUY 允许规则：
          - W3_in_progress（w2_formed）：大浪3推动中 ✅
          - W5_in_progress（w4_formed）：大浪5推动中 ✅
          - 大级别为 WAVE4/WAVEA/WAVEC（调整浪）时：
              子浪 w2_formed/w4_formed 可能对应调整浪内部，不买
              只买 C_BUY 熊市反转信号

        BUY 允许状态：W3_in_progress（w2_formed）、W5_in_progress（w4_formed）
        BUY 拒绝状态：W4_correction、W2_correction、W5_done、W1_or_W2
        """
        ws = str(wave_state)
        large = large_degree or 'UNKNOWN'

        # 基础映射（v2.0 不变）
        mapping = {
            'w1_formed':        'W2_correction',
            'w2_formed':        'W3_in_progress',
            'w3_formed':        'W4_correction',
            'w4_formed':        'W5_in_progress',
            'w4_in_progress':  'W4_correction',
            'w5_formed':        'W5_done',
        }
        base = mapping.get(ws, 'W1_or_W2')

        # v2.0 大级别上下文调整
        # 调整浪大级别（WAVE4/WAVEA/WAVEB/WAVEC）中，子浪推动不等于大浪推动
        # 大级别为调整浪时：
        #   - w2_formed 可能只是反弹中的2浪，不是大级别推动
        #   - 只有在 large_trend='bullish' 时才认为 w2_formed = W3_in_progress
        if large in ('WAVE4_DOWN', 'WAVEA_DOWN', 'WAVEC_DOWN', 'WAVEB_UP'):
            # 大级别调整/反弹中：严格控制 BUY
            # C浪/W4/A浪/B浪中的子浪不算大级别推动
            if ws == 'w2_formed':
                return 'W1_or_W2'   # 调整浪中的反弹，不买
            if ws == 'w4_formed':
                return 'W4_correction'  # 调整浪中的子浪4，不买

        return base

    def get_weekly_impulse_state(self, symbol: str, date_str: str) -> str:
        """
        获取某只股票在指定日期的周线波浪阶段（v2.0）

        优先使用预计算缓存，缓存未命中时实时计算。
        实时计算时使用修正后的 wave_state 和大级别上下文。

        Returns:
            'W3_in_progress'   → W2完成，W3推动浪进行中，允许 BUY ✅
            'W5_in_progress'   → W4完成，W5推动浪进行中，允许 BUY ✅
            'W4_correction'   → W3完成或W4进行中，调整浪，拒绝 BUY ❌
            'W2_correction'   → W1完成，W2调整中，趋势不明，拒绝 BUY ❌
            'W5_done'          → 5浪完成，新周期酝酿，拒绝 BUY ❌
            'W1_or_W2'         → 大级别调整浪中子浪重置，拒绝 BUY ❌
        """
        date_str = str(date_str)
        if symbol in self._weekly_state_cache and date_str in self._weekly_state_cache[symbol]:
            return self._weekly_state_cache[symbol][date_str]

        # 实时计算（使用修正后的 wave_state 和大级别）
        ws = 'initial'
        large_deg = self._large_degree_cache.get(symbol, 'UNKNOWN')

        # 优先使用修正后的 wave_state
        if symbol in self._corrected_wave_state_cache:
            ws = self._corrected_wave_state_cache[symbol].get(date_str, 'initial')
        elif hasattr(self, '_full_df') and not self._full_df.empty:
            row = self._full_df[
                (self._full_df['symbol'] == symbol) &
                (self._full_df['date'].astype(str) == date_str)
            ]
            if not row.empty:
                ws = str(row.iloc[0].get('wave_state', 'initial'))

        state = self._wave_state_to_impulse_state(ws, large_deg)
        # 回填缓存
        self._weekly_state_cache[symbol][date_str] = state
        return state

    # ------------------------------------------------------------------
    # L1 Fallback：L2无数据时从L1实时计算V3波浪信号
    # ------------------------------------------------------------------

    def _compute_l1_signal(self, symbol: str, current_date: str) -> Optional[dict]:
        """
        当L2缓存无数据时，从L1极值实时计算V3波浪信号

        使用 wave_recognizer 的 weekly 波浪分析：
          - identify_wave_stage: 获取趋势和波浪序列
          - label_wave_stage: 获取艾略特标签和铁律验证

        每天只计算一次每个 symbol，结果缓存到 self._l1_cache。
        缓存以 year 为粒度（同一年的信号通常不变）。

        Returns:
            dict with keys: has_signal, signal_type, total_score, wave_trend,
                           signal_status, wave_state, stop_loss
            or None if computation fails
        """
        # L1 预计算信号不再加载（2026-04-23 从 prepare 中移除）
        # 不依赖外部 L1 数据

        try:
            import datetime
            year = int(current_date[:4]) if current_date else datetime.datetime.now().year

            # ── Step 1: identify_wave_stage → 趋势 + 波浪序列 ──────────────
            # 注意：实际目录名是 stock-wave-recognition（含连字符），不是 stock_wave_recognition
            _wr_path = str(PROJECT_ROOT.parent / 'stock-wave-recognition')
            if _wr_path not in sys.path:
                sys.path.insert(0, _wr_path)
            from wave_recognizer import identify_wave_stage, label_wave_stage

            weekly_trend, wave_seq = identify_wave_stage(
                symbol, year, years=[year - 1, year]
            )

            # ── Step 2: label_wave_stage → 艾略特标签 + 铁律验证 ────────────
            label_result = label_wave_stage(symbol, year)

            labeled_waves = label_result.get('labeled_waves', [])
            iron_law_passed = label_result.get('iron_law_passed', False)
            cycle_type = label_result.get('cycle_type', 'unknown')

            if not labeled_waves:
                return None

            # ── Step 3: 从最新标注的浪判断信号类型 ──────────────────────────
            # 最新完成的浪（倒数第1或第2个，因为最后一个可能还在进行中）
            completed = [w for w in labeled_waves if not w.get('in_progress', False)]
            if not completed:
                return None

            latest = completed[-1]
            # 注意：用 eliott_label 而不是 wave（numeric）
            # eliott_label: 'W1','W2','W3','W4','W5','WA','WB','WC'
            wave_label = str(latest.get('elliott_label', ''))

            # 映射 elliott_label → V3 signal_type
            # W2 完成 → 当前处于 W3 推动 → W2_BUY
            # W4 完成 → 当前处于 W5 推动 → W4_BUY
            # WA/WB/WC 完成 → C_BUY 熊市反弹/新推动起点
            signal_map = {
                'W2': 'W2_BUY',
                'W4': 'W4_BUY',
                'W1': 'C_BUY',
                'WA': 'C_BUY',
                'WB': 'C_BUY',
                'WC': 'C_BUY',
            }
            signal_type = signal_map.get(wave_label, None)

            if signal_type is None:
                return None

            # ── Step 4: 计算评分 ────────────────────────────────────────────
            # 基于铁律验证 + 置信度
            confidence = latest.get('confidence', 0.5)
            iron_bonus = 20 if iron_law_passed else 0
            wave_bonus = 10 if cycle_type == 'impulse' else 5
            total_score = int(confidence * 60 + iron_bonus + wave_bonus)

            # ── Step 5: 判断 wave_state ──────────────────────────────────────
            ws_map = {
                'W1': 'w1_formed', 'W2': 'w2_formed',
                'W3': 'w3_formed', 'W4': 'w4_formed',
                'W5': 'w5_formed',
                'WA': 'w1_formed', 'WB': 'w2_formed', 'WC': 'w3_formed',
            }
            wave_state = ws_map.get(wave_label, 'initial')

            # ── Step 6: wave_trend ───────────────────────────────────────────
            wave_trend_map = {'up': 'long', 'down': 'down'}
            wave_trend = wave_trend_map.get(weekly_trend, 'neutral')

            # ── Step 7: signal_status ────────────────────────────────────────
            # L1 fallback 直接 confirmed（铁律已体现在 total_score 中）
            signal_status = 'confirmed'

            # ── Step 8: stop_loss ────────────────────────────────────────────
            # 从最近的低点计算：W2低点 × 0.97 或 W4低点 × 0.97
            recent_low = latest.get('low_price') or latest.get('start_price') or latest.get('end_price')
            if recent_low:
                stop_loss = round(recent_low * 0.97, 2)
            else:
                stop_loss = 0.0

            result = {
                'has_signal': True,
                'signal_type': signal_type,
                'total_score': total_score,
                'wave_trend': wave_trend,
                'signal_status': signal_status,
                'wave_state': wave_state,
                'stop_loss': stop_loss,
                '_weekly_dir': 'bullish' if weekly_trend == 'up' else 'neutral',
                '_impulse_state': self._wave_state_to_impulse_state(wave_state),
            }
            # 不缓存（每次都重新计算，确保自动修正生效）
            return result

        except Exception as e:
            logger.warning(f"[L1Fallback] {symbol}@{current_date} 计算失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------

    def filter_buy(self, daily_df: pd.DataFrame, date: str = None) -> pd.DataFrame:
        """
        过滤候选股票（含周线过滤 + 缠论补充）

        Args:
            daily_df: 当日候选股票数据（包含 has_signal/signal_type/wave_trend 等列）
            date: 当前日期（框架传入）

        信号入口（Plan A - 并行双入口，2026-04-16）：
          Wave入口: (wave信号 AND score≥阈值) OR (缠论V1/V2信号 AND confidence≥0.7)

        波浪条件（2026-04-16研究优化）：
          W2_BUY:
            - has_signal == True
            - total_score >= 50
            - rsi_14 < 52
            - wave_trend in (long, neutral, '')
            - signal_status == 'confirmed'

          W4_BUY:
            - has_signal == True
            - total_score >= 30
            - rsi_14 < 50
            - wave_trend in (long, neutral, '')
            - signal_status == 'confirmed'

          禁止: C_BUY / w5_formed状态 / RSI>60

        缠论补充条件:
          - signal_type in (V1_BUY, V2_BUY)
          - confidence >= 0.7
          - 独立于wave信号，不受周线过滤限制（缠论自带级别递归）

        【周线过滤】：周线 bullish（W3/W5推动中）接受所有买入；neutral只有日线上升趋势才买。
        """
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        current_date = date or (df['date'].iloc[0] if 'date' in df.columns else None)

        # ── 合并 L2 wave_cache 到每日数据 ────────────────────────────────
        # wave_cache 在 prepare() 中加载（包含全年数据）
        _wc_ok = (hasattr(self, '_wave_cache') and
            not self._wave_cache.empty and
            'date' in self._wave_cache.columns and
            'symbol' in self._wave_cache.columns and
            'has_signal' in self._wave_cache.columns)
        if _wc_ok:
            # 如果当前月份的 L2 数据不在缓存中，实时加载（支持 April 等后续月份）
            if current_date and not self._wave_cache.empty:
                cached_months = set(str(d)[:7] for d in self._wave_cache['date'].unique())
                current_month = str(current_date)[:7]
                if current_month not in cached_months:
                    # 当前月份不在缓存，重新加载（包含已写入的 L1→L2 回补数据）
                    from utils.data_loader import load_wavechan_signals
                    reloaded = load_wavechan_signals(current_date, current_date)
                    if not reloaded.empty:
                        self._wave_cache = pd.concat([self._wave_cache, reloaded], ignore_index=True)
                        logger.info(f"[WaveMerge] {current_date} 重新加载月份数据，当前缓存共 {len(self._wave_cache)} 行")

            wave_cols_extra = ['all_verified', 'iron_law_1', 'iron_law_2', 'iron_law_3']
            wave_cols = ['date', 'symbol', 'has_signal', 'signal_type', 'signal_status',
                         'total_score', 'wave_trend', 'wave_state', 'stop_loss']
            wave_cols += [c for c in wave_cols_extra if c in self._wave_cache.columns]
            wave_cols = [c for c in wave_cols if c in self._wave_cache.columns]
            # 只删除信号重叠列（不删 merge key: date, symbol）
            sig_cols_to_replace = [c for c in wave_cols if c not in ('date', 'symbol') and c in df.columns]
            if sig_cols_to_replace:
                df = df.drop(columns=sig_cols_to_replace)
            wave_sub = self._wave_cache[wave_cols].drop_duplicates(['date', 'symbol'])
            # 筛选当日数据减少 merge 规模
            if current_date:
                wave_sub = wave_sub[wave_sub['date'] == str(current_date)[:10]]
            n_sig = wave_sub['has_signal'].sum() if not wave_sub.empty and 'has_signal' in wave_sub.columns else 0
            if not wave_sub.empty:
                df = df.merge(wave_sub, on=['date', 'symbol'], how='left')
            logger.info(f"[WaveMerge] {current_date} wave_sub={len(wave_sub)} signals={n_sig}")

        # ── L2-only 模式：禁止 L1 fallback ─────────────────────────────────
        # 检测 L2 当日是否有信号（必须在 wave merge 之后）
        _force_l2_only = getattr(self, '_l2_only_mode', False)
        l2_has_data = (
            'has_signal' in df.columns and
            df['has_signal'].eq(True).any()
        )
        if _force_l2_only:
            if not l2_has_data:
                logger.info(f"[L2-Only] {current_date} L2无信号，返回空（不触发L1 fallback）")
                return pd.DataFrame()

        # ── L2-only：L2 无数据则直接返回空，不走任何 L1 fallback ──────────
        # 正确逻辑：L2 没有这个日期 → 用 L1 重建 L2 缓存（离线一次性）
        #         → 运行时只读 L2，不实时 fallback
        if not l2_has_data:
            logger.info(f"[L2-Only] {current_date} L2无信号，返回空（禁止L1 fallback）")
            return pd.DataFrame()

        # ── Step 1: 日线基础过滤 ───────────────────────────────────────
        _false = pd.Series(False, index=df.index)
        _zero = pd.Series(0, index=df.index)
        _empty = pd.Series('', index=df.columns)
        has_signal = df['has_signal'] if 'has_signal' in df.columns else _false
        total_score = df['total_score'] if 'total_score' in df.columns else _zero
        wave_trend = df['wave_trend'] if 'wave_trend' in df.columns else _empty

        # 【优化2026-04-16】差异化阈值 + RSI过滤
        # W2_BUY: score≥50 + RSI<52
        # W4_BUY: score≥30 + RSI<50
        # w5_formed状态：最危险，Step3的ALLOWED_WAVE_STATES已过滤
        # TODO: momentum_score>6过滤（列暂不存在，需要从价格数据计算）
        rsi = df['rsi_14'].fillna(100) if 'rsi_14' in df.columns else pd.Series(100, index=df.index)

        w2_mask = (
            has_signal.eq(True) &
            total_score.ge(50) &
            rsi.lt(52) &
            wave_trend.isin(['long', 'neutral', '']) &
            (df['signal_type'] == 'W2_BUY' if 'signal_type' in df.columns else pd.Series(False, index=df.index))
        )
        w4_mask = (
            has_signal.eq(True) &
            total_score.ge(30) &
            rsi.lt(50) &
            wave_trend.isin(['long', 'neutral', '']) &
            (df['signal_type'].isin({'W4_BUY', 'W4_BUY_ALERT', 'W4_BUY_CONFIRMED'}) if 'signal_type' in df.columns else pd.Series(False, index=df.index))
        )
        mask = w2_mask | w4_mask

        if 'signal_status' in df.columns:
            # L1 fallback 信号已内置 signal_status='confirmed'，直接接受
            # 非 L1 行 signal_status='' 会自然被筛掉
            signal_ok = df['signal_status'].eq('confirmed')
            # 如果全是空（无效数据），跳过此过滤
            if signal_ok.any():
                mask &= signal_ok

        candidates = df[mask].copy()

        # ── Step 0.5: 缠论高置信信号补充（P1接入，Plan A 2026-04-16）──────
        # (wave信号 AND score≥阈值) OR (缠论V1/V2信号 AND confidence≥0.7)
        if (hasattr(self, '_chanlun_cache') and
                not self._chanlun_cache.empty and
                current_date):
            chanlun_sub = self._chanlun_cache[
                (self._chanlun_cache['confidence'] >= 0.7) &
                (self._chanlun_cache['signal_type'].isin({'V1_BUY', 'V2_BUY'}))
            ].copy()
            if not chanlun_sub.empty:
                # cache 已按 calc_date=今日 加载，date 列是信号产生日（可能早于今日）
                # 缠论信号有滞后性，保留 cache 中所有高置信信号，不按 date 再过滤
                if not chanlun_sub.empty:
                    # 排除已经在 candidates 里的股票
                    already_selected = set(candidates['symbol'].unique()) if not candidates.empty else set()
                    chanlun_new = chanlun_sub[~chanlun_sub['symbol'].isin(already_selected)]
                    if not chanlun_new.empty:
                        logger.info(f"[ChanlunSupplement] {current_date} 缠论高置信补充 "
                                    f"{len(chanlun_new)} 只（独立于wave信号）: "
                                    f"{chanlun_new['symbol'].tolist()[:5]}{'...' if len(chanlun_new)>5 else ''}")
                        # 从 df 中提取这些股票的完整数据
                        chanlun_symbols = chanlun_new['symbol'].tolist()
                        chanlun_extra = df[df['symbol'].isin(chanlun_symbols)].copy()
                        if not chanlun_extra.empty:
                            # 建立 symbol → chanlun 信号 的映射
                            chanlun_map = chanlun_new.set_index('symbol')
                            # 直接用缠论信号覆盖 wave 信号信息（不依赖 .where 的 NaN 逻辑）
                            chanlun_extra['_signal_source'] = 'chanlun'
                            chanlun_extra['signal_type'] = chanlun_extra['symbol'].map(chanlun_map['signal_type'])
                            chanlun_extra['total_score'] = (chanlun_extra['symbol'].map(chanlun_map['confidence']) * 100).fillna(0)
                            chanlun_extra['signal_status'] = 'chanlun_confirmed'
                            candidates = pd.concat([candidates, chanlun_extra], ignore_index=True)

        if candidates.empty:
            return candidates

        # ── Step 1.5: 股票列表过滤 ─────────────────────────────────────
        if self._stock_list is not None:
            candidates = candidates[candidates['symbol'].isin(self._stock_list)].copy()
            if candidates.empty:
                return candidates

        # ── Step 1.6: 基本面过滤 ──────────────────────────────────────
        if self._fundamental_filter_enabled and hasattr(self, '_valid_symbols'):
            n_before = len(candidates)
            candidates = candidates[candidates['symbol'].isin(self._valid_symbols)].copy()
            n_rejected = n_before - len(candidates)
            if n_rejected > 0:
                logger.info(f"[FundamentalFilter] {current_date} 基本面过滤拒绝 {n_rejected}/{n_before} 个候选")
            if candidates.empty:
                return candidates

        # ── Step 1.7: 铁律过滤（wave_recognizer）─────────────────────────
        if self.iron_laws_strict and not candidates.empty:
            if 'all_verified' in candidates.columns:
                n_before = len(candidates)
                # NaN=未验证(拒绝)，0=验证失败(拒绝)，1=验证通过(保留)
                candidates = candidates[
                    candidates['all_verified'] == 1
                ].copy()
                n_rejected = n_before - len(candidates)
                if n_rejected > 0:
                    logger.info(f"[IronLawsFilter] {current_date} 铁律过滤拒绝 {n_rejected}/{n_before} 个候选")
            else:
                logger.warning(f"[IronLawsFilter] all_verified 字段不存在，跳过铁律过滤")

        # ── Step 2: 周线过滤 ───────────────────────────────────────────
        if self.use_weekly_filter:
            weekly_mask = pd.Series(True, index=candidates.index)

            for idx, row in candidates.iterrows():
                sym = row.get('symbol', '')
                row_date = str(row.get('date', current_date or ''))
                signal_type = str(row.get('signal_type', ''))

                # C_BUY 是熊市反转信号，跳过周线过滤
                if signal_type == 'C_BUY':
                    continue

                # 优先使用 L1 fallback 计算的周线方向（_weekly_dir 列）
                # 否则回退到缓存的 _get_weekly_direction
                if '_weekly_dir' in row.index and pd.notna(row.get('_weekly_dir')):
                    weekly_dir = str(row['_weekly_dir'])
                else:
                    weekly_dir = self._get_weekly_direction(sym, row_date)

                daily_dir = str(row.get('wave_trend', 'neutral'))

                # 周线 bullish（w1/w3/w4完成）→ 接受所有买入
                # 周线 neutral（W3调整/W4进行中/W5完成/无趋势）→ 只有日线上升趋势才买
                if weekly_dir == 'neutral':
                    if daily_dir != 'long':
                        weekly_mask[idx] = False
                        logger.debug(f"[WeeklyFilter] {sym}@{row_date} 拒绝: 周线={weekly_dir}, 日线={daily_dir}")
                # weekly_dir == 'bullish': 全部接受（默认 True）

            filtered = candidates[weekly_mask].copy()
            n_rejected = len(candidates) - len(filtered)
            if n_rejected > 0:
                logger.info(f"[WeeklyFilter] {current_date} 周线过滤拒绝 {n_rejected}/{len(candidates)} 个候选")
        else:
            # 不启用周线过滤（仅用日线条件）
            logger.debug(f"[WeeklyFilter] DISABLED - 跳过周线过滤")
            filtered = candidates

        # ── Step 3: 周线波浪数过滤（激进版）──────────────────────────────
        # 只在 W3/W5 冲动（推动）浪中允许 BUY
        # W4 调整浪 / 早期 W1/W2 / W5 完成 → 全部拒绝
        #
        # 【修复2026-04-16】L2 cache 没有 _impulse_state 列，WaveNumFilter
        # 会错误地调用 get_weekly_impulse_state() 算出 W4_correction 导致全灭。
        # 只有在 _impulse_state 列真实存在时（L1 fallback 数据）才执行此过滤。
        # L2 模式的 V3 BUY 信号已带 wave_state（日线级别），本身已足够。
        has_impulse_state = '_impulse_state' in filtered.columns
        if not filtered.empty and has_impulse_state:
            wave_state_mask = pd.Series(True, index=filtered.index)
            for idx, row in filtered.iterrows():
                sym = str(row.get('symbol', ''))
                signal_type = str(row.get('signal_type', ''))

                # C_BUY 是熊市反转信号，跳过波浪数过滤
                if signal_type == 'C_BUY':
                    continue

                row_date = str(row.get('date', current_date or ''))

                # 优先使用 L1 fallback 计算的 impulse_state
                if '_impulse_state' in row.index and pd.notna(row.get('_impulse_state')):
                    weekly_state = str(row['_impulse_state'])
                else:
                    weekly_state = self.get_weekly_impulse_state(sym, row_date)

                if weekly_state not in self.ALLOWED_WAVE_STATES:
                    wave_state_mask[idx] = False
                    logger.debug(f"[WaveNumFilter] {sym}@{row_date} 拒绝: 周线波浪={weekly_state}")

            filtered = filtered[wave_state_mask].copy()
            n_rejected_state = (~wave_state_mask).sum()
            if n_rejected_state > 0:
                logger.info(f"[WaveNumFilter] {current_date} 波浪数过滤拒绝 {n_rejected_state} 个候选")

        # ── Step 1.9: 记录信号日期 ─────────────────────────────────────
        if current_date and not filtered.empty:
            filtered['signal_date'] = current_date

        # 记录信号日期
        if current_date and not filtered.empty:
            filtered['signal_date'] = current_date

        return filtered

    def _get_weekly_direction(self, symbol: str, date_str: str) -> str:
        """
        获取某只股票在指定日期的周线方向

        使用过去4周的日线 wave_state 众数作为周线状态代理：
          - bullish: 过去4周内，推动浪（w2_formed=W3/w4_formed=W5）天数 > 调整浪（w3_formed=W4/w4_in_progress）天数
          - neutral: 调整浪主导或其他情况

        解决了"接飞刀"问题：下跌趋势中（周线W3/W4调整）日线反弹不买。

        Returns:
            'bullish': 周线处于 W3/W5 推动浪（整体上升）
            'neutral': 周线处于 W2/W4 调整或无趋势
        """
        date_str = str(date_str)

        # 缓存命中
        if symbol in self._weekly_dir_cache and date_str in self._weekly_dir_cache[symbol]:
            return self._weekly_dir_cache[symbol][date_str]

        # 缓存未命中：实时计算
        direction = self._compute_weekly_dir_from_daily(symbol, date_str)

        # 回填缓存
        self._weekly_dir_cache[symbol][date_str] = direction
        return direction

    def _compute_weekly_dir_from_daily(self, symbol: str, date_str: str) -> str:
        """
        基于过去4周日线 wave_state 统计判断周线方向

        逻辑：
          - 驱动浪主导（w3/w4_formed 天数 > w2/w4_in_progress 天数）→ bullish
          - 调整浪主导或无趋势 → neutral

        由于波浪是分形的，日线的 w3/w4_formed 通常对应周线的 W1/W3 驱动浪。
        """
        if not hasattr(self, '_full_df') or self._full_df is None or self._full_df.empty:
            return 'neutral'

        sym_df = self._full_df[self._full_df['symbol'] == symbol].copy()
        if sym_df.empty:
            return 'neutral'

        sym_df['dt'] = pd.to_datetime(sym_df['date'])
        cutoff = pd.to_datetime(date_str)

        # 取过去4周（约20个交易日）的数据
        four_weeks_ago = cutoff - pd.Timedelta(days=28)
        hist = sym_df[(sym_df['dt'] >= four_weeks_ago) & (sym_df['dt'] <= cutoff)].sort_values('dt')

        if len(hist) < 10:
            return 'neutral'

        wave_state = hist.get('wave_state', pd.Series(dtype=str))
        if wave_state.empty or wave_state.isna().all():
            return 'neutral'

        # 驱动浪天数（推动浪=已完成的推动段后正处于下一个推动中）
        # w2_formed: W2完成 → 当前W3推动中 → 推动浪 ✅
        # w4_formed: W4完成 → 当前W5推动中 → 推动浪 ✅
        bullish_count = int((wave_state.isin(['w2_formed', 'w4_formed'])).sum())
        # 调整浪天数
        # w3_formed: W3完成 → 当前W4调整中 → 调整浪 ❌
        # w4_in_progress: W4调整进行中 → 调整浪 ❌
        correction_count = int((wave_state.isin(['w3_formed', 'w4_in_progress'])).sum())

        # 调整浪主导：neutral（接飞刀风险高）
        if correction_count > bullish_count:
            return 'neutral'
        # 驱动浪主导或均衡：bullish
        return 'bullish'

    def _aggregate_to_weekly(self, daily_df: pd.DataFrame) -> list:
        """
        将日线 DataFrame 聚合为周线 OHLCV 列表
        """
        if daily_df.empty or len(daily_df) < 5:
            return []

        df = daily_df.copy()
        df['dt'] = pd.to_datetime(df['date'])
        df['year_week'] = df['dt'].dt.isocalendar().year.astype(str) + '_' + df['dt'].dt.isocalendar().week.astype(str).str.zfill(2)

        weekly = df.groupby('year_week').agg({
            'date': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'symbol': 'first',
        }).reset_index(drop=True)

        return weekly.to_dict('records')

    def _weekly_bar_to_bi(self, wb: dict, seq: int):
        """
        将一根周线 OHLCV bar 转换为一个 BiRecord（周线笔）
        如果与上一根bar方向相反或幅度太小，返回 None
        """
        if wb.get('close') is None or wb.get('open') is None:
            return None

        direction = 'up' if wb['close'] >= wb['open'] else 'down'

        return BiRecord(
            seq=seq,
            direction=direction,
            start_price=float(wb['open']),
            end_price=float(wb['close']),
            start_date=str(wb['date'])[:10] if wb.get('date') else '',
            end_date=str(wb['date'])[:10] if wb.get('date') else '',
            volume=float(wb.get('volume', 0)),
        )

    def _calc_trading_days(self, start_date: str, n: int) -> str:
        """
        计算 n 个交易日后的日期字符串
        简化实现：每个日历日算一个（交易日概念由回测框架保证）
        """
        from datetime import datetime, timedelta
        try:
            start = pd.to_datetime(start_date)
            # TODO: 实盘需替换为真实交易日历（考虑节假日）
            end = start + timedelta(days=n * 2)
            return end.strftime('%Y-%m-%d')
        except:
            return start_date

    def _check_w1_failure(self, date: str, price: float, pos: dict) -> tuple:
        """检查W1是否失败，返回 (failed, reason)"""
        if 'w1_state' not in pos:
            return False, None

        w1 = pos['w1_state']
        if w1.confirmed or w1.failed:
            # 价格跌破C浪低点 = C_BUY失败（C浪还在延伸）
            if price < w1.c_low:
                w1.failed = True
                return True, f"C_BUY_FAILED(broke_c_low {w1.c_low:.2f})"
            if w1.confirmed and self.wave_trend == 'down':
                # W1确认后，趋势转空仍然要止损
                w1.failed = True
                return True, "W1_TREND_BREAK"
            return w1.failed, "W1_CONFIRMED" if w1.confirmed else "W1_FAILED"

        # 记录每日低点
        w1.daily_lows.append(price)

        # 检查是否超过确认deadline
        if date > w1.confirm_deadline:
            # 5天后：不创新低 → W1确认
            entry_low = min(w1.daily_lows)
            if entry_low > 0 and price >= entry_low * 0.95:  # 不创新低（5%容差）
                w1.confirmed = True
                logger.debug(f"[W1] W1确认: date={date}, entry_low={entry_low:.2f}")
                return False, "W1_CONFIRMED"
            else:
                w1.failed = True
                logger.debug(f"[W1] W1失败: date={date}, price={price:.2f}, entry_low={entry_low:.2f}")
                return True, "W1_FAILED"

        # 检查趋势破坏（立即失败）
        wave_trend = pos.get('_last_wave_trend', '')
        if wave_trend == 'down':
            w1.failed = True
            logger.debug(f"[W1] W1失败(趋势破坏): date={date}")
            return True, "W1_FAILED"

        return False, None

    def score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        评分排序

        直接使用 total_score 作为评分，降序排列
        """
        if candidates.empty:
            return pd.DataFrame()

        df = candidates.copy()
        df['score'] = df.get('total_score', 0)
        df = df.sort_values('score', ascending=False)
        return df

    def should_sell(
        self,
        row: pd.Series,
        pos: dict,
        market: dict = None
    ) -> Tuple[bool, str]:
        """
        判断是否出场（分类出场逻辑）

        出场条件（优先级）：
          1. 止损：今天收盘价 < 止损位
          2. 止盈：今天收盘价 > 入场价 * (1 + take_profit)
          3. 时间止损（分类）：W2_BUY=35天 / W4_BUY=25天 / C_BUY=15天
          4. 波浪出场：signal in (W5_SELL, SELL)
          5. W5失败浪出场（W4 BUY专用）：wave_state='w5_formed' + wave_trend='down'
          6. B浪见顶出场（C BUY专用）：wave_state='w5_formed'
          7. 波浪空头出场（W2/W4 BUY）：wave_trend='down' + consecutive_bad >= 1

        Args:
            row: 当日行情（含 next_open, close, wave_trend, wave_state, signal_type 等）
            pos: 持仓信息（avg_cost, entry_date, entry_signal, consecutive_bad_days, days_held 等）

        Returns:
            (should_sell: bool, reason: str)
        """
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

        entry_price = pos.get('avg_cost', 0)
        if entry_price <= 0:
            return False, "INVALID_POSITION"

        # 用今天收盘价判断（不用next_open，因为T日不知道T+1开盘价）
        current_close = row_dict.get('close', entry_price)
        if current_close <= 0:
            current_close = row_dict.get('open', entry_price)

        # ---------- 1. 止损（优先用信号自带的W2动态止损位，其次用固定百分比） ----------
        dyn_stop = row_dict.get('stop_loss', 0)
        if dyn_stop and dyn_stop > 0:
            stop_price = dyn_stop
            stop_label = f"W2止损@{dyn_stop:.2f}"
        else:
            stop_price = entry_price * (1 - self.stop_loss_pct)
            stop_label = f"固定止损@{stop_price:.2f}"

        # ---------- 止损pending确认机制 ----------
        # T日收盘价跌破止损位 → 标记pending不立即出场
        # T+1日开盘价继续低于止损位 → 确认出场
        # T+1日开盘价回升高于止损位 → 撤销pending继续持有
        if current_close < stop_price:
            if pos['extra'].get('stop_pending'):
                # T+1：昨日已pending，今日确认止损
                return True, f"STOP_LOSS_CONFIRMED @{current_close:.2f}({stop_label})"
            else:
                # T：首次触发，标记pending
                pos['extra']['stop_pending'] = True
                pos['extra']['stop_pending_date'] = current_date
                return False, f"STOP_PENDING @{current_close:.2f}<{stop_price:.2f}({stop_label})"

        # ---------- 1b. W1 失败检查（C_BUY 专用） ----------
        # C_BUY 入场后，5天内不创新低则W1确认（继续持有）；趋势破坏则立即失败
        current_date = row_dict.get('date', '')
        w1_failed, w1_reason = self._check_w1_failure(current_date, current_close, pos)
        if w1_failed:
            return True, f"W1_EXIT({w1_reason})"

        # ---------- 2. 止盈（固定20%） ----------
        profit_price = entry_price * (1 + self.take_profit_pct)
        if current_close >= profit_price:
            return True, f"TAKE_PROFIT @{current_close:.2f}"

        # ---------- 2b. 移动止盈（追踪高点回撤8%） ----------
        # 当已有持仓高点且超过20%目标时，启动移动止盈
        high_since_entry = pos.get('high_since_entry', 0)
        if high_since_entry > 0 and entry_price > 0:
            trailing_stop = high_since_entry * 0.92
            # 只有超过20%目标后才启用移动止盈
            if trailing_stop > profit_price:
                if current_close <= trailing_stop:
                    drawdown = (1 - current_close / high_since_entry) * 100
                    return True, f"TRAILING_STOP({high_since_entry:.2f}→{current_close:.2f},-{drawdown:.1f}%)"

        # ---------- 3. 时间止损（分类出场） ----------
        # V3 卖点信号研究：不同买点类型使用不同时间止损
        # W2_BUY: 无时间止损（给足够时间走出W5）
        # W4_BUY: 25天（调整浪，正常幅度）
        # C_BUY:  15天（反弹浪，快速结束）
        signal_type = pos.get('entry_signal', 'W2_BUY')
        max_days_map = {'W4_BUY': 25, 'C_BUY': 15}
        max_days = max_days_map.get(signal_type, 20)
        if signal_type == 'W2_BUY':
            max_days = 99999  # 无时间止损，用极大值替代
        # W1确认后：延长C_BUY的"耐心"时间，不因短期震荡而触发时间止损
        if signal_type == 'C_BUY' and 'w1_state' in pos:
            w1_state = pos['w1_state']
            if w1_state.confirmed:
                max_days = 99999  # W1确认后无时间止损
                logger.debug(f"[W1] C_BUY W1已确认，延长持仓时间限制")
        days_held = pos.get('days_held', 0)
        if days_held >= max_days:
            return True, f"TIME_EXIT({days_held}天,{signal_type})"

        # ---------- 4. 波浪出场信号 ----------
        row_signal_type = row_dict.get('signal_type', '')
        if row_signal_type in ('W5_SELL', 'SELL'):
            return True, f"WAVE_SIGNAL({row_signal_type})"

        # ---------- 5. W5失败浪出场（W4 BUY专用） ----------
        # W5 < W3 = 推动结构失败，说明W4_BUY的假设错误，立即出场
        # 检测方式：wave_state='w5_formed' + wave_trend='down'（W5未能突破W3高点）
        if signal_type == 'W4_BUY':
            wave_state = row_dict.get('wave_state', '')
            wave_trend = row_dict.get('wave_trend', '')
            if wave_state == 'w5_formed' and wave_trend == 'down':
                return True, "W5_FAILED_EXIT"

        # ---------- 5b. W5终结出场（改动3：W5未能突破W3高点，触发即出场） ----------
        # W5_FAILED_EXIT_V2：wave_state='w5_formed'（W5已形成）
        # W5未能有效突破W3高点（wave_trend='down'确认推动失败）
        if signal_type in ('W2_BUY', 'W4_BUY'):
            wave_state = row_dict.get('wave_state', '')
            wave_trend = row_dict.get('wave_trend', '')
            # W5终结确认：W5已形成 + 趋势转空头
            if wave_state == 'w5_formed' and wave_trend == 'down':
                return True, "W5_END_EXIT"

        # ---------- 6. B浪见顶出场（C BUY专用） ----------
        # C_BUY 入场后，等待B浪反弹结束（即 wave_state='w5_formed'，新一轮下跌开始）
        # 此时 wave_trend='down' 确认B浪反弹已结束，应离场
        if signal_type == 'C_BUY':
            wave_state = row_dict.get('wave_state', '')
            wave_trend = row_dict.get('wave_trend', '')
            if wave_state == 'w5_formed' and wave_trend == 'down':
                return True, "B_WAVE_TOP_EXIT"

        # ---------- 7. 波浪空头出场（W2/W4 BUY） ----------
        # wave_trend 变空头时出场，不再等3天
        consecutive_bad = pos.get('consecutive_bad_days', 0)
        wave_trend = row_dict.get('wave_trend', '')
        if wave_trend == 'down' and consecutive_bad >= 1:
            return True, f"WAVE_TREND_EXIT({consecutive_bad}天)"

        return False, ""

    def on_tick(self, row: pd.Series, pos: dict):
        """
        每日 tick 回调（框架调用）
        用于更新连续空头计数器、持仓天数、最高价追踪
        """
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        date = row_dict.get('date', '')
        low = row_dict.get('low', 0)

        # 更新持仓天数
        pos['days_held'] = pos.get('days_held', 0) + 1

        # 更新连续空头计数器
        wave_trend = row_dict.get('wave_trend', '')
        if wave_trend == 'down':
            pos['consecutive_bad_days'] = pos.get('consecutive_bad_days', 0) + 1
        else:
            pos['consecutive_bad_days'] = 0
        # 记录当日 wave_trend 供 W1 检查使用
        pos['_last_wave_trend'] = wave_trend

        # 追踪持仓期间最高价（用于移动止盈）
        high = row_dict.get('high', 0)
        if high and high > 0:
            pos['high_since_entry'] = max(pos.get('high_since_entry', 0), high)

        # C_BUY 入场时初始化 W1 跟踪状态
        entry_signal = pos.get('entry_signal', '')
        if entry_signal == 'C_BUY' and 'w1_state' not in pos:
            pos['w1_state'] = W1TrackingState(
                entry_date=date,
                entry_price=pos.get('avg_cost', 0),
                c_low=low if low > 0 else pos.get('avg_cost', 0),
                confirm_deadline=self._calc_trading_days(date, 5)
            )
            logger.debug(f"[W1] C_BUY 入场初始化 W1 跟踪: date={date}, deadline={pos['w1_state'].confirm_deadline}")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_entry_conditions(self) -> dict:
        """返回当前入场条件（用于日志/报告）"""
        return {
            "signal": "W2_BUY / W4_BUY / C_BUY",
            "threshold": f"total_score >= {self.threshold}",
            "trend": "wave_trend in (up, neutral)",
        }

    def __repr__(self) -> str:
        return (
            f"<WaveChanStrategy "
            f"threshold={self.threshold} "
            f"stop_loss={self.stop_loss_pct:.0%} "
            f"take_profit={self.take_profit_pct:.0%} "
            f"max_hold={self.max_hold_days}天>"
        )
