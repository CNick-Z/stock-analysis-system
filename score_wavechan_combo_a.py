#!/usr/bin/env python3
"""
Score + WaveChan V3 组合方向A 回测
======================================

策略逻辑（方向A）：
  Score 每日选 Top N 强势股（主升浪）
      ↓
  WaveChan 检查是否处于 W2/W4 回调买点
      ↓
  有买点信号 → 买入

集成方式：
  - ScoreStrategy.get_signals() → 每日候选股
  - WaveEngine.get_snapshot() → W2/W4 状态检查
  - 只在 WaveChan 发出 W2/W4 买点信号时买入

不修改现有 Score v4 框架代码，纯增量实现。
"""

import sys, os, time, json, logging, glob
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from strategies.score_strategy import ScoreStrategy
from strategies.wavechan_v3 import WaveEngine, WaveSnapshot
from backtest_bias_corrector import (
    BiasCorrector, AdjustmentHandler, LookAheadBiasFixer,
    create_bias_corrector
)

# ============================================================
# 日线数据加载
# ============================================================

WAREHOUSE_PATH = "/root/.openclaw/workspace/data/warehouse"


def load_daily_data(start_date: str, end_date: str,
                    symbols: list = None) -> pd.DataFrame:
    """加载日线数据（多年份合并）"""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4]) + 1
    dfs = []
    for year in range(start_year, end_year):
        parquet_file = os.path.join(WAREHOUSE_PATH, f"daily_data_year={year}", "data.parquet")
        if os.path.exists(parquet_file):
            try:
                df = pd.read_parquet(parquet_file)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if symbols:
                    df = df[df['symbol'].isin(symbols)]
                dfs.append(df)
            except Exception as e:
                logging.warning(f"加载 {year} 年数据失败: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True).sort_values(['symbol', 'date'])
    return pd.DataFrame()


def build_price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """构建价格矩阵 (date × symbol × field)"""
    mat = df.pivot_table(
        index='date',
        columns='symbol',
        values=['open', 'close', 'high', 'low', 'volume']
    )
    return mat


def get_bar(symbol: str, date: str, price_df: pd.DataFrame) -> dict:
    """从价格矩阵中提取某日某股的 bar 数据"""
    try:
        date_idx = price_df.index.get_loc(date)
    except KeyError:
        return None

    sym_col = (slice(None), symbol)
    try:
        open_p = price_df.loc[date, ('open', symbol)]
        close_p = price_df.loc[date, ('close', symbol)]
        high_p = price_df.loc[date, ('high', symbol)]
        low_p = price_df.loc[date, ('low', symbol)]
        vol_p = price_df.loc[date, ('volume', symbol)]
    except KeyError:
        return None

    if pd.isna(open_p) or pd.isna(close_p):
        return None

    return {
        'date': date,
        'symbol': symbol,
        'open': float(open_p),
        'high': float(high_p),
        'low': float(low_p),
        'close': float(close_p),
        'volume': float(vol_p) if not pd.isna(vol_p) else 0.0,
    }


# ============================================================
# Score + WaveChan 组合引擎
# ============================================================

class ScoreWaveChanComboA:
    """
    Score v4 + WaveChan V3 组合策略（方向A）

    顺势而为：Score 选出主升浪股票，WaveChan 找回调买点
    集成点：蹭 get_snapshot() 接口串接 Score 和 WaveChan
    """

    # WaveChan 认可的买点信号（状态级别）
    BUY_SIGNALS = {'W2_BUY', 'W4_BUY'}
    BUY_STATUS = {'ALERT', 'CONFIRMED'}

    def __init__(self,
                 db_path: str = None,
                 top_n: int = 5,
                 wave_cache_dir: str = '/tmp/wavechan_combo_a_cache',
                 max_wave_engine_symbols: int = 50):
        """
        Args:
            top_n: Score 每日选股数
            wave_cache_dir: WaveChan 引擎缓存目录
            max_wave_engine_symbols: 最多同时缓存多少只股票的 WaveEngine
                                   （避免内存溢出，按需创建/销毁）
        """
        self.top_n = top_n
        self.wave_cache_dir = wave_cache_dir
        self.max_wave_engine_symbols = max_wave_engine_symbols
        os.makedirs(wave_cache_dir, exist_ok=True)

        # ScoreStrategy（不改框架，新建实例）
        self.score_strategy = ScoreStrategy(db_path=db_path, config={'top_n': top_n})

        # WaveEngine 缓存 {symbol: WaveEngine}
        self._wave_engines: dict = {}
        # WaveEngine 已加载的最后日期 {symbol: date_str}
        self._wave_last_date: dict = {}

    def _get_wave_engine(self, symbol: str) -> WaveEngine:
        """获取或创建 WaveEngine（带缓存）"""
        if symbol not in self._wave_engines:
            # 缓存满了就清掉最老的（按插入顺序）
            if len(self._wave_engines) >= self.max_wave_engine_symbols:
                oldest = next(iter(self._wave_engines))
                del self._wave_engines[oldest]
                del self._wave_last_date[oldest]

            cache_dir = os.path.join(self.wave_cache_dir, symbol)
            os.makedirs(cache_dir, exist_ok=True)
            engine = WaveEngine(symbol=symbol, cache_dir=cache_dir)
            self._wave_engines[symbol] = engine
            self._wave_last_date[symbol] = None
        return self._wave_engines[symbol]

    def _feed_history_to_engine(self,
                                 symbol: str,
                                 price_df: pd.DataFrame,
                                 up_to_date: str) -> WaveSnapshot:
        """
        把 symbol 在 up_to_date 之前的所有历史 K 线都喂给 WaveEngine

        WaveChan 需要累积 K 线才能形成笔（至少9根），
        所以回测时每天都要从第一天开始逐根喂入（Engine内部有去重逻辑）。
        """
        engine = self._get_wave_engine(symbol)
        last_loaded = self._wave_last_date.get(symbol)

        # 找出从 last_loaded 之后到 up_to_date 之前的所有日期
        available_dates = [d for d in price_df.index
                           if d >= (last_loaded or '1900-01-01') and d <= up_to_date]
        available_dates.sort()

        for date in available_dates:
            bar = get_bar(symbol, date, price_df)
            if bar is None:
                continue
            engine.feed_daily(bar)

        if available_dates:
            self._wave_last_date[symbol] = available_dates[-1]

        # get_snapshot() 在 WaveCounterV3 上，通过 daily_cache.counter 访问
        return engine.daily_cache.counter.get_snapshot()

    def warmup(self, symbols: list, price_df: pd.DataFrame, up_to_date: str):
        """
        预热 WaveChan 引擎：将 price_df 中所有历史 K 线都喂入对应 symbol 的 WaveEngine

        必须在回测开始前调用，确保 WaveChan 有足够的 K 线形成笔。
        """
        for symbol in symbols:
            self._feed_history_to_engine(symbol, price_df, up_to_date)

    def check_wavechan_signal(self,
                              symbol: str,
                              price_df: pd.DataFrame,
                              date: str) -> dict:
        """
        检查某只股票在指定日期的 WaveChan 买点信号

        Returns:
            {'has_signal': bool, 'state': str, 'signal': str,
             'status': str, 'stop_loss': float, 'reason': str}
        """
        engine = self._wave_engines.get(symbol)
        if engine is None:
            return {'has_signal': False, 'state': '', 'signal': '',
                    'status': '', 'stop_loss': None, 'reason': '', 'snapshot': None}

        # 触发 wave 计算
        self._feed_history_to_engine(symbol, price_df, date)

        # Snapshot via WaveCounterV3.get_snapshot()
        snapshot = engine.daily_cache.counter.get_snapshot()
        # Signal via WaveEngine.get_signal() (内部调用 counter.get_buy_sell_signals())
        sig_dict = engine.get_signal()

        signal = sig_dict.get('signal', 'NO_SIGNAL')
        status = sig_dict.get('status', '')

        has_signal = (
            signal in self.BUY_SIGNALS and
            status in self.BUY_STATUS
        )

        return {
            'has_signal': has_signal,
            'state': snapshot.state,
            'signal': signal,
            'status': status,
            'stop_loss': sig_dict.get('stop_loss'),
            'reason': sig_dict.get('reason', ''),
            'snapshot': snapshot,
        }

    def get_combo_signals(self,
                           start_date: str,
                           end_date: str,
                           price_df: pd.DataFrame,
                           trading_dates: list) -> tuple:
        """
        获取组合信号

        步骤：
        1. 调用 ScoreStrategy.get_signals() 获取每日候选股
        2. 对每个候选股调用 WaveEngine.get_snapshot() 检查 W2/W4 状态
        3. 双确认 → 加入买入列表

        Returns:
            (buy_signals_df, sell_signals_df)
        """
        logging.info(f"[ComboA] 生成信号 {start_date} ~ {end_date}")

        # Step 1: Score 原始信号
        score_result = self.score_strategy.get_signals(start_date, end_date)
        buy_candidates, sell_signals = score_result

        # ===== Look-Ahead Bias 验证 =====
        if buy_candidates is not None and not buy_candidates.empty:
            is_valid, msg = LookAheadBiasFixer.validate_signal_index(
                buy_candidates.reset_index(), date_col='date'
            )
            if not is_valid:
                logging.warning(f"[LookAhead] ⚠️ Score 信号存在 look-ahead bias: {msg}")
            else:
                logging.debug(f"[LookAhead] Score 信号 look-ahead 校验: {msg}")

        if buy_candidates is None or buy_candidates.empty:
            logging.info("[ComboA] Score 无买入候选")
            return pd.DataFrame(), sell_signals or pd.DataFrame()

        # Step 2: WaveChan 过滤
        filtered_rows = []
        dates_with_candidates = buy_candidates.index.unique()

        for date in dates_with_candidates:
            # 确保使用 Timestamp 作为 key（buy_candidates 的索引是 Timestamp）
            ts = pd.Timestamp(date)
            date_str = ts.strftime('%Y-%m-%d')
            daily_candidates = buy_candidates.loc[ts]
            if isinstance(daily_candidates, pd.Series):
                daily_candidates = daily_candidates.to_frame().T

            for _, row in daily_candidates.iterrows():
                symbol = row['symbol']
                # 日期来自当前迭代的 ts
                row_date = ts  # 已经是 Timestamp

                # 检查 WaveChan 信号
                wave_result = self.check_wavechan_signal(symbol, price_df, date_str)

                if wave_result['has_signal']:
                    new_row = row.copy()
                    new_row['wave_state'] = wave_result['state']
                    new_row['wave_signal'] = wave_result['signal']
                    new_row['wave_status'] = wave_result['status']
                    new_row['wave_reason'] = wave_result['reason']
                    new_row['wave_stop_loss'] = wave_result['stop_loss']
                    new_row['date'] = row_date  # 显式设置 date 字段
                    filtered_rows.append(new_row)
                    logging.info(
                        f"  ✅ {date_str} {symbol}: Score + WaveChan 双确认 "
                        f"[{wave_result['signal']}] {wave_result['reason']}"
                    )
                else:
                    logging.debug(
                        f"  ⏭ {date_str} {symbol}: WaveChan 无买点 "
                        f"(state={wave_result['state']})"
                    )

        if filtered_rows:
            buy_signals = pd.DataFrame(filtered_rows)
            buy_signals = buy_signals.set_index('date')
        else:
            buy_signals = pd.DataFrame()

        return buy_signals, sell_signals


# ============================================================
# 简化回测引擎（独立运行，不依赖 backtester.py）
# ============================================================

class SimpleComboBacktester:
    """Score + WaveChan 组合回测器（独立版）"""

    def __init__(self,
                 start_date: str = '2025-01-01',
                 end_date: str = '2025-12-31',
                 initial_cash: float = 500_000,
                 max_positions: int = 5,
                 position_size: float = 0.3,
                 stop_loss_pct: float = 0.08,
                 profit_target_pct: float = 0.20,
                 max_hold_days: int = 20,
                 top_n: int = 5,
                 wave_cache_dir: str = '/tmp/wavechan_combo_a_bt'):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.max_hold_days = max_hold_days
        self.top_n = top_n
        self.wave_cache_dir = wave_cache_dir

        self.combo: ScoreWaveChanComboA = None
        self.price_df: pd.DataFrame = None
        self.trading_dates: list = None
        # Bias 修正器（_load_data 后初始化）
        self.bias_corrector: BiasCorrector = None
        self.adj_handler: AdjustmentHandler = None

    def _load_data(self, lookback_days: int = 60):
        """
        加载数据（扩展 lookback 以便 WaveChan 积累足够笔）

        WaveChan 需要至少 9 根 K 线才能形成笔，
        czsc 需要约 20+ 根历史 K 线才能稳定识别笔。
        60 个交易日（约 3 个月）足以形成 W1-W3 结构，
        兼顾计算速度和波浪质量。
        lookback 仅用于喂给 WaveEngine 构建波浪结构，
        回测信号和交易仅在 start_date~end_date 内发生。
        """
        from datetime import timedelta
        lookback_start = (pd.to_datetime(self.start_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        logging.info(f"📊 加载日线数据（lookback: {lookback_start} ~ {self.end_date}）...")
        df = load_daily_data(lookback_start, self.end_date)
        if df.empty:
            raise ValueError("无市场数据，请检查数据路径")

        logging.info(f"  数据范围: {df['date'].min()} ~ {df['date'].max()}")
        logging.info(f"  股票数量: {df['symbol'].nunique()}")
        logging.info(f"  记录数: {len(df)}")

        self.price_df = build_price_matrix(df)
        # 回测交易日仅从 start_date 开始
        self.trading_dates = sorted([
            d for d in self.price_df.index.unique()
            if self.start_date <= d <= self.end_date
        ])
        logging.info(f"  回测交易日数量: {len(self.trading_dates)}")

        # ===== Bias 修正器初始化（复权校验 + 涨跌停/停牌过滤） =====
        logging.info("🔧 初始化 BiasCorrector + AdjustmentHandler...")
        self.adj_handler = AdjustmentHandler(self.price_df)
        # 使用回测交易日的 DatetimeIndex 构建 BiasCorrector
        trading_dates_idx = pd.DatetimeIndex(self.trading_dates)
        self.bias_corrector = BiasCorrector(self.price_df, trading_dates_idx)
        logging.info("✅ BiasCorrector + AdjustmentHandler 初始化完成")

    def _run(self) -> dict:
        """运行回测"""
        # ===== 前复权数据校验（fail-fast） =====
        if self.adj_handler:
            try:
                self.adj_handler.assert_adjusted()
            except (RuntimeError, ValueError) as e:
                logging.error(f"[BiasCorrector] {e}")
                raise

        logging.info("🚀 初始化组合引擎...")
        self.combo = ScoreWaveChanComboA(
            top_n=self.top_n,
            wave_cache_dir=self.wave_cache_dir,
        )

        # 按年分段（复用 backtester.py 的分年逻辑，节省内存）
        years = []
        cur = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        while cur <= end:
            y_end = pd.to_datetime(f"{cur.year}-12-31")
            years.append((cur.strftime('%Y-%m-%d'), min(y_end, end).strftime('%Y-%m-%d')))
            cur = pd.to_datetime(f"{cur.year + 1}-01-01")

        # 持仓状态
        cash = self.initial_cash
        positions: dict = {}  # {symbol: {buy_date, buy_price, shares, stop_loss, days, buy_reason}}
        # T+1 信号执行：存储上一日产生的Pending信号，{symbol: {wave, stop_loss}}
        pending_signals: dict = {}
        trades: list = []
        equity_curve: list = []

        total_score_signals = 0
        total_wavechan_filtered = 0
        total_buy_executed = 0
        total_bias_filtered = 0  # Bias 涨跌停过滤次数
        total_sell_delay = 0     # Bias 卖出延迟次数

        # 预热 WaveChan：先把 Score 候选股在回测首日之前的历史 K 线都喂入
        # 这样第一天的回测时 WaveChan 已有足够 K 线形成笔
        if self.trading_dates:
            first_date = self.trading_dates[0]
            # 获取 Score 所有候选股 symbol
            score_result = self.combo.score_strategy.get_signals(
                self.start_date, self.end_date
            )
            buy_candidates, _ = score_result
            if buy_candidates is not None and not buy_candidates.empty:
                warmup_symbols = buy_candidates['symbol'].unique().tolist()
                logging.info(f"🌡️ 预热 WaveChan：{len(warmup_symbols)} 只候选股到 {first_date}")
                self.combo.warmup(warmup_symbols, self.price_df, first_date)

        for year_start, year_end in years:
            logging.info(f"\n📅 {year_start} ~ {year_end}")

            # 按年获取 Score 候选股列表（只取 symbol，避免每天重复计算特征）
            year_buy_raw, _ = self.combo.score_strategy.get_signals(year_start, year_end)
            # {date_str -> [symbol, ...]}
            score_daily: dict = {}
            if year_buy_raw is not None and not year_buy_raw.empty:
                for ts, grp in year_buy_raw.groupby(level=0):
                    date_key = pd.Timestamp(ts).strftime('%Y-%m-%d')
                    score_daily[date_key] = grp['symbol'].tolist()

            # 过滤当年交易日
            year_dates = [d for d in self.trading_dates
                         if year_start <= d <= year_end]

            for date in year_dates:
                date_str = date

                # --- X. T+1 执行：使用 date_str 开盘价执行上一日产生的 Pending 信号 ---
                for sym in list(pending_signals.keys()):
                    sig_info = pending_signals[sym]
                    wave = sig_info['wave']
                    stop_loss = sig_info['stop_loss']
                    signal_date = sig_info.get('signal_date', date_str)

                    # ===== Bias 修正：T+1 执行日流动性检查 =====
                    # 涨跌停/停牌过滤 — 信号日本身涨停则次日高开，实际买不进
                    if self.bias_corrector:
                        can_buy, reason = self.bias_corrector.can_buy(sym, signal_date)
                        if not can_buy:
                            del pending_signals[sym]
                            total_bias_filtered += 1
                            logging.info(f"  🚫 Bias过滤 T+1买入 {sym}：{reason}")
                            continue

                    # 获取 date_str 开盘价作为实际买入价（T+1 执行价）
                    bar_today = get_bar(sym, date_str, self.price_df)
                    price = bar_today['open'] if bar_today else 0.0

                    if price <= 0:
                        del pending_signals[sym]
                        logging.warning(f"  ⚠️ T+1买入失败 {sym}：无法获取 {date_str} 开盘价")
                        continue

                    available = self.max_positions - len(positions)
                    if available <= 0:
                        break

                    pos_value = cash * self.position_size
                    shares = int(pos_value / price / 100) * 100
                    if shares < 100 or shares * price > cash:
                        del pending_signals[sym]
                        continue

                    positions[sym] = {
                        'buy_date': date_str,
                        'buy_price': price,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'days': 0,
                        'buy_reason': f"T+1:{wave.get('signal', '?')}:{wave.get('reason', '')}",
                        '_wave_state': wave.get('state', ''),
                    }
                    cash -= shares * price
                    total_buy_executed += 1
                    del pending_signals[sym]
                    logging.info(
                        f"  ✅ T+1买入 {sym} @ {price:.2f} "
                        f"[{wave.get('signal', '?')}][{wave.get('state', '?')}]"
                        f" {wave.get('reason', '')[:50]}"
                    )

                # --- A. 增量喂入 WaveEngine（当日新 K 线）---
                # 同时喂入候选股 + 持仓股（保持波浪状态连续）
                symbols_to_feed = set()
                # 持仓股一定喂
                symbols_to_feed.update(positions.keys())
                # 候选股：当天 Score 选出的
                for sym in score_daily.get(date_str, []):
                    symbols_to_feed.add(sym)

                for sym in symbols_to_feed:
                    bar = get_bar(sym, date_str, self.price_df)
                    if bar is None:
                        continue
                    self.combo._feed_history_to_engine(sym, self.price_df, date_str)

                # --- B. 获取当日 WaveChan 买点信号（仅持仓股需要实时状态）---
                # 已持仓的股票：当日 wave 状态用于监控止损/止盈
                for sym in list(positions.keys()):
                    wave = self.combo.check_wavechan_signal(sym, self.price_df, date_str)
                    positions[sym]['_wave_state'] = wave.get('state', '')

                # --- C. 处理当日 Score 候选股 + WaveChan 过滤 ---
                today_candidates = score_daily.get(date_str, [])
                total_score_signals += len(today_candidates)

                # ===== Bias 修正：涨跌停/停牌过滤 =====
                # 剔除信号日已涨停/跌停/停牌的股票（实盘无法买入）
                if self.bias_corrector and today_candidates:
                    candidates_df = pd.DataFrame({'symbol': today_candidates})
                    filtered_df = self.bias_corrector.filter_buy_signals(
                        date_str, candidates_df
                    )
                    if filtered_df is not None and not filtered_df.empty:
                        today_candidates = filtered_df['symbol'].tolist()
                    else:
                        today_candidates = []
                    # 日志：若涨跌停比例过高，记录以供分析
                    n_filtered = len(score_daily.get(date_str, [])) - len(today_candidates)
                    if n_filtered > 0:
                        total_bias_filtered += n_filtered
                        logging.debug(f"[BiasFilter] {date_str} Score候选过滤 {n_filtered} 只（涨跌停/停牌）")

                for sym in today_candidates:
                    if sym in positions:
                        continue

                    wave = self.combo.check_wavechan_signal(sym, self.price_df, date_str)
                    if not wave['has_signal']:
                        continue

                    # WaveChan 确认买点 → 存入 Pending（下一交易日执行）
                    total_wavechan_filtered += 1

                    stop_loss = float(wave.get('stop_loss')) if wave.get('stop_loss') else 0.0

                    pending_signals[sym] = {
                        'wave': wave,
                        'stop_loss': stop_loss,
                        'signal_date': date_str,  # 记录信号产生日
                    }
                    logging.info(
                        f"  ⏳ 信号记录 {sym} [{wave.get('signal', '?')}][{wave.get('state', '?')}]"
                        f" {wave.get('reason', '')[:50]} → T+1执行"
                    )

                # --- D. 处理持仓（止损/止盈/时间止损）---
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    pos['days'] += 1

                    bar = get_bar(sym, date_str, self.price_df)
                    if bar is None:
                        continue

                    close = bar['close']
                    ret = (close - pos['buy_price']) / pos['buy_price']
                    exit_reason = ''

                    # 止损
                    if close <= pos['stop_loss']:
                        exit_reason = 'STOP_LOSS'
                    # 止盈
                    elif ret >= self.profit_target_pct:
                        exit_reason = 'PROFIT_TAKE'
                    # 时间止损
                    elif pos['days'] >= self.max_hold_days:
                        exit_reason = 'TIME_EXIT'

                    if exit_reason:
                        # ===== Bias 修正：卖出流动性检查 =====
                        # 跌停日无法以合理价卖出，延迟卖出信号到下一日
                        if self.bias_corrector:
                            can_sell, sell_reason = self.bias_corrector.can_sell(sym, date_str)
                            if not can_sell:
                                # 跳过本次检查，不卖出（下一日继续检查）
                                total_sell_delay += 1
                                logging.debug(f"  ⏳ {sym} 卖出延迟：{sell_reason}")
                                continue

                        pnl = (close - pos['buy_price']) * pos['shares']
                        cash += pos['shares'] * close
                        trades.append({
                            'symbol': sym,
                            'buy_date': pos['buy_date'],
                            'sell_date': date_str,
                            'buy_price': pos['buy_price'],
                            'sell_price': close,
                            'shares': pos['shares'],
                            'pnl': pnl,
                            'return': ret,
                            'exit_reason': exit_reason,
                            'buy_reason': pos['buy_reason'],
                            'hold_days': pos['days'],
                        })
                        logging.info(
                            f"  📤 卖出 {sym} @ {close:.2f} ({exit_reason}) "
                            f"收益: {pnl:+,.0f} ({ret:.2%})"
                        )
                        del positions[sym]

                # 记录净值
                pos_value = sum(
                    get_bar(s, date_str, self.price_df)['close'] * positions[s]['shares']
                    for s in positions
                    if get_bar(s, date_str, self.price_df) is not None
                )
                equity = cash + pos_value
                equity_curve.append({
                    'date': date_str,
                    'cash': cash,
                    'position_value': pos_value,
                    'total_equity': equity,
                    'positions': len(positions),
                })

            # 最终清仓
            if year_end == self.end_date:
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    bar = get_bar(sym, year_end, self.price_df)
                    close = bar['close'] if bar else pos['buy_price']
                    pnl = (close - pos['buy_price']) * pos['shares']
                    cash += pos['shares'] * close
                    trades.append({
                        'symbol': sym,
                        'buy_date': pos['buy_date'],
                        'sell_date': year_end,
                        'buy_price': pos['buy_price'],
                        'sell_price': close,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'return': (close - pos['buy_price']) / pos['buy_price'],
                        'exit_reason': 'FINAL_EXIT',
                        'buy_reason': pos['buy_reason'],
                        'hold_days': pos['days'],
                    })
                    del positions[sym]

        final_equity = cash
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_equity': final_equity,
            'stats': {
                'total_score_signals': total_score_signals,
                'total_wavechan_filtered': total_wavechan_filtered,
                'total_buy_executed': total_buy_executed,
                'total_bias_filtered': total_bias_filtered,
                'total_sell_delay': total_sell_delay,
            }
        }

    def run(self) -> dict:
        """主入口"""
        self._load_data()
        return self._run()


# ============================================================
# 报告生成
# ============================================================

def generate_report(result: dict, config: dict):
    """打印回测报告"""
    trades = result.get('trades', [])
    equity_curve = result.get('equity_curve', [])
    stats = result.get('stats', {})
    final_equity = result.get('final_equity', config['initial_cash'])

    print("\n" + "=" * 60)
    print("Score + WaveChan V3 组合方向A 回测报告")
    print("=" * 60)

    print(f"\n📈 信号统计:")
    print(f"  Score 买入候选总数:   {stats.get('total_score_signals', 0)}")
    print(f"  WaveChan 过滤后:      {stats.get('total_wavechan_filtered', 0)}")
    print(f"  Bias涨跌停/停牌过滤:  {stats.get('total_bias_filtered', 0)}")
    print(f"  Bias卖出延迟:        {stats.get('total_sell_delay', 0)}")
    print(f"  实际执行买入:          {stats.get('total_buy_executed', 0)}")
    if stats.get('total_score_signals', 0) > 0:
        filter_rate = (1 - stats.get('total_wavechan_filtered', 0) /
                       stats.get('total_score_signals', 1)) * 100
        print(f"  WaveChan 过滤率:       {filter_rate:.1f}%")

    print(f"\n💰 收益概况:")
    print(f"  初始资金: {config['initial_cash']:,.0f}")
    print(f"  最终资金: {final_equity:,.0f}")
    total_return = (final_equity / config['initial_cash'] - 1) * 100
    print(f"  总收益率: {total_return:.2f}%")

    # 计算年化收益率
    start_dt = datetime.strptime(config['start_date'], '%Y-%m-%d')
    end_dt = datetime.strptime(config['end_date'], '%Y-%m-%d')
    years = (end_dt - start_dt).days / 365.25
    if years > 0:
        annualized = ((final_equity / config['initial_cash']) ** (1 / years) - 1) * 100
        print(f"  年化收益率: {annualized:.2f}%")

    if equity_curve:
        values = [e['total_equity'] for e in equity_curve]
        peak = max(values)
        dd = (min(values) - peak) / peak * 100 if peak > 0 else 0
        print(f"  最大回撤:   {dd:.2f}%")

    if trades:
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        print(f"\n📊 交易统计:")
        print(f"  总交易次数: {len(trades)}")
        print(f"  盈利次数:  {len(win_trades)} ({len(win_trades)/len(trades)*100:.1f}%)")
        print(f"  亏损次数:  {len(loss_trades)}")

        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            print(f"  平均盈利:   {avg_win:,.0f}")
            print(f"  平均亏损:   {avg_loss:,.0f}")
            if avg_loss != 0:
                print(f"  盈亏比:     {abs(avg_win/avg_loss):.2f}")

        # 按退出原因统计
        exit_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            exit_stats[t['exit_reason']]['count'] += 1
            exit_stats[t['exit_reason']]['pnl'] += t['pnl']
        print(f"\n🚪 退出原因:")
        for reason, s in sorted(exit_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {reason}: {s['count']}次, 盈亏 {s['pnl']:+,.0f}")

        # 按 WaveChan 信号统计
        sig_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            sig = t['buy_reason'].split(':')[0]
            sig_stats[sig]['count'] += 1
            sig_stats[sig]['pnl'] += t['pnl']
        print(f"\n🌊 按 WaveChan 信号:")
        for sig, s in sorted(sig_stats.items(), key=lambda x: -x[1]['count']):
            print(f"  {sig}: {s['count']}次, 盈亏 {s['pnl']:+,.0f}")

    # 保存结果
    out_dir = './backtestresult'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(out_dir, f'score_wavechan_combo_a_{ts}.json')

    save_data = {
        'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                   for k, v in config.items()},
        'stats': stats,
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'trades': trades,
        'equity_curve': equity_curve[-30:],  # 只保留最后30天
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 结果已保存: {out_file}")


# ============================================================
# 主程序
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Score + WaveChan V3 组合方向A 回测')
    parser.add_argument('--start', type=str, default='2025-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2025-12-31', help='结束日期')
    parser.add_argument('--cash', type=float, default=500_000, help='初始资金')
    parser.add_argument('--max-pos', type=int, default=5, help='最大持仓数')
    parser.add_argument('--top-n', type=int, default=5, help='Score 每日选股数')
    parser.add_argument('--stop-loss', type=float, default=0.08, help='止损比例')
    parser.add_argument('--profit-target', type=float, default=0.20, help='止盈比例')
    parser.add_argument('--max-hold', type=int, default=20, help='最大持仓天数')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    config = {
        'start_date': args.start,
        'end_date': args.end,
        'initial_cash': args.cash,
        'max_positions': args.max_pos,
        'top_n': args.top_n,
        'stop_loss_pct': args.stop_loss,
        'profit_target_pct': args.profit_target,
        'max_hold_days': args.max_hold,
        'wave_cache_dir': '/tmp/wavechan_combo_a_bt',
    }

    print("=" * 60)
    print("Score + WaveChan V3 组合方向A 回测")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")

    start_time = time.time()
    try:
        tester = SimpleComboBacktester(**config)
        result = tester.run()
        elapsed = time.time() - start_time
        generate_report(result, config)
        print(f"\n⏱️ 总耗时: {elapsed:.1f}秒")
    except Exception as e:
        logging.error(f"回测失败: {e}", exc_info=True)


if __name__ == '__main__':
    main()
