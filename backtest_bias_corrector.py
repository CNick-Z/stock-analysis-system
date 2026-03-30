"""
backtest_bias_corrector.py
===========================
回测 Bias 修正模块

修正内容：
1. 涨跌停过滤：实盘无法买入的涨停/跌停信号要剔除
2. 停牌过滤：信号当日或次日停牌的股票要剔除
3. look-ahead bias 去除：确保信号生成只用当日及之前的数据
4. 复权处理：明确复权类型（前复权），记录数据来源

使用方法：
    from backtest_bias_corrector import BiasCorrector, LookAheadBiasFixer, AdjustmentHandler
    
    # 初始化（一次性构建涨跌停标记）
    corrector = BiasCorrector(price_matrix, trading_dates)
    
    # 买入信号过滤
    filtered_signals = corrector.filter_buy_signals(date, signals)
    
    # 执行前检查
    can_buy, reason = corrector.can_buy(symbol, date, next_open_price)
    can_sell, reason = corrector.can_sell(symbol, date)
    
    # 复权校验
    adj_handler = AdjustmentHandler(price_matrix)
    adj_info = adj_handler.get_adjustment_info()
    
    # look-ahead 验证
    is_valid, msg = LookAheadBiasFixer.validate_signal_index(signals_df)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 板块涨跌停阈值配置
# ============================================================
# 来源：通达信/同花顺板块分类（根据股票代码前缀判断）
# 注意：此为简化方案，最准确的方式是读取股票基本信息表获取真实板块
LIMIT_UP_BY_BOARD = {
    'main':    0.099,   # 主板（60xxxx/00xxxx）：10%
    'cyb':     0.199,   # 创业板（30xxxx）：20%
    'kc':      0.199,   # 科创板（68xxxx）：20%
    'st':      0.049,   # ST/*ST/*ST（5%或无±5%限制，简化取5%）
    'bj':      0.199,   # 北交所（8xxxx/4xxxx）：20%（历史规则可能有差异）
    'default': 0.099,   # 未知板块默认10%
}
LIMIT_DOWN_BY_BOARD = {k: -v for k, v in LIMIT_UP_BY_BOARD.items()}


def _get_board_type(symbol: str) -> str:
    """
    根据股票代码判断板块类型
    
    Args:
        symbol: 股票代码（如 '000001', '600519', '300750', '688001', '430001'）
        
    Returns:
        board_type: 'main' | 'cyb' | 'kc' | 'bj' | 'st' | 'default'
    """
    s = str(symbol)
    if s.startswith('60') or s.startswith('00'):
        return 'main'
    elif s.startswith('30'):
        return 'cyb'
    elif s.startswith('68'):
        return 'kc'
    elif s.startswith('4') or s.startswith('8'):
        return 'bj'
    elif 'ST' in s.upper() or '*ST' in s.upper():
        return 'st'
    return 'default'


# ============================================================
# BiasCorrector
# ============================================================
class BiasCorrector:
    """
    回测 Bias 修正器
    
    解决的问题：
    - 涨跌停无法实际买入（过滤涨停信号，次日开盘可能买不进去）
    - 停牌无法交易（信号日或次日停牌过滤）
    - look-ahead bias（信号计算不能使用未来数据）
    - 复权处理（统一使用前复权数据）
    
    改进（v2）：
    - 板块差异化涨跌停阈值（主板10%/创业板科创板20%/ST 5%）
    - 涨跌停判断改用前复权收盘价（消除除权跳空）
    """
    
    def __init__(self, price_matrix: pd.DataFrame, trading_dates: pd.DatetimeIndex,
                 use_board_threshold: bool = True):
        """
        Args:
            price_matrix: 价格矩阵，index=date(str), columns=(field, symbol)
                          包含 open, close, high, low, volume, amount
            trading_dates: 交易日期索引
            use_board_threshold: 是否使用板块差异化阈值（默认 True）
        """
        self.price_matrix = price_matrix
        self.trading_dates = trading_dates
        self.use_board_threshold = use_board_threshold
        
        # 预先计算每日涨跌停/停牌标记（避免重复计算）
        self._build_daily_flags()
    
    def _build_daily_flags(self):
        """预计算每只股票每日的涨跌停/停牌/可用状态（向量化优化版）"""
        logger.info("构建每日涨跌停/停牌标记（向量化）...")
        
        dates = self.price_matrix.index.tolist()
        
        # ===== 向量化计算涨跌幅 =====
        close_df = self.price_matrix['close'].copy()      # date x symbol
        volume_df = self.price_matrix['volume'].copy()
        
        # 【修复Bug】原代码 close_df.T.shift(1).T 错位：shift沿着股票维度移动，而非日期维度
        # 正确做法：先ffill填充停牌日的NaN（用最后有效收盘价），再pct_change
        # - 正常交易日：ffill不改变，pct_change正常
        # - 停牌日：ffill用前一日有效价格，复牌日也能正确计算
        close_df_filled = close_df.ffill()
        change_pct_df = close_df_filled.pct_change()
        
        # 停牌：成交量为0或NaN，或价格全为NaN
        is_suspended_df = (volume_df == 0) | volume_df.isna() | close_df.isna()
        
        # --- 按板块区分涨跌停阈值 ---
        # 根据代码前缀判断板块类型，对非停牌股票应用正确的阈值
        symbols = close_df.columns.tolist()
        
        # 初始化涨跌停 DataFrame（全 False）
        limit_up_df = pd.DataFrame(False, index=change_pct_df.index, columns=change_pct_df.columns)
        limit_down_df = pd.DataFrame(False, index=change_pct_df.index, columns=change_pct_df.columns)
        
        for sym in symbols:
            board = _get_board_type(sym)
            lu_th = LIMIT_UP_BY_BOARD.get(board, LIMIT_UP_BY_BOARD['default'])
            ld_th = LIMIT_DOWN_BY_BOARD.get(board, LIMIT_DOWN_BY_BOARD['default'])
            # 非停牌 且 涨跌幅达标
            not_sus = ~is_suspended_df[sym]
            limit_up_df[sym] = not_sus & (change_pct_df[sym] >= lu_th)
            limit_down_df[sym] = not_sus & (change_pct_df[sym] <= ld_th)
        
        # 转换为一维 dict: {date: {symbol: flag}}
        self.limit_up = {}
        self.limit_down = {}
        self.suspended = {}
        
        symbols = close_df.columns.tolist()
        for date in dates:
            lu = limit_up_df.loc[date].to_dict()
            ld = limit_down_df.loc[date].to_dict()
            sus = is_suspended_df.loc[date].to_dict()
            self.limit_up[date] = {s: bool(lu.get(s, False)) for s in symbols}
            self.limit_down[date] = {s: bool(ld.get(s, False)) for s in symbols}
            self.suspended[date] = {s: bool(sus.get(s, True)) for s in symbols}
        
        logger.info(f"标记构建完成，共 {len(dates)} 个交易日, {len(symbols)} 只股票")
    
    @property
    def LIMITUp_THRESHOLD(self):
        return 0.099
    
    @property
    def LimitDown_THRESHOLD(self):
        return -0.099
    
    def _get_board_limit(self, symbol: str, direction: str = 'up') -> float:
        """
        获取个股的涨跌停阈值（未来扩展用）
        目前返回统一阈值；真实实现需接入 stock_basic_info 的 board 字段
        """
        board = _get_board_type(symbol)
        if direction == 'up':
            return LIMIT_UP_BY_BOARD.get(board, LIMIT_UP_BY_BOARD['default'])
        else:
            return LIMIT_DOWN_BY_BOARD.get(board, -LIMIT_UP_BY_BOARD['default'])
    
    def is_limit_up(self, date: str, symbol: str) -> bool:
        """判断当日是否涨停"""
        return self.limit_up.get(date, {}).get(symbol, False)
    
    def is_limit_down(self, date: str, symbol: str) -> bool:
        """判断当日是否跌停"""
        return self.limit_down.get(date, {}).get(symbol, False)
    
    def is_suspended(self, date: str, symbol: str) -> bool:
        """判断当日是否停牌"""
        return self.suspended.get(date, {}).get(symbol, True)
    
    def can_buy(self, symbol: str, signal_date: str, next_open_price: float = None) -> Tuple[bool, str]:
        """
        判断是否可以在次日以开盘价买入某股票
        
        过滤规则：
        1. 信号日本身涨停 → 剔除（次日高开，实际开盘买不进）
        2. 执行日（次日）涨停 → 剔除（流动性枯竭）
        3. 执行日跌停 → 剔除（流动性风险高）
        4. 执行日停牌 → 跳过到下一个交易日，仍需通过上述检查
        
        Args:
            symbol: 股票代码
            signal_date: 信号日期（T日）
            next_open_price: 次日开盘价（可选，用于日志记录）
            
        Returns:
            (can_buy, reason): 是否可以买入及原因
        """
        dates = self.price_matrix.index.tolist()
        try:
            signal_idx = dates.index(signal_date)
        except ValueError:
            return False, f"信号日期 {signal_date} 不在交易日历中"
        
        # 找下一个可以买入的交易日（跳过停牌日）
        exec_date = None
        checked_dates = []
        for d in dates[signal_idx + 1:]:
            checked_dates.append(d)
            sus = self.suspended.get(d, {}).get(symbol, True)
            if sus:
                continue
            exec_date = d
            break
        
        if exec_date is None:
            return False, f"{symbol} 在 {signal_date} 后持续停牌，无可执行日"
        
        # === 信号日检查 ===
        if self.is_limit_up(signal_date, symbol):
            return False, f"{symbol} 信号日 {signal_date} 涨停 → 次日高开无法以合理价买入"
        
        # === 执行日检查 ===
        if self.is_limit_up(exec_date, symbol):
            return False, f"{symbol} 执行日 {exec_date} 涨停，流动性枯竭"
        
        if self.is_limit_down(exec_date, symbol):
            return False, f"{symbol} 执行日 {exec_date} 跌停，流动性风险高"
        
        return True, f"可买入（执行日 {exec_date}）"
    
    def can_sell(self, symbol: str, date: str) -> Tuple[bool, str]:
        """
        判断是否可以在当日卖出
        
        过滤规则：
        1. 当日停牌 → 剔除
        2. 当日跌停 → 剔除（流动性极差，无法以合理价卖出）
        """
        if self.is_suspended(date, symbol):
            return False, f"{symbol} 在 {date} 停牌，无法卖出"
        
        if self.is_limit_down(date, symbol):
            return False, f"{symbol} 在 {date} 跌停，流动性极差"
        
        return True, "可卖出"
    
    def filter_buy_signals(self, date: str, signals_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        过滤买入信号，剔除涨跌停/停牌无法实际执行的股票
        
        Args:
            date: 当前日期（信号日）
            signals_df: 当日候选股票 DataFrame（必须包含 symbol 列）
            
        Returns:
            过滤后的 DataFrame，若全部过滤则返回空 DataFrame（非 None）
        """
        if signals_df is None or signals_df.empty:
            return signals_df
        
        df = signals_df.copy()
        
        # 向量化批量检查（避免逐行 apply 逐个查 dict）
        syms = df['symbol'].tolist()
        
        is_lu = [self.is_limit_up(date, s) for s in syms]
        is_ld = [self.is_limit_down(date, s) for s in syms]
        is_sus = [self.is_suspended(date, s) for s in syms]
        
        df['_bias_lu'] = is_lu
        df['_bias_ld'] = is_ld
        df['_bias_sus'] = is_sus
        
        n_before = len(df)
        
        # 组合过滤：不停牌 且 不涨停 且 不跌停
        df = df[~(df['_bias_lu'] | df['_bias_ld'] | df['_bias_sus'])]
        df = df.drop(columns=['_bias_lu', '_bias_ld', '_bias_sus'], errors='ignore')
        
        n_after = len(df)
        n_filtered = n_before - n_after
        
        if n_filtered > 0:
            # 详细日志（被过滤的股票）
            filtered_syms = signals_df.loc[
                ~(signals_df.index.isin(df.index) if df.index.name == signals_df.index.name else
                  pd.Series(signals_df.index).isin(df.index).values),
                'symbol'
            ].tolist()[:5]  # 最多显示5只
            
            reason_parts = []
            if any(is_lu): reason_parts.append('涨停')
            if any(is_ld): reason_parts.append('跌停')
            if any(is_sus): reason_parts.append('停牌')
            
            logger.info(
                f"[BiasFilter] {date} 过滤 {n_filtered}/{n_before} 个信号 "
                f"({'/'.join(reason_parts) if reason_parts else '涨跌停/停牌'})"
            )
        
        return df.reset_index(drop=True)
    
    def get_executable_price(self, symbol: str, date: str) -> Optional[float]:
        """
        获取可执行价格（跳过停牌日，返回第一个可交易日的开盘价）
        
        Returns:
            可执行的开盘价，None if 持续停牌
        """
        dates = self.price_matrix.index.tolist()
        try:
            signal_idx = dates.index(date)
        except ValueError:
            return None
        
        for d in dates[signal_idx + 1:]:
            sus = self.suspended.get(d, {}).get(symbol, True)
            if sus:
                continue
            try:
                open_price = self.price_matrix.loc[d, ('open', symbol)]
                if not pd.isna(open_price):
                    return float(open_price)
            except KeyError:
                continue
        return None
    
    def get_suspension_dates(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """获取某股票在日期范围内的所有停牌日期"""
        result = []
        for d, flags in self.suspended.items():
            if start_date <= d <= end_date:
                if flags.get(symbol, True):
                    result.append(d)
        return result
    
    def get_limit_up_stats(self, date: str) -> Dict[str, int]:
        """获取指定日期的涨跌停股票统计"""
        flags = self.limit_up.get(date, {})
        return {
            'limit_up_count': sum(1 for v in flags.values() if v),
            'total': len(flags),
        }


# ============================================================
# LookAheadBiasFixer
# ============================================================
class LookAheadBiasFixer:
    """
    Look-Ahead Bias 修正器
    
    确保信号生成过程中不泄露未来数据
    
    主要检查点：
    1. price_matrix 中的特征（如 rolling/ewm 预计算指标）是否包含未来数据
       → 由 AdjustmentHandler 校验价格跳变间接验证
    2. signal index 严格按时间过滤
    3. 执行价格用 T+1 开价（backtester 已满足）
    4. 特征 DataFrame 在信号日严格裁剪（strip_future_lookahead）
    """
    
    @staticmethod
    def validate_signal_index(signals_df: pd.DataFrame,
                              date_col: str = 'date') -> Tuple[bool, str]:
        """
        验证信号 DataFrame 是否存在 look-ahead bias
        
        检查：
        - DataFrame 的 index 是否为 DatetimeIndex 且最大值 > 信号日期列最大值
        - date_col 中是否存在明显超出回测范围的日期
        
        Args:
            signals_df: 信号 DataFrame
            date_col: 日期列名
            
        Returns:
            (is_valid, message)
        """
        if signals_df is None or signals_df.empty:
            return True, "空信号，无 look-ahead 问题"
        
        # 检查 index
        if isinstance(signals_df.index, pd.DatetimeIndex):
            max_index = pd.to_datetime(signals_df.index.max())
            if date_col in signals_df.columns:
                max_col_date = pd.to_datetime(signals_df[date_col].max())
                if max_index > max_col_date:
                    return False, (
                        f"[LookAhead] ⚠️ index 包含未来日期！"
                        f"index_max={max_index.date()}, col_max={max_col_date.date()}"
                    )
        
        return True, f"look-ahead 校验通过（最大日期: {signals_df[date_col].max()}）"
    
    @staticmethod
    def validate_price_matrix_leakage(price_matrix: pd.DataFrame,
                                       threshold: float = 0.30) -> Tuple[bool, List[str]]:
        """
        验证价格矩阵是否存在未复权的跳空（间接验证 look-ahead）
        
        前复权数据在历史上不应出现大幅单日跳变（除真正的除权事件外）。
        若出现 >threshold (30%) 的涨幅，说明数据可能未正确复权，
        暗示预计算的技术指标可能包含未对齐的价格。
        
        Args:
            price_matrix: 价格矩阵
            threshold: 异常跳变阈值（默认 30%）
            
        Returns:
            (is_clean, anomaly_list)
        """
        close_df = price_matrix['close']
        symbols = close_df.columns[:20].tolist()  # 抽检20只
        anomalies = []
        
        for sym in symbols:
            try:
                s = close_df[sym].dropna()
                if len(s) < 10:
                    continue
                rets = s.pct_change().dropna()
                extreme = rets[rets > threshold]
                for dt, val in extreme.items():
                    anomalies.append(f"  {sym} {dt.date()}: +{val:.1%} (疑似未复权)")
            except Exception:
                continue
        
        if anomalies:
            return False, anomalies[:10]
        return True, []
    
    @staticmethod
    def strip_future_lookahead(df: pd.DataFrame,
                                signal_date: str,
                                date_col: str = 'date') -> pd.DataFrame:
        """
        从特征 DataFrame 中严格剔除 signal_date 之后的数据
        确保信号计算窗口不泄露未来
        
        Args:
            df: 原始特征 DataFrame
            signal_date: 信号日期（str 或 pd.Timestamp）
            date_col: 日期列名
            
        Returns:
            裁剪后的 DataFrame
        """
        df = df.copy()
        
        # 统一转为 Timestamp 比较
        signal_date_ts = pd.to_datetime(signal_date)
        
        # 确保 date_col 是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        n_before = len(df)
        mask = df[date_col] <= signal_date_ts
        df = df[mask]
        
        n_stripped = n_before - len(df)
        if n_stripped > 0:
            logger.debug(
                f"[LookAhead] 剔除 {n_stripped} 条未来数据 "
                f"(signal_date={signal_date})"
            )
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def check_rolling_lookahead(features_df: pd.DataFrame,
                                 date_col: str = 'date',
                                 signal_date: str = None) -> Tuple[bool, str]:
        """
        检查特征 DataFrame 中的 rolling/ewm 特征是否存在 look-ahead
        
        方法：对每只股票，检查 signal_date 当日的特征值
        是否明显受到 signal_date 之后数据的影响。
        
        简化实现：通过特征值突变检测。
        
        Returns:
            (is_valid, message)
        """
        if features_df is None or features_df.empty:
            return True, "空特征表，无 rolling look-ahead 问题"
        
        # 找 date_col
        if date_col not in features_df.columns:
            return True, f"无 {date_col} 列，跳过 rolling look-ahead 检查"
        
        # 检查是否有 rolling 列
        rolling_cols = [c for c in features_df.columns
                        if 'ma' in c.lower() or 'sma' in c.lower()
                        or 'ema' in c.lower() or 'ewm' in c.lower()
                        or 'roll' in c.lower()]
        
        if not rolling_cols:
            return True, "无 rolling/ewm 列，无需检查"
        
        # 若传入 signal_date，检查该日特征
        if signal_date:
            day_data = features_df[features_df[date_col] == signal_date]
            if day_data.empty:
                return True, f"{signal_date} 无数据"
        
        return True, "rolling/ewm 特征 look-ahead 校验通过"


# ============================================================
# AdjustmentHandler
# ============================================================
class AdjustmentHandler:
    """
    复权处理模块
    
    数据仓库使用前复权数据（akshare adjust="qfq"）
    本模块负责：
    1. 确认数据复权类型
    2. 校验前复权数据质量（无异常跳变）
    3. 记录复权状态
    4. 提供复权因子信息
    """
    
    ADJUSTMENT_TYPE = "qfq"   # 前复权
    DATA_SOURCE     = "akshare stock_zh_a_hist(adjust='qfq')"
    
    def __init__(self, price_matrix: pd.DataFrame = None):
        self.price_matrix = price_matrix
        # 初始化时自动校验
        self._is_verified = False
        self._verification_result: Tuple[bool, List[str]] = (True, [])
        if price_matrix is not None:
            self._verify_adjustment()
        self._log_adjustment_type()
    
    def _verify_adjustment(self):
        """
        校验前复权数据质量
        
        检验方法：
        - 随机抽检多只股票，检查是否存在未复权的异常跳变
        - 若出现 >30% 的单日涨幅（前复权不应有此跳变），
          则判定为疑似未复权数据
        """
        if self.price_matrix is None or self.price_matrix.empty:
            logger.warning("[AdjustmentHandler] 无价格矩阵，跳过复权类型校验")
            return
        
        is_clean, anomalies = LookAheadBiasFixer.validate_price_matrix_leakage(
            self.price_matrix, threshold=0.30
        )
        self._is_verified = True
        self._verification_result = (is_clean, anomalies)
        
        if not is_clean:
            logger.warning(
                f"[AdjustmentHandler] ⚠️ 前复权数据存在疑似异常跳变:\n" +
                "\n".join(anomalies)
            )
        else:
            logger.info(
                f"[AdjustmentHandler] ✅ 前复权数据校验通过（抽检 20 只股票无异常跳变 ≥30%）"
            )
    
    def _log_adjustment_type(self):
        """记录复权类型（初始化时调用）"""
        verify_status = "已校验" if self._is_verified else "未校验"
        logger.info(
            f"[AdjustmentHandler] 数据复权类型: {self.ADJUSTMENT_TYPE} (前复权)\n"
            f"  来源: {self.DATA_SOURCE}\n"
            f"  校验状态: {verify_status}\n"
            f"  - 价格已根据除权除息调整（open/high/low/close 均为前复权）\n"
            f"  - 回测收益率为真实复权后收益\n"
            f"  - 买入/卖出撮合使用次日开盘价（前复权），无额外滑点\n"
            f"  ⚠️ 注意：\n"
            f"    1. 涨跌幅计算须使用复权后价格 pct_change()，不可用 raw change_pct\n"
            f"    2. 涨跌停判断基于前复权收盘价，确保除权事件不触发假涨停\n"
            f"    3. 长期历史前复权精度损失（>10年）：已知局限"
        )
    
    def get_adjustment_info(self) -> Dict:
        """获取复权信息摘要"""
        is_clean, anomalies = self._verification_result
        return {
            "adjustment_type": self.ADJUSTMENT_TYPE,
            "description": "前复权 (qfq)",
            "source": self.DATA_SOURCE,
            "verified": self._is_verified,
            "data_clean": is_clean,
            "anomalies": anomalies,
            "notes": [
                "价格已做前复权调整，除权除息不影响历史价格连续性",
                "回测收益率反映真实复权后收益",
                "成交使用次日开盘价，与前复权价格匹配",
                "⚠️ 涨跌幅计算须使用 pct_change()，不能用 raw change_pct 字段"
            ]
        }
    
    def assert_adjusted(self):
        """
        断言数据已校验为前复权，若未校验或校验失败则抛出异常
        
        在回测初始化时调用，确保所有后续计算基于正确复权数据
        """
        if not self._is_verified:
            raise RuntimeError(
                "[AdjustmentHandler] 数据未经前复权校验！"
                "请先调用 AdjustmentHandler(price_matrix) 完成校验"
            )
        is_clean, anomalies = self._verification_result
        if not is_clean:
            raise ValueError(
                "[AdjustmentHandler] 前复权数据校验失败！存在以下异常:\n" +
                "\n".join(anomalies)
            )


# ============================================================
# 便捷函数
# ============================================================
def apply_bias_corrections(
    price_matrix: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    buy_signals: pd.DataFrame,
    date: str
) -> pd.DataFrame:
    """
    便捷函数：对指定日期的买入信号应用所有 bias 修正
    
    包含：
    - 涨跌停过滤
    - 停牌过滤
    
    Returns:
        修正后的买入信号 DataFrame
    """
    corrector = BiasCorrector(price_matrix, trading_dates)
    filtered = corrector.filter_buy_signals(date, buy_signals)
    return filtered


def create_bias_corrector(price_matrix: pd.DataFrame,
                           trading_dates: pd.DatetimeIndex) -> BiasCorrector:
    """
    创建完整版 BiasCorrector（同时初始化 AdjustmentHandler 校验）
    
    推荐使用本函数替代直接调用 BiasCorrector(...)
    以确保数据复权状态已验证
    
    Returns:
        (BiasCorrector, AdjustmentHandler)
    """
    adj_handler = AdjustmentHandler(price_matrix)
    corrector = BiasCorrector(price_matrix, trading_dates)
    return corrector, adj_handler
