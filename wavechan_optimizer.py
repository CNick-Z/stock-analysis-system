#!/usr/bin/env python3
"""
WaveChan 选股策略 Optuna 超参优化（向量化版）

目标：最大化夏普比率
优化：2024-2025年数据

【缓存架构】
  L1 历史归档层（Parquet，按年分区）- 2021-2024，永不重建
  L2 热数据层（Parquet，按年分区）- 2025+，当年重建
  L3 参数缓存层（SQLite）- (算法版本+参数hash) → 结果，LRU 100组

用法：
    python3 wavechan_optimizer.py --trials 100
    python3 wavechan_optimizer.py --trials 100 --use-l3  # 启用L3参数缓存
    python3 wavechan_optimizer.py --cache-status          # 查看缓存状态
"""

import argparse
import sys
import os
import time
import json
import optuna
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 默认使用三层缓存（向后兼容：优先读新缓存，找不到则读旧缓存）
USE_LAYERED_CACHE = True
FALLBACK_CACHE = "/data/warehouse/wavechan/l1_history"
CACHE_STATUS_CMD = False


def load_and_prepare_cache(start_year=2024, end_year=2025, use_l3=False, algo_version="v3.0"):
    """
    加载并准备缓存数据
    - 优先使用三层缓存（L1+L2）
    - 找不到时 fallback 到旧单文件缓存
    """
    # 优先尝试三层缓存
    if USE_LAYERED_CACHE:
        try:
            from utils.wavechan_cache import WaveChanCacheManager
            cm = WaveChanCacheManager()
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            df = cm.load(start_date, end_date, algo_version=algo_version)
            if not df.empty:
                print(f"  [L1/L2] 加载: {len(df):,} 行 | 股票: {df['symbol'].nunique()} | 交易日: {df['date'].nunique()}")
                # 计算未来收益（向量化）
                df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
                df['future_close_5d'] = df.groupby('symbol')['close'].shift(-5)
                df['return_5d'] = (df['future_close_5d'] - df['close']) / df['close']
                df = df.dropna(subset=['return_5d'])
                print(f"  return_5d计算后: {len(df):,} 行")
                return df, cm
        except Exception as e:
            print(f"  [L1/L2] 加载失败，回退到旧缓存: {e}")

    # Fallback 到旧单文件缓存
    cache_path = FALLBACK_CACHE
    if not os.path.exists(cache_path):
        print(f"❌ 缓存不存在: {cache_path}")
        print(f"   请先运行: python3 wavechan_cache_migrate.py")
        sys.exit(1)

    print(f"加载缓存(Fallback): {cache_path}")
    df = pd.read_parquet(cache_path)
    print(f"  记录: {len(df):,} 行 | 股票: {df['symbol'].nunique()} | 交易日: {df['date'].nunique()}")

    # 过滤年份
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f"  过滤 {start_year}-{end_year}: {len(df):,} 行")

    # 计算未来收益（向量化）
    df['future_close_5d'] = df.groupby('symbol')['close'].shift(-5)
    df['return_5d'] = (df['future_close_5d'] - df['close']) / df['close']
    df = df.dropna(subset=['return_5d'])

    return df, None


def run_backtest_vectorized(df, w_signal, w_struct, w_momentum, w_chan,
                            threshold, holding_days=5, top_n=10, position_limit=5,
                            commission_rate=0.0003, stamp_tax=0.001, slippage=0.001):
    """
    向量化回测 - 加入真实成本 + T+1持仓跟踪
    """
    # 归一化权重
    total = w_signal + w_struct + w_momentum + w_chan
    w_s = w_signal / total
    w_st = w_struct / total
    w_m = w_momentum / total
    # 信号类型+状态过滤（只取有效的买入信号）
    BUY_SIGNALS = {'C_BUY', 'W2_BUY', 'W4_BUY'}
    VALID_STATUSES = {'alert', 'confirmed'}
    valid_signals = df[
        (df['signal_type'].isin(BUY_SIGNALS)) &
        (df['signal_status'].isin(VALID_STATUSES))
    ].copy()

    if len(valid_signals) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'win_rate': 0, 'n_trades': 0, 'annual_return': 0, 'n_days': 0}

    # 直接用预计算的total_score排序（已经是各组分求和，不需要重加权）
    # 权重参数改为对total_score的缩放因子（0.5-2.0范围更合理）
    score_multiplier = 1.0  # 固定为1，直接用total_score
    signals = valid_signals[valid_signals['total_score'] * score_multiplier >= threshold].copy()

    if len(signals) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'win_rate': 0, 'n_trades': 0, 'annual_return': 0, 'n_days': 0}

    # 每日选top_n，按分数排序
    signals = signals.sort_values(['date', 'total_score'], ascending=[True, False])
    signals['rank'] = signals.groupby('date')['total_score'].rank(method='first', ascending=False)
    top_signals = signals[signals['rank'] <= top_n].copy()

    if len(top_signals) == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'win_rate': 0, 'n_trades': 0, 'annual_return': 0, 'n_days': 0}

    # T+1持仓跟踪
    hold_until = {}  # symbol -> date
    trades = []
    daily_pnl = {}
    dates = sorted(top_signals['date'].unique())

    for date in dates:
        daily_signals = top_signals[top_signals['date'] == date]

        # 清理已释放的持仓
        expired = [s for s, exp_d in hold_until.items() if date >= exp_d]
        for s in expired:
            del hold_until[s]

        # 当日新开仓的股票（受持仓上限限制）
        new_positions = []
        for _, row in daily_signals.iterrows():
            if len(hold_until) >= position_limit:
                break  # 达到持仓上限，不再开新仓
            sym = row['symbol']
            if sym not in hold_until:
                new_positions.append(row)
                idx = dates.index(date)
                if idx + holding_days < len(dates):
                    expire_date = dates[idx + holding_days]
                else:
                    expire_date = dates[-1]
                hold_until[sym] = expire_date

        if not new_positions:
            daily_pnl[date] = 0.0
            continue

        returns = [r['return_5d'] for r in new_positions]
        buy_cost = commission_rate + slippage
        sell_cost = commission_rate + stamp_tax + slippage
        total_cost = buy_cost + sell_cost
        daily_cost = total_cost / holding_days
        avg_return = np.mean(returns) - daily_cost

        daily_pnl[date] = avg_return

        for r in new_positions:
            trades.append({
                'date': date,
                'symbol': r['symbol'],
                'return': r['return_5d'] - daily_cost
            })

    if not daily_pnl:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0, 'win_rate': 0, 'n_trades': 0, 'annual_return': 0, 'n_days': 0}

    daily_returns = pd.Series(daily_pnl)
    total_return = (1 + daily_returns).prod() - 1
    n_days = max(len(daily_returns), 1)
    annual_return = (1 + total_return) ** (252 / n_days) - 1

    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (daily_returns > 0).mean()
    n_trades = len(trades)

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'n_days': n_days
    }


def objective(trial, df, cache_mgr=None, algo_version="v3.0", use_l3=False):
    """Optuna 目标函数"""
    # 权重参数
    w_signal = trial.suggest_float('w_signal', 0.1, 1.0)
    w_struct = trial.suggest_float('w_struct', 0.1, 1.0)
    w_momentum = trial.suggest_float('w_momentum', 0.1, 1.0)
    w_chan = trial.suggest_float('w_chan', 0.1, 1.0)

    # 阈值
    threshold = trial.suggest_float('threshold', 20, 70)

    # Top N
    top_n = trial.suggest_int('top_n', 5, 30)

    # 构造 params 用于 L3 缓存
    params = {
        'w_signal': w_signal,
        'w_struct': w_struct,
        'w_momentum': w_momentum,
        'w_chan': w_chan,
        'threshold': threshold,
        'top_n': top_n,
    }

    # L3 缓存查询
    if use_l3 and cache_mgr is not None:
        cached = cache_mgr.get_cached_backtest_result(algo_version, params)
        if cached is not None:
            return cached['sharpe']

    result = run_backtest_vectorized(
        df, w_signal, w_struct, w_momentum, w_chan,
        threshold, holding_days=5, top_n=top_n, position_limit=5
    )

    # L3 缓存写入
    if use_l3 and cache_mgr is not None and result['n_trades'] >= 30:
        cache_mgr.put_cached_backtest_result(algo_version, params, result)

    # 至少30笔交易才有效
    if result['n_trades'] < 30:
        return 0

    return result['sharpe']


def main():
    global USE_LAYERED_CACHE, CACHE_STATUS_CMD

    parser = argparse.ArgumentParser(description='WaveChan Optuna 优化')
    parser.add_argument('--trials', type=int, default=100, help='优化轮数')
    parser.add_argument('--start-year', type=int, default=2024, help='回测开始年份')
    parser.add_argument('--end-year', type=int, default=2025, help='回测结束年份')
    parser.add_argument('--use-l3', action='store_true', help='启用L3参数缓存')
    parser.add_argument('--no-layered-cache', action='store_true', help='禁用三层缓存，使用旧单文件')
    parser.add_argument('--cache-status', action='store_true', help='仅显示缓存状态')
    parser.add_argument('--algo-version', type=str, default='v3.0', help='算法版本标识')
    args = parser.parse_args()

    # 仅显示缓存状态
    if args.cache_status:
        from utils.wavechan_cache import WaveChanCacheManager
        cm = WaveChanCacheManager()
        status = cm.status()
        print("=" * 60)
        print("WaveChan 缓存状态")
        print("=" * 60)
        print(f"\nL1 历史归档（2021-2024）:")
        for k, v in status['l1_partitions'].items():
            print(f"  {k}: {v['rows']:,} 行, {v['size_mb']:.1f} MB")
        print(f"\nL2 热数据（2025+）:")
        for k, v in status['l2_partitions'].items():
            print(f"  {k}: {v['rows']:,} 行, {v['size_mb']:.1f} MB")
        print(f"\nL2 总大小: {status['l2_size_mb']:.1f} MB")
        print(f"\nL3 参数缓存: {status['l3_count']} 条, 版本: {status['l3_versions']}")
        print()
        return

    if args.no_layered_cache:
        USE_LAYERED_CACHE = False

    print("=" * 60)
    print("WaveChan 选股策略优化（向量化版）")
    print(f"三层缓存: {'启用' if USE_LAYERED_CACHE else '禁用'}")
    print(f"L3 参数缓存: {'启用' if args.use_l3 else '禁用'}")
    print("=" * 60)

    # 加载数据
    df, cache_mgr = load_and_prepare_cache(
        start_year=args.start_year,
        end_year=args.end_year,
        use_l3=args.use_l3,
        algo_version=args.algo_version,
    )

    print(f"\n回测期间: {pd.to_datetime(df['date']).min().date()} ~ {pd.to_datetime(df['date']).max().date()}")
    print(f"记录数: {len(df):,} | 股票数: {df['symbol'].nunique()}")

    # 预计算每日收益
    print("\n开始优化...")
    start = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, df, cache_mgr, args.algo_version, args.use_l3),
        n_trials=args.trials,
        show_progress_bar=True
    )

    elapsed = time.time() - start
    print(f"\n优化完成! 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")

    # 最佳结果
    print("\n" + "=" * 60)
    print("最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n最佳夏普: {study.best_value:.3f}")

    # 用最佳参数跑完整回测
    best = study.best_params
    result = run_backtest_vectorized(
        df,
        best['w_signal'],
        best['w_struct'],
        best['w_momentum'],
        best['w_chan'],
        best['threshold'],
        top_n=best['top_n'],
        position_limit=5
    )

    print(f"\n回测结果:")
    print(f"  总收益: {result['total_return']*100:.1f}%")
    print(f"  年化收益: {result['annual_return']*100:.1f}%")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_drawdown']*100:.1f}%")
    print(f"  胜率: {result['win_rate']*100:.1f}%")
    print(f"  交易次数: {result['n_trades']}")
    print(f"  交易天数: {result['n_days']}")

    # 保存
    output_path = '/root/.openclaw/workspace/projects/stock-analysis-system/wavechan_best_params.json'
    with open(output_path, 'w') as f:
        json.dump({
            'best_params': best,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in result.items()},
            'algo_version': args.algo_version,
            'l3_cached': args.use_l3,
        }, f, indent=2)
    print(f"\n已保存: {output_path}")


if __name__ == '__main__':
    main()
