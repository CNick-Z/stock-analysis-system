#!/usr/bin/env python3
"""
T1.3: 验证 V4/V8 策略模块对接
测试 simulator 能否正确 import 和调用 V4/V8 策略
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

from simulator.base_portfolio import BasePortfolio
from simulator.multi_simulator import MultiSimulator
from strategies.score.v4.strategy import ScoreV4Strategy
from strategies.score.v8.strategy import ScoreV8Strategy


def create_mock_data():
    """创建模拟日线数据（包含所有必需字段）"""
    dates = pd.date_range('2026-01-01', periods=5, freq='B')
    symbols = ['000001', '000002', '000004']
    
    records = []
    for sym in symbols:
        for i, date in enumerate(dates):
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': sym,
                'open': 10.0 + np.random.rand() * 2,
                'high': 10.5 + np.random.rand() * 2,
                'low': 9.5 + np.random.rand() * 2,
                'close': 10.2 + np.random.rand() * 2,
                'volume': 1000000 + np.random.rand() * 500000,
                'sma_5': 10.0 + np.random.rand() * 0.5,
                'sma_10': 10.1 + np.random.rand() * 0.5,
                'sma_20': 10.3 + np.random.rand() * 0.5,
                'sma_55': 10.5 + np.random.rand() * 0.5,
                'sma_240': 10.0 + np.random.rand() * 0.5,
                'vol_ma5': 900000,
                'macd': -0.05 + np.random.rand() * 0.1,
                'macd_signal': -0.06 + np.random.rand() * 0.1,
                'rsi_14': 50 + np.random.rand() * 10,
                'kdj_k': 50 + np.random.rand() * 20,
                'cci_20': -50 + np.random.rand() * 100,
                'williams_r': -50 + np.random.rand() * 30,
                'bb_upper': 12.0,
                'vol_ratio': 1.0 + np.random.rand() * 0.5,
                'turnover_rate': 1.0 + np.random.rand(),
                'growth': 0.5 + np.random.rand() * 2,
                'money_flow_positive': True,
                'money_flow_increasing': True,
                'money_flow_trend': True,
                'money_flow_weekly': True,
                'money_flow_weekly_increasing': True,
                '主生量': 100000,
                '量基线': 50000,
                '量增幅': 0.5,
                '周增幅': 0.3,
                'market_heat': 0.5,
                'angle_ma_10': 30 + np.random.rand() * 10,
            })
    
    df = pd.DataFrame(records)
    # 添加 next_open（次日开盘价）
    df = df.sort_values(['symbol', 'date'])
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    return df


def test_v4_strategy():
    """测试 V4 策略"""
    print("\n" + "="*60)
    print("测试 V4 Strategy")
    print("="*60)
    
    strategy = ScoreV4Strategy()
    print(f"Strategy: {strategy}")
    print(f"Entry conditions: {strategy.get_entry_conditions()}")
    
    # 测试 filter_buy
    df = create_mock_data()
    filtered = strategy.filter_buy(df)
    print(f"\nfilter_buy 结果: {len(filtered)}/{len(df)} 只股票通过")
    
    # 测试 score
    if not filtered.empty:
        scored = strategy.score(filtered)
        print(f"score 结果: top score = {scored.iloc[0]['score']:.4f}")
    
    # ========== should_sell 测试（使用固定值避免随机数据干扰）==========
    
    # 测试 should_sell - 止损
    # V4 默认止损 5%，所以 next_open < 10.0 * 0.95 = 9.5 触发止损
    row = df.iloc[0].copy()
    row['next_open'] = 9.4  # 低于止损价 9.5
    row['sma_20'] = 10.0
    row['sma_55'] = 10.5
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"\nshould_sell (止损测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发止损，实际 sell={sell}, reason={reason}"
    assert 'STOP_LOSS' in reason, f"期望 STOP_LOSS，实际 reason={reason}"

    # 测试 should_sell - 止盈
    # V4 默认止盈 15%，所以 next_open > 10.0 * 1.15 = 11.5 触发止盈
    row = df.iloc[0].copy()
    row['next_open'] = 11.6  # 高于止盈价 11.5
    row['sma_20'] = 10.0  # sma_20 < sma_55，不触发 MA死叉
    row['sma_55'] = 10.5
    row['money_flow_trend'] = True  # 资金流正常，不触发趋势破坏
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (止盈测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发止盈，实际 sell={sell}, reason={reason}"
    assert 'TAKE_PROFIT' in reason, f"期望 TAKE_PROFIT，实际 reason={reason}"

    # 测试 should_sell - MA死叉（V4/V8 均有）
    # 条件：入场时 sma20 <= sma55，当前 sma20 > sma55
    row = df.iloc[0].copy()
    row['next_open'] = 10.05  # 不触发止损/止盈（10.05 在 [9.5, 11.5] 区间外）
    row['sma_20'] = 10.4  # 当前 sma_20 > sma_55 = 死叉已发生
    row['sma_55'] = 10.2
    row['money_flow_trend'] = True  # 资金流正常，不触发趋势破坏
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}  # 入场时 sma20 <= sma55
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (MA死叉测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发MA死叉，实际 sell={sell}, reason={reason}"
    assert 'MA_DEATH_CROSS' in reason, f"期望 MA_DEATH_CROSS，实际 reason={reason}"

    # 测试 should_sell - 趋势破坏（仅 V4）
    # 条件：close < SMA20 且 money_flow_trend == False
    row = df.iloc[0].copy()
    row['next_open'] = 10.05  # 不触发止损/止盈
    row['close'] = 9.8  # close < sma_20
    row['sma_20'] = 10.0  # close(9.8) < sma_20(10.0)
    row['sma_55'] = 10.5  # sma_20 < sma_55，不触发MA死叉
    row['money_flow_trend'] = False  # 资金流趋势为负
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (趋势破坏测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发趋势破坏，实际 sell={sell}, reason={reason}"
    assert 'TREND_BREAK' in reason, f"期望 TREND_BREAK，实际 reason={reason}"

    # 测试 should_sell - 正常持仓（不应触发任何出场）
    row = df.iloc[0].copy()
    row['next_open'] = 10.05  # 不触发止损/止盈
    row['close'] = 10.2  # close > sma_20(10.0)，不触发趋势破坏
    row['sma_20'] = 10.0  # sma_20 < sma_55(10.2)，不触发MA死叉
    row['sma_55'] = 10.2
    row['money_flow_trend'] = True  # 资金流趋势正常
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (正常持仓测试): sell={sell}, reason={reason}")
    assert sell is False, f"期望不触发出场，实际 sell={sell}, reason={reason}"
    
    return True


def test_v8_strategy():
    """测试 V8 策略"""
    print("\n" + "="*60)
    print("测试 V8 Strategy")
    print("="*60)
    
    strategy = ScoreV8Strategy()
    print(f"Strategy: {strategy}")
    print(f"Entry conditions: {strategy.get_entry_conditions()}")
    
    # 测试 filter_buy
    df = create_mock_data()
    filtered = strategy.filter_buy(df)
    print(f"\nfilter_buy 结果: {len(filtered)}/{len(df)} 只股票通过")
    
    # 测试 score
    if not filtered.empty:
        scored = strategy.score(filtered)
        print(f"score 结果: top score = {scored.iloc[0]['score']:.4f}")
    
    # ========== should_sell 测试（使用固定值避免随机数据干扰）==========
    # V8 默认止损 5%、止盈 15%
    
    # 测试 should_sell - 止损
    # close < 10.0 * 0.95 = 9.5 触发止损
    row = df.iloc[0].copy()
    row['close'] = 9.4  # 低于止损价 9.5
    row['sma_20'] = 10.0
    row['sma_55'] = 10.5
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"\nshould_sell (止损测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发止损，实际 sell={sell}, reason={reason}"
    assert 'STOP_LOSS' in reason, f"期望 STOP_LOSS，实际 reason={reason}"

    # 测试 should_sell - 止盈
    # close > 10.0 * 1.15 = 11.5 触发止盈
    row = df.iloc[0].copy()
    row['close'] = 11.6  # 高于止盈价 11.5
    row['sma_20'] = 10.0  # sma_20 < sma_55，不触发 MA死叉
    row['sma_55'] = 10.5
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (止盈测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发止盈，实际 sell={sell}, reason={reason}"
    assert 'TAKE_PROFIT' in reason, f"期望 TAKE_PROFIT，实际 reason={reason}"

    # 测试 should_sell - MA死叉（V4/V8 均有）
    # 条件：入场时 sma20 <= sma55，当前 sma20 > sma55
    row = df.iloc[0].copy()
    row['next_open'] = 10.05  # 不触发止损/止盈
    row['sma_20'] = 10.4  # 当前 sma_20 > sma_55 = 死叉已发生
    row['sma_55'] = 10.2
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}  # 入场时 sma20 <= sma55
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (MA死叉测试): sell={sell}, reason={reason}")
    assert sell is True, f"期望触发MA死叉，实际 sell={sell}, reason={reason}"
    assert 'MA_DEATH_CROSS' in reason, f"期望 MA_DEATH_CROSS，实际 reason={reason}"

    # 注意：V8 策略没有 趋势破坏 条件，此处不测试

    # 测试 should_sell - 正常持仓（不应触发任何出场）
    # V8用收盘价判断止损/止盈，close=10.2在[9.5, 12.0]之间，不触发
    row = df.iloc[0].copy()
    row['close'] = 10.2  # 10.2在[9.5, 12.0]之间，不触发止损/止盈
    row['sma_20'] = 10.0  # sma_20 < sma_55(10.2)，不触发MA死叉
    row['sma_55'] = 10.2
    pos = {'avg_cost': 10.0, 'entry_sma20_le_sma55': True}
    sell, reason = strategy.should_sell(row, pos)
    print(f"should_sell (正常持仓测试): sell={sell}, reason={reason}")
    assert sell is False, f"期望不触发出场，实际 sell={sell}, reason={reason}"
    
    return True


def test_base_portfolio_integration():
    """测试 BasePortfolio 对接 V4/V8"""
    print("\n" + "="*60)
    print("测试 BasePortfolio 对接 V4/V8")
    print("="*60)
    
    df = create_mock_data()
    
    # 测试 V4
    v4 = ScoreV4Strategy()
    portfolio_v4 = BasePortfolio("V4", 1_000_000, v4)
    
    # 模拟买入
    candidates = v4.filter_buy(df)
    if not candidates.empty:
        scored = v4.score(candidates)
        top = scored.iloc[0]
        
        # 模拟建仓
        entry_sma20_le_sma55 = top['sma_20'] <= top['sma_55']
        portfolio_v4._open_position(
            symbol=top['symbol'],
            row=top,
            qty=1000,
            price=top['close'],
            date=top['date'],
            entry_sma20_le_sma55=entry_sma20_le_sma55,
        )
        print(f"\nV4 持仓建仓: {portfolio_v4.positions}")
    
    # 测试 on_day
    day2_df = df[df['date'] == df['date'].iloc[1]]
    trades = portfolio_v4.on_day(df['date'].iloc[1], day2_df)
    print(f"V4 on_day 交易: {len(trades)} 笔")
    
    return True


def test_multi_simulator_integration():
    """测试 MultiSimulator 对接 V4/V8"""
    print("\n" + "="*60)
    print("测试 MultiSimulator 对接 V4/V8")
    print("="*60)
    
    df = create_mock_data()
    
    sim = MultiSimulator(initial_cash_per_strategy=1_000_000)
    sim.add_strategy("V4", ScoreV4Strategy())
    sim.add_strategy("V8", ScoreV8Strategy())
    
    print(f"\nMultiSimulator: {sim}")
    print(f"Portfolios: {list(sim.portfolios.keys())}")
    
    # 运行
    sim.run(
        start_date='2026-01-01',
        end_date='2026-01-05',
        daily_data=df,
    )
    
    # 获取结果
    results = sim.get_results()
    for name, res in results.items():
        print(f"\n{name}: final_value={res['final_value']:,.2f}, "
              f"return={res['total_return_pct']}, "
              f"trades={res['n_trades']}")
    
    sim.print_summary()
    
    return True


if __name__ == '__main__':
    print("="*60)
    print("T1.3: V4/V8 策略模块对接测试")
    print("="*60)
    
    all_pass = True
    
    try:
        all_pass &= test_v4_strategy()
    except Exception as e:
        print(f"❌ V4 策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    try:
        all_pass &= test_v8_strategy()
    except Exception as e:
        print(f"❌ V8 策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    try:
        all_pass &= test_base_portfolio_integration()
    except Exception as e:
        print(f"❌ BasePortfolio 对接测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    try:
        all_pass &= test_multi_simulator_integration()
    except Exception as e:
        print(f"❌ MultiSimulator 对接测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("✅ T1.3 验证通过: V4/V8 策略模块对接正常")
    else:
        print("❌ T1.3 验证失败")
    print("="*60)
    
    sys.exit(0 if all_pass else 1)
