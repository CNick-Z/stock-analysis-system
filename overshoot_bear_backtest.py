#!/usr/bin/env python3
"""
超跌反弹策略（熊市专用）独立回测
不依赖backtester，自包含简单回测引擎
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'rsi_threshold': 25,          # RSI低于此值视为超跌
    'bias6_threshold': -8.0,   # BIAS6低于此值
    'bias12_threshold': -6.0,  # BIAS12低于此值
    'volume_ratio_min': 1.2,    # 量比最低要求
    'sell_rsi_threshold': 50,    # RSI>50则卖出
    'stop_loss_pct': 0.05,      # 止损5%
    'profit_target_pct': 0.10,  # 止盈10%
    'max_hold_days': 5,          # 最大持有5天
    'initial_cash': 1_000_000,  # 初始资金100万
    'position_size': 0.30,       # 每只股票仓位30%
    'max_positions': 5,          # 最多5只
    'buy_slots_per_day': 5,      # 每天最多买5只（分散持仓）
}


def load_year_data(year: int) -> pd.DataFrame:
    """加载指定年份数据并计算指标"""
    pq_path = f'/root/.openclaw/workspace/data/warehouse/daily_data_year={year}/'
    if not os.path.exists(pq_path):
        return pd.DataFrame()

    files = [f for f in os.listdir(pq_path) if f.endswith('.parquet')]
    if not files:
        return pd.DataFrame()

    df = pd.read_parquet(os.path.join(pq_path, files[0]))
    df['date'] = df['date'].astype(str).str[:10]
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # 计算RSI(14) - 向量化方式
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = df.groupby('symbol', sort=False)['close'].transform(lambda x: calc_rsi(x, 14))

    # 计算BIAS6和BIAS12
    df['ma_6'] = df.groupby('symbol', sort=False)['close'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['ma_12'] = df.groupby('symbol', sort=False)['close'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['bias6'] = (df['close'] - df['ma_6']) / (df['ma_6'] + 1e-10) * 100
    df['bias12'] = (df['close'] - df['ma_12']) / (df['ma_12'] + 1e-10) * 100

    # 计算量比
    df['vol_prev'] = df.groupby('symbol', sort=False)['volume'].shift(1)
    df['vol_ratio'] = df['volume'] / df['vol_prev'].fillna(1)
    df.loc[df['vol_ratio'] <= 0, 'vol_ratio'] = 1.0

    return df


def find_buy_signals(df: pd.DataFrame, dates: list) -> dict:
    """找出所有买入信号（向量化）"""
    cfg = CONFIG
    # 一次性计算所有信号
    # 排除新股：上市不足30天的不考虑（RSI数据不可靠）
    df = df.copy()
    df['trade_count'] = df.groupby('symbol').cumcount()
    buy_mask = (
        (df['trade_count'] >= 30) &
        (df['close'] >= 3.0) &  # 股价至少3元，避免炒作
        (df['rsi_14'] < cfg['rsi_threshold']) &
        (df['bias6'] < cfg['bias6_threshold']) &
        (df['bias12'] < cfg['bias12_threshold']) &
        (df['vol_ratio'] > cfg['volume_ratio_min'])
    )
    candidates = df[buy_mask].copy()
    if candidates.empty:
        return {}

    # 给每条信号打分：RSI越低分越高
    candidates['score'] = cfg['rsi_threshold'] - candidates['rsi_14']

    buy_signals = {}
    for date in dates:
        day_df = candidates[candidates['date'] == date]
        if day_df.empty:
            continue
        top = day_df.nlargest(cfg['buy_slots_per_day'], 'score')
        buy_signals[date] = []
        for _, row in top.iterrows():
            buy_signals[date].append({
                'symbol': row['symbol'],
                'price': row['close'],
                'date': date,
                'score': row['score'],
                'rsi': row['rsi_14'],
                'bias6': row['bias6'],
                'vol_ratio': row['vol_ratio'],
            })
    return buy_signals


def run_backtest(df: pd.DataFrame, buy_signals: dict, start_date: str, end_date: str) -> dict:
    """运行简单回测"""
    cfg = CONFIG
    cash = cfg['initial_cash']
    positions = {}  # symbol -> {'qty': int, 'buy_price': float, 'buy_date': str, 'buy_day_count': int}
    trades = []  # list of {date, symbol, type, price, qty, pnl, reason}
    equity_curve = []  # list of {date, equity}

    dates = sorted(df[df['date'].between(start_date, end_date)]['date'].unique())
    date_set = set(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # 创建价格查询表
    price_lookup = df.pivot_table(index='date', columns='symbol', values='close')

    def get_price(date, symbol):
        try:
            if date not in date_set or symbol not in price_lookup.columns:
                return None
            p = price_lookup.loc[date, symbol]
            return p if not pd.isna(p) else None
        except:
            return None

    def get_open_price(date, symbol):
        """用次日收盘作为模拟开盘价（简化处理）"""
        idx = date_to_idx.get(date, -1)
        if idx >= 0 and idx + 1 < len(dates):
            p = get_price(dates[idx + 1], symbol)
            if p is not None and not pd.isna(p):
                return p
        return None

    for date in dates:
        # 1. 卖出检查
        sell_list = []
        for sym, pos in list(positions.items()):
            pos['buy_day_count'] = pos.get('buy_day_count', 0) + 1
            cur_price = get_price(date, sym)
            if cur_price is None:
                continue

            buy_price = pos['buy_price']
            ret_pct = (cur_price - buy_price) / buy_price
            reason = ''

            # 卖出条件判断
            day_df = df[(df['date'] == date) & (df['symbol'] == sym)]
            if not day_df.empty:
                cur_rsi = day_df.iloc[0]['rsi_14']
                if cur_rsi > cfg['sell_rsi_threshold']:
                    reason = f'RSI反弹>{cfg["sell_rsi_threshold"]}'
                elif ret_pct <= -cfg['stop_loss_pct']:
                    reason = f'止损-{abs(ret_pct):.2%}'
                elif ret_pct >= cfg['profit_target_pct']:
                    reason = f'止盈+{ret_pct:.2%}'
                elif pos['buy_day_count'] >= cfg['max_hold_days']:
                    reason = f'超时持有{pos["buy_day_count"]}天'

            if reason:
                sell_list.append((sym, cur_price, reason))

        # 执行卖出
        for sym, price, reason in sell_list:
            pos = positions.pop(sym)
            qty = pos['qty']
            buy_px = pos['buy_price']
            pnl = (price - buy_px) * qty
            cash += price * qty
            commission = price * qty * 0.0003
            cash -= commission
            trades.append({
                'date': date, 'symbol': sym, 'type': 'sell',
                'price': price, 'qty': qty,
                'buy_price': buy_px, 'pnl': pnl - commission,
                'return_pct': (price - buy_px) / buy_px,
                'reason': reason,
                'hold_days': pos['buy_day_count']
            })

        # 2. 买入检查
        available_slots = cfg['max_positions'] - len(positions)
        if available_slots > 0 and date in buy_signals:
            for sig in buy_signals[date]:
                if available_slots <= 0:
                    break
                sym = sig['symbol']
                if sym in positions:
                    continue

                buy_price = get_open_price(date, sym)
                if buy_price is None or pd.isna(buy_price) or buy_price <= 0:
                    continue

                # 计算买入数量
                max_investment = cash * cfg['position_size']
                max_qty = int(max_investment / (buy_price * 1.0003) / 100) * 100
                if max_qty < 100:
                    continue

                cost = buy_price * max_qty * 1.0003
                if pd.isna(cost) or cost > cash:
                    continue

                cash -= cost
                positions[sym] = {
                    'qty': max_qty,
                    'buy_price': buy_price,
                    'buy_date': date,
                    'buy_day_count': 0,
                    'signal_rsi': sig['rsi'],
                    'signal_bias6': sig['bias6'],
                }
                trades.append({
                    'date': date, 'symbol': sym, 'type': 'buy',
                    'price': buy_price, 'qty': max_qty,
                    'buy_price': buy_price, 'pnl': 0,
                    'return_pct': 0,
                    'reason': f"RSI={sig['rsi']:.1f} BIAS={sig['bias6']:.1f}",
                    'hold_days': 0
                })
                available_slots -= 1

        # 3. 计算当日权益
        total_value = cash if not pd.isna(cash) else 0
        for sym, pos in positions.items():
            cur_price = get_price(date, sym)
            if cur_price is not None and not pd.isna(cur_price) and cur_price > 0:
                total_value += cur_price * pos['qty']

        if pd.isna(total_value):
            total_value = cfg['initial_cash']

        equity_curve.append({
            'date': date,
            'cash': cash if not pd.isna(cash) else 0,
            'position_value': total_value - (cash if not pd.isna(cash) else 0),
            'total_value': total_value
        })

    # 最终结算
    final_date = dates[-1] if dates else end_date
    for sym, pos in list(positions.items()):
        cur_price = get_price(final_date, sym)
        if cur_price and not pd.isna(cur_price) and cur_price > 0:
            qty = pos['qty']
            pnl = (cur_price - pos['buy_price']) * qty
            cash += cur_price * qty
            trades.append({
                'date': final_date, 'symbol': sym, 'type': 'final_sell',
                'price': cur_price, 'qty': qty,
                'buy_price': pos['buy_price'], 'pnl': pnl,
                'return_pct': (cur_price - pos['buy_price']) / pos['buy_price'],
                'reason': '期末清算',
                'hold_days': pos['buy_day_count']
            })
            positions.pop(sym)

    # 处理NaN情况
    total_value = cash if not pd.isna(cash) else cfg['initial_cash']
    peak_value = cfg['initial_cash']
    max_drawdown = 0.0
    for e in equity_curve:
        tv = e['total_value']
        if pd.isna(tv):
            continue
        if tv > peak_value:
            peak_value = tv
        dd = (peak_value - tv) / peak_value if peak_value > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    return {
        'initial_cash': cfg['initial_cash'],
        'final_value': total_value,
        'total_return': (total_value - cfg['initial_cash']) / cfg['initial_cash'],
        'max_drawdown': max_drawdown,
        'trades': trades,
        'equity_curve': equity_curve,
        'num_trades': len([t for t in trades if t['type'] in ('buy', 'sell', 'final_sell')]),
        'num_buys': len([t for t in trades if t['type'] == 'buy']),
        'num_sells': len([t for t in trades if t['type'] in ('sell', 'final_sell')]),
    }


def print_result(name: str, result: dict, market: str, years: list = None):
    """打印回测结果"""
    emoji = "🟢" if result['total_return'] > 0 else "🔴"
    print(f"{emoji} {name}")
    print(f"   收益率: {result['total_return']:+.2%} | 最大回撤: {result['max_drawdown']:+.2%}")
    print(f"   净值: {result['final_value']:,.0f} | 交易次数: {result['num_buys']}买/{result['num_sells']}卖")
    sells = [t for t in result['trades'] if t['type'] in ('sell', 'final_sell')]
    if sells:
        wins = [t for t in sells if t['pnl'] > 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        avg_pnl = np.mean([t['pnl'] for t in sells])
        print(f"   胜率: {win_rate:.0f}% | 平均单笔: {avg_pnl:,.0f}")
    print()


if __name__ == '__main__':
    cfg = CONFIG
    print("=" * 60)
    print("超跌反弹策略（熊市专用）独立回测")
    print("=" * 60)
    print(f"买入: RSI<{cfg['rsi_threshold']} & BIAS6<{cfg['bias6_threshold']}% & BIAS12<{cfg['bias12_threshold']}% & 量比>{cfg['volume_ratio_min']}")
    print(f"卖出: RSI>{cfg['sell_rsi_threshold']} | 止损{cfg['stop_loss_pct']:.0%} | 止盈{cfg['profit_target_pct']:.0%} | 持有>{cfg['max_hold_days']}天 | 仓位{cfg['position_size']:.0%}")
    print()

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_results = {}

    for year in years:
        print(f"📅 {year} 年加载数据中...")
        df_year = load_year_data(year)
        if df_year.empty:
            print(f"  ⚠️  无数据")
            continue
        print(f"  ✅ {len(df_year)} 条记录, {df_year['symbol'].nunique()} 只股票")

        dates = sorted(df_year['date'].unique())
        buy_signals = find_buy_signals(df_year, dates)
        total_sig = sum(len(v) for v in buy_signals.values())
        print(f"  📊 买入信号: {total_sig} 个")

        result = run_backtest(df_year, buy_signals, f'{year}-01-01', f'{year}-12-31')
        market = "震荡" if year in (2021, 2023) else ("牛市" if year in (2020, 2025) else "熊市")
        print_result(f"{year}年({market})", result, market)
        all_results[year] = {**result, 'market': market}

    # 汇总
    print("=" * 60)
    print("📊 各年表现汇总")
    print("=" * 60)
    print(f"{'年份':<6} {'市场':<8} {'收益率':>10} {'最大回撤':>10} {'买/卖':>8}")
    print("-" * 50)
    for year in years:
        if year not in all_results:
            print(f"{year:<6} {'无数据':<8}")
            continue
        r = all_results[year]
        ret = r['total_return']
        dd = r['max_drawdown']
        nb = r['num_buys']
        ns = r['num_sells']
        flag = "✅" if ret > 0 else "❌"
        print(f"{year:<6} {r['market']:<8} {ret:>+10.2%} {dd:>10.2%} {nb:>3}买/{ns:>3}卖 {flag}")

    valid = {y: v for y, v in all_results.items() if 'total_return' in v}
    if valid:
        rets = [v['total_return'] for v in valid.values()]
        dds = [v['max_drawdown'] for v in valid.values()]
        avg_ret = sum(rets) / len(rets)
        print(f"\n平均收益率: {avg_ret:+.2%} | 平均最大回撤: {sum(dds)/len(dds):+.2%}")

        bull_years = {y: v for y, v in valid.items() if y in (2020, 2023, 2025)}
        bear_years = {y: v for y, v in valid.items() if y in (2021, 2022, 2024)}
        if bull_years:
            br = sum(v['total_return'] for v in bull_years.values()) / len(bull_years)
            print(f"牛市/震荡年平均: {br:+.2%}")
        if bear_years:
            br = sum(v['total_return'] for v in bear_years.values()) / len(bear_years)
            print(f"熊市年平均: {br:+.2%}")

        # 对比Score策略
        print("\n" + "=" * 60)
        print("📊 vs Score策略 对比")
        print("=" * 60)
        score_plain = {2020:0.0129, 2021:-0.0098, 2022:-0.1868, 2023:0.0396, 2024:-0.1011, 2025:0.0507}
        print(f"{'年份':<6} {'Score Plain':>12} {'超跌策略':>12} {'差异':>10}")
        print("-" * 45)
        for year in years:
            if year not in valid:
                continue
            sp = score_plain.get(year, 0)
            ov = valid[year]['total_return']
            diff = ov - sp
            flag = "⬆️" if diff > 0 else "⬇️"
            print(f"{year:<6} {sp:>+12.2%} {ov:>+12.2%} {diff:>+10.2%} {flag}")
