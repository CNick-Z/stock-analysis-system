#!/usr/bin/env python3
"""
v8_offline_sim.py — V8 熊市过滤离线模拟
核心思路：用 xlsx 交易记录重放，买入时降仓（position_limit），
         卖出时直接用原始 qty（不受影响），绩效直接用 xlsx 的 pnl 累计
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

INITIAL_CASH = 1_000_000.0
POSITION_SIZE = 0.20

PARAM_GRID = []
PARAM_GRID.append({'neutral': 1.0,  'bear': 1.0,  'confirm': 2, 'name': 'baseline'})
PARAM_GRID.append({'neutral': 1.0,  'bear': 0.40, 'confirm': 2, 'name': 'v2.5旧'})
PARAM_GRID.append({'neutral': 0.70, 'bear': 0.30, 'confirm': 1, 'name': '建议配置'})
for neutral in [0.50, 0.60, 0.70, 0.80]:
    for bear in [0.10, 0.20, 0.30, 0.40]:
        for confirm in [1, 2]:
            name = f"n{int(neutral*100)}_b{int(bear*100)}_c{confirm}"
            PARAM_GRID.append({'neutral': neutral, 'bear': bear, 'confirm': confirm, 'name': name})

XLSX_PATH = PROJECT_ROOT / "backtestresult" / "trades_report_1774334705.xlsx"
INDEX_PATH = "/data/warehouse/indices/CSI300.parquet"


def load_trades(path):
    print(f"📂 加载交易记录: {path}")
    df = pd.read_excel(path, sheet_name='Transactions')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  共 {len(df)} 条, {df['symbol'].nunique()} 只股票, {df['date'].min()} ~ {df['date'].max()}")
    return df


def load_index(path):
    print(f"📂 加载指数: {path}")
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    print(f"  共 {len(df)} 行")
    return df[['date', 'close']]


def build_regime_map(start_date, end_date, neutral, bear, confirm):
    """预计算 regime limit map（date → position_limit）"""
    from simulator.market_regime import MarketRegimeFilter
    mrf = MarketRegimeFilter(
        index_path=INDEX_PATH,
        confirm_days=confirm,
        neutral_position=neutral,
        bear_position=bear,
        regime_persist_days=3,
    )
    regime_df = mrf.prepare(start_date, end_date)
    regime_df['date_str'] = pd.to_datetime(regime_df['date']).dt.strftime('%Y-%m-%d')
    date_to_limit = regime_df.set_index('date_str')['position_limit'].to_dict()
    return date_to_limit


def run_simulation(trades_df, index_df, params, start_date, end_date):
    neutral = params['neutral']
    bear = params['bear']
    confirm = params['confirm']
    name = params['name']

    # 过滤
    df = trades_df[
        (trades_df['date'] >= start_date) & (trades_df['date'] <= end_date)
    ].copy().sort_values('date').reset_index(drop=True)
    if df.empty:
        return None

    # regime limit map
    date_to_limit = build_regime_map(start_date, end_date, neutral, bear, confirm)

    # ── 核心模拟 ──────────────────────────────────────────
    cash = float(INITIAL_CASH)
    positions = {}  # {symbol: {'qty': int, 'cost': float}}
    equity_dates = []   # [date]
    equity_values = []  # [value]
    reduced_count = 0
    total_sell_value = 0.0  # 累计卖出的钱
    trade_log = []  # [{date, symbol, type, qty, price}]

    dates = sorted(df['date'].unique())

    for date in dates:
        day = df[df['date'] == date]
        limit = date_to_limit.get(date, 1.0)

        for _, row in day.iterrows():
            sym = str(row['symbol'])
            typ = row['type']
            qty_orig = int(row['quantity'])
            price = float(row['price'])

            if typ == 'sell':
                # 卖出：用原始数量（不受 position_limit 影响）
                # 但不能超过持仓
                if sym not in positions or positions[sym]['qty'] <= 0:
                    continue
                hold_qty = positions[sym]['qty']
                actual_sell = min(qty_orig, hold_qty)
                avg_cost = positions[sym]['cost'] / hold_qty
                cost_sold = avg_cost * actual_sell
                commission = max(5.0, actual_sell * price * 0.0003)
                pnl = actual_sell * price - cost_sold - commission
                cash += actual_sell * price - commission
                total_sell_value += actual_sell * price - commission
                positions[sym]['qty'] -= actual_sell
                positions[sym]['cost'] -= cost_sold
                if positions[sym]['qty'] <= 0:
                    del positions[sym]
                trade_log.append({'date': date, 'symbol': sym, 'type': 'sell',
                                  'qty': actual_sell, 'price': price, 'pnl': pnl})

            else:  # buy
                # 最大可买量 = cash × position_limit × position_size
                max_invest = cash * limit * POSITION_SIZE
                max_qty = int(max_invest / price / 100) * 100
                actual_qty = min(qty_orig, max_qty)
                actual_qty = (actual_qty // 100) * 100

                if actual_qty < qty_orig:
                    reduced_count += 1

                if actual_qty <= 0:
                    continue

                commission = max(5.0, actual_qty * price * 0.0003)
                cost = actual_qty * price
                cash -= (cost + commission)

                if sym in positions:
                    old = positions[sym]
                    positions[sym] = {
                        'qty': old['qty'] + actual_qty,
                        'cost': old['cost'] + cost + commission
                    }
                else:
                    positions[sym] = {'qty': actual_qty, 'cost': cost + commission}

                trade_log.append({'date': date, 'symbol': sym, 'type': 'buy',
                                  'qty': actual_qty, 'price': price})

        # 当日 equity：用持仓成本法估算
        pos_cost = sum(p['cost'] for p in positions.values())
        equity_dates.append(date)
        equity_values.append(cash + pos_cost)

    # ── 绩效 ──────────────────────────────────────────
    equity_dates = np.array(equity_dates)
    equity_values = np.array(equity_values)
    n_days = len(equity_values)
    years = n_days / 252

    final_value = equity_values[-1]
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    peak = np.maximum.accumulate(equity_values)
    drawdown = (equity_values - peak) / peak
    max_dd = drawdown.min()

    rets = pd.Series(equity_values).pct_change().dropna()
    daily_rf = 0.03 / 252
    sharpe = (rets.mean() - daily_rf) / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    trades_count = len(trade_log)
    avg_hold = n_days / max(trades_count / 2, 1) if trades_count > 0 else 0

    return {
        'name': name, 'neutral': neutral, 'bear': bear, 'confirm': confirm,
        'start': equity_dates[0], 'end': equity_dates[-1],
        'years': round(years, 1),
        'final_value': round(final_value, 0),
        'total_return': f"{total_return:.2%}",
        'annualized': f"{annualized:.2%}",
        'max_drawdown': f"{max_dd:.2%}",
        'sharpe': f"{sharpe:.2f}",
        'total_trades': trades_count,
        'avg_hold_days': f"{avg_hold:.0f}",
        'reduced_count': reduced_count,
    }


def main():
    print("=" * 60)
    print("V8 熊市过滤离线模拟")
    print("=" * 60)

    trades_df = load_trades(XLSX_PATH)
    index_df = load_index(INDEX_PATH)

    samples = [
        ('样本内',  '2010-01-01', '2019-12-31'),
        ('样本外',  '2020-01-01', '2025-12-31'),
    ]

    all_results = []

    for params in PARAM_GRID:
        print(f"\n{'─'*50}")
        print(f"▶ {params['name']} (n={params['neutral']}, b={params['bear']}, c={params['confirm']})")
        for sample_name, start, end in samples:
            result = run_simulation(trades_df, index_df, params, start, end)
            if result:
                result['sample'] = sample_name
                all_results.append(result)
                print(f"  [{sample_name}] 年化={result['annualized']} MaxDD={result['max_drawdown']} "
                      f"夏普={result['sharpe']} 降仓={result['reduced_count']}次 最终={result['final_value']:,.0f}")
            else:
                print(f"  [{sample_name}] 无数据")

    # CSV
    csv_path = PROJECT_ROOT / "V8_OFFLINE_SIM_DETAILS.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV: {csv_path}")

    # Markdown
    md_path = PROJECT_ROOT / "V8_OFFLINE_SIM_RESULTS.md"
    key_groups = ['baseline', 'v2.5旧', '建议配置']

    lines = ["# V8 熊市过滤离线模拟 — 结果\n\n"]
    lines.append(f"**参数网格**: {len(PARAM_GRID)} 组 | **样本内**: 2010-2019 | **样本外**: 2020-2025\n\n")

    lines.append("## 关键对比\n\n")
    lines.append("| 组别 | neutral | bear | confirm | 样本 | 年化 | MaxDD | 夏普 | 降仓次数 | 最终资金 |\n")
    lines.append("|------|---------|------|---------|------|------|------|------|----------|---------|\n")
    for g in key_groups:
        for r in all_results:
            if r['name'] == g:
                lines.append(f"| {g} | {r['neutral']} | {r['bear']} | {r['confirm']} "
                             f"| {r['sample']} | {r['annualized']} | {r['max_drawdown']} | {r['sharpe']} "
                             f"| {r['reduced_count']} | {r['final_value']:,.0f} |\n")

    lines.append("\n## 完整结果（样本内，按年化降序）\n\n")
    in_sample = sorted([r for r in all_results if r['sample'] == '样本内'],
                       key=lambda x: float(x['annualized'].replace('%', '')), reverse=True)
    lines.append("| 组别 | neutral | bear | confirm | 年化 | MaxDD | 夏普 | 降仓 | 年数 |\n")
    lines.append("|------|---------|------|---------|------|------|------|------|------|\n")
    for r in in_sample:
        lines.append(f"| {r['name']} | {r['neutral']} | {r['bear']} | {r['confirm']} "
                     f"| {r['annualized']} | {r['max_drawdown']} | {r['sharpe']} "
                     f"| {r['reduced_count']} | {r['years']}年 |\n")

    lines.append("\n## 样本外验证（按年化降序）\n\n")
    out_sample = sorted([r for r in all_results if r['sample'] == '样本外'],
                        key=lambda x: float(x['annualized'].replace('%', '')), reverse=True)
    lines.append("| 组别 | neutral | bear | confirm | 年化 | MaxDD | 夏普 | 降仓 | 年数 |\n")
    lines.append("|------|---------|------|---------|------|------|------|------|------|\n")
    for r in out_sample:
        lines.append(f"| {r['name']} | {r['neutral']} | {r['bear']} | {r['confirm']} "
                     f"| {r['annualized']} | {r['max_drawdown']} | {r['sharpe']} "
                     f"| {r['reduced_count']} | {r['years']}年 |\n")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✅ Markdown: {md_path}")
    print("\n🎉 全部完成!")


if __name__ == '__main__':
    main()
