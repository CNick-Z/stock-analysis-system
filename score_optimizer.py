#!/usr/bin/env python3
"""
Score 策略 Optuna 超参优化

目标：最大化夏普比率（Sharpe Ratio）
同时考察：总收益、最大回撤、胜率

用法：
    python3 score_optimizer.py --trials 100 --start 2025-01-01 --end 2025-06-30
"""

import argparse
import sys
import os
import time
import optuna
import pandas as pd
import numpy as np

# 解决路径问题
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_backtest_with_weights(weights_config, start_date, end_date, position_limit=5):
    """
    用给定权重配置运行回测，返回绩效指标

    weights_config 格式（会被传入 ScoreStrategy 的 config）：
    {
        'weights': {'technical': 0.4483, 'capital_flow': 0.2947, 'market_heat': 0.257, 'fundamental': 0.0},
        'tech_weights': {...},
        'cap_weights': {...},
        'thresholds': {...}
    }
    """
    from backtester import BacktestOrchestrator
    from strategies.score_strategy import ScoreStrategy

    # 直接通过 config 传权重给 ScoreStrategy
    strategy_config = weights_config.get('weights', {})
    strategy_config['tech_weights'] = weights_config.get('tech_weights', {})
    strategy_config['cap_weights'] = weights_config.get('cap_weights', {})
    strategy_config['thresholds'] = weights_config.get('thresholds', {})

    # 创建 orchestrator（内部会创建带 config 的 ScoreStrategy）
    orchestrator = BacktestOrchestrator(
        db_path=None,
        live_plot=False,
        position_limit=position_limit,
        strategy_name='score'
    )
    # 用自定义权重覆盖策略 config
    orchestrator.strategy.config.update(strategy_config)

    # 禁用数据库写入（只返回结果，不写DB）
    def no_write_report(simulator):
        result = {'summary': {}, 'trades': pd.DataFrame()}
        all_history = simulator.portfolio['history']
        daily_records = [r for r in all_history if 'value' in r and 'type' not in r]
        trade_records_list = [r for r in all_history if 'type' in r]

        df = pd.DataFrame(daily_records).set_index('date')
        if len(df) > 0:
            df['returns'] = df['value'].pct_change()
            result['summary']['final_value'] = df['value'].iloc[-1]
            result['summary']['total_return'] = df['value'].iloc[-1] / 5e5 - 1
            result['summary']['max_drawdown'] = (df['value'] / df['value'].cummax() - 1).min()

            # 计算夏普比率
            returns = df['returns'].dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            result['summary']['sharpe'] = sharpe if not np.isnan(sharpe) else 0
        else:
            result['summary']['sharpe'] = 0
            result['summary']['total_return'] = 0
            result['summary']['max_drawdown'] = 0
            result['summary']['final_value'] = 5e5

        trades = pd.DataFrame(trade_records_list)
        result['trades'] = trades

        # 胜率
        if not trades.empty and 'pnl' in trades.columns:
            winners = (trades[trades['type'] == 'sell']['pnl'] > 0).sum()
            n_sells = len(trades[trades['type'] == 'sell'])
            result['summary']['win_rate'] = winners / n_sells if n_sells > 0 else 0
        else:
            result['summary']['win_rate'] = 0

        result['summary']['n_trades'] = len(trades)
        return result

    saved_generate_report = orchestrator._generate_report
    orchestrator._generate_report = no_write_report

    try:
        report = orchestrator.run(start_date, end_date)
        return {
            'sharpe': report['summary'].get('sharpe', 0),
            'total_return': report['summary'].get('total_return', 0),
            'max_drawdown': report['summary'].get('max_drawdown', 0),
            'win_rate': report['summary'].get('win_rate', 0),
            'final_value': report['summary'].get('final_value', 5e5),
            'n_trades': report['summary'].get('n_trades', 0),
            'status': 'ok'
        }
    except Exception as e:
        import traceback
        return {
            'sharpe': -999,
            'total_return': -1,
            'max_drawdown': -1,
            'win_rate': 0,
            'final_value': 0,
            'n_trades': 0,
            'status': f'error: {str(e)[:100]}'
        }


class WeightObjective:
    """Optuna 目标函数对象"""

    def __init__(self, start_date, end_date, position_limit=5):
        self.start_date = start_date
        self.end_date = end_date
        self.position_limit = position_limit

    def __call__(self, trial):
        # === 采样三大类权重（归一化和=1） ===
        w_tech = trial.suggest_float('w_technical', 0.1, 0.8)
        w_cap = trial.suggest_float('w_capital_flow', 0.05, 0.6)
        w_heat = trial.suggest_float('w_market_heat', 0.0, 0.5)
        total = w_tech + w_cap + w_heat
        w_tech /= total
        w_cap /= total
        w_heat /= total

        # === 技术面子因子权重（归一化）===
        tw_ma = trial.suggest_float('tw_ma_condition', 0.05, 0.30)
        tw_angle = trial.suggest_float('tw_angle_condition', 0.02, 0.20)
        tw_macd = trial.suggest_float('tw_macd_condition', 0.05, 0.25)
        tw_vol = trial.suggest_float('tw_volume_score', 0.05, 0.30)
        tw_rsi = trial.suggest_float('tw_rsi_oversold', 0.01, 0.15)
        tw_kdj = trial.suggest_float('tw_kdj_oversold', 0.01, 0.15)
        tw_cci = trial.suggest_float('tw_cci_oversold', 0.01, 0.15)
        tw_bb = trial.suggest_float('tw_bollinger_condition', 0.03, 0.25)
        tw_jc = trial.suggest_float('tw_macd_jc', 0.02, 0.20)
        tw_pg = trial.suggest_float('tw_price_growth', 0.01, 0.15)
        tw_total = tw_ma + tw_angle + tw_macd + tw_vol + tw_rsi + tw_kdj + tw_cci + tw_bb + tw_jc + tw_pg

        weights_config = {
            'weights': {
                'technical': w_tech,
                'capital_flow': w_cap,
                'market_heat': w_heat,
                'fundamental': 0.0
            },
            'tech_weights': {
                'ma_condition': tw_ma / tw_total,
                'angle_condition': tw_angle / tw_total,
                'macd_condition': tw_macd / tw_total,
                'volume_score': tw_vol / tw_total,
                'rsi_oversold': tw_rsi / tw_total,
                'kdj_oversold': tw_kdj / tw_total,
                'cci_oversold': tw_cci / tw_total,
                'bollinger_condition': tw_bb / tw_total,
                'macd_jc': tw_jc / tw_total,
                'price_growth': tw_pg / tw_total,
            },
            'cap_weights': {
                'positive_flow': -0.0072, 'flow_increasing': 0.0108,
                'trend_strength': 0.0147, 'weekly_flow': 0.0072,
                'weekly_increasing': 0.0036, 'volume_gain_ratio': 0.1524,
                'volume_baseline': 0.0437, 'primary_volume': 0.0159,
                'weekly_growth': 0.2438, 'volume_gain_multiplier': 0.3434,
                'volume_loss_multiplier': 0.1717
            },
            'thresholds': {
                'volume_gain_threshold': 1.3, 'volume_loss_threshold': -0.65,
                'growth_min': 0.5, 'growth_max': 6.0, 'angle_min': 30,
            }
        }

        result = run_backtest_with_weights(weights_config, self.start_date, self.end_date, self.position_limit)

        # 日历年化
        n_years = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days / 365
        annual_return = (1 + result['total_return']) ** (1 / n_years) - 1 if n_years > 0 else 0

        trial.set_user_attr('total_return', result['total_return'])
        trial.set_user_attr('max_drawdown', result['max_drawdown'])
        trial.set_user_attr('sharpe', result['sharpe'])
        trial.set_user_attr('win_rate', result['win_rate'])
        trial.set_user_attr('n_trades', result['n_trades'])
        trial.set_user_attr('annual_return', annual_return)

        # 优化目标：夏普比率（如果夏普太差，惩罚）
        if result['sharpe'] < -10:
            return -999
        return result['sharpe']


def main():
    parser = argparse.ArgumentParser(description='Score 策略 Optuna 优化')
    parser.add_argument('--trials', type=int, default=50, help='优化试验次数')
    parser.add_argument('--start', type=str, default='2025-01-01', help='回测开始日期')
    parser.add_argument('--end', type=str, default='2025-06-30', help='回测结束日期')
    parser.add_argument('--position-limit', type=int, default=5, help='最大持仓数')
    parser.add_argument('--name', type=str, default='default', help='实验名称')
    args = parser.parse_args()

    study_name = f"score_optimization_{args.name}_{int(time.time())}"

    print(f"🎯 Score 策略优化开始")
    print(f"   时间段: {args.start} ~ {args.end}")
    print(f"   试验次数: {args.trials}")
    print(f"   实验名: {study_name}")
    print()

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # 最大化夏普比率
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f'sqlite:///./data/optuna_{args.name}.db',
        load_if_exists=True
    )

    objective = WeightObjective(args.start, args.end, args.position_limit)

    print("开始优化...（会在后台运行）")
    print()

    # 快速预览：先跑一个 baseline
    baseline = run_backtest_with_weights({
        'weights': {'technical': 0.4483, 'capital_flow': 0.2947, 'market_heat': 0.257, 'fundamental': 0.0},
        'tech_weights': {
            'ma_condition': 0.1622, 'angle_condition': 0.0854, 'macd_condition': 0.1366,
            'volume_score': 0.1704, 'rsi_oversold': 0.0597, 'kdj_oversold': 0.0597,
            'cci_oversold': 0.0597, 'bollinger_condition': 0.1191, 'macd_jc': 0.0873, 'price_growth': 0.06
        },
        'cap_weights': {
            'positive_flow': -0.0072, 'flow_increasing': 0.0108, 'trend_strength': 0.0147,
            'weekly_flow': 0.0072, 'weekly_increasing': 0.0036, 'volume_gain_ratio': 0.1524,
            'volume_baseline': 0.0437, 'primary_volume': 0.0159, 'weekly_growth': 0.2438,
            'volume_gain_multiplier': 0.3434, 'volume_loss_multiplier': 0.1717
        },
        'thresholds': {'volume_gain_threshold': 1.3, 'volume_loss_threshold': -0.65, 'growth_min': 0.5, 'growth_max': 6.0, 'angle_min': 30}
    }, args.start, args.end, args.position_limit)

    print(f"📊 Baseline 绩效（当前权重）:")
    print(f"   夏普比率: {baseline['sharpe']:.3f}")
    print(f"   总收益: {baseline['total_return']:.2%}")
    print(f"   最大回撤: {baseline['max_drawdown']:.2%}")
    print(f"   交易次数: {baseline['n_trades']}")
    print()

    # 开始优化
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print()
    print(f"✅ 优化完成！最佳试验: #{study.best_trial.number}")
    print(f"   最佳夏普比率: {study.best_value:.3f}")
    print()
    print(f"📈 最佳权重配置:")
    best = study.best_trial
    for key, value in best.params.items():
        print(f"   {key}: {value:.4f}")
    print()
    print(f"   总收益: {best.user_attrs['total_return']:.2%}")
    print(f"   最大回撤: {best.user_attrs['max_drawdown']:.2%}")
    print(f"   年化收益: {best.user_attrs['annual_return']:.2%}")
    print(f"   胜率: {best.user_attrs['win_rate']:.2%}")
    print(f"   交易次数: {best.user_attrs['n_trades']}")

    # 保存最佳配置
    import json
    best_config = {
        'weights': {
            'technical': best.params['w_technical'] / (best.params['w_technical'] + best.params['w_capital_flow'] + best.params['w_market_heat']),
            'capital_flow': best.params['w_capital_flow'] / (best.params['w_technical'] + best.params['w_capital_flow'] + best.params['w_market_heat']),
            'market_heat': best.params['w_market_heat'] / (best.params['w_technical'] + best.params['w_capital_flow'] + best.params['w_market_heat']),
            'fundamental': 0.0
        },
        'tech_weights': {
            'ma_condition': best.params['tw_ma_condition'],
            'angle_condition': best.params['tw_angle_condition'],
            'macd_condition': best.params['tw_macd_condition'],
            'volume_score': best.params['tw_volume_score'],
            'rsi_oversold': best.params['tw_rsi_oversold'],
            'kdj_oversold': best.params['tw_kdj_oversold'],
            'cci_oversold': best.params['tw_cci_oversold'],
            'bollinger_condition': best.params['tw_bollinger_condition'],
            'macd_jc': best.params['tw_macd_jc'],
            'price_growth': best.params['tw_price_growth'],
        },
        'metrics': best.user_attrs
    }
    config_path = f'./data/best_score_weights_{args.name}.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print()
    print(f"💾 最佳配置已保存: {config_path}")


if __name__ == '__main__':
    main()
