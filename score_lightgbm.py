#!/usr/bin/env python3
"""
Score_LightGBM 高效版

优化策略：
1. 特征预生成+缓存（pickle），避免每次重复加载 8 年 parquet 数据
2. LightGBM 训练直接读缓存
3. 回测也读缓存，只替换评分逻辑

用法：
  python3 score_lightgbm.py --generate      # 预生成特征缓存
  python3 score_lightgbm.py --train         # 训练模型
  python3 score_lightgbm.py --backtest      # 回测（需要先 --generate）
  python3 score_lightgbm.py --all            # 一次性完成所有步骤
"""

import argparse
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
from backtester import BacktestOrchestrator
from strategies.score_strategy import ScoreStrategy

CACHE_FILE = './data/lgb_features_cache.pkl'
MODEL_FILE = './data/lgb_model.txt'
COMPARISON_FILE = './data/lgb_vs_baseline_comparison.json'

FEATURE_COLS = [
    # 技术面
    'growth', 'ma_condition', 'angle_ma_10', 'macd_condition', 'macd_jc',
    'rsi_14', 'kdj_k', 'kdj_d', 'cci_20', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower',
    'volume', 'amount', 'vol_ma5',
    'sma_5', 'sma_10', 'sma_20', 'sma_55',
    # 资金流
    '资金量', '主生量', '量增幅', '周量', '周增幅', 'XVL', 'LIJIN',
    'money_flow_positive', 'money_flow_increasing', 'money_flow_trend',
    # 市场热度
    'market_heat',
]


# ============================================================================
# 步骤1: 预生成特征缓存
# ============================================================================

def generate_feature_cache(start_date: str, end_date: str):
    """只调用一次，生成特征后序列化到 pickle 缓存"""
    print(f"📦 预生成特征缓存: {start_date} ~ {end_date}")

    strategy = ScoreStrategy()
    df = strategy.generate_features(start_date, end_date)

    if df.empty:
        print("⚠️ 特征数据为空！")
        return

    # 过滤
    if 'name' in df.columns:
        df = df[~df['name'].str.contains('ST', na=False)]
    df = df[df['close'] > 1.0]
    df = df.dropna(subset=['close', 'volume'])

    # 计算未来5日收益率（label）
    df = df.sort_values(['symbol', 'date'])
    df['future_close'] = df.groupby('symbol')['close'].shift(-5)
    df['label_5d_return'] = (df['future_close'] / df['close'] - 1)
    df = df.dropna(subset=['label_5d_return'])

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(df, f)

    print(f"✅ 缓存已保存: {CACHE_FILE}")
    print(f"   数据量: {len(df):,} 行")
    print(f"   时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   股票数: {df['symbol'].nunique()}")


# ============================================================================
# 步骤2: 训练 LightGBM 模型
# ============================================================================

def load_cache():
    """加载特征缓存"""
    if not os.path.exists(CACHE_FILE):
        print(f"❌ 缓存不存在: {CACHE_FILE}")
        print("   请先运行: python3 score_lightgbm.py --generate")
        return None
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)


def prepare_X(df: pd.DataFrame):
    """从 DataFrame 提取特征矩阵"""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in X.select_dtypes(bool).columns:
        X[col] = X[col].astype(int)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    return X[FEATURE_COLS], available


def sample_training_data(df: pd.DataFrame, sample_per_month: int = 100) -> pd.DataFrame:
    """
    每季度抽样：选成交额最高的 top_n + bottom_n，增强极端样本
    """
    df = df.copy()
    df['year_month'] = df['date'].astype(str).str[:7]

    samples = []
    for ym, group in df.groupby('year_month'):
        if len(group) < 20:
            continue
        group = group.sort_values('label_5d_return')
        n = len(group)
        k = min(sample_per_month, n // 4)

        # 极端样本：top/bottom 各 k 个
        # 中间样本：随机选 k 个
        extreme = pd.concat([group.iloc[:k], group.iloc[-k:]])
        mid = group.iloc[n//4:3*n//4].sample(min(k, n//2), random_state=42)
        samples.append(pd.concat([extreme, mid]))

    return pd.concat(samples, ignore_index=True) if samples else df.head(0)


def train_model():
    """训练 LightGBM"""
    print("\n🌲 训练 LightGBM 模型")
    df = load_cache()
    if df is None:
        return

    # 训练集：2017-2021
    train_mask = df['date'] < '2022-01-01'
    val_mask = (df['date'] >= '2022-01-01') & (df['date'] < '2024-01-01')

    train_df = sample_training_data(df[train_mask])
    val_df = sample_training_data(df[val_mask])

    print(f"   训练样本: {len(train_df):,}")
    print(f"   验证样本: {len(val_df):,}")

    X_train, fn_train = prepare_X(train_df)
    y_train = train_df['label_5d_return'].values
    X_val, _ = prepare_X(val_df)
    y_val = val_df['label_5d_return'].values

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'min_data_in_leaf': 50,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=fn_train)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=fn_train)

    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dtrain, dval], valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(30, verbose=True),
            lgb.log_evaluation(50)
        ]
    )

    # 验证集评估
    preds = model.predict(X_val)
    rmse = np.sqrt(np.mean((preds - y_val)**2))
    corr = np.corrcoef(preds, y_val)[0, 1]
    print(f"\n📊 验证集表现:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   预测-实际相关系数: {corr:.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': fn_train,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    print(f"\n📊 Top-10 特征重要性:")
    for _, r in importance.head(10).iterrows():
        print(f"   {r['feature']}: {r['importance']:.1f}")

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    model.save_model(MODEL_FILE)
    print(f"\n💾 模型已保存: {MODEL_FILE}")
    return model


# ============================================================================
# 步骤3: 回测
# ============================================================================

class LightGBMStrategy(ScoreStrategy):
    """使用 LightGBM 模型评分的策略"""

    def __init__(self, db_path=None, config=None, lgb_model=None):
        super().__init__(db_path, config)
        self.lgb_model = lgb_model

    def _score_stocks(self, signals: pd.DataFrame) -> pd.DataFrame:
        if self.lgb_model is None or signals.empty:
            return super()._score_stocks(signals)

        available = [c for c in FEATURE_COLS if c in signals.columns]
        X = signals[available].copy()
        for col in X.select_dtypes(bool).columns:
            X[col] = X[col].astype(int)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        for col in FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0
        X = X[FEATURE_COLS]

        preds = self.lgb_model.predict(X)
        signals = signals.copy()
        signals['total_score'] = preds
        return signals.sort_values('total_score', ascending=False)


def run_backtest_and_compare(model, test_start, test_end):
    """LightGBM vs 原始 Score 基线对比回测"""
    print(f"\n📈 回测: {test_start} ~ {test_end}")

    # LightGBM 策略
    lgb_strat = LightGBMStrategy(lgb_model=model, config={'top_n': 5})
    orch1 = BacktestOrchestrator(db_path=None, live_plot=False, position_limit=5, strategy_name='score')
    orch1.strategy = lgb_strat
    orch1.strategy.db_manager = orch1.db._parquet

    # 临时禁用数据库写入
    orig_gen = orch1._generate_report
    def no_write(sim):
        r = orig_gen(sim)
        r['summary'] = {}
        all_h = sim.portfolio['history']
        daily = [x for x in all_h if 'value' in x and 'type' not in x]
        df = pd.DataFrame(daily).set_index('date') if daily else pd.DataFrame()
        if not df.empty:
            r['summary']['final_value'] = df['value'].iloc[-1]
            r['summary']['total_return'] = df['value'].iloc[-1] / 5e5 - 1
            r['summary']['max_drawdown'] = (df['value'] / df['value'].cummax() - 1).min()
            returns = df['value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            r['summary']['sharpe'] = sharpe
        return r
    orch1._generate_report = no_write

    print("   跑 LightGBM 策略...")
    lgb_rep = orch1.run(test_start, test_end)
    lgb_sum = lgb_rep['summary']

    # 基线策略
    print("   跑 Score 基线...")
    orch2 = BacktestOrchestrator(db_path=None, live_plot=False, position_limit=5, strategy_name='score')
    orch2._generate_report = no_write
    base_rep = orch2.run(test_start, test_end)
    base_sum = base_rep['summary']

    # 对比结果
    print(f"\n{'='*50}")
    print(f"LightGBM 策略: 净值={lgb_sum['final_value']:,.0f}, 收益={lgb_sum['total_return']:.2%}, 回撤={lgb_sum['max_drawdown']:.2%}, 夏普={lgb_sum['sharpe']:.2f}")
    print(f"Score  基线:  净值={base_sum['final_value']:,.0f}, 收益={base_sum['total_return']:.2%}, 回撤={base_sum['max_drawdown']:.2%}")
    print(f"增量收益: {lgb_sum['total_return'] - base_sum['total_return']:+.2%}")
    print(f"增量回撤: {lgb_sum['max_drawdown'] - base_sum['max_drawdown']:+.2%}")
    print(f"{'='*50}")

    result = {
        'test_period': f"{test_start} ~ {test_end}",
        'lightgbm': {
            'final_value': float(lgb_sum['final_value']),
            'total_return': float(lgb_sum['total_return']),
            'max_drawdown': float(lgb_sum['max_drawdown']),
            'sharpe': float(lgb_sum.get('sharpe', 0)),
        },
        'baseline_score': {
            'final_value': float(base_sum['final_value']),
            'total_return': float(base_sum['total_return']),
            'max_drawdown': float(base_sum['max_drawdown']),
        },
    }

    with open(COMPARISON_FILE, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 对比结果已保存: {COMPARISON_FILE}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Score_LightGBM 高效版')
    parser.add_argument('--generate', action='store_true', help='预生成特征缓存')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--backtest', action='store_true', help='回测')
    parser.add_argument('--all', action='store_true', help='一次性完成所有步骤')
    parser.add_argument('--start', type=str, default='2017', help='开始年份')
    parser.add_argument('--end', type=str, default='2024', help='结束年份')
    args = parser.parse_args()

    if args.generate or args.all:
        t0 = time.time()
        generate_feature_cache(f"{args.start}-01-01", f"{args.end}-12-31")
        print(f"   耗时: {time.time()-t0:.1f}秒")

    if args.train or args.all:
        t0 = time.time()
        model = train_model()
        print(f"   耗时: {time.time()-t0:.1f}秒")

    if args.backtest or args.all:
        if not os.path.exists(MODEL_FILE):
            print(f"❌ 模型不存在: {MODEL_FILE}")
            return
        model = lgb.Booster(model_file=MODEL_FILE)
        run_backtest_and_compare(model, f"{args.start}-01-01", f"{args.end}-12-31")

    if not (args.generate or args.train or args.backtest or args.all):
        parser.print_help()


if __name__ == '__main__':
    main()
