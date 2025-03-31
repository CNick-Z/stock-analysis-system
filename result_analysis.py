# result_analysis.py
# -*- coding: utf-8 -*-
import ast
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.strategy import StockScorer, EnhancedTDXStrategy
from tabulate import tabulate

# ======================
# 全局配置
# ======================
PROFIT_WEIGHT_CONFIG = {
    'thresholds': {
        'small': 0.04,   # 小额盈利阈值
        'medium': 0.12,  # 中额盈利阈值 
        'large': 0.25    # 大幅盈利阈值
    },
    'weights': {
        'loss': 0.6,     # 亏损交易权重
        'small': 1.0,    # 小额盈利权重
        'medium': 1.5,   # 中额盈利权重
        'large': 2.0,    # 大幅盈利权重
        'huge': 3.0      # 超额盈利权重
    },
    'auto_optimize': True
}
def safe_literal_eval(value):
    """安全解析字典字符串"""
    try:
        if pd.isna(value) or value == "":
            return {}
        value = str(value).replace("'", "\"")
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"解析失败: {value}，错误: {str(e)}")
        return {}
# ======================
# 核心功能函数
# ======================
def get_current_config():
    """获取策略当前配置"""
    return StockScorer().config

def load_and_pair_transactions(directory_path):
    """加载并配对交易数据"""
    paired_data = []
    for file in [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]:
        df = pd.read_excel(os.path.join(directory_path, file), sheet_name='Transactions')
        buys = df[df['type'] == 'buy']
        sells = df[df['type'] == 'sell']
        
        for _, sell in sells.iterrows():
            match = buys[(buys['symbol'] == sell['symbol']) & 
                       (buys['date'] < sell['date']) & 
                       (buys['quantity'] == sell['quantity'])]
            if not match.empty:
                paired_data.append({
                    'symbol': sell['symbol'],
                    'buy_date': match.iloc[0]['date'],
                    'sell_date': sell['date'],
                    'buy_price': match.iloc[0]['price'],
                    'sell_price': sell['price'],
                    'pnl': sell['pnl'],
                    'signal': safe_literal_eval(match.iloc[0]['signal']),
                    'quantity': sell['quantity']
                })
    return pd.DataFrame(paired_data)

def parse_buy_signals(paired_df):
    """解析信号字典（增强容错）"""
    # 信号特征映射表
    feature_mapping = {
        'technical': ['ma_condition', 'angle_condition', 'macd_condition',
                     'volume_condition', 'rsi_oversold', 'kdj_oversold',
                     'cci_oversold', 'bollinger_condition', 'macd_jc'],
        'capital_flow': ['money_flow_positive', 'money_flow_increasing',
                        'money_flow_trend', 'money_flow_weekly',
                        'money_flow_weekly_increasing'],
        'raw_indicators': ['量增幅', '量基线', '主生量', '周增幅', 'growth', 'market_heat']
    }
    
    # 解析所有信号
    for feat in [f for sublist in feature_mapping.values() for f in sublist]:
        paired_df[feat] = paired_df['signal'].apply(
            lambda x: x.get(feat, np.nan) if isinstance(x, dict) else np.nan)
        if paired_df[feat].dtype == bool:
            paired_df[feat] = paired_df[feat].astype(float)
    
    # 计算主特征分数
    paired_df['technical'] = paired_df[feature_mapping['technical']].sum(axis=1)
    paired_df['capital_flow'] = paired_df[feature_mapping['capital_flow']].sum(axis=1)
    
    return paired_df

def prepare_features(parsed_df):
    """特征工程（含动态阈值优化）"""
    current_config = get_current_config()['thresholds']
    
    # 基础特征计算
    parsed_df['return_rate'] = (parsed_df['sell_price'] - parsed_df['buy_price']) / parsed_df['buy_price']
    parsed_df['is_profit'] = (parsed_df['return_rate'] > 0).astype(int)
    
    # 自动优化阈值
    if PROFIT_WEIGHT_CONFIG['auto_optimize']:
        PROFIT_WEIGHT_CONFIG['thresholds'] = auto_detect_thresholds(
            parsed_df['return_rate'], 
            current_config
        )
    
    # 动态权重分配
    conditions = [
        parsed_df['return_rate'] <= 0,
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['small'],
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['medium'],
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['large'],
        True
    ]
    choices = [PROFIT_WEIGHT_CONFIG['weights'][k] for k in ['loss', 'small', 'medium', 'large', 'huge']]
    parsed_df['weight'] = np.select(conditions, choices)
    
    # 量价特征增强
    if '主生量' in parsed_df and '量基线' in parsed_df:
        parsed_df['trend_strength'] = parsed_df['主生量'] / parsed_df['量基线'].replace(0, 1e-6)
    if 'volume' in parsed_df and 'volume_ma5' in parsed_df:
        parsed_df['volume_ratio'] = parsed_df['volume'] / parsed_df['volume_ma5'].replace(0, 1e-6)
    
    return parsed_df

def auto_detect_thresholds(return_series, current_thresholds):
    """自动检测阈值（带边界保护）"""
    thresholds = current_thresholds.copy()
    
    # 收益率阈值优化
    positive_returns = return_series[return_series > 0]
    if len(positive_returns) >= 10:
        kde = stats.gaussian_kde(positive_returns)
        x = np.linspace(0, positive_returns.max(), 100)
        y = kde(x)
        inflection_points = np.where(np.diff(np.sign(np.gradient(y))))[0]
        
        if len(inflection_points) >= 1:
            thresholds['small'] = max(0.03, round(float(x[inflection_points[0]]), 3))
        if len(inflection_points) >= 2:
            thresholds['medium'] = round(float(x[inflection_points[1]]), 3)
        if len(inflection_points) >= 3:
            thresholds['large'] = round(float(x[inflection_points[2]]), 3)
    
    # Growth阈值优化
    if 'growth' in return_series.index:
        growth_series = return_series['growth'].dropna()
        if len(growth_series) > 10:
            q1, q3 = np.percentile(growth_series, [25, 75])
            iqr = q3 - q1
            thresholds['growth_min'] = max(current_thresholds['growth_min'], 
                                          round(q1 - 0.5*iqr, 1))
            thresholds['growth_max'] = min(current_thresholds['growth_max'], 
                                         round(q3 + 0.5*iqr, 1))
    
    # 边界保护
    thresholds.update({
        'volume_gain_threshold': np.clip(thresholds.get('volume_gain_threshold', 1.3), 1.1, 2.5),
        'volume_loss_threshold': np.clip(thresholds.get('volume_loss_threshold', -0.65), -1.0, -0.3),
        'growth_min': max(0.5, thresholds.get('growth_min', 1.0)),
        'growth_max': min(10.0, thresholds.get('growth_max', 5.0))
    })
    
    return thresholds

def analyze_features(final_df):
    """增强的特征分析"""
    # 数据校验
    assert {'is_profit', 'weight', 'return_rate'}.issubset(final_df.columns)
    
    # 特征选择
    feature_cols = [c for c in final_df.columns 
                   if c not in ['symbol', 'buy_date', 'sell_date', 'sell_price', 
                              'buy_price', 'signal', 'pnl', 'quantity', 
                              'is_profit', 'weight', 'return_rate']
                   and pd.api.types.is_numeric_dtype(final_df[c])]
    
    # 模型训练
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(final_df[feature_cols], final_df['is_profit'], 
             sample_weight=final_df['weight'])
    
    # 特征重要性
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Impact': model.coef_[0] * final_df[feature_cols].std()
    }).sort_values('Impact', key=abs, ascending=False)
    
    # 生成建议
    current_config = get_current_config()
    suggestions = generate_optimization_suggestions(
        importance,
        current_config['weights'],
        current_config['technical_weights'],
        current_config['capital_flow_weights']
    )
    
    return {
        'feature_importance': importance,
        'optimization_suggestions': suggestions
    }

def generate_optimization_suggestions(importance_df, main_weights, tech_weights, cf_weights):
    """生成优化建议（含配置验证）"""
    # 初始化建议模板
    suggestions = {
        'main': main_weights.copy(),
        'technical': tech_weights.copy(),
        'capital_flow': cf_weights.copy(),
        'thresholds': get_current_config()['thresholds'].copy()
    }
    
    # 特征映射表
    feature_map = {
        'technical': ('main', 'technical'),
        'capital_flow': ('main', 'capital_flow'),
        'market_heat': ('main', 'market_heat'),
        'ma_condition': ('technical', 'ma_condition'),
        'angle_condition': ('technical', 'angle_condition'),
        'macd_condition': ('technical', 'macd_condition'),
        'volume_condition': ('technical', 'volume_score'),
        'rsi_oversold': ('technical', 'rsi_oversold'),
        'kdj_oversold': ('technical', 'kdj_oversold'),
        'cci_oversold': ('technical', 'cci_oversold'),
        'bollinger_condition': ('technical', 'bollinger_condition'),
        'macd_jc': ('technical', 'macd_jc'),
        'money_flow_positive': ('capital_flow', 'positive_flow'),
        'money_flow_increasing': ('capital_flow', 'flow_increasing'),
        'money_flow_trend': ('capital_flow', 'trend_strength'),
        'money_flow_weekly': ('capital_flow', 'weekly_flow'),
        'money_flow_weekly_increasing': ('capital_flow', 'weekly_increasing'),
        '量增幅': ('capital_flow', 'volume_gain_ratio'),
        '量基线': ('capital_flow', 'volume_baseline'),
        '主生量': ('capital_flow', 'primary_volume'),
        '周增幅': ('capital_flow', 'weekly_growth'),
        'growth': ('technical', 'price_growth')
    }
    
    # 动态调整权重
    for _, row in importance_df.iterrows():
        feat = row['Feature']
        if feat in feature_map:
            category, key = feature_map[feat]
            adjustment = np.tanh(row['Impact'] * 0.5)  # 非线性调整
            suggestions[category][key] *= (1 + adjustment)
    
    # 权重归一化
    def normalize_weights(weights):
        total = sum(weights.values())
        return {k: round(v/total, 4) for k, v in weights.items()}
    
    suggestions['main'] = normalize_weights(suggestions['main'])
    suggestions['technical'] = normalize_weights(suggestions['technical'])
    suggestions['capital_flow'] = normalize_weights(suggestions['capital_flow'])
    
    # 阈值优化
    suggestions['thresholds'].update(
        auto_detect_thresholds(pd.Series(), suggestions['thresholds'])
    )
    
    # 配置验证
    validate_config(suggestions)
    
    # 生成配置代码
    print("\n=== 可替换配置 ===")
    print(f"optimized_config = {json.dumps(suggestions, indent=4)}")
    
    return suggestions

def validate_config(config):
    """配置校验"""
    thresholds = config['thresholds']
    assert thresholds['growth_min'] < thresholds['growth_max'], "growth阈值范围错误"
    assert 1.0 < thresholds['volume_gain_threshold'] < 3.0, "量增阈值异常"
    assert -1.0 < thresholds['volume_loss_threshold'] < 0, "量减阈值异常"
    
    weights = config['main']
    assert abs(sum(weights.values()) - 1.0) < 0.01, "主权重和不等于1"
    
# ======================
# 可视化功能
# ======================
def plot_analysis(final_df, importance_df, config):
    """增强可视化分析"""
    plt.figure(figsize=(20, 12))
    
    # 1. 权重分布
    plt.subplot(2, 2, 1)
    sns.boxplot(x='is_profit', y='weight', data=final_df)
    plt.title('权重分布对比')
    
    # 2. 特征重要性
    plt.subplot(2, 2, 2)
    sns.barplot(y='Feature', x='Impact', data=importance_df.head(10))
    plt.title('Top 10特征重要性')
    
    # 3. 阈值对比
    plt.subplot(2, 2, 3)
    threshold_names = ['volume_gain_threshold', 'volume_loss_threshold']
    current = get_current_config()['thresholds']
    optimized = config['thresholds']
    for i, name in enumerate(threshold_names):
        plt.bar([i-0.2, i+0.2], [current[name], optimized[name]], 
               width=0.4, label=name)
    plt.legend()
    plt.title('阈值优化对比')
    
    # 4. 收益率分布
    plt.subplot(2, 2, 4)
    sns.histplot(final_df['return_rate'], bins=50, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('收益率分布')
    
    plt.tight_layout()
    plt.show()

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 初始化设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 数据加载
    paired_df = load_and_pair_transactions("./backtestresult/")
    print(f"成功加载 {len(paired_df)} 组配对交易")
    
    # 2. 特征工程
    parsed_df = parse_buy_signals(paired_df)
    final_df = prepare_features(parsed_df)
    
    # 3. 特征分析
    results = analyze_features(final_df)
    
    # 4. 结果展示
    print("\n特征重要性:")
    print(tabulate(results['feature_importance'].head(10), headers="keys", tablefmt="psql"))
    
    # 5. 可视化
    plot_analysis(final_df, 
                 results['feature_importance'], 
                 results['optimization_suggestions'])