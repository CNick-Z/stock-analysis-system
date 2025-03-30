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
from utils.strategy import StockScorer
from tabulate import tabulate

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
    'auto_optimize': True  # 是否自动优化阈值
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
# # ======================
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
    """解析信号字典，严格对齐strategy.py的输出"""
    # 信号特征映射（主类别 -> 子特征）
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
    for category in ['technical', 'capital_flow']:
        paired_df[category] = paired_df[feature_mapping[category]].sum(axis=1)
    
    return paired_df

def prepare_features(parsed_df):
    """特征工程（含动态权重计算）"""
    # 计算盈利率
    parsed_df['return_rate'] = (parsed_df['sell_price'] - parsed_df['buy_price']) / parsed_df['buy_price']
    parsed_df['is_profit'] = (parsed_df['return_rate'] > 0).astype(int)
    
    # 动态权重分配
    if PROFIT_WEIGHT_CONFIG['auto_optimize']:
        PROFIT_WEIGHT_CONFIG['thresholds'] = auto_detect_thresholds(parsed_df['return_rate'])
    
    conditions = [
        parsed_df['return_rate'] <= 0,
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['small'],
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['medium'],
        parsed_df['return_rate'] < PROFIT_WEIGHT_CONFIG['thresholds']['large'],
        True
    ]
    choices = [PROFIT_WEIGHT_CONFIG['weights'][k] for k in ['loss', 'small', 'medium', 'large', 'huge']]
    parsed_df['weight'] = np.select(conditions, choices)

    if '主生量' in parsed_df and '量基线' in parsed_df:
        parsed_df['trend_strength'] = parsed_df['主生量'] / parsed_df['量基线'].replace(0, 1e-6)
    if 'volume' in parsed_df and 'volume_ma5' in parsed_df:
        parsed_df['volume_ratio'] = parsed_df['volume'] / parsed_df['volume_ma5'].replace(0, 1e-6)
    return parsed_df



def auto_detect_thresholds(return_series):
    """自动检测盈利阈值"""
    positive_returns = return_series[return_series > 0]
    if len(positive_returns) < 10:
        return PROFIT_WEIGHT_CONFIG['thresholds']
    
    kde = stats.gaussian_kde(positive_returns)
    x = np.linspace(0, positive_returns.max(), 100)
    y = kde(x)
    inflection_points = np.where(np.diff(np.sign(np.gradient(y))))[0]
    
    thresholds = PROFIT_WEIGHT_CONFIG['thresholds'].copy()
    if len(inflection_points) >= 1:
        thresholds['small'] = round(float(x[inflection_points[0]]), 3)
    if len(inflection_points) >= 2:
        thresholds['medium'] = round(float(x[inflection_points[1]]), 3)
    if len(inflection_points) >= 3:
        thresholds['large'] = round(float(x[inflection_points[2]]), 3)
    
    return thresholds

def get_feature_columns(df, exclude_cols=None):
    """获取用于分析的特征列"""
    if exclude_cols is None:
        exclude_cols = ['symbol', 'buy_date', 'sell_date', 'sell_price', 
                       'buy_price', 'signal', 'pnl', 'quantity', 
                       'is_profit', 'weight', 'return_rate']
    
    return [c for c in df.columns 
           if c not in exclude_cols
           and pd.api.types.is_numeric_dtype(df[c])]

def analyze_features(final_df):
    """完整特征分析流程"""
    # 数据校验
    required_cols = {'is_profit', 'weight', 'return_rate'}
    assert required_cols.issubset(final_df.columns)
    
    # 特征选择
    feature_cols = get_feature_columns(final_df)
    
    # 模型训练
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(final_df[feature_cols], final_df['is_profit'], sample_weight=final_df['weight'])
    
    # 特征重要性
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Impact': model.coef_[0] * final_df[feature_cols].std()
    }).sort_values('Impact', key=abs, ascending=False)
    
    # 生成建议
    scorer = StockScorer()
    suggestions = generate_optimization_suggestions(
        importance, 
        scorer.config['weights'],
        scorer.config['technical_weights'],
        scorer.config['capital_flow_weights']
    )
    
    return {
        'feature_importance': importance,
        'optimization_suggestions': suggestions
    }


def plot_analysis_results(df, feature_importance, feature_diff, top_n=5):
    """增强的可视化分析"""
    plt.figure(figsize=(20, 8))
    
    # 1. 权重分布
    plt.subplot(2, 2, 1)
    sns.boxplot(x='is_profit', y='weight', data=df)
    plt.title('盈利/亏损交易的权重分布', fontsize=12)
    plt.xlabel('是否盈利', fontsize=10)
    plt.ylabel('权重', fontsize=10)
    plt.xticks([0, 1], ['亏损', '盈利'])
    
    # 2. 特征重要性
    plt.subplot(2, 2, 2)
    sns.barplot(
        y='Feature', x='Impact', 
        data=feature_importance.head(top_n),
        palette='viridis'
    )
    plt.title(f'最重要的 {top_n} 个预测特征', fontsize=12)
    plt.xlabel('影响力', fontsize=10)
    plt.ylabel('特征', fontsize=10)
    
    # 3. 原始指标分布
    plt.subplot(2, 2, 3)
    raw_indicators = ['量增幅', '量基线', '主生量', '周增幅', 'growth', 'market_heat']
    for indicator in raw_indicators:
        if indicator in df.columns:
            sns.kdeplot(
                data=df, x=indicator, hue='is_profit',
                label=indicator, fill=True, alpha=0.3
            )
    plt.title('原始指标分布 (按盈利/亏损)', fontsize=12)
    plt.xlabel('指标值', fontsize=10)
    plt.ylabel('密度', fontsize=10)
    plt.legend(title='指标', fontsize=8)
    
    # 4. 最优阈值分析
    plt.subplot(2, 2, 4)
    threshold_features = [f for f in feature_diff.index 
                         if f in ['量增幅', '周增幅', 'growth']]
    for feat in threshold_features:
        profit_mean = feature_diff.loc[feat, 'profit_mean']
        loss_mean = feature_diff.loc[feat, 'loss_mean']
        plt.axvline(x=profit_mean, color='g', linestyle='--', label=f'{feat} 盈利均值')
        plt.axvline(x=loss_mean, color='r', linestyle='--', label=f'{feat} 亏损均值')
        plt.axvline(x=(profit_mean+loss_mean)/2, color='b', label=f'{feat} 最优阈值')
    plt.title('最优阈值分析', fontsize=12)
    plt.xlabel('指标值', fontsize=10)
    plt.ylabel('', fontsize=10)
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

def initialize_weights(base_weights, required_keys):
    """权重字典安全初始化"""
    return {k: base_weights.get(k, 0.1) for k in required_keys}
#def generate_optimization_suggestions(feature_importance, feature_diff, current_weights, capital_flow_weights, technical_weights):
def generate_optimization_suggestions(feature_importance, main_weights, tech_weights, cf_weights):
    """生成权重优化建议（健壮版）"""
    # 权重字典初始化（确保所有键存在）
    suggestions = {
        'main': initialize_weights(main_weights, [
            'technical', 'capital_flow', 'market_heat'
        ]),
        'technical': initialize_weights(tech_weights, [
            'ma_condition', 'angle_condition', 'macd_condition',
            'volume_score', 'rsi_oversold', 'kdj_oversold',
            'cci_oversold', 'bollinger_condition', 'macd_jc',
            'price_growth'
        ]),
        'capital_flow': initialize_weights(cf_weights, [
            'positive_flow', 'flow_increasing', 'trend_strength',
            'weekly_flow', 'weekly_increasing', 'volume_gain_ratio',
            'volume_baseline', 'primary_volume', 'weekly_growth'
        ]),
        'threshold_suggestions': {}
    }
    signal_suggestions = {'add': [], 'remove': [], 'adjust': []}

    # 完整的特征映射表（与strategy.py严格一致）
    feature_mapping = {
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
    # 权重调整
    for _, row in feature_importance.iterrows():
        feat = row['Feature']
        if feat in feature_mapping:
            category, key = feature_mapping[feat]
            adjust = np.clip(row['Impact'] * 0.1, -0.2, 0.2)  # 限制调整幅度
            suggestions[category][key] += adjust
    # 归一化处理并格式化
    def normalize_and_format_weights(weights_dict):
        total = sum(weights_dict.values())
        if total > 0:
            normalized = {k: round(v/total, 4) for k, v in weights_dict.items()}
            # 确保总和为1（处理浮点精度问题）
            diff = 1.0 - sum(normalized.values())
            if diff != 0:
                max_key = max(normalized.items(), key=lambda x: x[1])[0]
                normalized[max_key] = round(normalized[max_key] + diff, 4)
            return normalized
        return {k: round(v, 4) for k, v in weights_dict.items()}
    # 对每个类别进行归一化
    suggestions['main'] = normalize_and_format_weights(suggestions['main'])
    suggestions['technical'] = normalize_and_format_weights(suggestions['technical'])
    suggestions['capital_flow'] = normalize_and_format_weights(suggestions['capital_flow'])
    # 阈值建议
    for indicator in ['量增幅', '周增幅', 'growth']:
        if indicator in feature_importance['Feature'].values:
            suggestions['threshold_suggestions'][f"{indicator}_threshold"] = round(
                feature_importance[feature_importance['Feature'] == indicator]['Impact'].iloc[0] * 0.5, 3)
    
    return suggestions
    
# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 初始化字体设置
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    mpl.rcParams['font.size'] = 12  # 设置默认字体大小
    
    # 初始化 StockScorer
    scorer = StockScorer()
    
    # 1. 数据加载
    paired_df = load_and_pair_transactions("./backtestresult/")
    print(f"Loaded {len(paired_df)} paired transactions")
    
    # 2. 特征工程
    parsed_df = parse_buy_signals(paired_df)
    final_df = prepare_features(parsed_df)
    
    # 3. 分析执行
    results = analyze_features(final_df)
    
    # 4. 结果输出
    print("\n=== 特征重要性 ===")
    print(tabulate(results['feature_importance'].head(10), headers="keys", tablefmt="psql"))
    print("\n=== 核心优化建议 ===")
    # 输出优化建议
    suggestions = results['optimization_suggestions']
    
    print("\n主权重调整建议:")
    print(tabulate(
        pd.DataFrame({
            '参数': list(suggestions['main'].keys()),
            '当前值': [scorer.config['weights'].get(k, 'N/A') for k in suggestions['main'].keys()],
            '建议值': list(suggestions['main'].values())
        }),
        headers='keys', tablefmt='psql'
    ))
    
    print("\n技术指标权重调整建议:")
    print(tabulate(
        pd.DataFrame({
            '参数': list(suggestions['technical'].keys()),
            '当前值': [scorer.config['technical_weights'].get(k, 'N/A') for k in suggestions['technical'].keys()],
            '建议值': list(suggestions['technical'].values())
        }),
        headers='keys', tablefmt='psql'
    ))
    
    print("\n资金流权重调整建议:")
    print(tabulate(
        pd.DataFrame({
            '参数': list(suggestions['capital_flow'].keys()),
            '当前值': [scorer.config['capital_flow_weights'].get(k, 'N/A') for k in suggestions['capital_flow'].keys()],
            '建议值': list(suggestions['capital_flow'].values())
        }),
        headers='keys', tablefmt='psql'
    ))
    
    if suggestions['threshold_suggestions']:
        print("\n=== 阈值优化建议 ===")
        print(tabulate(
            pd.DataFrame(suggestions['threshold_suggestions'].items(),
                       columns=['Indicator', 'Suggested Threshold']),
            tablefmt='psql'
        ))
    # 5. 可直接复制的配置代码
    print("\n配置代码片段:")
    print(f"'weights': {json.dumps(suggestions['main'], indent=4)}")
    print(f"'technical_weights': {json.dumps(suggestions['technical'], indent=4)}")
    print(f"'capital_flow_weights': {json.dumps(suggestions['capital_flow'], indent=4)}")
    # 6. 可视化分析
    # 计算盈利/亏损组的特征均值差异
    profit_mean = final_df[final_df['is_profit'] == 1].mean(numeric_only=True)
    loss_mean = final_df[final_df['is_profit'] == 0].mean(numeric_only=True)
    feature_diff = pd.DataFrame({
        'profit_mean': profit_mean,
        'loss_mean': loss_mean
    })
    
    # 调用可视化函数
    plot_analysis_results(final_df, results['feature_importance'], feature_diff)
    
    # 6. 新增可视化 - 盈利率分布
    plt.figure(figsize=(12, 6))
    sns.histplot(final_df['return_rate'], bins=50, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('收益率分布', fontsize=14)
    plt.xlabel('收益率', fontsize=12)
    plt.ylabel('交易数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    # 7. 新增可视化 - 特征相关性热图
    numeric_cols = final_df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(16, 12))
    sns.heatmap(final_df[get_feature_columns(final_df)].corr(), annot=True, 
           fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 8})
    plt.title('特征相关性热图', fontsize=14)
    plt.tight_layout()
    plt.show()