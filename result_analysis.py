# result_analysis.py
import ast
import os
import pandas as pd
import re
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from utils.strategy import StockScorer

def safe_literal_eval(value):
    """安全解析字典字符串，处理异常和空值"""
    try:
        if pd.isna(value) or value == "":
            return {}
        # 处理可能存在的单引号问题
        value = str(value).replace("'", "\"")
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"解析失败: {value}，错误: {str(e)}")
        return {}

def load_and_pair_transactions(directory_path):
    """加载交易数据并匹配买卖记录"""
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    all_data = pd.DataFrame()
    
    for file_name in all_files:
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_excel(file_path, sheet_name='Transactions')
        df = df[df['symbol'].notna()]
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    buys = all_data[all_data['type'] == 'buy'].copy()
    sells = all_data[all_data['type'] == 'sell'].copy()
    
    paired_data = []
    for symbol in sells['symbol'].unique():
        sell_group = sells[sells['symbol'] == symbol].sort_values('date')
        buy_group = buys[buys['symbol'] == symbol].sort_values('date')
        buy_idx = 0
        
        for _, sell_row in sell_group.iterrows():
            while buy_idx < len(buy_group):
                buy_row = buy_group.iloc[buy_idx]
                if (buy_row['date'] < sell_row['date']) and (buy_row['quantity'] == sell_row['quantity']):
                    merged = {
                        'symbol': symbol,
                        'buy_date': buy_row['date'],
                        'sell_date': sell_row['date'],
                        'pnl': sell_row['pnl'],
                        'signal': ast.literal_eval(buy_row['signal']),
                        'quantity': buy_row['quantity']
                    }
                    paired_data.append(merged)
                    buy_idx += 1
                    break
                else:
                    buy_idx += 1
    return pd.DataFrame(paired_data)

def parse_buy_signals(paired_df):
    """解析信号字典，覆盖所有策略字段"""
    # 动态获取字段映射（与 strategy.py 的评分规则严格一致）
    feature_mapping = {
        'technical': [
            'ma_condition', 'angle_condition', 'macd_condition',
            'volume_condition', 'rsi_oversold', 'kdj_oversold',
            'cci_oversold', 'bollinger_condition'
        ],
        'capital_flow': [
            'money_flow_positive', 'money_flow_increasing',
            'money_flow_trend', 'money_flow_weekly',
            'money_flow_weekly_increasing', '量基线', '主生量'
        ],
        'market_heat': ['market_heat'],
        
    }
    
    # 创建特征列，处理嵌套字段和缺失值
    for category in feature_mapping:
        for feat in feature_mapping[category]:
            paired_df[feat] = paired_df['signal'].apply(lambda x: x.get(feat, 0) if isinstance(x, dict) else 0)
            # 转换布尔值为数值
            if paired_df[feat].dtype == bool:
                paired_df[feat] = paired_df[feat].astype(int)
    # 在parse_buy_signals函数末尾添加：
    for category in feature_mapping:
        # 计算主特征总分（假设子特征已转换为数值）
        paired_df[category] = paired_df[feature_mapping[category]].sum(axis=1)
    
    return paired_df

def prepare_features(parsed_df):
    """特征工程，包含字段存在性检查"""
    parsed_df['is_profit'] = parsed_df['pnl'].apply(lambda x: 1 if x > 0 else 0)
    
    # 复合特征计算
    if '主生量' in parsed_df and '量基线' in parsed_df:
        parsed_df['trend_strength'] = parsed_df['主生量'] / parsed_df['量基线'].replace(0, 1e-6)
    if 'volume' in parsed_df and 'volume_ma5' in parsed_df:
        parsed_df['volume_ratio'] = parsed_df['volume'] / parsed_df['volume_ma5'].replace(0, 1e-6)
    
    # 选择最终特征（涵盖所有信号）
    features = [
        # 主维度
        'technical', 'capital_flow', 'market_heat',
        
        # 技术面子项
        'ma_condition', 'angle_condition', 'macd_condition',
        'volume_condition', 'rsi_oversold', 'kdj_oversold',
        'cci_oversold', 'bollinger_condition',
        
        # 资金流子项
        'money_flow_positive', 'money_flow_increasing',
        'money_flow_trend', 'money_flow_weekly',
        'money_flow_weekly_increasing',
        
        # 复合特征
        'trend_strength', 'volume_ratio'
    ]
    # 仅保留存在的字段
    selected_features = [f for f in features if f in parsed_df]
    return parsed_df[selected_features + ['is_profit']]

def analyze_loss_signals(parsed_df):
    """分析亏损交易的特征分布"""
    loss_df = parsed_df[parsed_df['is_profit'] == 0].copy()
    profit_df = parsed_df[parsed_df['is_profit'] == 1].copy()

    # 计算特征差异
    feature_diff = {}
    for col in parsed_df.columns:
        if col in ['is_profit', 'pnl', 'buy_date', 'sell_date']:
            continue
        if parsed_df[col].dtype in [int, float]:
            profit_mean = profit_df[col].mean()
            loss_mean = loss_df[col].mean()
            feature_diff[col] = profit_mean - loss_mean

    return pd.Series(feature_diff).sort_values(ascending=False)

def analyze_features(final_df):
    """分析特征并生成优化建议"""
    # 新增亏损特征分析
    feature_diff = analyze_loss_signals(final_df)
    print("\n=== 盈亏特征差异分析 ===")
    print(feature_diff)
    """分析特征并生成优化建议"""
    # 1. 相关性分析
    corr_matrix = final_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('特征与盈利相关性热力图')
    plt.show()
    
    # 2. 逻辑回归建模
    X = final_df.drop('is_profit', axis=1)
    y = final_df['is_profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 3. 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    print("\n=== 特征重要性排序 ===")
    print(feature_importance)
    
    # 4. 动态获取权重配置
    scorer = StockScorer()
    current_weights = scorer.config['weights'].copy()
    capital_flow_weights = scorer.config['capital_flow_weights'].copy()
    technical_weights = scorer.config['technical_weights'].copy()
    
    # 5. 生成优化建议
    weight_suggestions, signal_suggestions = generate_optimization_suggestions(
        feature_importance, feature_diff, current_weights,  # 新增 feature_diff
        capital_flow_weights, technical_weights
    )
    return feature_importance, weight_suggestions, signal_suggestions

def generate_optimization_suggestions(feature_importance,feature_diff, current_weights, capital_flow_weights,technical_weights):
    """生成权重调整建议"""
    POSITIVE_ADJUST = 0.05
    NEGATIVE_ADJUST = -0.05
    
    feature_mapping = {
        'technical': ('weights', 'technical'),
        'capital_flow': ('weights', 'capital_flow'),
        'market_heat': ('weights', 'market_heat'),
        'ma_condition': ('technical_weights', 'ma_condition'),
        'angle_condition': ('technical_weights', 'angle_condition'),
        'macd_condition': ('technical_weights', 'macd_condition'),
        'volume_condition': ('technical_weights', 'volume_score'),
        'rsi_oversold': ('technical_weights', 'rsi_oversold'),
        'kdj_oversold': ('technical_weights', 'kdj_oversold'),
        'cci_oversold': ('technical_weights', 'cci_oversold'),
        'bollinger_condition': ('technical_weights', 'bollinger_condition'),
        'money_flow_positive': ('capital_flow_weights', 'positive_flow'),
        'money_flow_trend': ('capital_flow_weights', 'trend_strength'),
        'trend_strength': ('scoring_rules', 'max_trend_strength')
    }
    
    weight_suggestions = current_weights.copy()
    cf_weight_suggestions = capital_flow_weights.copy()
    tech_weight_suggestions = technical_weights.copy()
    signal_suggestions = {'add': [], 'remove': []}
    
    for _, row in feature_importance.iterrows():
        feat = row['Feature']
        imp = row['Importance']
        
        if feat in feature_mapping:
            category, sub_feature = feature_mapping[feat]
            # 根据特征重要性调整权重
            if imp > 0.1:  # 正向调整
                if category == 'weights':
                    # 主权重调整
                    weight_suggestions[sub_feature] += POSITIVE_ADJUST
                elif category == 'technical_weights':
                    # technical_weights 子项调整
                    tech_weight_suggestions[sub_feature] += POSITIVE_ADJUST
                elif category == 'capital_flow_weights':
                    # capital_flow_weights 子项调整
                    cf_weight_suggestions[sub_feature] += POSITIVE_ADJUST
            
            elif imp < -0.1:  # 负向调整
                if category == 'weights':
                    weight_suggestions[sub_feature] += NEGATIVE_ADJUST
                elif category == 'technical_weights':
                    tech_weight_suggestions[sub_feature] += NEGATIVE_ADJUST
                elif category == 'capital_flow_weights':
                    cf_weight_suggestions[sub_feature] += NEGATIVE_ADJUST
    # 新增基于特征差异的调整
    DIFF_THRESHOLD = 0.2  # 特征差异阈值
    for feat, diff in feature_diff.items():
        if feat in feature_mapping:
            category, sub_feature = feature_mapping[feat]
            
            # 根据差异方向调整权重
            if diff > DIFF_THRESHOLD:
                adjust = min(0.1, diff * 0.05)
                if category == 'weights':
                    weight_suggestions[sub_feature] += adjust
                elif category == 'technical_weights':
                    tech_weight_suggestions[sub_feature] += adjust
                elif category == 'capital_flow_weights':
                    cf_weight_suggestions[sub_feature] += adjust
            elif diff < -DIFF_THRESHOLD:
                adjust = max(-0.1, diff * 0.05)
                if category == 'weights':
                    weight_suggestions[sub_feature] += adjust
                elif category == 'technical_weights':
                    tech_weight_suggestions[sub_feature] += adjust
                elif category == 'capital_flow_weights':
                    cf_weight_suggestions[sub_feature] += adjust

    # 归一化处理
    def normalize_sub_weights(weights, max_total=1.0):
        total = sum(weights.values())
        if total == 0:
            return {k: round(v, 2) for k, v in weights.items()}
        adjusted = {k: round(v / total * max_total, 2) for k, v in weights.items()}
        # 处理四舍五入误差
        if sum(adjusted.values()) != max_total:
            max_key = max(adjusted, key=lambda x: adjusted[x])
            adjusted[max_key] += round(max_total - sum(adjusted.values()), 2)
        return adjusted
    
    # 对所有子权重归一化
    tech_weight_suggestions = normalize_sub_weights(tech_weight_suggestions)
    cf_weight_suggestions = normalize_sub_weights(cf_weight_suggestions)

    # 新增：对主权重归一化
    main_total = sum(weight_suggestions.values())
    if main_total != 1.0:
        weight_suggestions = {
            k: round(v / main_total, 2) for k, v in weight_suggestions.items()
        }
    # 处理四舍五入误差
    diff = 1.0 - sum(weight_suggestions.values())
    max_key = max(weight_suggestions, key=lambda x: weight_suggestions[x])
    weight_suggestions[max_key] += round(diff, 2)
    
    return {
        'main': weight_suggestions,
        'technical': tech_weight_suggestions,  # 新增 technical_weights 建议
        'capital_flow': cf_weight_suggestions
    }, signal_suggestions

if __name__ == "__main__":
    paired_df = load_and_pair_transactions("./backtestresult/")
    print(f"成功匹配 {len(paired_df)} 笔交易")
    
    parsed_df = parse_buy_signals(paired_df)
    final_df = prepare_features(parsed_df)
    
    feature_importance, weight_suggestions, signal_suggestions = analyze_features(final_df)
    
    print("\n=== 自动优化建议 ===")
    print("\n主权重调整建议:")
    print(json.dumps(weight_suggestions['main'], indent=4))
    print("\n技术面子权重调整:")  # 新增
    print(json.dumps(weight_suggestions['technical'], indent=4))
    print("\n资金流子权重调整:")
    print(json.dumps(weight_suggestions['capital_flow'], indent=4))
    if signal_suggestions['add']:
        print("\n建议新增信号:")
        for feat in signal_suggestions['add']:
            print(f"- {feat}")
    if signal_suggestions['remove']:
        print("\n建议移除信号:")
        for feat in signal_suggestions['remove']:
            print(f"- {feat}")
    
    print("\n配置代码片段:")
    print(f"'weights': {json.dumps(weight_suggestions['main'], indent=4)}")
    print(f"'technical_weights': {json.dumps(weight_suggestions['technical'], indent=4)}")
    print(f"'capital_flow_weights': {json.dumps(weight_suggestions['capital_flow'], indent=4)}")