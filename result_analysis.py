#result_analysis.py
import pandas as pd
import re
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from utils.strategy import StockScorer

# 1. 数据加载与预处理
def load_and_pair_transactions(directory_path):
    # 遍历目录中的所有文件
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    # 初始化一个空的 DataFrame 用于存储所有文件的数据
    all_data = pd.DataFrame()
    
    # 遍历每个文件
    for file_name in all_files:
        file_path = os.path.join(directory_path, file_name)
        
        # 加载单个文件的数据
        df = pd.read_excel(file_path, sheet_name='Transactions')
        df = df[df['symbol'].notna()].reset_index(drop=True)
        
        # 将单个文件的数据合并到总的 DataFrame 中
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # 分离买入和卖出记录
    buys = all_data[all_data['type'] == 'buy'].copy()
    sells = all_data[all_data['type'] == 'sell'].copy()
    
    # 按股票代码分组并排序
    buys_grouped = buys.sort_values('date').groupby('symbol')
    sells_grouped = sells.sort_values('date').groupby('symbol')
    
    paired_data = []
    
    # 匹配买卖交易（FIFO原则）
    for symbol, sell_group in sells_grouped:
        if symbol not in buys_grouped.groups:
            continue  # 没有对应的买入记录
        
        buy_group = buys_grouped.get_group(symbol)
        buy_idx = 0
        
        for _, sell_row in sell_group.iterrows():
            # 查找匹配的买入记录（同一股票，卖出时间晚于买入）
            while buy_idx < len(buy_group):
                buy_row = buy_group.iloc[buy_idx]
                if buy_row['date'] < sell_row['date'] and \
                   buy_row['quantity'] == sell_row['quantity']:
                    # 匹配成功，合并记录
                    merged = {
                        'symbol': symbol,
                        'buy_date': buy_row['date'],
                        'sell_date': sell_row['date'],
                        'pnl': sell_row['pnl'],
                        'signal_info': buy_row['signal_info'],  # 仅保留买入时的信号
                        'quantity': buy_row['quantity']
                    }
                    paired_data.append(merged)
                    buy_idx += 1
                    break
                else:
                    buy_idx += 1
    
    return pd.DataFrame(paired_data)

# 2. 信号解析（仅解析买入时的信号）
def parse_buy_signals(paired_df):
    # 定义正则表达式模式
    patterns = {
        'tech_score': r'技术面\((\d+\.\d+)',
        'funds_score': r'资金流向分析\((\d+\.\d+)',
        'heat_score': r'市场热度分析（热度值：(\d+\.\d+)）',
        'macd_cross': r'MACD金叉信号: (存在|不存在)',
        'main_fund_direction': r'主力资金方向: (\S+)',
        'short_term_trend': r'短期趋势: (\S+)',
        'bollinger_position': r'当前价格(.*?)布林带'
    }
    
    # 初始化存储解析结果的列
    for col in patterns.keys():
        paired_df[col] = None
    
    # 仅解析买入时的信号
    for idx, row in paired_df.iterrows():
        text = row['signal_info']
        
        # 技术面评分
        tech_match = re.search(patterns['tech_score'], text)
        paired_df.at[idx, 'tech_score'] = float(tech_match.group(1)) if tech_match else 0.0
        
        # 资金流向评分
        funds_match = re.search(patterns['funds_score'], text)
        paired_df.at[idx, 'funds_score'] = float(funds_match.group(1)) if funds_match else 0.0
        
        # 市场热度评分
        heat_match = re.search(patterns['heat_score'], text)
        paired_df.at[idx, 'heat_score'] = float(heat_match.group(1)) if heat_match else 0.0
        
        # MACD金叉信号
        macd_match = re.search(patterns['macd_cross'], text)
        paired_df.at[idx, 'macd_cross'] = 1 if macd_match and macd_match.group(1) == '存在' else 0
        
        # 主力资金方向
        fund_dir_match = re.search(patterns['main_fund_direction'], text)
        paired_df.at[idx, 'main_fund_direction'] = fund_dir_match.group(1) if fund_dir_match else 'unknown'
        
        # 短期趋势
        trend_match = re.search(patterns['short_term_trend'], text)
        paired_df.at[idx, 'short_term_trend'] = trend_match.group(1) if trend_match else 'unknown'
        
        # 布林带位置
        boll_match = re.search(patterns['bollinger_position'], text)
        if boll_match:
            position = '超买' if '上轨' in boll_match.group(0) else \
                     '超卖' if '下轨' in boll_match.group(0) else '正常'
            paired_df.at[idx, 'bollinger_position'] = position
        else:
            paired_df.at[idx, 'bollinger_position'] = 'unknown'
    
    return paired_df

# 3. 特征工程
def prepare_features(parsed_df):
    # 目标变量：是否盈利
    parsed_df['is_profit'] = parsed_df['pnl'].apply(lambda x: 1 if x > 0 else 0)
    
    # 分类变量编码
    categorical_cols = ['main_fund_direction', 'short_term_trend', 'bollinger_position']
    le = LabelEncoder()
    for col in categorical_cols:
        parsed_df[col] = le.fit_transform(parsed_df[col])
    
    # 选择特征列
    features = ['tech_score', 'funds_score', 'heat_score', 'macd_cross',
               'main_fund_direction', 'short_term_trend', 'bollinger_position']
    
    return parsed_df[features + ['is_profit']]

# 4. 分析与建模
def analyze_features(final_df):

    # 相关性热力图
    corr_matrix = final_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('买入信号与盈利相关性')
    plt.show()
    
    # 逻辑回归分析
    X = final_df.drop('is_profit', axis=1)
    y = final_df['is_profit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    print("\n=== 特征重要性分析 ===")
    print(feature_importance)

    # 新增：动态获取当前权重配置
    scorer = StockScorer()  # 实例化 StockScorer
    current_weights = scorer.config['weights']
    capital_flow_weights = scorer.config.get('capital_flow_weights', {})
    # 生成优化建议
    weight_suggestions, signal_suggestions = generate_optimization_suggestions(
        feature_importance, 
        current_weights,
        capital_flow_weights
    )
    
    return feature_importance, weight_suggestions, signal_suggestions
    
def generate_optimization_suggestions(feature_importance, current_weights, capital_flow_weights):
    """根据特征重要性生成优化建议"""
    # 权重调整规则
    POSITIVE_ADJUST = 0.05
    NEGATIVE_ADJUST = -0.05
    
    # 映射特征到权重类别
    feature_mapping = {
        'tech_score': ('technical', None),
        'funds_score': ('capital_flow', None),
        'heat_score': ('market_heat', None),
        'macd_cross': ('technical', 'macd_cross'),
        'bollinger_position': ('technical', 'bollinger'),
        'main_fund_direction': ('capital_flow', 'positive_flow'),
        'flow_increasing': ('capital_flow', 'flow_increasing'),
        'weekly_flow': ('capital_flow', 'weekly_flow')
    }
    
    # 初始化建议配置
    weight_suggestions = current_weights.copy()
    cf_weight_suggestions = capital_flow_weights.copy()
    signal_suggestions = {'add': [], 'remove': []}
    
    for _, row in feature_importance.iterrows():
        feat = row['Feature']
        importance = row['Importance']
        
        # 主权重调整
        if feat in feature_mapping:
            category, sub_feature = feature_mapping[feat]
            if importance > 0.1:
                if sub_feature:  # 处理子权重
                    cf_weight_suggestions[sub_feature] = min(
                        cf_weight_suggestions.get(sub_feature, 0) + POSITIVE_ADJUST, 
                        0.4  # 防止子项权重过高
                    )
                else:  # 处理主权重
                    weight_suggestions[category] += POSITIVE_ADJUST
            elif importance < -0.1:
                if sub_feature:
                    cf_weight_suggestions[sub_feature] = max(
                        cf_weight_suggestions.get(sub_feature, 0) + NEGATIVE_ADJUST,
                        0.0  # 防止负权重
                    )
                else:
                    weight_suggestions[category] += NEGATIVE_ADJUST
        
        # 信号增删建议
        if importance > 0.2 and feat not in current_weights:
            signal_suggestions['add'].append(feat)
        elif importance < -0.2 and feat in current_weights.values():
            signal_suggestions['remove'].append(feat)
    
    # 权重归一化处理
    total_main = sum(weight_suggestions.values())
    for k in weight_suggestions:
        weight_suggestions[k] = round(weight_suggestions[k]/total_main, 2)
        
    total_cf = sum(cf_weight_suggestions.values())
    for k in cf_weight_suggestions:
        cf_weight_suggestions[k] = round(cf_weight_suggestions[k]/total_cf, 2)
    
    return {'main': weight_suggestions, 'capital_flow': cf_weight_suggestions}, signal_suggestions


# 主流程
if __name__ == "__main__":
    # 数据加载与匹配
    paired_df = load_and_pair_transactions("./backtestresult/")
    print(f"成功匹配 {len(paired_df)} 笔完整交易")
    
    # 信号解析
    parsed_df = parse_buy_signals(paired_df)
    
    # 特征工程
    final_df = prepare_features(parsed_df)
    
    # 获取分析结果
    feature_importance, weight_suggestions, signal_suggestions = analyze_features(final_df)
    
    # 输出优化建议
    print("\n=== 自动生成的优化建议 ===")
    
    # 1. 主权重调整建议
    print("\n主权重调整建议（strategy.py 的 weights 配置）:")
    print("当前配置:", json.dumps(StockScorer().config['weights'], indent=4))
    print("建议调整:")
    print(json.dumps(weight_suggestions['main'], indent=4))
    
    # 2. 资金流子权重建议
    print("\n资金流向子项调整（capital_flow_weights 配置）:")
    print("当前配置:", json.dumps(StockScorer().config['capital_flow_weights'], indent=4))
    print("建议调整:")
    print(json.dumps(weight_suggestions['capital_flow'], indent=4))
    
    # 3. 信号增删建议
    print("\n信号优化建议:")
    if signal_suggestions['add']:
        print("需增加的信号:")
        for feat in signal_suggestions['add']:
            imp = feature_importance[feature_importance['Feature'] == feat]['Importance'].values[0]
            print(f"  - {feat}（重要性: {imp:.2f}）")
    if signal_suggestions['remove']:
        print("\n需移除的信号:")
        for feat in signal_suggestions['remove']:
            imp = feature_importance[feature_importance['Feature'] == feat]['Importance'].values[0]
            print(f"  - {feat}（重要性: {imp:.2f}）")
    
    # 4. 配置代码示例
    print("\n配置代码修改示例（更新 strategy.py）:")
    print('''
    # strategy.py 修改部分
    'weights': %s,
    'capital_flow_weights': %s
    ''' % (
        json.dumps(weight_suggestions['main'], indent=4),
        json.dumps(weight_suggestions['capital_flow'], indent=4)
    ))
# 使用说明：
# 1. 依赖库：pandas, openpyxl, scikit-learn, matplotlib, seaborn
# 2. 确保Excel文件路径正确
# 3. 输出包含：
#    - 交易匹配数量
#    - 特征相关性热力图
#    - 逻辑回归特征重要性排序
#    - 优化建议