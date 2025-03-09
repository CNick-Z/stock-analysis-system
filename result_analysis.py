#result_analysis.py
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与预处理
def load_and_pair_transactions(file_path):
    # 加载原始数据
    df = pd.read_excel(file_path, sheet_name='Transactions')
    df = df[df['symbol'].notna()].reset_index(drop=True)
    
    # 分离买入和卖出记录
    buys = df[df['type'] == 'buy'].copy()
    sells = df[df['type'] == 'sell'].copy()
    
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
    
    return feature_importance

# 主流程
if __name__ == "__main__":
    # 数据加载与匹配
    paired_df = load_and_pair_transactions("trades_report.xlsx")
    print(f"成功匹配 {len(paired_df)} 笔完整交易")
    
    # 信号解析
    parsed_df = parse_buy_signals(paired_df)
    
    # 特征工程
    final_df = prepare_features(parsed_df)
    
    # 分析
    feature_importance = analyze_features(final_df)
    
    # 输出优化建议
    print("\n=== 优化建议 ===")
    print("应加强的指标：")
    for feat in feature_importance[feature_importance['Importance'] > 0]['Feature']:
        print(f"  - {feat}")
        
    print("\n需谨慎处理的指标：")
    for feat in feature_importance[feature_importance['Importance'] < 0]['Feature']:
        print(f"  - {feat}")

# 使用说明：
# 1. 依赖库：pandas, openpyxl, scikit-learn, matplotlib, seaborn
# 2. 确保Excel文件路径正确
# 3. 输出包含：
#    - 交易匹配数量
#    - 特征相关性热力图
#    - 逻辑回归特征重要性排序
#    - 优化建议