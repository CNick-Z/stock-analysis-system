import akshare as ak
import pandas as pd

def get_financial_scores(symbol):
    # 获取财务数据
    abstract = ak.stock_financial_abstract(stock=symbol)
    indicators = ak.stock_financial_analysis_indicator(stock=symbol)
    
    # 提取关键指标
    scores = {
        "ROE(TTM)": indicators["净资产收益率(%)"].mean(),
        "营收增长率": (abstract["主营业务收入"].iloc[-1] / abstract["主营业务收入"].iloc[-4] - 1),
        "毛利率": indicators["销售毛利率(%)"].iloc[-1],
        "负债率": indicators["资产负债率(%)"].iloc[-1]
    }
    return pd.Series(scores)

# 示例：获取沪深300成分股的财务评分
hs300 = ak.index_stock_cons_sina(symbol="hs300")
financial_scores = hs300["000001"].apply(get_financial_scores)