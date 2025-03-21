# stock_report.py
import pandas as pd
import akshare as ak
from typing import Dict, List
from datetime import datetime,date,timedelta
from strategy import EnhancedTDXStrategy



class StockReport:
    def __init__(self):
        self.report = []

       

    def generate_report(self, selected_stocks: pd.DataFrame) -> str:
        """生成带依据的报告"""
        self.report = []
        for _, row in selected_stocks.iterrows():
            # 市场热度分析逻辑（基于 market_heat 列）
            heat_analysis = [
                f"- 市场热度分析（热度值：{row['market_heat']:.2f}）"
            ]
            if row['market_heat'] > 1.2:
                heat_analysis.append("    存在板块异动信号；市场活跃")
            else:
                heat_analysis.append("    近期板块无明显异动；市场一般")
            # 新增资金流分析段落
            flow_analysis = [
                f"- 资金流向分析({row['capital_flow']:.2f}):",
                f"  主力资金方向: {'流入' if row['money_flow_positive'] else '流出'}",
                f"  短期趋势: {'加速' if row['money_flow_increasing'] else '减速'}",
                f"  趋势强度: {row['主生量']:.2f} (基线: {row['量基线']:.2f})",
                f"  周级别资金: {'净流入' if row['money_flow_weekly'] else '净流出'}",
                f"  周趋势: {'加速' if row['money_flow_weekly_increasing'] else '减速'}",
                f"  量增幅: {row['量增幅']:.2f}%",
                f"  当日涨幅：{row['growth']}"
                    ]
            # 技术分析部分增加MACD信息
            macd_analysis = [
                f"- MACD分析：",
                f"  当前MACD值: {row['macd']:.2f}",
                f"  信号线值: {row['macd_signal']:.2f}",
                f"  MACD柱状图: {'正值' if row['macd'] > 0 else '负值'}",
                f"  MACD与信号线关系: {'金叉' if row['macd'] > row['macd_signal'] else '死叉' if row['macd'] < row['macd_signal'] else '平行'}",
                f"  MACD金叉信号: {'存在' if row['macd_jc'] else '不存在'}"
            ]
            analysis = [
                f"{row['symbol']} 评分 {row['total_score']:.2f}：",
                f"- 技术面({row['technical']:.2f}): \
                    {'多头排列' if row['ma_5'] > row['ma_20'] else '空头'}；\
                    RSI：{row['rsi_14']:.2f}；CCI：{row['cci_20']}",
                "\n".join(flow_analysis),
                "\n".join(heat_analysis),
                "\n".join(macd_analysis),
                f"- 财务({row['fundamental']:.2f}): 盈利成长股" if row['fundamental'] > 0.6 else "- 财务: 稳健",
                f"- 超买超卖信号：\n" \
                f"    - RSI (14): {'超卖' if row['rsi_14'] < 30 else '超买' if row['rsi_14'] > 70 else '正常'}\n" \
                f"    - KDJ K: {'超卖' if row['kdj_k'] < 20 else '超买' if row['kdj_k'] > 80 else '正常'}\n" \
                f"    - KDJ D: {'超卖' if row['kdj_d'] < 20 else '超买' if row['kdj_d'] > 80 else '正常'}\n" \
                f"    - CCI (20): {'超卖' if row['cci_20'] < -100 else '超买' if row['cci_20'] > 100 else '正常'}\n" \
                f"    - Williams R: {'超卖' if row['williams_r'] > -20 else '超买' if row['williams_r'] < -80 else '正常'}\n" \
                f"    - 布林带中轨: {'超卖' if row['bb_middle'] < row['close'] * 0.95 else '超买' if row['bb_middle'] > row['close'] * 1.05 else '正常'}"
            ]
            # 补充布林带位置信息
            if row['close'] >= row['bb_upper']:
                analysis.append("当前价格 > 布林带上轨（超买）")
            elif row['close'] <= row['bb_lower']:
                analysis.append("当前价格 < 布林带下轨（超卖）")
            else:
                analysis.append("当前价格处于布林带中间区域")
            
            self.report.append("\n".join(analysis))
        
        return "\n\n".join(self.report)

# 使用示例
if __name__ == "__main__":
    # 生成策略信号
    date =date.today().strftime("%Y-%m-%d")
    db_url = "c:/db/stock_data.db"
    strategy = EnhancedTDXStrategy(db_url)
    #signals = strategy.get_buy_signals(start_date,end_date)
    signals = strategy.get_signals('2024-08-21','2024-10-12')
    daily_buy_signals = signals[0]
    if daily_buy_signals.empty or daily_buy_signals is None: 
        print("没有符合策略条件的股票。")
        exit()
    else:
        # 执行选股评分
        reporter = StockReport()
        report = reporter.generate_report(daily_buy_signals)        
        print("最终选股结果：")
        print(daily_buy_signals.reset_index()[['date', 'symbol',  'name',  'industry','total_score']])
        print("\n详细分析：")
        print(report)