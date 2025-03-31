# stock_report.py
import pandas as pd
import akshare as ak
from typing import Dict, List
from datetime import datetime,date,timedelta
from utils.strategy import EnhancedTDXStrategy,StockScorer



class StockReport:
    def __init__(self, scorer: StockScorer):
        self.report = []
        self.scorer = scorer  # 注入评分器实例
        self.weight_info = self._get_weight_explanations()

    def _get_weight_explanations(self) -> Dict:
        """获取权重配置说明"""
        return {
            'technical': {
                'weights': self.scorer.config['technical_weights'],
                'desc': "技术指标权重配置："
            },
            'capital_flow': {
                'weights': self.scorer.config['capital_flow_weights'],
                'desc': "资金流向子权重："
            },
            'market_heat': {
                'window': self.scorer.config['market_heat']['window'],
                'desc': "市场热度计算窗口："
            }
        }

    def _format_weight_table(self, weight_type: str) -> str:
        """生成权重说明表格"""
        info = self.weight_info[weight_type]
        table = [f"{info['desc']}"]
        if 'weights' in info:
            for k, v in info['weights'].items():
                table.append(f"  - {k.replace('_', ' ').title():<20}: {v:.2f}")
        else:
            table.append(f"  - 计算窗口: {info['window']}天")
        return "\n".join(table)

    def generate_report(self, selected_stocks: pd.DataFrame) -> str:
        # 在报告开头添加权重配置说明
        #self.report.append("# 策略权重配置说明")
        #self.report.append(self._format_weight_table('technical'))
        #self.report.append("\n" + self._format_weight_table('capital_flow'))
        #self.report.append("\n" + self._format_weight_table('market_heat'))

        # 修改后的个股分析部分
        for _, row in selected_stocks.iterrows():
            self.report=[]
            analysis = [
                "------------------------------------------------------------------------------------------------------------------"
                f"\n## {row['symbol']} {row['name']} 综合评分：{row['total_score']:.2f}",
                "### 评分构成分析:",
                f"- 技术面评分：{row['technical']:.2f} (权重 {self.scorer.config['weights']['technical']})",
                f"- 资金流向评分：{row['capital_flow']:.2f} (权重 {self.scorer.config['weights']['capital_flow']})",
                f"- 市场热度评分：{row['market_heat']:.2f} (权重 {self.scorer.config['weights']['market_heat']})",
                
                "### 技术面信号详情:",
                f"- 5日均线 > 20日均线：{'是' if row['ma_5'] > row['ma_20'] else '否'} (权重 +{self.scorer.config['technical_weights']['ma_condition']})",
                f"- 10日均线角度：{row['angle_ma_10']:.1f}° (阈值 >30°, 权重 +{self.scorer.config['technical_weights']['angle_condition']})",
                f"- MACD金叉：{'是' if row['macd'] > row['macd_signal'] else '否'} (权重 +{self.scorer.config['technical_weights']['macd_condition']})",
                f"- 成交量动能：{min(row['volume']/row['volume_ma5'], self.scorer.config['scoring_rules']['max_volume_ratio']):.2f}x (权重 x{self.scorer.config['technical_weights']['volume_score']})",
                f"- RSI_14：{row['rsi_14']:.1f} (超卖阈值 <70, 权重 +{self.scorer.config['technical_weights']['rsi_oversold']})",
                f"- KDJ_K：{row['kdj_k']:.1f} (超卖阈值 <80, 权重 +{self.scorer.config['technical_weights']['kdj_oversold']})",
                f"- CCI_20：{row['cci_20']:.1f} (超卖阈值 <100, 权重 +{self.scorer.config['technical_weights']['cci_oversold']})",
                f"- 布林带上轨突破：{'否' if row['close'] < row['bb_upper'] else '是'} (权重 +{self.scorer.config['technical_weights']['bollinger_condition']})",
                
                "### 资金流向信号详情:",
                f"- 主力资金方向：{'净流入' if row['money_flow_positive'] else '净流出'} (增益系数 x1.4)",
                f"- 资金流持续增长：{'是' if row['money_flow_increasing'] else '否'} (权重 +{self.scorer.config['capital_flow_weights']['flow_increasing']})",
                f"- 趋势强度：{min(row['主生量']/row['量基线'], self.scorer.config['scoring_rules']['max_trend_strength']):.2f}x (阈值 {self.scorer.config['scoring_rules']['max_trend_strength']}x)",
                f"- 周资金流方向：{'正向' if row['money_flow_weekly'] else '负向'} (权重 +{self.scorer.config['capital_flow_weights']['weekly_flow']})",
                f"- 周资金流增长：{'是' if row['money_flow_weekly_increasing'] else '否'} (增益系数 x1.3)",
                f"- 量增幅：{row['量增幅']:.1f}% → 触发阈值：{'增益' if row['量增幅'] > self.scorer.config['thresholds']['volume_gain_threshold'] else '损失' if row['量增幅'] < self.scorer.config['thresholds']['volume_loss_threshold'] else '无'} (乘数 x{self.scorer.config['capital_flow_weights']['volume_gain_multiplier'] if row['量增幅'] > self.scorer.config['thresholds']['volume_gain_threshold'] else self.scorer.config['capital_flow_weights']['volume_loss_multiplier'] if row['量增幅'] < self.scorer.config['thresholds']['volume_loss_threshold'] else '1.0'})",
                f"- 资金流最终得分：{row['capital_flow']:.2f} (上限 {self.scorer.config['scoring_rules']['max_flow_score']})"
                "------------------------------------------------------------------------------------------------------------------"
            ]
            self.report.append("\n".join(analysis))
        
        return "\n".join(self.report)

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
        scorer = StockScorer()  # 创建评分器实例
        reporter = StockReport(scorer)  # 传入评分器
        report = reporter.generate_report(daily_buy_signals)        
        print("最终选股结果：")
        print(daily_buy_signals.reset_index()[['date', 'symbol',  'name',  'industry','total_score']])
        print("\n详细分析：")
        print(report)