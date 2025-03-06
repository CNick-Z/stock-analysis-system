# stock_selector.py
import pandas as pd
import akshare as ak
from db_operations import *
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime,date,timedelta

class StockScorer:
    def __init__(self, config: Dict = None):
        self.db_manager = DatabaseManager(db_url = "sqlite:///c:/db/stock_data.db")
        self.db_manager.ensure_tables_exist()
        self.config = config or {
            'weights': {
                'technical': 0.3,  # 调整权重
                'capital_flow': 0.3,  # 新增资金流向权重
                'fundamental': 0.2,
                'market_heat': 0.2  # 调整市场热度权重
            },
             # 新增资金流子项权重配置
            'capital_flow_weights': {
                'positive_flow': 0.3,    # 资金流入
                'flow_increasing': 0.25,  # 流入加速
                'trend_strength': 0.2,    # 趋势强度
                'weekly_flow': 0.15,      # 周级别流入
                'weekly_increasing': 0.1  # 周流入加速
            },
            'fundamental_metrics': ['pe_ratio', 'roe', 'profit_growth'],
            'heat_window': 30  # 市场热度计算窗口
        }
        
    def _get_fundamental_score(self, symbol: str) -> float:
        """获取财务指标评分（带缓存机制）"""
        '''
        try:
            # 从数据库获取预存财务数据
            conn = sqlite3.connect('./db/stock_data.db')
            query = f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                latest = df.iloc[-1]
                return sum(
                    latest[metric] * self.config['fundamental_weights'][metric]
                    for metric in self.config['fundamental_metrics']
                )
                
            # 实时获取作为后备
            df = ak.stock_financial_report_sina(symbol=symbol, indicator="主要指标")
            return df['net_profit'].iloc[-1] / df['revenue'].iloc[-1]
        except Exception as e:
            print(f"财务数据获取失败 {symbol}: {str(e)}")
            return 0
        '''
        return 0

    def _calculate_technical_score(self, row: pd.Series) -> float:
        """技术指标评分（基于策略信号）"""
        tech_score = 0
        # 均线系统评分
        tech_score += (row['ma_5'] > row['ma_20']) * 0.2
        tech_score += (row['angle_ma_10'] > 30) * 0.2
        tech_score += (row['macd'] > row['macd_signal']) * 0.3
        
        # 成交量动能
        volume_score = min(row['volume'] / row['volume_ma5'], 3)  # 限制最大3倍
        tech_score += volume_score * 0.3

        # 超买超卖评分（新增）
        tech_score += (row['rsi_14'] < 70) * 0.1  # 未超买加分
        tech_score += (row['kdj_k'] < 80) * 0.1   # KDJ 未超买加分
        tech_score += (row['cci_20'] < 100) * 0.1  # CCI 未超买加分
        tech_score += (row['close'] < row['bb_upper']) * 0.1  # 布林带未超买加分
        
        return tech_score

    def _calculate_capital_flow(self, row: pd.Series) -> float:
        """基于策略生成的新资金流信号进行评分"""
        flow_score = 0
        
        # 资金流入基础分
        if row['money_flow_positive']:
            flow_score += self.config['capital_flow_weights']['positive_flow'] * 1.2  # 正值强化
            
        # 流入加速
        if row['money_flow_increasing']:
            flow_score += self.config['capital_flow_weights']['flow_increasing'] * 1.0
            
        # 趋势强度
        if row['money_flow_trend']:
            trend_strength = min(row['主生量'] / row['量基线'], 2.0)  # 限制最大2倍
            flow_score += self.config['capital_flow_weights']['trend_strength'] * trend_strength
            
        # 周级别资金流
        if row['money_flow_weekly']:
            flow_score += self.config['capital_flow_weights']['weekly_flow'] * 1.0
            
        # 周流入加速
        if row['money_flow_weekly_increasing']:
            flow_score += self.config['capital_flow_weights']['weekly_increasing'] * 1.5  # 加速给予更高权重
            
        # 量增幅强化
        if row['量增幅'] > 10:  # 显著增长
            flow_score *= 1.2
        elif row['量增幅'] < -5:  # 显著减少
            flow_score *= 0.8
            
        return min(flow_score, 1.0)  # 限制最大1分

    def _get_market_heat(self, symbol: str, date: str) -> float:
        """市场热度评分（基于板块）"""
        try:
            # 获取板块数据
            with self.db_manager.get_session() as session:
                sector_query = session.query(StockBasicInfo.industry).filter_by(symbol=symbol).first()
                if sector_query:
                    sector = sector_query[0]
                else:
                    return 0
                
                # 计算板块热度
                heat_query = session.query(DailyData.volume).filter(
                    DailyData.symbol == symbol,
                    DailyData.date >= (date - timedelta(days=self.config['heat_window'])).strftime("%Y-%m-%d"),
                    DailyData.date <= date.strftime("%Y-%m-%d")
                ).all()
                
                if heat_query:
                    sector_vol = sum([row[0] for row in heat_query]) / len(heat_query)
                    current_vol = session.query(DailyData.volume).filter(
                        DailyData.symbol == symbol,
                        DailyData.date == date.strftime("%Y-%m-%d")
                    ).first()
                    if current_vol:
                        return current_vol[0] / sector_vol
        except Exception as e:
            print(f"Error calculating market heat: {e}")
        return 0

    def score_daily_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """每日信号综合评分"""
        signals = signals.copy()  # 避免修改原始数据
        for _, row in tqdm(signals.iterrows(), total=len(signals), desc="每日评分"):
            if row['signal_type'] != 'buy':
                continue
                
            # 计算各维度评分
            technical_score = self._calculate_technical_score(row)
            capital_flow_score = self._calculate_capital_flow(row)
            fundamental_score = self._get_fundamental_score(row['symbol'])
            market_heat_score = self._get_market_heat(row['symbol'], row['date'])
            
            # 加权总分
            weights = self.config['weights']
            total_score = (
                technical_score * weights['technical'] +
                capital_flow_score * weights['capital_flow'] +
                fundamental_score * weights['fundamental'] +
                market_heat_score * weights['market_heat']
            )
            
            # 将评分结果添加到行中
            signals.loc[row.name, 'technical'] = technical_score
            signals.loc[row.name, 'capital_flow'] = capital_flow_score
            signals.loc[row.name, 'fundamental'] = fundamental_score
            signals.loc[row.name, 'market_heat'] = market_heat_score
            signals.loc[row.name, 'total_score'] = total_score
        
        # 保留有效的评分记录（enter_long 为 True）
        return signals[signals['signal_type']=='buy'].reset_index(drop=True)

class StockSelector:
    def __init__(self):
        self.scorer = StockScorer()
        
    def select_top_stocks(self, signals: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """选股主流程"""
        # 评分
        scored_df = self.scorer.score_daily_signals(signals)
        if scored_df.empty:
            return None
        else:
            # 每日TopN选择
            final_picks = []
            for date in scored_df['date'].unique():
                daily = scored_df[scored_df['date'] == date]
                top = daily.nlargest(top_n, 'total_score')
                final_picks.append(top)
                
            return pd.concat(final_picks).reset_index(drop=True)
       

    def generate_report(self, selected_stocks: pd.DataFrame) -> str:
        """生成带依据的报告"""
        report = []
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
                f"  量增幅: {row['量增幅']:.2f}%"
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
            
            report.append("\n".join(analysis))
        
        return "\n\n".join(report)

# 使用示例
if __name__ == "__main__":
    # 生成策略信号
    date =date.today().strftime("%Y-%m-%d")
    from strategy import EnhancedTDXStrategy
    strategy = EnhancedTDXStrategy()
    #signals = strategy.get_buy_signals(start_date,end_date)
    signals = strategy.get_buy_signals('2025-03-04')
    if signals.empty: 
        print("没有符合策略条件的股票。")
        exit()
    else:
        # 执行选股评分
        selector = StockSelector()
        scored_stocks = selector.select_top_stocks(signals)
        report = selector.generate_report(scored_stocks)
        
        print("最终选股结果：")
        print(scored_stocks[['date', 'symbol',  'name',  'industry','total_score']])
        print("\n详细分析：")
        print(report)                                                   