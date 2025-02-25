# stock_selector.py
import pandas as pd
import akshare as ak
import sqlite3
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime

class StockScorer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'weights': {
                'technical': 0.4,
                'capital_flow': 0.3,
                'fundamental': 0.2,
                'market_heat': 0.1
            },
            'fundamental_metrics': ['pe_ratio', 'roe', 'profit_growth'],
            'heat_window': 30  # 市场热度计算窗口
        }
        
    def _get_fundamental_score(self, symbol: str) -> float:
        """获取财务指标评分（带缓存机制）"""
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

    def _calculate_technical_score(self, row: pd.Series) -> float:
        """技术指标评分（基于策略信号）"""
        tech_score = 0
        # 均线系统评分
        tech_score += (row['ma_5'] > row['ma_20']) * 0.2
        tech_score += (row['angle_ma_10'] > 30) * 0.2
        tech_score += (row['macd'] > row['macd_signal']) * 0.3
        
        # 成交量动能
        volume_score = min(row['volume'] / row['vol_ma5'], 3)  # 限制最大3倍
        tech_score += volume_score * 0.3
        
        return tech_score

    def _calculate_capital_flow(self, symbol: str, date: str) -> float:
        """资金动向评分（示例）"""
        try:
            # 获取历史数据计算资金流
            conn = sqlite3.connect('./db/stock_data.db')
            query = f"""
                SELECT close, volume 
                FROM daily_data 
                WHERE symbol='{symbol}' 
                AND date <= '{date}' 
                ORDER BY date DESC 
                LIMIT 5
            """
            df = pd.read_sql(query, conn)
            conn.close()
            
            # 简单资金流模型
            price_change = (df['close'].iloc[0] - df['close'].iloc[-1]) / df['close'].iloc[-1]
            volume_change = (df['volume'].iloc[0] / df['volume'].iloc[1:].mean())
            return price_change * volume_change
        except:
            return 0

    def _get_market_heat(self, symbol: str, date: str) -> float:
        """市场热度评分（基于板块）"""
        try:
            # 获取板块数据
            conn = sqlite3.connect('./db/stock_data.db')
            # 假设有板块信息表
            sector_query = f"SELECT sector FROM stock_info WHERE symbol='{symbol}'"
            sector = pd.read_sql(sector_query, conn).iloc[0]['sector']
            
            # 计算板块热度
            heat_query = f"""
                SELECT AVG(volume) as avg_vol 
                FROM daily_data 
                WHERE sector='{sector}' 
                AND date BETWEEN date('{date}', '-{self.config['heat_window']} days') AND '{date}'
            """
            sector_vol = pd.read_sql(heat_query, conn).iloc[0]['avg_vol']
            current_vol = pd.read_sql(f"SELECT volume FROM daily_data WHERE symbol='{symbol}' AND date='{date}'", conn).iloc[0]['volume']
            conn.close()
            
            return current_vol / sector_vol
        except:
            return 0

    def score_daily_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """每日信号综合评分"""
        signals = signals.copy()  # 避免修改原始数据
        for _, row in tqdm(signals.iterrows(), total=len(signals), desc="每日评分"):
            if not row['enter_long']:
                continue
                
            # 计算各维度评分
            technical_score = self._calculate_technical_score(row)
            capital_flow_score = self._calculate_capital_flow(row['symbol'], row['date'])
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
        return signals[signals['enter_long']].reset_index(drop=True)

class StockSelector:
    def __init__(self):
        self.scorer = StockScorer()
        
    def select_top_stocks(self, signals: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """选股主流程"""
        # 评分
        scored_df = self.scorer.score_daily_signals(signals)
        
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
            analysis = [
                f"{row['symbol']} 评分 {row['total_score']:.2f}：",
                f"- 技术面({row['technical']:.2f}): {'多头排列' if row['ma_5'] > row['ma_20'] else ''} MACD金叉",
                f"- 资金流({row['capital_flow']:.2f}): 近期量价齐升" if row['capital_flow'] > 0 else "- 资金流: 平淡",
                f"- 财务({row['fundamental']:.2f}): 盈利成长股" if row['fundamental'] > 0.6 else "- 财务: 稳健"
            ]
            report.append("\n".join(analysis))
        
        return "\n\n".join(report)

# 使用示例
if __name__ == "__main__":
    # 生成策略信号
    from strategy import EnhancedTDXStrategy
    strategy = EnhancedTDXStrategy()
    signals = strategy.generate_signals("2024-09-25", "2024-09-26")
    
    # 执行选股评分
    selector = StockSelector()
    scored_stocks = selector.select_top_stocks(signals)
    report = selector.generate_report(scored_stocks)
    
    print("最终选股结果：")
    print(scored_stocks[['date', 'symbol', 'total_score']])
    print("\n详细分析：")
    print(report)