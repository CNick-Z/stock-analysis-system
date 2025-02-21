# main.py
from data_fetcher import DataFetcher
from strategy import MAStrategy
from backtester import Backtester

def main():
    # 1. 获取数据
    fetcher = DataFetcher()
    
    # 获取沪深300成分股
    hs300 = fetcher.get_index_constituents()
    
    # 遍历成分股进行回测
    for symbol in hs300['代码'][:10]:  # 示例只跑前10只
        print(f"Processing {symbol}")
        
        # 获取数据
        df = fetcher.fetch_daily_data(symbol, "20200101", "20231231")
        
        # 2. 执行策略
        strategy = MAStrategy(df)
        signals = strategy.generate_signals()
        
        # 3. 回测
        backtester = Backtester(df, signals)
        report = backtester.run_backtest()
        
        # 输出结果
        print(f"{symbol} 回测结果:")
        print(f"累计收益: {report['累计收益']:.2%}")
        print(f"最大回撤: {report['最大回撤']:.2%}\n")

if __name__ == "__main__":
    main()