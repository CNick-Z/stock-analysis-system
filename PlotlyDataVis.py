# PlotlyDataVis.py
import matplotlib.pyplot as plt

def plot_results(data, signals, positions):
    plt.figure(figsize=(12, 8))
    
    # 绘制价格和均线
    plt.subplot(211)
    plt.plot(data['close'], label='Price')
    plt.plot(data['MA5'], label='5日均线')
    plt.plot(data['MA20'], label='20日均线')
    plt.scatter(signals[signals['golden_cross']].index, 
                data['close'][signals['golden_cross']], 
                marker='^', color='g', label='买入')
    plt.scatter(signals[signals['death_cross']].index,
                data['close'][signals['death_cross']],
                marker='v', color='r', label='卖出')
    
    # 绘制资产曲线
    plt.subplot(212)
    plt.plot(positions['total'], label='组合净值')    
    plt.show()



def plot_signals(data, signals):
    plt.figure(figsize=(12, 8))
    plt.plot(data['close'], label='Price')
    plt.plot(data['MA5'], label='MA5')
    plt.plot(data['MA20'], label='MA20')
    plt.scatter(
        signals[signals['enter_long']].index,
        data.loc[signals['enter_long'], 'close'],
        marker='^', color='g', label='Buy'
    )
    plt.scatter(
        signals[signals['exit_long']].index,
        data.loc[signals['exit_long'], 'close'],
        marker='v', color='r', label='Sell'
    )
    plt.legend()
    plt.show()