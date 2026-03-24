# strategies/base.py
# 策略抽象基类，所有策略插件必须继承此类

from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    策略抽象基类

    所有策略插件必须实现以下方法：
    - get_signals(): 获取买卖信号
    - generate_features(): 生成特征数据
    - select_top_stocks(): 选股
    """

    def __init__(self, db_path: str = None, config: dict = None):
        """
        参数:
            db_path: 数据库路径
            config: 策略配置参数
        """
        self.db_path = db_path
        self.config = config or {}
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    def get_signals(self, start_date: str, end_date: str):
        """
        获取买卖信号

        参数:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        返回:
            [buy_signals, sell_signals] 买入和卖出信号列表
        """
        pass

    @abstractmethod
    def get_buy_signals(self, date: str, candidates: pd.DataFrame, **kwargs) -> list:
        """
        从候选股票中筛选买入信号

        参数:
            date: 当前日期
            candidates: 当日候选股票 DataFrame
            **kwargs: 其他参数（如持仓信息、资金等）

        返回:
            买入信号列表，每项包含 [symbol, price, date, quantity, signal_info, extra_data]
        """
        pass

    @abstractmethod
    def generate_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成特征数据

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            包含所有特征的 DataFrame
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} strategy_name={self.strategy_name}>"
