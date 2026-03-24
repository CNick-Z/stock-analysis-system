# stock_basic_info.py
import akshare as ak
from utils.db_operations import *
import logging
from sqlalchemy import distinct
from datetime import datetime
# 配置日志
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_and_save_stock_basic_info(db_url, symbol):
    """
    获取并保存单个股票的基本信息到数据库。
    :param db_url: 数据库连接 URL
    :param symbol: 股票代码
    """
    db_manager = DatabaseManager(db_url=db_url)
    db_manager.ensure_tables_exist()

    try:
        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        
        # 转换格式（akshare新版本返回 item/value 格式）
        info_dict = dict(zip(stock_info['item'], stock_info['value']))
        
        # 提取关键信息
        basic_info = {
            "symbol": symbol,
            "name": info_dict.get("股票简称", ""),
            "total_shares": int(info_dict.get("总股本", 0)),
            "circulating_shares": int(info_dict.get("流通股", 0)),
            "market_value": float(info_dict.get("总市值", 0)),
            "circulating_market_value": float(info_dict.get("流通市值", 0)),
            "industry": info_dict.get("行业", ""),
            "listing_date": datetime.strptime(str(info_dict.get("上市时间", "")), "%Y%m%d").date()
        }

        # 检查是否已存在该股票的基本信息
        with db_manager.get_session() as session:
            existing_info = session.query(StockBasicInfo).filter_by(symbol=symbol).first()
            if existing_info:
                logging.info(f"Updating existing info for {symbol}...")
                existing_info.name = basic_info["name"]
                existing_info.total_shares = basic_info["total_shares"]
                existing_info.circulating_shares = basic_info["circulating_shares"]
                existing_info.market_value = basic_info["market_value"]
                existing_info.circulating_market_value = basic_info["circulating_market_value"]
                existing_info.industry = basic_info["industry"]
                existing_info.listing_date = basic_info["listing_date"]
            else:
                logging.info(f"Saving new info for {symbol}...")
                db_manager.bulk_insert(StockBasicInfo, pd.DataFrame([basic_info]))

    except Exception as e:
        logging.error(f"Error fetching or saving info for {symbol}: {e}")



def fetch_symbols_from_daily_data(db_url):
    """
    从 daily_data 表中获取所有唯一的股票代码。
    :param db_url: 数据库连接 URL
    :return: 股票代码列表
    """
    db_manager = DatabaseManager(db_url=db_url)
    symbols = db_manager.load_data(DailyDataBase, distinct_column="symbol" )['symbol'].tolist()
    return symbols

def fetch_and_save_all_stock_basic_info(db_url):
    """
    从 daily_data 表中获取所有股票代码，并批量获取和保存它们的基本信息。
    :param db_url: 数据库连接 URL
    """
    symbols = fetch_symbols_from_daily_data(db_url)
    for symbol in symbols:
        fetch_and_save_stock_basic_info(db_url, symbol)

if __name__ == "__main__":
    db_url = "sqlite:///c:/db/stock_data.db"
    fetch_and_save_all_stock_basic_info(db_url)
