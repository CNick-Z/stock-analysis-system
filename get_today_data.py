#get_today_data.py
import os
from utils.data_fetcher import DataFetcher
from utils.DataProcessor import TechnicalIndicatorCalculator
from datetime import date,timedelta,datetime
from utils.db_operations import *
from Signale_Traderecord import SignalTraderecord
from utils.get_notion_database_info import NotionDatabaseManager
import configparser
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'get_today_data.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def get_today_data(start_date, end_date,db_url):
    fetcher = DataFetcher(db_url)
    fetcher.fetch_and_save_all_data(start_date, end_date)
    processor = TechnicalIndicatorCalculator(db_url)
    processor.process_all_data(start_date)

def get_data_last_day(db_url):
    db_manager = DatabaseManager(db_url=db_url)
    last_day = db_manager.load_data(DailyDataBase,distinct_column='date', order_by=[{'column': 'date', 'direction': 'desc'}], limit=1)['date'].iloc[0]
    return last_day

if __name__ == "__main__":
    setup_logging()
    logging.info('开始获取数据')
    config = configparser.ConfigParser()
    config.read('./conf/config.cfg')
    db_path=config['local_db']['path']
    db_url = f"sqlite:///{db_path}"
    data_last_day = get_data_last_day(db_url)
    start_date = datetime.strptime(data_last_day,'%Y-%m-%d')+timedelta(days=1)
    current_time = datetime.now()
    # 获取当前时间的小时数
    hour = current_time.hour
    if hour>18:
        end_date = datetime.today()
    else:
        end_date = datetime.today()+timedelta(days=-1)
    get_today_data(datetime.strftime(start_date,"%Y%m%d"), datetime.strftime(end_date,"%Y%m%d"),db_url)
    recorder=SignalTraderecord(db_path)
    notion=NotionDatabaseManager()
    datelist=recorder.get_trading_data(datetime.strftime(start_date,"%Y-%m-%d"), datetime.strftime(end_date,"%Y-%m-%d"))
    if len(datelist)==0:
        logging.info('没有交易数据')        
    else:
        for date in datelist:
            buylist_day,selllist_day = notion.query_notion_database(datetime.strftime(date,'%Y-%m-%d'))
            advice,notion_update_dic = recorder.run(buylist_day,selllist_day,datetime.strftime(date,'%Y-%m-%d'))
            notion.update_task_database(datetime.strftime(date,'%Y-%m-%d'),advice)
            notion.update_stock_database(notion_update_dic)
    logging.info('The app is end.')