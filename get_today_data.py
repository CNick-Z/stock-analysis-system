#get_today_data.py
import data_fetcher
import DataProcessor
from datetime import date,timedelta,datetime
from db_operations import *

def get_today_data(start_date, end_date,db_url):
    fetcher = data_fetcher.DataFetcher(db_url)
    fetcher.fetch_and_save_all_data(start_date, end_date)
    processor = DataProcessor.TechnicalIndicatorCalculator(db_url)
    processor.process_by_stock()

def get_data_last_day(db_url):
    db_manager = DatabaseManager(db_url=db_url)
    last_day = db_manager.load_data(DailyDataBase,distinct_column='date', order_by=[{'column': 'date', 'direction': 'desc'}], limit=1)['date'].iloc[0]
    return last_day

if __name__ == "__main__":
    db_url = "sqlite:///c:/db/stock_data.db"
    data_last_day = get_data_last_day(db_url)
    start_date = datetime.strptime(data_last_day,'%Y-%m-%d')+timedelta(days=1)
    end_date = datetime.today()+timedelta(days=-1)
    end_date = datetime.strftime(end_date,'%Y%m%d')
    #end_date = date.today().strftime("%Y%m%d")
    get_today_data(start_date.strftime("%Y%m%d"), end_date,db_url)