#get_today_data.py
import data_fetcher
import DataProcessor
from datetime import date

def get_today_data(start_date, end_date,db_url):
    fetcher = data_fetcher.DataFetcher(db_url)
    fetcher.fetch_and_save_all_data(start_date, end_date)
    processor = DataProcessor.TechnicalIndicatorCalculator(db_url)
    processor.process_unprocessed_data()

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db" 
    start_date =date.today().strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")
    get_today_data(start_date, end_date,db_url)