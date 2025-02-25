#get_today_data.py
import data_fetcher
import DataProcessor
from datetime import date

def get_today_data(start_date, end_date):
    fetcher = data_fetcher.DataFetcher()
    fetcher.fetch_and_save_all_data(start_date, end_date)
    processor = DataProcessor.TechnicalIndicatorCalculator()
    processor.process_unprocessed_data()

if __name__ == "__main__":
    start_date =date.today().strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")
    get_today_data(start_date, end_date)