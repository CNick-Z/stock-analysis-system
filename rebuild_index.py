# -*- coding: utf-8 -*-
# rebuild_all_indexes.py
from utils.db_operations import DatabaseManager, create_daily_data_table, create_technical_indicators_table, DailyDataBase, TechnicalIndicatorsBase
from datetime import datetime
from sqlalchemy import text


def rebuild_all_indexes(db_manager,startyear):
    """重建所有分年表的索引"""
    years = range(startyear, datetime.now().year)  # 假设有2020至今的表
    
    with db_manager.get_his_session() as hist_session:
        for year in years:
            # 重建技术指标表索引
            hist_session.execute(text(f"REINDEX technical_indicators_{year}"))
            hist_session.execute(text(f"ANALYZE technical_indicators_{year}"))
            
            # 重建日线数据表索引
            hist_session.execute(text(f"REINDEX daily_data_{year}"))
            hist_session.execute(text(f"ANALYZE daily_data_{year}"))
        
        hist_session.commit()

if __name__ == "__main__":
    # 创建所有分年表的索引
    # 初始化数据库管理器
    db_manager = DatabaseManager(db_url='sqlite:///c:/db/stock_data.db')
    current_year = str(datetime.now().year)
    rebuild_all_indexes(db_manager,2002)
    print("索引重建完成。")