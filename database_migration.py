# 数据迁移脚本
# database_migration.py
from utils.db_operations import DatabaseManager, create_daily_data_table, create_technical_indicators_table, DailyDataBase, TechnicalIndicatorsBase
from datetime import datetime

# 初始化数据库管理器
db_manager = DatabaseManager(db_url='sqlite:///c:/db/stock_data_his3.db', historical_db_url='sqlite:///c:/db/stock_data_his.db')
current_year = str(datetime.now().year)

# 获取所有历史数据
with db_manager.get_session() as session:
    # 查询所有年份
    years = session.query(DailyDataBase.date).distinct().all()
    years = list(set([date[:4] for date, in years]))

    for year in years:
        if year == current_year:
            continue
        # 获取历史数据
        daily_data = session.query(DailyDataBase).filter(DailyDataBase.date.startswith(year)).all()
        technical_indicators = session.query(TechnicalIndicatorsBase).filter(TechnicalIndicatorsBase.date.startswith(year)).all()

        # 插入到历史数据库
        with db_manager.get_his_session() as hist_session:
            # 创建历史表
            historical_daily_data_table = create_daily_data_table(year)
            historical_technical_indicators_table = create_technical_indicators_table(year)
            historical_daily_data_table.__table__.create(db_manager.historical_engine, checkfirst=True)
            historical_technical_indicators_table.__table__.create(db_manager.historical_engine, checkfirst=True)
            hist_session.bulk_insert_mappings(historical_daily_data_table, [item.__dict__ for item in daily_data])
            hist_session.bulk_insert_mappings(historical_technical_indicators_table, [item.__dict__ for item in technical_indicators])
            hist_session.commit()