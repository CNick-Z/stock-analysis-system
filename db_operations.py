# db_operations.py
from sqlalchemy import create_engine, Column, String, Float, Boolean, inspect, Index
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import logging
from contextlib import contextmanager
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

Base = declarative_base()

class DailyData(Base):
    __tablename__ = "daily_data"
    date = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    processed = Column(Boolean, default=False)
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_processed_symbol', 'processed', 'symbol'),
    )

class TechnicalIndicators(Base):
    __tablename__ = "technical_indicators"
    date = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_55 = Column(Float)
    sma_240 = Column(Float)
    vol_ma5 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    rsi_14 = Column(Float)
    kdj_k = Column(Float)
    kdj_d = Column(Float)
    kdj_j = Column(Float)
    cci_20 = Column(Float)
    williams_r = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    __table_args__ = (
        Index('idx_tec_symbol_date', 'symbol', 'date'),
    )

class DatabaseManager:
    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(self.db_url,connect_args={'check_same_thread': False})
        self.Session = scoped_session(sessionmaker(bind=self.engine))
    
    def ensure_tables_exist(self):
        """确保表存在"""
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        if "daily_data" not in existing_tables:
            DailyData.__table__.create(self.engine)
        if "technical_indicators" not in existing_tables:
            TechnicalIndicators.__table__.create(self.engine)
    
    @contextmanager
    def get_session(self):
        """获取数据库会话"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database error: {e}")
        finally:
            session.close()
    
    def load_data(self, table, filter_conditions, distinct_column=None, limit=None):
        """增强版数据加载方法，支持范围查询"""
        with self.get_session() as session:
            # 处理 distinct 查询
            if distinct_column:
                query = session.query(getattr(table, distinct_column).distinct())
            else:
                query = session.query(table)
            
            # 处理过滤条件
            for key, values in filter_conditions.items():
                if isinstance(values, list) and len(values) == 2:
                    # 范围查询处理
                    query = query.filter(
                        getattr(table, key).between(values[0], values[1])
                    )
                else:
                    # 精确匹配查询
                    query = query.filter(getattr(table, key) == values)
            
            # 处理结果限制
            if limit:
                query = query.limit(limit)
            
            # 获取数据
            df = pd.read_sql(query.statement, session.bind)
        
        return df
    
    def bulk_insert(self, table, data):
        """批量插入数据"""
        retries = 1
        n= 1
        while retries:
            try:
                with self.get_session() as session:
                    session.bulk_insert_mappings(table, data.to_dict(orient='records'))
                    retries = 0
                return  # 成功插入数据
            except Exception as e:
                n+=1
                logging.error(f"Insert failed, retrying: {e}")
                time.sleep(2 ** n)  # 指数退避

    
    def bulk_update(self, table, data, update_fields, filter_fields):
        """批量更新数据"""
        with self.get_session() as session:
            # 构建批量更新字典
            update_values = []
            for item in data:
                filter_dict = {}
                update_dict = {}
                for field in filter_fields:
                    if field not in item:
                        raise KeyError(f"Missing field '{field}' in data")
                    filter_dict[field] = item[field]
                for field in update_fields:
                    update_dict[field] = item[field]
                update_values.append({
                    **filter_dict,
                    **update_dict
                })
            
            # 批量更新
            session.bulk_update_mappings(table, update_values)
    
    def query_count(self, table, filter_conditions=None):
        """查询数据数量"""
        with self.get_session() as session:
            query = session.query(table)
            if filter_conditions:
                for key, value in filter_conditions.items():
                    query = query.filter(getattr(table, key) == value)
            count = query.count()
        return count