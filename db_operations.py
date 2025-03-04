# db_operations.py
from sqlalchemy import create_engine, Column, String, Float, Boolean, inspect, Index,event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import text
from sqlalchemy.sql import or_, and_
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
    amount = Column(Float)
    amplitude = Column(Float)
    change_pct = Column(Float)
    change_amount = Column(Float)
    turnover_rate = Column(Float)
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

class StockBasicInfo(Base):
    __tablename__ = "stock_basic_info"
    symbol = Column(String(10), primary_key=True,nullable=False, unique=True)  # 股票代码
    name = Column(String(50), nullable=False)  # 股票名称
    total_shares = Column(Float, nullable=False)  # 总股本
    circulating_shares = Column(Float, nullable=False)  # 流通股本
    market_value = Column(Float)  # 总市值
    circulating_market_value = Column(Float)  # 流通市值
    industry = Column(String(50))  # 行业
    listing_date = Column(String)  # 上市日期
    __table_args__ = (
        Index('idx_info_symbol', 'symbol'),
    )

class DatabaseManager:
    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(self.db_url,connect_args={'check_same_thread': False})
        self.Session = scoped_session(sessionmaker(bind=self.engine))
    
     # 监听连接事件，在连接建立时设置 PRAGMA journal_mode = WAL
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA cache_size=-200000;")  # 设置缓存大小为 200MB
            cursor.execute("PRAGMA mmap_size=4294967296;")  # 设置内存映射大小为 10GB
            cursor.execute("PRAGMA page_size=4096;")  # 设置页面大小为 4KB
            cursor.execute("PRAGMA synchronous=NORMAL;")  # 设置同步模式为 NORMAL
            cursor.close()
    def ensure_tables_exist(self):
        """确保表存在"""
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        if "daily_data" not in existing_tables:
            DailyData.__table__.create(self.engine)
        if "technical_indicators" not in existing_tables:
            TechnicalIndicators.__table__.create(self.engine)
        if "stock_basic_info" not in existing_tables:
            StockBasicInfo.__table__.create(self.engine)
    
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
    
    def load_data(self, table, filter_conditions=None, distinct_column=None, limit=None, columns=None):
        """增强版数据加载方法，支持动态过滤条件和指定列查询"""
        from sqlalchemy import text
        from sqlalchemy.sql import or_, and_
        from sqlalchemy import inspect

        with self.get_session() as session:
            # 处理指定列查询
            if columns:
                if not isinstance(columns, list):
                    columns = [columns]
                # 检查指定的列是否存在
                mapper = inspect(table)
                columns_attr = []
                for col in columns:
                    if isinstance(col, str):
                        col_attr = getattr(table, col, None)
                        if col_attr is None:
                            raise AttributeError(f"Column '{col}' not found in table '{table.__name__}'")
                        columns_attr.append(col_attr)
                    else:
                        columns_attr.append(col)
                # 构建查询对象
                if distinct_column:
                    # 如果存在 distinct_column，则需要单独处理
                    if distinct_column not in columns:
                        # 确保 distinct_column 总是被包含
                        columns_attr.append(getattr(table, distinct_column))
                    query = session.query(*columns_attr).distinct()
                else:
                    query = session.query(*columns_attr)
            else:
                if distinct_column:
                    query = session.query(getattr(table, distinct_column).distinct())
                else:
                    query = session.query(table)
            
            # 操作符映射
            operator_mapping = {
                '$eq': '__eq__',
                '$ne': '__ne__',
                '$gt': '__gt__',
                '$lt': '__lt__',
                '$gte': '__ge__',
                '$lte': '__le__',
                '$like': 'like',
                '$in': 'in_',
                '$between': 'between',  # 添加 `$between` 操作符
                '$not': '__neg__',
            }
            if filter_conditions:
                # 处理过滤条件
                for key, value in filter_conditions.items():
                    column = getattr(table, key, None)
                    if not column:
                        raise AttributeError(f"Attribute '{key}' not found in table '{table.__name__}'")
                    
                    # 处理复合条件
                    if isinstance(value, dict):
                        conditions = []
                        for op, op_value in value.items():
                            if op not in operator_mapping:
                                raise ValueError(f"Unsupported operator '{op}' for field '{key}'")
                            method = operator_mapping[op]
                            if method == 'like':
                                # 处理模糊匹配
                                conditions.append(getattr(column, 'like')(text(f"'%{op_value}%'")))
                            elif method == 'in_':
                                # 处理 IN 查询
                                conditions.append(getattr(column, 'in_')(op_value))
                            elif method == 'between':
                                # 处理范围查询
                                if not isinstance(op_value, list) or len(op_value) != 2:
                                    raise ValueError(f"Expected list of two elements for '$between' in field '{key}'")
                                conditions.append(getattr(column, 'between')(op_value[0], op_value[1]))
                            else:
                                # 处理其他二元操作符
                                conditions.append(getattr(column, method)(op_value))
                        query = query.filter(and_(*conditions)) if len(conditions) > 1 else query.filter(*conditions)
                    else:
                        # 精确匹配
                        query = query.filter(column == value)
            
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
        data_dict=data.to_dict(orient='records')
        while retries:
            try:
                with self.get_session() as session:
                    session.bulk_insert_mappings(table, data_dict)
                    session.commit()
                    retries = 0
                return 1 # 成功插入数据
            except OperationalError as e:
                if "database is locked" in str(e).lower():
                    logging.error(f"Insert failed (Attempt {retries}), retrying in {2 ** n} seconds: {e}")
                    time.sleep(2 ** n)  # 指数退避
                    n += 1
                else:
                    # 其他 OperationalError 异常
                    logging.error(f"Unexpected OperationalError: {e}")
                    raise
            except SQLAlchemyError as e:
                logging.error(f"SQLAlchemy error occurred: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise

    
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
                update_dict = update_fields.copy() 
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