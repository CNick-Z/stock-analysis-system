# db_operations.py
from sqlalchemy import create_engine, Column, String, Float, Boolean, inspect, Index,event,INTEGER 
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import text,asc,desc
from sqlalchemy.sql import or_, and_
from sqlalchemy.ext.declarative import declarative_base
import logging
from contextlib import contextmanager
import pandas as pd
from datetime import datetime
# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

Base = declarative_base()

dynamic_tables_cache = {}

class DailyDataBase(Base):
    __tablename__ = "daily_data"  # 映射到原始表
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
        Index('idx_date_symbol', 'date', 'symbol'),
        Index('idx_processed_symbol', 'processed', 'symbol'),
        Index('idx_date', 'date'),
    )

class TechnicalIndicatorsBase(Base):
    __tablename__ = "technical_indicators"  # 映射到原始表
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
        Index('idx_tec_date_symbol', 'date', 'symbol'),
        Index('idx_tec_date', 'date'),
    )
from sqlalchemy import Table, Column, String, Float, Boolean, Index

def create_daily_data_table(year):
    table_name = f"daily_data_{year}"
    # 检查表类是否已经存在缓存中
    if table_name in dynamic_tables_cache:
        return dynamic_tables_cache[table_name]
    
    # 检查表是否已经存在于元数据中
    if table_name in Base.metadata.tables:
        # 如果表已经存在，直接返回对应的表类
        dynamic_table = type(f"DailyData_{year}", (Base,), {
            "__tablename__": table_name,
            "__table__": Base.metadata.tables[table_name],
            "__abstract__": False,
            "__extend_existing__": True
        })
        dynamic_tables_cache[table_name] = dynamic_table
        return dynamic_table
    
    # 动态创建表
    table = Table(
        table_name,
        Base.metadata,
        Column("date", String, primary_key=True),
        Column("symbol", String, primary_key=True),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Float),
        Column("amount", Float),
        Column("amplitude", Float),
        Column("change_pct", Float),
        Column("change_amount", Float),
        Column("turnover_rate", Float),
        Column("processed", Boolean, default=False),
        Index(f'idx_{table_name}_date_symbol', 'date', 'symbol'),
        Index(f'idx_{table_name}_processed_symbol', 'processed', 'symbol'),
        Index(f'idx_{table_name}_date', 'date'),
    )
    
    # 创建表类
    dynamic_table = type(f"DailyData_{year}", (Base,), {
        "__tablename__": table_name,
        "__table__": table,
        "__abstract__": False,
        "__extend_existing__": True
    })

    # 缓存表类
    dynamic_tables_cache[table_name] = dynamic_table
    return dynamic_table

def create_technical_indicators_table(year):
    table_name = f"technical_indicators_{year}"
    # 检查表类是否已经存在缓存中
    if table_name in dynamic_tables_cache:
        return dynamic_tables_cache[table_name]
    
    # 检查表是否已经存在于元数据中
    if table_name in Base.metadata.tables:
        # 如果表已经存在，直接返回对应的表类
        dynamic_table = type(f"technical_indicators_{year}", (Base,), {
            "__tablename__": table_name,
            "__table__": Base.metadata.tables[table_name],
            "__abstract__": False,
            "__extend_existing__": True
        })
        dynamic_tables_cache[table_name] = dynamic_table
        return dynamic_table
    
    # 动态创建表
    table = Table(
        table_name,
        Base.metadata,
        Column("date", String, primary_key=True),
        Column("symbol", String, primary_key=True),
        Column("sma_5", Float),
        Column("sma_10", Float),
        Column("sma_20", Float),
        Column("sma_55", Float),
        Column("sma_240", Float),
        Column("vol_ma5", Float),
        Column("macd", Float),
        Column("macd_signal", Float),
        Column("macd_histogram", Float),
        Column("rsi_14", Float),
        Column("kdj_k", Float),
        Column("kdj_d", Float),
        Column("kdj_j", Float),
        Column("cci_20", Float),
        Column("williams_r", Float),
        Column("bb_upper", Float),
        Column("bb_middle", Float),
        Column("bb_lower", Float),
        Index(f'idx_{table_name}_date_symbol', 'date', 'symbol'),
        Index(f'idx_{table_name}_date', 'date'),
    )
    
    # 创建表类
    dynamic_table = type(f"TechnicalIndicators_{year}", (Base,), {
        "__tablename__": table_name,
        "__table__": table,
        "__abstract__": False,
        "__extend_existing__": True
    })

    # 缓存表类
    dynamic_tables_cache[table_name] = dynamic_table
    return dynamic_table

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

class PositionDetail(Base):
    """持仓明细表"""
    __tablename__ = 'position_details'    
    #id = Column(INTEGER , primary_key=True, autoincrement=True)
    date = Column(String, primary_key=True,nullable=False, comment='交易日期') 
    symbol = Column(String(10),primary_key=True, nullable=False, comment='股票代码')
    price = Column(Float, nullable=False, comment='成交价格')
    newprice = Column(Float, comment='最新价格')
    highprice = Column(Float, comment='最高价格')
    quantity = Column(Float, nullable=False, comment='成交数量')
    commission = Column(Float, nullable=False, comment='手续费')
    sell_date = Column(String, comment='卖出日期')  # 卖出相关字段允许为空
    sell_price = Column(Float, comment='卖出价格')
    pnl = Column(Float, comment='盈亏金额')
    
    # 复合索引
    __table_args__ = (
        Index('idx_detail_symbol_date', 'symbol', 'date'),  # 按股票+日期查询
        Index('idx_sell_date', 'sell_date'),  # 卖出日期查询
    )

class PositionStatus(Base):
    """持仓情况表"""
    __tablename__ = 'position_status'
    date = Column(String, primary_key=True, comment='统计日期')
    total_assets = Column(Float, nullable=False, comment='总资产')
    stock_value = Column(Float, nullable=False, comment='持仓市值')
    cash = Column(Float, nullable=False, comment='可用现金')
    position_ratio = Column(Float, nullable=False, comment='仓位比例')
    available_position = Column(Float, nullable=False, comment='可用仓位额度')
    
    # 时间索引
    __table_args__ = (
        Index('idx_status_date', 'date'),  # 日期主键自动建立索引
    )


class DatabaseManager:
    def __init__(self, db_url,historical_db_url=None):
        self.db_url = db_url
        self.historical_db_url = db_url.replace('.db','_his.db') if historical_db_url is None else historical_db_url            
        self.engine = create_engine(self.db_url,
                        pool_size=10, 
                        max_overflow=20,
                        pool_pre_ping=True,
                        connect_args={
                            'check_same_thread': False
                            })
        self.historical_engine = create_engine(self.historical_db_url,
                        pool_size=10, 
                        max_overflow=20,
                        pool_pre_ping=True,
                         connect_args={
                            'check_same_thread': False
                            })
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.HistoricalSession = scoped_session(sessionmaker(bind=self.historical_engine))
    
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
            DailyDataBase.__table__.create(self.engine)
        if "technical_indicators" not in existing_tables:
            TechnicalIndicatorsBase.__table__.create(self.engine)
        if "stock_basic_info" not in existing_tables:
            StockBasicInfo.__table__.create(self.engine)
        if "position_details" not in existing_tables:
            PositionDetail.__table__.create(self.engine)
        if  "position_status" not in existing_tables:
            PositionStatus.__table__.create(self.engine)
    
    def ensure_dynamic_table_exists(self, table_class, year):
        """确保动态分表存在"""
        if table_class == DailyDataBase:
            dynamic_table = create_daily_data_table(year)
        elif table_class == TechnicalIndicatorsBase:
            dynamic_table = create_technical_indicators_table(year)
        else:
            return
        # 检查历史库中是否存在该表
        inspector = inspect(self.historical_engine)
        if dynamic_table.__tablename__ not in inspector.get_table_names():
            dynamic_table.__table__.create(self.historical_engine)

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

    @contextmanager
    def get_his_session(self):
        """获取数据库会话"""
        his_session = self.HistoricalSession()
        try:
            yield his_session
            his_session.commit()
        except Exception as e:
            his_session.rollback()
            logging.error(f"Database error: {e}")
        finally:
            his_session.close()

    def _get_dynamic_table(self, base_table, year):
        if base_table == DailyDataBase:
            dynamic_table = create_daily_data_table(year)
        elif base_table == TechnicalIndicatorsBase:
            dynamic_table = create_technical_indicators_table(year)
        # 显式创建表到历史数据库
        inspector = inspect(self.historical_engine)
        if dynamic_table.__tablename__ not in inspector.get_table_names():
            dynamic_table.__table__.create(self.historical_engine)
        return dynamic_table

    def load_data(self, table_class, filter_conditions=None, distinct_column=None, limit=None, columns=None, order_by=None):
        """优化版数据加载方法（保留原有查询条件判断）"""
        # 如果传入的是基类，则需要动态确定表名
        if table_class in [DailyDataBase, TechnicalIndicatorsBase]:
            return self._load_dynamic_data(table_class, filter_conditions, distinct_column, limit, columns, order_by)
        else:
            return self._optimized_query_data(table_class, filter_conditions, distinct_column, limit, columns, order_by)
        
    def _load_dynamic_data(self, table_class, filter_conditions=None, distinct_column=None, limit=None, columns=None, order_by=None):
        """处理动态分表查询（保留原有条件判断）"""
        # 保留原有的日期条件解析逻辑
        if 'date' in (filter_conditions or {}):
            date_value = filter_conditions['date']
            current_year = datetime.now().year
            query_history = False
            query_real_time = False
            start_year = None
            end_year = None

            if isinstance(date_value, dict):
                # 保留原有的嵌套条件处理
                for op, op_value in date_value.items():
                    if op == '$between':
                        if isinstance(op_value, list) and len(op_value) == 2:
                            start_date = op_value[0]
                            end_date = op_value[1]
                            start_year = int(start_date[:4])
                            end_year = int(end_date[:4])
                    elif op in ['$lt', '$lte']:
                        end_date = op_value
                        end_year = int(end_date[:4])
                        start_year = None
                    elif op in ['$gt', '$gte']:
                        start_date = op_value
                        start_year = int(start_date[:4])
                        end_year = None

                # 保留原有的查询范围判断
                if end_year is not None and end_year < current_year:
                    query_history = True
                elif start_year is not None and start_year < current_year:
                    query_history = True
                    query_real_time = True
                elif end_year is not None and end_year >= current_year:
                    query_real_time = True
            elif isinstance(date_value, str):
                year = int(date_value[:4])
                if year < current_year:
                    query_history = True
                else:
                    query_real_time = True
            elif isinstance(date_value, list):
                start_year = int(date_value[0][:4])
                end_year = int(date_value[1][:4])
                if end_year < current_year:
                    query_history = True
                elif start_year < current_year and end_year >= current_year:
                    query_history = True
                    query_real_time = True

            # 使用优化后的查询方法
            results = []
            if query_history:
                if start_year is None:
                    start_year = end_year - 1 if end_year is not None else current_year - 1
                if end_year is None:
                    end_year = current_year

                # 并行查询历史数据
                historical_dfs = self._parallel_query_history(
                    table_class, 
                    range(start_year, end_year + 1),
                    filter_conditions,
                    distinct_column,
                    limit,
                    columns,
                    order_by
                )
                results.extend(historical_dfs)
            
            if query_real_time:
                real_time_result = self._optimized_query_data(
                    table_class, 
                    filter_conditions, 
                    distinct_column, 
                    limit, 
                    columns, 
                    order_by
                )
                results.append(real_time_result)

            return pd.concat(results) if results else pd.DataFrame()
        else:
            # 没有date条件，默认查询实时数据
            return self._optimized_query_data(
                table_class, 
                filter_conditions, 
                distinct_column, 
                limit, 
                columns, 
                order_by
            )

    def _parallel_query_history(self, table_class, years, filter_conditions, distinct_column, limit, columns, order_by):
        """并行查询历史数据表"""
        from concurrent.futures import ThreadPoolExecutor
        
        def query_single_year(year):
            table = self._get_dynamic_table(table_class, year)
            return self._optimized_query_data(
                table,
                filter_conditions,
                distinct_column,
                limit,
                columns,
                order_by
            )
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(query_single_year, year) for year in years]
            return [future.result() for future in futures]

    def _optimized_query_data(self, table, filter_conditions=None, distinct_column=None, limit=None, columns=None, order_by=None, chunk_size=100000):
        """优化后的基础查询方法（保留原有过滤条件处理）"""
        is_history = table.__tablename__.startswith(('daily_data_', 'technical_indicators_'))
        session_maker = self.get_his_session if is_history else self.get_session
        
        with session_maker() as session:
            # 1. 构建基础查询（保留原有列选择逻辑）
            if columns:
                if not isinstance(columns, list):
                    columns = [columns]
                columns_attr = []
                for col in columns:
                    if isinstance(col, str):
                        col_attr = getattr(table, col, None)
                        if col_attr is None:
                            raise AttributeError(f"Column '{col}' not found in table '{table.__name__}'")
                        columns_attr.append(col_attr)
                    else:
                        columns_attr.append(col)
                if distinct_column:
                    if distinct_column not in columns:
                        columns_attr.append(getattr(table, distinct_column))
                    query = session.query(*columns_attr).distinct()
                else:
                    query = session.query(*columns_attr)
            else:
                if distinct_column:
                    query = session.query(getattr(table, distinct_column).distinct())
                else:
                    query = session.query(table)
            
            # 2. 应用过滤条件（保留原有条件处理逻辑）
            if filter_conditions:
                operator_mapping = {
                    '$eq': '__eq__',
                    '$ne': '__ne__',
                    '$gt': '__gt__',
                    '$lt': '__lt__',
                    '$gte': '__ge__',
                    '$lte': '__le__',
                    '$like': 'like',
                    '$not_like': 'notlike',
                    '$in': 'in_',
                    '$between': 'between',
                    '$not': '__neg__',
                }
                for key, value in filter_conditions.items():
                    column = getattr(table, key, None)
                    if not column:
                        raise AttributeError(f"Attribute '{key}' not found in table '{table.__name__}'")
                    
                    if isinstance(value, dict):
                        conditions = []
                        for op, op_value in value.items():
                            if op not in operator_mapping:
                                raise ValueError(f"Unsupported operator '{op}' for field '{key}'")
                            method = operator_mapping[op]
                            if method == 'like':
                                conditions.append(getattr(column, 'like')(text(f"'%{op_value}%'")))
                            elif method == 'notlike':
                                conditions.append(getattr(column, 'notlike')(text(f"'%{op_value}%'")))
                            elif method == 'in_':
                                conditions.append(getattr(column, 'in_')(op_value))
                            elif method == 'between':
                                if not isinstance(op_value, list) or len(op_value) != 2:
                                    raise ValueError(f"Expected list of two elements for '$between' in field '{key}'")
                                conditions.append(getattr(column, 'between')(op_value[0], op_value[1]))
                            else:
                                conditions.append(getattr(column, method)(op_value))
                        query = query.filter(and_(*conditions)) if len(conditions) > 1 else query.filter(*conditions)
                    else:
                        query = query.filter(column == value)
            
            # 3. 应用排序（保留原有排序逻辑）
            if order_by:
                order_conditions = []
                for condition in order_by:
                    if isinstance(condition, str):
                        column = getattr(table, condition, None)
                        if not column:
                            raise AttributeError(f"Column '{condition}' not found in table '{table.__name__}'")
                        order_conditions.append(asc(column))
                    elif isinstance(condition, dict):
                        if 'column' not in condition or 'direction' not in condition:
                            raise ValueError("Invalid order_by condition format. Expected 'column' and 'direction' keys.")
                        column = getattr(table, condition['column'], None)
                        if not column:
                            raise AttributeError(f"Column '{condition['column']}' not found in table '{table.__name__}'")
                        direction = condition['direction'].lower()
                        if direction == 'asc':
                            order_conditions.append(asc(column))
                        elif direction == 'desc':
                            order_conditions.append(desc(column))
                        else:
                            raise ValueError(f"Invalid direction '{direction}' for order_by. Use 'asc' or 'desc'.")
                    else:
                        raise ValueError("Invalid order_by condition format. Use string or dictionary.")
                query = query.order_by(*order_conditions)
            
            # 4. 分块获取数据（新增优化）
            chunks = []
            offset = 0
            while True:
                chunk_query = query.offset(offset).limit(chunk_size) if limit is None else query.limit(limit)
                df_chunk = pd.read_sql(chunk_query.statement, session.bind)
                
                if df_chunk.empty:
                    break
                    
                chunks.append(df_chunk)
                offset += chunk_size
                
                # 如果设置了limit或者获取的行数少于chunk_size，则退出循环
                if limit is not None or len(df_chunk) < chunk_size:
                    break
            
            return pd.concat(chunks) if chunks else pd.DataFrame()


    
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
                # 根据 update_fields 的类型构建更新字典
                if isinstance(update_fields, list):
                    # 如果 update_fields 是列表，使用列表元素作为键，从 item 中获取值
                    update_dict = {field: item[field] for field in update_fields if field in item}
                    if len(update_dict) != len(update_fields):
                        missing_fields = [field for field in update_fields if field not in item]
                        raise KeyError(f"Missing update fields in data: {missing_fields}")
                elif isinstance(update_fields, dict):
                    # 如果 update_fields 是字典，使用字典的键作为键，字典的值作为更新的值
                    update_dict = update_fields.copy()
                else:
                    raise TypeError("update_fields must be a list or a dictionary")
            
                # 合并过滤条件和更新字典
                update_values.append({
                    **filter_dict,
                    **update_dict
                })
            
            # 批量更新
            session.bulk_update_mappings(table, update_values)
            session.commit()
    
    def query_count(self, table, filter_conditions=None):
        """查询数据数量"""
        with self.get_session() as session:
            query = session.query(table)
            if filter_conditions:
                for key, value in filter_conditions.items():
                    query = query.filter(getattr(table, key) == value)
            count = query.count()
        return count

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    db_url = "sqlite:///c:/db/stock_data.db"
    filter_conditions = {
    'date': {'$between': ['2020-01-01', '2023-12-31']}
    }
    db_manager = DatabaseManager(db_url=db_url)
    df = db_manager.load_data(DailyDataBase, filter_conditions=filter_conditions)
    print(df)