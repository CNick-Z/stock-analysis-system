import os
import struct
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
import mmap

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('tdx_reader.log'), logging.StreamHandler()]
)

class TdxDayReader:
    def __init__(
        self,
        root_dir: str = "e:/tdx/vipdoc",  # 默认通达信安装路径
        security_coefficients: Dict[str, List[float]] = None  # 不同类型证券的缩放系数
    ):
        """
        初始化解析器
        :param root_dir: 通达信vipdoc根目录
        :param security_coefficients: 不同类型证券的缩放系数
        """
        self.root_dir = root_dir
        self.security_coefficients = security_coefficients or {
            'SH_A_STOCK': [0.01, 0.01],    # 沪市A股
            'SZ_A_STOCK': [0.01, 0.01],    # 深市A股
            'SH_INDEX': [0.01, 1.0],       # 沪市指数
            'SZ_INDEX': [0.01, 1.0],       # 深市指数
            'SH_FUND': [0.001, 1.0],       # 沪市基金
            'SZ_FUND': [0.001, 0.01],      # 深市基金
            'SH_BOND': [0.001, 1.0],       # 沪市债券
            'SZ_BOND': [0.001, 0.01],      # 深市债券
            'BJ_STOCK': [0.01, 0.01],      # 北交所股票
            'DEFAULT': [0.01, 0.01]        # 默认值
        }
        self._validate_root_dir()

    def _validate_root_dir(self):
        """验证根目录是否存在"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"根目录不存在: {self.root_dir}")

    def get_security_type(self, file_path: str) -> str:
        """
        根据文件路径确定证券类型
        参考 daily_bar_reader.py 中的实现
        """
        # 从路径中提取交易所和市场代码
        file_name = os.path.basename(file_path)
        exchange = file_name[:2].lower()  # 如 'sh', 'sz'
        code_head = file_name[2:4]       # 股票代码前两位
        
        # 沪市证券
        if exchange == 'sh':
            if code_head.startswith('60') or code_head.startswith('68'):
                return 'SH_A_STOCK'  # 沪市A股
            elif code_head.startswith('90'):
                return 'SH_B_STOCK'  # 沪市B股
            elif code_head.startswith(('00', '88', '99')):
                return 'SH_INDEX'    # 沪市指数
            elif code_head.startswith(('50', '51')):
                return 'SH_FUND'      # 沪市基金
            elif code_head.startswith(('01', '10', '11', '12', '13', '14')):
                return 'SH_BOND'      # 沪市债券
            else:
                return 'SH_A_STOCK'   # 默认为沪市A股
        
        # 深市证券
        elif exchange == 'sz':
            if code_head.startswith(('00', '30')):
                return 'SZ_A_STOCK'  # 深市A股
            elif code_head.startswith('20'):
                return 'SZ_B_STOCK'  # 深市B股
            elif code_head.startswith('39'):
                return 'SZ_INDEX'    # 深市指数
            elif code_head.startswith(('15', '16')):
                return 'SZ_FUND'     # 深市基金
            elif code_head.startswith(('10', '11', '12', '13', '14')):
                return 'SZ_BOND'     # 深市债券
            else:
                return 'SZ_A_STOCK'  # 默认为深市A股
        
        # 北交所证券
        elif exchange == 'bj':
            return 'BJ_STOCK'  # 北交所股票
        
        # 其他证券
        else:
            return 'DEFAULT'  # 使用默认缩放系数

    def unpack_records(self, format_str: str, data: bytes):
        """
        批量解析记录 (参考 base_reader.py)
        :param format_str: 结构格式字符串
        :param data: 二进制数据
        :return: 生成器，每次返回一条解析后的记录
        """
        record_struct = struct.Struct(format_str)
        return (record_struct.unpack_from(data, offset)
                for offset in range(0, len(data), record_struct.size))

    def _parse_single_file_fast(self, file_path: str) -> pd.DataFrame:
        """高效解析单个.day文件（使用内存映射和批量处理）"""
        try:
            # 获取证券类型以确定缩放系数
            security_type = self.get_security_type(file_path)
            price_coeff, vol_coeff = self.security_coefficients.get(
                security_type, self.security_coefficients['DEFAULT']
            )
            
            # 使用内存映射文件提高读取效率
            with open(file_path, 'rb') as f:
                # 创建内存映射文件对象
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    file_size = len(mm)
                    if file_size % 32 != 0:
                        logging.warning(f"文件大小异常（非32字节整数倍）: {file_path}")
                        return pd.DataFrame()
                    
                    # 批量解析所有记录
                    records = list(self.unpack_records('<IIIIIfII', mm))
                    
                    # 如果没有记录，返回空DataFrame
                    if not records:
                        return pd.DataFrame()
                    
                    # 将记录转换为NumPy数组以便高效处理
                    records_arr = np.array(records, dtype=[
                        ('date', 'u4'), 
                        ('open', 'u4'), 
                        ('high', 'u4'), 
                        ('low', 'u4'), 
                        ('close', 'u4'), 
                        ('amount', 'f4'), 
                        ('volume', 'u4'),
                        ('reserved', 'u4')  # 保留字段，忽略
                    ])
                    
                    # 提取字段
                    dates = records_arr['date']
                    opens = records_arr['open'] * price_coeff
                    highs = records_arr['high'] * price_coeff
                    lows = records_arr['low'] * price_coeff
                    closes = records_arr['close'] * price_coeff
                    amounts = records_arr['amount']
                    volumes = records_arr['volume'] * vol_coeff
                    
                    # 转换日期格式 (使用向量化操作)
                    # 将日期整数转换为 datetime 对象
                    date_objs = pd.to_datetime(dates.astype(str), format='%Y%m%d', errors='coerce')
                    
                    # 创建DataFrame
                    df = pd.DataFrame({
                        'date': date_objs,
                        'open': opens,
                        'high': highs,
                        'low': lows,
                        'close': closes,
                        'amount': amounts,
                        'vol': volumes
                    })
                    
                    return df
        except Exception as e:
            logging.error(f"读取文件失败: {file_path} - {str(e)}", exc_info=True)
            return pd.DataFrame()

    def read_directory(self, output_format: str = 'pandas') -> Dict[str, Union[pd.DataFrame, Dict[str, str]]]:
        """
        批量读取目录下所有.day文件
        :param output_format: 输出格式（'pandas'或'dict'）
        :return: 字典，键为文件名，值为DataFrame或路径信息
        """
        all_data = {}
        for market in ['sh', 'sz', 'bj']:
            market_dir = os.path.join(self.root_dir, market, 'lday')
            if not os.path.exists(market_dir):
                logging.warning(f"目录不存在: {market_dir}")
                continue

            for file_name in os.listdir(market_dir):
                if not file_name.endswith('.day'):
                    continue

                file_path = os.path.join(market_dir, file_name)
                stock_code = file_name[2:8]  # 提取代码（如sh688001 → 688001）
                df = self._parse_single_file_fast(file_path)

                if output_format == 'pandas':
                    all_data[stock_code] = df
                elif output_format == 'dict':
                    all_data[stock_code] = {
                        'path': file_path,
                        'columns': df.columns.tolist() if not df.empty else []
                    }

        return all_data
    
    def read_by_code(self, stock_code: str, market: Optional[str] = None) -> pd.DataFrame:
        """
        按股票代码读取单只股票数据
        :param stock_code: 6位股票代码（如688308）
        :param market: 市场标识（可选，自动识别时传None）
        :return: 包含日线数据的DataFrame
        """
        # 自动识别市场
        if market is None:
            if stock_code.startswith(('688', '600', '601', '603', '605', '689')):
                market = 'sh'  # 上海主板和科创板
            elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
                market = 'sz'  # 深圳主板和创业板
            elif stock_code.startswith(('43', '83', '87', '88')):
                market = 'bj'  # 北交所
            else:
                # 尝试自动查找文件
                possible_markets = ['sh', 'sz', 'bj']
                for m in possible_markets:
                    file_name = f"{m}{stock_code.zfill(6)}.day"
                    file_path = os.path.join(self.root_dir, m, 'lday', file_name)
                    if os.path.exists(file_path):
                        market = m
                        break
                else:
                    raise ValueError(f"无法识别的股票代码: {stock_code}，请手动指定market参数")

        # 构建文件路径
        file_name = f"{market}{stock_code.zfill(6)}.day"  # 补齐6位代码
        market_dir = os.path.join(self.root_dir, market, 'lday')
        file_path = os.path.join(market_dir, file_name)

        # 检查文件存在性
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path} (代码: {stock_code}, 市场: {market})")

        # 使用高效方法解析文件
        return self._parse_single_file_fast(file_path)


if __name__ == "__main__":
    # 初始化读取器（自动识别科创板数据）
    reader = TdxDayReader(root_dir="e:/tdx/vipdoc")  # 替换为实际路径
    
    # 测试读取速度
    import time
    
    # 测试科创板股票
    try:
        start_time = time.time()
        df_688 = reader.read_by_code("688001")
        elapsed = time.time() - start_time
        print(f"读取科创板股票数据耗时: {elapsed:.4f}秒")
        print("科创板股票数据：")
        print(df_688[['date', 'open', 'close']].head())
        
        # 测试大盘股
        start_time = time.time()
        df_600 = reader.read_by_code("600000")
        elapsed = time.time() - start_time
        print(f"读取沪市A股数据耗时: {elapsed:.4f}秒")
        
        # 测试创业板股票
        start_time = time.time()
        df_300 = reader.read_by_code("300750")
        elapsed = time.time() - start_time
        print(f"读取创业板股票数据耗时: {elapsed:.4f}秒")
        
        # 测试北交所股票
        start_time = time.time()
        df_bj = reader.read_by_code("430510")
        elapsed = time.time() - start_time
        print(f"读取北交所股票数据耗时: {elapsed:.4f}秒")
        
    except Exception as e:
        print(f"错误: {str(e)}")