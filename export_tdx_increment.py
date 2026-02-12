#!/usr/bin/env python3
"""
通达信每日增量数据导出脚本
从本地通达信数据读取今日数据，导出为 CSV/JSON
参考原项目 tdx_reader.py 实现
"""

import os
import sys
import struct
import json
import csv
import logging
import mmap
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'tdx_export_{datetime.now().strftime("%Y%m%d")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class TdxDayReader:
    """通达信日线数据读取器（参考原项目实现）"""

    def __init__(self, tdx_root_dir: str):
        self.tdx_root_dir = tdx_root_dir
        if not os.path.exists(tdx_root_dir):
            raise FileNotFoundError(f"通达信目录不存在: {tdx_root_dir}")

    def get_file_path(self, symbol: str) -> Optional[str]:
        """获取通达信数据文件路径"""
        if symbol.startswith(('6', '5')):
            market = 'sh'
        elif symbol.startswith(('0', '3')):
            market = 'sz'
        elif symbol.startswith(('4', '8')):
            market = 'bj'
        else:
            market = 'sh'
        
        file_path = os.path.join(
            self.tdx_root_dir, market, 'lday', f"{market}{symbol}.day"
        )
        return file_path if os.path.exists(file_path) else None

    def get_security_type(self, file_path: str) -> str:
        """根据文件路径确定证券类型"""
        file_name = os.path.basename(file_path)
        exchange = file_name[:2].lower()
        code_head = file_name[2:4]
        
        if exchange == 'sh':
            if code_head.startswith(('60', '68')):
                return 'SH_A_STOCK'
            elif code_head.startswith('90'):
                return 'SH_INDEX'
            elif code_head.startswith(('50', '51')):
                return 'SH_FUND'
            else:
                return 'SH_A_STOCK'
        elif exchange == 'sz':
            if code_head.startswith(('00', '30')):
                return 'SZ_A_STOCK'
            elif code_head.startswith('39'):
                return 'SZ_INDEX'
            elif code_head.startswith(('15', '16')):
                return 'SZ_FUND'
            else:
                return 'SZ_A_STOCK'
        else:
            return 'DEFAULT'

    def get_coefficients(self, file_path: str) -> tuple:
        """获取缩放系数"""
        security_type = self.get_security_type(file_path)
        coeffs = {
            'SH_A_STOCK': [0.01, 0.01],
            'SZ_A_STOCK': [0.01, 0.01],
            'SH_INDEX': [0.01, 1.0],
            'SZ_INDEX': [0.01, 1.0],
            'SH_FUND': [0.001, 1.0],
            'SZ_FUND': [0.001, 0.01],
            'DEFAULT': [0.01, 0.01]
        }
        return coeffs.get(security_type, coeffs['DEFAULT'])

    def read_stock_data(self, symbol: str, target_date: str = None) -> List[dict]:
        """读取单只股票的数据"""
        file_path = self.get_file_path(symbol)
        if not file_path:
            return []

        try:
            price_coeff, vol_coeff = self.get_coefficients(file_path)
            
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    file_size = len(mm)
                    if file_size % 32 != 0:
                        return []
                    
                    # 批量解析所有记录
                    records = []
                    record_struct = struct.Struct('<IIIIIfII')
                    
                    for offset in range(0, file_size, 32):
                        chunk = mm[offset:offset+32]
                        if len(chunk) < 32:
                            break
                        
                        try:
                            fields = record_struct.unpack(chunk)
                        except struct.error:
                            continue
                        
                        # 解析日期
                        date_int = fields[0]
                        year = date_int // 10000
                        month = (date_int // 100) % 100
                        day = date_int % 100
                        date_str = f"{year:04d}-{month:02d}-{day:02d}"
                        
                        # 日期过滤
                        if target_date and date_str != target_date:
                            continue
                        
                        records.append({
                            'date': date_str,
                            'symbol': symbol,
                            'open': round(fields[1] * price_coeff, 3),
                            'close': round(fields[4] * price_coeff, 3),
                            'high': round(fields[2] * price_coeff, 3),
                            'low': round(fields[3] * price_coeff, 3),
                            'volume': round(fields[6] * vol_coeff, 2),
                            'amount': round(fields[5], 2),
                        })
                    
                    return records
                    
        except Exception as e:
            logging.warning(f"读取 {symbol} 时出错: {e}")
            return []

    def get_all_symbols(self) -> List[str]:
        """获取通达信目录下所有股票代码"""
        symbols = set()
        for market in ['sh', 'sz', 'bj']:
            lday_dir = os.path.join(self.tdx_root_dir, market, 'lday')
            if os.path.exists(lday_dir):
                for f in os.listdir(lday_dir):
                    if f.endswith('.day'):
                        s = f[2:-4]
                        if s:
                            symbols.add(s)
        return sorted(symbols)

    def get_latest_date(self) -> str:
        """获取通达信数据中最新的日期"""
        for symbol in self.get_all_symbols()[:100]:
            file_path = self.get_file_path(symbol)
            if not file_path:
                continue
            
            try:
                with open(file_path, 'rb') as f:
                    f.seek(-32, 2)
                    chunk = f.read(32)
                    if len(chunk) < 32:
                        continue
                    fields = struct.unpack('<IIIIIfII', chunk)
                    date_int = fields[0]
                    return f"{date_int // 10000}-{(date_int // 100) % 100:02d}-{date_int % 100:02d}"
            except:
                continue
        
        return datetime.now().strftime('%Y-%m-%d')


class TdxExport:
    def __init__(self, tdx_root_dir: str, export_dir: str, logger=None):
        self.tdx_root_dir = tdx_root_dir
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.reader = TdxDayReader(tdx_root_dir)
        self.logger = logger or logging.getLogger(__name__)

    def calculate_derived_fields(self, records: List[dict]) -> List[dict]:
        for r in records:
            r['amplitude'] = round((r['high'] - r['low']) / r['close'] * 100, 4) if r['close'] > 0 else 0
            r['change_amount'], r['change_pct'], r['turnover_rate'] = 0, 0, 0
        return records

    def export(self, target_date: str = None) -> tuple:
        target_date = target_date or self.reader.get_latest_date()
        self.logger.info(f"读取 {target_date} 的数据...")
        
        all_data = []
        for symbol in self.reader.get_all_symbols():
            records = self.reader.read_stock_data(symbol, target_date)
            all_data.extend(records)
        
        if not all_data:
            return None, None
        
        all_data = self.calculate_derived_fields(all_data)
        ds = target_date.replace('-', '')
        csv_p = self.export_dir / f"tdx_increment_{ds}.csv"
        json_p = self.export_dir / f"tdx_increment_{ds}.json"

        with open(csv_p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, ['date','symbol','open','high','low','close',
                                   'volume','amount','amplitude','change_pct','change_amount','turnover_rate'])
            w.writeheader()
            w.writerows(all_data)

        with open(json_p, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"导出 {len(all_data)} 条, {csv_p.stat().st_size/1024:.1f}KB")
        return str(csv_p), str(json_p)

    def show_status(self):
        ld = self.reader.get_latest_date()
        print(f"目录: {self.tdx_root_dir}")
        print(f"最新日期: {ld}")
        print(f"股票数: {len(self.reader.get_all_symbols())}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tdx-dir', '-t', default='e:/tdx/vipdoc')
    parser.add_argument('--export-dir', '-e', default='./exports')
    parser.add_argument('--date', '-d')
    parser.add_argument('--status', '-s', action='store_true')
    args = parser.parse_args()
    
    logger = setup_logging(args.export_dir)
    
    try:
        exporter = TdxExport(args.tdx_dir, args.export_dir, logger)
        if args.status:
            exporter.show_status()
        else:
            exporter.export(args.date)
    except Exception as ex:
        logger.error(str(ex))


if __name__ == "__main__":
    main()
