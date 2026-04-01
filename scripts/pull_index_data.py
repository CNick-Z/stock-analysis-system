#!/usr/bin/env python3
"""
下载关键指数历史数据并存为 Parquet
====================================
指数：沪深300、上证指数、创业板指、科创50
数据源：baostock（免费，无需Token）
存储：/data/warehouse/indices/
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import baostock as bs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# (bs_code, parquet_name, display_name, start_year)
INDICES = [
    ("sh.000300", "CSI300", "沪深300",   2006),
    ("sh.000001", "SSE",    "上证指数",   2006),
    ("sz.399006", "GEM",    "创业板指",   2010),
    ("sh.000688", "STAR50", "科创50",    2019),
]

FIELDS = ["date", "code", "open", "high", "low", "close", "volume", "amount"]

OUT_DIR = Path("/data/warehouse/indices")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_index(code: str, start_year: int = 2006) -> pd.DataFrame:
    """从 baostock 下载单个指数历史数据（分年抓取，避免超时）"""
    bs.login()
    records = []
    for year in range(start_year, 2027):
        rs = bs.query_history_k_data_plus(
            code, ",".join(FIELDS),
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            frequency="d", adjustflag="3"
        )
        if rs.error_msg != "success":
            print(f"    警告 [{year}]: {rs.error_msg}", flush=True)
            continue
        while rs.next():
            records.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(records, columns=FIELDS)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume","amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df


def main():
    for bs_code, fname, name, start_year in INDICES:
        out_path = OUT_DIR / f"{fname}.parquet"
        if out_path.exists():
            # 增量：只取最新一条之后的数据
            existing = pq.read_table(out_path).to_pandas()
            last_date = existing["date"].max()
            print(f"  {name}: 已存在，到 {last_date.date()}，增量抓取...")
            new_data = fetch_index(bs_code, start_year=last_date.year + 1 if last_date.year < 2026 else 2026)
            new_data = new_data[new_data["date"] > last_date]
            if not new_data.empty:
                combined = pd.concat([existing, new_data]).drop_duplicates("date").sort_values("date")
                combined.to_parquet(out_path, index=False)
                print(f"    +{len(new_data)} 条新数据，已追加")
            else:
                print(f"    无新数据")
        else:
            print(f"  {name}: 首次下载...")
            df = fetch_index(bs_code, start_year=start_year)
            df.to_parquet(out_path, index=False)
            print(f"    {len(df)} 条，{df['date'].min().date()} ~ {df['date'].max().date()}")

        # 验证
        t = pq.read_table(out_path).to_pandas()
        print(f"    → {fname}.parquet: {len(t)} 行 | {t['date'].min().date()} ~ {t['date'].max().date()}")

    print("\n✅ 全部完成")


if __name__ == "__main__":
    main()
