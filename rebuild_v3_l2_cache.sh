#!/bin/bash
# ============================================================
# V3 L2 缓存批量重建脚本
# 重建所有缺失的月份（2018-2024 全年）
# ============================================================

LOG_FILE="/tmp/rebuild_v3_l2.log"
PYTHON="/usr/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "V3 L2 缓存批量重建"
echo "开始时间: $(date)"
echo "日志: $LOG_FILE"
echo "=========================================="

# 年份列表（2018-2024）
YEARS=(2018 2019 2020 2021 2022 2023 2024)

for YEAR in "${YEARS[@]}"; do
    echo ""
    echo ">>> 开始重建 $YEAR 年 L2 缓存"
    echo "时间: $(date)"

    # 重建全年12个月
    for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
        MON_STR="$YEAR-$MONTH"
        echo "--- 重建 $MON_STR ---"
        
        $PYTHON $SCRIPT_DIR/wavechan_daily_incremental.py \
            --rebuild-month $MON_STR \
            2>&1 | tee -a $LOG_FILE
        
        # 记录完成状态
        echo "[$(date)] 完成 $MON_STR" >> $LOG_FILE
    done

    echo "<<< $YEAR 年完成"
done

echo ""
echo "=========================================="
echo "全部完成！结束时间: $(date)"
echo "=========================================="
