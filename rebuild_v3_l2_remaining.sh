#!/bin/bash
# 重建缺失年份：2020-2024（共5年）
# 每个年重建一次性处理12个月，~17分钟/年 ≈ 85分钟总时长

PYTHON="/usr/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/rebuild_v3_l2_remaining.log"

YEARS=(2020 2021 2022 2023 2024)

echo "开始重建 ${#YEARS[@]} 年..."
echo "日志: $LOG"
echo "开始: $(date)"

for YEAR in "${YEARS[@]}"; do
    echo ""
    echo ">>> [$YEAR] 开始: $(date '+%H:%M:%S')"
    $PYTHON $SCRIPT_DIR/rebuild_l2_fast.py --year $YEAR >> $LOG 2>&1
    echo "<<< [$YEAR] 完成 (退出码: $?)"
done

echo ""
echo "=========================================="
echo "全部完成！结束: $(date)"
echo "=========================================="
