#!/bin/bash
# 增量更新指数数据
# 每天 16:05（A股收盘后）运行
# 用法: bash update_index_data.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/var/log/index_data_update.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始更新指数数据" >> "$LOG"

cd "$SCRIPT_DIR/.." || exit 1
python3 scripts/pull_index_data.py >> "$LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 更新完成" >> "$LOG"
