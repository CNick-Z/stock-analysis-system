#!/usr/bin/env python3
"""
Wavechan 信号流程脚本
每天16:05执行：更新数据 → 计算信号 → 发送QQ报告
"""
import os, sys, time, json, subprocess
from datetime import datetime

# ========== 1. 更新Parquet数据 ==========
print(f"[{time.strftime('%H:%M:%S')}] Step1: 更新Parquet数据...")
try:
    result = subprocess.run(
        [sys.executable, '/root/.openclaw/workspace/scripts/export_increment_to_parquet.py'],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        print(f"[{time.strftime('%H:%M:%S')}] Parquet更新成功")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Parquet更新失败: {result.stderr[:100]}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] Parquet更新异常: {e}")

# ========== 2. 生成Wavechan信号 ==========
print(f"[{time.strftime('%H:%M:%S')}] Step2: 生成Wavechan信号...")
try:
    result = subprocess.run(
        [sys.executable, '/root/.openclaw/workspace/projects/stock-analysis-system/wavechan_daily_signal.py'],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout[-500:] if result.stdout else "")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] 信号计算异常: {e}")

# ========== 3. 读取信号并发送QQ ==========
print(f"[{time.strftime('%H:%M:%S')}] Step3: 发送QQ报告...")
signal_file = '/tmp/wavechan_signals.json'
try:
    if os.path.exists(signal_file):
        with open(signal_file) as f:
            sig = json.load(f)
        
        date = sig.get('date', '')
        buy_count = sig['summary']['buy_signals']
        hold_count = sig['summary']['hold_signals']
        top_signals = sig.get('top_buy_signals', [])
        
        # 构建消息
        lines = [f"📊 Wavechan每日信号 [{date}]"]
        lines.append(f"买入信号: {buy_count} 个")
        lines.append(f"持有信号: {hold_count} 个")
        lines.append("")
        
        if top_signals:
            lines.append("Top5买入信号:")
            for i, s in enumerate(top_signals[:5], 1):
                sl = f"{s['stop_loss']:.2f}" if s.get('stop_loss') else "N/A"
                tgt = f"{s['target']:.2f}" if s.get('target') else "N/A"
                lines.append(f"{i}. {s['symbol']} | {s['wave_stage']} | 置信{s['confidence']:.2f} | 止损{sl} | 目标{tgt}")
        else:
            lines.append("今日无买入信号，市场无趋势")
        
        message = '\n'.join(lines)
        print(message)
        
        # 发送QQ
        try:
            sys.path.insert(0, '/root/.openclaw/workspace/scripts')
            from qqbot_notifier import QQBotNotifier
            notifier = QQBotNotifier()
            result = notifier.send_message(message, openid='B1DE0C2788382B67AF73F1C189A6A5C5')
            if result.get('success'):
                print(f"[{time.strftime('%H:%M:%S')}] QQ报告已发送")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] QQ发送失败: {result.get('error')}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] QQ发送失败: {e}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 信号文件不存在")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] 发送报告异常: {e}")

print(f"[{time.strftime('%H:%M:%S')}] Wavechan每日流程完成")
