"""
预计算 2026 年 L1 波浪信号 → Parquet
每个股票只算一次（按年缓存），结果存 Parquet 供模拟盘直接读取
"""
import sys, os
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/root/.openclaw/workspace/projects/stock-analysis-system')
_WV_PATH = str(PROJECT_ROOT.parent / 'stock-wave-recognition')
L1_OUT = Path('/data/warehouse/wavechan_l1_signals')
L1_OUT.mkdir(parents=True, exist_ok=True)

def compute_for_symbol(args):
    """为单个股票计算 L1 信号（进程池 worker）"""
    sym, year = args
    try:
        sys.path.insert(0, _WV_PATH)
        from wave_recognizer import identify_wave_stage, label_wave_stage

        weekly_trend, wave_seq = identify_wave_stage(sym, year, years=[year - 1, year])
        label_result = label_wave_stage(sym, year)

        labeled_waves = label_result.get('labeled_waves', [])
        iron_law_passed = label_result.get('iron_law_passed', False)
        cycle_type = label_result.get('cycle_type', 'unknown')

        if not labeled_waves:
            return None

        completed = [w for w in labeled_waves if not w.get('in_progress', False)]
        if not completed:
            return None

        latest = completed[-1]
        wave_label = str(latest.get('elliott_label', ''))

        signal_map = {
            'W2': 'W2_BUY', 'W4': 'W4_BUY',
            'W1': 'C_BUY', 'WA': 'C_BUY', 'WB': 'C_BUY', 'WC': 'C_BUY',
        }
        signal_type = signal_map.get(wave_label, None)
        if signal_type is None:
            return None

        confidence = latest.get('confidence', 0.5)
        iron_bonus = 20 if iron_law_passed else 0
        wave_bonus = 10 if cycle_type == 'impulse' else 5
        total_score = int(confidence * 60 + iron_bonus + wave_bonus)

        ws_map = {
            'W1': 'w1_formed', 'W2': 'w2_formed',
            'W3': 'w3_formed', 'W4': 'w4_formed', 'W5': 'w5_formed',
            'WA': 'w1_formed', 'WB': 'w2_formed', 'WC': 'w3_formed',
        }
        wave_state = ws_map.get(wave_label, 'initial')
        wave_trend_map = {'up': 'long', 'down': 'down'}
        wave_trend = wave_trend_map.get(weekly_trend, 'neutral')

        recent_low = latest.get('low_price') or latest.get('start_price') or latest.get('end_price')
        stop_loss = round(float(recent_low) * 0.97, 2) if recent_low else 0.0

        return {
            'symbol': sym,
            'year': year,
            'has_signal': True,
            'signal_type': signal_type,
            'total_score': total_score,
            'wave_trend': wave_trend,
            'signal_status': 'confirmed',
            'wave_state': wave_state,
            'stop_loss': stop_loss,
            '_weekly_dir': 'bullish' if weekly_trend == 'up' else 'neutral',
            '_impulse_state': _wave_state_to_impulse_state(wave_state),
        }
    except Exception:
        return None

def _wave_state_to_impulse_state(ws):
    m = {
        'w1_formed': 'W2_correction', 'w2_formed': 'W3_in_progress',
        'w3_formed': 'W4_correction', 'w4_formed': 'W5_in_progress',
        'w4_in_progress': 'W4_correction', 'w5_formed': 'W5_done',
    }
    return m.get(ws, 'W1_or_W2')

def main():
    year = 2026

    # 获取 2026 年有数据的股票列表
    l1_dir = Path(f'/data/warehouse/wavechan_l1/extrema_year={year}')
    stocks = sorted([f.replace('.parquet', '') for f in os.listdir(l1_dir) if f.endswith('.parquet')])
    logger.info(f"将计算 {len(stocks)} 只股票的 2026 年 L1 信号...")

    # 并行计算（用 CPU 核数）
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_for_symbol, (sym, year)): sym for sym in stocks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            if r:
                results.append(r)
            if (i + 1) % 500 == 0:
                logger.info(f"  进度 {i+1}/{len(stocks)}...")

    logger.info(f"计算完成: {len(results)}/{len(stocks)} 只有效信号")

    if results:
        df = pd.DataFrame(results)
        out_path = L1_OUT / f"year={year}" / "data.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"已保存: {out_path}")
        print(df.head(5).to_string())
    else:
        logger.warning("无有效信号！")

if __name__ == '__main__':
    main()
