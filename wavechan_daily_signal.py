#!/usr/bin/env python3
"""
Wavechan 每日信号脚本（增量版）
✅ 缓存预热特征，每天只算新的一天
✅ 信号结果保存到文件
"""
import sys, os, time, json
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.parquet_db import ParquetDatabaseIntegrator
from strategies.wavechan_fast import compute_wavechan_features_fast

CONFIG = {
    'wave_threshold_pct': 0.025,
    'decline_threshold': -0.15,
    'consolidation_threshold': 0.05,
    'stop_loss_pct': 0.03,
    'profit_target_pct': 0.25,
}

CACHE_DIR = '/root/.openclaw/workspace/data/wavechan_cache'
SIGNAL_FILE = '/tmp/wavechan_signals.json'
LOG_FILE = '/tmp/wavechan_signal.log'

os.makedirs(CACHE_DIR, exist_ok=True)

def get_latest_trade_date(df):
    """从数据中获取最近有数据的交易日"""
    if 'date' in df.columns:
        dates = sorted(df['date'].unique())
        if len(dates) > 0:
            return dates[-1]
    return None

def get_cache_path():
    """获取缓存文件路径"""
    return os.path.join(CACHE_DIR, 'features_cache.parquet')

def load_cached_features():
    """加载缓存的特征数据"""
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        try:
            import pandas as pd
            df = pd.read_parquet(cache_path)
            print(f"[{time.strftime('%H:%M:%S')}] 加载缓存: {len(df)}行, 最近日期:{df['date'].max()}")
            return df
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 缓存加载失败: {e}")
    return None

def save_cached_features(features):
    """保存特征到缓存"""
    cache_path = get_cache_path()
    features.to_parquet(cache_path, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] 缓存已保存: {len(features)}行")

def compute_signals():
    """计算最新信号"""
    db = ParquetDatabaseIntegrator()
    
    # 先尝试加载缓存
    cached = load_cached_features()
    
    # 确定要加载的数据范围
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    if cached is not None:
        # 有缓存：只加载比缓存更新的数据
        last_date = cached['date'].max()
        start_date = last_date  # 从缓存最后一天开始
        print(f"[{time.strftime('%H:%M:%S')}] 增量模式: 从 {start_date} 开始加载新数据")
        
        t0 = time.time()
        df = db.fetch_daily_data(
            start_date, end_date,
            columns=['date','symbol','open','high','low','close','volume']
        )
        
        if df.empty or df['date'].max() == last_date:
            print(f"[{time.strftime('%H:%M:%S')}] 没有新数据")
            trade_date = last_date
            features = cached
        else:
            # 增量模式：需要把缓存里的历史数据合并进来再计算
            # 因为compute_wavechan_features需要每只股票至少60天数据
            t1 = time.time()
            new_date = df['date'].max()
            
            # 确保symbol类型一致（增量文件是int64，缓存是string）
            df['symbol'] = df['symbol'].astype(str)
            
            # 获取新数据中涉及的股票列表
            new_symbols = set(df['symbol'].unique())
            
            # 从缓存中获取这些股票的全部历史数据
            history = cached[cached['symbol'].isin(new_symbols)]
            
            # 合并：全部历史 + 新数据（去重，保留最新的）
            # concat后按symbol和date排序，然后用drop_duplicates保留每只股票每天最后的记录
            df_combined = pd.concat([history, df], ignore_index=True)
            df_combined = df_combined.sort_values(['symbol', 'date']).drop_duplicates(
                subset=['symbol', 'date'], keep='last'
            ).reset_index(drop=True)
            
            print(f"[{time.strftime('%H:%M:%S')}] 增量计算: {len(df)}行新数据 + {len(history)}行历史, 去重后{len(df_combined)}行")
            new_features = compute_wavechan_features_fast(df_combined, CONFIG)
            
            # 只保留新数据的特征结果
            new_features = new_features[new_features['date'] == new_date]
            print(f"[{time.strftime('%H:%M:%S')}] 新特征计算完成: {len(new_features)}行, {time.time()-t1:.1f}s")
            
            # 合并缓存：保留缓存中该日期之前的数据 + 新特征
            features = pd.concat([cached[cached['date'] < new_date], new_features], ignore_index=True)
            save_cached_features(features)
            trade_date = new_date
    else:
        # 无缓存：从头计算120天
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
        print(f"[{time.strftime('%H:%M:%S')}] 全量模式: {start_date} ~ {end_date}")
        
        t0 = time.time()
        df = db.fetch_daily_data(
            start_date, end_date,
            columns=['date','symbol','open','high','low','close','volume']
        )
        print(f"[{time.strftime('%H:%M:%S')}] 数据: {len(df)}条, {time.time()-t0:.1f}s")
        
        t1 = time.time()
        features = compute_wavechan_features_fast(df, CONFIG)
        print(f"[{time.strftime('%H:%M:%S')}] 特征: {len(features)}行, {time.time()-t1:.1f}s")
        
        save_cached_features(features)
        trade_date = get_latest_trade_date(features)
    
    # 获取当日信号
    day_df = features[features['date'] == trade_date]
    if day_df.empty:
        print(f"[{time.strftime('%H:%M:%S')}] {trade_date} 无数据")
        return None
    
    buy_cands = day_df[day_df['daily_signal'] == '买入'].sort_values('daily_confidence', ascending=False)
    hold_cands = day_df[day_df['daily_signal'] == 'hold']
    
    print(f"\n{'='*60}")
    print(f"📊 {trade_date} Wavechan 信号报告")
    print(f"{'='*60}")
    print(f"市场状态: 共 {len(day_df)} 只股票")
    print(f"买入信号: {len(buy_cands)} 个")
    print(f"持有信号: {len(hold_cands)} 个")
    
    if len(buy_cands) > 0:
        print(f"\n📈 Top买入信号 (按置信度):")
        print(f"{'代码':<8} {'波浪':<8} {'阶段':<10} {'置信度':<8} {'止损价':<8} {'目标价':<8}")
        for _, r in buy_cands.head(10).iterrows():
            sl = r.get('stop_loss', 0)
            tgt = r.get('target', 0)
            print(f"{r.symbol:<8} {str(r.get('wave_trend','')):<8} {str(r.get('wave_stage','')):<10} {r.get('daily_confidence',0):.3f}     {sl:.2f}     {tgt:.2f}")
    
    # 构建结果
    result = {
        'date': trade_date,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_stocks': int(len(day_df)),
            'buy_signals': int(len(buy_cands)),
            'hold_signals': int(len(hold_cands)),
        },
        'top_buy_signals': [],
    }
    
    for _, r in buy_cands.head(10).iterrows():
        sl = r.get('stop_loss', 0)
        tgt = r.get('target', 0)
        result['top_buy_signals'].append({
            'symbol': r.symbol,
            'wave_trend': str(r.get('wave_trend', '')),
            'wave_stage': str(r.get('wave_stage', '')),
            'confidence': float(r.get('daily_confidence', 0)),
            'stop_loss': float(sl) if sl else None,
            'target': float(tgt) if tgt else None,
        })
    
    # 保存信号
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n信号已保存到 {SIGNAL_FILE}")
    return result

if __name__ == '__main__':
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wavechan每日信号开始")
    compute_signals()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 完成")
