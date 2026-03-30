#!/usr/bin/env python3
"""
因子 IC/IR 分析器 - 年度滚动方案 + 性能监控
研究 Score 策略各因子的预测能力（IC/IR）
"""

import pandas as pd
import numpy as np
import tracemalloc
import psutil
import os
import warnings
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ============ 配置 ============
DATA_DIR = '/root/.openclaw/workspace/data/warehouse'
OUT_DIR = '/root/.openclaw/workspace/projects/stock-analysis-system/factor_research/results'
YEAR_RANGE = range(2018, 2026)  # 2018-2025 完整年

# 内存安全阈值
MEMORY_LIMIT_GB = 1.0  # 超过1GB立即停止
CPU_LIMIT_PCT = 95     # CPU超过95%持续30秒停止

# ============ 性能监控 ============
class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()
        self.peak_memory_gb = 0
        self.alerts = []
        self.should_abort = False
        
    def check(self, context=""):
        """检查资源使用，超限则报警"""
        mem_info = self.process.memory_info()
        mem_gb = mem_info.rss / 1024**3
        
        # CPU（采样）
        cpu_pct = self.process.cpu_percent(interval=0.1)
        
        self.peak_memory_gb = max(self.peak_memory_gb, mem_gb)
        
        alert = {
            'time': datetime.now().isoformat(),
            'context': context,
            'mem_gb': round(mem_gb, 2),
            'peak_mem_gb': round(self.peak_memory_gb, 2),
            'cpu_pct': round(cpu_pct, 1)
        }
        self.alerts.append(alert)
        
        # 系统可用内存
        sys_mem = psutil.virtual_memory()
        sys_available_gb = sys_mem.available / 1024**3
        
        print(f"  [{alert['time'][11:19]}] MEM={mem_gb:.1f}GB(峰值{self.peak_memory_gb:.1f}GB) | CPU={cpu_pct:.0f}% | 系统空闲={sys_available_gb:.1f}GB | {context}")
        
        # 终止条件
        if mem_gb > MEMORY_LIMIT_GB:
            print(f"  🚨【中断】内存超过 {MEMORY_LIMIT_GB}GB 限制！")
            self.should_abort = True
            return False
            
        return True

    def report(self):
        print(f"\n{'='*60}")
        print(f"📊 资源使用报告")
        print(f"{'='*60}")
        print(f"  运行时长: {datetime.now() - self.start_time}")
        print(f"  峰值内存: {self.peak_memory_gb:.2f} GB")
        print(f"  触发告警数: {len(self.alerts)}")
        if self.alerts:
            # 只显示最后5条
            print("  最近告警:")
            for a in self.alerts[-5:]:
                print(f"    {a['time'][11:19]} MEM={a['mem_gb']:.1f}GB {a['context']}")
        return self.peak_memory_gb


# ============ 因子定义 ============
def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """基于日线+技术指标数据计算所有候选因子"""
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 未来N日收益率（label）
    for n in [5, 10, 20]:
        df[f'return_{n}d'] = df.groupby('symbol')['close'].pct_change(n).shift(-n)
    
    # ===== 因子1: 均线多头层数（0~5层） =====
    ma_cols = ['sma_5','sma_10','sma_20','sma_55','sma_240']
    df['ma_layers'] = 0
    for i, col in enumerate(ma_cols[:-1]):
        df['ma_layers'] += (df[col] > df[col].shift(1)).astype(int)
    
    # ===== 因子2: MACD 金叉/死叉 =====
    df['macd_cross'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & 
           (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_cross'] = 1   # 金叉
    df.loc[(df['macd'] < df['macd_signal']) & 
           (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_cross'] = -1  # 死叉
    
    # ===== 因子3: RSI 区间 =====
    df['rsi_bucket'] = pd.cut(df['rsi_14'], 
                               bins=[0, 30, 50, 70, 100], 
                               labels=['oversold', 'weak', 'strong', 'overbought'])
    
    # ===== 因子4: BOLL 位置（0=下轨, 0.5=中轨, 1=上轨） =====
    df['boll_pos'] = (df['sma_5'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['boll_pos'] = df['boll_pos'].clip(0, 1)
    
    # ===== 因子5: 量比 =====
    df['volume_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    
    # ===== 因子6: CCI 超买超卖 =====
    df['cci_signal'] = 0
    df.loc[df['cci_20'] < -100, 'cci_signal'] = 1   # 超卖
    df.loc[df['cci_20'] > 100, 'cci_signal'] = -1   # 超买
    
    # ===== 因子7: Williams%R 信号 =====
    df['wr_signal'] = 0
    df.loc[df['williams_r'] < -80, 'wr_signal'] = 1   # 超卖
    df.loc[df['williams_r'] > -20, 'wr_signal'] = -1  # 超买
    
    # ===== 因子8: KDJ 金叉/死叉 =====
    df['kdj_cross'] = 0
    df.loc[(df['kdj_k'] > df['kdj_d']) & 
           (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1)), 'kdj_cross'] = 1
    df.loc[(df['kdj_k'] < df['kdj_d']) & 
           (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1)), 'kdj_cross'] = -1
    
    # ===== 因子9: 涨幅 =====
    df['price_change_pct'] = df['change_pct']
    
    # ===== 因子10: 换手率 =====
    df['turnover_bucket'] = pd.cut(df['turnover_rate'], 
                                    bins=[-np.inf, 3, 7, 15, np.inf],
                                    labels=['极低','中等','较高','极高'])
    
    return df


def compute_ic(df: pd.DataFrame, factor_col: str, label_col: str = 'return_5d') -> dict:
    """计算单个因子的 IC（信息系数）"""
    valid = df[[factor_col, label_col]].dropna()
    if len(valid) < 100:
        return None
    
    # Spearman 相关系数（秩相关，更稳健）
    from scipy.stats import spearmanr
    corr, pval = spearmanr(valid[factor_col], valid[label_col], nan_policy='omit')
    
    return {'ic': corr, 'pval': pval, 'n': len(valid)}


def compute_yearly_ic(factor_col: str, label_col: str = 'return_5d') -> dict:
    """年度滚动计算某因子的 IC/IR"""
    yearly_results = []
    
    for year in YEAR_RANGE:
        tech_path = f'{DATA_DIR}/technical_indicators_year={year}/data.parquet'
        daily_path = f'{DATA_DIR}/daily_data_year={year}/data.parquet'
        
        if not os.path.exists(tech_path) or not os.path.exists(daily_path):
            print(f"  ⚠️ {year} 年数据不存在，跳过")
            continue
        
        # 加载数据
        tech = pd.read_parquet(tech_path)
        daily = pd.read_parquet(daily_path)
        
        # 合并
        df = pd.merge(tech, daily, on=['date', 'symbol'], how='inner')
        del tech, daily
        
        # 计算因子
        df = compute_factors(df)
        
        # 计算 IC
        ic_result = compute_ic(df, factor_col, label_col)
        if ic_result:
            yearly_results.append({
                'year': year,
                'ic': ic_result['ic'],
                'pval': ic_result['pval'],
                'n': ic_result['n']
            })
        
        del df
        import gc; gc.collect()
    
    if not yearly_results:
        return None
    
    results_df = pd.DataFrame(yearly_results)
    
    # 年化 ICIR = IC均值 / IC标准差
    ic_mean = results_df['ic'].mean()
    ic_std = results_df['ic'].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    
    return {
        'factor': factor_col,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_positive_ratio': (results_df['ic'] > 0).mean(),
        'yearly_count': len(results_df),
        'yearly_detail': results_df.to_dict('records')
    }


def run_full_analysis():
    """运行完整因子分析"""
    monitor = ResourceMonitor()
    monitor.start_time = datetime.now()
    
    print(f"{'='*60}")
    print(f"📈 因子 IC/IR 分析 - Score策略候选因子")
    print(f"{'='*60}")
    print(f"分析区间: {min(YEAR_RANGE)}-{max(YEAR_RANGE)}")
    print(f"数据目录: {DATA_DIR}")
    print(f"内存限制: {MEMORY_LIMIT_GB}GB")
    print()
    
    monitor.check("启动")
    
    # 候选因子列表
    factors = [
        ('ma_layers',         '均线多头层数(0~5)'),
        ('macd_cross',        'MACD金叉(+1)/死叉(-1)'),
        ('rsi_14',            'RSI绝对值'),
        ('boll_pos',           'BOLL位置(0~1)'),
        ('volume_ratio',       '量比'),
        ('cci_signal',        'CCI信号'),
        ('wr_signal',         'Williams%R信号'),
        ('kdj_cross',         'KDJ金叉(+1)/死叉(-1)'),
        ('price_change_pct',  '当日涨跌幅%'),
        ('turnover_rate',     '换手率%'),
        ('kdj_k',             'KDJ_K绝对值'),
        ('cci_20',            'CCI绝对值'),
    ]
    
    results = []
    
    for factor_col, factor_name in factors:
        print(f"\n🔍 分析因子: {factor_name} ({factor_col})")
        
        # 检查资源
        if not monitor.check(f"因子:{factor_col}"):
            print("  ⚠️ 资源超限，停止分析")
            break
        
        result = compute_yearly_ic(factor_col, label_col='return_5d')
        
        if result:
            result['factor_name'] = factor_name
            results.append(result)
            print(f"  ✅ IC均值={result['ic_mean']:.4f} | IC标准差={result['ic_std']:.4f} | "
                  f"ICIR={result['ic_ir']:.2f} | 正IC率={result['ic_positive_ratio']:.0%} | "
                  f"年={result['yearly_count']}年")
        else:
            print(f"  ❌ 数据不足，无法计算")
        
        # 每完成一个因子做一次GC
        import gc; gc.collect()
        
        # 再检查一次
        if not monitor.check(f"完成:{factor_col}"):
            break
    
    # 输出报告
    print(f"\n{monitor.report()}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ic_ir', ascending=False)
        
        print(f"\n{'='*60}")
        print(f"🏆 因子 IC/IR 排行榜（按 ICIR 降序）")
        print(f"{'='*60}")
        print(f"{'因子':<25} {'IC均值':>8} {'IC标准差':>8} {'ICIR':>6} {'正IC率':>8} {'年数':>5}")
        print("-" * 65)
        for _, r in results_df.iterrows():
            print(f"{r['factor_name']:<22} {r['ic_mean']:>8.4f} {r['ic_std']:>8.4f} "
                  f"{r['ic_ir']:>6.2f} {r['ic_positive_ratio']:>8.0%} {r['yearly_count']:>5}")
        
        # 保存结果
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        out_file = f'{OUT_DIR}/ic_ir_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 结果已保存: {out_file}")
        
        return results
    else:
        print("❌ 没有成功计算出任何因子IC")
        return None


if __name__ == '__main__':
    result = run_full_analysis()
