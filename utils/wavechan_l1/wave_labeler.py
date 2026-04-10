"""
WaveChan L1 - 大浪标注模块 v4
跨年大浪标注：使用多年 BI 数据确保浪编号连续

【重构 v4 - 标准艾略特波浪命名】

波浪序列（循环往复）：
  推动浪：1 → 2 → 3 → 4 → 5
  调整浪：a → b → c
  然后重复：1 → 2 → 3 → 4 → 5 → a → b → c → ...

规则：
  - 1/3/5 = 推动浪，与主趋势同向（consecutive same-type major points）
  - 2/4   = 调整浪，与主趋势反向（alternating high/low）
  - a/b/c = 更大级别的调整浪（接在 5 之后）
  - W2 不能跌破 W1 起点（牛市）/ W2 不能突破 W1 起点（熊市）
  - W4 不能进入 W1 价格区间

Major Point 定义：
  - 创新高的笔终点（end_price > 历史所有笔终点价格最大值）→ major_high
  - 创新低的笔终点（end_price < 历史所有笔终点价格最小值）→ major_low
  - 两者互斥
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from ._path import weekly_bi_path, wave_labels_path, ensure_dirs

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 波浪标注引擎
# ------------------------------------------------------------------

# 标准波浪标签顺序
_WAVE_SEQUENCE = ["1", "2", "3", "4", "5", "a", "b", "c"]
_WAVE_SEQUENCE_REPEAT = _WAVE_SEQUENCE + _WAVE_SEQUENCE + _WAVE_SEQUENCE


def _get_wave_info(wave_label: str) -> Tuple[str, str, int]:
    """
    解析波浪标签，返回 (wave_role, trend, wave_order)。
    
    wave_role: 'impulse' | 'correction' | 'cycle_correction'
    trend:     'up' | 'down' | 'unknown'
    wave_order: 在波浪序列中的位置（0-7）
    """
    label = wave_label.rstrip("_forming").rstrip("*")
    try:
        idx = _WAVE_SEQUENCE.index(label)
    except ValueError:
        return ("unknown", "unknown", -1)
    
    if label in ("1", "3", "5"):
        return ("impulse", "unknown", idx)
    elif label in ("2", "4"):
        return ("correction", "unknown", idx)
    elif label in ("a", "b", "c"):
        return ("cycle_correction", "unknown", idx)
    return ("unknown", "unknown", idx)


class WaveLabelEngine:
    """
    标准艾略特波浪标注引擎。
    
    规则：
      - 相邻两个 major point 组成一个浪段
      - consecutive same-type（high→high 或 low→low）→ 推动浪（1/3/5）
      - alternating（high→low 或 low→high）→ 调整浪（2/4）
      - 5 浪完成后进入 a-b-c 调整，然后循环
      - 标注挂在浪段终点（segment ending point）
    """
    
    def __init__(self):
        # 波浪计数器：在 _WAVE_SEQUENCE 中的位置
        self.pos: int = 0
        # 当前主趋势方向（由第一个 impulse wave 决定）
        self.trend_direction: str = "unknown"
        # 完成的 5 浪数量（用于判断牛市/熊市）
        self.impulse_count: int = 0
        # 是否已完成第一个 5 浪
        self.cycle_started: bool = False
        
    def next_wave_label(self, prev_type: str, curr_type: str,
                        prev_price: float, curr_price: float) -> Tuple[str, str, str]:
        """
        给定两个相邻 major point，返回浪标签。
        
        返回 (wave_label, wave_role, trend_type)
          - wave_label: 标准波浪编号 1/2/3/4/5/a/b/c
          - wave_role: 'impulse' | 'correction' | 'cycle_correction'
          - trend_type: 'uptrend' | 'downtrend' | 'oscillation'
        """
        same_type = (prev_type == curr_type)
        
        # ---- 判断是 impulse 还是 correction ----
        if same_type:
            wave_role = "impulse"
            # 同向：价格抬升 → uptrend，价格降低 → downtrend
            if curr_price > prev_price:
                trend_type = "uptrend"
            elif curr_price < prev_price:
                trend_type = "downtrend"
            else:
                trend_type = "oscillation"
            
            # 第一个 impulse wave 决定主趋势
            if self.trend_direction == "unknown":
                self.trend_direction = trend_type
                self.cycle_started = True
                self.impulse_count = 1
        else:
            wave_role = "correction"
            trend_type = "oscillation"
        
        # ---- 分配波浪编号 ----
        # 在 cycle_correction 阶段（a/b/c）后，循环回到 1
        if self.pos >= len(_WAVE_SEQUENCE):
            self.pos = 0
            self.impulse_count = 0
            self.cycle_started = False
        
        wave_label = _WAVE_SEQUENCE[self.pos]
        
        # 推动浪（1/3/5）推进 impulse 计数
        if wave_label in ("1", "3", "5"):
            if wave_label == "5":
                self.impulse_count += 1
        
        # 调整浪（2/4）不改变 impulse 计数
        # a-b-c 完成后：下一个自动回到 1（循环）
        self.pos += 1
        
        return wave_label, wave_role, trend_type
    
    def label_for_point(self, idx: int, n: int,
                         major: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        为所有 major points 分配浪标签。
        
        返回 (wave_labels, trend_types, wave_roles) 三个列表，
        每个列表长度 = n（每个 major point 一个标签）。
        
        标签挂在浪段终点（第 i 个 point = 第 i-1 个浪段的终点）。
        """
        if n < 2:
            return ["1_forming"], ["unknown"], ["unknown"]
        
        wave_labels = [""] * n
        trend_types = ["unknown"] * n
        wave_roles = ["unknown"] * n
        
        # 第一个 point：浪段起点，标签由下一个 point 决定
        # 我们从 i=1 开始处理（浪段 0→1，标签挂在 point 1）
        # 最后 point n-1：未完成浪
        pending_label = None
        
        for i in range(1, n):
            prev = major.iloc[i - 1]
            curr = major.iloc[i]
            
            wave_label, wave_role, trend_type = self.next_wave_label(
                prev["point_type"], curr["point_type"],
                prev["price"], curr["price"],
            )
            
            # 标签挂在当前 point（浪段终点）
            wave_labels[i] = wave_label
            trend_types[i] = trend_type
            wave_roles[i] = wave_role
        
        # 处理最后一个 point（未完成浪）
        if wave_labels[n - 1] == "":
            next_label = _WAVE_SEQUENCE[self.pos] if self.pos < len(_WAVE_SEQUENCE) else "1"
            wave_labels[n - 1] = f"{next_label}_forming"
            trend_types[n - 1] = "unknown"
            wave_roles[n - 1] = "unknown"
        
        return wave_labels, trend_types, wave_roles


def label_symbol_waves(bi_df: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的所有 BI 数据标注标准艾略特波浪（1-2-3-4-5-a-b-c）。

    算法：
      1. 提取所有 major points（is_major_high 或 is_major_low）
      2. 相邻两个 major point 组成一个浪段
      3. consecutive same-type → 推动浪（1/3/5）
         alternating → 调整浪（2/4）
         5 浪完成后进入 a-b-c
      4. 标签挂在浪段终点
      5. 应用波浪约束验证

    返回：
      DataFrame，列：symbol, date, price, bi_index, point_type,
                   wave_label, trend_type, wave_role,
                   is_warning, warning_msg
    """
    if bi_df.empty:
        return pd.DataFrame()

    df = bi_df.sort_values("end_date").reset_index(drop=True)

    # ---- Step 1: 提取 major points ----
    major = df[(df["is_major_high"]) | (df["is_major_low"])].copy()
    if major.empty:
        return pd.DataFrame()

    major = major.rename(columns={"end_date": "date", "end_price": "price"})
    major = major.reset_index(drop=True)
    major["point_type"] = major["is_major_high"].apply(lambda x: "high" if x else "low")

    n = len(major)
    if n < 2:
        result = major[["symbol", "date", "price", "bi_index", "point_type"]].copy()
        result["wave_label"] = "1_forming"
        result["trend_type"] = "unknown"
        result["wave_role"] = "unknown"
        result["is_warning"] = False
        result["warning_msg"] = ""
        return result.reset_index(drop=True)

    # ---- Step 2: 标注 ----
    engine = WaveLabelEngine()
    wave_labels, trend_types, wave_roles = engine.label_for_point(0, n, major)

    # ---- Step 3: 波浪约束验证 ----
    warnings_list = _check_wave_constraints(major, wave_labels, trend_types, wave_roles)

    # ---- Step 4: 组装结果 ----
    result = major[["symbol", "date", "price", "bi_index", "point_type"]].copy()
    result["wave_label"] = wave_labels
    result["trend_type"] = trend_types
    result["wave_role"] = wave_roles
    result["is_warning"] = [w[0] for w in warnings_list]
    result["warning_msg"] = [w[1] for w in warnings_list]

    return result.reset_index(drop=True)


def _check_wave_constraints(
    major: pd.DataFrame,
    wave_labels: List[str],
    trend_types: List[str],
    wave_roles: List[str],
) -> List[Tuple[bool, str]]:
    """
    应用波浪理论约束，检测违规并发出警告。

    约束：
      1. W2 不能跌破 W1 起点（牛市）/ W2 不能突破 W1 起点（熊市）
         → 主趋势向上时，W2低点 > W1起点；主趋势向下时，W2高点 < W1起点
      2. W4 不能进入 W1 价格区间
         → 牛市：W4低点 < W1高点
         → 熊市：W4高点 > W1低点（不进入 W1 区间）
      3. W3 通常不是最短的推动浪（W1/W3/W5 中 W3 不应最短）
    """
    n = len(major)
    warnings_list: List[Tuple[bool, str]] = [(False, "")] * n

    if n < 3:
        return warnings_list

    # 构建 wave_label → index 的映射
    w_to_idx: dict = {}
    for i, label in enumerate(wave_labels):
        lbl = label.rstrip("_forming")
        if lbl in _WAVE_SEQUENCE:
            w_to_idx[lbl] = i

    # 判断主趋势（取第一个 impulse wave 的趋势）
    trend_direction = "unknown"
    for label, idx in w_to_idx.items():
        if label in ("1", "3", "5"):
            # 从 wave_role 和 trend_types 反推
            if idx < n - 1:
                next_idx = idx + 1
                if wave_roles[next_idx] == "impulse":
                    # 相邻同向 → impulse
                    pass
            if label == "1" and idx < n - 1:
                # wave 1 的趋势就是主趋势
                # prev 是 point idx-1, curr 是 point idx
                if idx > 0:
                    p_prev = major.iloc[idx - 1]
                    p_curr = major.iloc[idx]
                    if p_curr["price"] > p_prev["price"]:
                        trend_direction = "up"
                    elif p_curr["price"] < p_prev["price"]:
                        trend_direction = "down"
                break

    # ---- 约束 1: W2 vs W1 ----
    if "1" in w_to_idx and "2" in w_to_idx:
        w1_idx = w_to_idx["1"]
        w2_idx = w_to_idx["2"]
        
        # W1 的起点：第一个 wave 之前的 point（major[0] 或 wave 1 起点）
        # wave 1 的起点就是 major point w1_idx（浪段从 w1_idx-1 到 w1_idx）
        # 简化：W1 浪段的终点就是 wave_label="1" 的那个 major point
        # W1 浪的起点价格 = wave 1 段开始前那个 point 的 price
        if w1_idx > 0:
            w1_start_price = major.iloc[w1_idx - 1]["price"]
        else:
            w1_start_price = major.iloc[w1_idx]["price"]  # 第一段
        
        w2_point = major.iloc[w2_idx]
        w2_price = w2_point["price"]
        
        if trend_direction == "down":
            # 熊市：W2 是向上的调整浪，不应突破 W1 起点
            # W2 高点不应 > W1 起点
            if w2_point["point_type"] == "high" and w2_price > w1_start_price:
                msg = (f"[波浪约束] 熊市W2高点({w2_price:.2f}) > W1起点({w1_start_price:.2f})，"
                       f"W2突破W1起点")
                warnings_list[w2_idx] = (True, msg)
        else:
            # 牛市：W2 低点不应 < W1 起点
            if w2_point["point_type"] == "low" and w2_price < w1_start_price:
                msg = (f"[波浪约束] W2低点({w2_price:.2f}) < W1起点({w1_start_price:.2f})，"
                       f"违反W2不破W1起点规则")
                warnings_list[w2_idx] = (True, msg)

    # ---- 约束 2: W4 不能进入 W1 价格区间 ----
    if "1" in w_to_idx and "4" in w_to_idx:
        w1_idx = w_to_idx["1"]
        w4_idx = w_to_idx["4"]
        
        # W1 的高点：wave 1 段的高点
        # 简化：取 wave 1 终点（major point）和 wave 1 起点中较高的
        if w1_idx > 0:
            w1_start = major.iloc[w1_idx - 1]["price"]
            w1_end = major.iloc[w1_idx]["price"]
            w1_high = max(w1_start, w1_end)
        else:
            w1_high = major.iloc[w1_idx]["price"]
        
        w4_point = major.iloc[w4_idx]
        w4_price = w4_point["price"]
        
        if trend_direction == "down":
            # 熊市：W4 低点不应 > W1 高点（W4 不进入 W1 区间）
            if w4_point["point_type"] == "low" and w4_price > w1_high:
                msg = (f"[波浪约束] 熊市W4低点({w4_price:.2f}) > W1高点({w1_high:.2f})，"
                       f"W4进入W1区间")
                warnings_list[w4_idx] = (True, msg)
        else:
            # 牛市：W4 低点不应 < W1 高点
            if w4_point["point_type"] == "low" and w4_price < w1_high:
                msg = (f"[波浪约束] W4低点({w4_price:.2f}) < W1高点({w1_high:.2f})，"
                       f"W4进入W1区间")
                warnings_list[w4_idx] = (True, msg)

    # ---- 约束 3: W3 通常不是最短的推动浪 ----
    impulse_indices = {lbl: i for lbl, i in w_to_idx.items() if lbl in ("1", "3", "5")}
    if len(impulse_indices) >= 3:
        impulse_lengths = {}
        for w_label, idx in impulse_indices.items():
            if idx > 0:
                seg_start_price = major.iloc[idx - 1]["price"]
                seg_end_price = major.iloc[idx]["price"]
                impulse_lengths[w_label] = abs(seg_end_price - seg_start_price)
            else:
                impulse_lengths[w_label] = 0.0

        if all(v > 0 for v in impulse_lengths.values()):
            min_label = min(impulse_lengths, key=impulse_lengths.get)
            if min_label == "3":
                w3_idx = impulse_indices["3"]
                msg = "[波浪约束] W3是W1/W3/W5中最短的推动浪（通常W3不是最短）"
                warnings_list[w3_idx] = (True, msg)

    return warnings_list


# ------------------------------------------------------------------
# 文件读写接口
# ------------------------------------------------------------------

def _read_symbol_bi_all_years(sym: str, year: int, lookback_years: int) -> pd.DataFrame:
    """读取某只股票多年（含 lookback）的 BI 数据。"""
    from .bi_recognizer import list_symbols_for_year

    years_to_read = list(range(max(2018, year - lookback_years), year + 1))
    all_dfs = []

    for yr in years_to_read:
        p = weekly_bi_path(sym, yr)
        if p.exists():
            df = pd.read_parquet(p)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["bi_index"], keep="last")
    return combined.sort_values("end_date").reset_index(drop=True)


def label_year_waves(
    year: int,
    symbols: Optional[List[str]] = None,
    use_lookback: bool = True,
    lookback_years: int = 3,
) -> int:
    """
    标注某年所有股票的大浪（标准 1-2-3-4-5-a-b-c 命名）。
    """
    from .bi_recognizer import list_symbols_for_year
    ensure_dirs(year)

    if symbols is None:
        symbols = list_symbols_for_year(year)

    written = 0
    for sym in symbols:
        if use_lookback:
            bi_df = _read_symbol_bi_all_years(sym, year, lookback_years=lookback_years)
        else:
            bi_p = weekly_bi_path(sym, year)
            if not bi_p.exists():
                continue
            bi_df = pd.read_parquet(bi_p)

        if bi_df.empty:
            continue

        labels_df = label_symbol_waves(bi_df)
        if labels_df.empty:
            continue

        labels_year = labels_df[labels_df["date"].dt.year == year].copy()
        if labels_year.empty:
            continue

        out_path = wave_labels_path(sym, year)
        labels_year.to_parquet(out_path, index=False, engine="pyarrow")
        written += 1

    logger.info(f"标注 {year} 年大浪完成，写入 {written} 只股票")
    return written
