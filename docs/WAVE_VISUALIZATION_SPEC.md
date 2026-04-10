# 波浪图画图规范

**最后更新**: 2026-04-10

## 技术栈

- **库**: mplfinance
- **脚本**: `/root/.openclaw/workspace/projects/stock-analysis-system/scripts/wave_mplfinance_v6.py`

## 画图规则

### 1. K线底层
- 使用 mplfinance 默认蜡烛图
- 颜色：中国习惯（红涨绿跌）

### 2. 波浪线叠加
- 使用 `mpf.make_addplot()` 叠加波浪连接线
- **线性插值**：每段波浪从起止价格插值填满整个交易日区间，避免断开
- **颜色**：
  - 上涨推动浪 = 红色
  - 下跌调整浪 = 绿色
- 连接方式：高→低→高→低交替

### 3. 极值点标记
- 圆形标记（双层 scatter：外圈有色 + 内圈白色）
- 颜色与接下来波浪线颜色对应

### 4. 背景
- 深色背景 (`#1a1a2e`)
- 文字/标签颜色适配深色背景

## 输出格式

- 分辨率：约 2700×1800
- 格式：PNG

## 调用示例

```bash
python3 /root/.openclaw/workspace/projects/stock-analysis-system/scripts/wave_mplfinance_v6.py --symbol 000001 --year 2020
```

## 相关文件

| 文件 | 描述 |
|------|------|
| `scripts/wave_mplfinance_v6.py` | 画图脚本（最新版本） |
| `scripts/wave_mplfinance.py` | 旧版本（有问题） |
| `output/wave_candlestick_*.png` | matplotlib 版本输出 |