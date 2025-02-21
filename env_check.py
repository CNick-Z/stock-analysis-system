import sys
import platform

print(f"""
环境验证报告
=============
系统信息：
- 操作系统：{platform.system()} {platform.release()}
- Python版本：{sys.version}
- 处理器架构：{platform.machine()}
""")

dependencies = {
    "akshare": "1.12.1",
    "pandas": "2.0.3",
    "numpy": "1.24.4",
    "talib": "0.6.3",
    "matplotlib": "3.7.2"
}

print("\n核心依赖检查：")
for lib, expected in dependencies.items():
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', '未知版本')
        status = "✓" if version == expected else f"× 需{expected}，当前{version}"
        print(f"{lib.ljust(15)} {status}")
    except ImportError:
        print(f"{lib.ljust(15)} × 未安装")

print("\nTA-Lib功能测试：")
try:
    import talib
    from talib import abstract
    import numpy as np  # 导入 numpy
     # 将输入数据转换为 numpy.ndarray
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)  # 确保数据类型为 float64
    print("TA-Lib基本功能正常")
    print("MA计算示例：", talib.MA(data, timeperiod=2))
except Exception as e:
    print(f"TA-Lib异常：{str(e)}")

print("\nAKShare连接测试：")
try:
    import akshare as ak
    data = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20230101", end_date="20231022")
    print(data.head())
except Exception as e:
    print(f"数据获取失败：{e}")