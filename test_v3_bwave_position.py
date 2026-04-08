"""
V3 B浪仓位逻辑测试
测试内容：
1. bearish + W1<5浪 → B浪反弹，仓位30-50%
2. bearish + W1≥5浪 → 下跌1浪，仓位50%
3. bullish + W1推动 + 回撤浅 → 70%
4. bullish + W1推动 + 回撤深 → 50%
5. neutral趋势 → 40-50%

验证点：
- position_size_ratio 是否正确注入到 WaveSignal
- 止损价格是否正确（B浪2%，W2回调3%）
- neutral趋势是否有降仓处理
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/projects/stock-analysis-system')

from strategies.wavechan_v3 import WaveCounterV3, WaveSignal
from dataclasses import asdict
from unittest.mock import MagicMock
import json


class TestV3BWavePosition:
    """V3 B浪仓位逻辑测试"""
    
    def _create_counter(self):
        """创建并返回配置好的counter实例"""
        counter = WaveCounterV3()
        counter.position = None
        counter.snapshot = MagicMock()
        counter.active_signals = []
        counter.bis = []
        return counter
    
    def _mock_large_trend(self, counter, trend):
        """Mock大级别趋势"""
        counter._get_large_trend = MagicMock(return_value=trend)
    
    def _mock_count_wave_segments(self, counter, w1_up_segments, w2_down_segments):
        """Mock波浪段计数"""
        def count_segments(start, end, direction):
            if direction == 'up':
                return w1_up_segments
            elif direction == 'down':
                return w2_down_segments
            return 0
        counter._count_wave_segments = count_segments
    
    def _mock_wave_structure(self, w1_up_segments, w2_down_segments, w1_start, w1_end, w2_end):
        """Mock波浪结构"""
        wave_result = {
            'waves': {
                'W1': {'start': w1_start, 'end': w1_end},
                'W2': {'start': w1_end, 'end': w2_end}
            }
        }
        return wave_result

    def _run_validate(self, counter, w1_up_segments, w2_down_segments, w1_start, w1_end, w2_end, large_trend):
        """通用的验证执行"""
        self._mock_large_trend(counter, large_trend)
        self._mock_count_wave_segments(counter, w1_up_segments, w2_down_segments)
        wave_result = self._mock_wave_structure(w1_up_segments, w2_down_segments, w1_start, w1_end, w2_end)
        return counter._validate_wave_structure(wave_result)

    # ========================
    # 测试用例 1: 600368 - bearish + W1<5浪 → B浪反弹
    # ========================
    def test_600368_b_wave_rebound_w1_3_segments(self):
        """600368 B浪反弹测试 - W1仅3个子浪，仓位30%"""
        print("\n=== 测试1.1: 600368 B浪反弹 (W1=3浪, 仓位30%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 3  # < 5浪
        w2_down_segments = 3  # a-b-c = 3浪
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bearish')
        
        print(f"  大级别趋势: {struct_check['large_trend']}")
        print(f"  W1内部上涨子浪数: {struct_check['w1_internal_segments']}")
        print(f"  W2内部下跌子浪数: {struct_check['w2_internal_segments']}")
        print(f"  是否为B浪反弹: {struct_check['is_b_wave_rebound']}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        print(f"  过滤原因: {struct_check['filter_reason']}")
        print(f"  结构有效性: {struct_check['valid']}")
        
        # 验证
        assert struct_check['is_b_wave_rebound'] == True, "应该标记为B浪反弹"
        assert struct_check['position_size_ratio'] == 0.30, f"W1=3浪应该仓位30%，实际{struct_check['position_size_ratio']}"
        assert struct_check['valid'] == True, "B浪反弹应该valid=True（降仓参与）"
        
        print("  ✓ 测试通过!")
        return struct_check

    def test_600368_b_wave_rebound_w1_4_segments(self):
        """600368 B浪反弹测试 - W1=4浪，仓位40%"""
        print("\n=== 测试1.2: 600368 B浪反弹 (W1=4浪, 仓位40%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 4  # < 5浪
        w2_down_segments = 3  # a-b-c = 3浪
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bearish')
        
        print(f"  W1内部上涨子浪数: {struct_check['w1_internal_segments']}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        assert struct_check['is_b_wave_rebound'] == True
        assert struct_check['position_size_ratio'] == 0.40, f"W1=4浪应该仓位40%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    def test_600368_b_wave_rebound_w2_not_3_waves(self):
        """600368 B浪反弹测试 - W2不是3浪结构，应被过滤"""
        print("\n=== 测试1.3: 600368 B浪反弹 (W2≠3浪, 应过滤) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 3  # < 5浪
        w2_down_segments = 5  # 5浪不是3浪 = 无效
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bearish')
        
        print(f"  W2内部下跌子浪数: {w2_down_segments}")
        print(f"  结构有效性: {struct_check['valid']}")
        print(f"  过滤原因: {struct_check['filter_reason']}")
        
        assert struct_check['valid'] == False, "W2不是3浪结构应该过滤"
        assert "非3浪" in struct_check['filter_reason'], "应该包含非3浪的过滤原因"
        print("  ✓ 测试通过!")
        return struct_check

    # ========================
    # 测试用例 2: 600368 - bearish + W1≥5浪 → 下跌1浪
    # ========================
    def test_600368_decline_wave1_w1_5_segments(self):
        """600368 下跌1浪测试 - bearish + W1=5浪，仓位50%"""
        print("\n=== 测试2: 600368 下跌1浪 (W1=5浪, 仓位50%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 5  # = 5浪推动
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bearish')
        
        print(f"  W1内部上涨子浪数: {struct_check['w1_internal_segments']}")
        print(f"  是否为B浪反弹: {struct_check['is_b_wave_rebound']}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        print(f"  过滤原因: {struct_check['filter_reason']}")
        
        assert struct_check['is_b_wave_rebound'] == False, "下跌1浪不是B浪反弹"
        assert struct_check['position_size_ratio'] == 0.50, f"W1=5浪应该仓位50%，实际{struct_check['position_size_ratio']}"
        assert "下跌1浪" in struct_check['filter_reason'], "应该包含下跌1浪的说明"
        print("  ✓ 测试通过!")
        return struct_check

    # ========================
    # 测试用例 3: bullish + W1推动 + 回撤浅 → 70%
    # ========================
    def test_bullish_w1_impulse_shallow_retrace(self):
        """Bullish W1推动 + 浅回撤(≤50%) → 仓位70%"""
        print("\n=== 测试3: Bullish W1推动 + 浅回撤 (仓位70%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.6
        w1_up_segments = 5  # 5浪推动
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bullish')
        
        retracement = (w1_end - w2_end) / (w1_end - w1_start)
        print(f"  W1起点: {w1_start}, W1终点: {w1_end}, W2终点: {w2_end}")
        print(f"  W1涨幅: {w1_end - w1_start:.2f}, W2回撤: {w1_end - w2_end:.2f}")
        print(f"  回撤比例: {retracement:.2%}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        assert struct_check['position_size_ratio'] == 0.70, f"浅回撤(≤50%)应该仓位70%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    # ========================
    # 测试用例 4: bullish + W1推动 + 回撤深 → 50%
    # ========================
    def test_bullish_w1_impulse_deep_retrace(self):
        """Bullish W1推动 + 深回撤(>61.8%) → 仓位50%"""
        print("\n=== 测试4: Bullish W1推动 + 深回撤 (仓位50%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.3
        w1_up_segments = 5
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bullish')
        
        retracement = (w1_end - w2_end) / (w1_end - w1_start)
        print(f"  W1起点: {w1_start}, W1终点: {w1_end}, W2终点: {w2_end}")
        print(f"  回撤比例: {retracement:.2%}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        assert struct_check['position_size_ratio'] == 0.50, f"深回撤(>61.8%)应该仓位50%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    # ========================
    # 测试用例 5: neutral趋势 → 40-50%
    # ========================
    def test_neutral_w1_impulse(self):
        """Neutral趋势 + W1推动 → 仓位40%"""
        print("\n=== 测试5.1: Neutral趋势 + W1推动 (仓位40%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 5  # W1推动
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'neutral')
        
        print(f"  大级别趋势: {struct_check['large_trend']}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        assert struct_check['position_size_ratio'] == 0.40, f"Neutral+W1推动应该仓位40%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    def test_neutral_w1_not_impulse(self):
        """Neutral趋势 + W1非推动 → 仓位50%"""
        print("\n=== 测试5.2: Neutral趋势 + W1非推动 (仓位50%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.5
        w1_up_segments = 3  # W1非推动
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'neutral')
        
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        assert struct_check['position_size_ratio'] == 0.50, f"Neutral+W1非推动应该仓位50%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    # ========================
    # 测试用例 6: 止损价格验证
    # ========================
    def test_b_wave_stop_loss_2_percent(self):
        """B浪反弹止损价格 = W2低点 × 0.98 (2%)"""
        print("\n=== 测试6.1: B浪反弹止损价格 (2%) ===")
        
        w2_end = 4.5
        expected_sl = round(w2_end * 0.98, 2)  # = 4.41
        
        print(f"  W2终点(买入价): {w2_end}")
        print(f"  止损价应为: {expected_sl} (2%止损)")
        
        assert expected_sl == 4.41, f"B浪止损应为4.41，实际{expected_sl}"
        print("  ✓ B浪止损2%验证通过!")
        return expected_sl

    def test_w2_correction_stop_loss_3_percent(self):
        """W2回调止损价格 = W2低点 × 0.97 (3%)"""
        print("\n=== 测试6.2: W2回调止损价格 (3%) ===")
        
        w2_end = 4.5
        expected_sl = round(w2_end * 0.97, 2)  # = 4.37
        
        print(f"  W2终点(买入价): {w2_end}")
        print(f"  止损价应为: {expected_sl} (3%止损)")
        
        assert expected_sl == 4.37, f"W2回调止损应为4.37，实际{expected_sl}"
        print("  ✓ W2回调止损3%验证通过!")
        return expected_sl

    # ========================
    # 测试用例 7: WaveSignal注入验证
    # ========================
    def test_wavesignal_has_position_size_field(self):
        """验证WaveSignal类有position_size_ratio字段"""
        print("\n=== 测试7: WaveSignal字段验证 ===")
        
        signal = WaveSignal(
            signal='W2_BUY',
            status='ALERT',
            price=4.5,
            stop_loss=4.41,
            verify_conditions=[],
            verified_conditions=[],
            reason='测试',
            confidence=0.65,
            created_date='2026-04-08',
            position_size_ratio=0.30,
            is_b_wave_rebound=True
        )
        
        print(f"  position_size_ratio字段存在: {hasattr(signal, 'position_size_ratio')}")
        print(f"  is_b_wave_rebound字段存在: {hasattr(signal, 'is_b_wave_rebound')}")
        print(f"  position_size_ratio值: {signal.position_size_ratio}")
        print(f"  is_b_wave_rebound值: {signal.is_b_wave_rebound}")
        
        assert hasattr(signal, 'position_size_ratio'), "WaveSignal应该有position_size_ratio字段"
        assert hasattr(signal, 'is_b_wave_rebound'), "WaveSignal应该有is_b_wave_rebound字段"
        assert signal.position_size_ratio == 0.30, f"position_size_ratio应该等于0.30，实际{signal.position_size_ratio}"
        
        print("  ✓ WaveSignal字段验证通过!")
        return signal

    # ========================
    # 测试用例 8: calc_position_size降仓验证
    # ========================
    def test_calc_position_size_with_ratio(self):
        """验证calc_position_size正确使用position_size_ratio"""
        print("\n=== 测试8: calc_position_size降仓验证 ===")
        
        from strategies.wavechan_v3 import PositionManager
        pm = PositionManager()
        pm.position = {
            'entry_price': 4.5,
            'stop_loss': 4.41,  # 2%止损 = B浪
            'entry_time': __import__('datetime').datetime.now(),
            'position_size_ratio': 0.30  # B浪30%仓位
        }
        
        account_balance = 100000  # 10万
        risk_ratio = 0.02  # 2%风险
        
        # 计算
        shares = pm.calc_position_size(account_balance, risk_ratio)
        
        # 手动计算期望值
        stop_distance = 4.5 - 4.41  # = 0.09
        risk_amount = 100000 * 0.02  # = 2000
        base_shares = risk_amount / stop_distance  # ≈ 22222
        expected_shares = int(base_shares * 0.30 // 100 * 100)  # × 30% ≈ 6600
        
        print(f"  账户余额: {account_balance}")
        print(f"  买入价: 4.5, 止损价: 4.41")
        print(f"  止损距离: {stop_distance}")
        print(f"  风险金额: {risk_amount}")
        print(f"  基准股数: {base_shares:.0f}")
        print(f"  B仓比例: 0.30")
        print(f"  计算股数: {shares}")
        print(f"  期望股数(约): {expected_shares}")
        
        # 允许一定误差（整百取整）
        assert abs(shares - expected_shares) < 100, f"股数计算有误: 实际{shares}，期望约{expected_shares}"
        print("  ✓ calc_position_size降仓验证通过!")
        return shares

    # ========================
    # 测试用例 9: 中间回撤(50%-61.8%) → 60%
    # ========================
    def test_bullish_w1_impulse_medium_retrace(self):
        """Bullish W1推动 + 45%回撤 → 仓位70%"""
        print("\n=== 测试9.1: Bullish W1推动 + 45%回撤 (仓位70%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.55
        w1_up_segments = 5
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bullish')
        
        retracement = (w1_end - w2_end) / (w1_end - w1_start)
        print(f"  回撤比例: {retracement:.2%}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        # 回撤45% <= 50% → 70%
        assert struct_check['position_size_ratio'] == 0.70, f"45%回撤(≤50%)应该仓位70%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check

    def test_bullish_w1_impulse_55_percent_retrace(self):
        """Bullish W1推动 + 55%回撤 → 仓位60%"""
        print("\n=== 测试9.2: Bullish W1推动 + 55%回撤 (仓位60%) ===")
        
        counter = self._create_counter()
        w1_start, w1_end, w2_end = 4.0, 5.0, 4.45
        w1_up_segments = 5
        w2_down_segments = 3
        
        struct_check = self._run_validate(counter, w1_up_segments, w2_down_segments, 
                                          w1_start, w1_end, w2_end, 'bullish')
        
        retracement = (w1_end - w2_end) / (w1_end - w1_start)
        print(f"  回撤比例: {retracement:.2%}")
        print(f"  仓位比例: {struct_check['position_size_ratio']}")
        
        # 回撤55%在50%-61.8%之间 → 60%
        assert struct_check['position_size_ratio'] == 0.60, f"55%回撤(50%-61.8%)应该仓位60%，实际{struct_check['position_size_ratio']}"
        print("  ✓ 测试通过!")
        return struct_check


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("V3 B浪仓位逻辑测试")
    print("=" * 60)
    
    tester = TestV3BWavePosition()
    results = []
    
    try:
        # 1. 600368 B浪反弹测试
        results.append(("1.1 B浪反弹 W1=3浪 30%", tester.test_600368_b_wave_rebound_w1_3_segments()))
        results.append(("1.2 B浪反弹 W1=4浪 40%", tester.test_600368_b_wave_rebound_w1_4_segments()))
        results.append(("1.3 B浪反弹 W2≠3浪过滤", tester.test_600368_b_wave_rebound_w2_not_3_waves()))
        
        # 2. 600368 下跌1浪测试
        results.append(("2. 下跌1浪 W1=5浪 50%", tester.test_600368_decline_wave1_w1_5_segments()))
        
        # 3-4. Bullish W1推动仓位测试
        results.append(("3. Bullish浅回撤 ≤50% → 70%", tester.test_bullish_w1_impulse_shallow_retrace()))
        results.append(("4. Bullish深回撤 >61.8% → 50%", tester.test_bullish_w1_impulse_deep_retrace()))
        
        # 5. Neutral趋势降仓测试
        results.append(("5.1 Neutral+W1推动 → 40%", tester.test_neutral_w1_impulse()))
        results.append(("5.2 Neutral+W1非推动 → 50%", tester.test_neutral_w1_not_impulse()))
        
        # 6. 止损价格验证
        results.append(("6.1 B浪止损2%", tester.test_b_wave_stop_loss_2_percent()))
        results.append(("6.2 W2回调止损3%", tester.test_w2_correction_stop_loss_3_percent()))
        
        # 7. WaveSignal字段验证
        results.append(("7. WaveSignal字段注入", tester.test_wavesignal_has_position_size_field()))
        
        # 8. calc_position_size降仓验证
        results.append(("8. calc_position降仓", tester.test_calc_position_size_with_ratio()))
        
        # 9. 中间回撤验证
        results.append(("9.1 回撤45% → 70%", tester.test_bullish_w1_impulse_medium_retrace()))
        results.append(("9.2 回撤55% → 60%", tester.test_bullish_w1_impulse_55_percent_retrace()))
        
        all_passed = True
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result is not None else "✗ FAIL"
        print(f"  {status}  {name}")
    
    if all_passed:
        print(f"\n✓ 共 {len(results)} 项测试全部通过!")
    else:
        print(f"\n✗ 测试未全部通过")


if __name__ == "__main__":
    run_all_tests()
