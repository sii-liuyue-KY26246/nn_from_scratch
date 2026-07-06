"""初始化方法测试。

运行: pytest tests/test_initializers.py -v
"""
import numpy as np
from nn_from_scratch.initializers import zeros, random_normal, xavier, he, visualize_initializations


class TestZeros:
    def test_all_zero(self):
        W = zeros((3, 5))
        assert np.all(W == 0)
        assert W.shape == (3, 5)


class TestRandomNormal:
    def test_shape(self):
        W = random_normal((100, 50))
        assert W.shape == (100, 50)

    def test_scale(self):
        np.random.seed(0)
        W = random_normal((1000, 500), scale=0.01)
        np.testing.assert_allclose(W.std(), 0.01, atol=0.002)


class TestXavier:
    def test_variance(self):
        np.random.seed(0)
        fan_in, fan_out = 784, 256
        W = xavier((fan_in, fan_out))
        expected_std = np.sqrt(2.0 / (fan_in + fan_out))
        np.testing.assert_allclose(W.std(), expected_std, atol=0.01)


class TestHe:
    def test_variance(self):
        np.random.seed(0)
        fan_in, fan_out = 784, 256
        W = he((fan_in, fan_out))
        expected_std = np.sqrt(2.0 / fan_in)
        np.testing.assert_allclose(W.std(), expected_std, atol=0.01)


class TestVisualize:
    def test_returns_dict(self):
        results = visualize_initializations(in_features=100, out_features=50, n_layers=3)
        assert isinstance(results, dict)
        assert len(results) >= 3  # 至少 random_normal, xavier, he

    def test_he_stable_std(self):
        """He 初始化 + ReLU 应保持激活值的 std 大致稳定"""
        np.random.seed(42)
        results = visualize_initializations(in_features=256, out_features=256, n_layers=5)
        he_stats = results.get('he', results.get('He', []))
        stds = [s for _, s in he_stats]
        # std 不应该坍缩到接近 0 也不应该爆炸
        assert all(s > 0.1 for s in stds), f"std 坍缩了: {stds}"
        assert all(s < 100 for s in stds), f"std 爆炸了: {stds}"
