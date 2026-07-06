"""激活函数测试。

运行: pytest tests/test_activations.py -v
"""
import numpy as np
import pytest
from nn_from_scratch.activations import Sigmoid, ReLU, LeakyReLU, Tanh, ELU
from nn_from_scratch.gradient_check import eval_numerical_gradient_array


# ── Sigmoid ──────────────────────────────────────────────

class TestSigmoid:
    def test_forward_known_values(self):
        sig = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])
        out = sig(x)
        expected = np.array([0.5, 0.7310585786, 0.2689414214])
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_forward_shape(self):
        sig = Sigmoid()
        x = np.random.randn(8, 5)
        out = sig(x)
        assert out.shape == x.shape

    def test_output_range(self):
        sig = Sigmoid()
        x = np.random.randn(100, 50) * 10
        out = sig(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_backward_gradient_check(self):
        sig = Sigmoid()
        x = np.random.randn(4, 5)
        dout = np.random.randn(4, 5)
        _ = sig(x)
        analytic = sig.backward(dout)
        numerical = eval_numerical_gradient_array(lambda x: Sigmoid()(x), x, dout)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)


# ── ReLU ─────────────────────────────────────────────────

class TestReLU:
    def test_forward_known_values(self):
        relu = ReLU()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out, expected)

    def test_backward_gradient_check(self):
        relu = ReLU()
        x = np.random.randn(4, 5)
        x[np.abs(x) < 0.1] = 0.5  # 避免 x≈0 处的不可导点
        dout = np.random.randn(4, 5)
        _ = relu(x)
        analytic = relu.backward(dout)
        numerical = eval_numerical_gradient_array(lambda x: ReLU()(x), x, dout)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_backward_mask(self):
        relu = ReLU()
        x = np.array([-1.0, 2.0, -3.0, 4.0])
        _ = relu(x)
        dx = relu.backward(np.ones_like(x))
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(dx, expected)


# ── LeakyReLU ────────────────────────────────────────────

class TestLeakyReLU:
    def test_forward_known_values(self):
        lrelu = LeakyReLU(alpha=0.1)
        x = np.array([-2.0, 0.0, 2.0])
        out = lrelu(x)
        expected = np.array([-0.2, 0.0, 2.0])
        np.testing.assert_allclose(out, expected)

    def test_backward_gradient_check(self):
        lrelu = LeakyReLU(alpha=0.1)
        x = np.random.randn(4, 5)
        x[np.abs(x) < 0.1] = 0.5
        dout = np.random.randn(4, 5)
        _ = lrelu(x)
        analytic = lrelu.backward(dout)
        numerical = eval_numerical_gradient_array(lambda x: LeakyReLU(alpha=0.1)(x), x, dout)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)


# ── Tanh ─────────────────────────────────────────────────

class TestTanh:
    def test_forward_known_values(self):
        tanh = Tanh()
        x = np.array([0.0, 1.0, -1.0])
        out = tanh(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(out, expected)

    def test_backward_gradient_check(self):
        tanh = Tanh()
        x = np.random.randn(4, 5)
        dout = np.random.randn(4, 5)
        _ = tanh(x)
        analytic = tanh.backward(dout)
        numerical = eval_numerical_gradient_array(lambda x: Tanh()(x), x, dout)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)


# ── ELU ──────────────────────────────────────────────────

class TestELU:
    def test_forward_positive(self):
        elu = ELU(alpha=1.0)
        x = np.array([1.0, 2.0, 3.0])
        out = elu(x)
        np.testing.assert_array_equal(out, x)

    def test_forward_negative(self):
        elu = ELU(alpha=1.0)
        x = np.array([-1.0, -2.0])
        out = elu(x)
        expected = np.exp(x) - 1.0
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_backward_gradient_check(self):
        elu = ELU(alpha=1.0)
        x = np.random.randn(4, 5)
        x[np.abs(x) < 0.1] = 0.5
        dout = np.random.randn(4, 5)
        _ = elu(x)
        analytic = elu.backward(dout)
        numerical = eval_numerical_gradient_array(lambda x: ELU(alpha=1.0)(x), x, dout)
        np.testing.assert_allclose(analytic, numerical, atol=1e-5)
