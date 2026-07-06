"""梯度检验工具测试。

运行: pytest tests/test_gradient_check.py -v
注意: 这个模块的测试不依赖其他模块，应该最先通过。
"""
import numpy as np
from nn_from_scratch.gradient_check import (
    eval_numerical_gradient, eval_numerical_gradient_array, gradient_check
)


class TestEvalNumericalGradient:
    def test_linear_function(self):
        """f(x) = sum(3*x)，梯度应该全是 3"""
        x = np.random.randn(4, 3)
        grad = eval_numerical_gradient(lambda x: np.sum(3 * x), x)
        np.testing.assert_allclose(grad, 3.0, atol=1e-5)

    def test_quadratic_function(self):
        """f(x) = sum(x^2)，梯度应该是 2*x"""
        x = np.random.randn(5)
        grad = eval_numerical_gradient(lambda x: np.sum(x ** 2), x)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-5)

    def test_shape_preserved(self):
        x = np.random.randn(3, 4)
        grad = eval_numerical_gradient(lambda x: np.sum(x), x)
        assert grad.shape == x.shape


class TestEvalNumericalGradientArray:
    def test_identity(self):
        """f(x) = x，dout 全 1，梯度应该全 1"""
        x = np.random.randn(3, 4)
        dout = np.ones_like(x)
        grad = eval_numerical_gradient_array(lambda x: x.copy(), x, dout)
        np.testing.assert_allclose(grad, 1.0, atol=1e-5)

    def test_scaling(self):
        """f(x) = 2*x，dout 全 1，梯度应该全 2"""
        x = np.random.randn(3, 4)
        dout = np.ones_like(x)
        grad = eval_numerical_gradient_array(lambda x: 2 * x, x, dout)
        np.testing.assert_allclose(grad, 2.0, atol=1e-5)


class TestGradientCheck:
    def test_correct_gradient_passes(self):
        f = lambda x: np.sum(x ** 2)
        x = np.random.randn(5)
        analytic_grad = 2 * x
        passed = gradient_check(f, x, analytic_grad, verbose=False)
        assert passed

    def test_wrong_gradient_fails(self):
        f = lambda x: np.sum(x ** 2)
        x = np.random.randn(5)
        wrong_grad = np.ones_like(x)  # 故意给错
        passed = gradient_check(f, x, wrong_grad, verbose=False)
        assert not passed
