"""层模块测试。

运行: pytest tests/test_layers.py -v
"""
import numpy as np
from nn_from_scratch.layers import Affine, Sequential
from nn_from_scratch.activations import ReLU
from nn_from_scratch.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient


class TestAffine:
    def test_forward_shape(self):
        layer = Affine(784, 256)
        x = np.random.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 256)

    def test_forward_with_reshape(self):
        """输入是高维张量时，应自动 flatten"""
        layer = Affine(3 * 4, 10)
        x = np.random.randn(2, 3, 4)
        out = layer(x)
        assert out.shape == (2, 10)

    def test_backward_dx_gradient_check(self):
        np.random.seed(0)
        layer = Affine(5, 3)
        x = np.random.randn(4, 5)
        dout = np.random.randn(4, 3)

        _ = layer(x)
        analytic_dx = layer.backward(dout)

        def f(x):
            temp = Affine(5, 3)
            temp.W = layer.W.copy()
            temp.b = layer.b.copy()
            return temp(x)

        numerical_dx = eval_numerical_gradient_array(f, x, dout)
        np.testing.assert_allclose(analytic_dx, numerical_dx, atol=1e-5)

    def test_backward_dW_gradient_check(self):
        np.random.seed(1)
        layer = Affine(5, 3)
        x = np.random.randn(4, 5)
        dout = np.random.randn(4, 3)

        _ = layer(x)
        _ = layer.backward(dout)
        analytic_dW = layer.dW

        def f(W):
            temp = Affine(5, 3)
            temp.W = W
            temp.b = layer.b.copy()
            return temp(x)

        numerical_dW = eval_numerical_gradient_array(f, layer.W.copy(), dout)
        np.testing.assert_allclose(analytic_dW, numerical_dW, atol=1e-5)

    def test_parameters(self):
        layer = Affine(10, 5)
        params = layer.parameters()
        assert len(params) == 2
        assert params[0][0].shape == (10, 5)  # W
        assert params[1][0].shape == (5,)     # b


class TestSequential:
    def test_forward(self):
        model = Sequential(Affine(784, 256), ReLU(), Affine(256, 10))
        x = np.random.randn(8, 784)
        out = model(x)
        assert out.shape == (8, 10)

    def test_backward_runs(self):
        model = Sequential(Affine(5, 3), ReLU(), Affine(3, 2))
        x = np.random.randn(4, 5)
        out = model(x)
        dout = np.random.randn(4, 2)
        dx = model.backward(dout)
        assert dx.shape == (4, 5)

    def test_parameters_collected(self):
        model = Sequential(Affine(5, 3), ReLU(), Affine(3, 2))
        params = model.parameters()
        # 2 Affine layers * 2 params each = 4
        assert len(params) == 4

    def test_repr(self):
        model = Sequential(Affine(5, 3), ReLU())
        r = repr(model)
        assert "Affine" in r
        assert "ReLU" in r
