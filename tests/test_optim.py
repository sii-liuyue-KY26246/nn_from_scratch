"""优化器测试。

运行: pytest tests/test_optim.py -v
"""
import numpy as np
from nn_from_scratch.optim import SGD, SGDMomentum


class TestSGD:
    def test_step(self):
        W = np.array([1.0, 2.0, 3.0])
        dW = np.array([0.1, 0.2, 0.3])
        opt = SGD([(W, dW)], lr=1.0)
        opt.step()
        np.testing.assert_allclose(W, [0.9, 1.8, 2.7])

    def test_zero_grad(self):
        W = np.array([1.0, 2.0])
        dW = np.array([0.5, 0.5])
        opt = SGD([(W, dW)], lr=0.1)
        opt.zero_grad()
        np.testing.assert_array_equal(dW, [0.0, 0.0])

    def test_inplace_modification(self):
        """确保是原地修改，不是创建新数组"""
        W = np.array([1.0, 2.0])
        dW = np.array([1.0, 1.0])
        W_id = id(W)
        opt = SGD([(W, dW)], lr=0.1)
        opt.step()
        assert id(W) == W_id  # 同一个数组对象


class TestSGDMomentum:
    def test_first_step_matches_sgd(self):
        """momentum=0 时应该和普通 SGD 一样"""
        W = np.array([1.0, 2.0, 3.0])
        dW = np.array([0.1, 0.2, 0.3])
        opt = SGDMomentum([(W, dW)], lr=1.0, momentum=0.0)
        opt.step()
        np.testing.assert_allclose(W, [0.9, 1.8, 2.7])

    def test_momentum_accelerates(self):
        """有动量时，连续同方向更新应该越来越大"""
        W1 = np.array([0.0])
        dW1 = np.array([1.0])
        opt1 = SGD([(W1, dW1)], lr=0.1)

        W2 = np.array([0.0])
        dW2 = np.array([1.0])
        opt2 = SGDMomentum([(W2, dW2)], lr=0.1, momentum=0.9)

        for _ in range(5):
            opt1.step()
            opt2.step()

        # 有动量的应该走得更远
        assert abs(W2[0]) > abs(W1[0])

    def test_zero_grad(self):
        W = np.array([1.0])
        dW = np.array([1.0])
        opt = SGDMomentum([(W, dW)], lr=0.1, momentum=0.9)
        opt.zero_grad()
        np.testing.assert_array_equal(dW, [0.0])
