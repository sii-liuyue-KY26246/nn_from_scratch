"""数据工具测试。

运行: pytest tests/test_data_utils.py -v
"""
import numpy as np
from nn_from_scratch.data_utils import (
    normalize, standardize, one_hot_encode, train_val_test_split, DataLoader
)


class TestNormalize:
    def test_default_range(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_norm, _ = normalize(X)
        np.testing.assert_allclose(X_norm.min(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_norm.max(axis=0), 1.0, atol=1e-10)

    def test_custom_range(self):
        X = np.array([[0.0], [5.0], [10.0]])
        X_norm, _ = normalize(X, feature_range=(-1, 1))
        np.testing.assert_allclose(X_norm.flatten(), [-1.0, 0.0, 1.0], atol=1e-10)


class TestStandardize:
    def test_zero_mean_unit_var(self):
        np.random.seed(42)
        X = np.random.randn(1000, 5) * 3 + 7
        X_std, _ = standardize(X)
        np.testing.assert_allclose(X_std.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_std.std(axis=0), 1.0, atol=0.01)

    def test_reuse_params(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        _, params = standardize(X_train)
        X_test = np.array([[5.0, 6.0]])
        X_test_std, _ = standardize(X_test, mean=params['mean'], std=params['std'])
        assert X_test_std.shape == (1, 2)


class TestOneHotEncode:
    def test_basic(self):
        y = np.array([0, 1, 2, 1])
        oh = one_hot_encode(y, num_classes=3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(oh, expected)

    def test_auto_num_classes(self):
        y = np.array([0, 3, 1])
        oh = one_hot_encode(y)
        assert oh.shape == (3, 4)

    def test_sum_is_one(self):
        y = np.random.randint(0, 5, size=20)
        oh = one_hot_encode(y, num_classes=5)
        np.testing.assert_array_equal(oh.sum(axis=1), 1)


class TestTrainValTestSplit:
    def test_sizes(self):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(
            X, y, val_ratio=0.1, test_ratio=0.1, seed=42
        )
        assert len(Xtr) + len(Xv) + len(Xte) == 100

    def test_reproducibility(self):
        X = np.random.randn(50, 5)
        y = np.arange(50)
        split1 = train_val_test_split(X, y, seed=123)
        split2 = train_val_test_split(X, y, seed=123)
        np.testing.assert_array_equal(split1[0][1], split2[0][1])


class TestDataLoader:
    def test_len(self):
        X = np.random.randn(105, 10)
        y = np.random.randint(0, 2, 105)
        loader = DataLoader(X, y, batch_size=32)
        assert len(loader) == 4  # ceil(105/32)

    def test_iteration_covers_all(self):
        X = np.random.randn(100, 5)
        y = np.arange(100)
        loader = DataLoader(X, y, batch_size=32, shuffle=False)
        total = sum(xb.shape[0] for xb, _ in loader)
        assert total == 100

    def test_last_batch_smaller(self):
        X = np.random.randn(10, 3)
        y = np.arange(10)
        loader = DataLoader(X, y, batch_size=4, shuffle=False)
        sizes = [xb.shape[0] for xb, _ in loader]
        assert sizes == [4, 4, 2]

    def test_shuffle_changes_order(self):
        X = np.arange(20).reshape(20, 1).astype(float)
        y = np.arange(20)
        loader = DataLoader(X, y, batch_size=20, shuffle=True)
        batches = list(loader)
        # 非常不太可能 shuffle 后顺序完全一样
        x_batch = batches[0][0].flatten()
        assert not np.array_equal(x_batch, np.arange(20, dtype=float))
