"""nn_from_scratch: 从零实现深度学习基础组件。

一个纯 NumPy 实现的神经网络工具包，用于学习深度学习的底层原理。
"""

from .module import Module
from .activations import Sigmoid, ReLU, LeakyReLU, Tanh, ELU
from .losses import mse_loss, binary_cross_entropy_loss, softmax_cross_entropy_loss, svm_loss
from .gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, gradient_check
from .data_utils import normalize, standardize, one_hot_encode, train_val_test_split, DataLoader
from .knn import compute_distances_vectorized, KNNClassifier
from .linear_classifier import SoftmaxClassifier, SVMClassifier
from .layers import Affine, Sequential
from .initializers import zeros, random_normal, xavier, he
from .optim import SGD, SGDMomentum
