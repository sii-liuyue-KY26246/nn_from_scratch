# 实验 09：优化器

> **对应模块**: `nn_from_scratch/optim.py`  
> **依赖**: `layers.py`, `activations.py`, `losses.py`  
> **验证**: `pytest tests/test_optim.py -v`

## 学习目标

实现 SGD 和 SGD+Momentum，理解动量如何加速收敛。

## 理论背景

- **SGD**: param -= lr * grad
- **Momentum**: v = momentum * v - lr * grad; param += v

动量的直觉：球在损失曲面上滚动时积累惯性，加速穿过峡谷，减小震荡。

**zero_grad 注意事项**: 用 `grad[...] = 0`（原地），不能用 `grad = np.zeros_like(grad)`（断开引用）。

## 需要阅读的资料

- CS231n: SGD and variants — https://cs231n.github.io/neural-networks-3/#sgd
- d2l.ai 第 12.6 节：动量法

## 要实现的类

`SGD` → `SGDMomentum`

## 验证

```bash
pytest tests/test_optim.py -v
```

**全部 9 个模块通过后**:

```bash
pytest tests/ -v
```

恭喜，第〇阶段工程实现部分完成！
