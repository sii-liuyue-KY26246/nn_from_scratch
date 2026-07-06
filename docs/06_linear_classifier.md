# 实验 06：线性分类器

> **对应模块**: `nn_from_scratch/linear_classifier.py`  
> **依赖**: `losses.py`  
> **验证**: `pytest tests/test_linear_classifier.py -v`

## 学习目标

实现完整的"前向 → loss → 反向 → 更新"闭环。第一次写完整训练循环。

## 理论背景

- scores = X @ W + b
- L2 正则: loss += 0.5 * reg * ||W||², 梯度加 reg * W
- Mini-batch SGD: 每次随机取一批数据

## 需要阅读的资料

- CS231n: Linear Classification — https://cs231n.github.io/linear-classify/
- CS231n: Optimization — https://cs231n.github.io/optimization-1/

## 要实现的类

`LinearClassifier`（基类）→ `SoftmaxClassifier` → `SVMClassifier`

## fit 方法核心流程

1. `idx = np.random.choice(N, batch_size)` 采样
2. `scores = X_batch @ W + b`
3. `loss, dscores = self._compute_loss_and_grad(scores, y_batch)`
4. `dW = X_batch.T @ dscores + reg * W`
5. `db = np.sum(dscores, axis=0)`
6. `W -= lr * dW; b -= lr * db`
7. `loss += 0.5 * reg * np.sum(W**2)`

## 验证

```bash
pytest tests/test_linear_classifier.py -v
```
