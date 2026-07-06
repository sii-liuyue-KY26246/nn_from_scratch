# 实验 05：向量化距离计算与 k-NN

> **对应模块**: `nn_from_scratch/knn.py`  
> **依赖**: 无  
> **验证**: `pytest tests/test_knn.py -v`

## 学习目标

练习 NumPy 向量化——实现 L2 距离矩阵的三个版本（双循环/单循环/零循环）。

## 理论背景

展开公式: ||a - b||² = ||a||² + ||b||² - 2a·b

用矩阵乘法一次性算出所有距离对，避免 Python 循环。

## 需要阅读的资料

- CS231n: k-Nearest Neighbor — https://cs231n.github.io/classification/#nn
- NumPy 广播机制文档

## 要实现的函数/类

`compute_distances_two_loops` → `compute_distances_one_loop` → `compute_distances_vectorized` → `KNNClassifier`

## 性能对比（推荐）

用 `time.time()` 对比三个版本速度，零循环版本应快 50-100 倍。

## 验证

```bash
pytest tests/test_knn.py -v
```
