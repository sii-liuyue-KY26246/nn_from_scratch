# 实验 04：数据预处理与加载

> **对应模块**: `nn_from_scratch/data_utils.py`  
> **依赖**: 无  
> **验证**: `pytest tests/test_data_utils.py -v`

## 学习目标

实现数据预处理和 mini-batch 加载的完整流水线。

## 理论背景

- **归一化**: 缩放到 [0,1] 或 [-1,1]，使不同特征在同一量级
- **标准化**: 均值0、方差1。训练集算参数，测试集复用
- **One-hot**: 整数标签 → 向量，用于和 softmax 输出对齐
- **Mini-batch**: 每次用一小批数据算梯度，batch_size 通常 32-256

## 需要阅读的资料

- CS231n: Data Preprocessing — https://cs231n.github.io/neural-networks-2/#datapre

## 要实现的函数/类

`normalize` → `standardize` → `one_hot_encode` → `train_val_test_split` → `DataLoader`

## 关键提示

- standardize 的 mean/std 从训练集算，测试集复用
- DataLoader 用 yield（生成器），注意全程用 self.X 不引用外部变量

## 验证

```bash
pytest tests/test_data_utils.py -v
```
