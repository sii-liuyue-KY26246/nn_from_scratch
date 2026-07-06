# 实验 08：权重初始化

> **对应模块**: `nn_from_scratch/initializers.py`  
> **依赖**: 无  
> **验证**: `pytest tests/test_initializers.py -v`

## 学习目标

理解不同初始化方法的数学原理和对训练的影响。

## 理论背景

- **全零**: 对称性无法打破，网络学不到东西
- **Xavier**: Var(W) = 2/(fan_in + fan_out)，适合 Sigmoid/Tanh
- **He**: Var(W) = 2/fan_in，适合 ReLU

## 需要阅读的资料

- CS231n: Weight Initialization — https://cs231n.github.io/neural-networks-2/#init
- 邱书第 7.3 节：参数初始化

## 要实现的函数

`zeros` → `random_normal` → `xavier` → `he` → `visualize_initializations`

## 可视化实验（强烈推荐）

实现后画每层激活值 std 的变化。你会看到 He+ReLU 保持稳定，其他方法坍缩或爆炸。

## 验证

```bash
pytest tests/test_initializers.py -v
```
