# 实验 07：全连接层与 Sequential

> **对应模块**: `nn_from_scratch/layers.py`  
> **依赖**: `module.py`, `activations.py`, `gradient_check.py`  
> **验证**: `pytest tests/test_layers.py -v`

## 学习目标

实现全连接层的前向和反向传播——神经网络的原子积木。

## 理论背景

Affine 层: out = x @ W + b

反向传播（背下来！）:
- dx = dout @ W.T
- dW = x.T @ dout
- db = sum(dout, axis=0)

## 需要阅读的资料

- CS231n: Backpropagation — https://cs231n.github.io/optimization-2/
- 邱书第 4.3 节：反向传播算法
- d2l.ai 第 5.3 节：计算图

## 要实现的类

`Affine` → `Sequential`

## 关键提示

- Affine forward 处理高维输入：reshape 成 (N, D)，backward 还原
- Sequential backward 倒序遍历
- 用 gradient_check 验证 dx 和 dW

## 验证

```bash
pytest tests/test_layers.py -v
```
