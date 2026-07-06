# 实验 02：激活函数

> **对应模块**: `nn_from_scratch/activations.py`  
> **依赖**: `gradient_check.py`（验证 backward）  
> **验证**: `pytest tests/test_activations.py -v`

## 学习目标

实现 5 种激活函数的前向和反向传播，理解每种函数的导数形式和数值特性。

## 理论背景

| 函数 | 公式 | 导数 | 特点 |
|------|------|------|------|
| Sigmoid | 1/(1+e⁻ˣ) | σ(1-σ) | 导数最大0.25，深层梯度消失 |
| ReLU | max(0,x) | 1 if x>0 else 0 | 简单高效，但死神经元 |
| LeakyReLU | x if x>0 else αx | 1 if x>0 else α | 解决死神经元 |
| Tanh | tanh(x) | 1-tanh²(x) | 输出均值0，仍有梯度消失 |
| ELU | x if x>0 else α(eˣ-1) | 1 if x>0 else out+α | 负半轴平滑饱和 |

## 需要阅读的资料

- CS231n: Activation functions — https://cs231n.github.io/neural-networks-1/#actfun
- 邱书第 4.1.3 节：激活函数

## 要实现的类

按顺序: `Sigmoid` → `ReLU` → `LeakyReLU` → `Tanh` → `ELU`

## 关键提示

- Sigmoid/Tanh 缓存**输出**，ReLU/LeakyReLU/ELU 缓存**输入**
- ELU backward 中 x≤0 部分梯度为 `dout * (out + alpha)`
- 每写完一个立刻用 gradient_check 验证

## 验证

```bash
pytest tests/test_activations.py -v
```
