# 实验 03：损失函数

> **对应模块**: `nn_from_scratch/losses.py`  
> **依赖**: `gradient_check.py`  
> **验证**: `pytest tests/test_losses.py -v`

## 学习目标

实现 4 种损失函数及其梯度。

## 理论背景

- **MSE**: L = (1/N)Σ(pred-target)², grad = (2/N)(pred-target)
- **Binary CE**: L = -(1/N)Σ[y·log(p)+(1-y)·log(1-p)]，grad = (1/N)(p-y)
- **Softmax CE**: 先 softmax 得概率，再交叉熵。梯度: grad = (probs - one_hot) / N
- **SVM Hinge**: L_i = Σ_{j≠y_i} max(0, s_j - s_{y_i} + 1)

## 需要阅读的资料

- CS231n: Linear Classification — https://cs231n.github.io/linear-classify/
- 邱书第 4.2 节：损失函数
- d2l.ai 第 3.4 节：Softmax 回归

## 要实现的函数

`mse_loss` → `binary_cross_entropy_loss` → `softmax_cross_entropy_loss` → `svm_loss`

## 关键提示

- Softmax 必须先减最大值保证数值稳定
- SVM 梯度：违反 margin 的位置 +1/N，正确类 -= count/N
- 用 np.clip 防止 log(0)

## 验证

```bash
pytest tests/test_losses.py -v
```
