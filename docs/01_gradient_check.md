# 实验 01：梯度检验工具

> **对应模块**: `nn_from_scratch/gradient_check.py`  
> **依赖**: 无（第一个要实现的模块）  
> **验证**: `pytest tests/test_gradient_check.py -v`

## 学习目标

掌握数值梯度检验的原理和实现，这是你后续所有 backward 实现的验证工具。

## 理论背景

**有限差分近似**：对于标量函数 f(x)，用中心差分近似导数：

    df/dx ≈ (f(x + h) - f(x - h)) / (2h)

这比单侧差分更精确（误差 O(h²) vs O(h)）。

**相对误差**：比较数值梯度 num 和解析梯度 ana 时：

    rel_error = |num - ana| / max(|num|, |ana|, 1e-8)

经验法则：< 1e-5 通常正确，> 1e-3 几乎肯定有 bug。

## 需要阅读的资料

- CS231n: Gradient Check — https://cs231n.github.io/neural-networks-3/#gradcheck
- d2l.ai 第 2.4 节：自动微分的数值验证部分

## 要实现的函数

1. `eval_numerical_gradient(f, x, h)` — 标量函数的数值梯度
2. `eval_numerical_gradient_array(f, x, dout, h)` — 数组函数的数值梯度
3. `gradient_check(f, x, analytic_grad, ...)` — 完整检验流程

## 实现提示

- 遍历 x 每个元素时可用 `it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])`
- `eval_numerical_gradient_array` 定义 `scalar_f = lambda x: np.sum(f(x) * dout)` 然后复用第一个函数
- 修改 x 的值后要改回来

## 验证

```bash
pytest tests/test_gradient_check.py -v
```
