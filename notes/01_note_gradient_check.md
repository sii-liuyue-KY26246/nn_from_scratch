# 01 · 梯度检验（Gradient Check）学习笔记

对应文件：`nn_from_scratch/gradient_check.py`

梯度检验：用**有限差分**近似梯度（数值梯度），和你 backward 推导的**解析梯度**对比。
相对误差 < `1e-5` 说明解析梯度大概率是对的。每写完一个新的 backward，第一件事就跑它。

---

## 一、`ndarray` 能不能在后面追加数据

不能原地追加。`ndarray` 底层是一块固定大小的连续内存，不像 Python `list` 那样能 `append` 增长。

需要"增长"时的做法：

| 场景 | 做法 |
|------|------|
| 循环中逐步收集 | 先用 Python `list` 收集，最后一次 `np.array(buf)` 转换（最快） |
| 已知最终形状 | `np.empty`/`np.zeros` 预分配后按下标填充（无复制） |
| 少量几次拼接 | `np.concatenate([a, b])` |

⚠️ 循环里反复 `np.append` 是 O(n²)（每次都复制整个数组），避免。

---

## 二、`eval_numerical_gradient`：标量函数的数值梯度

数值梯度是**逐元素**求偏导：每次只扰动 `x` 的一个元素，`f` 在完整的 `x` 上求值。

```python
grad = np.zeros_like(x)
for idx in np.ndindex(x.shape):
    old = x[idx]
    x[idx] = old + h
    fpos = f(x)
    x[idx] = old - h
    fneg = f(x)
    x[idx] = old                 # 改回来，关键
    grad[idx] = (fpos - fneg) / (2 * h)
return grad
```

**为什么用 `np.ndindex` 而不是 `enumerate`**

- `enumerate(x)` 只拆**第 0 维**，二维数组拿到的是"一整行"，到不了单个元素；维度不确定（1维/2维/4维）时还得写不同层数的循环。
- 拆包出来的变量是临时值，重新赋值改不动原数组，没法"写回"。
- `enumerate` 只给第 0 维的 `i`，拿不到深层元素的完整下标。

`np.ndindex(x.shape)` 依次生成所有多维下标 `(0,0),(0,1),...`，配合 `x[idx]` 读写 → **任意维度一套代码、能按下标精确写回**。它就是"多维版的 `range`"。

**为什么每轮只需存 `old`（不算创建副本）**

- `old = x[idx]` 存的是**一个标量**（~8 字节，常数大小，与 `x` 多大无关）→ O(1)。
- 若改用 `x + hei`（`hei` 是和 `x` 同尺寸的 one-hot 数组）则每轮分配/复制整个数组 → O(n)。
- "不创建副本"指不复制整个数组，`old` 存单个数是必要且极廉价的。

> 补充 `np.ndindex` vs `np.nditer`：两者都遍历多维数组。`ndindex` **只给下标**（"多维 range"），`nditer` 给**元素值**并可选下标/读写/多数组。梯度检验只需下标，`ndindex` 更简洁，且是普通 `for`，**没有忘记 `it.iternext()` 导致死循环的坑**。

---

## 三、`eval_numerical_gradient_array`：输出是数组的函数

层的 `f(x)` 返回**数组**，没法直接套标量版。技巧：用上游梯度 `dout` 把数组压成标量。

```python
scalar_f = lambda x: np.sum(f(x) * dout)
return eval_numerical_gradient(scalar_f, x, h)
```

**`scalar_f = lambda x: np.sum(f(x) * dout)` 的含义**

- `lambda x: 表达式` = 一行匿名函数，自动返回表达式的值，等价于
  ```python
  def scalar_f(x):
      return np.sum(f(x) * dout)
  ```
- 用 lambda 是因为它短、临时、只当参数传给 `eval_numerical_gradient`。
- 它通过**闭包**记住定义时环境里的 `f` 和 `dout`（这两个不是它的参数，是从外层函数捕获的）。

**为什么这样构造是对的**

令 L = Σ(f(x)·dout)，则

```
∂L/∂x = Σ_j dout_j · ∂f_j/∂x
```

正是链式法则里"上游梯度 `dout` 经过 `f` 回传到 `x`"的梯度，即数值版的 `dx`。所以对 `scalar_f` 求数值梯度 = 数值版的该层 backward 结果，用来检验解析 backward。

`np.sum(f(x) * dout)` 本质：用 `dout` 作权重，把 `f(x)` 的所有输出加权求和成一个标量。

---

## 四、`gradient_check`：对比数值梯度与解析梯度

相对误差 = |num − ana| / max(|num|, |ana|, 1e-8)，取最大值和阈值比较。

```python
num_grad = eval_numerical_gradient(f, x, h)
ana_grad = analytic_grad
err = np.abs(num_grad - ana_grad) / np.maximum(np.maximum(np.abs(num_grad), np.abs(ana_grad)), 1e-8)
if np.max(err) < threshold:
    if verbose:
        print(f"number_grad:{num_grad},analytic_grad{analytic_grad}")
    return True
else:
    return False
```

**`np.maximum` vs `np.max`（关键区别）**

| 函数 | 作用 |
|------|------|
| `np.max(a)` | 把**一个**数组规约成**一个**最大值 |
| `np.maximum(a, b)` | **两个**数组**逐元素**取较大值 |

分母要的是"对每个元素，取 `|num|`、`|ana|`、`1e-8` 三者中的最大"→ 逐元素操作 → 用 `np.maximum`。
`np.maximum` 一次只比两个，三者取最大就**嵌套**：`np.maximum(np.maximum(|num|, |ana|), 1e-8)`。
`1e-8` 会**广播**到每个元素，作用是**防除零**（两个梯度都接近 0 时）。

而第 98 行的 `np.max(err)` 是把整个误差数组规约成一个最大值（判断"最坏元素的误差是否达标"），是 `np.max` 的正确用法。

---

## 五、附：编辑器杂项

- `fpos`、`nditer` 等变量名报 `Unknown word` 是 **Code Spell Checker**（拼写检查插件）在管闲事，与代码正确性无关。加进 `cSpell.words` 词典或忽略即可。
- 禁用该插件只失去**英文拼写提示**；语法检查/报错/补全/跳转/格式化都由 Pylance、解释器等其他工具负责，不受影响。

---

## 术语速记（本节相关）

| 英文 | 中文 |
|------|------|
| Deep Learning (DL) | 深度学习 |
| Forward / Backward Propagation | 前向 / 反向传播 |
| Gradient Check | 梯度检验 |
| Numerical Gradient | 数值梯度 |
| Analytic Gradient | 解析梯度 |
| Finite Difference | 有限差分 |
| Relative Error | 相对误差 |
| Chain Rule | 链式法则 |
| Upstream / Downstream Gradient (`dout`) | 上游梯度 |
