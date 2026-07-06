# 02 · 激活函数（Activations）学习笔记

对应文件：`nn_from_scratch/activations.py`

实现 5 个激活函数（Sigmoid、ReLU、LeakyReLU、Tanh、ELU）的前向与反向。
每个都继承 `Module`，无可学习参数，靠 `self.cache` 在 forward 存信息给 backward 用。

---

## 一、x 与 out 的缓存技巧

激活层的 backward 要用到 forward 时的信息，但**缓存什么，取决于导数用什么表达最方便**，分三种情况：

| 激活 | 缓存 | 原因（导数形式） |
|------|------|-----------------|
| Sigmoid、Tanh | **输出 out** | 导数能用输出表示：σ' = out(1−out)，tanh' = 1−out² |
| ReLU、LeakyReLU | **输入 x** | 导数取决于 x 的符号：`x > 0` 决定梯度是否通过 |
| ELU | **x 和 out 都缓存** | x>0 分支用 x 判断，x≤0 分支导数 = out + alpha |

选择标准很简单：**backward 的公式里出现的是谁，就缓存谁**。

### 关键坑：缓存输出后，backward 里不要再调一次激活函数

这是我在 Tanh 上踩过的错。导数是 `1 − tanh²(原始x)`，而 `self.cache` **已经是** `tanh(原始x)` 了：

```python
def forward(self, x):
    self.cache = np.tanh(x)          # cache 就是 out
    return self.cache

def backward(self, dout):
    out = self.cache                 # 已经是 tanh(x)
    return dout * (1 - out**2)       # ✅ 直接平方
    # ✗ 错误写法：dout * (1 - np.tanh(out)**2)
    #   会变成 1 - tanh(tanh(x))²，多套了一层 tanh
```

对照写对了的 Sigmoid：`out = σ(x)`，backward 用 `out * (1 - out)`，同样**没有再调 sigmoid**。
缓存输出的意义正是"省掉重算"——所以 backward 里绝不能对缓存的输出再套一次激活函数。

ReLU 则相反，缓存的是**输入 x**，backward 直接用符号掩码：

```python
def forward(self, x):
    self.cache = x                   # 缓存输入
    return np.maximum(0, x)          # 输出是 max(0, x)

def backward(self, dout):
    return dout * (self.cache > 0)   # x>0 处梯度通过，否则为 0
```

> 另一个易错点：ReLU forward 必须返回 `np.maximum(0, x)`，**不能**直接 `return self.cache`（那样等于没做激活，负数没被截断）。缓存和返回值是两回事。

---

## 二、`np.maximum` vs `np.max`

ReLU 前向用到，这两个名字像但作用完全不同：

| 函数 | 作用 |
|------|------|
| `np.max(a)` | 把**一个**数组**规约**成一个最大值（沿某轴） |
| `np.maximum(a, b)` | **两个**数组**逐元素**取较大值 |

ReLU 的 `max(0, x)` 是"对每个元素与 0 比大小" → **逐元素** → 用 `np.maximum(0, x)`（标量 `0` 广播到每个元素）：

```python
np.maximum(0, np.array([-2, 3, -1, 5]))   # array([0, 3, 0, 5]) ✅
np.max(np.array([-2, 3, -1, 5]))          # 5  ✗ 整个数组压成一个标量
```

用 `np.max` 会把整个数组变成一个标量，形状都错了。
（这和 note1 里 gradient_check 分母用 `np.maximum` 取"三者最大"是同一个区分点，见 [[01_note_gradient_check]]。）

---

## 术语速记（本节相关）

| 英文 | 中文 |
|------|------|
| Activation Function | 激活函数 |
| Sigmoid / Tanh / ReLU / LeakyReLU / ELU | （同名） |
| Vanishing Gradient | 梯度消失（Sigmoid/Tanh 深层易出现） |
| Dead Neuron | 死神经元（ReLU 的缺点，LeakyReLU 缓解） |
| Forward / Backward Pass | 前向 / 反向传播 |
| Cache | 缓存（forward 存、backward 取） |
