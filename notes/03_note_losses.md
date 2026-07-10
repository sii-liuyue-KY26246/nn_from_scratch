# 03 · 损失函数（Losses）学习笔记

对应文件：`nn_from_scratch/losses.py`

4 个损失（MSE、Binary CE、Softmax CE、SVM Hinge），函数式设计：接收预测和标签，返回 `(loss, grad)`。
本节重点记几个**跨损失复用**的通用技巧，最后留一个待复习项。

---

## 一、花式索引：按行取"正确类"的值

softmax CE 和 SVM 都要"每个样本取出它正确类别对应的那个值"。别写循环，用**整数数组索引**一行搞定：

```python
N = scores.shape[0]
correct = scores[np.arange(N), targets]     # (N,) 每个样本正确类的分数/概率
```

- `np.arange(N)` = 行下标 `[0,1,...,N-1]`
- `targets` = 每行要取的列下标
- 两者**逐位置配对** → `scores[0,targets[0]], scores[1,targets[1]], ...`

例：`scores=[[2,1,0.5],[0.3,1.5,2.2]]`, `targets=[0,2]` → `[2.0, 2.2]`。

比 Python 循环**更快、更短、维度通用**，是 CS231n / PyTorch gather 正确类的标准写法。
> 广播小技巧：要和 `(N,C)` 相减时先 `.reshape(-1,1)` 变 `(N,1)`，让每行减自己的正确类值。

---

## 二、数值稳定性：softmax 减 max vs BCE clip（为什么不同）

两种损失都碰到 `log`/`exp` 的数值风险，但**处理手段不同**：

| 损失 | 风险 | 处理 |
|------|------|------|
| Softmax CE | `exp(大数)` 上溢成 inf | **减每行最大值**：`z -= np.max(z, axis=1, keepdims=True)` |
| Binary CE | `sigmoid` 饱和到恰好 0 或 1 → `log(0)=-inf` | **clip**：`np.clip(p, 1e-12, 1-1e-12)` |

**为什么 softmax 减 max 不改结果**：分子分母同乘 `e^{-max}` 约掉，数学等价，只是躲开溢出：

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

减完后最大类 `exp(0)=1`，分母 ≥ 1（良态），概率是干净的 `[0,1]`。

**为什么 BCE 更需要 clip**：sigmoid 极易饱和到 0 和 1**两端**，而 BCE 同时要算 `log(p)` 和 `log(1-p)`，**两个** log(0) 危险点，所以必须两边夹住。softmax CE 只有一个（`log(p_correct)`）且减 max 后很少触发，所以本练习不 clip 也能过；要根治用 log-sum-exp（`log_softmax`）而非 clip。

> softmax+CE、sigmoid+BCE 组合后梯度都极简：`grad = (probs - onehot)/N`、`(p - y)/N`。不是巧合，是配对的甜头。

---

## 三、逐样本损失 → 标量：归一化要除对数

所有损失都是"**每个样本一个 `L_i`（概念上 (N,)），再对 N 个样本求平均**"：

$$
L = \frac{1}{N}\sum_{i} L_i \quad(\text{标量})
$$

formula 常只写单样本的 `L_i`，外层平均藏在"最后除以 N"里。

### ⚠️ 我踩的坑：SVM 用了 np.mean，除错了数

```python
margins = np.maximum(0, scores - correct + 1.0)   # (N, C)
loss = np.mean(margins)        # ✗ 除以 N×C（所有元素），多除了 C 倍
loss = np.sum(margins) / N     # ✓ 只除以样本数 N
```

`np.mean` 会除以**总元素数 N×C**，但 SVM 损失只该除 N。当时 grad 那边除的是 N，loss 除的是 N×C，**两边归一化不一致 → 梯度检验差 C 倍**（测试报 `0.125 vs 0.025`，正好 5=C 倍）。

**教训**：loss 和 grad 必须除以**同一个 N**；`(N,C)` 矩阵想"按样本平均"要用 `np.sum(...)/N`，不是 `np.mean`。

---

## 四、⚠️ 待复习：SVM Hinge（margin 违反）

> 本节 SVM 的 **margin 违反机制和梯度推导当时没完全理解**，之后回来补。要点先记在这：
>
> - `L_i = Σ_{j≠y_i} max(0, s_j - s_{y_i} + margin)`，margin 默认 1。
> - "**margin 被违反**" = 正确类没比某错误类高出至少一个 margin，即 `s_j - s_{y_i} + margin > 0`。即使分对了，只要没拉开 margin 也罚。
> - `max(0,·)`：没违反的项归 0（不罚），只留违反的正值。
> - 梯度：每个被违反的错误类 `grad[i,j] += 1`；正确类 `grad[i,y_i] -= 该样本违反次数`；最后 `/N`。
> - 回来复习时重点想清楚：为什么正确类的梯度是"减去违反次数"（因为 `s_{y_i}` 出现在每个违反项里，各贡献 −1）。

---

## 术语速记

| 英文 | 中文 |
|------|------|
| Loss / Cost Function | 损失函数 |
| MSE (Mean Squared Error) | 均方误差 |
| Cross-Entropy | 交叉熵 |
| Softmax | 归一化指数函数 |
| Hinge Loss / SVM Loss | 合页损失 |
| Margin | 间隔 |
| Logits | 未经激活的原始分数 |
| Numerical Stability | 数值稳定性 |
| Fancy / Integer Array Indexing | 花式索引 |

相关：[[01_note_gradient_check]]（np.maximum、梯度检验）、[[02_note_activations]]（sigmoid、缓存技巧）。
