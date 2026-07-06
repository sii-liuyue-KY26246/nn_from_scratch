# nn_from_scratch

> 纯 NumPy 从零实现神经网络基础组件，配合邱锡鹏《神经网络与深度学习》教材的第〇阶段实验。

## 这是什么

这不是一个框架，而是一个**学习项目**。

所有深度学习的核心计算——激活函数、损失函数、反向传播、优化器——都用纯 NumPy 从零实现，没有 PyTorch、TensorFlow 或任何自动微分工具。目标是在使用框架之前，真正理解每一行 `loss.backward()` 背后在发生什么。

## 背景

本项目是跟随邱锡鹏[《神经网络与深度学习》](https://nndl.github.io/)（"蒲公英书"）自学深度学习的配套实验。教材理论扎实但缺少动手实验，这个项目填补这个空缺。

设计理念：**先手撕，再框架**。每个组件先用 NumPy 从零实现并通过梯度检验，在后续阶段再用 PyTorch 重写，体会框架为你做了什么。

## 项目结构

```
phase0/
├── nn_from_scratch/            # Python 包（你来实现）
│   ├── module.py               # 基类 Module（已实现）
│   ├── gradient_check.py       # 数值梯度检验工具
│   ├── activations.py          # Sigmoid, ReLU, LeakyReLU, Tanh, ELU
│   ├── losses.py               # MSE, Binary CE, Softmax CE, SVM Hinge
│   ├── data_utils.py           # 归一化, 标准化, one-hot, DataLoader
│   ├── knn.py                  # 向量化距离计算 + k-NN 分类器
│   ├── linear_classifier.py    # Softmax / SVM 线性分类器
│   ├── layers.py               # Affine 全连接层 + Sequential 容器
│   ├── initializers.py         # zeros, Xavier, He 初始化 + 可视化
│   └── optim.py                # SGD, SGD+Momentum
├── tests/                      # pytest 测试（每个模块一个）
├── docs/                       # 实验文档（每个模块一份）
└── README.md
```

每个源文件包含完整的函数签名、docstring、数学公式和实现提示，但核心逻辑标记为 `raise NotImplementedError`，由你来填充。

## 快速开始

### 环境

```bash
# Python 3.10+，只需要 numpy 和 pytest
pip install numpy pytest matplotlib
```

### 实现流程

模块之间有依赖关系，按以下顺序实现：

```
01 gradient_check  →  你的调试武器，后面每个 backward 都靠它验证
       ↓
02 activations     →  Sigmoid, ReLU, LeakyReLU, Tanh, ELU
       ↓
03 losses          →  MSE, Binary CE, Softmax CE, SVM Hinge
       ↓
04 data_utils      →  归一化, DataLoader 等工具
       ↓
05 knn             →  练习 NumPy 向量化（核心目标不是 k-NN 本身）
       ↓
06 linear_classifier → 第一次写完整的训练循环
       ↓
07 layers          →  Affine 前向/反向 + Sequential（神经网络的原子积木）
       ↓
08 initializers    →  Xavier, He 初始化 + 可视化激活值分布
       ↓
09 optim           →  SGD, SGD+Momentum
```

### 验证

每完成一个模块：

```bash
pytest tests/test_gradient_check.py -v    # 单个模块
```

全部完成后：

```bash
pytest tests/ -v                          # 一次性验证所有模块
```

## 实验文档

`docs/` 目录下每个模块有一份实验文档，包含：

- 学习目标和理论背景
- 需要阅读的参考资料（CS231n、d2l.ai、邱书对应章节）
- 要实现的函数/类列表及数学定义
- 实现提示（方向性提示，不是答案）
- 验证命令

## 设计原则

**梯度检验驱动**：每个 backward 实现都通过数值梯度检验验证。测试中使用 `eval_numerical_gradient_array` 对比解析梯度和有限差分梯度，相对误差 < 1e-5 即通过。

**模块化继承**：所有层继承自 `Module` 基类，支持 `layer(x)` 调用语法（通过 `__call__`）、`train()/eval()` 模式切换、`parameters()` 参数收集。这和 PyTorch 的 `nn.Module` 设计一致，后续迁移到 PyTorch 时会感到熟悉。

**渐进式依赖**：前面的模块是后面的基础。gradient_check 验证 activations，activations 和 losses 被 layers 和 linear_classifier 使用，最终 optim 把一切串起来。

## 配套资源

| 资源 | 用途 |
|------|------|
| [邱锡鹏《神经网络与深度学习》](https://nndl.github.io/) | 理论主线 |
| [CS231n Notes](https://cs231n.github.io/) | 实现参考（本项目的很多设计直接对应 CS231n Assignment 1-2） |
| [d2l.ai 动手学深度学习](https://zh.d2l.ai/) | 代码参考 |
| [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 从零构建的视频教程 |

## 学习路线全景

本项目是一条更长学习路线的第〇阶段（Python + NumPy 基础）。完整路线：

```
第〇阶段  Python/NumPy 速成 + 本项目          ← 你在这里
第一阶段  前馈神经网络（micrograd, MLP from scratch）
第二阶段  卷积神经网络（手撕卷积, LeNet → ResNet）
第三阶段  循环神经网络（makemore, LSTM 文本生成）
第四阶段  Transformer（手撕 GPT, nanoGPT）
第五阶段  深度生成模型（VAE, GAN, Diffusion）
第六阶段  高级训练技术（BatchNorm, 混合精度, 完整 pipeline）
第七阶段  强化学习（Q-Learning → PPO）
第八阶段  AI Infra（分布式训练, CUDA, 推理优化）
```

## License

MIT
