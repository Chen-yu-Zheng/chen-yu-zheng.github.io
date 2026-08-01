# 各类优化器中的动量设计: SGD、Adam 与 Muon

最近重温 Muon 优化器时，注意到一个有意思的细节：现有实现里能看到两种看起来很不一样的 momentum 写法。原始的版本（[链接](https://github.com/KellerJordan/modded-nanogpt/blob/973030408364f8738b4ad9e8f912d8cbbf56e4d4/train_gpt2.py)）使用的是 SGD 里的非归一化 plain momentum，更新的版本（如 Pytorch 版本 [链接](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/_muon.py#L164) ）使用的是 Adam 里的 EMA momentum；再叠加 Nesterov 写法之后，表面上差异更大。

于是本人花了一点时间把这个 “bug” 弄明白，并顺带复习了一下 SGD 和 Adam 的两类 momentum。这篇博客就简单 review 这些 momentum 的设计。这篇文章围绕三个问题展开：

1. 为什么 SGD momentum 通常使用 plain momentum，而不用 EMA？
2. 为什么 Adam 通常使用 EMA 风格的动量？
3. Muon 中 plain momentum / EMA momentum / Nesterov momentum 的实现关系是什么？

本文统一用如下符号：

- 参数为 $\theta_t$。
- 当前梯度为 $g_t = \nabla_\theta L(\theta_t)$。
- 学习率为 $\eta$。
- SGD / Muon 中常用动量系数记为 $\mu$。
- Adam 中一阶、二阶动量系数记为 $\beta_1, \beta_2$。

## SGD momentum: plain momentum

SGD momentum 用的是非归一化 plain momentum：

$$
b_t = \mu b_{t-1} + g_t,\ b_0 = 0
$$

$$
\theta_{t+1} = \theta_t - \eta b_t
$$

其中 $b_t$ 是 momentum buffer，$\mu \in [0, 1)$ 是 momentum 系数。实际 SGD 还涉及 dampening、Nesterov、weight decay 等细节，这里我们只考虑一般的动量。

### Plain momentum 的好处: 平滑方向，同时放大稳定方向

SGD momentum 的历史来源更接近 heavy-ball 方法中的 velocity。它不是在估计“梯度的均值”这个统计量，而是在构造一个带惯性的更新方向：

$$
\text{new direction}
=
\text{old velocity}
+
\text{current force}
$$

这个设计有一个很直接的好处：如果连续若干步的梯度方向比较一致，历史方向会被不断累积，更新会沿着这个稳定方向加速；如果某些方向上的梯度来回震荡，正负贡献会部分抵消，更新在这些方向上会被抑制。所以 plain momentum 同时做了两件事：

1. 在高曲率方向上，梯度方向经常反复变化，momentum 会互相抵消，减少 zig-zag；
2. 在低曲率方向上，梯度方向比较稳定，momentum 会持续累积，加速前进。

我们把递推展开来，理解一下为什么在方向稳定时放大有效步长：

$$
b_t
=
g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots \mu^{t-1}g_1
$$

如果一段时间内梯度近似恒定，即 $g_t \approx g$，那么稳态下：

$$
b_t \approx \frac{1-\mu^t}{1-\mu} g \approx \frac{1}{1-\mu} g
$$

此时参数更新近似为：

$$
\theta_{t+1}
\approx
\theta_t
-
\eta \frac{1}{1-\mu} g
$$

因此可以把有效学习率粗略理解为：

$$
\eta_{\text{eff}}
\approx
\frac{\eta}{1-\mu}
$$

当 $\mu = 0.9$ 时，这个因子约为 10，这正是 plain momentum 的一个重要 insight：**在梯度持续稳定的地方加速，通常是那些低曲率、梯度方向稳定、普通梯度下降收敛很慢的方向**。

### Plain momentum 而非 EMA

EMA 形式的动量（常用于 Adam 等现代优化器）通常写作：

$$
m_t = \mu m_{t-1} + (1-\mu) g_t
$$

展开后：

$$
m_t
=
(1-\mu)(g_t + \mu g_{t-1}
+\mu^2 g_{t-2} + \cdots \mu^t g_1)
$$

在稳态恒定梯度下，$m_t \approx (1-\mu)\frac{1-\mu^t}{1-\mu} g = (1-\mu^t)g \approx g$。也就是说，EMA 的设计目标是估计原本梯度的尺度，而非对梯度做累计。**这样本质上就退化成了普通的 SGD，失去了 SGD momentum 里那种天然的梯度累积加速的行为**。

## Adam: EMA 动量

Adam 的一阶动量和二阶动量通常写成：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\ m_0 = 0
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, v_0 = 0
$$

其中 $g_t^2$ 表示逐坐标平方。在更新前，还对动量还会额外使用 bias correction：

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

最最后的更新为：

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这里的重点是：**Adam 的 $m_t$ 和 $v_t$ 的目的不是和 SGD 里面一样做累积，而是对梯度和梯度尺度的在线估计**。

### EMA 动量估计尺度

Adam 的核心结构是 coordinate-wise normalized update：

$$
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

分子估计一阶矩，分母估计二阶矩的平方根。为了让这个比值的尺度有可解释性，$m_t$ 应该和 $g_t$ 具有相同尺度，$v_t$ 应该和 $g_t^2$ 具有相同尺度。**在上面提到了，为了估计尺度，我们应当用 EMA 动量**，而不是 SGD 的形式。

注意到，我们额外还做了 bias correction，这是因为动量从 0 初始化时，即使用了归一化 EMA，早期的权重和也还没有达到 1。例如，假设梯度恒定为 $g$，则：
$$
m_t =
(1-\beta_1)(g_t + \beta_1 g_{t-1}
+\beta_1^2 g_{t-2} + \cdots \beta_1^t g_1) \approx (1-\beta_1^t)g
$$

所以未校正的 $m_t$ 在训练初期会偏小。bias correction 用：

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}
$$

把这个初始化偏差校正回来。二阶动量同理。

### 根本差异：velocity vs moment estimate

SGD momentum 的 buffer 是速度：

$$
b_t = \mu b_{t-1} + g_t
$$

Adam 的 $m_t$ 是尺度估计：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

这不是表面公式差一个 $(1-\beta_1)$ 那么简单。真正的区别在于后续使用方式：

- SGD 直接用 buffer 作为更新方向，因此 buffer 的尺度会直接改变有效步长。
- Adam 用 $\hat m_t / (\sqrt{\hat v_t}+\epsilon)$ 做逐坐标归一化，因此 $m_t$ 和 $v_t$ 都应作为统计量估计保持尺度一致。

所以，SGD 里非 EMA 是合理的 velocity convention；Adam 里 EMA 是其矩估计解释和超参数 convention 的一部分。

## Muon: 两种动量实现方式

现有 Muon 实现里能看到之前说的两种原理很不一样的 momentum 写法。原始的版本（[链接](https://github.com/KellerJordan/modded-nanogpt/blob/973030408364f8738b4ad9e8f912d8cbbf56e4d4/train_gpt2.py)）使用的是 SGD 里的 plain momentum，更新的版本（如 Pytorch 版本 [链接](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/_muon.py#L164)）使用的是 Adam 里的 EMA momentum。我们现在说明这两种实现是等价的。

### 原始版本: plain 动量

原始的写法沿用 SGD 风格的动量 buffer，

$$
b_t = \mu b_{t-1} + g_t,\ b_0 = 0
$$

然后构造 Nesterov 方向，例如：

$$
u_t = \mu b_t + g_t
$$

在 Muon 中，$u_t$ 通常不会直接作为参数更新，而是先经过矩阵 orthogonalization：

$$
\bar u_t = \operatorname{Msign}(u_t)
$$

再更新：

$$
\theta_{t+1} = \theta_t - \eta \bar u_t
$$

### 新版写法: EMA 动量

另一种写法使用 EMA momentum：

$$
m_t = \mu m_{t-1} + (1-\mu)g_t,\ m_0 = 0
$$

然后构造 Nesterov 风格方向，例如：

$$
\tilde u_t = \mu m_t + (1-\mu)g_t
$$

### 两者等价

现在我们具体推导一下这两种写法的尺等价关系，核心的 insight（其实之前的推导已经能看出来了）是：**两种写法只影响动量的尺度，而在正交化的过程中尺度的影响会被抹去**。

把 $m_t$ 展开：

$$
m_t
=
(1-\mu)(g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots + \mu^{t-1}g_1)
$$

而 $b_t$ 展开为：

$$
b_t
=
g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots + \mu^{t-1}g_1
$$

因此两者满足精确的尺度关系：

$$
m_t = (1-\mu)b_t
$$

所以，只要初始化和递推顺序一致，EMA momentum 就是 plain momentum 乘了一个常数 $(1-\mu)$。

再看 Nesterov 方向。原始 plain momentum 写法中，进入 Muon orthogonalization 的方向是：

$$
u_t = \mu b_t + g_t
$$

新版 EMA 写法中，对应方向是：

$$
\tilde u_t = \mu m_t + (1-\mu)g_t
$$

把 $m_t=(1-\mu)b_t$ 代入：

$$
\begin{aligned}
\tilde u_t
&= \mu (1-\mu)b_t + (1-\mu)g_t \\
&= (1-\mu)(\mu b_t + g_t) \\
&= (1-\mu)u_t.
\end{aligned}
$$

因此，plain momentum + Nesterov 得到的 $u_t$，和 EMA momentum + Nesterov 得到的 $\tilde u_t$，只差一个正的标量 $(1-\mu)$。即两者的方向相同，尺度不同。这正是 Muon 中“看起来不同但可以等价”的关键，由于后续正交化操作满足

$$
\operatorname{Msign}(cU)=\operatorname{Msign}(U),\quad c>0
$$

那么：

$$
\operatorname{Msign}(\tilde u_t)
=
\operatorname{Msign}((1-\mu)u_t)
=
\operatorname{Msign}(u_t)
$$

得出最终参数更新方向就相同。换句话说，在 Muon 这种先构造 momentum direction、再做 scale-invariant matrix normalization 的结构里，plain momentum 和 EMA momentum 的常数尺度差异会被后续的 $\operatorname{Msign}$ 吸收掉。不过 EMA 可能更利于数值稳定，因为 $1-\mu$ 通常为 0.05 左右，得到更小的动量。

## Takeaways

- SGD 里的 plain momentum 不是为了估计梯度均值，而是一个 velocity buffer。它会在稳定方向上累积梯度，把有效步长放大到约 $\eta/(1-\mu)$，这也是它区别于普通 SGD 的关键。
- EMA momentum 的作用是保持梯度尺度可解释。Adam 必须使用这种 convention，因为它的更新依赖 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$，分子和分母都应该是矩估计，而不是任意缩放过的 velocity。
- Muon 里的 plain momentum 和 EMA momentum 看起来差别很大，但在相同零初始化和更新顺序下的动量仅是有尺度差别。因为 Muon 后续会对矩阵方向做 $\operatorname{Msign}$，而这个操作通常对正标量缩放无关，所以两种 momentum 写法的尺度差异会被吸收，最终更新方向可以相同。
