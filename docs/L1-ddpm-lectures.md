# 扩散模型推导笔记 L1：DDPM 原理剖析（李宏毅课程听课笔记）

本文是本系列的听课笔记，整理李宏毅《机器学习 2023 春》课程 4 月 7 日的选修内容《【生成式AI】Diffusion Model 原理剖析》四段视频，讲义题为“Diffusion Model 背後的數學原理”。[L0](L0-image-generation-models.md) 第 2 章介绍了 DDPM 的训练算法和采样算法，课上说这两个算法“暗藏玄机”；这四讲用最大似然和变分下界解释两个算法为什么写成那样。[R1](R1-ddpm-paper-walkthrough.md) 把同一套推导写成定义、定理与证明，并逐式对照原论文。

读法同 L0：每节按 **术语**、**要点**、**公式**、**补充**、**联系** 排列。课上对代数推导点到为止，许多步骤只给结果，并注明“硬推”可得；本文也只写出关键的中间结果和出处，完整的逐步推导属于本系列的推导篇。第 0 章是一页速览，第 5 章把课程内容与推导篇对应起来，第 6、7 章是自测题和答案。

记号：模型的分布写作 $`P_\theta`$，前向（加噪）过程的分布写作 $`q`$，与课程一致；KL 散度写作 $`\mathrm{KL}(q\Vert P)`$，即 P0 中的 $`D_{\mathrm{KL}}`$。噪声调度同 L0：$`T=1000`$，$`\beta_t`$ 从 $`10^{-4}`$ 线性增加到 $`0.02`$，$`\alpha_t=1-\beta_t`$，$`\bar\alpha_t=\alpha_1\alpha_2\cdots\alpha_t`$，并约定 $`\bar\alpha_0=1`$。

引用说明：同 L0。文字是笔记作者自己的概括，课上引用的推导都对照原论文核实过式号；三幅插图由 `code/make_figures_l0_l1.py` 绘制，文中的数字由 `code/l0_l1_check.py` 核对。版本 1.0，2026-10-20。

## 0 一页速览

| 讲次 | 课上回答的问题 | 一句话结论 | 本文 |
|---|---|---|---|
| 1/4 | 两个算法里藏着哪些细节？ | 训练时只加一次噪声；网络只预测噪声；采样时去噪之后还要再加噪声 | 第 1 章 |
| 2/4 | 生成模型到底在优化什么？ | 最大似然等价于最小化 $`\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$；似然算不出来，改为最大化下界，VAE 与 DDPM 是同一个套路 | 第 2 章 |
| 3/4 | 下界拆开以后，怎样变成预测噪声？ | 前向过程一步到位；下界三项中只有去噪匹配项要训练；它要求网络的均值靠近后验均值，换元后等价于预测噪声 | 第 3 章 |
| 4/4 | 采样时为什么还要加噪声？ | 老师的推测：概率最高的结果不一定最好，每一步都要保留随机性；扩散模型可以看作“N 次到位”；文字也能做扩散 | 第 4 章 |

## 1 训练与采样算法逐行读（原理剖析 1/4）

### 1.1 训练算法

**要点**：DDPM 算法 1 逐行读：

1. 从训练集取一张干净图像 $`x_0`$；
2. 从 $`\{1,\ldots,T\}`$ 中均匀地抽一个步数 $`t`$；
3. 抽一份噪声 $`\epsilon\sim\mathcal{N}(0,I)`$；
4. 把 $`x_0`$ 和 $`\epsilon`$ 分别乘以 $`\sqrt{\bar\alpha_t}`$ 和 $`\sqrt{1-\bar\alpha_t}`$ 后相加，得到带噪图像，连同 $`t`$ 一起送进噪声预测器；
5. 让预测器的输出接近 $`\epsilon`$，即对 $`\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\Vert^2`$ 做一步梯度下降；
6. 重复以上步骤，直到收敛。

$`\bar\alpha_1,\ldots,\bar\alpha_T`$ 是事先算好的一串数，从接近 1 逐渐减小到接近 0：$`t`$ 越大，干净图像的成分越少，噪声的成分越多。

### 1.2 想象中与实际上：只加一次噪声

**要点**：按 L0 第 2.4 节的直观图景，训练好像是逐步加噪：第 1 步加一点，第 2 步再加一点，网络学的是“去掉一步的噪声”。但算法 1 只加一次噪声：直接由 $`x_0`$ 合成第 $`t`$ 步的带噪图像，正确答案是这一次加入的全部噪声 $`\epsilon`$，并没有“加很多次噪声”这件事。为什么可以这样做，见第 3.1 节：多次加入的独立高斯噪声可以合并成一份。

### 1.3 采样算法：去噪之后还要加噪声

**要点**：DDPM 算法 2 逐行读：先抽一份纯噪声 $`x_T\sim\mathcal{N}(0,I)`$；然后 $`t`$ 从 $`T`$ 到 1，每一步先抽一份新的噪声 $`z\sim\mathcal{N}(0,I)`$（最后一步取 $`z=0`$），再按下式由 $`x_t`$ 算出 $`x_{t-1}`$；最后输出 $`x_0`$。

**公式**：

```math
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)+\sigma_t z
```

括号里是从输入中减去按比例缩放的预测噪声，再整体乘以 $`1/\sqrt{\alpha_t}`$；最后加上的 $`\sigma_t z`$ 是一份新抽的噪声。讲义在这里打了一个问号：明明在去噪，为什么还要加噪声？第 3.5 节说明括号里的部分从哪里来，第 4 章讨论为什么要加 $`\sigma_t z`$。

**补充**：$`\sigma_t`$ 怎么取？DDPM 第 3.2 节试了 $`\sigma_t^2=\beta_t`$ 和 $`\sigma_t^2=\tilde\beta_t`$ 两种（$`\tilde\beta_t`$ 见第 3.3 节），结果相近。前者在数据本身服从标准正态分布时最优，后者在数据只有一个固定点时最优，两者是两个极端。

## 2 生成模型的共同目标（原理剖析 2/4）

### 2.1 让生成分布接近真实分布

**术语**：$`P_{\mathrm{data}}(x)`$ 是真实图像的分布。$`P_\theta(x)`$ 是生成网络的输出分布：从简单分布抽 $`z`$，经参数为 $`\theta`$ 的网络得到 $`x`$，所有可能的输出构成一个分布。

**要点**：VAE、Flow、扩散模型、GAN 的共同目标，都是让 $`P_\theta(x)`$ 尽量接近 $`P_{\mathrm{data}}(x)`$。有文字条件时，对每一段文字都要求如此，下面为简洁省略条件。剩下的问题是：怎样衡量两个分布“接近”？

### 2.2 最大似然等价于最小化 KL 散度

**术语**：最大似然估计（Maximum Likelihood Estimation, MLE）：选择参数 $`\theta`$，使训练样本在模型下出现的概率最大。

**要点**：

1. 从 $`P_{\mathrm{data}}`$ 中收集 $`m`$ 张图 $`x^1,\ldots,x^m`$。假如能算出每张图在模型下的概率 $`P_\theta(x^i)`$，就选使 $`\prod_i P_\theta(x^i)`$ 最大的 $`\theta`$。讲义在“能算出 $`P_\theta(x^i)`$”旁打了问号：对生成网络而言，这一步恰恰很难，第 2.3 节以后专门处理。
2. 取对数，把乘积变成求和；样本足够多时，样本平均接近期望。
3. 期望可以拆成两项，其中 $`P_{\mathrm{data}}`$ 的熵与 $`\theta`$ 无关，剩下的就是 KL 散度。

**公式**：

```math
\theta^*=\arg\max_\theta\prod_{i=1}^m P_\theta(x^i)=\arg\max_\theta\sum_{i=1}^m\log P_\theta(x^i)\approx\arg\max_\theta\,\mathbb{E}_{x\sim P_{\mathrm{data}}}[\log P_\theta(x)]
```

```math
\mathbb{E}_{x\sim P_{\mathrm{data}}}[\log P_\theta(x)]=\int P_{\mathrm{data}}(x)\log P_\theta(x)\,dx=-H(P_{\mathrm{data}})-\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)
```

所以 $`\theta^*\approx\arg\min_\theta\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$：最大似然就是最小化真实分布到模型分布的 KL 散度。

**补充（数值例子）**：设真实分布 $`P_{\mathrm{data}}=\mathcal{N}(1,0.5^2)`$，模型 $`P_\theta=\mathcal{N}(\mu,1)`$，参数只有 $`\mu`$。使平均对数似然最大的 $`\mu`$ 就是样本均值：用 10、1000、100000 个样本时分别是 1.0423、0.9750、0.9999，样本越多越接近 1。而 $`\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$ 恰好在 $`\mu=1`$ 处最小，最小值为 0.3181。也可以核对上面的恒等式：$`H(P_{\mathrm{data}})=0.7258`$；$`\mu=1`$ 时 $`\mathbb{E}[\log P_\theta]=-1.0439=-0.7258-0.3181`$，$`\mu=0.5`$ 时 $`\mathbb{E}[\log P_\theta]=-1.1689=-0.7258-0.4431`$。

**联系**：交叉熵等于熵加 KL 散度，见 [P0 第 3.6 节](P0-probability-and-information.md#36-交叉熵)；KL 散度的非负性见 [P0 第 3.4 节](P0-probability-and-information.md#34-kl-散度的性质)。

### 2.3 VAE 怎样计算似然

**要点**：VAE 先抽 $`z\sim P(z)`$（标准正态分布），再由网络 $`G`$ 算出 $`G(z)`$。把所有 $`z`$ 的贡献加起来，就是一张图的概率：

**公式**：

```math
P_\theta(x)=\int_z P(z)\,P_\theta(x\mid z)\,dz
```

难点在 $`P_\theta(x\mid z)`$。如果把网络看成确定的映射，即 $`G(z)=x`$ 时概率为 1、否则为 0，那么对几乎每一张真实图像，都没有 $`z`$ 恰好生成它，$`P_\theta(x)`$ 几乎处处为 0，无法用来训练。VAE 的做法是把 $`G(z)`$ 看成一个高斯分布的均值：$`G(z)`$ 离 $`x`$ 越近，$`x`$ 的概率越大，

```math
P_\theta(x\mid z)\propto\exp\big(-\Vert G(z)-x\Vert^2\big)
```

**补充**：讲义上写的是 $`\exp(-\Vert G(z)-x\Vert_2)`$。严格地说，方差为 $`\sigma^2`$ 的各向同性高斯，指数是距离平方乘以 $`-1/(2\sigma^2)`$；这个比例系数不影响这里的讨论。

### 2.4 VAE 的证据下界

**术语**：证据下界（Evidence Lower BOund, ELBO）是对数似然 $`\log P_\theta(x)`$ 的一个下界。似然本身算不出来时，就改为最大化这个下界。

**要点**：推导只用一个技巧：任取一个关于 $`z`$ 的分布 $`q(z\mid x)`$，因为它的积分为 1，可以把 $`\log P_\theta(x)`$ 写成对它的积分，再在对数里同时乘、除 $`q(z\mid x)`$。

**公式**：

```math
\log P_\theta(x)=\int_z q(z\mid x)\log P_\theta(x)\,dz=\int_z q(z\mid x)\log\frac{P(z,x)}{P(z\mid x)}\,dz
```

```math
=\int_z q(z\mid x)\log\frac{P(z,x)}{q(z\mid x)}\,dz+\mathrm{KL}\big(q(z\mid x)\Vert P(z\mid x)\big)\ \ge\ \mathbb{E}_{q(z\mid x)}\Big[\log\frac{P(x,z)}{q(z\mid x)}\Big]
```

第二行的 KL 散度非负，去掉它就得到下界。$`q(z\mid x)`$ 可以是任何分布，在 VAE 中它就是编码器。把下界推高，$`\log P_\theta(x)`$ 也随之被推高。

**联系**：ELBO 的完整推导，以及它为什么等于“重建项减去 KL 项”，见 [D2–D3 笔记](D2-D3-elbo-and-vae-loss.md)第 4 至 6 章。

### 2.5 DDPM 怎样计算似然

**要点**：DDPM 的生成过程是一条链：先抽 $`x_T\sim P(x_T)=\mathcal{N}(0,I)`$，再一步步去噪。课上把每一步去噪也看成高斯分布，网络的输出 $`G(x_t)`$ 是它的均值。

**公式**：

```math
P_\theta(x_{t-1}\mid x_t)\propto\exp\big(-\Vert G(x_t)-x_{t-1}\Vert^2\big)
```

```math
P_\theta(x_0)=\int P(x_T)\,P_\theta(x_{T-1}\mid x_T)\cdots P_\theta(x_{t-1}\mid x_t)\cdots P_\theta(x_0\mid x_1)\,dx_1\cdots dx_T
```

要对 $`x_1,\ldots,x_T`$ 全部积分，比 VAE 的情形更难计算。

### 2.6 DDPM 的证据下界

**要点**：套用 VAE 的技巧，把 $`z`$ 换成 $`x_1,\ldots,x_T`$，把 $`q(z\mid x)`$ 换成 $`q(x_1,\ldots,x_T\mid x_0)`$。区别在于：VAE 的 $`q`$ 是要训练的编码器，DDPM 的 $`q`$ 就是前向（加噪）过程，事先规定好，没有参数。

**公式**：

```math
\log P_\theta(x_0)\ \ge\ \mathbb{E}_{q(x_{1:T}\mid x_0)}\Big[\log\frac{P(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big],\qquad q(x_{1:T}\mid x_0)=q(x_1\mid x_0)\,q(x_2\mid x_1)\cdots q(x_T\mid x_{T-1})
```

| | VAE | DDPM |
|---|---|---|
| 隐变量 | $`z`$ | $`x_1,\ldots,x_T`$ |
| $`q`$ | 编码器 $`q(z\mid x)`$，要训练 | 前向过程 $`q(x_{1:T}\mid x_0)`$，固定 |
| 生成 | 解码器 $`P_\theta(x\mid z)`$，一步 | $`T`$ 步去噪 $`P_\theta(x_{t-1}\mid x_t)`$ |
| 训练目标 | 最大化 ELBO | 最大化 ELBO |

**补充**：因此 DDPM 可以看作一个有 $`T`$ 层隐变量、编码器固定不训练的层级 VAE；Luo (2022) 称之为变分扩散模型（Variational Diffusion Model）。

## 3 下界拆开以后（原理剖析 3/4）

### 3.1 前向过程一步到位

**术语**：噪声调度（noise schedule）$`\beta_1,\ldots,\beta_T`$ 是事先给定的一串小正数，规定每一步加多少噪声。

**要点**：

1. 前向过程每一步是 $`x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t`$，其中 $`\epsilon_t\sim\mathcal{N}(0,I)`$，各步的噪声相互独立；也就是 $`q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1},\beta_tI)`$。
2. 连续两步：把 $`x_1`$ 的式子代入 $`x_2`$，得到 $`x_2=\sqrt{1-\beta_2}\sqrt{1-\beta_1}\,x_0+\sqrt{1-\beta_2}\sqrt{\beta_1}\,\epsilon_1+\sqrt{\beta_2}\,\epsilon_2`$。后两项是独立的零均值高斯，它们的和仍是零均值高斯，方差是两者方差之和 $`(1-\beta_2)\beta_1+\beta_2=1-(1-\beta_1)(1-\beta_2)`$。所以两份噪声可以合并成一份，见下面第一个式子。
3. 依此类推，任意一步都可以由 $`x_0`$ 一步算出，中间各步的噪声根本不必抽出来。这就是训练时只加一次噪声的原因（第 1.2 节）。

**公式**：

```math
x_2=\sqrt{(1-\beta_1)(1-\beta_2)}\,x_0+\sqrt{1-(1-\beta_1)(1-\beta_2)}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
```

```math
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad q(x_t\mid x_0)=\mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\big)
```

**补充（数值）**：

- 例：$`\beta_1=0.1`$、$`\beta_2=0.2`$ 时，$`x_0`$ 的系数是 $`\sqrt{0.72}=0.8485`$，合并后噪声的标准差是 $`\sqrt{0.28}=0.5292`$；100 万次模拟给出的均值和标准差与之吻合。
- 线性调度下，$`\bar\alpha_t`$ 在 $`t=1`$、10、100、250、500、750、1000 时分别约为 0.9999、0.9981、0.8970、0.5241、0.0786、0.0034、$`4.0\times10^{-5}`$，所有 $`\beta_t`$ 之和为 10.05。图 1(a) 画出了两个系数随 $`t`$ 的变化。逐步加噪 1000 次的模拟结果，与一步公式给出的均值、方差一致。

![图 1：(a) 前向过程中干净图像与噪声的系数；(b) 后验均值中两个系数与方差之比，横轴为对数刻度。](figures/l1_fig1_schedule.png)

**联系**：两份独立高斯噪声的合并就是 [D0(c) 噪声合并](D0-D1-gaussian-and-kl.md#3-d0c-噪声合并)；对任意 $`t`$ 的归纳证明是推导题 D4。

### 3.2 下界的三项

**要点**：把第 2.6 节的下界展开、整理（Luo, 2022，式 (47) 至 (58)；课上跳过了这段代数，直接引用结果），得到三项：

**公式**：

```math
\mathbb{E}_{q(x_1\mid x_0)}[\log P_\theta(x_0\mid x_1)]-\mathrm{KL}\big(q(x_T\mid x_0)\Vert P(x_T)\big)-\sum_{t=2}^{T}\mathbb{E}_{q(x_t\mid x_0)}\Big[\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\Vert P_\theta(x_{t-1}\mid x_t)\big)\Big]
```

Luo (2022) 把三项分别称为重建项（reconstruction term）、先验匹配项（prior matching term）和去噪匹配项（denoising matching term），见图 2。

1. 第二项与 $`\theta`$ 无关：$`q(x_T\mid x_0)`$ 由前向过程决定，$`P(x_T)`$ 是标准正态分布，两者都没有参数，训练时可以忽略。它本身也几乎为 0：$`x_0=1`$ 时，每一维约为 $`2.0\times10^{-5}`$。
2. 要训练的是第一项和第三项。课上的重点是第三项：对每个 $`t`$，让模型的一步去噪 $`P_\theta(x_{t-1}\mid x_t)`$ 接近 $`q(x_{t-1}\mid x_t,x_0)`$。DDPM 的简化损失把第一项也并入同样的形式（DDPM 第 3.4 节），所以下面只讨论第三项。

![图 2：DDPM 下界的三项。橙色箭头是固定的前向加噪，蓝色箭头是要训练的反向去噪。](figures/l1_fig2_elbo_terms.png)

**联系**：从下界到三项的逐步展开是推导题 D6。

### 3.3 前向过程的后验

**术语**：后验 $`q(x_{t-1}\mid x_t,x_0)`$ 指已知干净图像 $`x_0`$ 和第 $`t`$ 步的带噪图像 $`x_t`$ 时，第 $`t-1`$ 步图像的分布。

**要点**：由贝叶斯规则，再利用前向过程的马尔可夫性质 $`q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1})`$，后验可以写成三个已知分布的组合：

**公式**：

```math
q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
```

右边三个分布都是第 3.1 节已知的高斯分布。课上说经过“硬推”可得结果（Luo, 2022，式 (71) 至 (84)，用的是配方法）：后验仍是高斯分布（DDPM 式 (6)(7)），

```math
q(x_{t-1}\mid x_t,x_0)=\mathcal{N}\big(\tilde\mu_t,\tilde\beta_tI\big),\qquad \tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
```

均值是 $`x_0`$ 与 $`x_t`$ 的加权组合，方差只与 $`t`$ 有关。

**补充（数值）**：两个系数和方差之比 $`\tilde\beta_t/\beta_t`$ 随 $`t`$ 的变化见图 1(b)，几个代表值如下。

| $`t`$ | $`x_0`$ 的系数 | $`x_t`$ 的系数 | $`\tilde\beta_t/\beta_t`$ |
|---|---|---|---|
| 2 | 0.5453 | 0.4547 | 0.4547 |
| 10 | 0.1473 | 0.8527 | 0.8528 |
| 500 | 0.0031 | 0.9941 | 0.9991 |
| 1000 | 0.0001 | 0.9899 | 1.0000 |

$`t`$ 大时，后验均值几乎就是略作缩放的 $`x_t`$，每一步只去掉一点噪声；$`t`$ 小时，$`x_0`$ 的分量才变得重要。用联合高斯的条件公式可以得到同样的结果，40 万次模拟后的回归也与之吻合。

**联系**：贝叶斯规则见 [P0 第 2 章](P0-probability-and-information.md#2-贝叶斯规则311-节)；“高斯先验乘高斯似然”这一类后验，在 D0–D1 的 [5.2 节](D0-D1-gaussian-and-kl.md#52-后验分布套路二)和 [5.3 节](D0-D1-gaussian-and-kl.md#53-另一种方法联合高斯的条件公式)有逐步推导；这一后验的完整推导是推导题 D5。

### 3.4 最小化 KL：只需让均值靠近

**要点**：第三项中的 KL 散度，左边 $`q(x_{t-1}\mid x_t,x_0)`$ 的均值和方差都已确定；右边 $`P_\theta(x_{t-1}\mid x_t)`$ 的方差也事先固定为 $`\sigma_t^2`$，只有均值，即网络的输出，可以调整。讲义画了两个圆：一个均值、方差都固定，一个方差固定、均值可调。要让 KL 最小，就把可调的均值移到固定的均值上。

**公式**：两个方差相同的高斯分布之间的 KL 散度只取决于均值之差，

```math
\mathrm{KL}\big(\mathcal{N}(\mu_1,\sigma^2I)\Vert\mathcal{N}(\mu_2,\sigma^2I)\big)=\frac{\Vert\mu_1-\mu_2\Vert^2}{2\sigma^2}
```

例如 $`\mathrm{KL}(\mathcal{N}(0.3,0.25)\Vert\mathcal{N}(1.1,0.25))=0.8^2/(2\times0.25)=1.28`$。两边方差不相同时，KL 只多出一个与均值无关的常数，结论不变。所以最小化第三项，就是让网络输出的均值尽量接近后验均值 $`\tilde\mu_t`$。

**联系**：一般的高斯 KL 公式与方差相同的特例，见 D0–D1 的 [6.3 节](D0-D1-gaussian-and-kl.md#63-两个特例-c)。

### 3.5 把干净图像换成噪声：网络只需预测噪声

**要点**：

1. 目标均值 $`\tilde\mu_t`$ 里含有 $`x_0`$，而采样时 $`x_0`$ 正是要生成的东西，不能当作输入。
2. 训练时 $`x_t`$ 是由 $`x_0`$ 和 $`\epsilon`$ 合成的：$`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`$，反解得 $`x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}`$。代入 $`\tilde\mu_t`$ 整理后，$`x_0`$ 消失了。
3. $`x_t`$ 是网络的输入，已知；唯一的未知量是噪声 $`\epsilon`$。所以网络只需预测噪声，再按同样的式子组合出均值。

**公式**（DDPM 式 (10)(11)）：

```math
\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big),\qquad \mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
```

第二个式子正是采样算法第 4 行中去掉 $`\sigma_t z`$ 的部分（第 1.3 节）。用随机抽取的 1000 组 $`(t,x_0,\epsilon)`$ 核对，两种写法的差不超过 $`10^{-12}`$。

**补充**：代回 KL 项后，第三项中的每一项都变成带权重的噪声预测误差（DDPM 式 (12)）：

```math
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\Vert\epsilon-\epsilon_\theta(x_t,t)\Vert^2
```

取 $`\sigma_t^2=\beta_t`$ 时，这个权重在 $`t=1`$、10、100、500、1000 时约为 0.50、0.074、0.010、0.0055、0.010。DDPM 去掉权重，得到简化损失 $`L_{\mathrm{simple}}`$（式 (14)），也就是训练算法第 5 行。论文第 3.4 节解释：与完整的下界相比，简化损失降低了小 $`t`$ 各项（噪声很少、很容易去噪）的权重，让网络把精力放在 $`t`$ 较大、更难的去噪上，样本质量反而更好。

**联系**：从 KL 到预测噪声的逐步推导是推导题 D7。

## 4 采样为什么要加噪声，以及推广（原理剖析 4/4）

### 4.1 为什么不直接取均值

**要点**：去噪网络给出的是一个高斯分布的均值，采样算法却不直接取这个均值，而是从这个高斯分布中抽一个样本，即加上 $`\sigma_t z`$。为什么？讲义在这一段前写明“以下只是猜测”：第 4.2 至 4.4 节的说法是老师本人的推测，不是论文的结论。

**补充**：从概率模型的角度，答案是确定的。DDPM 定义的模型是一条马尔可夫链，每一步 $`P_\theta(x_{t-1}\mid x_t)`$ 都是一个分布；沿着链逐步抽样，最后得到的 $`x_0`$ 才是来自模型分布 $`P_\theta(x_0)`$ 的样本。每一步只取均值，得到的就不再是这个分布的样本。第 4.5 节的小实验显示了两者的差别。

### 4.2 文本生成：概率最高的句子不一定好

**要点**：语言模型每一步给出下一个字的概率分布。每次都取概率最高的字，看起来最稳妥，效果却不好。Holtzman et al. (2020) 发现，以概率最大为目标的解码方法（如束搜索，beam search）生成的文本平淡、不连贯，还常常陷入重复；而人写的文本并不是概率最高的那一句。所以生成文本时，需要按概率抽样。

### 4.3 语音合成：推理时保留 dropout

**要点**：Tacotron 2（Shen et al., 2018）在自回归解码器的预处理网络（pre-net）中使用概率为 0.5 的 dropout，并且推理时也不关闭，目的是让输出有所变化（论文第 2.2 节）。课上播放了一位同学的对比实验：推理时保留 dropout 与关闭 dropout 的合成语音。

### 4.4 扩散模型是“N 次到位”的自回归

**要点**：课程较早的一讲区分过两种生成策略：自回归（各个击破，一次生成一个元素）和非自回归（一次到位，一次生成全部元素）。扩散模型把“一次到位”改成“N 次到位”：每一步都重新生成整张图，并以上一步的结果为条件，这与自回归“每一步以前面的结果为条件”相似。老师据此推测：自回归生成需要在每一步抽样以保持多样性，扩散模型的每一步大概也需要。

### 4.5 一个可以自己验证的小实验

**要点**：讲义第 40 页给出了助教的对比：$`\sigma_t`$ 按论文取值时生成正常的图像，$`\sigma_t=0`$ 时只得到一片几乎没有细节的色块。下面用一个可以精确计算的例子说明原因。

数据是二维平面上的三团高斯，团中心在半径为 1 的圆上，每团每个方向的标准差是 0.1。这时最优的噪声预测器有闭式解，不需要训练。从同一批 3000 个 $`x_T`$ 出发，比较三种采样方法（图 3）：

1. DDPM，$`\sigma_t^2=\beta_t`$：团内标准差约 0.10，团中心到原点的距离约 1.00，与训练数据一致。
2. 去掉 $`\sigma_t z`$，只做均值更新：每团缩成一个小点，团内标准差约 0.014；三个小点还被拉向中心，到原点的距离约 0.90。样本失去了多样性，又偏向“平均”。
3. DDIM（$`\eta=0`$，Song et al., 2021）：同样不加任何噪声，但用的是另一种更新式。团内标准差约 0.10，团中心距离约 1.00，与数据一致。

可见问题不在“确定性”本身，而在于把为随机采样设计的 DDPM 均值更新单独拿来用，又去掉了与之配套的噪声。每一步均值更新都把样本往密度高处拉，没有噪声把它们重新推散，迭代 1000 次后样本就挤到了一起。三种方法下三团各占约三分之一，都没有丢失某一团。

![图 3：三团二维高斯上的三种采样方法，都从同一批 3000 个纯噪声出发。去掉 σt z 后，每团缩成一个小点，并被拉向中心。](figures/l1_fig3_samplers.png)

**联系**：DDIM 的推导是推导题 D8。

### 4.6 文字的扩散

**要点**：文字是离散的，直接给“你好吗”加高斯噪声没有意义。讲义列出了两条路：

1. 在连续的潜空间里加噪声：Diffusion-LM（Li et al., 2022）对词向量做扩散，最后取最近的词，还能用分类器的梯度控制生成结果；DiffuSeq（Gong et al., 2023）把这种做法用于序列到序列的任务，只对目标序列加噪。
2. 不加高斯噪声：DiffusER（Reid et al., 2022）用编辑操作破坏文本，再学习用编辑操作把它还原；D3PM（Austin et al., 2021）在离散状态上定义扩散，例如逐步把词换成 [MASK]，再逐步填回。

**补充**：语音也可以做扩散。WaveGrad（Chen et al., 2020）以梅尔频谱为条件，从白噪声出发逐步生成波形。

### 4.7 Mask-Predict 与 MaskGIT

**要点**：

1. Mask-Predict（Ghazvininejad et al., 2019）是一种非自回归的文本生成方法：先让所有位置都是 [MASK]，并行预测每个位置的字；保留把握大的，把把握小的重新换成 [MASK] 再预测；重复固定的次数。课上的例子是一个有两个正确答案的问题，答“演员”或“老师”都对。如果各位置独立地取最可能的字，可能拼出“演师”“老员”这样的错误组合，这与 L0 第 1.1 节黑狗、白狗的问题是同一回事；反复遮住、重填可以逐步消除这种不一致。
2. MaskGIT（Chang et al., 2022）把同样的做法用到图像上：先用 VQ 编码器把图像变成离散的视觉 token，训练一个双向 Transformer 预测被遮住的 token；生成时从全部遮住出发，每一轮保留把握最大的预测，其余按余弦调度重新遮住，256×256 分辨率下共解码 8 轮。Muse（Chang et al., 2023）把它扩展到文生图。

老师的看法是，这类“遮住、填上、再遮住”的反复修正，与扩散模型“加噪、去噪”的反复修正很像，只是把高斯噪声换成了 [MASK]。

## 5 小结：与本系列推导篇的对应

| 本文 | 课上的结论 | 推导在哪里 |
|---|---|---|
| 2.2 | 最大似然等价于最小化 $`\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$ | [P0 第 3.6 节](P0-probability-and-information.md#36-交叉熵)：交叉熵等于熵加 KL |
| 2.4、2.6 | 对数似然不小于 ELBO | [D2–D3 第 4、5 章](D2-D3-elbo-and-vae-loss.md#4-d2a-jensen-推法)；[R1 定理 1](R1-ddpm-paper-walkthrough.md#31-引理-1-与定理-1) |
| 3.1 | 前向过程一步到位 | [D0(c) 噪声合并](D0-D1-gaussian-and-kl.md#3-d0c-噪声合并)；[R1 命题 1](R1-ddpm-paper-walkthrough.md#24-命题-1前向过程一步到位)；推导题 D4（后续） |
| 3.2 | 下界拆成三项，第二项与 $`\theta`$ 无关 | [R1 定理 2](R1-ddpm-paper-walkthrough.md#32-定理-2三项式)；推导题 D6（后续） |
| 3.3 | 后验 $`q(x_{t-1}\mid x_t,x_0)`$ 是高斯 | [D0–D1 第 5 章](D0-D1-gaussian-and-kl.md#5-d0a-线性高斯模型)；[R1 定理 3](R1-ddpm-paper-walkthrough.md#4-定理-3前向过程的后验)；推导题 D5（后续） |
| 3.4 | 方差固定时，KL 只取决于均值之差 | [D0–D1 第 6.3 节](D0-D1-gaussian-and-kl.md#63-两个特例-c)；[R1 定理 4](R1-ddpm-paper-walkthrough.md#51-引理-2-与定理-4) |
| 3.5 | 网络只需预测噪声 | [R1 定理 5](R1-ddpm-paper-walkthrough.md#52-定理-5ε-参数化)；推导题 D7（后续） |
| 4.5 | DDIM 的确定性采样 | [R1 命题 7](R1-ddpm-paper-walkthrough.md#64-命题-7采样为什么要加噪声)；推导题 D8（后续） |

## 6 自测题

做完再看第 7 章的答案。

1. 训练算法为什么只加一次噪声，而不必真的逐步加 $`t`$ 次？
2. 设 $`\beta_1=0.1`$、$`\beta_2=0.2`$，写出 $`x_2`$ 关于 $`x_0`$ 的一步公式。
3. 最大似然为什么等价于最小化 $`\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$？推导中哪一项与 $`\theta`$ 无关？
4. VAE 为什么不把 $`P_\theta(x\mid z)`$ 定义成“$`G(z)=x`$ 时为 1，否则为 0”？
5. DDPM 下界的三项中，哪一项可以在训练时忽略？为什么？
6. $`t=500`$ 时，后验均值中 $`x_0`$ 和 $`x_t`$ 的系数约是多少？这说明了什么？
7. 为什么最小化去噪匹配项只需要让均值靠近？
8. 采样时去掉 $`\sigma_t z`$ 会怎样？DDIM 也不加噪声，为什么没有这个问题？

## 7 自测答案

1. 各步加入的是独立的高斯噪声，它们的和仍是高斯噪声，可以合并成一份。所以第 $`t`$ 步的带噪图像可以由 $`x_0`$ 一步算出：$`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`$。
2. $`x_2=\sqrt{0.72}\,x_0+\sqrt{0.28}\,\epsilon\approx0.8485\,x_0+0.5292\,\epsilon`$，其中 $`\epsilon\sim\mathcal{N}(0,I)`$。
3. 对数似然的样本平均近似于期望 $`\mathbb{E}_{P_{\mathrm{data}}}[\log P_\theta]`$，而这个期望等于 $`-H(P_{\mathrm{data}})-\mathrm{KL}(P_{\mathrm{data}}\Vert P_\theta)`$。熵 $`H(P_{\mathrm{data}})`$ 与 $`\theta`$ 无关，所以最大化似然就是最小化 KL。
4. 那样的话，几乎没有 $`z`$ 能恰好生成一张给定的真实图像，$`P_\theta(x)`$ 几乎处处为 0，无法用来训练。把 $`G(z)`$ 看成高斯分布的均值，离 $`x`$ 越近概率越大，似然才有意义。
5. 先验匹配项 $`\mathrm{KL}(q(x_T\mid x_0)\Vert P(x_T))`$。两边都没有参数，与 $`\theta`$ 无关，而且它本身几乎为 0。
6. 约 0.0031 和 0.9941。$`t`$ 大时后验均值几乎就是 $`x_t`$，每一步只去掉很少的噪声。
7. 两边的方差都事先固定，只有网络输出的均值可以调整。方差相同的两个高斯之间的 KL 等于均值差的平方除以 $`2\sigma^2`$；方差不同时只多一个与均值无关的常数。所以只需让均值靠近。
8. 在第 4.5 节的例子中，样本缩成三个小点（团内标准差约 0.014，真实值为 0.10），并被拉向中心。DDIM 用的是另一种更新式，不加噪声也能得到与数据一致的分布；问题出在把为随机采样设计的 DDPM 均值更新单独拿来用。

## 参考资料

课程与视频：

1. 李宏毅. 机器学习 2023 春（课程页）. <https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php> 。4/7 条目“Diffusion Model 原理剖析 (optional)”下有本文的四段视频和讲义 PDF。
2. 【生成式AI】Diffusion Model 原理剖析 (1/4) (optional). <https://youtu.be/ifCDXFdeaaM>
3. 【生成式AI】Diffusion Model 原理剖析 (2/4) (optional). <https://youtu.be/73qwu77ZsTM>
4. 【生成式AI】Diffusion Model 原理剖析 (3/4) (optional). <https://youtu.be/m6QchXTx6wA>
5. 【生成式AI】Diffusion Model 原理剖析 (4/4) (optional). <https://youtu.be/67_M2qP5ssY>

论文：

6. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems 33*. arXiv:2006.11239.
7. Luo, C. (2022). Understanding diffusion models: A unified perspective. arXiv:2208.11970. 课程页将其列为选读。
8. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR 2014*. arXiv:1312.6114.
9. Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR 2021*. arXiv:2010.02502.
10. Holtzman, A., et al. (2020). The curious case of neural text degeneration. *ICLR 2020*. arXiv:1904.09751.
11. Shen, J., et al. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. *ICASSP 2018*. arXiv:1712.05884.
12. Chen, N., et al. (2020). WaveGrad: Estimating gradients for waveform generation. arXiv:2009.00713.
13. Li, X. L., et al. (2022). Diffusion-LM improves controllable text generation. arXiv:2205.14217.
14. Gong, S., et al. (2023). DiffuSeq: Sequence to sequence text generation with diffusion models. *ICLR 2023*. arXiv:2210.08933.
15. Reid, M., et al. (2022). DiffusER: Discrete diffusion via edit-based reconstruction. arXiv:2210.16886.
16. Austin, J., et al. (2021). Structured denoising diffusion models in discrete state-spaces. arXiv:2107.03006.
17. Ghazvininejad, M., et al. (2019). Mask-Predict: Parallel decoding of conditional masked language models. *EMNLP-IJCNLP 2019*, pp. 6112–6121.
18. Chang, H., et al. (2022). MaskGIT: Masked generative image transformer. arXiv:2202.04200.
19. Chang, H., et al. (2023). Muse: Text-to-image generation via masked generative transformers. arXiv:2301.00704.
