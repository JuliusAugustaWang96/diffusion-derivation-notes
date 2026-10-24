# 扩散模型推导笔记

*Diffusion Model Derivation Notes*

本仓库收录扩散模型相关推导的逐步笔记与配套验证代码。每篇笔记都把推导拆成可以单独检验的小步，写明每一步做了什么、依据哪条规则、为什么这样做，并用脚本核对文中的全部数值例子。听课笔记 L0、L1 整理李宏毅《机器学习 2023 春》中与扩散模型有关的 7 段视频，先建立整体图景；L2 整理李宏毅两门课中讲变分自编码器（VAE）的内容；直观篇 I0 把 VAE 的来龙去脉连成一条路线，回答编码器学什么、噪声加在哪里、“混合高斯”指哪一个；论文精读 R0 把 VAE 原论文的推导写成定义、定理与证明，并逐式对照原文，R1 对 DDPM 原论文做同样的事；前置篇 P0 是《深度学习》（花书）第 3 章的读书笔记，整理推导要用到的高斯分布、贝叶斯规则和信息论。

## 内容

| 编号 | 主题 | 文档 | 配套代码 |
|---|---|---|---|
| L0 | 听课笔记：四类图像生成模型、扩散模型怎样生成和训练、文生图的三个模块与 FID、CLIP score | [L0 笔记](docs/L0-image-generation-models.md) | [`l0_l1_check.py`](code/l0_l1_check.py)、[`make_figures_l0_l1.py`](code/make_figures_l0_l1.py) |
| L1 | 听课笔记：DDPM 原理剖析，从最大似然、证据下界到预测噪声，以及采样时为什么要加噪声 | [L1 笔记](docs/L1-ddpm-lectures.md) | [`l0_l1_check.py`](code/l0_l1_check.py)、[`make_figures_l0_l1.py`](code/make_figures_l0_l1.py) |
| L2 | 听课笔记：变分自编码器 VAE，从文字生图缺少的信息、编码器与码中的噪声，到最大似然与证据下界、条件 VAE 和逐像素误差的局限 | [L2 笔记](docs/L2-vae-lectures.md) | [`l2_check.py`](code/l2_check.py)、[`make_figures_l2.py`](code/make_figures_l2.py) |
| I0 | 直观篇：VAE 的路线；编码器近似的是解码器的真实后验，重参数化把噪声加在隐空间，两个“混合”分别在 x 空间和 z 空间（聚合后验）；在 2 维隐空间里把真实后验算出来与编码器比较 | [I0 笔记](docs/I0-vae-intuition.md) | [`i0_check.py`](code/i0_check.py)、[`make_figures_i0.py`](code/make_figures_i0.py)、[`03_vae_latent_mnist.py`](code/03_vae_latent_mnist.py) |
| R0 | 论文精读：《Auto-Encoding Variational Bayes》的推导写成 4 条定理、3 条引理、3 条命题，每条附证明、白话和对照（原论文式号、苏剑林博客、L2、D2–D3）；联合分布的 KL、IWAE 界、AEVB 算法与代码逐行对照 | [R0 笔记](docs/R0-aevb-paper-walkthrough.md) | [`r0_check.py`](code/r0_check.py)、[`make_figures_r0.py`](code/make_figures_r0.py) |
| R1 | 论文精读：《Denoising Diffusion Probabilistic Models》的推导写成 2 条定义、5 条定理、2 条引理、8 条命题，每条附证明、白话和对照（原论文式号、苏剑林《生成扩散模型漫谈》、L1）；前向一步到位、变分界与三项式、前向过程的后验、ε 参数化、简化目标、两种 σ 的来历、采样为什么要加噪；两个算法与代码逐行对照 | [R1 笔记](docs/R1-ddpm-paper-walkthrough.md) | [`r1_check.py`](code/r1_check.py)、[`make_figures_r1.py`](code/make_figures_r1.py)、[`04_ddpm_toy2d.py`](code/04_ddpm_toy2d.py) |
| P0 | 前置：高斯分布、贝叶斯规则、自信息与熵、KL 散度与交叉熵（花书第 3 章读书笔记） | [P0 笔记](docs/P0-probability-and-information.md) | [`00_probability_check.py`](code/00_probability_check.py)、[`make_figures_p0.py`](code/make_figures_p0.py) |
| D0 | 高斯分布预备：线性高斯模型、两个高斯密度相乘、独立噪声合并 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`make_figures.py`](code/make_figures.py) |
| D1 | 两个对角高斯分布之间的 KL 散度及其两个特例 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`01_gaussian_kl_check.py`](code/01_gaussian_kl_check.py) |
| D2 | 证据下界 ELBO：Jensen 推法、分解推法（缺口就是 KL）与“重建项减 KL”的两项形式 | [D2–D3 笔记](docs/D2-D3-elbo-and-vae-loss.md) | [`d2_d3_check.py`](code/d2_d3_check.py)、[`make_figures_d2_d3.py`](code/make_figures_d2_d3.py) |
| D3 | 重参数化与 VAE 的损失：梯度为什么不能直接进期望、KL 的闭式、伯努利解码器与二元交叉熵；在 MNIST 上训练一个 VAE，把损失的每一项对应到代码行 | [D2–D3 笔记](docs/D2-D3-elbo-and-vae-loss.md) | [`02_vae_mnist.py`](code/02_vae_mnist.py)、[`d2_d3_check.py`](code/d2_d3_check.py) |

题号 D0–D14 是本系列推导题的统一编号，P 开头的是前置篇，L 开头的是听课笔记，I 开头的是直观篇，R 开头的是论文精读。目前包含 L0、L1、L2、I0、R0、R1、P0 与 D0–D3（Release v1.0 的 PDF 只含 D0–D1，其余各篇暂无 PDF），后续篇目将陆续加入。建议先读 L0、L1、L2 建立整体图景（L2 讲 VAE，也可以放在 L1 之前读），读完 L2 接着读 I0 理清 VAE 的直观，再读 P0、D0–D1 和 D2–D3；I0 引用了 D2–D3 的结论，第一遍读时可以先跳过其中的公式。读原论文时配合 R0：它把整篇论文的推导收成定理与证明，并标出每条结论在 L2、D2–D3、I0 中的位置。读 DDPM 原论文时配合 R1：它接在 L1 之后，把论文的推导收成定理与命题，第 8 章逐项对照 R0，说明 DDPM 哪里是 VAE 的老朋友、哪里是新东西。

## 阅读方式

- 在线阅读：直接打开 `docs/` 下的 Markdown 文件，GitHub 会渲染其中的公式、表格和插图。公式采用 GitHub 的写法：行内公式写作 `` $`…`$ ``，独立公式写在标记为 `math` 的代码块中。部分本地 Markdown 编辑器不识别这种写法，此时请阅读打印版。
- 打印版：每个版本的 PDF 附在本仓库的 [Releases](../../releases) 页面。

笔记的结构：

- L0、L1、L2 的每一节按“术语、要点、公式、补充、联系”排列，“补充”中的细节都对照原论文核实过，“联系”指向 P0、D0–D1 中的对应位置。L0 讲图像生成模型与扩散模型的工作方式，L1 讲 DDPM 训练与采样算法背后的数学，L2 讲 VAE 的结构、直观与证据下界；三篇的最后都有小结、自测题及答案。
- I0 是直观篇：第 0 章一页速览，第 1、2 章说明自编码器为什么不能直接生成、解码器为什么就是生成模型本身，第 3、4、5 章分别回答编码器学什么、噪声加在哪里、“混合高斯”指哪一个，第 6 章把训练与生成串起来，第 7 章是常见误解，第 8、9 章是自测题及答案。每个说法都用两个例子核对：一维玩具例子（与 L2 图 5(b) 同一个模型）和 D2–D3 第 8 章在 MNIST 上训练的 VAE。
- R0 是论文精读：第 0 章一页速览（六步链条、恒等式与三处近似），第 1 章对齐论文、苏剑林、李宏毅三套记号，第 2 到 6 章是定义与定理，每条按“正式陈述、证明、白话、对照”排列，第 7 章从定理到 AEVB 算法和代码，第 8 章是初学者十问，第 9 章是论文式号与各篇的总对照表，第 10、11 章是自测题及答案。玩具例子与 L2、D2–D3 相同。
- R1 是 DDPM 原论文的精读：第 0 章一页速览（七步链条、哪些是恒等式、哪些是近似或选择、三种讲法的起点），第 1 章对齐论文、苏剑林、李宏毅三套记号（苏剑林的 α、β 是论文的平方根，p、q 与论文相反），第 2 到 6 章是定义、定理与命题，每条按“正式陈述、证明、白话、对照”排列，第 7 章把论文其余部分一页读完，第 8 章与 R0 逐项对照，第 9 章对照 `04_ddpm_toy2d.py` 的行号，第 10 章是初学者十问，第 11 章是论文式号与各篇的总对照表，第 12、13 章是自测题及答案。文中的数值例子（后验的系数、损失的权重、一个三步的线性高斯玩具上联合 KL 的逐项核对）都由 `r1_check.py` 复算。
- P0 的每一节按“要点、公式、读书批注、补充、联系”排列，公式注明原书式号，“联系”指向 D0–D1 中用到它的位置；第 4 章是小结表，第 5、6 章是自测题及答案。
- D0–D1：第 1 章列出推导所需的 8 条基本规则，每条配一个数值例子；第 2 章给出推导顺序和两个通用套路；第 3–6 章逐题推导，每题附“常见卡点”和“重推骨架”；第 7 章是代码核对与概念小结；第 8、9 章是自测题及答案。
- D2–D3：第 1 章回顾 D0–D1 的 5 条规则，再新增 9 条（T9–T17），编号接着 D0–D1；第 2 章给出推导顺序和第三个套路“凑出 KL”；第 3 章说明为什么需要证据下界；第 4–7 章逐题推导，每题附“常见卡点”和“重推骨架”；第 8 章在 MNIST 上训练一个 VAE，对照公式找出代码中的重参数化、KL 与重建；第 9 章把重推骨架汇总成一页；第 10、11 章是自测题及答案。玩具例子与 L2 第 3 章相同，相同的量数值相同。

## 环境与复现

需要 Python 3.10 或更高版本。已在 Python 3.11、PyTorch 2.11、NumPy 2.1、Matplotlib 3.10 下测试。

```bash
pip install -r requirements.txt
python code/l0_l1_check.py
python code/make_figures_l0_l1.py
python code/l2_check.py
python code/make_figures_l2.py
python code/i0_check.py
python code/make_figures_i0.py
python code/03_vae_latent_mnist.py
python code/r0_check.py
python code/make_figures_r0.py
python code/r1_check.py
python code/make_figures_r1.py
python code/04_ddpm_toy2d.py
python code/00_probability_check.py
python code/make_figures_p0.py
python code/01_gaussian_kl_check.py
python code/make_figures.py
python code/d2_d3_check.py
python code/make_figures_d2_d3.py
python code/02_vae_mnist.py
```

- `l0_l1_check.py` 核对 L0、L1 中出现的全部数字，包括噪声调度、后验的系数、FID 的样本量偏差和三种采样方法的对比；只用 NumPy，约 5 秒完成，最后一行应为“全部通过”。
- `make_figures_l0_l1.py` 生成 `docs/figures/` 下 L0、L1 的 8 张插图（文件名以 `l0_`、`l1_` 开头）。示意图都用代码绘制，示例图像是程序生成的合成风景，随机种子固定，结果可复现。
- `l2_check.py` 核对 L2 中出现的全部数字，包括讲义中 KL 项的两种约定、码空间重叠处的最优输出、无穷多个高斯混合的数值积分、证据下界玩具例子的闭式值与蒙特卡洛估计、重参数化的梯度，以及逐像素误差的例子；只用 NumPy，约 2 秒完成，最后一行应为“全部通过”。
- `make_figures_l2.py` 生成 `docs/figures/` 下 L2 的 7 张插图（文件名以 `l2_` 开头）。示意图都用代码绘制，图中的月亮和数字 7 是程序生成的简单图形，结果可复现。
- `i0_check.py` 核对 I0 中玩具例子的全部数字，包括两个混合、真实后验的平均等于先验、互信息的两种算法、每个数据点使 ELBO 最大的高斯、KL 项拆成互信息与聚合后验的 KL，并核对正文引用的 `02_vae_mnist.py` 行号；只用 NumPy，约 3 秒完成，最后一行应为“全部通过”。
- `make_figures_i0.py` 生成 `docs/figures/` 下 I0 的图 1、图 4、图 5（文件名以 `i0_` 开头），都是示意图或玩具例子的精确计算，结果可完全复现。
- `03_vae_latent_mnist.py` 用与 `02_vae_mnist.py` 相同的设置训练同一个 VAE，在同一台机器的同一设备上得到的模型逐位相同；然后在网格上算出前 1000 张测试图像的真实后验，与编码器比较，并把 6 万张训练图像的编码器输出叠成聚合后验，生成 I0 的图 2、图 3、图 6，打印正文用到的数字。数据与 `02_vae_mnist.py` 共用 `data/mnist/`。测试机上 GPU 用时约 20 秒，CPU 约 1.5 分钟。与 `02_vae_mnist.py` 一样，换一台机器或改用 CPU，训练出的模型会略有不同：I0 正文的数字来自测试机的 GPU；在 CPU 上，前 1000 张测试图像的缺口中位数是 1.57 nat（GPU 上 1.67），聚合后验与先验的 KL 是 0.40 nat（GPU 上 0.36），按规则选出的三张测试图像也不同，结论不变。
- `r0_check.py` 核对 R0 中出现的全部数字，包括证据分解的玩具例子、两项拉锯的最小点、KL 闭式中 $`u-1-\log u`$ 的取值、图 4 中 80 个专属高斯叠加后与先验的 KL、在数据与模型不一致时用二维数值积分逐项核对联合 KL 的两种拆法、两种梯度估计的标准差、IWAE 界随样本数的变化，以及引用的 `02_vae_mnist.py` 行号和 D2–D3 的训练数字；只用 NumPy，约 5 秒完成，最后一行应为“全部通过”。
- `make_figures_r0.py` 生成 `docs/figures/` 下 R0 的图 1、图 4、图 6（文件名以 `r0_` 开头），都是示意图或玩具例子的精确计算，结果可完全复现；R0 的其余插图取自 L2 与 D2–D3。
- `r1_check.py` 核对 R1 中出现的全部数字，包括噪声调度的代表值、两步例子与逐步加噪 1000 步的模拟、先验匹配项的大小、后验系数表与回归核对、式 (7) 代入 $`x_0`$ 后与式 (10) 的一致性、三种参数化之间的因子、式 (12) 的权重、两种 $`\sigma_t^2`$ 的回归核对、三步线性高斯玩具上联合 KL 的逐项数值，并核对正文引用的 `04_ddpm_toy2d.py` 行号；只用 NumPy，约 1 秒完成，最后一行应为“全部通过”。
- `make_figures_r1.py` 生成 `docs/figures/` 下 R1 的图 1、图 2、图 5、图 6（文件名以 `r1_` 开头），都是示意图或精确计算，随机种子固定，结果可复现；R1 的其余插图取自 L1。
- `04_ddpm_toy2d.py` 在二维瑞士卷数据上从零实现 DDPM（R1 第 9 章引用了它的行号）：线性噪声调度、一步到位的前向采样、后验的均值与方差、$`L_{\mathrm{simple}}`$ 训练与算法 2 的采样。瑞士卷由文件开头的函数生成，与 scikit-learn 的 `make_swiss_roll` 逐位相同，不需要安装 scikit-learn。训练 20000 步后 $`L_{\mathrm{simple}}`$ 约 0.23；测试机上 GPU 用时约 37 秒，CPU 约 3 分钟；采样 3000 个点各走 1000 步不到 1 秒。输出 `code/figs/` 下的 4 张图和 `code/ddpm_toy2d.pt`，两者都不提交。
- `00_probability_check.py` 核对 P0 中出现的全部数字，只用 NumPy，约 10 秒完成，最后一行应为“全部通过”。
- `make_figures_p0.py` 生成 `docs/figures/` 下 P0 的 3 张插图（文件名以 `p0_` 开头），图中的最优参数由网格搜索求得，结果可复现。
- `01_gaussian_kl_check.py` 用数值方法核对 D1 的闭式 KL，同时核对本系列后续 D4、D5、D9 的结论。CPU 上约 10 秒完成，最后一行应为“全部通过”。
- `make_figures.py` 重新生成 `docs/figures/` 下 D0–D1 的 4 张插图，随机种子固定，结果可复现。图中使用中文字体，在 Linux 或 macOS 上请先安装 Noto Sans CJK SC 等中文字体。
- `d2_d3_check.py` 核对 D2–D3 中出现的全部数字，包括证据下界玩具例子的闭式值、缺口与 KL、蒙特卡洛估计、两种梯度估计的理论标准差与模拟结果、二元交叉熵的例子，并核对正文引用的 `02_vae_mnist.py` 行号；只用 NumPy，约 1 秒完成，最后一行应为“全部通过”。
- `make_figures_d2_d3.py` 生成 `docs/figures/` 下 D2–D3 的图 1 至图 4（文件名以 `d2d3_fig1` 至 `d2d3_fig4` 开头），都是闭式计算或示意图，结果可完全复现。
- `02_vae_mnist.py` 在 MNIST 上训练一个隐变量 2 维的 VAE（10 个 epoch），生成 D2–D3 的图 5 至图 8。第一次运行会把 MNIST 的原始文件（约 11 MB）下载到 `data/mnist/`，这个目录不提交。测试机上 GPU 用时约 13 秒，CPU 约 20 秒。随机种子固定，但不同硬件上浮点运算的顺序不同，结果的小数点后几位会不同：10 个 epoch 后的负 ELBO 在测试机的 GPU 上是 143.048，在 CPU 上是 143.373。需要 PyTorch，不需要 torchvision。

## 记号

记号按 Ho et al. (2020)。$`\mathcal{N}(\mu,\sigma^2)`$ 的第二个参数是方差，log 指自然对数。P0 沿用《深度学习》原书的记号，例如 KL 散度写作 $`D_{\mathrm{KL}}(P\Vert Q)`$。L0、L1 沿用课程的写法，模型的分布写作 $`P_\theta`$，前向过程的分布写作 $`q`$；L2 沿用 2016 年讲义的写法，先验、解码器、编码器给出的分布分别写作 $`P(z)`$、$`P(x\mid z)`$、$`q(z\mid x)`$，下界写作 $`L_b`$。D2–D3、I0 与 R0 按 Kingma & Welling (2014) 的写法，解码器写作 $`p_\theta(x\mid z)`$，编码器写作 $`q_\phi(z\mid x)`$，下界写作 ELBO；I0 中的聚合后验写作 $`q_\phi(z)`$。R1 与 Ho et al. (2020) 完全相同，约定 $`\bar\alpha_0=1`$，第 1 章给出苏剑林博客记号（他的 $`\alpha_t`$、$`\beta_t`$ 是论文对应量的平方根）与论文记号的换算。

## 目录结构

```
.
├── docs/
│   ├── L0-image-generation-models.md        听课笔记 L0
│   ├── L1-ddpm-lectures.md                  听课笔记 L1
│   ├── L2-vae-lectures.md                   听课笔记 L2
│   ├── I0-vae-intuition.md                  直观篇 I0
│   ├── R0-aevb-paper-walkthrough.md         论文精读 R0
│   ├── R1-ddpm-paper-walkthrough.md         论文精读 R1
│   ├── P0-probability-and-information.md    前置篇正文
│   ├── D0-D1-gaussian-and-kl.md             正文 D0–D1
│   ├── D2-D3-elbo-and-vae-loss.md           正文 D2–D3
│   ├── figures/                             插图，由 code/ 下的绘图脚本、02_vae_mnist.py 和 03_vae_latent_mnist.py 生成
│   └── LICENSE.md                           文档与插图的许可
├── code/
│   ├── l0_l1_check.py                       L0、L1 的数值核对
│   ├── make_figures_l0_l1.py                L0、L1 的绘图
│   ├── l2_check.py                          L2 的数值核对
│   ├── make_figures_l2.py                   L2 的绘图
│   ├── i0_check.py                          I0 的数值核对（玩具例子）
│   ├── make_figures_i0.py                   I0 的绘图（图 1、图 4、图 5）
│   ├── r0_check.py                          R0 的数值核对
│   ├── make_figures_r0.py                   R0 的绘图（图 1、图 4、图 6）
│   ├── r1_check.py                          R1 的数值核对
│   ├── make_figures_r1.py                   R1 的绘图（图 1、图 2、图 5、图 6）
│   ├── 04_ddpm_toy2d.py                     R1 第 9 章对照的 DDPM 玩具实现（瑞士卷）
│   ├── 00_probability_check.py              P0 的数值核对
│   ├── make_figures_p0.py                   P0 的绘图
│   ├── 01_gaussian_kl_check.py              D0–D1 的数值核对
│   ├── make_figures.py                      D0–D1 的绘图
│   ├── d2_d3_check.py                       D2–D3 的数值核对
│   ├── make_figures_d2_d3.py                D2–D3 的绘图（图 1 至图 4）
│   ├── 02_vae_mnist.py                      D2–D3 的 VAE 训练（图 5 至图 8）
│   └── 03_vae_latent_mnist.py               I0 的隐空间分析（图 2、图 3、图 6）
├── requirements.txt
└── LICENSE                                  代码的许可
```

## 许可

- 代码（`code/`）采用 MIT License，见 [LICENSE](LICENSE)。
- 文档与插图（`docs/`）采用知识共享署名 4.0 国际许可协议（CC BY 4.0），见 [docs/LICENSE.md](docs/LICENSE.md)。
- L0、L1、L2 是听课笔记：文字是笔记作者自己的概括，插图由代码重新绘制。课程视频与讲义的版权属于原作者，本仓库不包含讲义截图；视频与讲义的链接见各篇的参考资料。I0、R0 与 R1 概括了苏剑林的博客与李宏毅的课程，同样没有照录原文，链接见各篇的参考资料；R0、R1 按原论文的编号引用公式。
- D2–D3 的图 5 至图 8、I0 的图 2、图 3、图 6 含有 MNIST 测试集中的手写数字图像，或由在 MNIST 上训练的模型生成。MNIST 数据集的版权与使用条件以其发布方为准（LeCun et al., 1998）；本仓库不包含数据集本身，运行脚本时从公开镜像下载。

## 勘误与反馈

如发现推导或代码中的错误，欢迎通过 Issues 提出，请注明章节号或代码行号。

## 参考文献

- 李宏毅. 机器学习 2023 春（课程页）. <https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php> 。L0 依据 3/24 条目下的三段视频，L1 依据 4/7 条目下的《Diffusion Model 原理剖析》四段视频。
- 李宏毅. 生成式AI导论 2024（课程页）. <https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php> 。L2 第 1 章依据 5/31 条目下的第 18 讲。
- 李宏毅. 机器学习 2016 秋（课程页）. <https://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html> 。L2 第 2–4 章依据 2016/12/02 条目下的 VAE 一讲。
- 苏剑林 (2018). 变分自编码器（一）：原来是这么一回事；（二）：从贝叶斯观点出发；（三）：这样做为什么能成？ 科学空间. <https://kexue.fm/archives/5253> 、<https://kexue.fm/archives/5343> 、<https://kexue.fm/archives/5383> 。I0 依据这三篇博客。
- 苏剑林 (2022). 生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼；（二）：DDPM = 自回归式VAE；（三）：DDPM = 贝叶斯 + 去噪；（四）：DDIM = 高观点DDPM. 科学空间. <https://kexue.fm/archives/9119> 、<https://kexue.fm/archives/9152> 、<https://kexue.fm/archives/9164> 、<https://kexue.fm/archives/9181> 。R1 依据这四篇博客。
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML 2015*, pp. 2256–2265. R1 中前向过程、变分界与三项式的出处。
- Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems 32*, pp. 11895–11907. R1 第 7 章与得分匹配的联系。
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 1661–1674.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR 2021*. arXiv:2010.02502. L1 第 4.5 节与 R1 第 10 章的 DDIM。
- Vahdat, A., & Kautz, J. (2020). NVAE: A deep hierarchical variational autoencoder. *Advances in Neural Information Processing Systems 33*. arXiv:2007.03898. R1 第 8 章。
- Hoffman, M. D., & Johnson, M. J. (2016). ELBO surgery: Yet another way to carve up the variational evidence lower bound. *NIPS 2016 Workshop on Advances in Approximate Bayesian Inference*. I0 第 5.3 节的分解。
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). Importance weighted autoencoders. *International Conference on Learning Representations*. arXiv:1509.00519. R0 第 7.4 节的 IWAE 界。
- Luo, C. (2022). Understanding diffusion models: A unified perspective. arXiv:2208.11970.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. 中译本《深度学习》，人民邮电出版社，2017。P0 依据第 3 章。
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems 33*. arXiv:2006.11239.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations*. arXiv:1312.6114.
- Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. *Foundations and Trends in Machine Learning*, 12(4), 307–392. arXiv:1906.02691.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. D2–D3 第 8 章使用的 MNIST 数据集。
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

## English summary

Step-by-step derivation notes on diffusion models, written in Chinese, with scripts that check every numerical example in the text. The lecture notes L0 and L1 summarize seven videos on image generation and diffusion models from Hung-yi Lee's Machine Learning course (NTU, spring 2023): L0 covers the four families of image generators, how a diffusion model generates and is trained, and the common structure of text-to-image systems; L1 covers the mathematics behind the DDPM training and sampling algorithms, from maximum likelihood and the evidence lower bound to noise prediction, plus a small exactly computable experiment on why sampling adds noise. The lecture note L2 covers Hung-yi Lee's lectures on the variational auto-encoder (the first half of lecture 18 of Introduction to Generative AI, NTU, spring 2024, and the 2016 Machine Learning lecture on VAE that it links to): the information a text prompt leaves out, the encoder and the noise added to the code, maximum likelihood and the evidence lower bound, conditional VAEs, and why a pixel-wise error favors blurred outputs. The text of the lecture notes is the note author's own summary, all figures are drawn by code, and no lecture slides are included. The intuition note I0 lays out the VAE as one route and answers three common questions: the encoder learns an approximation of the decoder's true posterior; the reparameterization trick adds the noise in the latent space, between encoder and decoder; and the "Gaussian mixture" of the lectures lives in data space, while the mixture of the encoder's outputs over the training set is the aggregate posterior in latent space, which the KL term pulls towards the prior. Every claim is checked on an exactly computable 1-D toy model and on the 2-D MNIST VAE of D3, whose true posterior is computed on a grid and compared with the encoder. The paper walkthrough R0 restates the derivations of Auto-Encoding Variational Bayes as four theorems, three lemmas and three propositions, each with a compact proof, a plain-language reading and pointers to the paper's equation numbers, to Su Jianlin's blog posts and to L2 and D2-D3; it covers the joint-distribution form of the objective, the importance-weighted bound and the line-by-line correspondence between equation (10) and the code. The paper walkthrough R1 does the same for Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020): two definitions, five theorems, two lemmas and eight propositions cover the one-shot forward formula, the variational bound and its three-term decomposition, the Gaussian forward-process posterior proved by completing the square, the noise-prediction parameterization, the simplified objective, the origin of the two choices of the reverse variance and why sampling must sample; chapter 1 reconciles the paper's notation with Su Jianlin's blog series and Hung-yi Lee's lectures, chapter 8 compares DDPM with the VAE of R0 item by item, and chapter 9 maps every step to the lines of a from-scratch DDPM on a two-dimensional swiss roll. Release v1.0 covers D0 (Gaussian preliminaries: the linear Gaussian model, the product of two Gaussian densities, and merging independent Gaussian noise) and D1 (the KL divergence between diagonal Gaussians). D2 derives the evidence lower bound twice, with Jensen's inequality and as an exact decomposition whose gap is a KL divergence, and rewrites it as a reconstruction term minus a KL term; D3 covers the reparameterization trick and the VAE loss (the closed-form KL, a Bernoulli decoder and the binary cross-entropy). A small VAE is trained on MNIST and every term of its loss is located in the code. The prerequisite note P0 is a reading note on Chapter 3 of *Deep Learning* (Goodfellow, Bengio & Courville): the Gaussian distribution, Bayes' rule, self-information and entropy, the KL divergence and cross-entropy, with its own check script and redrawn figures. The code is released under the MIT License; the documents and figures are released under CC BY 4.0.
