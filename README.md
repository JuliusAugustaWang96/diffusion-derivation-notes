# 扩散模型推导笔记

*Diffusion Model Derivation Notes*

本仓库收录扩散模型相关推导的逐步笔记与配套验证代码。每篇笔记都把推导拆成可以单独检验的小步，写明每一步做了什么、依据哪条规则、为什么这样做，并用脚本核对文中的全部数值例子。听课笔记 L0、L1 整理李宏毅《机器学习 2023 春》中与扩散模型有关的 7 段视频，先建立整体图景；L2 整理李宏毅两门课中讲变分自编码器（VAE）的内容；前置篇 P0 是《深度学习》（花书）第 3 章的读书笔记，整理推导要用到的高斯分布、贝叶斯规则和信息论。

## 内容

| 编号 | 主题 | 文档 | 配套代码 |
|---|---|---|---|
| L0 | 听课笔记：四类图像生成模型、扩散模型怎样生成和训练、文生图的三个模块与 FID、CLIP score | [L0 笔记](docs/L0-image-generation-models.md) | [`l0_l1_check.py`](code/l0_l1_check.py)、[`make_figures_l0_l1.py`](code/make_figures_l0_l1.py) |
| L1 | 听课笔记：DDPM 原理剖析，从最大似然、证据下界到预测噪声，以及采样时为什么要加噪声 | [L1 笔记](docs/L1-ddpm-lectures.md) | [`l0_l1_check.py`](code/l0_l1_check.py)、[`make_figures_l0_l1.py`](code/make_figures_l0_l1.py) |
| L2 | 听课笔记：变分自编码器 VAE，从文字生图缺少的信息、编码器与码中的噪声，到最大似然与证据下界、条件 VAE 和逐像素误差的局限 | [L2 笔记](docs/L2-vae-lectures.md) | [`l2_check.py`](code/l2_check.py)、[`make_figures_l2.py`](code/make_figures_l2.py) |
| P0 | 前置：高斯分布、贝叶斯规则、自信息与熵、KL 散度与交叉熵（花书第 3 章读书笔记） | [P0 笔记](docs/P0-probability-and-information.md) | [`00_probability_check.py`](code/00_probability_check.py)、[`make_figures_p0.py`](code/make_figures_p0.py) |
| D0 | 高斯分布预备：线性高斯模型、两个高斯密度相乘、独立噪声合并 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`make_figures.py`](code/make_figures.py) |
| D1 | 两个对角高斯分布之间的 KL 散度及其两个特例 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`01_gaussian_kl_check.py`](code/01_gaussian_kl_check.py) |
| D2 | 证据下界 ELBO：Jensen 推法、分解推法（缺口就是 KL）与“重建项减 KL”的两项形式 | [D2–D3 笔记](docs/D2-D3-elbo-and-vae-loss.md) | [`d2_d3_check.py`](code/d2_d3_check.py)、[`make_figures_d2_d3.py`](code/make_figures_d2_d3.py) |
| D3 | 重参数化与 VAE 的损失：梯度为什么不能直接进期望、KL 的闭式、伯努利解码器与二元交叉熵；在 MNIST 上训练一个 VAE，把损失的每一项对应到代码行 | [D2–D3 笔记](docs/D2-D3-elbo-and-vae-loss.md) | [`02_vae_mnist.py`](code/02_vae_mnist.py)、[`d2_d3_check.py`](code/d2_d3_check.py) |

题号 D0–D14 是本系列推导题的统一编号，P 开头的是前置篇，L 开头的是听课笔记。目前包含 L0、L1、L2、P0 与 D0–D3（Release v1.0 的 PDF 只含 D0–D1，其余各篇暂无 PDF），后续篇目将陆续加入。建议先读 L0、L1、L2 建立整体图景（L2 讲 VAE，也可以放在 L1 之前读），再读 P0、D0–D1 和 D2–D3。

## 阅读方式

- 在线阅读：直接打开 `docs/` 下的 Markdown 文件，GitHub 会渲染其中的公式、表格和插图。公式采用 GitHub 的写法：行内公式写作 `` $`…`$ ``，独立公式写在标记为 `math` 的代码块中。部分本地 Markdown 编辑器不识别这种写法，此时请阅读打印版。
- 打印版：每个版本的 PDF 附在本仓库的 [Releases](../../releases) 页面。

笔记的结构：

- L0、L1、L2 的每一节按“术语、要点、公式、补充、联系”排列，“补充”中的细节都对照原论文核实过，“联系”指向 P0、D0–D1 中的对应位置。L0 讲图像生成模型与扩散模型的工作方式，L1 讲 DDPM 训练与采样算法背后的数学，L2 讲 VAE 的结构、直观与证据下界；三篇的最后都有小结、自测题及答案。
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
- `00_probability_check.py` 核对 P0 中出现的全部数字，只用 NumPy，约 10 秒完成，最后一行应为“全部通过”。
- `make_figures_p0.py` 生成 `docs/figures/` 下 P0 的 3 张插图（文件名以 `p0_` 开头），图中的最优参数由网格搜索求得，结果可复现。
- `01_gaussian_kl_check.py` 用数值方法核对 D1 的闭式 KL，同时核对本系列后续 D4、D5、D9 的结论。CPU 上约 10 秒完成，最后一行应为“全部通过”。
- `make_figures.py` 重新生成 `docs/figures/` 下 D0–D1 的 4 张插图，随机种子固定，结果可复现。图中使用中文字体，在 Linux 或 macOS 上请先安装 Noto Sans CJK SC 等中文字体。
- `d2_d3_check.py` 核对 D2–D3 中出现的全部数字，包括证据下界玩具例子的闭式值、缺口与 KL、蒙特卡洛估计、两种梯度估计的理论标准差与模拟结果、二元交叉熵的例子，并核对正文引用的 `02_vae_mnist.py` 行号；只用 NumPy，约 1 秒完成，最后一行应为“全部通过”。
- `make_figures_d2_d3.py` 生成 `docs/figures/` 下 D2–D3 的图 1 至图 4（文件名以 `d2d3_fig1` 至 `d2d3_fig4` 开头），都是闭式计算或示意图，结果可完全复现。
- `02_vae_mnist.py` 在 MNIST 上训练一个隐变量 2 维的 VAE（10 个 epoch），生成 D2–D3 的图 5 至图 8。第一次运行会把 MNIST 的原始文件（约 11 MB）下载到 `data/mnist/`，这个目录不提交。测试机上 GPU 用时约 13 秒，CPU 约 20 秒。随机种子固定，但不同硬件上浮点运算的顺序不同，结果的小数点后几位会不同：10 个 epoch 后的负 ELBO 在测试机的 GPU 上是 143.048，在 CPU 上是 143.373。需要 PyTorch，不需要 torchvision。

## 记号

记号按 Ho et al. (2020)。$`\mathcal{N}(\mu,\sigma^2)`$ 的第二个参数是方差，log 指自然对数。P0 沿用《深度学习》原书的记号，例如 KL 散度写作 $`D_{\mathrm{KL}}(P\Vert Q)`$。L0、L1 沿用课程的写法，模型的分布写作 $`P_\theta`$，前向过程的分布写作 $`q`$；L2 沿用 2016 年讲义的写法，先验、解码器、编码器给出的分布分别写作 $`P(z)`$、$`P(x\mid z)`$、$`q(z\mid x)`$，下界写作 $`L_b`$。D2–D3 按 Kingma & Welling (2014) 的写法，解码器写作 $`p_\theta(x\mid z)`$，编码器写作 $`q_\phi(z\mid x)`$，下界写作 ELBO。

## 目录结构

```
.
├── docs/
│   ├── L0-image-generation-models.md        听课笔记 L0
│   ├── L1-ddpm-lectures.md                  听课笔记 L1
│   ├── L2-vae-lectures.md                   听课笔记 L2
│   ├── P0-probability-and-information.md    前置篇正文
│   ├── D0-D1-gaussian-and-kl.md             正文 D0–D1
│   ├── D2-D3-elbo-and-vae-loss.md           正文 D2–D3
│   ├── figures/                             插图，由 code/ 下的绘图脚本和 02_vae_mnist.py 生成
│   └── LICENSE.md                           文档与插图的许可
├── code/
│   ├── l0_l1_check.py                       L0、L1 的数值核对
│   ├── make_figures_l0_l1.py                L0、L1 的绘图
│   ├── l2_check.py                          L2 的数值核对
│   ├── make_figures_l2.py                   L2 的绘图
│   ├── 00_probability_check.py              P0 的数值核对
│   ├── make_figures_p0.py                   P0 的绘图
│   ├── 01_gaussian_kl_check.py              D0–D1 的数值核对
│   ├── make_figures.py                      D0–D1 的绘图
│   ├── d2_d3_check.py                       D2–D3 的数值核对
│   ├── make_figures_d2_d3.py                D2–D3 的绘图（图 1 至图 4）
│   └── 02_vae_mnist.py                      D2–D3 的 VAE 训练（图 5 至图 8）
├── requirements.txt
└── LICENSE                                  代码的许可
```

## 许可

- 代码（`code/`）采用 MIT License，见 [LICENSE](LICENSE)。
- 文档与插图（`docs/`）采用知识共享署名 4.0 国际许可协议（CC BY 4.0），见 [docs/LICENSE.md](docs/LICENSE.md)。
- L0、L1、L2 是听课笔记：文字是笔记作者自己的概括，插图由代码重新绘制。课程视频与讲义的版权属于原作者，本仓库不包含讲义截图；视频与讲义的链接见各篇的参考资料。
- D2–D3 的图 5 至图 8 含有 MNIST 测试集中的手写数字图像，或由在 MNIST 上训练的模型生成。MNIST 数据集的版权与使用条件以其发布方为准（LeCun et al., 1998）；本仓库不包含数据集本身，运行脚本时从公开镜像下载。

## 勘误与反馈

如发现推导或代码中的错误，欢迎通过 Issues 提出，请注明章节号或代码行号。

## 参考文献

- 李宏毅. 机器学习 2023 春（课程页）. <https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php> 。L0 依据 3/24 条目下的三段视频，L1 依据 4/7 条目下的《Diffusion Model 原理剖析》四段视频。
- 李宏毅. 生成式AI导论 2024（课程页）. <https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php> 。L2 第 1 章依据 5/31 条目下的第 18 讲。
- 李宏毅. 机器学习 2016 秋（课程页）. <https://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html> 。L2 第 2–4 章依据 2016/12/02 条目下的 VAE 一讲。
- Luo, C. (2022). Understanding diffusion models: A unified perspective. arXiv:2208.11970.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. 中译本《深度学习》，人民邮电出版社，2017。P0 依据第 3 章。
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems 33*. arXiv:2006.11239.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations*. arXiv:1312.6114.
- Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. *Foundations and Trends in Machine Learning*, 12(4), 307–392. arXiv:1906.02691.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. D2–D3 第 8 章使用的 MNIST 数据集。
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

## English summary

Step-by-step derivation notes on diffusion models, written in Chinese, with scripts that check every numerical example in the text. The lecture notes L0 and L1 summarize seven videos on image generation and diffusion models from Hung-yi Lee's Machine Learning course (NTU, spring 2023): L0 covers the four families of image generators, how a diffusion model generates and is trained, and the common structure of text-to-image systems; L1 covers the mathematics behind the DDPM training and sampling algorithms, from maximum likelihood and the evidence lower bound to noise prediction, plus a small exactly computable experiment on why sampling adds noise. The lecture note L2 covers Hung-yi Lee's lectures on the variational auto-encoder (the first half of lecture 18 of Introduction to Generative AI, NTU, spring 2024, and the 2016 Machine Learning lecture on VAE that it links to): the information a text prompt leaves out, the encoder and the noise added to the code, maximum likelihood and the evidence lower bound, conditional VAEs, and why a pixel-wise error favors blurred outputs. The text of the lecture notes is the note author's own summary, all figures are drawn by code, and no lecture slides are included. Release v1.0 covers D0 (Gaussian preliminaries: the linear Gaussian model, the product of two Gaussian densities, and merging independent Gaussian noise) and D1 (the KL divergence between diagonal Gaussians). D2 derives the evidence lower bound twice, with Jensen's inequality and as an exact decomposition whose gap is a KL divergence, and rewrites it as a reconstruction term minus a KL term; D3 covers the reparameterization trick and the VAE loss (the closed-form KL, a Bernoulli decoder and the binary cross-entropy). A small VAE is trained on MNIST and every term of its loss is located in the code. The prerequisite note P0 is a reading note on Chapter 3 of *Deep Learning* (Goodfellow, Bengio & Courville): the Gaussian distribution, Bayes' rule, self-information and entropy, the KL divergence and cross-entropy, with its own check script and redrawn figures. The code is released under the MIT License; the documents and figures are released under CC BY 4.0.
