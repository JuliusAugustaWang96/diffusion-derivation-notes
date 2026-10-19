# 扩散模型推导笔记

*Diffusion Model Derivation Notes*

本仓库收录扩散模型相关推导的逐步笔记与配套验证代码。每篇笔记都把推导拆成可以单独检验的小步，写明每一步做了什么、依据哪条规则、为什么这样做，并用脚本核对文中的全部数值例子。前置篇 P0 是《深度学习》（花书）第 3 章的读书笔记，整理推导要用到的高斯分布、贝叶斯规则和信息论。

## 内容

| 编号 | 主题 | 文档 | 配套代码 |
|---|---|---|---|
| P0 | 前置：高斯分布、贝叶斯规则、自信息与熵、KL 散度与交叉熵（花书第 3 章读书笔记） | [P0 笔记](docs/P0-probability-and-information.md) | [`00_probability_check.py`](code/00_probability_check.py)、[`make_figures_p0.py`](code/make_figures_p0.py) |
| D0 | 高斯分布预备：线性高斯模型、两个高斯密度相乘、独立噪声合并 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`make_figures.py`](code/make_figures.py) |
| D1 | 两个对角高斯分布之间的 KL 散度及其两个特例 | [D0–D1 笔记](docs/D0-D1-gaussian-and-kl.md) | [`01_gaussian_kl_check.py`](code/01_gaussian_kl_check.py) |

题号 D0–D14 是本系列推导题的统一编号，P 开头的是前置篇。目前包含 P0、D0 与 D1（Release v1.0 的 PDF 只含 D0–D1），后续篇目将陆续加入。建议先读 P0，再读 D0–D1。

## 阅读方式

- 在线阅读：直接打开 `docs/` 下的 Markdown 文件，GitHub 会渲染其中的公式、表格和插图。公式采用 GitHub 的写法：行内公式写作 `` $`…`$ ``，独立公式写在标记为 `math` 的代码块中。部分本地 Markdown 编辑器不识别这种写法，此时请阅读打印版。
- 打印版：每个版本的 PDF 附在本仓库的 [Releases](../../releases) 页面。

笔记的结构：

- P0 的每一节按“要点、公式、读书批注、补充、联系”排列，公式注明原书式号，“联系”指向 D0–D1 中用到它的位置；第 4 章是小结表，第 5、6 章是自测题及答案。
- D0–D1：第 1 章列出推导所需的 8 条基本规则，每条配一个数值例子；第 2 章给出推导顺序和两个通用套路；第 3–6 章逐题推导，每题附“常见卡点”和“重推骨架”；第 7 章是代码核对与概念小结；第 8、9 章是自测题及答案。

## 环境与复现

需要 Python 3.10 或更高版本。已在 Python 3.11、PyTorch 2.11、NumPy 2.1、Matplotlib 3.10 下测试。

```bash
pip install -r requirements.txt
python code/00_probability_check.py
python code/make_figures_p0.py
python code/01_gaussian_kl_check.py
python code/make_figures.py
```

- `00_probability_check.py` 核对 P0 中出现的全部数字，只用 NumPy，约 10 秒完成，最后一行应为“全部通过”。
- `make_figures_p0.py` 生成 `docs/figures/` 下 P0 的 3 张插图（文件名以 `p0_` 开头），图中的最优参数由网格搜索求得，结果可复现。
- `01_gaussian_kl_check.py` 用数值方法核对 D1 的闭式 KL，同时核对本系列后续 D4、D5、D9 的结论。CPU 上约 10 秒完成，最后一行应为“全部通过”。
- `make_figures.py` 重新生成 `docs/figures/` 下 D0–D1 的 4 张插图，随机种子固定，结果可复现。图中使用中文字体，在 Linux 或 macOS 上请先安装 Noto Sans CJK SC 等中文字体。

## 记号

记号按 Ho et al. (2020)。$`\mathcal{N}(\mu,\sigma^2)`$ 的第二个参数是方差，log 指自然对数。P0 沿用《深度学习》原书的记号，例如 KL 散度写作 $`D_{\mathrm{KL}}(P\Vert Q)`$。

## 目录结构

```
.
├── docs/
│   ├── P0-probability-and-information.md    前置篇正文
│   ├── D0-D1-gaussian-and-kl.md             正文
│   ├── figures/                             插图，由 code/ 下的两个绘图脚本生成
│   └── LICENSE.md                           文档与插图的许可
├── code/
│   ├── 00_probability_check.py              P0 的数值核对
│   ├── make_figures_p0.py                   P0 的绘图
│   ├── 01_gaussian_kl_check.py              D0–D1 的数值核对
│   └── make_figures.py                      D0–D1 的绘图
├── requirements.txt
└── LICENSE                                  代码的许可
```

## 许可

- 代码（`code/`）采用 MIT License，见 [LICENSE](LICENSE)。
- 文档与插图（`docs/`）采用知识共享署名 4.0 国际许可协议（CC BY 4.0），见 [docs/LICENSE.md](docs/LICENSE.md)。

## 勘误与反馈

如发现推导或代码中的错误，欢迎通过 Issues 提出，请注明章节号或代码行号。

## 参考文献

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. 中译本《深度学习》，人民邮电出版社，2017。P0 依据第 3 章。
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems 33*. arXiv:2006.11239.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations*. arXiv:1312.6114.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

## English summary

Step-by-step derivation notes on diffusion models, written in Chinese, with scripts that check every numerical example in the text. Release v1.0 covers D0 (Gaussian preliminaries: the linear Gaussian model, the product of two Gaussian densities, and merging independent Gaussian noise) and D1 (the KL divergence between diagonal Gaussians). The prerequisite note P0 is a reading note on Chapter 3 of *Deep Learning* (Goodfellow, Bengio & Courville): the Gaussian distribution, Bayes' rule, self-information and entropy, the KL divergence and cross-entropy, with its own check script and redrawn figures. The code is released under the MIT License; the documents and figures are released under CC BY 4.0.
