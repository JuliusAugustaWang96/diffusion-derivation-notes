# -*- coding: utf-8 -*-
"""Figures for the paper walkthrough R0 (docs/R0-aevb-paper-walkthrough.md). Schematics and a 1-D toy model only.
Usage: python code/make_figures_r0.py [outdir]      (default: docs/figures)
r0_fig1 the chain from log p(x) to the two lines of code, with the identities and the three approximations marked
r0_fig4 one dedicated Gaussian per training sample: (a) the usual textbook picture, (b) one posterior per sample,
        (c) 80 dedicated Gaussians pulled towards N(0, I) add up to roughly N(0, I)
r0_fig6 the tug of war between the reconstruction term and the KL term in the toy model of D2-D3 and L2
        (p(z) = N(0,1), p(x|z) = N(2z, 1), x = 3), and the variance part u - 1 - log u of the closed-form KL
Figures 2, 3, 5 and 7 of R0 are reused from L2 and D2-D3. The numbers quoted in R0 are checked by code/r0_check.py.
PNG, 200 dpi, 16 cm wide."""
import sys, math, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC")
installed = {f.name for f in font_manager.fontManager.ttflist}
cjk = [name for name in CJK_FONTS if name in installed]      # only installed fonts: no per-glyph warnings
if not cjk:
    print("No Chinese font found; install e.g. Noto Sans CJK SC, otherwise Chinese labels show as boxes.")
plt.rcParams.update({"font.family": cjk + ["DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "mathtext.fontset": "dejavusans"})
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
W = 16 * CM
BLUE, CORAL, TEAL, GRAY, PURPLE = "#185FA5", "#D85A30", "#0F6E56", "#888780", "#6B4C9A"
# toy model of D2-D3 and L2: p(z) = N(0,1), p(x|z) = N(a z, s0^2), observed x
A_, S0_, X_ = 2.0, 1.0, 3.0
POST_VAR = 1.0 / (1.0 + A_ ** 2 / S0_)                 # 0.2
POST_MEAN = POST_VAR * A_ * X_ / S0_                   # 1.2


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, **kw):
    fig.savefig(OUT / name, dpi=200, **kw)
    plt.close(fig)
    print("  saved", OUT / name)


def recon_neg(m, s2):
    """-E_q[log p(x|z)] for q = N(m, s2) in the toy model."""
    return 0.5 * ((X_ - A_ * m) ** 2 + A_ ** 2 * s2) / S0_ + 0.5 * np.log(2 * np.pi * S0_)


def kl_std(m, s2):
    """KL(N(m, s2) || N(0, 1))."""
    return 0.5 * (m ** 2 + s2 - 1.0 - np.log(s2))


def log_px():
    var = A_ ** 2 + S0_
    return -X_ ** 2 / (2 * var) - 0.5 * np.log(2 * np.pi * var)


# ---------------------------------------------------------------- r0_fig1: the chain with identities and approximations
def fig_chain():
    fig, ax = plt.subplots(figsize=(W, 12.2 * CM))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12.2)
    ax.axis("off")

    def box(y, title, body, fc="#EEF3FA", ec=BLUE, h=1.45):
        ax.add_patch(FancyBboxPatch((0.4, y), 10.4, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(0.65, y + h - 0.32, title, fontsize=8.5, fontweight="bold", va="center")
        ax.text(0.65, y + 0.52, body, fontsize=8, va="center")

    def arrow(y1, y2, label, color, x=5.6):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))
        ax.text(x + 0.25, (y1 + y2) / 2, label, fontsize=7.5, color=color, va="center")

    def note(y, text, color):
        ax.add_patch(FancyBboxPatch((11.25, y), 4.45, 1.45, boxstyle="round,pad=0.06", fc="white", ec=color, lw=1.0, ls="--"))
        ax.text(11.45, y + 0.725, text, fontsize=7, color=color, va="center")

    ys = [10.45, 7.95, 5.45, 2.95, 0.45]
    box(ys[0], "证据（想最大化，但算不出）",
        r"$\log p_\theta(x)=\log\int p(z)\,p_\theta(x|z)\,dz$，对 z 的高维积分没有闭式", fc="#F4F4F2", ec=GRAY)
    box(ys[1], "证据下界 ELBO　　论文式 (2)",
        r"$\mathcal{L}(\theta,\phi;x)=\mathrm{E}_{q_\phi(z|x)}\left[\log p_\theta(x,z)-\log q_\phi(z|x)\right]$")
    box(ys[2], "两项式　　论文式 (3)，KL 有闭式（附录 B）",
        r"$\mathcal{L}=\mathrm{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]-\mathrm{KL}\left(q_\phi(z|x)\,\Vert\,p(z)\right)$")
    box(ys[3], "单样本估计　　论文式 (7)、(10)",
        r"$\tilde{\mathcal{L}}^{B}=\log p_\theta(x\,|\,\mu+\sigma\odot\varepsilon)-\mathrm{KL}$，$\varepsilon\sim\mathcal{N}(0,I)$")
    box(ys[4], "代码　　02_vae_mnist.py 第 142、137、174 行",
        "loss = BCE(x, dec(z)) + KL，对一个批次取平均后反向传播", fc="#EDF6F2", ec=TEAL)
    arrow(ys[0], ys[1] + 1.45, "恒等式（定理 1）：缺口 = KL(q 对真实后验) ≥ 0", TEAL)
    arrow(ys[1], ys[2] + 1.45, r"恒等式（定理 2）：拆 $p_\theta(x,z)=p(z)\,p_\theta(x|z)$", TEAL)
    arrow(ys[2], ys[3] + 1.45, "近似 ②：重参数化 + 每张图只抽一个 z（无偏，定理 4）", CORAL)
    arrow(ys[3], ys[4] + 1.45, "近似 ③：小批量代替全数据集（无偏，论文式 (8)）", CORAL)
    note(ys[1], "近似 ①：q 只在对角高斯族里选\n缺口一般不为 0，所以 ELBO 是下界\n而不是等于 log p(x)", CORAL)
    note(ys[2], "恒等变形，没有新的近似\n第二项不用抽样，直接按闭式算", TEAL)
    note(ys[3], "随机误差，不是系统误差\n平均起来等于 ELBO", CORAL)
    note(ys[4], "三处近似：① 变分族 ② 单样本 ③ 小批量\n绿色箭头是恒等式，橙色是近似", GRAY)
    save(fig, "r0_fig1_three_approximations.png", bbox_inches="tight")


# ---------------------------------------------------------------- r0_fig4: one dedicated Gaussian per sample
def fig_per_sample():
    fig, axes = plt.subplots(1, 3, figsize=(W, 6.6 * CM))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.1, 3.1)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=7)
        clean(ax)
    ax = axes[0]                                   # (a) the usual textbook picture
    for r in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r, fill=False, ec=GRAY, ls="--", lw=0.8))
    zs = np.random.default_rng(45).standard_normal((45, 2))
    ax.scatter(zs[:, 0], zs[:, 1], s=7, c=GRAY, zorder=3)
    for k in range(3):
        ax.annotate("?", zs[k], xytext=(6, 4), textcoords="offset points", color=CORAL, fontsize=10, fontweight="bold")
    ax.text(0, 2.55, r"$z\sim\mathcal{N}(0,I)$", ha="center", fontsize=8, color=GRAY)
    ax.set_title("(a) 常见画法：从先验抽 z 再解码\n可是这个 z 该和哪个 x 比较？", fontsize=8)
    ax = axes[1]                                   # (b) one dedicated Gaussian per sample
    for r in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r, fill=False, ec=GRAY, ls="--", lw=0.6))
    mus = [(-1.25, 0.95), (1.15, 1.05), (0.35, -1.35)]
    sds = [(0.55, 0.32), (0.36, 0.6), (0.62, 0.4)]
    eps = [(0.6, -0.5), (0.4, -0.6), (-0.5, 0.4)]          # fixed offsets: each z_k lies inside its own ellipse
    for k, (mu_, sd, c, e) in enumerate(zip(mus, sds, (BLUE, TEAL, PURPLE), eps), 1):
        ax.add_patch(Ellipse(mu_, 4 * sd[0], 4 * sd[1], fc=c, alpha=0.18, ec=c, lw=1.0))
        ax.plot(*mu_, "+", color=c, ms=7, mew=1.2)
        zk = np.array(mu_) + np.array(sd) * np.array(e)
        ax.plot(*zk, "o", color=c, ms=4)
        ax.annotate(rf"$z_{k}$", zk, xytext=(4, -9), textcoords="offset points", fontsize=8, color=c)
        ax.text(mu_[0], mu_[1] + 2 * sd[1] + 0.12, rf"$x_{k}$ 的专属分布", ha="center", va="bottom", fontsize=7.5, color=c)
    ax.text(0.0, -2.75, r"$z_k\sim\mathcal{N}(\mu_k,\sigma_k^2)$，解码成 $\hat x_k$ 后与 $x_k$ 比较", ha="center", fontsize=7)
    ax.set_title("(b) VAE 的真实做法：\n每个样本一个专属正态分布", fontsize=8)
    ax = axes[2]                                   # (c) many dedicated Gaussians add up to roughly N(0, I)
    K, TAU2, SIG2 = 80, 0.6, 0.4                   # mu_k ~ N(0, TAU2 I), each N(mu_k, SIG2 I); TAU2 + SIG2 = 1
    mu_k = np.random.default_rng(80).standard_normal((K, 2)) * math.sqrt(TAU2)   # the same draws as code/r0_check.py
    for m in mu_k:
        ax.add_patch(Ellipse(m, 2 * math.sqrt(SIG2), 2 * math.sqrt(SIG2), fc=BLUE, alpha=0.04, ec=BLUE, lw=0.4))
    gx = np.linspace(-3.1, 3.1, 311)
    GX, GY = np.meshgrid(gx, gx)
    pts = np.stack([GX, GY], -1)
    mix = np.zeros_like(GX)
    for m in mu_k:
        mix += np.exp(-((pts - m) ** 2).sum(-1) / (2 * SIG2)) / (2 * np.pi * SIG2)
    mix /= K
    std2 = np.exp(-(GX ** 2 + GY ** 2) / 2) / (2 * np.pi)
    levels = [np.exp(-2) / (2 * np.pi), np.exp(-0.5) / (2 * np.pi)]          # N(0,I) density at r = 2 and r = 1
    ax.contour(GX, GY, std2, levels=levels, colors=GRAY, linestyles="--", linewidths=0.8)
    ax.contour(GX, GY, mix, levels=levels, colors=CORAL, linewidths=1.3)
    ax.plot([], [], color=CORAL, lw=1.3, label=f"{K} 个专属分布叠加")
    ax.plot([], [], color=GRAY, ls="--", lw=0.8, label=r"$\mathcal{N}(0,I)$")
    ax.legend(loc="lower center", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.02))
    ax.set_title("(c) KL 把每个专属分布拉向 N(0, I)\n全体叠加后接近 N(0, I)", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "r0_fig4_posterior_per_sample.png")


# ---------------------------------------------------------------- r0_fig6: tug of war in the toy model
def fig_tug_of_war():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 6.8 * CM), gridspec_kw={"width_ratios": [1.35, 1]})
    s2 = np.linspace(0.02, 2.0, 600)
    r_curve, k_curve = recon_neg(POST_MEAN, s2), kl_std(POST_MEAN, s2)
    ax1.plot(s2, r_curve, color=BLUE, lw=1.4, label=r"重建项  $-\mathrm{E}_q[\log p(x|z)]$：方差越小越好")
    ax1.plot(s2, k_curve, color=CORAL, lw=1.4, label=r"KL 项  $\mathrm{KL}(q\,\Vert\,\mathcal{N}(0,1))$：把方差拉向 1")
    ax1.plot(s2, r_curve + k_curve, color="black", lw=2.0, label=r"两项之和 $=-\mathrm{ELBO}$")
    ax1.axhline(-log_px(), color=GRAY, ls=":", lw=0.9)
    ax1.plot([POST_VAR], [recon_neg(POST_MEAN, POST_VAR) + kl_std(POST_MEAN, POST_VAR)], "o", color="black", ms=5)
    ax1.annotate(f"最小值在 $s^2={POST_VAR:.1f}$，正是真实后验的方差\n此时两项之和 $={-log_px():.4f}=-\\log p(x)$",
                 (POST_VAR, -log_px()), xytext=(0.62, 1.15), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax1.set_xlim(0, 2.0)
    ax1.set_ylim(0, 7.5)
    ax1.set_xlabel(r"编码器给出的方差 $s^2$（均值固定为后验均值 1.2）", fontsize=8)
    ax1.set_ylabel("损失（越小越好）", fontsize=8)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    ax1.set_title(r"(a) 玩具模型 $p(z)=\mathcal{N}(0,1)$，$p(x|z)=\mathcal{N}(2z,1)$，$x=3$", fontsize=8)
    clean(ax1)
    u = np.linspace(0.02, 3.0, 500)
    ax2.plot(u, u - 1 - np.log(u), color=CORAL, lw=1.6)
    for uu in (0.05, 0.2, 1.0, 3.0):
        ax2.plot([uu], [uu - 1 - np.log(uu)], "o", color=CORAL, ms=4)
        ax2.annotate(f"{uu - 1 - np.log(uu):.2f}", (uu, uu - 1 - np.log(uu)), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax2.axvline(1.0, color=GRAY, ls=":", lw=0.8)
    ax2.text(0.12, 2.75, r"$\sigma^2\to0$ 时 $-\log\sigma^2\to\infty$：" + "\n方差缩不到 0", fontsize=7.5, color=CORAL)
    ax2.set_xlim(0, 3.0)
    ax2.set_ylim(0, 3.2)
    ax2.set_xlabel(r"$u=\sigma^2$", fontsize=8)
    ax2.set_title(r"(b) 闭式里管方差的那一块 $u-1-\log u$" + "\n只在 u = 1 处为 0", fontsize=8)
    clean(ax2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "r0_fig6_tug_of_war.png")


print("R0 figures ->", OUT)
fig_chain()
fig_per_sample()
fig_tug_of_war()
