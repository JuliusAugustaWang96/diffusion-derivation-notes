# -*- coding: utf-8 -*-
"""Figures for the note I0 (docs/I0-vae-intuition.md) that need no training: two schematics and a 1-D toy model.
Usage: python code/make_figures_i0.py [outdir]      (default: docs/figures)
i0_fig1 the route: training (encoder, noise in the latent space, decoder, two loss terms) and generation (prior, decoder)
i0_fig4 where the noise goes: denoising auto-encoder (on the input), VAE (on the latent code), DDPM (on the data)
i0_fig5 two mixtures in the toy model of L2 fig. 5(b): in x space the decoder's Gaussians weighted by the prior give p(x);
        in z space the posteriors of the data points average to the prior; with Gaussian posteriors only approximately
Figures 2, 3 and 6 of I0 come from code/03_vae_latent_mnist.py (they need the trained MNIST model).
PNG, 200 dpi, 16 cm wide. Prints the numbers each figure shows so they can be checked against the text."""
import sys, math, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC")
installed = {f.name for f in font_manager.fontManager.ttflist}
cjk = [name for name in CJK_FONTS if name in installed]      # only installed fonts: no per-glyph warnings
if not cjk:
    print("No Chinese font found; install e.g. Noto Sans CJK SC, otherwise Chinese labels show as boxes.")
plt.rcParams.update({"font.family": cjk + ["DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 8.5, "mathtext.fontset": "dejavusans"})
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
BLUE, CORAL, TEAL, GRAY, GOLD, PURPLE = "#185FA5", "#D85A30", "#0F6E56", "#888780", "#BA7517", "#7B4FA0"
FILL = {"learn": "#E6F1FB", "fixed": "#FAECE7", "latent": "#E1F5EE", "data": "#F1EFE8", "loss": "#FBF3E0"}
EDGE = {"learn": BLUE, "fixed": CORAL, "latent": TEAL, "data": GRAY, "loss": GOLD}
INK = "#2C2C2A"


def canvas(height_cm):
    fig = plt.figure(figsize=(16 * CM, height_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, height_cm)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="learn", size=8, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[kind], ec=EDGE[kind], lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#444441", ls="-", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, linestyle=ls))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  saved", OUT / name)


# ---------------------------------------------------------------- i0_fig1: the route
def fig_route():
    fig, ax = canvas(8.0)
    h = 1.0
    ax.text(0.2, 7.6, "(a) 训练：每张训练图像都走一遍这条路", fontsize=8.5, color=INK)
    y = 5.35
    box(ax, 0.3, y, 1.6, h, "训练图像 $x$", "data", 7.5)
    box(ax, 2.4, y, 1.8, h, "编码器\n参数 $\\phi$", "learn", 7.5)
    box(ax, 4.7, y, 3.0, h, "$\\mu_\\phi(x)$、$\\sigma_\\phi(x)$\n专属分布 $q_\\phi(z|x)$", "latent", 7.3)
    box(ax, 8.3, y, 2.5, h, "$z=\\mu+\\sigma\\odot\\varepsilon$", "latent", 8, lw=1.8)
    box(ax, 11.3, y, 1.8, h, "解码器\n参数 $\\theta$", "learn", 7.5)
    box(ax, 13.6, y, 2.1, h, "$p_\\theta(x|z)$\n每个像素的概率", "data", 7)
    for a, b in ((1.9, 2.4), (4.2, 4.7), (7.7, 8.3), (10.8, 11.3), (13.1, 13.6)):
        arrow(ax, a, y + h / 2, b, y + h / 2)
    box(ax, 8.3, 6.85, 2.5, 0.6, "$\\varepsilon\\sim\\mathcal{N}(0,I)$ 外部噪声", "fixed", 7)
    arrow(ax, 9.55, 6.85, 9.55, y + h, color=CORAL, lw=1.3)
    ax.text(11.05, 7.15, "问题 2：噪声在这里加入\n隐空间里，编码器与解码器之间", fontsize=7, color=CORAL, va="center")
    ax.text(2.4, 7.0, "问题 1：编码器给每张图一个专属分布 $q_\\phi(z|x)$，\n用来近似解码器的真实后验 $p_\\theta(z|x)$",
            fontsize=7, color=TEAL, va="center")
    arrow(ax, 6.2, 6.62, 6.2, y + h + 0.03, color=TEAL)
    box(ax, 4.4, 3.55, 3.6, 0.95, "KL 项：$\\mathrm{KL}(q_\\phi(z|x)\\,\\Vert\\,\\mathcal{N}(0,I))$\n不让专属分布离先验太远、缩成一点",
        "loss", 6.6)
    box(ax, 12.6, 3.55, 3.3, 0.95, "重建项：$-\\log p_\\theta(x|z)$\n要求解码结果像输入的 $x$", "loss", 6.6)
    arrow(ax, 6.2, y, 6.2, 4.5)
    arrow(ax, 14.65, y, 14.65, 4.5)
    ax.text(10.3, 4.02, "两项相加 = 负 ELBO\n对 $\\theta$、$\\phi$ 一起求梯度", fontsize=6.8, ha="center", va="center", color="#444441")
    ax.plot([0.2, 15.8], [3.15, 3.15], color="#D3D1C7", lw=0.8, ls="--")
    ax.text(0.2, 2.75, "(b) 生成：训练好以后，不再需要编码器", fontsize=8.5, color=INK)
    yg = 0.75
    box(ax, 8.3, yg, 2.5, h, "$z\\sim\\mathcal{N}(0,I)$\n从先验抽", "fixed", 7.5)
    box(ax, 11.3, yg, 1.8, h, "解码器\n同一个 $\\theta$", "learn", 7.5)
    box(ax, 13.6, yg, 2.1, h, "新图像", "data", 7.5)
    arrow(ax, 10.8, yg + h / 2, 11.3, yg + h / 2)
    arrow(ax, 13.1, yg + h / 2, 13.6, yg + h / 2)
    ax.plot([12.2, 12.2], [y, yg + h], color=BLUE, lw=0.8, ls=":")
    ax.text(12.3, 2.45, "同一个网络", fontsize=6.5, color=BLUE, va="center")
    ax.plot([6.2, 6.2, 8.3], [3.55, yg + h / 2, yg + h / 2], color=GOLD, lw=1.0, ls="--")
    arrow(ax, 8.0, yg + h / 2, 8.3, yg + h / 2, color=GOLD)
    ax.text(0.3, 1.55, "问题 3：KL 项把所有专属分布拉向 $\\mathcal{N}(0,I)$，\n它们叠在一起就接近先验。"
            "生成时从先验抽的 $z$，\n落在训练时见过的地方，解码器才认得", fontsize=7, color=GOLD, va="center")
    save(fig, "i0_fig1_route.png")


# ---------------------------------------------------------------- i0_fig4: where the noise goes
def fig_noise():
    fig, ax = canvas(8.6)
    h = 0.8
    note_x = 12.15

    def note(y, text):
        ax.text(note_x, y, text, fontsize=6.7, va="center", color="#444441", linespacing=1.35)

    def plus(x, y):
        ax.add_patch(plt.Circle((x, y), 0.32, fc=FILL["fixed"], ec=CORAL, lw=1.6))
        ax.text(x, y, "+", ha="center", va="center", fontsize=11, color=CORAL)

    ax.text(0.2, 8.2, "(a) 去噪自编码器：噪声加在输入图像上", fontsize=8.3, color=INK)
    y = 6.75
    box(ax, 0.3, y, 1.4, h, "图像 $x$", "data", 7.5)
    plus(2.35, y + h / 2)
    ax.text(2.35, y + h + 0.12, "噪声", fontsize=6.5, ha="center", color=CORAL)
    box(ax, 3.0, y, 1.4, h, "$\\tilde{x}$", "data", 8)
    box(ax, 4.9, y, 1.6, h, "编码器", "learn", 7.5)
    box(ax, 7.0, y, 1.2, h, "码", "latent", 7.5)
    box(ax, 8.7, y, 1.6, h, "解码器", "learn", 7.5)
    box(ax, 10.6, y, 1.3, h, "还原 $x$", "data", 7.2)
    for a, b in ((1.7, 2.03), (2.67, 3.0), (4.4, 4.9), (6.5, 7.0), (8.2, 8.7), (10.3, 10.6)):
        arrow(ax, a, y + h / 2, b, y + h / 2)
    note(y + h / 2, "加在哪：输入图像\n多大：人为设定，固定不变\n为了：学到抗干扰的特征")

    ax.text(0.2, 5.6, "(b) VAE：噪声加在隐空间，编码器与解码器之间", fontsize=8.3, color=INK)
    y = 3.95
    box(ax, 0.3, y, 1.4, h, "图像 $x$", "data", 7.5)
    box(ax, 2.1, y, 1.6, h, "编码器", "learn", 7.5)
    box(ax, 4.1, y, 1.7, h, "$\\mu$、$\\sigma$", "latent", 8)
    box(ax, 6.25, y, 2.15, h, "$z=\\mu+\\sigma\\odot\\varepsilon$", "latent", 7.6, lw=1.8)
    box(ax, 8.8, y, 1.6, h, "解码器", "learn", 7.5)
    box(ax, 10.7, y, 1.3, h, "$p_\\theta(x|z)$", "data", 7.5)
    box(ax, 6.25, y + 1.0, 2.15, 0.5, "$\\varepsilon\\sim\\mathcal{N}(0,I)$", "fixed", 7)
    arrow(ax, 7.33, y + 1.0, 7.33, y + h, color=CORAL, lw=1.3)
    for a, b in ((1.7, 2.1), (3.7, 4.1), (5.8, 6.25), (8.4, 8.8), (10.4, 10.7)):
        arrow(ax, a, y + h / 2, b, y + h / 2)
    note(y + h / 2, "加在哪：隐变量 $z$\n多大：$\\sigma$ 由编码器学出，\n每张图、每一维都不同\n"
                    "为了：从 $q_\\phi(z|x)$ 采样，\n并能对 $\\phi$ 求梯度")

    ax.text(0.2, 2.75, "(c) 扩散模型（DDPM）的前向过程：噪声一步步加在图像本身上", fontsize=8.3, color=INK)
    y = 1.15
    box(ax, 0.3, y, 1.2, h, "$x_0$", "data", 8)
    xs = [1.5]
    for k, lab_ in enumerate(("$x_1$", "$x_2$")):
        plus(xs[-1] + 0.6, y + h / 2)
        box(ax, xs[-1] + 1.2, y, 1.2, h, lab_, "data", 8)
        arrow(ax, xs[-1], y + h / 2, xs[-1] + 0.28, y + h / 2)
        arrow(ax, xs[-1] + 0.92, y + h / 2, xs[-1] + 1.2, y + h / 2)
        xs.append(xs[-1] + 2.4)
    ax.text(xs[-1] + 0.45, y + h / 2, "…", fontsize=12, ha="center", va="center", color=INK)
    plus(xs[-1] + 1.25, y + h / 2)
    arrow(ax, xs[-1] + 0.75, y + h / 2, xs[-1] + 0.93, y + h / 2)
    box(ax, xs[-1] + 1.9, y, 2.4, h, "$x_T\\approx\\mathcal{N}(0,I)$", "fixed", 7.6)
    arrow(ax, xs[-1] + 1.57, y + h / 2, xs[-1] + 1.9, y + h / 2)
    note(y + h / 2, "加在哪：图像本身，一共 $T$ 步\n多大：按固定的时间表 $\\beta_t$\n网络学的是反过来一步步去噪")
    save(fig, "i0_fig4_where_noise.png")


# ---------------------------------------------------------------- i0_fig5: two mixtures in the toy model
A, OMEGA, PHI, S = 1.8, 2.8, 1.0, 0.2


def mu(z):
    return z + A * np.sin(OMEGA * z + PHI)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def elbo_closed(x, m, s):
    """same closed form as code/i0_check.py (checked there against Gauss-Hermite quadrature)"""
    c = np.exp(-0.5 * (OMEGA * s) ** 2)
    e_sin = c * np.sin(OMEGA * m + PHI)
    e_zsin = m * e_sin + s ** 2 * OMEGA * c * np.cos(OMEGA * m + PHI)
    e_sin2 = 0.5 - 0.5 * np.exp(-2 * (OMEGA * s) ** 2) * np.cos(2 * OMEGA * m + 2 * PHI)
    e_sq = x ** 2 - 2 * x * (m + A * e_sin) + m ** 2 + s ** 2 + 2 * A * e_zsin + A ** 2 * e_sin2
    return (-0.5 * math.log(2 * math.pi) - 0.5 * (m ** 2 + s ** 2) - 0.5 * math.log(2 * math.pi * S ** 2)
            - e_sq / (2 * S ** 2) + 0.5 * np.log(2 * math.pi * math.e * s ** 2))


def fit_gaussians(xs):
    """same search as code/i0_check.py: a grid over (m, log s), then five finer grids around the best point"""
    M0 = np.linspace(-4, 4, 401)
    S0 = np.exp(np.linspace(math.log(0.01), math.log(1.5), 41))
    ms, ss = np.empty(len(xs)), np.empty(len(xs))
    for k in range(0, len(xs), 100):
        e = elbo_closed(xs[k:k + 100, None, None], M0[None, :, None], S0[None, None, :])
        best = e.reshape(e.shape[0], -1).argmax(1)
        ms[k:k + 100], ss[k:k + 100] = M0[best // len(S0)], S0[best % len(S0)]
    dm, dl = 0.02, 0.1
    for _ in range(5):
        mg = ms[:, None, None] + np.linspace(-dm, dm, 21)[None, :, None]
        sg = ss[:, None, None] * np.exp(np.linspace(-dl, dl, 21))[None, None, :]
        e = elbo_closed(xs[:, None, None], mg, sg)
        best = e.reshape(len(xs), -1).argmax(1)
        ms, ss = mg[np.arange(len(xs)), best // 21, 0], sg[np.arange(len(xs)), 0, best % 21]
        dm, dl = dm / 4, dl / 4
    return ms, ss


def fig_two_mixtures():
    rng = np.random.default_rng(0)                       # the same draws as code/i0_check.py
    N = 2000
    zn = rng.standard_normal(N)
    xn = mu(zn) + S * rng.standard_normal(N)
    zg = np.linspace(-6, 6, 12001)
    dz = zg[1] - zg[0]
    pz = npdf(zg, 0, 1)
    lpost = -0.5 * zg[None] ** 2 - 0.5 * ((xn[:, None] - mu(zg)[None]) / S) ** 2
    lpost -= lpost.max(1, keepdims=True)
    pp = np.exp(lpost)
    pp /= pp.sum(1, keepdims=True) * dz
    avg = pp.mean(0)
    kl_avg = np.sum(avg * (np.log(avg) - np.log(pz))) * dz
    ms, ss = fit_gaussians(xn)
    qagg = npdf(zg[None], ms[:, None], ss[:, None]).mean(0)
    kl_q = np.sum(qagg * (np.log(qagg) - np.log(pz))) * dz
    targets = (-2.2, -0.4, 0.9, 2.0)
    ex = [int(np.argmin(np.abs(xn - t))) for t in targets]
    cols = (PURPLE, GOLD, TEAL, "#C2185B")
    hole = qagg < 0.1 * pz                                  # where the Gaussian aggregate is below a tenth of the prior
    print(f"  fig5: KL(average of exact posteriors ‖ prior) = {kl_avg:.4f}; KL(average of Gaussians ‖ prior) = {kl_q:.4f}")
    print(f"  fig5: prior mass where the Gaussian aggregate is below 0.1 x prior: {np.sum(pz[hole]) * dz:.4f}; "
          f"highest point of the Gaussian aggregate {qagg.max():.3f}")
    print("  fig5 examples:", [(k, round(float(xn[k]), 3), round(float(ms[k]), 3), round(float(ss[k]), 3)) for k in ex])

    fig, axs = plt.subplots(1, 3, figsize=(16 * CM, 6.4 * CM))
    ax = axs[0]
    xg = np.linspace(-4.5, 5, 1901)
    px = np.array([np.sum(pz * npdf(x, mu(zg), S)) * dz for x in xg])
    ax.hist(xn, bins=60, range=(-4.5, 5), density=True, color="#E4E2DA", edgecolor="none")
    zk = np.arange(-2.25, 2.26, 0.25)
    for z in zk:
        ax.plot(xg, npdf(z, 0, 1) * 0.25 * npdf(xg, mu(z), S), color=GRAY, lw=0.5)
    ax.plot(xg, px, color="black", lw=1.6)
    for k, c in zip(ex, cols):
        ax.plot([xn[k]], [0], marker="^", color=c, ms=6, clip_on=False, zorder=5)
    ax.set_xlim(-4.5, 5); ax.set_ylim(0, None)
    ax.set_xlabel("$x$", fontsize=8)
    ax.set_title("(a) x 空间（解码器一侧）\n每个 $z$ 给一个高斯，按 $p(z)$ 加权叠成 $p(x)$", fontsize=7.4)
    top = 0.8
    for ax, curves, agg, kl, title in ((axs[1], pp, avg, kl_avg, "(b) z 空间（编码器一侧）：真实后验\n2000 个后验的平均与先验的 KL = "),
                                       (axs[2], None, qagg, kl_q, "(c) z 空间：编码器只能给高斯\n2000 个高斯的平均与先验的 KL = ")):
        if curves is None:
            edges = np.flatnonzero(np.diff(np.r_[0, hole[(zg > -3.5) & (zg < 3.5)].astype(int), 0]))
            zz = zg[(zg > -3.5) & (zg < 3.5)]
            for a_, b_ in zip(edges[::2], edges[1::2] - 1):
                ax.axvspan(zz[a_], zz[b_], color="#E4E2DA", lw=0)
        ax.plot(zg, pz, color="black", lw=1.2, ls="--", label="先验 $\\mathcal{N}(0,1)$")
        for k, c in zip(ex, cols):
            y = pp[k] if curves is not None else npdf(zg, ms[k], ss[k])
            ax.plot(zg, 0.25 * top * y / y.max(), color=c, lw=0.8)
        ax.plot(zg, agg, color=BLUE if curves is not None else CORAL, lw=1.5, label="平均")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, top)
        ax.set_xlabel("$z$", fontsize=8)
        ax.set_title(title + (f"{kl:.3f}" if kl < 0.1 else f"{kl:.2f}"), fontsize=7.4)
        ax.legend(fontsize=6.3, frameon=False, loc="upper right")
    for ax in axs:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.8)
    save(fig, "i0_fig5_two_mixtures.png")


print("I0 figures ->", OUT)
fig_route()
fig_noise()
fig_two_mixtures()
