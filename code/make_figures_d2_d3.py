# -*- coding: utf-8 -*-
"""Figures 1-4 of the D2-D3 derivation notes (docs/D2-D3-elbo-and-vae-loss.md).
Usage: python code/make_figures_d2_d3.py [outdir]      (default: docs/figures)
fig1 Jensen's inequality for log (T11), fig2 ELBO of the one-dimensional linear Gaussian toy model against log p(x)
(D2(b): the gap is a KL divergence), fig3 computation graphs without and with the reparameterization trick (D3(a)(b)),
fig4 binary cross-entropy of one pixel (D3(d)). Figures 5-8 come from code/02_vae_mnist.py.
PNG, 200 dpi, 16 cm wide. Everything here is computed in closed form, so the figures are reproducible exactly.
The numbers quoted in the text are checked by code/d2_d3_check.py."""
import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
BLUE, CORAL, TEAL, GRAY, GOLD = "#185FA5", "#D85A30", "#0F6E56", "#888780", "#B8860B"


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


def logpdf(x, m, v):
    """log N(x; m, v); the second argument is the variance."""
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)


# ------------------------------------------------------------------ fig1: Jensen for log, two-point example
x1, x2 = 1.0, 4.0
mean_x = (x1 + x2) / 2
log_of_mean, mean_of_log = np.log(mean_x), (np.log(x1) + np.log(x2)) / 2
fig, ax = plt.subplots(figsize=(W, 6.5 * CM))
xs = np.linspace(0.3, 5.2, 400)
ax.plot(xs, np.log(xs), color=BLUE, lw=1.8, label=r"$\log x$（凹函数：弦在曲线下方）")
ax.plot([x1, x2], [np.log(x1), np.log(x2)], color=GRAY, lw=1.2, ls="--", label="连接两点的弦")
ax.scatter([x1, x2], [np.log(x1), np.log(x2)], color=BLUE, zorder=3, s=22)
ax.scatter([mean_x], [log_of_mean], color=TEAL, zorder=4, s=30)
ax.scatter([mean_x], [mean_of_log], color=CORAL, zorder=4, s=30)
ax.vlines(mean_x, mean_of_log, log_of_mean, color=GOLD, lw=2)
ax.annotate(rf"$\log\mathbb{{E}}[X]=\log 2.5={log_of_mean:.3f}$", (mean_x, log_of_mean), xytext=(0.55, 1.45),
            fontsize=8.5, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))
ax.annotate(rf"$\mathbb{{E}}[\log X]=\frac{{\log 1+\log 4}}{{2}}={mean_of_log:.3f}$", (mean_x, mean_of_log),
            xytext=(2.75, 0.18), fontsize=8.5, color=CORAL, arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.7))
ax.text(mean_x + 0.08, (mean_of_log + log_of_mean) / 2, f"差 {log_of_mean - mean_of_log:.3f}", fontsize=8, color=GOLD,
        va="center")
for xv, lab in ((x1, "$X=1$"), (x2, "$X=4$")):
    ax.text(xv, np.log(xv) - 0.22, lab, ha="center", fontsize=8.5)
ax.set_xlabel("x")
ax.set_ylabel(r"$\log x$")
ax.set_xlim(0.2, 5.3)
ax.set_ylim(-1.3, 1.9)
ax.set_title(r"Jensen 不等式：X 等概率取 1 或 4，$\log\mathbb{E}[X]\ \geq\ \mathbb{E}[\log X]$", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="lower right")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "d2d3_fig1_jensen.png", dpi=200)
plt.close(fig)
print(f"fig1: log E[X] = {log_of_mean:.4f}, E[log X] = {mean_of_log:.4f}, gap {log_of_mean - mean_of_log:.4f}")

# ------------------------------------------------------------------ fig2: ELBO of the linear Gaussian toy model
a, sig2, xobs = 2.0, 1.0, 3.0                    # z ~ N(0,1), x | z ~ N(a z, sig2), observed x = 3
m_post, v_post = a * xobs / (sig2 + a * a), sig2 / (sig2 + a * a)          # true posterior N(1.2, 0.2)
log_px = logpdf(xobs, 0.0, a * a + sig2)                                     # log N(3; 0, 5)


def elbo(m, s2):
    """Closed-form ELBO for q = N(m, s2): E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z)]."""
    t_lik = -0.5 * np.log(2 * np.pi * sig2) - ((xobs - a * m) ** 2 + a * a * s2) / (2 * sig2)
    t_prior = -0.5 * np.log(2 * np.pi) - (m * m + s2) / 2
    t_ent = 0.5 * np.log(2 * np.pi * np.e * s2)
    return t_lik + t_prior + t_ent


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 6.3 * CM))
ms = np.linspace(-0.8, 3.2, 400)
ax1.axhline(log_px, color=GRAY, lw=1.4, ls="--", label=rf"$\log p(x)={log_px:.3f}$")
ax1.plot(ms, elbo(ms, v_post), color=BLUE, lw=1.8, label=rf"ELBO，$s^2={v_post:g}$（后验方差）")
ax1.plot(ms, elbo(ms, 1.0), color=CORAL, lw=1.5, label=r"ELBO，$s^2=1$")
ax1.scatter([m_post], [elbo(m_post, v_post)], color=TEAL, zorder=4, s=30)
ax1.annotate(rf"$q$ = 真实后验 $\mathcal{{N}}({m_post:g},{v_post:g})$：缺口为 0", (m_post, log_px),
             xytext=(-0.75, log_px + 0.3), fontsize=7.5, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))
mm = 2.0
ax1.vlines(mm, elbo(mm, v_post), log_px, color=GOLD, lw=2)
ax1.text(mm + 0.06, (elbo(mm, v_post) + log_px) / 2, f"缺口 = KL = {log_px - elbo(mm, v_post):.2f}", fontsize=7.5,
         color=GOLD, va="center")
ax1.set_xlabel(r"q 的均值 $m$")
ax1.set_ylabel("ELBO")
ax1.set_ylim(log_px - 6.5, log_px + 0.8)
ax1.set_title("固定方差，改变均值", fontsize=9)
ax1.legend(frameon=False, fontsize=7.5, loc="lower left")
clean(ax1)
s2s = np.logspace(np.log10(0.02), np.log10(3.0), 400)
ax2.axhline(log_px, color=GRAY, lw=1.4, ls="--", label=rf"$\log p(x)={log_px:.3f}$")
ax2.plot(s2s, elbo(m_post, s2s), color=BLUE, lw=1.8, label=rf"ELBO，$m={m_post:g}$（后验均值）")
ax2.plot(s2s, elbo(0.0, s2s), color=CORAL, lw=1.5, label=r"ELBO，$m=0$")
ax2.scatter([v_post], [elbo(m_post, v_post)], color=TEAL, zorder=4, s=30)
ax2.axvline(v_post, color=TEAL, lw=0.7, ls=":")
ax2.set_xscale("log")
ax2.set_xlabel(r"q 的方差 $s^2$（对数刻度）")
ax2.set_ylim(log_px - 6.5, log_px + 0.8)
ax2.set_title("固定均值，改变方差", fontsize=9)
ax2.legend(frameon=False, fontsize=7.5, loc="lower left")
clean(ax2)
fig.tight_layout()
fig.savefig(OUT / "d2d3_fig2_elbo_gap.png", dpi=200)
plt.close(fig)
print(f"fig2: log p(x) = {log_px:.4f}; ELBO at the posterior N({m_post:g},{v_post:g}) = {elbo(m_post, v_post):.4f}; "
      f"gap at m = 2, s^2 = 0.2: {log_px - elbo(mm, v_post):.4f}")


# ------------------------------------------------------------------ fig3: computation graphs without / with reparameterization
def box(ax, x, y, w, h, text, color=BLUE, fill="#EAF2FB", dashed=False, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=color, lw=1.2,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222222")


def arrow(ax, p0, p1, color=GRAY, lw=1.2, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


fig, axes = plt.subplots(2, 1, figsize=(W, 9.2 * CM))
Y, H = 1.35, 0.9
cols = [(0.15, 1.35, "输入 $x$"), (1.85, 1.9, "编码器\n参数 $\\phi$"), (4.1, 1.6, "$\\mu,\\ \\log\\sigma^2$"),
        (6.05, 2.45, None), (8.85, 1.75, "解码器\n参数 $\\theta$"), (10.95, 1.0, "损失")]
titles = ["(a) 直接采样：梯度在采样处断开", "(b) 重参数化：噪声从外部进来，梯度穿过 $\\mu$、$\\sigma$ 回到编码器"]
for k, (ax, title) in enumerate(zip(axes, titles)):
    ax.set_xlim(0, 12.1)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left")
    for x0, w, text in cols:
        if text is None:
            if k == 0:
                box(ax, x0, Y, w, H, "采样\n$z\\sim\\mathcal{N}(\\mu,\\sigma^2)$", color=CORAL, fill="#FBE9E3", dashed=True)
            else:
                box(ax, x0, Y, w, H, "$z=\\mu+\\sigma\\odot\\varepsilon$", color=TEAL, fill="#E3F2EE")
        else:
            box(ax, x0, Y, w, H, text)
    for (x0, w, _), (x1, _, _) in zip(cols[:-1], cols[1:]):          # forward arrows between neighbouring boxes
        arrow(ax, (x0 + w, Y + H / 2), (x1, Y + H / 2))
    gy = Y - 0.42                                                    # backward (gradient) arrows below the boxes
    ax.text(11.45, gy - 0.33, "梯度", fontsize=7.5, color=CORAL, ha="center")
    pairs = list(zip(cols[1:], cols[:-1]))                           # (right box, left box)
    stop_at = 3 if k == 0 else 0                                     # (a): the gradient stops at the sampling box
    for j, ((xr, wr, _), (xl, wl, _)) in enumerate(pairs[::-1]):
        if len(pairs) - 1 - j < stop_at:
            break
        arrow(ax, (xr + wr / 2, gy), (xl + wl / 2, gy), color=CORAL, lw=1.3, ls="--")
    xs_, ws_, _ = cols[3]
    if k == 0:
        ax.text(xs_ + ws_ / 2, gy - 0.36, "随机采样不是 $\\mu,\\sigma$ 的可微函数，梯度到此为止", fontsize=7.5, color=CORAL,
                ha="center")
        ax.text(2.8, gy - 0.36, "编码器收不到梯度", fontsize=7.5, color=GRAY, ha="center")
    else:
        box(ax, xs_ + 0.25, Y + H + 0.55, ws_ - 0.5, 0.75, "$\\varepsilon\\sim\\mathcal{N}(0,I)$\n外部噪声，不含 $\\phi$",
            color=GRAY, fill="#F3F3F1", fs=7.5)
        arrow(ax, (xs_ + ws_ / 2, Y + H + 0.55), (xs_ + ws_ / 2, Y + H))
        ax.text(xs_ + ws_ / 2, gy - 0.36, "$\\partial z/\\partial\\mu=1$，$\\partial z/\\partial\\sigma=\\varepsilon$",
                fontsize=7.5, color=TEAL, ha="center")
fig.tight_layout()
fig.savefig(OUT / "d2d3_fig3_reparam.png", dpi=200)
plt.close(fig)
print("fig3: schematic")

# ------------------------------------------------------------------ fig4: binary cross-entropy of one pixel
p = np.linspace(0.005, 0.995, 600)


def bce(x, p):
    return -(x * np.log(p) + (1 - x) * np.log(1 - p))


fig, ax = plt.subplots(figsize=(W, 6.0 * CM))
for xv, col, lab in ((1.0, BLUE, "像素值 $x=1$：$-\\log p$"), (0.0, CORAL, "像素值 $x=0$：$-\\log(1-p)$"),
                     (0.5, TEAL, "灰度像素 $x=0.5$：$-0.5\\log p-0.5\\log(1-p)$")):
    ax.plot(p, bce(xv, p), color=col, lw=1.8, label=lab)
    pmin = np.clip(xv, 0.005, 0.995)
    ax.scatter([pmin], [bce(xv, pmin)], color=col, s=28, zorder=4)
ax.axvline(0.5, color=GRAY, lw=0.6, ls=":")
ax.text(0.5, 1.5, r"三条曲线的最低点都在 $p=x$；灰度像素的最小值是 $\log 2=0.693$", fontsize=7.5, color=TEAL, ha="center")
ax.set_xlabel(r"解码器输出的概率 $p$")
ax.set_ylabel("二元交叉熵 BCE")
ax.set_ylim(0, 5)
ax.set_title("一个像素的 BCE：预测概率离像素值越远，损失越大", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper center")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "d2d3_fig4_bce.png", dpi=200)
plt.close(fig)
print(f"fig4: BCE(x=0.5, p=0.5) = {bce(0.5, 0.5):.4f}")
print("saved to", OUT)
