# -*- coding: utf-8 -*-
"""Figures for the D0–D1 derivation notes (docs/D0-D1-gaussian-and-kl.md).
Usage: python code/make_figures.py [outdir]      (default: docs/figures)
fig1 noise merging (D0(c)), fig2 product of two Gaussian densities (D0(b)),
fig3 linear Gaussian model: joint samples and the posterior slice (D0(a)), fig4 KL integrands (D1).
PNG, 200 dpi, 16 cm wide. Prints the numbers each figure shows so they can be checked against the text."""
import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
BLUE, CORAL, TEAL, GRAY = "#185FA5", "#D85A30", "#0F6E56", "#888780"
rng = np.random.default_rng(0)
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


# fig1: 3 eps1 + 4 eps2 against N(0,25) and the wrong N(0,49)
e1, e2 = rng.standard_normal(200_000), rng.standard_normal(200_000)
s = 3 * e1 + 4 * e2
fig, ax = plt.subplots(figsize=(W, 6 * CM))
ax.hist(s, bins=120, range=(-25, 25), density=True, color="#B5D4F4", edgecolor="none",
        label=r"$3\varepsilon_1+4\varepsilon_2$ 的 20 万个样本")
x = np.linspace(-25, 25, 801)
ax.plot(x, npdf(x, 0, 25), color=BLUE, lw=1.8, label=r"$\mathcal{N}(0,25)$：方差相加，标准差 5")
ax.plot(x, npdf(x, 0, 49), color=CORAL, lw=1.5, ls="--", label=r"$\mathcal{N}(0,49)$：误把标准差相加，标准差 7")
ax.set_xlabel("取值")
ax.set_ylabel("密度")
ax.legend(frameon=False, fontsize=8, loc="upper right")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "fig1_noise_merge.png", dpi=200)
plt.close(fig)
print(f"fig1: sample mean {s.mean():.3f}, var {s.var():.3f}, std {s.std():.3f}")

# fig2: product of two densities, equal and unequal variances
cases = [(0, 1, 2, 1), (0, 1, 2, 4)]
fig, axes = plt.subplots(1, 2, figsize=(W, 6 * CM), sharey=True)
x = np.linspace(-4, 6, 2001)
for ax, (m1, v1, m2, v2) in zip(axes, cases):
    A, B = 1 / v1 + 1 / v2, m1 / v1 + m2 / v2
    m, v = B / A, 1 / A
    f = npdf(x, m1, v1) * npdf(x, m2, v2)
    fn = f / trap(f, x)
    ax.plot(x, npdf(x, m1, v1), color=GRAY, lw=1.2, label=rf"$\mathcal{{N}}(x;{m1},{v1})$")
    ax.plot(x, npdf(x, m2, v2), color=GRAY, lw=1.2, ls="--", label=rf"$\mathcal{{N}}(x;{m2},{v2})$")
    ax.plot(x, fn, color=BLUE, lw=2, label=rf"乘积除以面积：$\mathcal{{N}}(x;{m:g},{v:g})$")
    ax.axvline(m, color=BLUE, lw=0.8, ls=":")
    ax.set_title(rf"$\sigma_1^2={v1},\ \sigma_2^2={v2}$", fontsize=9)
    ax.set_xlabel("x")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    clean(ax)
    print(f"fig2 {m1, v1, m2, v2}: formula mean {m:.4f} var {v:.4f}; "
          f"numeric mean {trap(x * fn, x):.4f} var {trap((x - m) ** 2 * fn, x):.4f}")
axes[0].set_ylabel("密度")
axes[0].set_ylim(0, 0.75)
fig.tight_layout()
fig.savefig(OUT / "fig2_product.png", dpi=200)
plt.close(fig)

# fig3: linear Gaussian model a=2, sigma^2=1, observe x=3
a, s2, xobs, half = 2.0, 1.0, 3.0, 0.05
n = 2_000_000
z = rng.standard_normal(n)
xx = a * z + np.sqrt(s2) * rng.standard_normal(n)
sel = np.abs(xx - xobs) < half
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 6.4 * CM))
idx = rng.choice(n, 5000, replace=False)
ax1.scatter(xx[idx], z[idx], s=2, color=GRAY, alpha=0.35, linewidths=0, label="联合样本")
ax1.axvspan(xobs - half, xobs + half, color=CORAL, alpha=0.25, lw=0)
ax1.scatter(xx[sel][:800], z[sel][:800], s=2, color=CORAL, linewidths=0, label=r"$x\approx3$ 的样本")
ax1.set_xlabel("x（观测）")
ax1.set_ylabel("z（隐变量）")
ax1.set_xlim(-8, 8)
ax1.set_title(r"联合分布 $p(z,x)$，$a=2,\ \sigma^2=1$", fontsize=9)
ax1.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=4)
clean(ax1)
m_post, v_post = a * xobs / (s2 + a * a), s2 / (s2 + a * a)
zz = np.linspace(-1.5, 3.5, 500)
ax2.hist(z[sel], bins=60, density=True, color="#F5C4B3", edgecolor="none", label=f"竖条内 z 的直方图（{sel.sum()} 个）")
ax2.plot(zz, npdf(zz, m_post, v_post), color=BLUE, lw=2, label=rf"公式：$\mathcal{{N}}({m_post:g},\,{v_post:g})$")
ax2.plot(zz, npdf(zz, 0, 1), color=GRAY, lw=1.2, ls="--", label=r"先验 $\mathcal{N}(0,1)$")
ax2.set_xlabel("z")
ax2.set_ylabel("密度")
ax2.set_title(r"后验 $p(z\mid x=3)$", fontsize=9)
ax2.set_ylim(0, 1.25)
ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
clean(ax2)
fig.tight_layout()
fig.savefig(OUT / "fig3_posterior.png", dpi=200)
plt.close(fig)
print(f"fig3: band n={sel.sum()}, z mean {z[sel].mean():.4f} var {z[sel].var():.4f}; "
      f"formula {m_post:.4f} {v_post:.4f}; x var {xx.var():.4f}")

# fig4: KL(p||q) and KL(q||p) as net areas
m1, v1, m2, v2 = 1.0, 1.0, 0.0, 4.0
x = np.linspace(-8, 8, 3201)
p, q = npdf(x, m1, v1), npdf(x, m2, v2)
xw = np.linspace(-40, 40, 160001)          # the area is integrated on a wide grid: q has long tails


def logpdf(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)


par = {"p": (m1, v1), "q": (m2, v2)}
fig, axes = plt.subplots(1, 2, figsize=(W, 5.0 * CM), sharey=True)
for ax, (f, g, first, second, name, loc) in zip(axes, [(p, q, "p", "q", r"p\,\|\,q", "upper left"),
                                                       (q, p, "q", "p", r"q\,\|\,p", "upper right")]):
    integrand = f * np.log(f / g)
    (ma, va), (mb, vb) = par[first], par[second]
    val = trap(npdf(xw, ma, va) * (logpdf(xw, ma, va) - logpdf(xw, mb, vb)), xw)
    ax.plot(x, p, color=BLUE, lw=1.4, label=r"$p=\mathcal{N}(1,1)$")
    ax.plot(x, q, color=GRAY, lw=1.4, ls="--", label=r"$q=\mathcal{N}(0,4)$")
    ax.fill_between(x, integrand, 0, where=integrand >= 0, color=TEAL, alpha=0.35, lw=0, label="被积函数为正")
    ax.fill_between(x, integrand, 0, where=integrand < 0, color=CORAL, alpha=0.45, lw=0, label="被积函数为负")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_title(rf"$\mathrm{{KL}}({name})$：阴影净面积 = {val:.3f}", fontsize=9)
    ax.set_xlabel("x")
    ax.set_xlim(-6, 6)
    if loc == "upper left":                       # both panels show the same four items: one legend is enough
        ax.legend(frameon=False, fontsize=7, loc=loc)
    clean(ax)
    print(f"fig4 KL({name}) numeric {val:.4f}")
axes[0].set_ylabel("密度 / 被积函数")
fig.tight_layout()
fig.savefig(OUT / "fig4_kl.png", dpi=200)
plt.close(fig)
print("saved to", OUT)
