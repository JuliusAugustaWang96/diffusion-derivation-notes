# -*- coding: utf-8 -*-
"""Figures for the prerequisite notes P0 (docs/P0-probability-and-information.md), drawn from the formulas.
Usage: python code/make_figures_p0.py [outdir]      (default: docs/figures)
p0_fig1 Gaussian density: peak and inflection points of N(0,1); three values of sigma and the area within mu +- sigma
p0_fig2 entropy of a binary random variable, in nats (left axis) and bits (right axis)
p0_fig3 fitting one Gaussian q to a two-peak p: minimising KL(p||q) versus KL(q||p)
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
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def log_npdf(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


# p0_fig1: (a) N(0,1) with its peak and inflection points, (b) sigma = 0.25, 1, 2 and the area within mu +- sigma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 6.2 * CM))
x = np.linspace(-4, 4, 2001)
ax1.plot(x, npdf(x, 0, 1), color=BLUE, lw=1.5)
peak, infl = npdf(0, 0, 1), npdf(1, 0, 1)
ax1.plot([0], [peak], "o", color=CORAL, ms=4)
ax1.plot([-1, 1], [infl, infl], "o", color=TEAL, ms=4)
ax1.annotate(f"最大值 {peak:.3f}\n在 x = μ 处", (0, peak), xytext=(1.1, 0.37), fontsize=8,
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax1.annotate(f"拐点 x = μ ± σ\n密度 {infl:.3f}", (1, infl), xytext=(1.7, 0.2), fontsize=8,
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax1.set_title(r"(a) 标准正态分布 $\mathcal{N}(0,1)$", fontsize=9)
ax1.set_xlabel("x")
ax1.set_ylabel("概率密度 p(x)")
ax1.set_ylim(0, 0.45)
clean(ax1)
x = np.linspace(-5, 5, 4001)
for s, color in ((0.25, CORAL), (1.0, BLUE), (2.0, TEAL)):
    label = rf"$\sigma={s:g}$" + (f"，峰值 {npdf(0, 0, s * s):.3f}" if s == 0.25 else "")
    ax2.plot(x, npdf(x, 0, s * s), color=color, lw=1.4, label=label)
band = (x >= -1) & (x <= 1)
area = trap(npdf(x[band], 0, 1), x[band])
ax2.fill_between(x[band], npdf(x[band], 0, 1), color=BLUE, alpha=0.18, lw=0)
ax2.annotate(f"阴影面积 {area:.3f}\n= P(−1 ≤ X ≤ 1)", (0.55, 0.16), xytext=(1.6, 0.55), fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax2.axhline(1, color=GRAY, lw=0.7, ls=":")
ax2.text(4.9, 1.03, "密度 = 1", ha="right", fontsize=7.5, color=GRAY)
ax2.set_title(r"(b) $\mu=0$：$\sigma$ 越大，曲线越宽越扁", fontsize=9)
ax2.set_xlabel("x")
ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
clean(ax2)
fig.tight_layout()
fig.savefig(OUT / "p0_fig1_gaussian.png", dpi=200)
plt.close(fig)
print(f"p0_fig1: N(0,1) peak {peak:.4f}, density at the inflection points {infl:.4f}; "
      f"peak for sigma=0.25 {npdf(0, 0, 0.0625):.4f}; area within +-1 for sigma=1 {area:.4f}")

# p0_fig2: binary entropy H(p) = -p log p - (1-p) log(1-p), with 0 log 0 = 0
p = np.linspace(0, 1, 2001)
with np.errstate(divide="ignore", invalid="ignore"):
    H = -np.where(p > 0, p * np.log(p), 0) - np.where(p < 1, (1 - p) * np.log(1 - p), 0)
fig, ax = plt.subplots(figsize=(W, 5.6 * CM))
ax.plot(p, H, color=BLUE, lw=1.5)
h01 = -(0.1 * np.log(0.1) + 0.9 * np.log(0.9))
ax.plot([0.5], [np.log(2)], "o", color=CORAL, ms=4)
ax.plot([0.1, 0.9], [h01, h01], "o", color=TEAL, ms=4)
ax.annotate(f"p = 0.5 时最大：ln 2 = {np.log(2):.3f} 奈特 = 1 比特", (0.5, np.log(2)), xytext=(0.56, 0.74),
            fontsize=8, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax.annotate(f"p = 0.1 或 0.9：{h01:.3f}", (0.9, h01), xytext=(0.62, 0.2), fontsize=8,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax.text(0.5, 0.05, "p = 0 或 1 时结果确定，熵为 0", ha="center", fontsize=7.5, color=GRAY)
ax.set_xlabel("p = P(x = 1)")
ax.set_ylabel("香农熵 H（奈特）")
ax.set_ylim(0, 0.82)
sec = ax.secondary_yaxis("right", functions=(lambda h: h / np.log(2), lambda b: b * np.log(2)))
sec.set_ylabel("香农熵 H（比特）")
clean(ax)
ax.spines["right"].set_visible(True)
fig.tight_layout()
fig.savefig(OUT / "p0_fig2_binary_entropy.png", dpi=200)
plt.close(fig)
print(f"p0_fig2: max {np.log(2):.4f} nats at p=0.5; H(0.1) {h01:.4f}")

# p0_fig3: p = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.6^2); q = N(mu, sigma^2) minimising KL(p||q) or KL(q||p)
xs = np.linspace(-20, 20, 8001)                    # wide grid: q with a large sigma still integrates to 1 here
logp = np.logaddexp(np.log(0.5) + log_npdf(xs, -2.0, 0.36), np.log(0.5) + log_npdf(xs, 2.0, 0.36))
pm = np.exp(logp)
Ep, Ep2, Hp = trap(pm * xs, xs), trap(pm * xs ** 2, xs), -trap(pm * logp, xs)


def kl_forward(mu, s):                            # KL(p||q) = -H(p) - E_p[log q]
    return -Hp + 0.5 * np.log(2 * np.pi * s ** 2) + (Ep2 - 2 * mu * Ep + mu ** 2) / (2 * s ** 2)


def kl_reverse(mu, s):                            # KL(q||p) = -H(q) - E_q[log p]
    mu, s = np.broadcast_arrays(np.asarray(mu, float), np.asarray(s, float))
    out = np.empty(mu.shape)
    for idx in np.ndindex(mu.shape):
        out[idx] = -0.5 * np.log(2 * np.pi * np.e * s[idx] ** 2) - trap(npdf(xs, mu[idx], s[idx] ** 2) * logp, xs)
    return out


def grid_min(f, mu_lo, mu_hi, s_lo, s_hi):        # coarse grid, then refine around the best point
    for step in (0.05, 0.005, 0.0005, 0.00005):
        M, S = np.meshgrid(np.arange(mu_lo, mu_hi + step / 2, step), np.arange(s_lo, s_hi + step / 2, step))
        V = f(M, S)
        k = np.unravel_index(np.argmin(V), V.shape)
        mu_lo, mu_hi, s_lo, s_hi = M[k] - 10 * step, M[k] + 10 * step, S[k] - 10 * step, S[k] + 10 * step
    return M[k], S[k], V[k]


fits = [grid_min(kl_forward, -3.0, 3.0, 0.3, 4.0), grid_min(kl_reverse, 0.0, 3.0, 0.3, 2.0)]
x = np.linspace(-6, 6, 2401)
p_plot = np.exp(np.logaddexp(np.log(0.5) + log_npdf(x, -2.0, 0.36), np.log(0.5) + log_npdf(x, 2.0, 0.36)))
fig, axes = plt.subplots(1, 2, figsize=(W, 6.0 * CM), sharey=True)
titles = [r"(a) 最小化 $D_{\mathrm{KL}}(p\,\|\,q)$：$q$ 盖住两个峰", r"(b) 最小化 $D_{\mathrm{KL}}(q\,\|\,p)$：$q$ 只取一个峰"]
values = [r"$D_{\mathrm{KL}}(p\,\|\,q^*)$", r"$D_{\mathrm{KL}}(q^*\|\,p)$"]
for ax, (mu, s, val), title, vname in zip(axes, fits, titles, values):
    mu = 0.0 if abs(mu) < 5e-4 else mu            # no "-0.000" in the legend
    ax.plot(x, p_plot, color=BLUE, lw=1.5, label=r"$p$：两个峰的混合")
    ax.plot(x, npdf(x, mu, s * s), color=CORAL, lw=1.5, ls="--", label=rf"$q^*$：$\mu={mu:.3f},\ \sigma={s:.3f}$")
    ax.text(0.02, 0.70, f"{vname} = {val:.3f}", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    clean(ax)
    print(f"p0_fig3 {title[:3]}: mu* {mu:.4f}, sigma* {s:.4f}, minimum KL {val:.4f}")
axes[0].set_ylabel("概率密度")
fig.tight_layout()
fig.savefig(OUT / "p0_fig3_kl_direction.png", dpi=200)
plt.close(fig)
print("saved to", OUT)
