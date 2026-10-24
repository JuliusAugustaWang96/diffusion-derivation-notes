# -*- coding: utf-8 -*-
"""Figures for the paper walkthrough R1 (docs/R1-ddpm-paper-walkthrough.md). Schematics and exactly computable examples only.
Usage: python code/make_figures_r1.py [outdir]      (default: docs/figures)
r1_fig1 the seven-step chain from -log p(x_0) to the code, identities (green) versus modelling choices and approximations (orange)
r1_fig2 the forward process on the swiss roll of code/04_ddpm_toy2d.py at t = 0, 50, 200, 500, 1000 (one-shot formula, Ho eq. 4)
r1_fig3 the forward-process posterior in 1-D as "prior x likelihood": q(x_{t-1}|x_0) and q(x_t|x_{t-1}) as functions of x_{t-1},
        their normalised product is q(x_{t-1}|x_t,x_0); (a) t = 2 (balanced), (b) t = 100 (the likelihood dominates)
r1_fig4 (a) the weight of the epsilon-loss in Ho eq. (12) for both choices of sigma_t^2 versus the constant weight of L_simple,
        (b) the factor (1-abar_t)/abar_t that turns the x_0-loss into the epsilon-loss
Figures 3, 4 and 7 of R1 are reused from L1. The numbers quoted in R1 are checked by code/r1_check.py.
PNG, 200 dpi, 16 cm wide."""
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
BLUE, CORAL, TEAL, GRAY, GOLD, PURPLE = "#185FA5", "#D85A30", "#0F6E56", "#888780", "#B8860B", "#6B4C9A"
rng = np.random.default_rng(0)

# noise schedule of Ho et al. (2020): T = 1000, beta linear from 1e-4 to 0.02
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = np.cumprod(alphas)
abar_prev = np.concatenate([[1.0], abar[:-1]])
beta_tilde = betas * (1.0 - abar_prev) / (1.0 - abar)
coef_x0 = np.sqrt(abar_prev) * betas / (1.0 - abar)
coef_xt = np.sqrt(alphas) * (1.0 - abar_prev) / (1.0 - abar)


def idx(t):
    return t - 1


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


# ------------------------------------------------------------------ r1_fig1: the seven-step chain
fig, ax = plt.subplots(figsize=(W, 16.6 * CM))
ax.set_xlim(0, 16)
ax.set_ylim(0, 16.6)
ax.axis("off")


def box(y, title, body, fc="#EEF3FA", ec=BLUE, h=1.3, fs=8):
    ax.add_patch(FancyBboxPatch((0.4, y), 10.4, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(0.65, y + h - 0.3, title, fontsize=8.5, fontweight="bold", va="center")
    ax.text(0.65, y + 0.45, body, fontsize=fs, va="center")


def arrow(y1, y2, label, color, x=5.6):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))
    ax.text(x + 0.25, (y1 + y2) / 2, label, fontsize=7.3, color=color, va="center")


def note(y, text, color, h=1.3):
    ax.add_patch(FancyBboxPatch((11.25, y), 4.45, h, boxstyle="round,pad=0.06", fc="white", ec=color, lw=1.0, ls="--"))
    ax.text(11.45, y + h / 2, text, fontsize=6.8, color=color, va="center")


H, G = 1.3, 1.15
ys = [15.0 - k * (H + G) for k in range(7)]
box(ys[0], "① 证据（想最大化，但算不出）",
    r"$-\log p_\theta(x_0)=-\log\int p(x_T)\prod_t p_\theta(x_{t-1}|x_t)\,dx_{1:T}$，对 $T$ 个变量积分", fc="#F4F4F2", ec=GRAY)
box(ys[1], "② 变分界　　论文式 (3)",
    r"$L=\mathrm{E}_q\left[-\log p(x_T)-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\geq-\log p_\theta(x_0)$")
box(ys[2], "③ 三项式　　论文式 (5)，附录 A",
    r"$L=L_T+\sum_{t>1}L_{t-1}+L_0$，$L_{t-1}=\mathrm{E}_q\,\mathrm{KL}(q(x_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t))$", fs=7.2)
box(ys[3], "④ 去噪匹配项的闭式　　论文式 (6)(7)(8)",
    r"$L_{t-1}=\mathrm{E}_q\left[\frac{1}{2\sigma_t^2}\Vert\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\Vert^2\right]+C$")
box(ys[4], "⑤ ε 参数化　　论文式 (4)(10)(11)(12)",
    r"$L_{t-1}-C=\mathrm{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\Vert^2\right]$")
box(ys[5], "⑥ 简化目标　　论文式 (14)，算法 1、2",
    r"$L_{\mathrm{simple}}=\mathrm{E}_{t,x_0,\epsilon}\left[\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\Vert^2\right]$")
box(ys[6], "⑦ 代码　　04_ddpm_toy2d.py 第 60、108、116、119 行",
    "loss = mse(model(x_t, t), eps)；采样一步 = mean + σ_t·randn", fc="#EDF6F2", ec=TEAL, fs=7.5)
arrow(ys[0], ys[1] + H, r"不等式（定理 1）：缺口 $=\mathrm{KL}(q(x_{1:T}|x_0)\,\Vert\,p_\theta(x_{1:T}|x_0))\geq0$", TEAL)
arrow(ys[1], ys[2] + H, "恒等式（定理 2）：把 x₀ 凑进条件，伸缩相消", TEAL)
arrow(ys[2], ys[3] + H, r"恒等式（定理 3，后验是高斯）+ 建模选择 ②：$\Sigma_\theta=\sigma_t^2I$（定理 4）", CORAL)
arrow(ys[3], ys[4] + H, r"恒等式（定理 5）：用式 (4) 把 $x_0$ 换成 $\epsilon$，再让 $\mu_\theta$ 取同样形状", TEAL)
arrow(ys[4], ys[5] + H, "选择 ③：权重改成 1；近似 ④：L₀ 并入；L_T 是常数；随机抽一个 t（无偏）", CORAL)
arrow(ys[5], ys[6] + H, "一行 MSE；采样按算法 2 逐步抽样", TEAL)
note(ys[1], "近似 ①：q 固定为前向过程，不含参数\n缺口一般不为 0，L 是上界\n（对 −log p 而言）", CORAL)
note(ys[2], "恒等变形，没有新的近似\nL_T 与 θ 无关；每个 KL 都是\n两个高斯之间的，可闭式算", TEAL)
note(ys[3], "σ_t² 取 β_t 或 β̃_t，不训练\n预测 μ̃ 是最直接的参数化", CORAL)
note(ys[4], "只是换写法：预测 μ̃、x₀、ε\n三者线性互换（命题 4）", TEAL)
note(ys[5], "丢掉权重后不再是变分界\n样本质量更好（Table 1、2）", CORAL)
note(ys[6], "绿色箭头是恒等式\n橙色是建模选择或近似", GRAY)
fig.savefig(OUT / "r1_fig1_chain.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ r1_fig2: the swiss roll under the forward process
# the same swiss roll as code/04_ddpm_toy2d.py (NumPy replica of scikit-learn's make_swiss_roll, bit for bit)
g = np.random.RandomState(0)
tt = 1.5 * np.pi * (1 + 2 * g.uniform(size=3000))
yy = 21 * g.uniform(size=3000)
X3 = np.vstack((tt * np.cos(tt), yy, tt * np.sin(tt))) + 0.3 * g.standard_normal(size=(3, 3000))
X = X3.T[:, [0, 2]]
X = (X - X.mean(0)) / X.std()
snaps = [0, 50, 200, 500, 1000]
fig, axes = plt.subplots(1, len(snaps), figsize=(W, 4.8 * CM))
eps2 = rng.standard_normal(X.shape)
for ax, t in zip(axes, snaps):
    if t == 0:
        Xt, sa, sb = X, 1.0, 0.0
    else:
        sa, sb = np.sqrt(abar[idx(t)]), np.sqrt(1 - abar[idx(t)])
        Xt = sa * X + sb * eps2
    ax.scatter(Xt[:, 0], Xt[:, 1], s=1.2, color=BLUE if t < 1000 else GRAY, alpha=0.5, linewidths=0)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$t={t}$\n" + r"$\sqrt{\bar\alpha_t}=$" + f"{sa:.3f}\n" + r"$\sqrt{1-\bar\alpha_t}=$" + f"{sb:.3f}", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "r1_fig2_forward_swissroll.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ r1_fig3: the posterior as prior x likelihood (1-D)
fig, axes = plt.subplots(1, 2, figsize=(W, 6.0 * CM))
cases = [(2, 1.0, 1.02), (100, 1.0, 1.2)]
for k, (ax, (t, x0v, xtv)) in enumerate(zip(axes, cases)):
    i = idx(t)
    m_prev, s_prev = np.sqrt(abar_prev[i]) * x0v, np.sqrt(1 - abar_prev[i])
    s_lik = np.sqrt(betas[i])
    m_post = coef_x0[i] * x0v + coef_xt[i] * xtv
    lo = min(m_prev - 3.5 * s_prev, xtv / np.sqrt(alphas[i]) - 3.5 * s_lik)
    hi = max(m_prev + 3.5 * s_prev, xtv / np.sqrt(alphas[i]) + 3.5 * s_lik)
    gx = np.linspace(lo, hi, 4001)
    prior = np.exp(-0.5 * ((gx - m_prev) / s_prev) ** 2)
    lik = np.exp(-0.5 * ((xtv - np.sqrt(alphas[i]) * gx) / s_lik) ** 2)
    post = prior * lik
    ax.plot(gx, prior / prior.max(), color=BLUE, lw=1.3, label=r"$q(x_{t-1}\mid x_0)$")
    ax.plot(gx, lik / lik.max(), color=GOLD, lw=1.3, label=r"$q(x_t\mid x_{t-1})$，看成 $x_{t-1}$ 的函数")
    ax.plot(gx, post / post.max(), color=CORAL, lw=1.8, label=r"乘积归一化 $=q(x_{t-1}\mid x_t,x_0)$")
    ax.axvline(x0v, color=GRAY, lw=0.8, ls=":")
    ax.axvline(xtv, color=GRAY, lw=0.8, ls="--")
    ax.text(x0v, 1.04, r"$x_0$", ha="center", fontsize=8, color=GRAY)
    ax.text(xtv, 1.04, r"$x_t$", ha="center", fontsize=8, color=GRAY)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel(r"$x_{t-1}$")
    ax.set_title(f"({'ab'[k]}) $t={t}$：均值 {m_post:.3f}，系数 {coef_x0[i]:.3f}·$x_0$ + {coef_xt[i]:.3f}·$x_t$", fontsize=8.5)
    clean(ax)
axes[1].legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "r1_fig3_posterior_1d.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ r1_fig4: loss weights
w_beta = betas ** 2 / (2 * betas * alphas * (1 - abar))
w_tilde = np.full(T, np.nan)
w_tilde[1:] = betas[1:] ** 2 / (2 * beta_tilde[1:] * alphas[1:] * (1 - abar[1:]))
fig, axes = plt.subplots(1, 2, figsize=(W, 6.0 * CM))
ax = axes[0]
ts = np.arange(1, T + 1)
ax.semilogy(ts, w_beta, color=BLUE, lw=1.4, label=r"$\sigma_t^2=\beta_t$")
ax.semilogy(ts[1:], w_tilde[1:], color=PURPLE, lw=1.4, ls="--", label=r"$\sigma_t^2=\tilde\beta_t$（$t\geq2$）")
ax.axhline(1.0, color=CORAL, lw=1.4, label=r"$L_{\mathrm{simple}}$：权重恒为 1")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"式 (12) 的权重 $\beta_t^2/\left(2\sigma_t^2\alpha_t(1-\bar\alpha_t)\right)$", fontsize=8)
ax.set_title("(a) 变分界里 ε 损失的权重：小 t 大，大 t 小", fontsize=8.5)
ax.legend(fontsize=7, frameon=False, loc="upper right")
clean(ax)
ax = axes[1]
ax.semilogy(ts, (1 - abar) / abar, color=TEAL, lw=1.4)
ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$(1-\bar\alpha_t)/\bar\alpha_t$", fontsize=8)
ax.set_title(r"(b) $\Vert x_0-\hat x_0\Vert^2=\frac{1-\bar\alpha_t}{\bar\alpha_t}\Vert\epsilon-\epsilon_\theta\Vert^2$ 的系数", fontsize=8.5)
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "r1_fig4_weights.png", dpi=200)
plt.close(fig)
print("saved:", ", ".join(sorted(p.name for p in OUT.glob("r1_fig*.png"))))
