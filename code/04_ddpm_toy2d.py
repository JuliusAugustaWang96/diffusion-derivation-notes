# -*- coding: utf-8 -*-
"""04_ddpm_toy2d.py —— 二维玩具分布（瑞士卷）上的 DDPM，从零实现，对应 docs/R1-ddpm-paper-walkthrough.md 第 9 章
（正文引用了本文件的行号，改动后请运行 code/r1_check.py 核对）。Schedule 是 β_t、α_t、ᾱ_t 与后验系数（R1 命题 1、定理 3），
q_sample 是一步到位的前向采样（命题 1），q_posterior 是后验均值与方差（定理 3），p_losses 是 L_simple（命题 5，算法 1），
p_sample 是采样的一步（定理 5、命题 6，算法 2）。数组下标 i = t-1：betas[0] 是 β_1，模型输入的 t 也用下标 i（0..T-1）。
运行：python code/04_ddpm_toy2d.py [--stage forward]   （在仓库根目录执行；--stage forward 只画调度与加噪快照；用时见 README）
输出：code/figs/ 下的 ddpm_alpha_bar.png、ddpm_forward.png、ddpm_samples.png、ddpm_reverse_traj.png（不提交）和 code/ddpm_toy2d.pt。
"""


def make_swiss_roll(n_samples, noise=0.0, random_state=None):
    """与 scikit-learn 的 make_swiss_roll 同一公式、同一随机数顺序（RandomState），结果逐位相同，因此不需要安装 scikit-learn。"""
    g = np.random.RandomState(random_state)
    t = 1.5 * np.pi * (1 + 2 * g.uniform(size=n_samples)); y = 21 * g.uniform(size=n_samples)
    X = np.vstack((t * np.cos(t), y, t * np.sin(t))) + noise * g.standard_normal(size=(3, n_samples))
    return X.T, t
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 ᾱ ε 等符号会报错
import argparse, math, pathlib, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC") if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}] + ["DejaVu Sans"]   # 只用已安装的中文字体
plt.rcParams["axes.unicode_minus"] = False

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE / "figs"
CKPT = HERE / "ddpm_toy2d.pt"


# ------------------------------------------------------------------ 数据
def make_data(n=20000, seed=0):
    """瑞士卷的 (x, z) 两维，归一化到零均值、整体标准差 1。返回 tensor[n,2] 与 (mean, scale)。"""
    pass   # 瑞士卷由文件开头的 make_swiss_roll 生成（与 scikit-learn 逐位相同）
    X, _ = make_swiss_roll(n, noise=0.3, random_state=seed)
    X = X[:, [0, 2]].astype(np.float32)
    mean, scale = X.mean(0), X.std()
    X = (X - mean) / scale
    return torch.from_numpy(X), (mean, float(scale))


# ------------------------------------------------------------------ 噪声调度（D4、D5 的全部常数）
class Schedule:
    def __init__(self, T=1000, beta_1=1e-4, beta_T=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_1, beta_T, T, device=device)          # β_t（线性调度，Ho 2020）
        self.alphas = 1.0 - self.betas                                          # α_t = 1 − β_t
        self.alpha_bar = torch.cumprod(self.alphas, 0)                          # ᾱ_t = ∏_{s≤t} α_s
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])   # ᾱ_{t-1}，ᾱ_0 = 1
        self.sqrt_ab = self.alpha_bar.sqrt()                                    # √ᾱ_t        （D4 均值系数）
        self.sqrt_1m_ab = (1.0 - self.alpha_bar).sqrt()                         # √(1−ᾱ_t)    （D4 噪声系数）
        # D5：后验 q(x_{t-1}|x_t,x_0) = N(coef_x0·x_0 + coef_xt·x_t, β̃_t I)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)             # β̃_t
        self.coef_x0 = self.alpha_bar_prev.sqrt() * self.betas / (1.0 - self.alpha_bar)                     # √ᾱ_{t-1} β_t/(1−ᾱ_t)
        self.coef_xt = self.alphas.sqrt() * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)           # √α_t (1−ᾱ_{t-1})/(1−ᾱ_t)


def q_sample(sch, x0, t, eps):
    """D4：从 q(x_t|x_0) 采样（重参数化 M8）。t 为下标张量 [B]。"""
    return sch.sqrt_ab[t, None] * x0 + sch.sqrt_1m_ab[t, None] * eps


def q_posterior(sch, x0, xt, t):
    """D5：前向后验的均值 μ̃_t 与方差 β̃_t。"""
    mean = sch.coef_x0[t, None] * x0 + sch.coef_xt[t, None] * xt
    var = sch.posterior_var[t, None]
    return mean, var


# ------------------------------------------------------------------ 网络：MLP + 正弦时间嵌入
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=1)          # 正弦位置编码（Transformer 同款）
        return self.mlp(emb)


class EpsNet(nn.Module):
    """ε_θ(x_t, t)：输入 2 维坐标 + 128 维时间嵌入，3 层 256 的 MLP，输出 2 维噪声预测。"""
    def __init__(self, hidden=256, tdim=128):
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        self.net = nn.Sequential(nn.Linear(2 + tdim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=1))


# ------------------------------------------------------------------ 训练目标与采样（D7）
def p_losses(sch, model, x0):
    """D7(d) / Algorithm 1：抽 t、抽 ε、算 x_t、回归 ‖ε − ε_θ(x_t,t)‖²。"""
    B = x0.shape[0]
    t = torch.randint(0, sch.T, (B,), device=x0.device)            # t ~ Uniform{1..T}（下标 0..T-1）
    eps = torch.randn_like(x0)                                      # ε ~ N(0,I)
    xt = q_sample(sch, x0, t, eps)                                  # x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε
    return F.mse_loss(model(xt, t), eps)                            # L_simple


@torch.no_grad()
def p_sample(sch, model, xt, i):
    """D7(e) / Algorithm 2 的一步：i 是下标（t = i+1）。"""
    t = torch.full((xt.shape[0],), i, device=xt.device, dtype=torch.long)
    eps = model(xt, t)
    mean = (xt - sch.betas[i] / sch.sqrt_1m_ab[i] * eps) / sch.alphas[i].sqrt()   # μ_θ = (x_t − β_t/√(1−ᾱ_t) ε_θ)/√α_t
    if i == 0:
        return mean                                                               # t=1 时不加噪
    return mean + sch.posterior_var[i].sqrt() * torch.randn_like(xt)              # σ_t² = β̃_t


@torch.no_grad()
def p_sample_loop(sch, model, n, device, keep=()):
    """从 x_T ~ N(0,I) 走到 x_0；keep 里的下标处保存快照。"""
    x = torch.randn(n, 2, device=device)
    traj = {}
    for i in reversed(range(sch.T)):
        x = p_sample(sch, model, x, i)
        if i in keep:
            traj[i] = x.clone()
    return x, traj


# ------------------------------------------------------------------ 画图
def plot_schedule(sch):
    t = np.arange(1, sch.T + 1)
    ab = sch.alpha_bar.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, ab, label=r"$\bar\alpha_t$（信号保留比例）")
    ax.plot(t, np.sqrt(ab), label=r"$\sqrt{\bar\alpha_t}$（均值系数）")
    ax.plot(t, np.sqrt(1 - ab), label=r"$\sqrt{1-\bar\alpha_t}$（噪声系数）")
    half = int(np.argmax(ab < 0.5)) + 1
    ax.axvline(half, color="grey", ls="--", lw=0.8); ax.text(half + 10, 0.52, rf"$\bar\alpha_t$=0.5 @ t={half}", fontsize=8)
    ax.set_xlabel("t"); ax.set_title(rf"线性调度 $\beta$: 1e-4→0.02, T={sch.T}；$\bar\alpha_T$={ab[-1]:.1e}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / "ddpm_alpha_bar.png", dpi=150); plt.close(fig)
    return half


def plot_forward(sch, X):
    ts = [0, 50, 100, 250, 500, 999]
    fig, ax = plt.subplots(1, len(ts), figsize=(15, 2.8))
    x0 = X[:3000]
    for k, i in enumerate(ts):
        t = torch.full((x0.shape[0],), i, device=x0.device, dtype=torch.long)
        xt = x0 if i == 0 else q_sample(sch, x0, t, torch.randn_like(x0))
        ax[k].scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=2, alpha=0.5)
        ax[k].set_xlim(-3.5, 3.5); ax[k].set_ylim(-3.5, 3.5); ax[k].set_aspect("equal")
        ax[k].set_title(rf"t={i + 1 if i else 0}  $\bar\alpha$={float(sch.alpha_bar[i]) if i else 1:.3f}", fontsize=9)
    fig.suptitle(r"前向加噪 $q(x_t|x_0)$：信号按 $\sqrt{\bar\alpha_t}$ 缩小，噪声按 $\sqrt{1-\bar\alpha_t}$ 增大", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "ddpm_forward.png", dpi=150); plt.close(fig)


def plot_samples(X, samples):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(X[:3000, 0].cpu(), X[:3000, 1].cpu(), s=2, alpha=0.5); ax[0].set_title("真实数据")
    ax[1].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=2, alpha=0.5, color="C1"); ax[1].set_title("DDPM 采样（1000 步祖先采样）")
    for a in ax:
        a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIGS / "ddpm_samples.png", dpi=150); plt.close(fig)


def plot_reverse(traj):
    keys = sorted(traj.keys(), reverse=True)
    fig, ax = plt.subplots(1, len(keys), figsize=(15, 2.8))
    for k, i in enumerate(keys):
        x = traj[i]
        ax[k].scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=2, alpha=0.5, color="C1")
        ax[k].set_xlim(-3.5, 3.5); ax[k].set_ylim(-3.5, 3.5); ax[k].set_aspect("equal")
        ax[k].set_title(f"x_{{{i}}}（走完第 {i + 1} 步后）" if i else "x_0", fontsize=9)
    fig.suptitle("反向过程 p_θ(x_{t-1}|x_t)：从纯噪声逐步去噪", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "ddpm_reverse_traj.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["forward", "all"], default="all")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGS.mkdir(exist_ok=True)
    X, stats = make_data(seed=a.seed)
    X = X.to(dev)
    sch = Schedule(a.T, device=dev)
    half = plot_schedule(sch)
    plot_forward(sch, X)
    print(f"ᾱ_t 掉到 0.5 的步数 t={half}；ᾱ_T={float(sch.alpha_bar[-1]):.2e}；图：ddpm_alpha_bar.png, ddpm_forward.png")
    if a.stage == "forward":
        return

    model = EpsNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, X.shape[0], (a.batch,), device=dev)
        loss = p_losses(sch, model, X[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == 1:
            print(f"step {step:5d}  L_simple {loss.item():.4f}")
    train_time = time.time() - t0

    model.eval()
    t1 = time.time()
    samples, traj = p_sample_loop(sch, model, 3000, dev, keep={999, 750, 500, 250, 100, 0})
    sample_time = time.time() - t1
    plot_samples(X, samples)
    plot_reverse(traj)
    torch.save({"model": model.state_dict(), "T": a.T, "data_mean": stats[0].tolist(), "data_scale": stats[1], "seed": a.seed}, CKPT)
    print(f"训练 {a.steps} 步用时 {train_time:.1f} s；采样 3000 点 × {a.T} 步用时 {sample_time:.1f} s；最终 L_simple {loss.item():.4f}")
    print(f"图：ddpm_samples.png, ddpm_reverse_traj.png；模型已保存到 {CKPT.name}")


if __name__ == "__main__":
    main()
