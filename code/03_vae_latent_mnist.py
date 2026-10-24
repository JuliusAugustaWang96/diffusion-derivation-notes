# -*- coding: utf-8 -*-
"""03_vae_latent_mnist.py —— 看清 VAE 的隐空间，对应 docs/I0-vae-intuition.md 的图 2、图 3、图 6 和其中 MNIST 上的数字。

先用与 02_vae_mnist.py 完全相同的设置训练同一个 VAE（隐变量 2 维、10 个 epoch、批大小 128、学习率 1e-3、种子 0）。
在同一台机器的同一设备上，得到的模型与 02_vae_mnist.py 训练出的逐位相同：两者打印的每个 epoch 的损失应完全一致。然后：
  (1) 图 2，编码器与解码器：一张测试图像经编码器得到 q_φ(z|x)，即隐空间里的一个小椭圆；按重参数化 z = μ + σ⊙ε 抽 5 个 z，
      解码成 5 张图。对照：从先验 N(0,I) 抽 5 个 z，解码成 5 张新图，这是生成。
  (2) 图 3，编码器学的是什么：隐变量只有 2 维，所以可以在网格上把解码器的真实后验 p_θ(z|x) ∝ p(z) p_θ(x|z) 直接算出来，
      与编码器给出的高斯放在一起比较。对前 1000 张测试图像，算出 log p_θ(x)（网格求和）、ELBO（KL 闭式加重建项的
      蒙特卡洛估计）和两者之差，并核对“差 = KL(q ‖ 真实后验)”。
  (3) 图 6，聚合后验：6 万张训练图像的编码器输出叠加成 q_φ(z) = (1/N) Σ_n q_φ(z|x_n)，与先验 N(0,I) 比较；
      核对 E[KL(q(z|x) ‖ p(z))] = I(x;z) + KL(q(z) ‖ p(z))，并给出聚合后验的均值与协方差。
这里的 p_θ(x|z) 是训练时用的逐像素伯努利似然。MNIST 的像素是灰度，它不是严格归一化的分布（见 D2–D3 第 7.4 节），
但 ELBO 与缺口的关系只用到对 z 的积分，照样成立。

运行：python code/03_vae_latent_mnist.py [--device cpu]      （在仓库根目录执行；需要 PyTorch；数据与 02_vae_mnist.py 共用 data/mnist/）
测试图像按从 0 开始的下标编号，“18 号”就是 test_x[18]。
用时：测试机 GPU 约 20 秒，CPU 约 1.5 分钟。随机种子固定，但不同硬件上浮点运算的顺序不同，小数点后几位会不同，
      被选中的测试图像也可能不同（见 README）。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ ε 等符号会报错
import argparse, importlib.util, math, pathlib, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE.parent / "docs" / "figures"
spec = importlib.util.spec_from_file_location("vae02", HERE / "02_vae_mnist.py")
vae02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae02)                      # 只取用 VAE、loss_fn、load_mnist，不运行它的 main()

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC")
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [n for n in CJK_FONTS if n in installed] + ["DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "dejavusans"})
CM = 1 / 2.54
CORAL, GRAY, BLUE = "#D85A30", "#888780", "#185FA5"
LOG2PI = math.log(2 * math.pi)
HDR = (0.5, 0.9, 0.99)                               # 画等高线用的概率质量：最高密度区域含 50%、90%、99%


def gauss_r(p):
    """二维标准高斯中，含概率 p 的圆的半径（马氏距离）：r = sqrt(-2 ln(1 - p))。"""
    return math.sqrt(-2 * math.log(1 - p))


def train(dev, epochs=10, batch=128, zdim=2, seed=0):
    """与 02_vae_mnist.py 的 main() 相同的步骤与随机数调用顺序（跳过其中不消耗随机数的 KL 核对）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    train_x, train_y, test_x, test_y = vae02.load_mnist()
    Xtr = torch.from_numpy(train_x).to(dev)
    model = vae02.VAE(zdim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot_rec = tot_kl = 0.0
        for i in range(0, n, batch):
            xb = Xtr[perm[i:i + batch]]
            logits, mu, logvar = model(xb)
            rec, kl = vae02.loss_fn(logits, xb, mu, logvar)
            loss = rec + kl                                   # 负 ELBO
            opt.zero_grad(); loss.backward(); opt.step()
            tot_rec += rec.item() * xb.shape[0]; tot_kl += kl.item() * xb.shape[0]
        print(f"epoch {ep:2d}  重建项 {tot_rec / n:8.3f}  KL 项 {tot_kl / n:7.3f}  负ELBO {(tot_rec + tot_kl) / n:8.3f}")
    model.eval()
    return model, (train_x, train_y, test_x, test_y)


def log_lik(model, X, Z):
    """log p_θ(x|z) = Σ_i [x_i l_i − softplus(l_i)]，l = 解码器输出的 logits；返回 len(X) × len(Z) 的矩阵。"""
    logits = model.decode(Z)
    return X @ logits.T - F.softplus(logits).sum(1)[None]


def log_prior(Z):
    return -0.5 * (Z ** 2).sum(1) - LOG2PI


def hdr_levels(p):
    """给定网格上的密度 p（已归一化到求和为 1），返回含 HDR 中各概率质量的最高密度区域的密度阈值。"""
    s = np.sort(p.ravel())[::-1]
    c = np.cumsum(s)
    return [s[np.searchsorted(c, m)] for m in HDR]


def thumb(ax, img, title=None):
    ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRAY); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("== 训练（与 02_vae_mnist.py 相同的设置）")
    model, (train_x, train_y, test_x, test_y) = train(dev)
    print(f"训练用时 {time.time() - t0:.1f} s，设备 {dev}")
    Xtr = torch.from_numpy(train_x).to(dev)
    Xte = torch.from_numpy(test_x).to(dev)

    with torch.no_grad():
        mu_tr, lv_tr = model.encode(Xtr)
        mu_te, lv_te = model.encode(Xte)
    sd_tr, sd_te = torch.exp(0.5 * lv_tr), torch.exp(0.5 * lv_te)
    kl_tr = 0.5 * (mu_tr ** 2 + lv_tr.exp() - 1 - lv_tr).sum(1)
    qs = torch.tensor([0.05, 0.5, 0.95], device=dev)
    print("\n== 编码器的输出（训练集 60000 张）")
    print("σ 的 5%、50%、95% 分位，第 1 维：", [round(v, 4) for v in torch.quantile(sd_tr[:, 0], qs).tolist()],
          " 第 2 维：", [round(v, 4) for v in torch.quantile(sd_tr[:, 1], qs).tolist()])
    print(f"μ 的范围：第 1 维 [{mu_tr[:, 0].min().item():.2f}, {mu_tr[:, 0].max().item():.2f}]，"
          f"第 2 维 [{mu_tr[:, 1].min().item():.2f}, {mu_tr[:, 1].max().item():.2f}]")

    # ------------------------------------------------------------ (2) 真实后验：前 1000 张测试图像，网格求和
    print("\n== 真实后验与编码器（前 1000 张测试图像）")
    L = math.ceil(max(mu_te.abs().max().item(), 3.5) + 1.0)
    h = 0.01
    axis = torch.arange(-L, L + h / 2, h, device=dev)
    n1 = axis.numel()
    ZZ = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    print(f"网格：[-{L}, {L}]²，步长 {h}，共 {ZZ.shape[0]} 个点")
    nimg = 1000
    X, mu, sd = Xte[:nimg], mu_te[:nimg], sd_te[:nimg]
    lab = test_y[:nimg]
    m = torch.full((nimg,), -float("inf"), device=dev)
    S0 = torch.zeros(nimg, device=dev); S1 = torch.zeros(nimg, 2, device=dev); S2 = torch.zeros(nimg, 2, 2, device=dev)
    Sin, Bq, Cq = torch.zeros(nimg, device=dev), torch.zeros(nimg, device=dev), torch.zeros(nimg, device=dev)
    r90 = gauss_r(0.9)
    with torch.no_grad():
        for i in range(0, ZZ.shape[0], 32768):
            zc = ZZ[i:i + 32768]
            lj = log_lik(model, X, zc) + log_prior(zc)[None]           # log p(z) + log p(x|z)
            mn = torch.maximum(m, lj.max(1).values)                     # 逐块累加，随时按新的最大值缩放
            r = torch.exp(m - mn)
            S0 *= r; S1 *= r[:, None]; S2 *= r[:, None, None]; Sin *= r
            w = torch.exp(lj - mn[:, None])
            S0 += w.sum(1); S1 += w @ zc; S2 += torch.einsum("nc,ci,cj->nij", w, zc, zc)
            d2 = (((zc[None] - mu[:, None]) / sd[:, None]) ** 2).sum(-1)
            Sin += (w * (d2 <= r90 ** 2)).sum(1)                        # 后验落在 q 的 90% 椭圆内的部分
            lq = -0.5 * d2 - LOG2PI - torch.log(sd).sum(1)[:, None]
            wq = torch.exp(lq) * h * h
            Bq += (wq * (lj - lq)).sum(1); Cq += wq.sum(1)               # 网格上的 E_q[log p(x,z) − log q]
            m = mn
    logZ = m + torch.log(S0) + 2 * math.log(h)                          # log p_θ(x)
    pm = S1 / S0[:, None]
    pc = S2 / S0[:, None, None] - pm[:, :, None] * pm[:, None, :]
    corr = pc[:, 0, 1] / torch.sqrt(pc[:, 0, 0] * pc[:, 1, 1])
    inside = Sin / S0
    elbo = Bq / Cq                                                      # 网格求和得到的 ELBO
    gap = logZ - elbo                                                   # = KL(q ‖ 真实后验)
    # 核对：ELBO 的常规算法，KL 闭式 + 重建项用 1000 个重参数化样本的平均（与训练时的算法相同，只是样本多）
    g = torch.Generator(device=dev).manual_seed(2)
    K = 1000
    kl_q = 0.5 * (mu ** 2 + sd ** 2 - 1 - 2 * torch.log(sd)).sum(1)
    rec, rec_se = torch.zeros(nimg, device=dev), torch.zeros(nimg, device=dev)
    with torch.no_grad():
        for k in range(0, nimg, 50):
            eps = torch.randn(K, 2, device=dev, generator=g)
            for j in range(k, min(k + 50, nimg)):
                ll = log_lik(model, X[j:j + 1], mu[j] + sd[j] * eps)[0]
                rec[j], rec_se[j] = ll.mean(), ll.std() / math.sqrt(K)
    elbo_mc = rec - kl_q
    zdev = (elbo_mc - elbo) / rec_se
    gq = torch.quantile(gap, torch.tensor([0.1, 0.5, 0.9, 0.99], device=dev)).tolist()
    print(f"q 在网格上的总质量：{Cq.min().item():.6f} 到 {Cq.max().item():.6f}（应为 1）")
    print(f"负 ELBO 的平均：{(-elbo).mean().item():.3f}；−log p_θ(x) 的平均：{(-logZ).mean().item():.3f}")
    print(f"缺口 log p(x) − ELBO：平均 {gap.mean().item():.3f}，10%/50%/90%/99% 分位 {[round(v, 3) for v in gq]}，"
          f"最大 {gap.max().item():.2f}")
    print(f"缺口小于 1 nat 的比例 {(gap < 1).float().mean().item():.3f}，大于 10 nat 的比例 {(gap > 10).float().mean().item():.3f}")
    print(f"核对 ELBO：蒙特卡洛估计与网格求和之差除以蒙特卡洛标准误，绝对值最大 {zdev.abs().max().item():.2f}，"
          f"在 3 以内的比例 {(zdev.abs() < 3).float().mean().item():.3f}")
    print(f"后验落在 q 的 90% 椭圆内的比例：中位数 {inside.median().item():.3f}（q 与后验相同时应为 0.900）")
    print(f"后验相关系数的绝对值：中位数 {corr.abs().median().item():.3f}，90% 分位 {torch.quantile(corr.abs(), 0.9).item():.3f}")
    psd = torch.sqrt(torch.diagonal(pc, dim1=1, dim2=2))
    print("后验标准差的中位数：", [round(v, 4) for v in psd.median(0).values.tolist()],
          "；编码器 σ 的中位数：", [round(v, 4) for v in sd.median(0).values.tolist()])
    # 选三张图：缺口为中位数的一张（典型）；后验大部分仍在 q 附近、但倾斜最厉害的一张；缺口最大的一张
    k_typ = int(torch.argsort(gap)[nimg // 2])
    tilt_score = corr.abs() * (inside > 0.5)
    k_tilt = int(torch.argmax(tilt_score))
    k_bad = int(torch.argmax(gap))
    picks = [("典型", k_typ), ("倾斜", k_tilt), ("找错位置", k_bad)]

    def full_post(k):
        with torch.no_grad():
            lj = torch.cat([log_lik(model, X[k:k + 1], ZZ[i:i + 262144])[0] + log_prior(ZZ[i:i + 262144])
                            for i in range(0, ZZ.shape[0], 262144)])
        lp = lj - torch.logsumexp(lj, 0)
        return lp.reshape(n1, n1).cpu().numpy()

    ax_np = axis.cpu().numpy()
    posts = {}
    for name, k in picks:
        lp = full_post(k)
        i0, j0 = np.unravel_index(np.argmax(lp), lp.shape)
        zmode = np.array([ax_np[i0], ax_np[j0]])
        posts[k] = (lp, zmode)
        print(f"{name}：{k} 号（数字 {lab[k]}），缺口 {gap[k].item():.3f}，log p(x) {logZ[k].item():.2f}，"
              f"ELBO {elbo[k].item():.2f}；q 的中心 ({mu[k, 0].item():.3f}, {mu[k, 1].item():.3f})，"
              f"σ ({sd[k, 0].item():.3f}, {sd[k, 1].item():.3f})；后验峰值 ({zmode[0]:.2f}, {zmode[1]:.2f})，"
              f"后验均值 ({pm[k, 0].item():.3f}, {pm[k, 1].item():.3f})，后验标准差 "
              f"({math.sqrt(pc[k, 0, 0].item()):.3f}, {math.sqrt(pc[k, 1, 1].item()):.3f})，相关系数 {corr[k].item():.3f}，"
              f"落在 q 的 90% 椭圆内 {inside[k].item():.3f}")

    # ------------------------------------------------------------ 图 3：真实后验与编码器的高斯
    fig = plt.figure(figsize=(16 * CM, 8.2 * CM))
    titles = {"典型": "(a) 典型：位置大致对，形状有出入", "倾斜": "(b) 后验倾斜：对角高斯转不过来",
              "找错位置": "(c) 编码器找错了位置"}
    for t, (name, k) in enumerate(picks):
        lp, zmode = posts[k]
        p = np.exp(lp)
        muk, sdk = mu[k].cpu().numpy(), sd[k].cpu().numpy()
        lo = np.minimum(muk - 3.5 * sdk, zmode - 0.12)
        hi = np.maximum(muk + 3.5 * sdk, zmode + 0.12)
        half = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1], 0.36) * 1.1
        cen = 0.5 * (lo + hi)
        ax = fig.add_axes([0.045 + t * 0.33, 0.33, 0.27, 0.45])
        si = np.abs(ax_np - cen[0]) <= half
        sj = np.abs(ax_np - cen[1]) <= half
        lv = hdr_levels(p)
        ax.contourf(ax_np[si], ax_np[sj], p[np.ix_(si, sj)].T, levels=[lv[2], lv[1], lv[0], p.max() * 1.01],
                    colors=["#DCEAF7", "#9CC3E6", "#3F84C4"])
        for prob, lw in zip(HDR, (1.3, 0.9, 0.6)):
            r = gauss_r(prob)
            ax.add_patch(Ellipse(muk, 2 * r * sdk[0], 2 * r * sdk[1], fill=False, ec=CORAL, lw=lw))
        ax.plot(*muk, "+", color=CORAL, ms=7, mew=1.3)
        ax.plot(*zmode, "x", color=BLUE, ms=5, mew=1.2)
        ax.set_xlim(cen[0] - half, cen[0] + half); ax.set_ylim(cen[1] - half, cen[1] + half)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6.5, length=2)
        ax.set_xlabel("$z_1$", fontsize=7, labelpad=1); ax.set_ylabel("$z_2$", fontsize=7, labelpad=0)
        ax.set_title(titles[name] + f"\n{k} 号测试图像，缺口 {gap[k].item():.2f} nat", fontsize=7.5)
        with torch.no_grad():
            dec = torch.sigmoid(model.decode(torch.tensor(np.stack([muk, zmode]), device=dev, dtype=torch.float32))).cpu().numpy()
        for j, (img, cap) in enumerate(((test_x[k], "图像 x"), (dec[0], "解码\nq 的中心"), (dec[1], "解码\n后验峰值"))):
            tax = fig.add_axes([0.06 + t * 0.33 + j * 0.088, 0.085, 0.07, 0.137])
            thumb(tax, img)
            tax.set_xlabel(cap, fontsize=6.2, labelpad=1, linespacing=1.0)
    fig.text(0.5, 0.965, "蓝色：解码器的真实后验 p(z|x)，由深到浅分别含 50%、90%、99% 的概率；"
             "橙色椭圆：编码器给出的 q(z|x)，同样含 50%、90%、99%。", ha="center", fontsize=6.8, color="#333333")
    fig.text(0.5, 0.925, "橙色 + 是 q 的中心，蓝色 × 是后验峰值。下方三张小图：测试图像、解码 q 的中心、解码后验峰值。",
             ha="center", fontsize=6.8, color="#333333")
    fig.savefig(FIGS / "i0_fig3_posterior_vs_encoder.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------ 图 2：编码器与解码器（训练路径与生成路径）
    k2 = k_typ
    g2 = torch.Generator(device=dev).manual_seed(3)
    eps5 = torch.randn(5, 2, device=dev, generator=g2)
    z_tr = mu[k2] + sd[k2] * eps5                                        # 训练路径：z = μ + σ ⊙ ε
    z_gen = torch.randn(5, 2, device=dev, generator=g2)                  # 生成路径：z ~ N(0, I)
    with torch.no_grad():
        img_tr = torch.sigmoid(model.decode(z_tr)).cpu().numpy()
        img_gen = torch.sigmoid(model.decode(z_gen)).cpu().numpy()
    print(f"\n== 图 2：{k2} 号测试图像（数字 {lab[k2]}），μ = ({mu[k2, 0].item():.3f}, {mu[k2, 1].item():.3f})，"
          f"σ = ({sd[k2, 0].item():.4f}, {sd[k2, 1].item():.4f})")
    print("   ε：", [[round(v, 3) for v in r] for r in eps5.tolist()])
    print("   训练路径的 z：", [[round(v, 3) for v in r] for r in z_tr.tolist()])
    print("   生成路径的 z：", [[round(v, 3) for v in r] for r in z_gen.tolist()])
    fig = plt.figure(figsize=(16 * CM, 7.2 * CM))
    ax_in = fig.add_axes([0.0, 0.38, 0.11, 0.244])
    thumb(ax_in, test_x[k2])
    ax_in.set_title(f"测试图像 x\n（{k2} 号）", fontsize=7.5)
    axm = fig.add_axes([0.215, 0.11, 0.31, 0.72])
    mte = mu_te.cpu().numpy()
    axm.scatter(mte[:, 0], mte[:, 1], c=test_y, cmap="tab10", s=0.6, alpha=0.35, linewidths=0, rasterized=True)
    tt = np.linspace(0, 2 * np.pi, 200)
    for prob in HDR:
        r = gauss_r(prob)
        axm.plot(r * np.cos(tt), r * np.sin(tt), color=GRAY, ls="--", lw=0.7)
    zg = z_gen.cpu().numpy()
    axm.plot(zg[:, 0], zg[:, 1], "x", color="black", ms=5, mew=1.2)
    for j in range(5):
        axm.annotate(chr(ord("a") + j), zg[j], xytext=(3, 2), textcoords="offset points", fontsize=7)
    muk, sdk = mu[k2].cpu().numpy(), sd[k2].cpu().numpy()
    box = round(float(5 * sdk.max()), 2)                                # 放大框的半宽：q 的标准差的 5 倍
    axm.add_patch(Rectangle(muk - box, 2 * box, 2 * box, fill=False, ec=CORAL, lw=0.9))
    axm.set_xlim(-4, 4); axm.set_ylim(-4, 4); axm.set_aspect("equal")
    axm.tick_params(labelsize=6.5, length=2)
    axm.set_xlabel("$z_1$（横轴）、$z_2$（纵轴）", fontsize=7, labelpad=1)
    axm.set_title("隐空间：彩色点是 1 万张测试图像的 μ\n灰虚线：先验 N(0,I)，含 50%、90%、99% 的概率；× 是生成时抽的 z",
                  fontsize=6.8)
    axz = fig.add_axes([0.545, 0.47, 0.135, 0.30])
    for prob, lw in zip(HDR, (1.2, 0.8, 0.5)):
        r = gauss_r(prob)
        axz.add_patch(Ellipse(muk, 2 * r * sdk[0], 2 * r * sdk[1], fill=False, ec=CORAL, lw=lw))
    zt = z_tr.cpu().numpy()
    axz.plot(*muk, "+", color=CORAL, ms=6, mew=1.2)
    axz.plot(zt[:, 0], zt[:, 1], "o", color=BLUE, ms=2.6)
    offs = ((3, 1), (-7, 1), (3, 1), (3, -6), (3, 1))                  # 第 2、5 个点挨得近，标签分到两侧
    for j in range(5):
        axz.annotate(str(j + 1), zt[j], xytext=offs[j], textcoords="offset points", fontsize=6.5, color=BLUE)
    axz.set_xlim(muk[0] - box, muk[0] + box); axz.set_ylim(muk[1] - box, muk[1] + box); axz.set_aspect("equal")
    axz.set_xticks([]); axz.set_yticks([])
    for s in axz.spines.values():
        s.set_color(CORAL)
    axz.set_title("放大：q(z|x) 与 5 个 z", fontsize=6.5, pad=2, color=CORAL)
    for yy in (1, 0):
        con = ConnectionPatch(xyA=(muk[0] + box, muk[1] - box + 2 * box * yy), coordsA="data", axesA=axm,
                              xyB=(0, yy), coordsB="axes fraction", axesB=axz, color=CORAL, lw=0.5, alpha=0.7)
        fig.add_artist(con)
    fig.text(0.147, 0.56, "编码器\n给出 μ、σ", fontsize=7, ha="center", color=BLUE)
    fig.add_artist(matplotlib.patches.FancyArrowPatch((0.115, 0.51), (0.182, 0.51), transform=fig.transFigure,
                                                      arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1.0))
    rows = ((img_tr, "训练路径：z = μ + σ⊙ε，ε ~ N(0,I)", "12345", 0.60, (0.683, 0.62)),
            (img_gen, "生成路径：z 从先验 N(0,I) 抽，不用编码器", "abcde", 0.17, (0.525, 0.235)))
    for imgs, cap, lab_, y, start in rows:
        fig.text(0.85, y + 0.165, cap, fontsize=6.8, ha="center")
        for j in range(5):
            tax = fig.add_axes([0.715 + j * 0.057, y, 0.052, 0.116])
            thumb(tax, imgs[j])
            tax.set_xlabel(lab_[j], fontsize=6.3, labelpad=1)
        fig.add_artist(matplotlib.patches.FancyArrowPatch(start, (0.712, y + 0.058), transform=fig.transFigure,
                                                          arrowstyle="-|>", mutation_scale=8, color=BLUE, lw=0.9))
        fig.text(0.70, y + 0.085 if y > 0.5 else y + 0.10, "解码器", fontsize=6.5, ha="right", color=BLUE)
    fig.savefig(FIGS / "i0_fig2_encoder_decoder_mnist.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------ (3) 聚合后验
    print("\n== 聚合后验（训练集 60000 张）")
    N = mu_tr.shape[0]
    inv = torch.exp(-lv_tr)
    cst = -0.5 * lv_tr.sum(1) - LOG2PI

    def log_qagg(z, chunk=1024):
        out = []
        for i in range(0, z.shape[0], chunk):
            d2 = ((z[i:i + chunk, None, :] - mu_tr[None]) ** 2 * inv[None]).sum(-1)
            out.append(torch.logsumexp(cst[None] - 0.5 * d2, 1) - math.log(N))
        return torch.cat(out)

    g3 = torch.Generator(device=dev).manual_seed(4)
    M = 20000
    idx = torch.randint(0, N, (M,), device=dev, generator=g3)
    zs = mu_tr[idx] + sd_tr[idx] * torch.randn(M, 2, device=dev, generator=g3)     # 从聚合后验抽样
    with torch.no_grad():
        diff = log_qagg(zs) - log_prior(zs)
    kl_agg, kl_se = diff.mean().item(), diff.std().item() / math.sqrt(M)
    ekl = kl_tr.mean().item()
    print(f"E[KL(q(z|x) ‖ p(z))] = {ekl:.4f}（训练集平均）")
    print(f"KL(q(z) ‖ p(z)) = {kl_agg:.4f} ± {kl_se:.4f}（{M} 个样本的蒙特卡洛估计及其标准误）")
    print(f"I(x;z) = E[KL] − KL(q(z)‖p(z)) = {ekl - kl_agg:.4f}；上限 log N = {math.log(N):.4f}")
    mean_agg = mu_tr.mean(0)
    cov_mu = torch.cov(mu_tr.T)
    es2 = lv_tr.exp().mean(0)
    print(f"聚合后验的均值 ({mean_agg[0].item():.3f}, {mean_agg[1].item():.3f})")
    print(f"Cov(μ) = [[{cov_mu[0, 0].item():.3f}, {cov_mu[0, 1].item():.3f}], [{cov_mu[1, 0].item():.3f}, {cov_mu[1, 1].item():.3f}]]，"
          f"E[σ²] = ({es2[0].item():.4f}, {es2[1].item():.4f})")
    print(f"聚合后验的方差 = Var(μ) + E[σ²] = ({cov_mu[0, 0].item() + es2[0].item():.3f}, {cov_mu[1, 1].item() + es2[1].item():.3f})，"
          f"相关系数 {cov_mu[0, 1].item() / math.sqrt((cov_mu[0, 0] + es2[0]).item() * (cov_mu[1, 1] + es2[1]).item()):.3f}")
    zp = torch.randn(M, 2, device=dev, generator=g3)                                 # 从先验抽样
    with torch.no_grad():
        ratio = log_qagg(zp) - log_prior(zp)
    for rr in (0.5, 0.1):
        print(f"先验抽到的 z 中，聚合后验密度不到先验 {rr} 倍的比例：{(ratio < math.log(rr)).float().mean().item():.4f}")
    hg = 0.025
    gax = torch.arange(-4, 4 + hg / 2, hg, device=dev)
    GZ = torch.stack(torch.meshgrid(gax, gax, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        lqg = log_qagg(GZ).reshape(gax.numel(), gax.numel()).cpu().numpy()
    lpg = log_prior(GZ).reshape(gax.numel(), gax.numel()).cpu().numpy()
    qg = np.exp(lqg)
    print(f"网格 [-4,4]²、步长 {hg}：聚合后验的质量 {qg.sum() * hg * hg:.4f}，"
          f"网格上的 KL(q(z)‖p(z)) ≈ {np.sum(qg * (lqg - lpg)) * hg * hg:.4f}")
    gnp = gax.cpu().numpy()
    fig = plt.figure(figsize=(16 * CM, 6.6 * CM))
    ext = [gnp[0], gnp[-1], gnp[0], gnp[-1]]
    axa = fig.add_axes([0.035, 0.09, 0.28, 0.70])
    axa.imshow(np.maximum(qg, 1e-3).T, origin="lower", extent=ext, cmap="Blues",
               norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=qg.max()))
    for prob in HDR:
        r = gauss_r(prob)
        axa.plot(r * np.cos(tt), r * np.sin(tt), color=CORAL, ls="--", lw=0.7)
    bw = 0.6
    zb = mu[k2].cpu().numpy() - bw / 2                                   # 黑框以图 2 那张测试图像的 μ 为中心
    axa.add_patch(Rectangle(zb, bw, bw, fill=False, ec="black", lw=0.8))
    axa.set_title("(a) 6 万个 q(z|x) 叠加：聚合后验 q(z)\n颜色按对数刻度；橙色虚线：先验 N(0,I)", fontsize=7)
    axb = fig.add_axes([0.36, 0.09, 0.28, 0.70])
    mtr, str_ = mu_tr.cpu().numpy(), sd_tr.cpu().numpy()
    sel = np.nonzero((mtr[:, 0] > zb[0]) & (mtr[:, 0] < zb[0] + bw) & (mtr[:, 1] > zb[1]) & (mtr[:, 1] < zb[1] + bw))[0]
    show = np.random.default_rng(5).choice(sel, size=min(80, len(sel)), replace=False)
    cmap = plt.get_cmap("tab10")
    r50 = gauss_r(0.5)
    for n_ in show:
        c = cmap(int(train_y[n_]))
        axb.add_patch(Ellipse(mtr[n_], 2 * r50 * str_[n_, 0], 2 * r50 * str_[n_, 1], fc=c, ec=c, lw=0.5, alpha=0.35))
    axb.set_xlim(zb[0], zb[0] + bw); axb.set_ylim(zb[1], zb[1] + bw); axb.set_aspect("equal")
    axb.tick_params(labelsize=6, length=2)
    axb.set_title(f"(b) 放大黑框（框内 {len(sel)} 张训练图像）\n随机画其中 {len(show)} 个 q(z|x)，颜色表示数字",
                  fontsize=7)
    axc = fig.add_axes([0.675, 0.09, 0.28, 0.70])
    lr = np.clip((lqg - lpg) / math.log(10), -1.5, 1.5)
    GX, GY = np.meshgrid(gnp, gnp, indexing="ij")
    lr = np.where(GX ** 2 + GY ** 2 <= gauss_r(0.99) ** 2, lr, np.nan)  # 先验 99% 圆外几乎抽不到，不着色
    cm_ = plt.get_cmap("RdBu_r").copy(); cm_.set_bad("#EEEEEE")
    im = axc.imshow(lr.T, origin="lower", extent=ext, cmap=cm_, vmin=-1.5, vmax=1.5)
    for prob in HDR:
        r = gauss_r(prob)
        axc.plot(r * np.cos(tt), r * np.sin(tt), color="black", ls="--", lw=0.5)
    cax = fig.add_axes([0.958, 0.17, 0.011, 0.54])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["0.1", "1", "10"], fontsize=6)
    cb.ax.set_title("倍", fontsize=6, pad=2)
    axc.set_title("(c) 聚合后验与先验之比\n蓝：更稀；红：更密；灰：99% 圆外", fontsize=7)
    for axx in (axa, axc):
        axx.set_xlim(-4, 4); axx.set_ylim(-4, 4); axx.tick_params(labelsize=6, length=2)
        axx.set_xticks([-4, -2, 0, 2, 4]); axx.set_yticks([-4, -2, 0, 2, 4])
    fig.savefig(FIGS / "i0_fig6_aggregate_posterior.png", dpi=200)
    plt.close(fig)
    print(f"\n总用时 {time.time() - t0:.1f} s；图已保存到 {FIGS.relative_to(HERE.parent).as_posix()}")


if __name__ == "__main__":
    main()
