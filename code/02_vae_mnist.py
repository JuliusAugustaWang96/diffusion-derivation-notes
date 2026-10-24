# -*- coding: utf-8 -*-
"""02_vae_mnist.py —— 在 MNIST 上训练一个变分自编码器，对应 docs/D2-D3-elbo-and-vae-loss.md 第 8 章（正文引用了本文件的行号，改动后请运行 code/d2_d3_check.py 核对）。

结构：编码器 784→512→256→(μ, log σ²)，隐变量维度 2（便于画隐空间散点）；解码器 2→256→512→784（伯努利 logits）。
损失：BCE 重建项（D3(d)） + KL(q_φ(z|x) ‖ N(0,I))（D3(c) 闭式，即 D1(c)；每个 epoch 与 torch.distributions 核对一次）。
输出：docs/figures/ 下的 d2d3_fig5_vae_recon.png（测试图像与重建）、d2d3_fig6_vae_latent.png（隐空间散点）、
      d2d3_fig7_vae_interp.png（两张图之间的隐空间插值）、d2d3_fig8_vae_samples.png（先验采样）。
运行：python code/02_vae_mnist.py [--epochs 10]      （在仓库根目录执行；用时见 README）
数据：MNIST 原始 IDX 文件（约 11 MB），从 torchvision 官方镜像 ossci-datasets.s3.amazonaws.com 下载到 data/mnist/，
      失败则退回 sklearn.datasets.fetch_openml('mnist_784')（需要 scikit-learn）。不依赖 torchvision。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 ᾱ ε 等符号会报错
import argparse, gzip, pathlib, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributions as td
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC") if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}] + ["DejaVu Sans"]   # 只用已安装的中文字体
plt.rcParams["axes.unicode_minus"] = False

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE.parent / "docs" / "figures"
DATA = HERE.parent / "data" / "mnist"
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",            # torchvision 官方镜像
           "https://storage.googleapis.com/cvdf-datasets/mnist/"]        # 备用镜像
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
EXPECT = {"train_x": 16 + 60000 * 784, "train_y": 8 + 60000, "test_x": 16 + 10000 * 784, "test_y": 8 + 10000}   # 解压后字节数


def _read_idx(path, key):
    """解压并校验长度；不完整的文件抛异常。"""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) != EXPECT[key]:
        raise ValueError(f"{path.name} 解压后 {len(raw)} 字节，应为 {EXPECT[key]}")
    return raw


def _download(url, path, attempts=3):
    """流式下载到临时文件，完成后再改名；失败则删除残缺文件并重试。"""
    tmp = path.with_suffix(path.suffix + ".part")
    for k in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(path)
            return
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"  第 {k} 次下载失败：{e}")
    raise RuntimeError("下载失败：" + url)


def load_mnist():
    """返回 (train_x [60000,784] float32 in [0,1], train_y, test_x, test_y)。"""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["train_x"], d["train_y"], d["test_x"], d["test_y"]
    try:
        arrs = {}
        for key, fname in FILES.items():
            path = DATA / fname
            raw = None
            if path.exists():                      # 已有文件先校验，残缺则删掉重下
                try:
                    raw = _read_idx(path, key)
                except Exception as e:
                    print("  已有文件无效，重新下载：", e); path.unlink()
            if raw is None:
                for mirror in MIRRORS:
                    try:
                        print("下载", mirror + fname)
                        _download(mirror + fname, path)
                        raw = _read_idx(path, key)
                        break
                    except Exception as e:
                        print("  镜像失败：", e); path.unlink(missing_ok=True)
                if raw is None:
                    raise RuntimeError("所有镜像都失败：" + fname)
            if key.endswith("_x"):
                arrs[key] = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 784)   # IDX3：16 字节头
            else:
                arrs[key] = np.frombuffer(raw, dtype=np.uint8, offset=8)                     # IDX1：8 字节头
        train_x, train_y, test_x, test_y = arrs["train_x"], arrs["train_y"], arrs["test_x"], arrs["test_y"]
    except Exception as e:  # 镜像不可用时退回 sklearn
        print("镜像下载失败，改用 sklearn.datasets.fetch_openml：", e)
        from sklearn.datasets import fetch_openml
        m = fetch_openml("mnist_784", as_frame=False)
        X, y = m.data.astype(np.uint8), m.target.astype(np.uint8)
        train_x, train_y, test_x, test_y = X[:60000], y[:60000], X[60000:], y[60000:]
    train_x = (train_x.astype(np.float32) / 255.0)
    test_x = (test_x.astype(np.float32) / 255.0)
    np.savez_compressed(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    return train_x, train_y, test_x, test_y


class VAE(nn.Module):
    def __init__(self, zdim=2, h=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, h), nn.ReLU(), nn.Linear(h, 256), nn.ReLU())
        self.mu = nn.Linear(256, zdim)            # 编码器输出：均值 μ_φ(x)
        self.logvar = nn.Linear(256, zdim)        # 编码器输出：log σ_φ²(x)（对角高斯 q_φ(z|x)）
        self.dec = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, h), nn.ReLU(), nn.Linear(h, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """D3(b) 重参数化：z = μ + σ ⊙ ε，ε ~ N(0,I)。梯度可以穿过 μ、σ 回到编码器。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.dec(z)                        # 伯努利 logits（p_θ(x|z) 的参数）

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_to_std_normal(mu, logvar):
    """D3(c) = D1(c) 逐维求和：KL(N(μ,σ²) ‖ N(0,1)) = ½ Σ_j (μ_j² + σ_j² − 1 − log σ_j²)。返回每个样本的 KL。"""
    return 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=1)


def loss_fn(logits, x, mu, logvar):
    """负 ELBO 的单样本估计（D2(c) + D3(b)(d)）：重建项用 BCE（逐像素伯努利），KL 项用闭式。"""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)   # -log p_θ(x|z)
    kl = kl_to_std_normal(mu, logvar)
    return recon.mean(), kl.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--zdim", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()
    train_x, train_y, test_x, test_y = load_mnist()
    Xtr = torch.from_numpy(train_x).to(dev)
    Xte = torch.from_numpy(test_x).to(dev)
    print(f"数据：train {tuple(Xtr.shape)} test {tuple(Xte.shape)}，设备 {dev}")

    model = VAE(a.zdim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot_rec = tot_kl = 0.0
        for i in range(0, n, a.batch):
            xb = Xtr[perm[i:i + a.batch]]
            logits, mu, logvar = model(xb)
            rec, kl = loss_fn(logits, xb, mu, logvar)
            loss = rec + kl                                   # 负 ELBO
            opt.zero_grad(); loss.backward(); opt.step()
            tot_rec += rec.item() * xb.shape[0]; tot_kl += kl.item() * xb.shape[0]
        # 核对：手写 KL 闭式 vs torch.distributions（应为 ~1e-6）
        with torch.no_grad():
            mu, logvar = model.encode(Xte[:512])
            q = td.Independent(td.Normal(mu, torch.exp(0.5 * logvar)), 1)
            p = td.Independent(td.Normal(torch.zeros_like(mu), torch.ones_like(mu)), 1)
            diff = (td.kl_divergence(q, p) - kl_to_std_normal(mu, logvar)).abs().max().item()
        print(f"epoch {ep:2d}  重建项 {tot_rec / n:8.3f}  KL 项 {tot_kl / n:7.3f}  负ELBO {(tot_rec + tot_kl) / n:8.3f}  "
              f"| KL闭式 vs torch 最大差 {diff:.2e}")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        # 1) 重建图
        xb = Xte[:8]
        logits, mu, logvar = model(xb)
        rec = torch.sigmoid(logits)
        fig, ax = plt.subplots(2, 8, figsize=(10, 2.8))
        for j in range(8):
            ax[0, j].imshow(xb[j].view(28, 28).cpu(), cmap="gray"); ax[0, j].axis("off")
            ax[1, j].imshow(rec[j].view(28, 28).cpu(), cmap="gray"); ax[1, j].axis("off")
        ax[0, 0].set_title("测试图像", fontsize=9, loc="left"); ax[1, 0].set_title("重建", fontsize=9, loc="left")
        fig.suptitle("VAE 重建（上：测试图像，下：解码 μ_φ(x)+σ_φ(x)ε 后的均值）", fontsize=10)
        fig.tight_layout(); fig.savefig(FIGS / "d2d3_fig5_vae_recon.png", dpi=150); plt.close(fig)

        # 2) 隐空间散点（测试集 10000 张，取 μ_φ(x)）
        mu, _ = model.encode(Xte)
        mu = mu.cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5.5))
        sc = ax.scatter(mu[:, 0], mu[:, 1], c=test_y, cmap="tab10", s=3, alpha=0.6)
        fig.colorbar(sc, ax=ax, ticks=range(10), label="数字")
        ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_title("隐空间 μ_φ(x)，按数字上色（KL 项把它们拉向原点附近）")
        fig.tight_layout(); fig.savefig(FIGS / "d2d3_fig6_vae_latent.png", dpi=150); plt.close(fig)

        # 3) 隐空间插值：第一张 3 和第一张 7
        ia = int(np.where(test_y == 3)[0][0]); ib = int(np.where(test_y == 7)[0][0])
        za, _ = model.encode(Xte[ia:ia + 1]); zb, _ = model.encode(Xte[ib:ib + 1])
        lam = torch.linspace(0, 1, 10, device=dev).view(-1, 1)
        zs = (1 - lam) * za + lam * zb
        imgs = torch.sigmoid(model.decode(zs)).view(-1, 28, 28).cpu()
        fig, ax = plt.subplots(1, 10, figsize=(12, 1.6))
        for j in range(10):
            ax[j].imshow(imgs[j], cmap="gray"); ax[j].axis("off")
        fig.suptitle("隐空间线性插值：z = (1-λ) z_3 + λ z_7，λ 从 0 到 1", fontsize=10)
        fig.tight_layout(); fig.savefig(FIGS / "d2d3_fig7_vae_interp.png", dpi=150); plt.close(fig)

        # 4) 先验采样：在 N(0,1) 的分位点网格上取 z（zdim=2 时是整张"地图"）
        if a.zdim == 2:
            qs = td.Normal(0.0, 1.0).icdf(torch.linspace(0.05, 0.95, 10))
            g1, g2 = torch.meshgrid(qs, qs, indexing="ij")
            zs = torch.stack([g1.flatten(), g2.flatten()], 1).to(dev)
        else:
            zs = torch.randn(100, a.zdim, device=dev)
        imgs = torch.sigmoid(model.decode(zs)).view(10, 10, 28, 28).cpu()
        canvas = imgs.permute(0, 2, 1, 3).reshape(280, 280)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(canvas, cmap="gray"); ax.axis("off")
        ax.set_title("先验采样：z 取 N(0,1) 分位点网格后解码（D2 的生成过程）", fontsize=10)
        fig.tight_layout(); fig.savefig(FIGS / "d2d3_fig8_vae_samples.png", dpi=150); plt.close(fig)

    print(f"训练用时 {train_time:.1f} s；图已保存到 {FIGS.relative_to(HERE.parent).as_posix()}")


if __name__ == "__main__":
    main()
