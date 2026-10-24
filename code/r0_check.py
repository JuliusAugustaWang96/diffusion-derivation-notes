# -*- coding: utf-8 -*-
"""r0_check.py —— 用数值方法核对论文精读 R0（VAE 原论文推导精讲）中出现的全部数字。

玩具例子与 L2、D2–D3 相同：p(z) = N(0,1)，p(x|z) = N(2z, 1)，观测 x = 3。
  (a) 定理 1（R0 第 3 章）：log p(x)、真实后验，四个 q 的 ELBO 与缺口，缺口等于 KL(q ‖ 真实后验)
  (b) 命题 1（R0 第 4.3 节、图 6(a)）：重建项与 KL 项的闭式、三个方差处的数值、和的最小点
  (c) 引理 2（R0 第 4.2 节、图 6(b)）：u − 1 − log u 的四个值
  (d) 图 4(c)：80 个专属高斯叠加后与 N(0, I) 的 KL
  (e) 定理 3（R0 第 5 章）：数据分布与模型不一致、编码器也不是真实后验时，联合 KL 的两种拆法逐项数值积分
  (f) 定理 4 的白话（R0 第 6.3 节）：两种梯度估计的单样本标准差 √30 与 2
  (g) 命题 2（R0 第 7.4 节）：以先验为 q 时 IWAE 界 L_k 随 k 增大、不超过 log p(x)
  (h) R0 第 7.3 节引用的 code/02_vae_mnist.py 行号，以及从 D2–D3 第 8 章引用的训练数字

运行：python code/r0_check.py      （在仓库根目录执行；只需 NumPy，约 5 秒完成）
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半。随机数的种子都固定，结果可复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ ε 等符号会报错
import math, pathlib
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
HERE = pathlib.Path(__file__).resolve().parent
ok_all = True
A, S0, X = 2.0, 1.0, 3.0                      # p(x|z) = N(a z, s0²)，观测 x


def report(name, expected, computed, tol):
    global ok_all
    err = abs(float(expected) - float(computed))
    passed = err <= tol
    ok_all &= passed
    print(f"  {name:<46s} 应为 {float(expected): .6g}   计算得 {float(computed): .6g}   误差 {err:.1e}   "
          f"{'通过' if passed else '未通过'}")


def check(name, passed, detail):
    global ok_all
    ok_all &= bool(passed)
    print(f"  {name:<46s} {detail}   {'通过' if passed else '未通过'}")


def kl_gauss(m1, v1, m2, v2):
    """KL(N(m1, v1) ‖ N(m2, v2))，一维，第二个参数是方差。"""
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)


def recon_neg(x, m, s2):
    """−E_q[log p(x|z)]，q = N(m, s²)：0.5((x − a m)² + a² s²)/s0² + 0.5 log(2π s0²)。"""
    return 0.5 * ((x - A * m) ** 2 + A ** 2 * s2) / S0 + 0.5 * np.log(2 * np.pi * S0)


def elbo(x, m, s2):
    return -recon_neg(x, m, s2) - kl_gauss(m, s2, 0.0, 1.0)


print("(a) 定理 1：证据分解")
log_px = -X ** 2 / (2 * (A ** 2 + S0)) - 0.5 * math.log(2 * math.pi * (A ** 2 + S0))
post_v = 1 / (1 + A ** 2 / S0)
post_m = post_v * A * X / S0
report("log p(x)（正文 −2.6237）", -2.6237, log_px, 5e-5)
report("真实后验的均值（正文 1.2）", 1.2, post_m, 1e-12)
report("真实后验的方差（正文 0.2）", 0.2, post_v, 1e-12)
for m, v, e_txt, g_txt in ((0.0, 1.0, -7.4189, 4.7953), (1.2, 1.0, -3.8189, 1.1953), (1.2, 0.05, -2.9418, 0.3181),
                           (1.2, 0.2, -2.6237, 0.0)):
    e = elbo(X, m, v)
    report(f"q = N({m}, {v})：ELBO（正文 {e_txt}）", e_txt, e, 5e-5)
    report(f"q = N({m}, {v})：缺口 log p(x) − ELBO（正文 {g_txt}）", g_txt, log_px - e, 5e-5)
    report(f"q = N({m}, {v})：缺口 = KL(q ‖ 真实后验)", log_px - e, kl_gauss(m, v, post_m, post_v), 1e-12)

print("(b) 命题 1：固定均值时的拉锯")
report("重建项的常数部分（正文 1.0989）", 1.0989, recon_neg(X, 1.2, 0.0), 5e-5)
report("重建项对 s² 的斜率 a²/(2 s0²)（正文 2）", 2.0, A ** 2 / (2 * S0), 0)
for s2, r_txt, k_txt, t_txt in ((0.2, 1.4989, 1.1247, 2.6237), (0.05, 1.1989, 1.7429, 2.9418), (1.0, 3.0989, 0.7200, 3.8189)):
    r, k = recon_neg(X, 1.2, s2), kl_gauss(1.2, s2, 0.0, 1.0)
    report(f"s² = {s2}：重建项（正文 {r_txt}）", r_txt, r, 5e-5)
    report(f"s² = {s2}：KL 项（正文 {k_txt}）", k_txt, k, 5e-5)
    report(f"s² = {s2}：两项之和（正文 {t_txt}）", t_txt, r + k, 5e-5)
grid = np.linspace(0.01, 2.0, 199001)
best = grid[np.argmin(recon_neg(X, 1.2, grid) + kl_gauss(1.2, grid, 0.0, 1.0))]
report("网格搜索得到的最小点 s²", post_v, best, 1e-5)
mg = np.linspace(-1, 3, 400001)
best_m = mg[np.argmin(recon_neg(X, mg, 0.2) + kl_gauss(mg, 0.2, 0.0, 1.0))]
report("固定 s² = 0.2 时，网格搜索得到的最小点 m", post_m, best_m, 1e-5)

print("(c) 引理 2：u − 1 − log u")
for u, txt in ((0.05, 2.0457), (0.2, 0.8094), (1.0, 0.0), (3.0, 0.9014)):
    report(f"u = {u}（正文 {txt}）", txt, u - 1 - math.log(u), 5e-5)

print("(d) 图 4(c)：80 个专属高斯叠加")
K, TAU2, SIG2 = 80, 0.6, 0.4
mu_k = np.random.default_rng(80).standard_normal((K, 2)) * math.sqrt(TAU2)      # 与 make_figures_r0.py 相同的抽样
g = np.linspace(-7, 7, 1401)
GX, GY = np.meshgrid(g, g, indexing="ij")
mix = np.zeros_like(GX)
for m in mu_k:
    mix += np.exp(-((GX - m[0]) ** 2 + (GY - m[1]) ** 2) / (2 * SIG2)) / (2 * np.pi * SIG2)
mix /= K
std2 = np.exp(-(GX ** 2 + GY ** 2) / 2) / (2 * np.pi)
dA = (g[1] - g[0]) ** 2
report("混合的总概率", 1.0, mix.sum() * dA, 1e-6)
kl_mix = np.sum(mix * (np.log(mix) - np.log(std2))) * dA
report("叠加后与 N(0, I) 的 KL（正文 0.028）", 0.028, kl_mix, 0.0005)

print("(e) 定理 3：联合 KL 的两种拆法（数据 N(0.5, 4)，编码器 q(z|x) = N(0.3x + 0.1, 0.3)）")
DM, DV, QS2 = 0.5, 4.0, 0.3
xg = np.linspace(-15, 16, 3101)
zg = np.linspace(-9, 9, 1801)
XX, ZZ = np.meshgrid(xg, zg, indexing="ij")
log_pdata = -0.5 * (XX - DM) ** 2 / DV - 0.5 * np.log(2 * np.pi * DV)
log_q = -0.5 * (ZZ - (0.3 * XX + 0.1)) ** 2 / QS2 - 0.5 * np.log(2 * np.pi * QS2)
log_joint_model = -0.5 * ZZ ** 2 - 0.5 * np.log(2 * np.pi) - 0.5 * (XX - A * ZZ) ** 2 / S0 - 0.5 * np.log(2 * np.pi * S0)
w = np.exp(log_pdata + log_q)
kl_joint = trap(trap(w * (log_pdata + log_q - log_joint_model), zg, axis=1), xg)
pdata_x = np.exp(-0.5 * (xg - DM) ** 2 / DV) / np.sqrt(2 * np.pi * DV)
mean_elbo = trap(pdata_x * elbo(xg, 0.3 * xg + 0.1, QS2), xg)
H_data = 0.5 * math.log(2 * math.pi * math.e * DV)
split_a = -mean_elbo - H_data
split_b = kl_gauss(DM, DV, 0.0, A ** 2 + S0) + trap(pdata_x * kl_gauss(0.3 * xg + 0.1, QS2, 0.4 * xg, 0.2), xg)
report("联合 KL（二维数值积分）与拆法 A 相同", kl_joint, split_a, 1e-6)
report("拆法 B：KL(数据 ‖ 模型) + 平均缺口，与拆法 A 相同", split_a, split_b, 1e-9)
check("两项都不小于 0", kl_gauss(DM, DV, 0.0, A ** 2 + S0) >= 0 and split_b >= kl_gauss(DM, DV, 0.0, A ** 2 + S0),
      f"联合 KL = {split_a:.4f}")

print("(f) 两种梯度估计（f(z) = z²，q = N(μ, 1)，μ = 1）")
report("得分函数估计的单样本标准差 √30（正文 5.48）", 5.48, math.sqrt(30), 0.005)
report("两者之比 √30 / 2（正文 2.7）", 2.7, math.sqrt(30) / 2, 0.05)
eps = np.random.default_rng(0).standard_normal(2_000_000)
z = 1.0 + eps
report("重参数化估计 2(μ + ε) 的标准差（模拟，理论 2）", 2.0, (2 * z).std(), 0.01)
report("得分函数估计 z²(z − μ) 的标准差（模拟，理论 √30）", math.sqrt(30), (z ** 2 * (z - 1.0)).std(), 0.02)

print("(g) 命题 2：IWAE 界，以先验 N(0, 1) 作为 q")
rng = np.random.default_rng(7)
iw = {}
for k, reps in ((1, 2_000_000), (10, 400_000), (100, 40_000), (1000, 4_000)):
    zz = rng.standard_normal((reps, k))
    lw = -0.5 * (X - A * zz) ** 2 / S0 - 0.5 * np.log(2 * np.pi * S0)          # log [p(x,z)/q(z)] = log p(x|z)
    mx = lw.max(1, keepdims=True)
    lk = mx[:, 0] + np.log(np.exp(lw - mx).mean(1))
    iw[k] = (lk.mean(), lk.std() / math.sqrt(reps))
    print(f"    L_{k:<4d} = {iw[k][0]:.4f} ± {iw[k][1]:.4f}")
report("L_1 与 q = N(0, 1) 时的 ELBO 相同（正文 −7.42）", elbo(X, 0.0, 1.0), iw[1][0], 4 * iw[1][1])
for k, txt in ((10, -2.81), (100, -2.64), (1000, -2.626)):
    report(f"L_{k}（正文 {txt}）", txt, iw[k][0], 0.005 + 3 * iw[k][1])
check("L_k 随 k 增大，并且不超过 log p(x) = −2.6237",
      iw[1][0] < iw[10][0] < iw[100][0] < iw[1000][0] < log_px, "")

print("(h) R0 第 7.3 节引用的行号与 D2–D3 的训练数字")
src = (HERE / "02_vae_mnist.py").read_text(encoding="utf-8").splitlines()
cited = {115: "def encode", 116: "h = self.enc(x)", 117: "return self.mu(h), self.logvar(h)", 120: "def reparameterize",
         122: "std = torch.exp(0.5 * logvar)", 123: "eps = torch.randn_like(std)", 124: "return mu + std * eps",
         135: "def kl_to_std_normal", 137: "0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar)",
         142: "binary_cross_entropy_with_logits", 174: "loss = rec + kl", 175: "loss.backward()"}
for line, text in cited.items():
    check(f"02_vae_mnist.py 第 {line} 行含“{text}”", text in src[line - 1], "")
d23 = (HERE.parent / "docs" / "D2-D3-elbo-and-vae-loss.md").read_text(encoding="utf-8")
for s in ("181.420", "143.048", "136.607", "6.440", "4.094"):
    check(f"D2–D3 第 8 章的输出含 {s}", s in d23, "")
report("136.61 + 6.44（正文 143.05）", 143.05, 136.61 + 6.44, 1e-9)

print("全部通过" if ok_all else "有未通过的项目")
