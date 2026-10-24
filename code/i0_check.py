# -*- coding: utf-8 -*-
"""i0_check.py —— 用数值方法核对直观篇 I0（VAE 的直观路线）中玩具例子的全部数字，以及正文引用的代码行号。

玩具例子与 L2 图 5(b) 相同：z ~ N(0,1)，x | z ~ N(μ(z), 0.2²)，μ(z) = z + 1.8 sin(2.8z + 1)。数据直接从这个模型抽，
所以“模型完全拟合了数据”这件事在这里精确成立。
  (a) 两个混合（I0 第 5 章、图 5）：x 空间的 p(x) = ∫ p(z) p(x|z) dz 积分为 1；z 空间里，真实后验对数据取平均
      ∫ p(x) p(z|x) dx 正好等于先验 p(z)；抽 2000 个数据点时，平均只差抽样误差
  (b) 互信息的两种算法：I(x;z) = h(x) − h(x|z) = E_x[KL(p(z|x) ‖ p(z))]
  (c) 编码器只能给高斯时（I0 第 5.3 节、图 5(c)）：对每个数据点找使 ELBO 最大的高斯 q，ELBO 用闭式计算（与高斯–埃尔米特
      积分核对）；缺口、E[KL(q ‖ p)]、聚合后验与先验的 KL、互信息；分解 E[KL] = I + KL(q(z) ‖ p(z)) 的两边分别计算
  (d) 聚合后验的方差 = 均值的方差 + 方差的平均（全方差公式）
  (e) 二维标准高斯中含 50%、90%、99% 概率的圆的半径（图 2、图 3、图 6 的等高线）
  (f) 正文引用的 MNIST 数字之间的关系（这些数字本身由 code/03_vae_latent_mnist.py 算出并打印）
  (g) 正文引用的 code/02_vae_mnist.py 行号

运行：python code/i0_check.py      （在仓库根目录执行；只需 NumPy，约 10 秒完成）
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半。随机数的种子都固定，结果可复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ ε 等符号会报错
import math, pathlib
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
HERE = pathlib.Path(__file__).resolve().parent
ok_all = True
A, OMEGA, PHI, S = 1.8, 2.8, 1.0, 0.2          # μ(z) = z + A sin(ωz + φ)，解码器的标准差 S


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


def mu(z):
    return z + A * np.sin(OMEGA * z + PHI)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def elbo_closed(x, m, s):
    """q = N(m, s²) 时玩具模型的 ELBO 闭式：E_q[log p(z)] + E_q[log p(x|z)] + H[q]。
    用到 z ~ N(m, s²) 时 E[sin(ωz+φ)] = c·sin(ωm+φ)，c = exp(−ω²s²/2)；E[z sin(ωz+φ)] = m E[sin] + s² ω c cos(ωm+φ)
    （Stein 引理）；E[sin²(ωz+φ)] = 1/2 − 1/2·exp(−2ω²s²) cos(2ωm+2φ)。"""
    c = np.exp(-0.5 * (OMEGA * s) ** 2)
    e_sin = c * np.sin(OMEGA * m + PHI)
    e_zsin = m * e_sin + s ** 2 * OMEGA * c * np.cos(OMEGA * m + PHI)
    e_sin2 = 0.5 - 0.5 * np.exp(-2 * (OMEGA * s) ** 2) * np.cos(2 * OMEGA * m + 2 * PHI)
    e_mu = m + A * e_sin
    e_mu2 = m ** 2 + s ** 2 + 2 * A * e_zsin + A ** 2 * e_sin2
    e_sq = x ** 2 - 2 * x * e_mu + e_mu2                                        # E_q[(x − μ(z))²]
    return (-0.5 * math.log(2 * math.pi) - 0.5 * (m ** 2 + s ** 2)
            - 0.5 * math.log(2 * math.pi * S ** 2) - e_sq / (2 * S ** 2)
            + 0.5 * np.log(2 * math.pi * math.e * s ** 2))


def fit_gaussians(xs):
    """对每个数据点找使 ELBO 最大的 q = N(m, s²)：先在 (m, log s) 的网格上找，再在最好的点附近逐级加密 5 次。"""
    M0 = np.linspace(-4, 4, 401)
    S0 = np.exp(np.linspace(math.log(0.01), math.log(1.5), 41))
    ms, ss = np.empty(len(xs)), np.empty(len(xs))
    for k in range(0, len(xs), 100):
        x = xs[k:k + 100, None, None]
        e = elbo_closed(x, M0[None, :, None], S0[None, None, :])
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


print("(a) 两个混合：x 空间的 p(x) 与 z 空间的平均后验")
Zc = np.linspace(-6, 6, 2401)
Xc = np.linspace(-6, 7, 2601)
J = npdf(Zc[:, None], 0, 1) * npdf(Xc[None, :], mu(Zc)[:, None], S)          # p(z) p(x|z)，行是 z，列是 x
px = trap(J, Zc, axis=0)                                                       # p(x) = ∫ p(z) p(x|z) dz
report("p(x) 的积分", 1.0, trap(px, Xc), 1e-6)
post = J / px[None, :]                                                         # p(z|x)，每一列是一个 x 的后验
agg = trap(post * px[None, :], Xc, axis=1)                                     # ∫ p(x) p(z|x) dx
dev = np.abs(agg - npdf(Zc, 0, 1)).max()
check("∫ p(x) p(z|x) dx 与 p(z) 的最大差", dev < 1e-5, f"{dev:.1e}（数值积分误差）")
rng = np.random.default_rng(0)
N = 2000
zn = rng.standard_normal(N)
xn = mu(zn) + S * rng.standard_normal(N)                                       # 从模型本身抽数据
zg = np.linspace(-6, 6, 12001)
dz = zg[1] - zg[0]
pz = npdf(zg, 0, 1)
lpost = -0.5 * zg[None] ** 2 - 0.5 * ((xn[:, None] - mu(zg)[None]) / S) ** 2
lpost -= lpost.max(1, keepdims=True)
pp = np.exp(lpost)
pp /= pp.sum(1, keepdims=True) * dz                                            # 2000 个数据点各自的真实后验
avg = pp.mean(0)
kl_avg = np.sum(avg * (np.log(avg) - np.log(pz))) * dz
report("2000 个真实后验的平均与先验的 KL（正文 0.004）", 0.004, kl_avg, 0.0005)
report("平均与先验的最大差（正文 0.05）", 0.05, np.abs(avg - pz).max(), 0.005)
peaks = np.array([np.sum((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > 0.1 * p.max())) for p in pp])
report("有两个以上峰（次峰至少有主峰一成高）的比例（正文 94.5%）", 0.945, np.mean(peaks >= 2), 0.0005)
main = np.array([p[np.abs(zg - zg[np.argmax(p)]) < 0.25].sum() * dz for p in pp])
report("主峰左右 0.25 以内的概率，中位数（正文 0.50）", 0.50, np.median(main), 0.005)

print("(b) 互信息的两种算法")
hx = -trap(px * np.log(np.maximum(px, 1e-300)), Xc)
hxz = 0.5 * math.log(2 * math.pi * math.e * S ** 2)
kl_post = trap(post * (np.log(np.maximum(post, 1e-300)) - np.log(npdf(Zc, 0, 1))[:, None]), Zc, axis=0)
I_exact = trap(px * kl_post, Xc)
report("h(x) − h(x|z)（正文 1.94）", 1.94, hx - hxz, 0.005)
report("E_x[KL(p(z|x) ‖ p(z))] 与上一行相同", hx - hxz, I_exact, 1e-4)

print("(c) 编码器只能给高斯")
for m_, s_, x_ in ((0.3, 0.1, 1.0), (-1.2, 0.5, -0.4), (2.0, 1.2, 3.0)):
    t, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / w.sum()
    z = m_ + s_ * t
    lj = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - 0.5 * ((x_ - mu(z)) / S) ** 2 - math.log(S * math.sqrt(2 * math.pi))
    quad = np.sum(w * lj) + 0.5 * math.log(2 * math.pi * math.e * s_ ** 2)
    report(f"ELBO 闭式 vs 积分（m={m_}, s={s_}, x={x_}）", quad, elbo_closed(x_, m_, s_), 1e-8)
ms, ss = fit_gaussians(xn)
elbo_n = elbo_closed(xn, ms, ss)
logpx_n = np.log(np.sum(pz[None] * npdf(xn[:, None], mu(zg)[None], S), 1) * dz)
gap = logpx_n - elbo_n
check("缺口都不小于 0", gap.min() >= 0, f"最小 {gap.min():.4f}")
report("缺口的平均（正文 0.55）", 0.55, gap.mean(), 0.005)
report("最好的高斯的标准差，中位数（正文 0.065）", 0.065, np.median(ss), 0.0005)
ekl = np.mean(0.5 * (ms ** 2 + ss ** 2 - 1 - np.log(ss ** 2)))                # E[KL(q ‖ p(z))]，闭式
qagg = npdf(zg[None], ms[:, None], ss[:, None]).mean(0)                        # 聚合后验
kl_agg = np.sum(qagg * (np.log(qagg) - np.log(pz))) * dz
lq = np.log(npdf(zg[None], ms[:, None], ss[:, None]) + 1e-300)
I_q = np.mean(np.sum(np.exp(lq) * (lq - np.log(qagg)[None]), 1) * dz)        # 互信息 = E_n[KL(q_n ‖ q(z))]，直接算
report("E[KL(q(z|x) ‖ p(z))]（正文 2.46）", 2.46, ekl, 0.005)
report("KL(q(z) ‖ p(z))（正文 0.33）", 0.33, kl_agg, 0.005)
report("I(x;z) = E_n[KL(q_n ‖ q(z))]（正文 2.12）", 2.12, I_q, 0.005)
report("分解：E[KL] = I + KL(q(z) ‖ p(z))", ekl, I_q + kl_agg, 1e-6)
check("I 不超过 log N", I_q <= math.log(N), f"I = {I_q:.3f}，log {N} = {math.log(N):.3f}")
check("聚合后验与先验的 KL 不超过平均缺口（正文 0.33 与 0.55）", kl_agg <= gap.mean(),
      f"{kl_agg:.3f} ≤ {gap.mean():.3f}")

hole = qagg < 0.1 * pz
report("先验落在“聚合后验不到先验一成”区间的概率（正文 18%）", 0.18, np.sum(pz[hole]) * dz, 0.005)

print("(d) 全方差公式")
mean_agg = np.sum(zg * qagg) * dz
var_agg = np.sum(zg ** 2 * qagg) * dz - mean_agg ** 2
report("聚合后验的方差（正文 0.68）", 0.68, var_agg, 0.005)
report("Var(m) + E[s²] 与上一行相同", var_agg, ms.var() + np.mean(ss ** 2), 1e-6)

print("(e) 二维标准高斯：含概率 p 的圆的半径 sqrt(−2 ln(1 − p))")
for p_, r_ in ((0.5, 1.18), (0.9, 2.15), (0.99, 3.03)):
    g = np.random.default_rng(1).standard_normal((1_000_000, 2))
    inside = np.mean((g ** 2).sum(1) <= (math.sqrt(-2 * math.log(1 - p_))) ** 2)
    report(f"p = {p_}：半径（正文 {r_}）", r_, math.sqrt(-2 * math.log(1 - p_)), 0.005)
    report(f"p = {p_}：100 万个样本落在圆内的比例", p_, inside, 0.002)

print("(f) 正文引用的 MNIST 数字之间的关系（数字本身由 03_vae_latent_mnist.py 打印）")
report("I = E[KL] − KL(q(z)‖p(z))：6.560 − 0.360", 6.200, 6.560 - 0.360, 1e-9)
report("log 60000（正文 11.0）", 11.0, math.log(60000), 0.05)
report("聚合后验的方差：1.323 + 0.0028（正文 1.326）", 1.326, 1.323 + 0.0028, 0.0005)
report("平均缺口 = 平均负 ELBO − 平均 −log p(x)：142.960 − 138.321", 4.639, 142.960 - 138.321, 0.0005)
check("18 号图像的 σ = (0.0225, 0.0238) 约为先验的四十分之一", 40 <= 1 / 0.0238 and 1 / 0.0225 <= 45,
      f"1/σ = {1 / 0.0225:.1f}、{1 / 0.0238:.1f}")

print("(g) 正文引用的 code/02_vae_mnist.py 行号")
src = (HERE / "02_vae_mnist.py").read_text(encoding="utf-8").splitlines()
cited = {111: "self.mu = nn.Linear", 112: "self.logvar = nn.Linear", 120: "def reparameterize",
         122: "std = torch.exp(0.5 * logvar)", 123: "eps = torch.randn_like(std)", 124: "return mu + std * eps",
         137: "0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar)", 142: "binary_cross_entropy_with_logits",
         174: "loss = rec + kl"}
for line, text in cited.items():
    check(f"第 {line} 行含“{text}”", text in src[line - 1], "")

print("全部通过" if ok_all else "有未通过的项目")
