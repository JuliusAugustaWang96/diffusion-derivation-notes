# -*- coding: utf-8 -*-
"""d2_d3_check.py —— 用数值方法核对推导笔记 D2–D3（证据下界与 VAE 的损失）中出现的全部数字。

  (a) T9、T10（第 1.2、1.3 节）：玩具例子的 log p(x)；同一个积分换两个 q 写成期望，结果不变
  (b) T11 Jensen 不等式（第 1.4 节、图 1、自测题 1）：两点、三点的例子
  (c) T12 KL 非负（第 1.5 节）：例子的数值；随机抽一万对高斯，KL 都不小于 0
  (d) T15 换元（第 1.8 节）：E[(μ+ε)²] = μ² + 1
  (e) T17 伯努利与 BCE（第 1.10 节、图 4）：例子的数值；logit 版公式；灰度像素的最小点与最小值
  (f) 证据下界的玩具例子（第 4、5、6 章、图 2）：五个 q 的 ELBO、缺口与 KL；5.1 节的三项；D2(c) 的两项拆分；蒙特卡洛
  (g) 梯度的三种估计（第 7.1 节）：理论标准差与模拟结果
  (h) KL 闭式（第 7.3 节、自测题 3）
  (i) 第 8.1 节的输出：负 ELBO 的组成、浮点舍入单位、分位点
  (j) 第 7.5、8.3 节等处引用的 code/02_vae_mnist.py 行号

运行：python code/d2_d3_check.py      （在仓库根目录执行；只需 NumPy，约 1 秒完成）
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半；模拟结果的容差按标准误差设定。随机数的种子都固定，结果可复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ ε 等符号会报错
import math, pathlib
from statistics import NormalDist
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ok_all = True


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


def log_normal_pdf(x, m, v):
    """log N(x; m, v)，第二个参数是方差。"""
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)


def kl_gauss(m1, v1, m2, v2):
    """D1(a)：KL(N(m1, v1) || N(m2, v2))，一维，第二个参数是方差。"""
    return 0.5 * np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / (2 * v2) - 0.5


def kl_std(mu, s2):
    """D1(c) 特例二 = D3(c)：逐维的 ½(μ² + σ² − 1 − log σ²)。"""
    mu, s2 = np.asarray(mu, float), np.asarray(s2, float)
    return 0.5 * (mu ** 2 + s2 - 1 - np.log(s2))


def bce(x, p):
    return -(x * np.log(p) + (1 - x) * np.log(1 - p))


def softplus(l):
    return np.logaddexp(0.0, l)                    # log(1 + e^l)，不会溢出


# ---------------------------------------------------------------- (a)
print("(a) T9、T10：边缘分布与“乘一个、除一个”")
a, sig2, xobs = 2.0, 1.0, 3.0                      # z ~ N(0,1)，x | z ~ N(az, σ²)，观测 x = 3
log_px = log_normal_pdf(xobs, 0.0, a * a + sig2)
report("log N(3; 0, 5)（正文 −2.6237）", -2.6237, log_px, 0.5e-4)
z = np.linspace(-12, 12, 480001)                   # 边缘化 ∫ p(x|z) p(z) dz 的数值积分
px_num = trap(np.exp(log_normal_pdf(xobs, a * z, sig2) + log_normal_pdf(z, 0.0, 1.0)), z)
report("数值积分 ∫p(x|z)p(z)dz 再取 log", -2.6237, math.log(px_num), 0.5e-4)
u = np.linspace(0.0, 1.0, 200001)
direct = trap(u ** 2, u)
uniform = trap(1.0 * (u ** 2 / 1.0), u)            # q = (0,1) 上的均匀分布，密度 1
uu = u[1:]                                          # q(z) = 2z 在 z = 0 处为 0，那一点不参与
linear = trap(2 * uu * (uu ** 2 / (2 * uu)), uu)
report("∫₀¹ z² dz（正文 1/3）", 1 / 3, direct, 1e-9)
report("q 取均匀分布：E_q[z²/q(z)]", 1 / 3, uniform, 1e-9)
report("q(z) = 2z：E_q[z²/q(z)] = ½E_q[z]", 1 / 3, linear, 1e-8)
report("q(z) = 2z 下的 E_q[z]（正文 2/3）", 2 / 3, trap(2 * u * u, u), 1e-9)

# ---------------------------------------------------------------- (b)
print("(b) T11 Jensen 不等式")
report("X 等概率取 1、4：log E[X] = log 2.5（正文 0.916）", 0.916, math.log(2.5), 0.5e-3)
report("E[log X] = (0 + log 4)/2（正文 0.693）", 0.693, (math.log(1) + math.log(4)) / 2, 0.5e-3)
report("log 4（正文 1.386）", 1.386, math.log(4), 0.5e-3)
report("两者之差（正文 0.223）", 0.223, math.log(2.5) - math.log(4) / 2, 0.5e-3)
pts, probs = np.array([1.0, 2.0, 9.0]), np.array([0.5, 0.25, 0.25])
report("三点例子：E[X]（正文 3.25）", 3.25, probs @ pts, 1e-12)
report("三点例子：log E[X]（正文 1.179）", 1.179, math.log(probs @ pts), 0.5e-3)
report("三点例子：E[log X]（正文 0.723）", 0.723, probs @ np.log(pts), 0.5e-3)
report("自测题 1：log E[X] = log 5（正文 1.609）", 1.609, math.log(5), 0.5e-3)
report("自测题 1：log 2（正文 0.693）", 0.693, math.log(2), 0.5e-3)
report("自测题 1：log 8（正文 2.079）", 2.079, math.log(8), 0.5e-3)
report("自测题 1：E[log X]（正文 1.386）", 1.386, (math.log(2) + math.log(8)) / 2, 0.5e-3)
rng_j = np.random.default_rng(7)
xs = rng_j.uniform(0.1, 10, size=(2000, 5))
ws = rng_j.dirichlet(np.ones(5), size=2000)
check("随机 2000 个正值分布：log E[X] ≥ E[log X]", np.all(np.log((ws * xs).sum(1)) >= (ws * np.log(xs)).sum(1) - 1e-12),
      "一个反例也没有")

# ---------------------------------------------------------------- (c)
print("(c) T12 KL 散度非负")
report("KL(N(1,1) || N(0,4))（正文 0.4431）", 0.4431, kl_gauss(1, 1, 0, 4), 0.5e-4)
rng_k = np.random.default_rng(3)
m1, m2 = rng_k.normal(0, 3, 10000), rng_k.normal(0, 3, 10000)
v1, v2 = np.exp(rng_k.normal(0, 1.5, 10000)), np.exp(rng_k.normal(0, 1.5, 10000))
kls = kl_gauss(m1, v1, m2, v2)
check("随机一万对一维高斯：KL ≥ 0", kls.min() >= -1e-12, f"最小值 {kls.min():.3g}")
check("两个分布相同：KL = 0", abs(kl_gauss(0.7, 2.3, 0.7, 2.3)) < 1e-15, f"{kl_gauss(0.7, 2.3, 0.7, 2.3):.1e}")

# ---------------------------------------------------------------- (d)
print("(d) T15 换元")
mu = 1.0
eps = np.random.default_rng(1).standard_normal(2_000_000)
mc = float(((mu + eps) ** 2).mean())
se = float(((mu + eps) ** 2).std() / math.sqrt(eps.size))
check("E[(μ+ε)²] = μ² + 1 = 2（μ = 1，蒙特卡洛）", abs(mc - 2) < 4 * se, f"模拟 {mc:.4f}，标准误差 {se:.4f}")

# ---------------------------------------------------------------- (e)
print("(e) T17 伯努利分布与二元交叉熵")
report("BCE(x=1, p=0.9)（正文 0.1054）", 0.1054, bce(1, 0.9), 0.5e-4)
report("BCE(x=0, p=0.9)（正文 2.3026）", 2.3026, bce(0, 0.9), 0.5e-4)
report("BCE(x=0.5, p=0.5)（正文 0.6931）", 0.6931, bce(0.5, 0.5), 0.5e-4)
report("BCE(x=0.5, p=0.9)（正文 1.2040）", 1.2040, bce(0.5, 0.9), 0.5e-4)
l9 = math.log(0.9 / 0.1)
report("0.9 的 logit（正文 2.1972）", 2.1972, l9, 0.5e-4)
report("sigmoid(2.1972)（正文 0.9）", 0.9, 1 / (1 + math.exp(-2.1972)), 0.5e-4)
report("softplus(l) − l，l = 2.1972（正文 0.1054）", 0.1054, softplus(2.1972) - 2.1972, 0.5e-4)
rng_b = np.random.default_rng(5)
xx, ll = rng_b.uniform(0, 1, 100000), rng_b.uniform(-20, 20, 100000)
pp = 1 / (1 + np.exp(-ll))
diff = np.abs(bce(xx, pp) - (softplus(ll) - xx * ll)).max()
check("BCE(x, sigmoid(l)) = softplus(l) − x·l（十万组）", diff < 1e-6, f"最大差 {diff:.1e}")
check("log sigmoid(l) = −softplus(−l)", np.abs(np.log(pp) + softplus(-ll)).max() < 1e-6, "十万组")
check("log(1 − sigmoid(l)) = −softplus(l)", np.abs(np.log1p(-pp) + softplus(ll)).max() < 1e-6, "十万组")
grid = np.linspace(1e-4, 1 - 1e-4, 999999)
for xv in (0.3, 0.5, 0.8):
    pbest = grid[np.argmin(bce(xv, grid))]
    entropy = -xv * math.log(xv) - (1 - xv) * math.log(1 - xv)
    check(f"x = {xv}：BCE 的最小点在 p = x", abs(pbest - xv) < 2e-6, f"最小点 {pbest:.6f}")
    check(f"x = {xv}：最小值 = −x log x − (1−x) log(1−x)", abs(bce(xv, xv) - entropy) < 1e-12, f"{entropy:.4f}")
report("灰度像素 x = 0.5 的最小值 log 2（正文 0.693）", 0.693, bce(0.5, 0.5), 0.5e-3)

# ---------------------------------------------------------------- (f)
print("(f) 证据下界的玩具例子：a = 2，σ² = 1，x = 3")
m_post, v_post = a * xobs / (sig2 + a * a), sig2 / (sig2 + a * a)
report("后验均值（正文 1.2）", 1.2, m_post, 1e-12)
report("后验方差（正文 0.2）", 0.2, v_post, 1e-12)


def elbo_terms(m, s2):
    """q = N(m, s2) 时 ELBO 的三项：E_q[log p(x|z)]、E_q[log p(z)]、−E_q[log q(z)]（第 5.1 节）。"""
    t_lik = -0.5 * math.log(2 * math.pi * sig2) - ((xobs - a * m) ** 2 + a * a * s2) / (2 * sig2)
    t_prior = -0.5 * math.log(2 * math.pi) - (m * m + s2) / 2
    t_ent = 0.5 * math.log(2 * math.pi * s2) + 0.5
    return t_lik, t_prior, t_ent


table = [((1.2, 0.2), "−2.6237", "0"), ((0.0, 1.0), "−7.4189", "4.7953"), ((1.2, 1.0), "−3.8189", "1.1953"),
         ((2.0, 0.2), "−4.2237", "1.6000"), ((1.2, 0.05), "−2.9418", "0.3181")]
for (m, s2), e_txt, gap_txt in table:
    e = sum(elbo_terms(m, s2))
    kl = kl_gauss(m, s2, m_post, v_post)
    num = lambda t: float(t.replace("−", "-"))           # 正文里的负号是 U+2212
    report(f"q = N({m:g},{s2:g})：ELBO（正文 {e_txt}）", num(e_txt), e, 0.5e-4)
    report(f"q = N({m:g},{s2:g})：缺口 log p(x) − ELBO（正文 {gap_txt}）", num(gap_txt), log_px - e, 0.5e-4)
    check(f"q = N({m:g},{s2:g})：缺口 = KL(q || 后验)", abs((log_px - e) - kl) < 1e-12, f"KL {kl:.6f}")
t1, t2, t3 = elbo_terms(1.2, 0.2)
report("5.1 节：−½ log(2π)（正文 −0.9189）", -0.9189, -0.5 * math.log(2 * math.pi), 0.5e-4)
report("5.1 节：第一项（正文 −1.4989）", -1.4989, t1, 0.5e-4)
report("5.1 节：第二项（正文 −1.7389）", -1.7389, t2, 0.5e-4)
report("5.1 节：½ log(2π·0.2)（正文 0.1142）", 0.1142, 0.5 * math.log(2 * math.pi * 0.2), 0.5e-4)
report("5.1 节：第三项，高斯的熵（正文 0.6142）", 0.6142, t3, 0.5e-4)
report("5.1 节：三个四舍五入值之和（正文 −2.6236）", -2.6236, -1.4989 - 1.7389 + 0.6142, 1e-9)
report("第三项 = ½ log(2πe s²)", 0.5 * math.log(2 * math.pi * math.e * 0.2), t3, 1e-12)
kl_prior = kl_gauss(1.2, 0.2, 0.0, 1.0)
report("D2(c)：KL(N(1.2,0.2) || N(0,1))（正文 1.1247）", 1.1247, kl_prior, 0.5e-4)
report("D2(c)：0.64 + 1.6094 的一半", 1.1247, 0.5 * (0.64 + 1.6094), 0.5e-4)
report("D2(c)：−log 0.2（正文 1.6094）", 1.6094, -math.log(0.2), 0.5e-4)
report("D2(c)：第二项 + 第三项 = −KL（正文 −1.1247）", -1.1247, t2 + t3, 0.5e-4)
report("D2(c)：重建项 − KL = ELBO", sum(elbo_terms(1.2, 0.2)), t1 - kl_prior, 1e-12)
report("D2(c)：两个四舍五入值之差（正文 −2.6236）", -2.6236, -1.4989 - 1.1247, 1e-9)
rng = np.random.default_rng(0)                     # 下面两段共用这个生成器，顺序固定
zs = rng.normal(0.0, 1.0, 1_000_000)               # q = N(0,1)：ELBO 的蒙特卡洛估计
vals = log_normal_pdf(xobs, a * zs, sig2) + log_normal_pdf(zs, 0.0, 1.0) - log_normal_pdf(zs, 0.0, 1.0)
mc, se = float(vals.mean()), float(vals.std() / math.sqrt(vals.size))
check("q = N(0,1)：蒙特卡洛 ELBO 与闭式 −7.4189 一致", abs(mc + 7.4189) < 3 * se, f"模拟 {mc:.4f}，标准误差 {se:.4f}")
check("q = N(0,1)：蒙特卡洛 ELBO（正文 −7.4156）", abs(mc + 7.4156) < 3 * se, f"模拟 {mc:.4f}")

# ---------------------------------------------------------------- (g)
print("(g) 梯度的三种估计：f(z) = z²，q = N(μ, 1)，μ = 1，真梯度 2")
mom = {k: (math.prod(range(k - 1, 0, -2)) if k % 2 == 0 else 0) for k in range(0, 9)}   # ε 的 k 阶矩：(k−1)!!
report("E[ε²]（正文 1）", 1, mom[2], 0)
report("E[ε⁴]（正文 3）", 3, mom[4], 0)
report("E[ε⁶]（正文 15）", 15, mom[6], 0)
coef = np.polynomial.polynomial.polymul(np.polynomial.polynomial.polypow([1, 1], 4), [0, 0, 1])   # (1+ε)⁴ε²
second = sum(c * mom[k] for k, c in enumerate(coef))
report("E[(1+ε)⁴ε²]（正文 34）", 34, second, 1e-12)
mean_sf = sum(c * mom[k] for k, c in enumerate(np.polynomial.polynomial.polymul([1, 2, 1], [0, 1])))  # (1+ε)²ε
report("score-function 估计的均值", 2, mean_sf, 1e-12)
report("score-function：单样本方差（正文 30）", 30, second - mean_sf ** 2, 1e-12)
report("score-function：单样本标准差 √30（正文 5.48）", 5.48, math.sqrt(30), 0.5e-2)
report("重参数化：单样本标准差（正文 2）", 2, math.sqrt(4 * mom[2]), 0)
report("score-function：1000 个平均后的标准差（正文 0.173）", 0.173, math.sqrt(30 / 1000), 0.5e-3)
report("重参数化：1000 个平均后的标准差（正文 0.063）", 0.063, math.sqrt(4 / 1000), 0.5e-3)
report("两者之比 √30/2（正文 2.7）", 2.7, math.sqrt(30) / 2, 0.05)
reps, n = 2000, 1000
e = rng.standard_normal((reps, n))
zz = mu + e
g_rep = 2 * zz                                     # d/dμ (μ+ε)² = 2(μ+ε)
g_sf = zz ** 2 * (zz - mu)                         # f(z) · d/dμ log q(z) = z²(z − μ)
check("重参数化：单样本标准差的模拟值接近 2", abs(g_rep.std() - 2) < 0.01, f"模拟 {g_rep.std():.4f}")
check("score-function：单样本标准差的模拟值接近 √30", abs(g_sf.std() - math.sqrt(30)) < 0.05, f"模拟 {g_sf.std():.4f}")
sd_rep, sd_sf = g_rep.mean(1).std(), g_sf.mean(1).std()
check("重参数化：2000 次平均的标准差（正文 0.064）", abs(sd_rep - 0.064) < 0.0015 and abs(sd_rep - 0.0632) < 0.003,
      f"模拟 {sd_rep:.4f}")
check("score-function：2000 次平均的标准差（正文 0.174）", abs(sd_sf - 0.174) < 0.004 and abs(sd_sf - 0.1732) < 0.008,
      f"模拟 {sd_sf:.4f}")
check("两种估计的平均值都是 2.00", abs(g_rep.mean() - 2) < 0.005 and abs(g_sf.mean() - 2) < 0.005,
      f"重参数化 {g_rep.mean():.4f}，score-function {g_sf.mean():.4f}")
check("天真做法：抽样后只对 f 求导，f 不含 μ，梯度为 0", True, "0（错）")

# ---------------------------------------------------------------- (h)
print("(h) KL 闭式")
report("μ = 0.5，σ = 0.8（正文 0.1681）", 0.1681, kl_std(0.5, 0.64), 0.5e-4)
report("μ = 0，σ = 1（正文 0）", 0, kl_std(0.0, 1.0), 1e-15)
report("μ = 0，σ² = 0.01（正文 1.8076）", 1.8076, kl_std(0.0, 0.01), 0.5e-4)
report("μ = 0，σ² = 10⁻⁴（正文 4.1052）", 4.1052, kl_std(0.0, 1e-4), 0.5e-4)
uu = np.linspace(0.01, 10, 999001)
fu = uu - 1 - np.log(uu)
check("u − 1 − log u ≥ 0，只在 u = 1 处为 0", fu.min() >= 0 and abs(uu[np.argmin(fu)] - 1) < 2e-5,
      f"最小点 u = {uu[np.argmin(fu)]:.5f}")
report("闭式 = D1(a) 的一般公式代入 N(0,1)", kl_gauss(0.3, 0.5, 0, 1), kl_std(0.3, 0.5), 1e-15)
terms = kl_std([0.3, -0.4], [0.5, 2.0])
report("自测题 3：0.09 + 0.5 − 1（正文 −0.41）", -0.41, 0.09 + 0.5 - 1, 1e-12)
report("自测题 3：−log 0.5 = log 2（正文 0.6931）", 0.6931, math.log(2), 0.5e-4)
report("自测题 3：第 1 维（正文 0.1416）", 0.1416, terms[0], 0.5e-4)
report("自测题 3：第 2 维（正文 0.2334）", 0.2334, terms[1], 0.5e-4)
report("自测题 3：合计（正文 0.3750）", 0.3750, terms.sum(), 0.5e-4)

# ---------------------------------------------------------------- (i)
print("(i) 第 8.1 节的输出")
report("GPU：136.607 + 6.440（正文 143.048）", 143.048, 136.607 + 6.440, 1.5e-3)
report("GPU：重建项占负 ELBO 的比例（正文 95%）", 0.95, 136.607 / 143.048, 0.005)
report("CPU：136.916 + 6.457（正文 143.373）", 143.373, 136.916 + 6.457, 1.5e-3)
report("2⁻²¹（输出 4.77e-07）", 4.77e-7, 2.0 ** -21, 0.005e-7)
report("2⁻²⁰（输出 9.54e-07）", 9.54e-7, 2.0 ** -20, 0.005e-7)
f32 = np.float32
report("float32 在 [4, 8) 的间距 = 2⁻²¹", 2.0 ** -21, float(np.spacing(f32(6.0))), 0)
report("float32 在 [8, 16) 的间距 = 2⁻²⁰", 2.0 ** -20, float(np.spacing(f32(10.0))), 0)
report("N(0,1) 的 5% 分位点（正文 −1.64）", -1.64, NormalDist().inv_cdf(0.05), 0.005)
report("N(0,1) 的 95% 分位点（正文 1.64）", 1.64, NormalDist().inv_cdf(0.95), 0.005)

# ---------------------------------------------------------------- (j)
print("(j) 正文引用的 code/02_vae_mnist.py 行号")
src = (pathlib.Path(__file__).resolve().parent / "02_vae_mnist.py").read_text(encoding="utf-8").splitlines()
cited = {115: "def encode(self, x):", 117: "return self.mu(h), self.logvar(h)",
         120: "def reparameterize(mu, logvar):", 122: "std = torch.exp(0.5 * logvar)",
         123: "eps = torch.randn_like(std)", 124: "return mu + std * eps",
         126: "def decode(self, z):", 127: "return self.dec(z)",
         129: "def forward(self, x):", 130: "mu, logvar = self.encode(x)", 131: "z = self.reparameterize(mu, logvar)",
         132: "return self.decode(z), mu, logvar",
         135: "def kl_to_std_normal(mu, logvar):", 137: "0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=1)",
         142: 'F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)',
         149: '"--epochs", type=int, default=10', 150: '"--batch", type=int, default=128',
         151: '"--zdim", type=int, default=2', 152: '"--seed", type=int, default=0',
         164: "torch.optim.Adam(model.parameters(), lr=1e-3)",
         174: "loss = rec + kl", 175: "opt.zero_grad(); loss.backward(); opt.step()",
         178: "with torch.no_grad():", 182: "td.kl_divergence(q, p) - kl_to_std_normal(mu, logvar)",
         192: "rec = torch.sigmoid(logits)"}
for line_no, text in cited.items():
    check(f"第 {line_no} 行", text in src[line_no - 1], text[:40])

print("全部通过" if ok_all else "有未通过的项目")
