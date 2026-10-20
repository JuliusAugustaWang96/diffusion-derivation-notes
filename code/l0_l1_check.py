# -*- coding: utf-8 -*-
"""l0_l1_check.py —— 用数值方法核对听课笔记 L0、L1 中出现的全部数字。

  (a) 线性噪声调度与前向过程（L0 第 2 章、L1 第 3.1 节）：ᾱ_t 的取值、两步加噪合并成一步、一步加噪与逐步加噪一致、
      先验匹配项的大小
  (b) 后验 q(x_{t-1}|x_t,x_0)（L1 第 3.3 节）：两个系数与方差，和联合高斯条件化、蒙特卡洛回归三种算法一致
  (c) 把 x_0 换成 ε（L1 第 3.5 节）：均值改写恒等式，以及 DDPM 去掉的权重 λ_t
  (d) 方差相同的两个高斯之间的 KL（L1 第 3.4 节）
  (e) 最大似然与最小化 KL（L1 第 2.2 节）：样本越多，估计越接近 KL 的最小点；E[log P_θ] = −H − KL
  (f) FID（L0 第 3.5 节）：闭式值，以及样本量不足时的偏差
  (g) 反向过程（L0 第 2 章图 2）：训练集只有一张图时，最优噪声预测器有闭式解，采样算法逐步复原这张图
  (h) 采样时加不加噪声（L1 第 4.5 节、图 3）：三团二维高斯上比较 DDPM、去掉 σ_t z 的 DDPM 与 DDIM

运行：python code/l0_l1_check.py      （在仓库根目录执行；只需 NumPy，约 5 秒完成）
记号按 Ho et al. (2020)：T = 1000，β_t 从 0.0001 线性增加到 0.02，α_t = 1 − β_t，ᾱ_t = α_1 α_2 … α_t。
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半。随机数的种子都固定，结果可复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 ᾱ ε 等符号会报错
import math
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ok_all = True

T = 1000
BETA = np.linspace(1e-4, 0.02, T)          # BETA[t-1] 是 β_t
ALPHA = 1.0 - BETA
ABAR = np.cumprod(ALPHA)                     # ABAR[t-1] 是 ᾱ_t


def beta(t):
    return float(BETA[t - 1])


def alpha(t):
    return float(ALPHA[t - 1])


def abar(t):
    return 1.0 if t == 0 else float(ABAR[t - 1])


def report(name, expected, computed, tol):
    global ok_all
    err = abs(float(expected) - float(computed))
    passed = err <= tol
    ok_all &= passed
    print(f"  {name:<40s} 应为 {float(expected): .6g}   计算得 {float(computed): .6g}   误差 {err:.1e}   "
          f"{'通过' if passed else '未通过'}")


def check(name, passed, detail):
    global ok_all
    ok_all &= bool(passed)
    print(f"  {name:<40s} {detail}   {'通过' if passed else '未通过'}")


# ---------------------------------------------------------------- (a)
print("(a) 线性噪声调度与前向过程")
for t, text in ((1, 0.9999), (10, 0.9981), (100, 0.8970), (250, 0.5241), (500, 0.0786), (750, 0.0034)):
    report(f"ᾱ_{t}", text, abar(t), 0.5e-4)
report("ᾱ_1000（正文 4.0×10⁻⁵）", 4.0e-5, abar(1000), 0.05e-5)
report("√ᾱ_1000", 0.0064, math.sqrt(abar(1000)), 0.5e-4)
report("β_1 + … + β_1000", 10.05, BETA.sum(), 0.005)
for t, text in ((50, 0.985), (150, 0.888), (300, 0.630), (600, 0.161), (1000, 0.006)):   # L0 图 3 的标注
    report(f"√ᾱ_{t}（L0 图 3）", text, math.sqrt(abar(t)), 0.5e-3)

b1, b2 = 0.1, 0.2                                  # 讲义上两步合并的例子，取 β_1 = 0.1、β_2 = 0.2
coef, std = math.sqrt((1 - b1) * (1 - b2)), math.sqrt(1 - (1 - b1) * (1 - b2))
report("两步合并：x_0 的系数 √0.72", 0.8485, coef, 0.5e-4)
report("两步合并：噪声标准差 √0.28", 0.5292, std, 0.5e-4)
report("两步合并：两份噪声方差之和", 0.28, (1 - b2) * b1 + b2, 1e-12)
rng = np.random.default_rng(0)
n, x0 = 1_000_000, 1.0
x2 = math.sqrt(1 - b2) * (math.sqrt(1 - b1) * x0 + math.sqrt(b1) * rng.standard_normal(n)) \
    + math.sqrt(b2) * rng.standard_normal(n)
check("两步合并：100 万次模拟", abs(x2.mean() - coef) < 4 * std / math.sqrt(n) and abs(x2.std() - std) < 0.002,
      f"均值 {x2.mean():.4f}，标准差 {x2.std():.4f}")

n, x0 = 200_000, 0.8                               # 逐步加噪 1000 次，与一步公式比较均值和方差
x = np.full(n, x0)
for t in range(1, T + 1):
    x = math.sqrt(alpha(t)) * x + math.sqrt(beta(t)) * rng.standard_normal(n)
    if t in (10, 100, 1000):
        m, v = math.sqrt(abar(t)) * x0, 1 - abar(t)
        se_m, se_v = math.sqrt(v / n), v * math.sqrt(2 / n)
        check(f"逐步加噪到 t = {t}", abs(x.mean() - m) < 4 * se_m and abs(x.var() - v) < 4 * se_v + 1e-12,
              f"均值 {x.mean():.4f}（公式 {m:.4f}），方差 {x.var():.4f}（公式 {v:.4f}）")

kl_prior = 0.5 * ((1 - abar(T)) + abar(T) * 1.0 ** 2 - 1 - math.log(1 - abar(T)))   # KL(N(√ᾱ x0, 1−ᾱ) ‖ N(0,1))
report("先验匹配项（x_0 = 1，每维，正文 2.0×10⁻⁵）", 2.0e-5, kl_prior, 0.05e-5)


# ---------------------------------------------------------------- (b)
print("\n(b) 后验 q(x_{t-1}|x_t,x_0)")


def posterior(t):
    c0 = math.sqrt(abar(t - 1)) * beta(t) / (1 - abar(t))
    ct = math.sqrt(alpha(t)) * (1 - abar(t - 1)) / (1 - abar(t))
    var = (1 - abar(t - 1)) / (1 - abar(t)) * beta(t)
    return c0, ct, var


for t, (t0, tt, tr) in {2: (0.5453, 0.4547, 0.4547), 10: (0.1473, 0.8527, 0.8528),
                        500: (0.0031, 0.9941, 0.9991), 1000: (0.0001, 0.9899, 1.0000)}.items():
    c0, ct, var = posterior(t)
    report(f"t = {t}：x_0 的系数", t0, c0, 0.5e-4)
    report(f"t = {t}：x_t 的系数", tt, ct, 0.5e-4)
    report(f"t = {t}：β̃_t / β_t", tr, var / beta(t), 0.5e-4)
worst = 0.0
for t in range(2, T + 1):                          # 联合高斯条件化：均值 √ᾱ_{t-1}x0 + k (x_t − √ᾱ_t x0)
    k = math.sqrt(alpha(t)) * (1 - abar(t - 1)) / (1 - abar(t))
    c0, ct, var = posterior(t)
    worst = max(worst, abs((math.sqrt(abar(t - 1)) - k * math.sqrt(abar(t))) - c0), abs(k - ct),
                abs(((1 - abar(t - 1)) - k * math.sqrt(alpha(t)) * (1 - abar(t - 1))) - var))
check("t = 2…1000：与联合高斯条件化一致", worst < 1e-12, f"最大差 {worst:.1e}")
t, x0, n = 500, 0.8, 400_000                       # 模拟 (x_{t-1}, x_t)，用 x_t 回归 x_{t-1}
xp = math.sqrt(abar(t - 1)) * x0 + math.sqrt(1 - abar(t - 1)) * rng.standard_normal(n)
xt = math.sqrt(alpha(t)) * xp + math.sqrt(beta(t)) * rng.standard_normal(n)
slope, icpt = np.polyfit(xt, xp, 1)
resid = (xp - (slope * xt + icpt)).var()
c0, ct, var = posterior(t)
check("t = 500：40 万次模拟后回归", abs(slope - ct) < 0.003 and abs(icpt - c0 * x0) < 0.003 and abs(resid / var - 1) < 0.01,
      f"斜率 {slope:.4f}（{ct:.4f}），截距 {icpt:.4f}（{c0 * x0:.4f}），残差方差 {resid:.5f}（{var:.5f}）")


# ---------------------------------------------------------------- (c)
print("\n(c) 把 x_0 换成 ε")
worst = 0.0
for _ in range(1000):
    t = int(rng.integers(2, T + 1))
    x0, eps = rng.standard_normal(3), rng.standard_normal(3)
    xt = math.sqrt(abar(t)) * x0 + math.sqrt(1 - abar(t)) * eps
    c0, ct, _ = posterior(t)
    mu_x0 = c0 * x0 + ct * xt
    mu_eps = (xt - beta(t) / math.sqrt(1 - abar(t)) * eps) / math.sqrt(alpha(t))
    worst = max(worst, float(np.abs(mu_x0 - mu_eps).max()))
check("1000 组随机 (t, x_0, ε)：两种写法相同", worst < 1e-12, f"最大差 {worst:.1e}")


def lam(t):                                        # DDPM 式 (12) 的权重，取 σ_t² = β_t
    return beta(t) ** 2 / (2 * beta(t) * alpha(t) * (1 - abar(t)))


for t, text, tol in ((1, 0.50, 0.005), (10, 0.074, 0.0005), (100, 0.010, 0.0005), (500, 0.0055, 0.00005),
                     (1000, 0.010, 0.0005)):
    report(f"λ_{t}", text, lam(t), tol)


# ---------------------------------------------------------------- (d)
print("\n(d) 方差相同的两个高斯之间的 KL")
m1, m2, v = 0.3, 1.1, 0.25
grid = np.linspace(-6, 7, 200_001)
p = np.exp(-(grid - m1) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
q = np.exp(-(grid - m2) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
report("KL(N(0.3, 0.25) ‖ N(1.1, 0.25)) 闭式", 1.28, (m1 - m2) ** 2 / (2 * v), 1e-12)
report("同上，数值积分", 1.28, trap(p * np.log(p / q), grid), 1e-6)


# ---------------------------------------------------------------- (e)
print("\n(e) 最大似然与最小化 KL：P_data = N(1, 0.5²)，模型 P_θ = N(μ, 1)")
rng_mle = np.random.default_rng(0)
for m, text in ((10, 1.0423), (1000, 0.9750), (100_000, 0.9999)):
    xs = rng_mle.normal(1.0, 0.5, m)
    grid = np.linspace(0.0, 2.0, 20_001)              # 在网格上找平均对数似然最大的 μ
    ll = [-(0.5 * math.log(2 * math.pi)) - 0.5 * np.mean((xs - mu) ** 2) for mu in grid[::100]]
    mu_grid = grid[::100][int(np.argmax(ll))]
    report(f"m = {m}：μ̂（平均对数似然最大）", text, xs.mean(), 0.5e-4)
    check(f"m = {m}：网格搜索与样本均值一致", abs(mu_grid - xs.mean()) <= 0.01, f"网格 {mu_grid:.2f}")
H = 0.5 * math.log(2 * math.pi * math.e * 0.25)


def kl_data(mu):
    return math.log(1 / 0.5) + (0.25 + (1 - mu) ** 2) / 2 - 0.5


def elog(mu):
    return -0.5 * math.log(2 * math.pi) - 0.5 * (0.25 + (1 - mu) ** 2)


report("H(P_data)", 0.7258, H, 0.5e-4)
report("KL(P_data ‖ N(1, 1))（最小值）", 0.3181, kl_data(1.0), 0.5e-4)
report("KL(P_data ‖ N(0.5, 1))", 0.4431, kl_data(0.5), 0.5e-4)
report("E[log P_θ]，μ = 1", -1.0439, elog(1.0), 0.5e-4)
report("E[log P_θ]，μ = 0.5", -1.1689, elog(0.5), 0.5e-4)
check("E[log P_θ] = −H − KL（μ = 0、0.5、1、1.7）",
      max(abs(elog(mu) - (-H - kl_data(mu))) for mu in (0.0, 0.5, 1.0, 1.7)) < 1e-12, "恒等式成立")
mus = np.linspace(0, 2, 2001)
check("KL 的最小点", abs(mus[int(np.argmin([kl_data(mu) for mu in mus]))] - 1.0) < 1e-9, "μ = 1")


# ---------------------------------------------------------------- (f)
print("\n(f) FID：两团二维“特征”")


def sqrtm_psd(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fid(m1, S1, m2, S2):
    """‖m1 − m2‖² + Tr(S1 + S2 − 2 (S1 S2)^{1/2})；Tr((S1 S2)^{1/2}) = Tr((S1^{1/2} S2 S1^{1/2})^{1/2})。"""
    r = sqrtm_psd(S1)
    return float(np.sum((m1 - m2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.trace(sqrtm_psd(r @ S2 @ r)))


def draw(r, mean, cov, n):                          # 用 Cholesky 抽样，结果不依赖线性代数库的符号约定
    return mean + r.standard_normal((n, 2)) @ np.linalg.cholesky(cov).T


MR, SR = np.zeros(2), np.array([[1.0, 0.3], [0.3, 0.5]])
CLOSE = (np.array([0.2, 0.1]), np.array([[0.9, 0.2], [0.2, 0.6]]))
FAR = (np.array([1.5, -1.0]), np.array([[0.3, 0.0], [0.0, 0.3]]))
report("FID 闭式：接近的一组", 0.066, fid(MR, SR, *CLOSE), 0.5e-3)
report("FID 闭式：远离的一组", 3.52, fid(MR, SR, *FAR), 0.005)
m_any = np.array([0.4, -0.3])
S1, S2 = np.array([[1.0, 0.3], [0.3, 0.5]]), np.array([[0.9, 0.2], [0.2, 0.6]])
ev = np.linalg.eigvals(S1 @ S2)
check("Tr((S1 S2)^{1/2}) 的两种算法一致", abs(np.sum(np.sqrt(ev.real)) - np.trace(sqrtm_psd(sqrtm_psd(S1) @ S2 @ sqrtm_psd(S1)))) < 1e-12,
      "特征值法 = 对称化法")
r_fid = np.random.default_rng(1)
for nsamp, text, tol in ((25, 0.27, 0.005), (250, 0.083, 0.0005), (2500, 0.067, 0.0005)):
    vals = []
    for _ in range(50):
        xr, xg = draw(r_fid, MR, SR, nsamp), draw(r_fid, *CLOSE, nsamp)
        vals.append(fid(xr.mean(0), np.cov(xr.T), xg.mean(0), np.cov(xg.T)))
    report(f"接近的一组，每边 {nsamp} 个样本，50 次平均", text, np.mean(vals), tol)
r_fid = np.random.default_rng(2)
for nsamp, text, tol in ((25, 0.20, 0.005), (250, 0.022, 0.0005), (2500, 0.0020, 0.00005)):
    vals = []
    for _ in range(50):
        xr, xg = draw(r_fid, MR, SR, nsamp), draw(r_fid, MR, SR, nsamp)
        vals.append(fid(xr.mean(0), np.cov(xr.T), xg.mean(0), np.cov(xg.T)))
    report(f"同一分布，每边 {nsamp} 个样本，50 次平均", text, np.mean(vals), tol)


# ---------------------------------------------------------------- (g)
print("\n(g) 只有一张训练图时的反向过程")


def scene(size=64):
    """L0 图 2、图 3 用的合成图：天空、太阳、两层山坡，像素值在 [−1, 1]。"""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    img = np.empty((size, size, 3))
    img[..., 0], img[..., 1], img[..., 2] = 0.55 + 0.35 * yy, 0.75 + 0.2 * yy, 0.98 - 0.05 * yy   # 天空
    sun = (xx - 0.72) ** 2 + (yy - 0.28) ** 2 < 0.13 ** 2
    img[sun] = (1.0, 0.82, 0.25)
    far = yy > 0.62 + 0.07 * np.sin(2 * np.pi * (xx * 1.3 + 0.1))
    img[far] = (0.45, 0.68, 0.40)
    near = yy > 0.78 + 0.05 * np.sin(2 * np.pi * (xx * 0.9 + 0.55))
    img[near] = (0.25, 0.52, 0.28)
    return img * 2 - 1


x0 = scene()
r_rev = np.random.default_rng(3)
x = r_rev.standard_normal(x0.shape)
for t in range(T, 0, -1):
    eps_hat = (x - math.sqrt(abar(t)) * x0) / math.sqrt(1 - abar(t))     # 训练集只有 x0 时的最优噪声预测
    x = (x - beta(t) / math.sqrt(1 - abar(t)) * eps_hat) / math.sqrt(alpha(t))
    if t > 1:
        x = x + math.sqrt(beta(t)) * r_rev.standard_normal(x.shape)
check("DDPM 采样算法跑完 1000 步", float(np.abs(x - x0).max()) < 1e-9, f"与训练图的最大差 {float(np.abs(x - x0).max()):.1e}")


# ---------------------------------------------------------------- (h)
print("\n(h) 采样时加不加噪声：三团二维高斯，团中心在半径 1 的圆上，每团每个方向的标准差 0.1")
ANG = np.deg2rad([90, 210, 330])
CEN = np.stack([np.cos(ANG), np.sin(ANG)], 1)
S = 0.1


def eps_opt(x, t):
    """数据是三团等权高斯时，噪声预测器的最优解 E[ε | x_t] = −√(1−ᾱ_t) ∇log q_t(x_t)，有闭式。"""
    ab = abar(t)
    v = ab * S ** 2 + 1 - ab
    d2 = ((x[:, None, :] - math.sqrt(ab) * CEN[None]) ** 2).sum(-1)
    lw = -d2 / (2 * v)
    w = np.exp(lw - lw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return math.sqrt(1 - ab) * (x - math.sqrt(ab) * (w @ CEN)) / v


def sample(kind, n=3000):
    x = np.random.default_rng(10).standard_normal((n, 2))     # 三种采样器从同一批 x_T 出发
    rz = np.random.default_rng(11)
    for t in range(T, 0, -1):
        e = eps_opt(x, t)
        if kind == "ddim":                         # DDIM，η = 0：先估计 x_0，再按 ᾱ_{t-1} 重新组合
            x0_hat = (x - math.sqrt(1 - abar(t)) * e) / math.sqrt(abar(t))
            x = math.sqrt(abar(t - 1)) * x0_hat + math.sqrt(1 - abar(t - 1)) * e
        else:                                      # DDPM 算法 2 第 4 行
            x = (x - beta(t) / math.sqrt(1 - abar(t)) * e) / math.sqrt(alpha(t))
            if kind == "ddpm" and t > 1:
                x = x + math.sqrt(beta(t)) * rz.standard_normal(x.shape)
    return x


def cluster_stats(x):
    k = ((x[:, None, :] - CEN[None]) ** 2).sum(-1).argmin(1)
    frac = np.bincount(k, minlength=3) / len(x)
    radius, spread = [], []
    for j in range(3):
        xs = x[k == j]
        m = xs.mean(0)
        radius.append(float(np.linalg.norm(m)))
        spread.append(float(np.sqrt(((xs - m) ** 2).sum(1).mean() / 2)))   # 团内每个方向的标准差
    return frac, np.array(radius), np.array(spread)


r_data = np.random.default_rng(12)
data = CEN[r_data.integers(0, 3, 3000)] + S * r_data.standard_normal((3000, 2))
for label, x, rad_text, spr_text, tol_spr in (("数据本身", data, 1.00, 0.10, 0.005),
                                               ("DDPM（σ_t² = β_t）", sample("ddpm"), 1.00, 0.10, 0.005),
                                               ("去掉 σ_t z", sample("mean"), 0.90, 0.014, 0.0005),
                                               ("DDIM（η = 0）", sample("ddim"), 1.00, 0.10, 0.005)):
    frac, rad, spr = cluster_stats(x)
    check(f"{label}：三团各占约 1/3", np.all(np.abs(frac - 1 / 3) < 0.03), f"比例 {np.round(frac, 3)}")
    report(f"{label}：团中心到原点的距离（平均）", rad_text, rad.mean(), 0.005)
    report(f"{label}：团内每个方向的标准差（平均）", spr_text, spr.mean(), tol_spr)

print("\n全部通过" if ok_all else "\n有未通过的项目")
