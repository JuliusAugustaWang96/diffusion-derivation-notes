# -*- coding: utf-8 -*-
"""l2_check.py —— 用数值方法核对听课笔记 L2（变分自编码器）中出现的全部数字。

  (a) 讲义的 KL 项（L2 第 2.1、2.4、3.6 节）：讲义写法与两种约定下的 KL、原论文附录 B 的公式、蒙特卡洛估计
  (b) exp(σ) ≥ 1 + σ（L2 第 2.4 节）：方差项只在 σ = 0 处取最小值 0
  (c) 码空间中的重叠（L2 第 2.3 节）：两段带噪声的码重叠时，使平方误差最小的输出是两张图的加权平均
  (d) 高斯混合模型（L2 第 3.2 节）：密度积分为 1，先抽类别再抽 x 的两步抽样与混合密度一致
  (e) 无穷多个高斯的混合（L2 第 3.3 节、图 5）：P(x) = ∫P(z)P(x|z)dz 的数值积分与抽样结果一致
  (f) 证据下界的玩具例子（L2 第 3.4 至 3.6 节、图 6）：log P(x) = Lb + KL 对每个 q 都成立，Lb 的两种拆法一致
  (g) 解码器方差固定时（L2 第 3.6 节）：高斯对数似然与平方误差只差常数
  (h) 重参数化（L2 第 3.6 节）：c = m + exp(σ)·e 时，梯度可以穿过 m 和 σ
  (i) 逐像素比较的问题（L2 第 4.2 节、图 7）：差一个像素的两张图误差相同；平方误差偏爱模糊的输出

运行：python code/l2_check.py      （在仓库根目录执行；只需 NumPy，约 5 秒完成）
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半。随机数的种子都固定，结果可复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ ε 等符号会报错
import math
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
ok_all = True


def report(name, expected, computed, tol):
    global ok_all
    err = abs(float(expected) - float(computed))
    passed = err <= tol
    ok_all &= passed
    print(f"  {name:<44s} 应为 {float(expected): .6g}   计算得 {float(computed): .6g}   误差 {err:.1e}   "
          f"{'通过' if passed else '未通过'}")


def check(name, passed, detail):
    global ok_all
    ok_all &= bool(passed)
    print(f"  {name:<44s} {detail}   {'通过' if passed else '未通过'}")


def kl_gauss(m1, v1, m2, v2):
    """KL(N(m1, v1) || N(m2, v2))，一维，第二个参数是方差。"""
    return 0.5 * (v1 / v2 + (m1 - m2) ** 2 / v2 - 1 + math.log(v2 / v1))


def slide_term(s, m):
    """讲义的写法：exp(σ) − (1 + σ) + m²。"""
    return math.exp(s) - (1 + s) + m * m


# ---------------------------------------------------------------- (a)
print("(a) 讲义的 KL 项与两种约定")
for s, m in ((1.0, 0.0), (0.3, 0.5), (-5.0, 0.0), (-1.3, 0.7)):
    # 约定一：σ 是对数方差，方差 = exp(σ)；约定二：σ 是对数标准差，标准差 = exp(σ)（与 c = exp(σ)·e + m 一致）
    check(f"σ={s}, m={m}：约定一下讲义的式子 = 2 × KL", abs(slide_term(s, m) - 2 * kl_gauss(m, math.exp(s), 0, 1)) < 1e-12,
          f"讲义 {slide_term(s, m):.6f}，2×KL {2 * kl_gauss(m, math.exp(s), 0, 1):.6f}")
    appendix_b = -0.5 * (1 + s - m * m - math.exp(s))         # 原论文附录 B：−D_KL = ½(1 + log σ² − μ² − σ²)，σ 为标准差
    check(f"σ={s}, m={m}：附录 B 的公式（log σ² = σ）", abs(appendix_b - kl_gauss(m, math.exp(s), 0, 1)) < 1e-12,
          f"附录 B {appendix_b:.6f}")
report("σ=1, m=0：讲义的式子 e − 2（正文 0.7183）", 0.7183, slide_term(1, 0), 0.5e-4)
report("σ=1, m=0：σ 为对数方差时的 KL（正文 0.3591）", 0.3591, kl_gauss(0, math.exp(1), 0, 1), 0.5e-4)
report("σ=1, m=0：σ 为对数标准差时的 KL（正文 2.1945）", 2.1945, kl_gauss(0, math.exp(2), 0, 1), 0.5e-4)
report("½(e² − 3)", 2.1945, 0.5 * (math.e ** 2 - 3), 0.5e-4)
report("σ=−5：讲义的式子（正文 4.0067）", 4.0067, slide_term(-5, 0), 0.5e-4)
report("σ=−5：标准差 exp(−5)（正文 0.0067）", 0.0067, math.exp(-5), 0.5e-4)

rng = np.random.default_rng(0)
n, m, s = 2_000_000, 0.5, 0.3
for name, std in (("约定二 c = m + exp(σ)·e", math.exp(s)), ("约定一 c = m + exp(σ/2)·e", math.exp(s / 2))):
    c = m + std * rng.standard_normal(n)                                      # 编码器给出的带噪声的码
    log_q = -0.5 * math.log(2 * math.pi * std ** 2) - (c - m) ** 2 / (2 * std ** 2)
    log_p = -0.5 * math.log(2 * math.pi) - c ** 2 / 2
    mc = float((log_q - log_p).mean())                                         # KL = E_q[log q − log p]
    exact = kl_gauss(m, std ** 2, 0, 1)
    check(f"蒙特卡洛 KL，{name}", abs(mc - exact) < 4e-3, f"模拟 {mc:.4f}，闭式 {exact:.4f}")

# ---------------------------------------------------------------- (b)
print("(b) exp(σ) − (1 + σ) 的最小值")
grid = np.linspace(-6, 6, 120001)
f = np.exp(grid) - (1 + grid)
report("最小值所在的 σ", 0.0, grid[f.argmin()], 1e-4)
report("最小值", 0.0, f.min(), 1e-8)
check("处处非负（切线不等式 e^σ ≥ 1 + σ）", bool((f >= -1e-15).all()), f"网格上最小 {f.min():.2e}")
check("KL 项在两种约定下都只在 σ = 0、m = 0 处为 0", all(slide_term(t, 0) > 0 and kl_gauss(0, math.exp(2 * t), 0, 1) > 0
                                                for t in (-2, -0.5, 0.5, 2)) and slide_term(0, 0) == 0,
      "σ = ±0.5、±2 时都大于 0")

# ---------------------------------------------------------------- (c)
print("(c) 两段带噪声的码重叠时的最优输出")
S = 0.6                      # 码中的噪声标准差；满月的码在 −1，弦月的码在 +1，两张图各占一半


def w_full(c, s=S):
    """码为 c 时，平方误差最小的输出中满月所占的权重：P(满月 | c)。"""
    return 1 / (1 + math.exp(2 * c / s ** 2))


report("c = 0：满月的权重", 0.5, w_full(0), 1e-12)
report("c = 0.3：满月的权重（正文 0.16）", 0.16, w_full(0.3), 0.005)
report("c = −0.5：满月的权重（正文 0.94）", 0.94, w_full(-0.5), 0.005)
report("c = 1：满月的权重（正文 0.004）", 0.004, w_full(1), 0.0005)
rng = np.random.default_rng(1)
n = 400_000
label = rng.integers(0, 2, n)                         # 0 = 满月，1 = 弦月
code = np.where(label == 0, -1.0, 1.0) + S * rng.standard_normal(n)
for c0 in (-0.5, 0.0, 0.3):
    near = np.abs(code - c0) < 0.02
    frac = float((label[near] == 0).mean())           # 平方误差下，最优输出 = 该处目标的平均，即满月所占的比例
    check(f"c ≈ {c0}：模拟中满月的比例", abs(frac - w_full(c0)) < 0.02, f"模拟 {frac:.3f}，公式 {w_full(c0):.3f}")

# ---------------------------------------------------------------- (d)
print("(d) 高斯混合模型")
PM = np.array([0.30, 0.15, 0.15, 0.25, 0.15])        # P(m)，m = 1..5
MU = np.array([-3.0, -1.6, 0.2, 2.0, 3.6])
SD = np.array([0.45, 0.30, 0.50, 0.70, 0.25])
xs = np.linspace(-8, 8, 160001)
comp = PM[:, None] * np.exp(-(xs[None] - MU[:, None]) ** 2 / (2 * SD[:, None] ** 2)) / np.sqrt(2 * np.pi * SD[:, None] ** 2)
px = comp.sum(0)
report("∫P(x)dx", 1.0, trap(px, xs), 1e-9)
rng = np.random.default_rng(2)
n = 500_000
mm = rng.choice(5, size=n, p=PM)                      # 第一步：m ~ P(m)
x = MU[mm] + SD[mm] * rng.standard_normal(n)          # 第二步：x | m ~ N(μ^m, (σ^m)²)
hist, edges = np.histogram(x, bins=200, range=(-6, 6), density=True)
mid = (edges[1:] + edges[:-1]) / 2
dens = np.interp(mid, xs, px)
check("两步抽样的直方图与混合密度一致", float(np.abs(hist - dens).max()) < 0.03,
      f"最大偏差 {np.abs(hist - dens).max():.4f}")
report("混合分布的均值 Σ P(m) μ^m", float((PM * MU).sum()), float(x.mean()), 0.01)
report("混合密度的峰数（正文 5）", 5, int(((px[1:-1] > px[:-2]) & (px[1:-1] > px[2:])).sum()), 0)

# ---------------------------------------------------------------- (e)
print("(e) 无穷多个高斯的混合：z ~ N(0, 1)，x | z ~ N(μ(z), 0.2²)")


def mu_net(z):
    """代替神经网络的一个固定函数：把 z 映射成高斯分量的均值（与 make_figures_l2.py 相同）。"""
    return z + 1.8 * np.sin(2.8 * z + 1.0)


SIG_X = 0.2
zs = np.linspace(-6, 6, 6001)
pz = np.exp(-zs ** 2 / 2) / math.sqrt(2 * math.pi)
xg = np.linspace(-14, 14, 5601)
lik = np.exp(-(xg[:, None] - mu_net(zs)[None]) ** 2 / (2 * SIG_X ** 2)) / math.sqrt(2 * math.pi * SIG_X ** 2)
pxz = trap(lik * pz[None], zs, axis=1)                 # P(x) = ∫ P(z) P(x|z) dz，对每个 x 做数值积分
report("∫P(x)dx", 1.0, trap(pxz, xg), 1e-6)
rng = np.random.default_rng(3)
z = rng.standard_normal(500_000)
xsamp = mu_net(z) + SIG_X * rng.standard_normal(z.size)
hist, edges = np.histogram(xsamp, bins=140, range=(-7, 7), density=True)
mid = (edges[1:] + edges[:-1]) / 2
check("抽样直方图与数值积分一致", float(np.abs(hist - np.interp(mid, xg, pxz)).max()) < 0.03,
      f"最大偏差 {np.abs(hist - np.interp(mid, xg, pxz)).max():.4f}")
top = [i for i in np.where((pxz[1:-1] > pxz[:-2]) & (pxz[1:-1] > pxz[2:]))[0] + 1 if pxz[i] > 0.05 * pxz.max()]
report("P(x) 的明显峰数（高于最高峰的 5%，正文 3）", 3, len(top), 0)
for i, text in zip(top, (-2.6, -0.35, 1.9)):
    report(f"峰的位置（正文 {text}）", text, xg[i], 0.05)
report("最高峰（x ≈ 1.9）与最低峰（x ≈ −2.6）的高度之比（正文 1.6）", 1.6, pxz[top[2]] / pxz[top[0]], 0.05)
dips = [pxz[a:b].min() / min(pxz[a], pxz[b]) for a, b in zip(top, top[1:])]
check("相邻两峰之间的谷都低于较矮峰的 60%", max(dips) < 0.6, f"谷与较矮峰之比 {[round(float(d), 2) for d in dips]}")

# ---------------------------------------------------------------- (f)
print("(f) 证据下界的玩具例子：P(z) = N(0, 1)，P(x|z) = N(2z, 1)，观测 x = 3")
X, A = 3.0, 2.0
log_px = -0.5 * math.log(2 * math.pi * (1 + A * A)) - X * X / (2 * (1 + A * A))   # 边缘分布 x ~ N(0, 5)
report("log P(x)（正文 −2.6237）", -2.6237, log_px, 0.5e-4)
post_m, post_v = A * X / (1 + A * A), 1 / (1 + A * A)
report("后验 P(z|x) 的均值（正文 1.2）", 1.2, post_m, 1e-12)
report("后验 P(z|x) 的方差（正文 0.2）", 0.2, post_v, 1e-12)


def lb_parts(mq, vq):
    rec = -0.5 * math.log(2 * math.pi) - 0.5 * ((X - A * mq) ** 2 + A * A * vq)    # E_q[log P(x|z)]
    prior = kl_gauss(mq, vq, 0, 1)                                                   # KL(q || P(z))
    return rec, prior, rec - prior


rows = {(0.0, 1.0): (-7.4189, 0.0, -7.4189, 4.7953), (1.0, 0.5): (-2.4189, 0.5966, -3.0155, 0.3919),
        (1.2, 0.2): (-1.4989, 1.1247, -2.6237, 0.0)}
rng = np.random.default_rng(4)
for (mq, vq), (t_rec, t_prior, t_lb, t_gap) in rows.items():
    rec, prior, lb = lb_parts(mq, vq)
    gap = kl_gauss(mq, vq, post_m, post_v)                                            # KL(q || P(z|x))
    report(f"q = N({mq}, {vq})：E_q[log P(x|z)]", t_rec, rec, 0.5e-4)
    report(f"q = N({mq}, {vq})：KL(q || P(z))", t_prior, prior, 0.5e-4)
    report(f"q = N({mq}, {vq})：Lb", t_lb, lb, 0.5e-4)
    report(f"q = N({mq}, {vq})：KL(q || P(z|x))", t_gap, gap, 0.5e-4)
    check(f"q = N({mq}, {vq})：Lb + KL = log P(x)", abs(lb + gap - log_px) < 1e-12, f"{lb + gap:.6f}")
    zq = mq + math.sqrt(vq) * rng.standard_normal(1_000_000)
    log_joint = -math.log(2 * math.pi) - zq ** 2 / 2 - (X - A * zq) ** 2 / 2          # log P(x|z) + log P(z)
    log_q = -0.5 * math.log(2 * math.pi * vq) - (zq - mq) ** 2 / (2 * vq)
    mc = float((log_joint - log_q).mean())                                            # Lb = E_q[log(P(x|z)P(z)/q)]
    check(f"q = N({mq}, {vq})：Lb 的蒙特卡洛估计", abs(mc - lb) < 0.02, f"模拟 {mc:.4f}，闭式 {lb:.4f}")

# ---------------------------------------------------------------- (g)
print("(g) 解码器方差固定时，最大化 log P(x|z) 就是最小化平方误差")
rng = np.random.default_rng(5)
D, SIG = 784, 0.3
x = rng.random(D)
cands = [x + d * rng.standard_normal(D) for d in (0.05, 0.1, 0.2, 0.4)]               # 四个候选的解码器均值
loglik = [float(-((x - mu) ** 2).sum() / (2 * SIG ** 2) - D / 2 * math.log(2 * math.pi * SIG ** 2)) for mu in cands]
se = [float(((x - mu) ** 2).sum()) for mu in cands]
check("对数似然 = −平方误差/(2σ²) − 常数", all(abs(l + e / (2 * SIG ** 2) + D / 2 * math.log(2 * math.pi * SIG ** 2)) < 1e-8
                                            for l, e in zip(loglik, se)), "四个候选")
check("对数似然的排序与平方误差的排序相反", np.argsort(loglik).tolist() == np.argsort(se)[::-1].tolist(),
      f"平方误差 {[round(v, 1) for v in se]}")

# ---------------------------------------------------------------- (h)
print("(h) 重参数化：f(c) = c²，c = m + exp(σ)·e，e ~ N(0, 1)")
m, s = 0.8, -0.4
e = np.random.default_rng(6).standard_normal(2_000_000)
c = m + math.exp(s) * e
grad_m = float((2 * c).mean())                                  # ∂c²/∂m = 2c
grad_s = float((2 * c * math.exp(s) * e).mean())                # ∂c²/∂σ = 2c · exp(σ) e
report("E[c²] = m² + exp(2σ)", m * m + math.exp(2 * s), float((c ** 2).mean()), 3e-3)
report("对 m 的梯度：2m", 2 * m, grad_m, 3e-3)
report("对 σ 的梯度：2 exp(2σ)", 2 * math.exp(2 * s), grad_s, 3e-3)

# ---------------------------------------------------------------- (i)
print("(i) 逐像素比较：28×28 的合成数字 7")
N = 28


def seven(shift=0):
    """一笔画出的 7：先横后斜；shift 把整个数字向右平移若干像素。像素值 0（白）或 1（黑）。"""
    img = np.zeros((N, N))
    rr, cc = np.mgrid[0:N, 0:N]
    pts = [(6, 7), (6, 20), (22, 11)]
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        for t in np.linspace(0, 1, 200):
            r, cl = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t + shift
            img = np.maximum(img, (((rr - r) ** 2 + (cc - cl) ** 2) <= 1.3 ** 2).astype(float))
    return img


target = seven()
ink = np.argwhere(target > 0)
end = ink[ink[:, 0].argmax()]                                   # 斜笔画最下端的一个像素
realistic = target.copy()
realistic[end[0] + 1, end[1]] = 1                               # 把笔画再延长一个像素
fake = target.copy()
fake[12, 24] = 1                                                # 远离笔画的孤立像素
assert target[end[0] + 1, end[1]] == 0 and target[12, 24] == 0
report("墨迹像素数（正文 89）", 89, int(target.sum()), 0)
report("延长笔画：平方误差", 1, float(((realistic - target) ** 2).sum()), 0)
report("孤立像素：平方误差", 1, float(((fake - target) ** 2).sum()), 0)
shifted = seven(1)
blurred = (seven(-1) + target + seven(1)) / 3                   # 三个位置的平均：一张模糊的 7
report("向右平移 1 像素的清晰 7：平方误差（正文 38）", 38, float(((shifted - target) ** 2).sum()), 0)
report("模糊的 7：平方误差（正文 8.67）", 8.67, float(((blurred - target) ** 2).sum()), 0.005)
# 位置不确定时：目标等可能平移 −1、0、+1 像素，期望平方误差最小的预测是三者的平均（模糊的 7）
targets = [seven(-1), target, seven(1)]
exp_se = {name: float(np.mean([((pred - t) ** 2).sum() for t in targets]))
          for name, pred in (("清晰的 7", target), ("模糊的 7", blurred))}
report("位置不确定时，清晰 7 的期望平方误差（正文 25.33）", 25.33, exp_se["清晰的 7"], 0.005)
report("位置不确定时，模糊 7 的期望平方误差（正文 16.67）", 16.67, exp_se["模糊的 7"], 0.005)

print("全部通过" if ok_all else "有未通过的项目")
