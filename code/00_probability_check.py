# -*- coding: utf-8 -*-
"""00_probability_check.py —— 用数值方法核对前置篇 P0（花书第 3 章读书笔记）中出现的全部数字。

  (a) 高斯分布（§3.9.3）：归一化常数、精度写法、峰值、μ±σ 内的概率、拐点、多维高斯的两种写法与各向同性时的分解
  (b) 贝叶斯规则（§3.11）：检测阳性后的患病概率，两条计算路线与 100 万人模拟
  (c) 自信息与熵（§3.13）：抛硬币的信息量、奈特与比特、二值熵、均匀分布的熵最大、微分熵可以为负
  (d) KL 散度与交叉熵（§3.13）：两个方向的数值、编码长度的解释、非负性、H(P,Q) = H(P) + KL、分类损失
  (e) 图 3：用单个高斯拟合双峰分布，分别最小化 KL(p‖q) 与 KL(q‖p)

运行：python code/00_probability_check.py      （在仓库根目录执行；只需 NumPy，几秒内完成）
log 一律是自然对数，单位是奈特；以 2 为底时单位是比特。“正文 x”指笔记里写出的四舍五入值，容差取末位的一半。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 σ μ 等符号会报错
import math
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
rng = np.random.default_rng(0)
ok_all = True


def report(name, expected, computed, tol):
    global ok_all
    err = abs(float(expected) - float(computed))
    passed = err <= tol
    ok_all &= passed
    print(f"  {name:<34s} 应为 {float(expected): .6f}   计算得 {float(computed): .6f}   误差 {err:.1e}   "
          f"{'通过' if passed else '未通过'}")


def check(name, passed, detail):
    global ok_all
    ok_all &= bool(passed)
    print(f"  {name:<34s} {detail}   {'通过' if passed else '未通过'}")


def npdf(x, mu, var):
    """一维高斯密度 N(x; μ, σ²)，第二个参数是方差，式 (3.21)。"""
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def log_npdf(x, mu, var):
    return -(x - mu) ** 2 / (2 * var) - 0.5 * np.log(2 * np.pi * var)


# ---------------------------------------------------------------------------------------------
# (a) 高斯分布
# ---------------------------------------------------------------------------------------------
print("(a) 高斯分布（§3.9.3）")
x = np.linspace(-40, 40, 800_001)
report("高斯积分 ∫exp(-x²/2)dx = √(2π)", math.sqrt(2 * math.pi), trap(np.exp(-x ** 2 / 2), x), 1e-9)
for mu, s in ((0.0, 1.0), (1.5, 0.25), (-2.0, 3.0)):
    report(f"N({mu}, {s}²) 曲线下的总面积", 1.0, trap(npdf(x, mu, s ** 2), x), 1e-9)
beta = 1 / 0.25 ** 2                                                        # 精度 β = 1/σ² = 16
dens_prec = np.sqrt(beta / (2 * np.pi)) * np.exp(-0.5 * beta * (x - 1.5) ** 2)   # 式 (3.22)
report("精度写法 (3.22) 与 (3.21) 的最大差", 0.0, np.abs(dens_prec - npdf(x, 1.5, 0.25 ** 2)).max(), 1e-12)
for s, shown in ((0.25, 1.596), (1.0, 0.399), (2.0, 0.199)):
    report(f"σ = {s} 时的峰值（正文 {shown}）", shown, npdf(0.0, 0.0, s ** 2), 5e-4)
for s in (0.25, 1.0, 2.0):
    xs = np.linspace(-s, s, 200_001)
    prob = trap(npdf(xs, 0.0, s ** 2), xs)
    report(f"σ = {s}：P(μ-σ ≤ X ≤ μ+σ)（正文 0.683）", 0.683, prob, 5e-4)
report("同上，与 erf(1/√2) 比较", math.erf(1 / math.sqrt(2)), prob, 1e-9)
report("σ = 0.25：峰值大于 1，总面积仍为 1", 1.0, trap(npdf(x, 0.0, 0.25 ** 2), x), 1e-9)
xg = np.linspace(-4, 4, 80_001)
d2 = np.gradient(np.gradient(npdf(xg, 0.0, 1.0), xg), xg)                   # 二阶导数
roots = xg[np.where(np.diff(np.sign(d2)) != 0)[0]]
report("N(0,1) 的左拐点（二阶导变号处）", -1.0, roots[0], 1e-3)
report("N(0,1) 的右拐点", 1.0, roots[-1], 1e-3)
report("拐点处的密度（图 1 标注 0.242）", 0.242, npdf(1.0, 0.0, 1.0), 5e-4)


def mvn(xv, mu_v, Sig):
    """式 (3.23)：用协方差矩阵 Σ 写的多维高斯密度。"""
    d = xv - mu_v
    return math.sqrt(1 / ((2 * math.pi) ** len(xv) * np.linalg.det(Sig))) * math.exp(-0.5 * d @ np.linalg.solve(Sig, d))


def mvn_prec(xv, mu_v, B):
    """式 (3.24)：用精度矩阵 β = Σ⁻¹ 写的多维高斯密度。"""
    d = xv - mu_v
    return math.sqrt(np.linalg.det(B) / (2 * math.pi) ** len(xv)) * math.exp(-0.5 * d @ B @ d)


n, s2 = 3, 0.7 ** 2
mu_v = np.array([0.5, -1.0, 2.0])
pts = mu_v + rng.normal(size=(5, n))
Sig_iso = s2 * np.eye(n)                                                     # 各向同性：Σ = σ²I
report("Σ = σ²I 时 (3.23) 与一维密度之积的最大差", 0.0,
       max(abs(mvn(p, mu_v, Sig_iso) - np.prod(npdf(p, mu_v, s2))) for p in pts), 1e-15)
A = rng.normal(size=(n, n))
Sig_gen = A @ A.T + n * np.eye(n)                                            # 一般的对称正定矩阵
report("一般 Σ：(3.24) 与 (3.23) 的最大差", 0.0,
       max(abs(mvn(p, mu_v, Sig_gen) - mvn_prec(p, mu_v, np.linalg.inv(Sig_gen))) for p in pts), 1e-15)
gap = max(abs(mvn(p, mu_v, Sig_gen) - np.prod(npdf(p, mu_v, np.diag(Sig_gen)))) for p in pts)
check("有相关时不能拆成一维密度之积", gap > 1e-4, f"最大差 {gap:.2e}（应明显大于 0）")

# ---------------------------------------------------------------------------------------------
# (b) 贝叶斯规则
# ---------------------------------------------------------------------------------------------
print("(b) 贝叶斯规则（§3.11）")
p_ill, sens, false_pos = 0.01, 0.95, 0.05        # 先验 P(患病)，似然 P(阳性|患病)，误报率 P(阳性|未患病)
joint_ill = sens * p_ill                         # 联合概率 P(患病, 阳性) = P(阳性|患病) P(患病)
joint_ok = false_pos * (1 - p_ill)               # 联合概率 P(未患病, 阳性)
p_pos = joint_ill + joint_ok                     # 全概率：P(阳性) = Σ_x P(阳性|x) P(x)
post_def = joint_ill / p_pos                     # 路线一：条件概率的定义 P(x|y) = P(x,y)/P(y)
post_bayes = p_ill * sens / p_pos                # 路线二：式 (3.42)
report("P(阳性)（正文 0.059）", 0.059, p_pos, 5e-7)
report("P(患病|阳性)（正文 0.161）", 0.161, post_bayes, 5e-4)
report("两条路线的结果相同", post_def, post_bayes, 1e-15)
report("每 1 万人：真阳性 95 人", 95, 10_000 * joint_ill, 1e-9)
report("每 1 万人：假阳性 495 人", 495, 10_000 * joint_ok, 1e-9)
N = 1_000_000
ill = rng.random(N) < p_ill
pos = np.where(ill, rng.random(N) < sens, rng.random(N) < false_pos)
report("模拟 100 万人：阳性者中患病的比例", post_bayes, ill[pos].mean(), 5e-3)
p_pos10 = sens * 0.1 + false_pos * 0.9                                     # 自测题 4：患病率改为 10%
report("自测 4：P(阳性)（正文 0.14）", 0.14, p_pos10, 1e-12)
report("自测 4：P(患病|阳性)（正文 0.679）", 0.679, sens * 0.1 / p_pos10, 5e-4)

# ---------------------------------------------------------------------------------------------
# (c) 自信息与熵
# ---------------------------------------------------------------------------------------------
print("(c) 自信息与熵（§3.13）")
report("正面朝上的信息量 ln 2（正文 0.693 奈特）", 0.693, -math.log(0.5), 5e-4)
report("以 2 为底：1 比特", 1.0, -math.log2(0.5), 0)
report("两次都正面：信息量是一次的两倍", 2 * -math.log(0.5), -math.log(0.25), 1e-15)
report("1 奈特 = 1/ln 2 比特（正文 1.443）", 1.443, 1 / math.log(2), 5e-4)
report("必然事件 P = 1 的信息量", 0.0, -math.log(1.0), 0)


def h2(p):
    """二值熵，按约定 0 log 0 = 0。"""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.zeros_like(p)
    for q in (p, 1 - p):
        m = q > 0
        out[m] -= q[m] * np.log(q[m])
    return out


pg = np.linspace(0, 1, 100_001)
hg = h2(pg)
report("二值熵最大处的 p（正文 0.5）", 0.5, pg[hg.argmax()], 1e-9)
report("二值熵的最大值 ln 2（正文 0.693）", 0.693, hg.max(), 5e-4)
report("H(0.1) = H(0.9)（正文 0.325）", 0.325, h2(0.1)[0], 5e-4)
report("H(0) + H(1)：确定的分布熵为 0", 0.0, h2(0.0)[0] + h2(1.0)[0], 0)
D = rng.dirichlet(np.ones(4), size=100_000)                                 # 4 个结果上的随机分布
h_max = (-np.sum(D * np.log(D), axis=1)).max()
check("随机分布的熵都不超过 ln 4", h_max <= math.log(4), f"最大 {h_max:.4f} ≤ ln 4 = {math.log(4):.4f}")
for s, shown in ((1.0, 1.419), (0.1, -0.884)):
    xx = np.linspace(-12 * s, 12 * s, 400_001)
    pp = npdf(xx, 0.0, s ** 2)
    num = -trap(pp * np.log(pp), xx)
    report(f"N(0,{s}²) 的微分熵（正文 {shown}）", shown, num, 5e-4)
    report("  与 ½ln(2πeσ²) 比较", 0.5 * math.log(2 * math.pi * math.e * s ** 2), num, 1e-8)
report("U(0, 0.5) 的微分熵 ln 0.5（正文 -0.693）", -0.693, math.log(0.5), 5e-4)

# ---------------------------------------------------------------------------------------------
# (d) KL 散度与交叉熵
# ---------------------------------------------------------------------------------------------
print("(d) KL 散度与交叉熵（§3.13）")


def kl(a, b):
    m = a > 0                                                                 # 0 log 0 = 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))


def ent(a):
    m = a > 0
    return float(-np.sum(a[m] * np.log(a[m])))


def xent(a, b):
    m = a > 0
    return float(-np.sum(a[m] * np.log(b[m])))


P, Q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
report("KL(P‖Q)（正文 0.511）", 0.511, kl(P, Q), 5e-4)
report("KL(Q‖P)（正文 0.368）", 0.368, kl(Q, P), 5e-4)
report("KL(P‖P)", 0.0, kl(P, P), 0)
report("KL(P‖Q) 换成比特（正文 0.737）", 0.737, kl(P, Q) / math.log(2), 5e-4)
code_q = -np.log2(Q)                                                          # 按 Q 设计的理想码长（比特）
report("按 Q 设计的码长：结果 1（正文 0.152）", 0.152, code_q[0], 5e-4)
report("按 Q 设计的码长：结果 2（正文 3.322）", 3.322, code_q[1], 5e-4)
report("用这套码发送 P 的平均码长（正文 1.737）", 1.737, P @ code_q, 5e-4)
report("比按 P 设计（1 比特）多出的 = KL", kl(P, Q) / math.log(2), P @ code_q - 1.0, 1e-12)
report("H(P)（正文 0.693）", 0.693, ent(P), 5e-4)
report("H(P,Q)（正文 1.204）", 1.204, xent(P, Q), 5e-4)
report("H(P,Q) = H(P) + KL(P‖Q)", ent(P) + kl(P, Q), xent(P, Q), 1e-12)
A5, B5 = rng.dirichlet(np.ones(5), size=100_000), rng.dirichlet(np.ones(5), size=100_000)
kls = np.sum(A5 * np.log(A5 / B5), axis=1)
check("10 万对随机分布的 KL 都 ≥ 0", kls.min() >= 0, f"最小值 {kls.min():.2e}")
qg = np.linspace(0.001, 0.999, 999)
ce_curve = [xent(P, np.array([q, 1 - q])) for q in qg]
kl_curve = [kl(P, np.array([q, 1 - q])) for q in qg]
report("对 Q 最小化交叉熵：最优 q（应为 0.5）", 0.5, qg[int(np.argmin(ce_curve))], 1e-9)
report("对 Q 最小化 KL：最优 q 相同", qg[int(np.argmin(ce_curve))], qg[int(np.argmin(kl_curve))], 0)
onehot, pred = np.array([1.0, 0.0]), np.array([0.7, 0.3])
report("独热标签：H(P) = 0（用到 0 log 0 = 0）", 0.0, ent(onehot), 0)
report("分类损失 -log 0.7（正文 0.357）", 0.357, xent(onehot, pred), 5e-4)

# ---------------------------------------------------------------------------------------------
# (e) 图 3：KL 的两个方向。p 是双峰分布，q 只能是单个高斯 N(μ, σ²)
# ---------------------------------------------------------------------------------------------
print("(e) 图 3：用单个高斯拟合 p = 0.5 N(-2, 0.6²) + 0.5 N(2, 0.6²)")
xs = np.linspace(-20, 20, 8001)
logp = np.logaddexp(np.log(0.5) + log_npdf(xs, -2.0, 0.36), np.log(0.5) + log_npdf(xs, 2.0, 0.36))
pmix = np.exp(logp)
Ep, Ep2, Hp = trap(pmix * xs, xs), trap(pmix * xs ** 2, xs), -trap(pmix * logp, xs)


def kl_forward(mu, s):
    """KL(p‖q) = -H(p) - E_p[log q]，E_p[log q] 只用到 p 的一阶、二阶矩。"""
    return -Hp + 0.5 * np.log(2 * np.pi * s ** 2) + (Ep2 - 2 * mu * Ep + mu ** 2) / (2 * s ** 2)


def kl_reverse(mu, s):
    """KL(q‖p) = -H(q) - E_q[log p]，H(q) = ½ln(2πeσ²)，E_q[log p] 在网格上积分。"""
    mu, s = np.broadcast_arrays(np.asarray(mu, float), np.asarray(s, float))
    out = np.empty(mu.shape)
    for idx in np.ndindex(mu.shape):
        q = npdf(xs, mu[idx], s[idx] ** 2)
        out[idx] = -0.5 * np.log(2 * np.pi * np.e * s[idx] ** 2) - trap(q * logp, xs)
    return out


def grid_min(f, mu_lo, mu_hi, s_lo, s_hi):
    """先用步长 0.05 的粗网格，再在最优点附近逐级细化到步长 0.00005，返回 (μ*, σ*, KL 最小值)。"""
    for step_mu, step_s in ((0.05, 0.05), (0.005, 0.005), (0.0005, 0.0005), (0.00005, 0.00005)):
        M, S = np.meshgrid(np.arange(mu_lo, mu_hi + step_mu / 2, step_mu), np.arange(s_lo, s_hi + step_s / 2, step_s))
        V = f(M, S)
        k = np.unravel_index(np.argmin(V), V.shape)
        mu_best, s_best = M[k], S[k]
        mu_lo, mu_hi, s_lo, s_hi = mu_best - 10 * step_mu, mu_best + 10 * step_mu, s_best - 10 * step_s, s_best + 10 * step_s
    return mu_best, s_best, V[k]


report("p 的均值", 0.0, Ep, 1e-9)
report("p 的方差 = 0.36 + 4（正文 4.36）", 4.36, Ep2 - Ep ** 2, 1e-9)
mu_f, s_f, v_f = grid_min(kl_forward, -3.0, 3.0, 0.3, 4.0)
report("正向 KL 最优 μ（矩匹配：p 的均值 0）", 0.0, mu_f, 1e-3)
report("正向 KL 最优 σ（矩匹配：√4.36 = 2.088）", math.sqrt(4.36), s_f, 1e-3)
report("正向 KL 的最小值（正文 0.555）", 0.555, v_f, 5e-4)
mu_r, s_r, v_r = grid_min(kl_reverse, 0.0, 3.0, 0.3, 2.0)                    # 另一个最优解与之关于 0 对称
report("反向 KL 最优 μ（正文 1.997，贴住右峰）", 1.997, mu_r, 5e-4)
report("反向 KL 最优 σ（正文 0.605，接近单峰的 0.6）", 0.605, s_r, 5e-4)
report("反向 KL 的最小值（正文 0.692，略小于 ln 2）", 0.692, v_r, 5e-4)
report("左右两个解对称：左峰的反向 KL", v_r, kl_reverse(-mu_r, s_r).item(), 1e-9)
report("正向最优解的反向 KL（正文 1.27）", 1.27, kl_reverse(mu_f, s_f).item(), 5e-3)
report("反向最优解的正向 KL（正文 10.2）", 10.2, kl_forward(mu_r, s_r), 5e-2)

print("\n总结：", "全部通过" if ok_all else "有未通过项，请检查上面标记")
