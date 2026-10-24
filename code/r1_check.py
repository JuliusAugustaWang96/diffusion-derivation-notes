# -*- coding: utf-8 -*-
"""r1_check.py —— 用数值方法核对论文精读 R1（DDPM 原论文推导精讲）中出现的全部数字。

噪声调度与 Ho et al. (2020) 相同：T = 1000，β_t 从 1e-4 线性增加到 0.02。
  (a) 命题 1（R1 第 2.4 节）：ᾱ_t 的代表值、首次降到 0.5 以下的 t、恒等式 1 − ᾱ_t = α_t(1 − ᾱ_{t-1}) + β_t、
      两步例子 β = (0.1, 0.2) 的闭式与模拟、逐步加噪 1000 步的模拟与一步到位公式
  (b) 命题 2（R1 第 2.5 节）：KL(q(x_T|x_0) ‖ N(0, I)) 的每维数值，苏剑林（一）的估计
  (c) 定理 3（R1 第 4 章）：后验系数表、联合高斯条件公式、精度相加的恒等式、t = 10 处的回归核对、图 5 两例
  (d) 定理 5、命题 4（R1 第 5 章）：式 (7) 代入 x_0 后与式 (10) 相等、x_0 损失与 ε 损失的因子
  (e) 命题 5（R1 第 6.1 节）：式 (12) 的权重在两种 σ_t² 下的取值
  (f) 命题 6（R1 第 6.3 节）：数据为 N(0, 1) 时反向方差是 β_t，数据为单点时是 β̃_t
  (g) 命题 3（R1 第 3.4 节）：一维线性高斯玩具 T = 3 上联合 KL = E[L] − H 的逐项数值
  (h) 引理 2 的例子（R1 第 5.1 节）
  (i) R1 第 9 章引用的 code/04_ddpm_toy2d.py 行号

运行：python code/r1_check.py      （在仓库根目录执行；只需 NumPy，约 10 秒完成）
“正文 x”指笔记里写出的四舍五入值，容差取末位的一半；模拟值的随机种子固定，与正文逐位相同。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 ᾱ ε 等符号会报错
import math, pathlib, time
import numpy as np

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
HERE = pathlib.Path(__file__).resolve().parent
ok_all = True
t_start = time.time()


def check(label, text, value, tol):
    global ok_all
    good = abs(value - text) <= tol
    ok_all &= good
    print(f"  {label}: 正文 {text:g}  计算 {value:.6g}  {'通过' if good else '不符'}")


def check_true(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  {label}: {'通过' if cond else '不符'}")


rng = np.random.default_rng(0)          # 与绘图脚本同一种子、同一抽样顺序，模拟值才能逐位重现
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = np.cumprod(alphas)
abar_prev = np.concatenate([[1.0], abar[:-1]])
beta_tilde = betas * (1.0 - abar_prev) / (1.0 - abar)
coef_x0 = np.sqrt(abar_prev) * betas / (1.0 - abar)
coef_xt = np.sqrt(alphas) * (1.0 - abar_prev) / (1.0 - abar)
i = lambda t: t - 1

# ------------------------------------------------------------------ (a) 命题 1
print("(a) 命题 1：调度、两步例子、1000 步模拟")
check("β_1", 1e-4, betas[0], 1e-9)
check("β_T", 0.02, betas[-1], 1e-9)
check("Σβ_t", 10.05, betas.sum(), 0.005)
for t, v in [(1, 0.9999), (10, 0.9981), (100, 0.8970), (250, 0.5241), (500, 0.0786), (750, 0.0034)]:
    check(f"ᾱ_{t}", v, abar[i(t)], 5e-5)
check("ᾱ_1000 (4.0e-5)", 4.0e-5, abar[-1], 5e-7)
check("首次 ᾱ_t ≤ 0.5 的 t", 260, int(np.argmax(abar <= 0.5)) + 1, 0)
check_true("恒等式 1−ᾱ_t = α_t(1−ᾱ_{t−1}) + β_t（全部 t）", np.allclose(1 - abar, alphas * (1 - abar_prev) + betas, atol=1e-15))
b1, b2 = 0.1, 0.2
check("两步例子 x_0 的系数 √0.72", 0.8485, math.sqrt((1 - b1) * (1 - b2)), 5e-5)
check("两步例子 合并噪声标准差 √0.28", 0.5292, math.sqrt(1 - (1 - b1) * (1 - b2)), 5e-5)
n = 1_000_000
e1, e2 = rng.standard_normal(n), rng.standard_normal(n)
x2 = math.sqrt(1 - b2) * (math.sqrt(1 - b1) * 1.0 + math.sqrt(b1) * e1) + math.sqrt(b2) * e2
check("两步例子 模拟均值（x_0 = 1，一百万次）", 0.8492, x2.mean(), 5e-5)
check("两步例子 模拟标准差", 0.5288, x2.std(), 5e-5)
m = 20000
x = np.full(m, 1.0)
sim = {}
for t in range(1, T + 1):
    x = math.sqrt(alphas[i(t)]) * x + math.sqrt(betas[i(t)]) * rng.standard_normal(m)
    if t in (500, 1000):
        sim[t] = (x.mean(), x.std())
check("1000 步模拟 t=500 均值", 0.275, sim[500][0], 5e-4)
check("1000 步模拟 t=500 标准差", 0.959, sim[500][1], 5e-4)
check("闭式 t=500 均值 √ᾱ", 0.280, math.sqrt(abar[i(500)]), 5e-4)
check("闭式 t=500 标准差 √(1−ᾱ)", 0.960, math.sqrt(1 - abar[i(500)]), 5e-4)
check("1000 步模拟 t=1000 均值", 0.015, sim[1000][0], 5e-4)
check("1000 步模拟 t=1000 标准差", 1.005, sim[1000][1], 5e-4)
check("闭式 t=1000 均值", 0.006, math.sqrt(abar[-1]), 5e-4)
check("闭式 t=1000 标准差", 1.000, math.sqrt(1 - abar[-1]), 5e-4)
check_true("t=1000 的模拟均值与闭式之差在两万条链的抽样误差内（< 2 个标准误）", abs(sim[1000][0] - math.sqrt(abar[-1])) < 2 / math.sqrt(m))

# ------------------------------------------------------------------ (b) 命题 2
print("(b) 命题 2：信号被摧毁")
aT = abar[-1]
kl_T = 0.5 * (aT * 1.0 - aT - math.log(1 - aT))
check("KL(q(x_T|x_0=1) ‖ N(0,1)) 每维（纳特，2.0e-5）", 2.0e-5, kl_T, 5e-7)
check("同上，比特（2.9e-5）", 2.9e-5, kl_T / math.log(2), 5e-7)
check("苏剑林的估计：0.5·log ᾱ_T（−5.059）", -5.059, 0.5 * math.log(aT), 5e-4)
check("−0.005(T+1)", -5.005, -0.005 * (T + 1), 1e-9)
check_true("0.5·log ᾱ_T < −0.005(T+1)", 0.5 * math.log(aT) < -0.005 * (T + 1))
check("exp(−5.005)", 0.0067, math.exp(-5.005), 5e-5)
check("实际 √ᾱ_T", 0.0064, math.sqrt(aT), 5e-5)
check_true("|x_0i| ≤ 1 时 x_0i = 1 给出每维 KL 的上限", kl_T >= 0.5 * (aT * 0.5 ** 2 - aT - math.log(1 - aT)))

# ------------------------------------------------------------------ (c) 定理 3
print("(c) 定理 3：后验")
table = {1: (1.0000, 0.0000, 1.0000, 0.0), 2: (0.5453, 0.4547, 1.0000, 0.4547), 10: (0.1473, 0.8527, 1.0000, 0.8528),
         100: (0.0191, 0.9809, 1.0000, 0.9819), 500: (0.0031, 0.9941, 0.9972, 0.9991), 1000: (0.0001, 0.9899, 0.9901, 1.0000)}
for t, (c0, ct, s, r) in table.items():
    check(f"t={t} x_0 的系数", c0, coef_x0[i(t)], 5e-5)
    check(f"t={t} x_t 的系数", ct, coef_xt[i(t)], 5e-5)
    check(f"t={t} 两系数之和", s, coef_x0[i(t)] + coef_xt[i(t)], 5e-5)
    check(f"t={t} β̃_t/β_t", r, beta_tilde[i(t)] / betas[i(t)], 5e-5)
check("β̃_1", 0.0, beta_tilde[0], 1e-15)
check("β̃_100/β_100（6.3 节的 0.98）", 0.98, beta_tilde[i(100)] / betas[i(100)], 5e-3)
check("β̃_10/β_10（6.3 节的 0.85）", 0.85, beta_tilde[i(10)] / betas[i(10)], 5e-3)
check("β̃_1000/β_1000（6.3 节的 0.999999）", 0.999999, beta_tilde[-1] / betas[-1], 5e-7)
v_prev, v_t = 1 - abar_prev, 1 - abar
cov = np.sqrt(alphas) * v_prev
k = cov / v_t
check_true("联合高斯条件公式给出 x_t 的系数", np.allclose(k, coef_xt, rtol=1e-10))
check_true("联合高斯条件公式给出 x_0 的系数", np.allclose(np.sqrt(abar_prev) - k * np.sqrt(abar), coef_x0, rtol=1e-8, atol=1e-14))
check_true("联合高斯条件公式给出 β̃_t", np.allclose(v_prev - cov ** 2 / v_t, beta_tilde, rtol=1e-8, atol=1e-16))
tt = np.arange(2, T + 1)
check_true("1/β̃_t = α_t/β_t + 1/(1−ᾱ_{t−1})（t ≥ 2）", np.allclose(1 / beta_tilde[i(tt)], alphas[i(tt)] / betas[i(tt)] + 1 / (1 - abar_prev[i(tt)]), rtol=1e-10))
t = 10
e1, e2 = rng.standard_normal(400_000), rng.standard_normal(400_000)
xp = math.sqrt(abar_prev[i(t)]) * 1.0 + math.sqrt(1 - abar_prev[i(t)]) * e1
xt = math.sqrt(alphas[i(t)]) * xp + math.sqrt(betas[i(t)]) * e2
sl, ic = np.linalg.lstsq(np.c_[xt, np.ones_like(xt)], xp, rcond=None)[0]
res = xp - (sl * xt + ic)
check("t=10 回归斜率（x_0 = 1，四十万组）", 0.8531, sl, 5e-5)
check("t=10 回归截距", 0.1469, ic, 5e-5)
check("t=10 残差方差（2.385e-4）", 2.385e-4, res.var(), 5e-8)
check("t=10 β̃_10（2.382e-4）", 2.382e-4, beta_tilde[i(t)], 5e-8)
for (t, x0v, xtv, m_txt, v_txt) in [(2, 1.0, 1.02, 1.0091, 5.45e-5), (100, 1.0, 1.2, 1.1962, 2.04e-3)]:
    j = i(t)
    m_post, v_post = coef_x0[j] * x0v + coef_xt[j] * xtv, beta_tilde[j]
    check(f"图 5 t={t} 后验均值", m_txt, m_post, 5e-5)
    check(f"图 5 t={t} 后验方差", v_txt, v_post, 5e-3 * v_txt)
    m_prev, s_prev, s_lik = math.sqrt(abar_prev[j]) * x0v, math.sqrt(1 - abar_prev[j]), math.sqrt(betas[j])
    lo = min(m_prev - 3.5 * s_prev, xtv / math.sqrt(alphas[j]) - 3.5 * s_lik)
    hi = max(m_prev + 3.5 * s_prev, xtv / math.sqrt(alphas[j]) + 3.5 * s_lik)
    g = np.linspace(lo, hi, 4001)
    prod = np.exp(-0.5 * ((g - m_prev) / s_prev) ** 2) * np.exp(-0.5 * ((xtv - math.sqrt(alphas[j]) * g) / s_lik) ** 2)
    post = prod / trap(prod, g)
    mean_num = trap(g * post, g)
    var_num = trap((g - mean_num) ** 2 * post, g)
    check_true(f"图 5 t={t} 网格乘积的均值、方差与公式一致", abs(mean_num - m_post) < 1e-4 and abs(var_num / v_post - 1) < 1e-3)

# ------------------------------------------------------------------ (d) 定理 5、命题 4
print("(d) 定理 5、命题 4：ε 参数化")
ts = rng.integers(1, T + 1, 1000)
x0s = rng.uniform(-1, 1, 1000)
eps = rng.standard_normal(1000)
j = i(ts)
xts = np.sqrt(abar[j]) * x0s + np.sqrt(1 - abar[j]) * eps
mu7 = coef_x0[j] * x0s + coef_xt[j] * xts
mu10 = (xts - betas[j] / np.sqrt(1 - abar[j]) * eps) / np.sqrt(alphas[j])
d = np.max(np.abs(mu7 - mu10))
print(f"  式 (7) 代入 x_0 后与式 (10) 的最大差：{d:.1e}（正文：2.6e-14，浮点误差量级）")
check_true("最大差小于 1e-12", d < 1e-12)
check_true("式 (15) 由 ε 还原 x_0", np.max(np.abs((xts - np.sqrt(1 - abar[j]) * eps) / np.sqrt(abar[j]) - x0s)) < 1e-9)
for t, v, tol in [(10, 0.002, 5e-4), (100, 0.11, 5e-3), (500, 12, 0.5), (1000, 2.5e4, 500)]:
    check(f"x_0 损失/ε 损失的因子 (1−ᾱ_t)/ᾱ_t，t={t}", v, (1 - abar[i(t)]) / abar[i(t)], tol)
check("同上 t=1（0.0001）", 0.0001, (1 - abar[0]) / abar[0], 5e-5)

# ------------------------------------------------------------------ (e) 命题 5
print("(e) 命题 5：式 (12) 的权重")
w_beta = betas ** 2 / (2 * betas * alphas * (1 - abar))
w_tilde = betas[1:] ** 2 / (2 * beta_tilde[1:] * alphas[1:] * (1 - abar[1:]))
for t, v, tol in [(1, 0.50, 5e-3), (10, 0.074, 5e-4), (100, 0.010, 5e-4), (500, 0.0055, 5e-5), (1000, 0.010, 5e-4)]:
    check(f"σ²=β_t 权重 t={t}", v, w_beta[i(t)], tol)
check("σ²=β_t 权重的最小值", 0.0050, w_beta.min(), 5e-5)
check("最小值所在的 t", 349, int(np.argmin(w_beta)) + 1, 0)
check("σ²=β̃_t 权重 t=2", 0.60, w_tilde[0], 5e-3)
check_true("σ²=β̃_t 的权重在 t ≥ 10 时与 σ²=β_t 的相差不到 20%", np.max(np.abs(w_tilde[8:] / w_beta[9:] - 1)) < 0.2)

# ------------------------------------------------------------------ (f) 命题 6
print("(f) 命题 6：两种 σ_t²")
t = 10
n = 400_000
x0 = rng.standard_normal(n)
xp = math.sqrt(abar_prev[i(t)]) * x0 + math.sqrt(1 - abar_prev[i(t)]) * rng.standard_normal(n)
xt = math.sqrt(alphas[i(t)]) * xp + math.sqrt(betas[i(t)]) * rng.standard_normal(n)
sl, ic = np.linalg.lstsq(np.c_[xt, np.ones(n)], xp, rcond=None)[0]
res = xp - (sl * xt + ic)
check("数据 N(0,1)，t=10 回归斜率", 0.9998, sl, 5e-5)
check("√α_10", 0.9999, math.sqrt(alphas[i(t)]), 5e-5)
check("数据 N(0,1)，残差方差（2.789e-4）", 2.789e-4, res.var(), 5e-8)
check("β_10（2.793e-4）", 2.793e-4, betas[i(t)], 5e-8)
check("β̃_10（2.382e-4）", 2.382e-4, beta_tilde[i(t)], 5e-8)
check_true("精确值：数据 N(0,1) 时 Var(x_t) = 1，Cov(x_{t−1},x_t) = √α_t，条件方差 = β_t", abs((1 - alphas[i(t)]) - betas[i(t)]) < 1e-15)

# ------------------------------------------------------------------ (g) 命题 3
print("(g) 命题 3：联合 KL = E[L] − H（一维线性高斯玩具，T = 3）")
m0, s0 = 0.5, 1.5
bt = np.array([0.1, 0.2, 0.3]); at = 1 - bt; abt = np.cumprod(at); abt_prev = np.concatenate([[1.0], abt[:-1]])
btil = bt * (1 - abt_prev) / (1 - abt); cx0 = np.sqrt(abt_prev) * bt / (1 - abt); cxt = np.sqrt(at) * (1 - abt_prev) / (1 - abt)
a_t = np.array([0.95, 0.90, 0.85]); b_t = np.array([0.05, 0.10, 0.00]); s2_t = bt.copy(); Tt = 3
Aq = np.zeros((Tt + 1, Tt + 1)); cq = np.zeros(Tt + 1); Aq[0, 0] = s0; cq[0] = m0
for t in range(1, Tt + 1):
    Aq[t] = math.sqrt(at[t - 1]) * Aq[t - 1]; Aq[t, t] += math.sqrt(bt[t - 1]); cq[t] = math.sqrt(at[t - 1]) * cq[t - 1]
Ap = np.zeros((Tt + 1, Tt + 1)); cp = np.zeros(Tt + 1); Ap[Tt, Tt] = 1.0
for t in range(Tt, 0, -1):
    Ap[t - 1] = a_t[t - 1] * Ap[t]; Ap[t - 1, t - 1] += math.sqrt(s2_t[t - 1]); cp[t - 1] = a_t[t - 1] * cp[t] + b_t[t - 1]
Sq, muq, Sp, mup = Aq @ Aq.T, cq, Ap @ Ap.T, cp
Spi = np.linalg.inv(Sp); dd = mup - muq
kl_joint = 0.5 * (np.trace(Spi @ Sq) + dd @ Spi @ dd - (Tt + 1) + math.log(np.linalg.det(Sp) / np.linalg.det(Sq)))
E_sq = lambda c, c0: float(c @ c + c0 ** 2)
L_T = 0.5 * (abt[-1] * (m0 ** 2 + s0 ** 2) + (1 - abt[-1]) - 1 - math.log(1 - abt[-1]))
L_mid = 0.0
for t in range(2, Tt + 1):
    c = cx0[t - 1] * Aq[0] + cxt[t - 1] * Aq[t] - a_t[t - 1] * Aq[t]
    c0 = cx0[t - 1] * cq[0] + cxt[t - 1] * cq[t] - a_t[t - 1] * cq[t] - b_t[t - 1]
    L_mid += 0.5 * (math.log(s2_t[t - 1] / btil[t - 1]) + (btil[t - 1] + E_sq(c, c0)) / s2_t[t - 1] - 1)
c = Aq[0] - a_t[0] * Aq[1]; c0 = cq[0] - a_t[0] * cq[1] - b_t[0]
L_0 = 0.5 * math.log(2 * math.pi * s2_t[0]) + E_sq(c, c0) / (2 * s2_t[0])
E_L = L_T + L_mid + L_0
H0 = 0.5 * math.log(2 * math.pi * math.e * s0 ** 2)
kl_marg = 0.5 * (math.log(Sp[0, 0] / s0 ** 2) + (s0 ** 2 + (m0 - mup[0]) ** 2) / Sp[0, 0] - 1)
check("L_T", 0.7286, L_T, 5e-5)
check("L_1 + L_2", 1.0584, L_mid, 5e-5)
check("L_0", 0.3286, L_0, 5e-5)
check("E[L]", 2.1155, E_L, 5e-5)
check("H(q(x_0))", 1.8244, H0, 5e-5)
check("E[L] − H", 0.2911, E_L - H0, 5e-5)
check("联合 KL（四维高斯闭式）", 0.2911, kl_joint, 5e-5)
check("KL(q(x_0) ‖ p_θ(x_0))", 0.2640, kl_marg, 5e-5)
check("缺口 E KL(q(x_{1:3}|x_0) ‖ p_θ(x_{1:3}|x_0))", 0.0271, kl_joint - kl_marg, 5e-5)
check_true("E[L] − H 与联合 KL 相等（1e-9）", abs((E_L - H0) - kl_joint) < 1e-9)

# ------------------------------------------------------------------ (h) 引理 2
print("(h) 引理 2 的例子")
check("KL(N(0.3,0.25) ‖ N(1.1,0.25))", 1.28, 0.8 ** 2 / (2 * 0.25), 5e-3)

# ------------------------------------------------------------------ (i) 行号
print("(i) R1 第 9 章引用的 code/04_ddpm_toy2d.py 行号")
src = (HERE / "04_ddpm_toy2d.py").read_text(encoding="utf-8").split("\n")
expect = {48: "torch.linspace(beta_1, beta_T, T", 49: "self.alphas = 1.0 - self.betas", 50: "torch.cumprod", 51: "alpha_bar_prev",
          52: "self.sqrt_ab = ", 53: "self.sqrt_1m_ab = ", 55: "self.posterior_var = ", 56: "self.coef_x0 = ", 57: "self.coef_xt = ",
          60: "def q_sample", 62: "sch.sqrt_ab[t, None] * x0 + sch.sqrt_1m_ab[t, None] * eps", 65: "def q_posterior", 69: "return mean, var",
          102: "def p_losses", 105: "torch.randint(0, sch.T", 106: "torch.randn_like(x0)", 107: "q_sample(sch, x0, t, eps)",
          108: "F.mse_loss(model(xt, t), eps)", 112: "def p_sample", 116: "mean = (xt - sch.betas[i] / sch.sqrt_1m_ab[i] * eps) / sch.alphas[i].sqrt()",
          117: "if i == 0:", 118: "return mean", 119: "sch.posterior_var[i].sqrt() * torch.randn_like(xt)", 123: "def p_sample_loop", 131: "return x, traj"}
for ln, frag in expect.items():
    check_true(f"第 {ln} 行含 {frag[:40]}", frag in src[ln - 1])

print(f"用时 {time.time() - t_start:.1f} s")
print("全部通过" if ok_all else "有不符项，请检查")
sys.exit(0 if ok_all else 1)
