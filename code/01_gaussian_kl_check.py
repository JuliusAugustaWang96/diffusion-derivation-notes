# -*- coding: utf-8 -*-
"""01_gaussian_kl_check.py —— 用数值方法验证四条推导（对应本系列推导题 D1 / D4 / D5 / D9）。

  (a) D1  两个对角高斯的 KL 散度闭式：闭式 vs 蒙特卡洛 vs torch.distributions
  (b) D4  前向过程闭式 q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)：逐步采样的样本均值/方差 vs 闭式
  (c) D5  前向后验 q(x_{t-1}|x_t,x_0) 的均值 μ̃_t 与方差 β̃_t：一维网格上按贝叶斯公式数值归一化 vs 闭式
  (d) D9  score 与噪声的关系 ∇log q(x_t) = -E[ε|x_t]/√(1-ᾱ_t)：两分量高斯混合上的解析 score vs 条件期望

运行：python code/01_gaussian_kl_check.py      （在仓库根目录执行；CPU 即可，约 10 秒）
记号按 Ho et al. 2020：β_t 线性调度 1e-4→0.02，T=1000；α_t = 1-β_t；ᾱ_t = ∏_{s≤t} α_s。
数组下标 i = t-1（即 betas[0] 是 t=1 的 β_1）。全部用 float64，闭式与精确数值之间的差应在 1e-4 以内。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，打印 ᾱ ε 等符号会报错
import math
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000
betas = torch.linspace(1e-4, 0.02, T)                 # β_t，线性调度
alphas = 1.0 - betas                                   # α_t
alpha_bar = torch.cumprod(alphas, dim=0)               # ᾱ_t = ∏ α_s          （D4 的核心量）
alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])   # ᾱ_{t-1}，约定 ᾱ_0 = 1

ok_all = True


def report(name, formula, numeric, tol):
    global ok_all
    err = abs(float(formula) - float(numeric))
    passed = err < tol
    ok_all &= passed
    print(f"  {name:<34s} 公式值 {float(formula): .8f}   数值值 {float(numeric): .8f}   误差 {err:.2e}   {'通过' if passed else '未通过'}")


# ---------------------------------------------------------------------------------------------
# (a) D1：两个对角高斯的 KL
# ---------------------------------------------------------------------------------------------
def kl_diag_gaussians(mu1, s1, mu2, s2):
    """M4(a)(b)：KL(N(μ1,σ1²) ‖ N(μ2,σ2²)) = Σ_j [ log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2 ]"""
    return (torch.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5).sum()


print("(a) D1 两个对角高斯的 KL 散度")
mu1, s1 = torch.tensor([0.5, -1.0, 2.0]), torch.tensor([1.2, 0.7, 0.4])
mu2, s2 = torch.tensor([0.0, 0.5, 1.0]), torch.tensor([1.0, 1.0, 0.8])
kl_formula = kl_diag_gaussians(mu1, s1, mu2, s2)
p = torch.distributions.Independent(torch.distributions.Normal(mu1, s1), 1)
q = torch.distributions.Independent(torch.distributions.Normal(mu2, s2), 1)
kl_torch = torch.distributions.kl_divergence(p, q)
x = p.sample((1_000_000,))                                           # 蒙特卡洛：E_p[log p - log q]
kl_mc = (p.log_prob(x) - q.log_prob(x)).mean()
report("闭式 vs torch.distributions", kl_formula, kl_torch, 1e-6)
report("闭式 vs 蒙特卡洛(1e6 样本)", kl_formula, kl_mc, 2e-2)
# 特例 M4(e)：与标准正态的 KL = ½ Σ (μ² + σ² - 1 - log σ²)
kl_std = 0.5 * (mu1 ** 2 + s1 ** 2 - 1 - torch.log(s1 ** 2)).sum()
report("与 N(0,I) 的 KL 特例", kl_std, kl_diag_gaussians(mu1, s1, torch.zeros(3), torch.ones(3)), 1e-12)

# ---------------------------------------------------------------------------------------------
# (b) D4：前向闭式——逐步采样 x_t = √α_t x_{t-1} + √β_t ε_t，与闭式 N(√ᾱ_t x_0, (1-ᾱ_t) I) 比较
# ---------------------------------------------------------------------------------------------
print("(b) D4 前向过程闭式 q(x_t|x_0)（逐步采样 100000 条链，蒙特卡洛容差 2e-2）")
x0 = torch.tensor([1.0, -2.0, 0.5], device=dev)
N = 100_000
x = x0.repeat(N, 1)
checkpoints = {10, 100, 500, 1000}
for t in range(1, T + 1):
    i = t - 1
    x = torch.sqrt(alphas[i]) * x + torch.sqrt(betas[i]) * torch.randn_like(x)      # 前向一步 q(x_t|x_{t-1})
    if t in checkpoints:
        m_num, v_num = x.mean(0), x.var(0)
        m_formula, v_formula = torch.sqrt(alpha_bar[i]) * x0, (1 - alpha_bar[i])
        report(f"t={t:4d} 均值(第 1 维)", m_formula[0], m_num[0], 2e-2)
        report(f"t={t:4d} 方差(第 1 维)", v_formula, v_num[0], 2e-2)
print(f"      参考：ᾱ_10={alpha_bar[9]:.4f}  ᾱ_100={alpha_bar[99]:.4f}  ᾱ_500={alpha_bar[499]:.4f}  ᾱ_1000={alpha_bar[999]:.2e}")

# ---------------------------------------------------------------------------------------------
# (c) D5：前向后验 q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1}) q(x_{t-1}|x_0)，一维网格数值归一化
# ---------------------------------------------------------------------------------------------
print("(c) D5 前向后验 q(x_{t-1}|x_t,x_0) 的均值与方差（一维网格，步长 1e-5）")
x0_1d = 0.7
grid = torch.linspace(-8.0, 8.0, 1_600_001, device=dev)            # x_{t-1} 的取值网格
for t in (2, 10, 500):
    i = t - 1
    ab, ab_prev, a, b = alpha_bar[i], alpha_bar_prev[i], alphas[i], betas[i]
    xt = math.sqrt(ab) * x0_1d + math.sqrt(1 - ab) * 0.3                          # 任取一个 x_t（ε=0.3）
    # 两个因子的对数密度（M1(a)），常数项无关紧要
    logw = -(xt - torch.sqrt(a) * grid) ** 2 / (2 * b) - (grid - math.sqrt(ab_prev) * x0_1d) ** 2 / (2 * (1 - ab_prev))
    w = torch.softmax(logw, dim=0)                                                # 数值归一化（M3 的"配方法"由计算机代劳）
    mean_num = (w * grid).sum()
    var_num = (w * (grid - mean_num) ** 2).sum()
    # 闭式（D5）
    mean_formula = (torch.sqrt(ab_prev) * b / (1 - ab)) * x0_1d + (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * xt
    var_formula = (1 - ab_prev) / (1 - ab) * b                                    # β̃_t
    report(f"t={t:4d} 后验均值 μ̃_t", mean_formula, mean_num, 1e-4)
    report(f"t={t:4d} 后验方差 β̃_t", var_formula, var_num, 1e-4)
    print(f"      参考：β_t={float(b):.3e}，β̃_t/β_t={float(var_formula / b):.4f}")

# ---------------------------------------------------------------------------------------------
# (d) D9：score 与 E[ε|x_t]。数据 q(x_0) = 0.5 N(-2, 0.3²) + 0.5 N(2, 0.3²)
#     q(x_t) = Σ_k π_k N(x_t; √ᾱ μ_k, ᾱ s_k² + 1-ᾱ)    （M1(b)(c)：高斯经线性变换再加独立噪声仍是高斯）
#     解析 score 用自动微分算 ∇ log q(x_t)；E[x_0|x_t] 用"每个分量内 (x_0,x_t) 联合高斯"的条件公式 M2 算，
#     再由 x_t = √ᾱ x_0 + √(1-ᾱ) ε 得 E[ε|x_t] = (x_t - √ᾱ E[x_0|x_t]) / √(1-ᾱ)。
# ---------------------------------------------------------------------------------------------
print("(d) D9 score = -E[ε|x_t]/√(1-ᾱ_t)（两分量高斯混合，精确恒等式）")
pi_k = torch.tensor([0.5, 0.5])
mu_k = torch.tensor([-2.0, 2.0])
s_k = torch.tensor([0.3, 0.3])
for t in (50, 500):
    i = t - 1
    ab = alpha_bar[i]
    m_k = torch.sqrt(ab) * mu_k                       # x_t | k 的均值
    v_k = ab * s_k ** 2 + (1 - ab)                    # x_t | k 的方差
    for xv in (-3.0, -1.0, 0.0, 0.5, 2.5):
        xt = torch.tensor(xv, requires_grad=True)
        logp_k = torch.log(pi_k) - 0.5 * torch.log(2 * math.pi * v_k) - (xt - m_k) ** 2 / (2 * v_k)
        logq = torch.logsumexp(logp_k, dim=0)
        score, = torch.autograd.grad(logq, xt)                                     # 解析 score（自动微分）
        r_k = torch.softmax(logp_k, dim=0).detach()                                # 责任 q(k|x_t)
        cov_x0_xt = torch.sqrt(ab) * s_k ** 2                                       # Cov(x_0, x_t | k)
        e_x0_given_k = mu_k + cov_x0_xt / v_k * (xt.detach() - m_k)                # M2 条件均值
        e_x0 = (r_k * e_x0_given_k).sum()
        e_eps = (xt.detach() - torch.sqrt(ab) * e_x0) / torch.sqrt(1 - ab)         # E[ε|x_t]
        report(f"t={t:3d} x_t={xv:+.1f}  score", score, -e_eps / torch.sqrt(1 - ab), 1e-8)

print("\n总结：", "全部通过" if ok_all else "有未通过项，请检查上面标记")
