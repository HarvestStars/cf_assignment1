import numpy as np
from numpy.random import default_rng
from tqdm import tqdm


# step 1: 定义Heston模型的参数和初始条件
# 设置随机数生成器
rng = default_rng(seed=42)

# 模型参数
S0 = 100.0       # 初始资产价格
v0 = 0.04        # 初始波动率
r = 0.05         # 无风险利率
kappa = 2.0      # 均值回复速度
theta = 0.04     # 长期波动率均值
xi = 0.3         # 波动率的波动率（vol of vol）
rho = -0.2       # 资产和波动率间的相关性

# 期权参数
K = 100          # 行权价
T = 1.0          # 到期时间（年）
N = 252          # 时间步数（按日计算）
M = 10000        # 模拟路径数

dt = T / N       # 每步时间长度

# step 2: heston模型的模拟函数
def simulate_heston_paths_explicit(method='euler'):
    # 初始化
    S_euler = np.zeros((N, M))
    V_euler = np.zeros((N, M))
    S_milstein = np.zeros((N, M))
    V_milstein = np.zeros((N, M))

    S_euler[0, :] = S0
    V_euler[0, :] = v0
    S_milstein[0, :] = S0
    V_milstein[0, :] = v0

    # 生成 Z_S 和 Z_V（独立正态变量）
    Z_S = rng.standard_normal((N - 1, M))
    Z_V = rng.standard_normal((N - 1, M))

    # 相关布朗运动（可复用）
    Z2 = Z_V
    Z1 = Z_S
    dt_sq = np.sqrt(dt)

    # 时间步迭代
    for t in tqdm(range(0, N - 1), desc="Simulating Heston paths"):
        # ----- Euler -----
        Vt_e = np.maximum(V_euler[t, :], 0)
        sqrt_V_e = np.sqrt(Vt_e)
        V_euler[t + 1, :] = V_euler[t, :] + kappa * (theta - Vt_e) * dt + xi * sqrt_V_e * dt_sq * Z_V[t, :]
        V_euler[t + 1, :] = np.abs(V_euler[t + 1, :])  # 确保非负
        S_euler[t + 1, :] = S_euler[t, :] + r * S_euler[t, :] * dt + sqrt_V_e * dt_sq * S_euler[t, :] * Z_S[t, :]

        # ----- Milstein -----
        Vt_m = np.maximum(V_milstein[t, :], 0)
        sqrt_V_m = np.sqrt(Vt_m)
        V_milstein[t + 1, :] = V_milstein[t, :] + kappa * (theta - Vt_m) * dt \
                             + xi * sqrt_V_m * dt_sq * Z_V[t, :] \
                             + 0.25 * xi**2 * dt * (Z_V[t, :]**2 - 1)
        V_milstein[t + 1, :] = np.abs(V_milstein[t + 1, :])
        S_milstein[t + 1, :] = S_milstein[t, :] + r * S_milstein[t, :] * dt \
                             + sqrt_V_m * dt_sq * S_milstein[t, :] * Z_S[t, :] \
                             + 0.5 * Vt_m * S_milstein[t, :] * dt * (Z_S[t, :]**2 - 1)

    return {
        "euler": {"S": S_euler, "V": V_euler},
        "milstein": {"S": S_milstein, "V": V_milstein},
        "Z_S": Z_S,
        "Z_V": Z_V
    }

# step 3: 亚洲期权定价函数
def asian_call_payoff(S_paths):
    """
    输入：S_paths 是 (N, M) 的矩阵，按列是路径，按行是时间
    返回：每条路径的 payoff 向量 (M,)
    """
    # 去掉初始价格，只取 t = 1 到 t = N 的价格，按列平均
    S_avg = np.mean(S_paths[1:, :], axis=0)
    payoff = np.maximum(S_avg - K, 0)
    return payoff

# step 4: 定价器函数
def monte_carlo_asian_call(method='euler'):
    result = simulate_heston_paths_explicit(method='both')  # 同时返回 euler 和 milstein

    if method == 'euler':
        S_paths = result["euler"]["S"]
    elif method == 'milstein':
        S_paths = result["milstein"]["S"]
    else:
        raise ValueError("method must be 'euler' or 'milstein'.")

    payoff = asian_call_payoff(S_paths)
    discounted = np.exp(-r * T) * payoff

    price = np.mean(discounted)
    stderr = np.std(discounted) / np.sqrt(M)

    return price, stderr

# step 5: 验证ξ=0
# 保存原始 xi
xi_original = xi

# 设置 ξ = 0，使 Heston 模型退化为 GBM
xi = 0.0
price_gbm_euler, err_euler = monte_carlo_asian_call('euler')
price_gbm_milstein, err_milstein = monte_carlo_asian_call('milstein')

print(f"[ξ=0 - Euler]    Price: {price_gbm_euler:.4f}, StdErr: {err_euler:.4f}")
print(f"[ξ=0 - Milstein] Price: {price_gbm_milstein:.4f}, StdErr: {err_milstein:.4f}")

# 恢复原始 xi
xi = xi_original
price_heston_euler, err_euler = monte_carlo_asian_call('euler')
price_heston_milstein, err_milstein = monte_carlo_asian_call('milstein')

print(f"[Heston - Euler]    Price: {price_heston_euler:.4f}, StdErr: {err_euler:.4f}")
print(f"[Heston - Milstein] Price: {price_heston_milstein:.4f}, StdErr: {err_milstein:.4f}")
