import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import t1_1_sv_generator as sv_gen

# 验证xi=0时, heston模型和GBM模型一致
# 参数设定
S0 = 100
V0 = 0.04           # which is actually σ^2 
r = 0.05
kappa = 0.0         # 可以设成0
theta = V0
xi = 0.0            # volatility of volatility 设成 0
rho = 0.0           # 相关性设不设无所谓了
T = 1.0
N = 500
M = 500
K = 100

# 实验设置
num_trials = 50
heston_prices = []
gbm_prices = []

for trial in range(num_trials):
    # 生成固定的 Z2 路径
    Z2_list = [np.random.randn(M) for _ in range(N)]

    # Heston ξ=0 情形
    S_heston, V_heston = sv_gen.simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T, N, M,
        scheme='euler', Z2_list=Z2_list
    )
    payoffs = sv_gen.arithmetic_asian_call_payoff(S_heston, K)
    mc_price, _, _ = sv_gen.monte_carlo_estimator(payoffs, r, T)
    heston_prices.append(mc_price)

    # GBM 使用相同的 Z2
    sigma = np.sqrt(V0)
    S_gbm = sv_gen.simulate_gbm_paths(
        S0, sigma, r, T, N, M, Z_list=Z2_list
    )

    gbm_payoffs = sv_gen.arithmetic_asian_call_payoff(S_gbm, K)
    gbm_price, _, _ = sv_gen.monte_carlo_estimator(gbm_payoffs, r, T)
    gbm_prices.append(gbm_price)

# 绘图比较
plt.figure(figsize=(10, 6))
plt.plot(heston_prices, label='Heston (ξ=0)', marker='o')
plt.plot(gbm_prices, label='GBM', marker='x')
plt.xlabel('Trial')
plt.ylabel('Estimated Price')
plt.title('Arithmetic Asian Option Price Estimates: Heston (ξ=0) vs GBM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('heston_vs_gbm_asian_arithmetic.png', dpi=300)
plt.show()
