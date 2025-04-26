import numpy as np
from scipy.stats import norm
import t1_1_sv_generator as sv_gen

# analytic Asian call option pricing using closed-form formula
def geometric_asian_call_bs(S0, K, r, sigma, T, N):
    sigma_hat = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))
    r_hat = 0.5 * (r - 0.5 * sigma ** 2) + 0.5 * sigma_hat ** 2

    d1 = (np.log(S0 / K) + (r_hat + 0.5 * sigma_hat ** 2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)

    price = S0 * np.exp((r_hat - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

if __name__ == "__main__":

    # 参数设定
    S0 = 100
    V0 = 0.04           # which is actually σ^2 
    r = 0.05
    kappa = 0.0         # 可以设成0
    theta = V0
    xi = 0.0            # volatility of volatility 设成 0
    rho = 0.0           # 相关性设不设无所谓了
    T = 1.0
    N = 10000
    M = 10000         # 提高M减少误差
    K = 100

    # 模拟路径（其实就是GBM了）
    S_sim, V_sim = sv_gen.simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler'
    )

    # 计算 MC 估计算术均值亚洲期权
    payoffs = sv_gen.arithmetic_asian_call_payoff(S_sim, K)
    mc_price, mc_stderr = sv_gen.monte_carlo_estimator(payoffs, r, T)

    # 用几何平均重新计算 Monte Carlo payoff
    payoffs_geo = sv_gen.geometric_asian_call_payoff(S_sim, K)
    mc_price_geo, mc_stderr_geo = sv_gen.monte_carlo_estimator(payoffs_geo, r, T)

    # 几何均值亚洲期权封闭公式 baseline
    sigma = np.sqrt(V0)
    bs_price = geometric_asian_call_bs(S0, K, r, sigma, T, N)

    # 打印结果
    print(f"Monte Carlo (Arithmetic Average) Asian Call: {mc_price:.4f} ± {mc_stderr:.4f}")
    print(f"Monte Carlo (Geometric Average) Asian Call: {mc_price_geo:.4f} ± {mc_stderr_geo:.4f}")
    print(f"Closed-form (Geometric Average) Asian Call: {bs_price:.4f}")
