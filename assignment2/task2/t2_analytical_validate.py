import numpy as np
from scipy.stats import norm
import t1_1_sv_generator as sv_gen
import matplotlib.pyplot as plt

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
    N = 1000
    M = 1000         # 提高M减少误差
    K = 100

    # # 模拟路径（其实就是GBM了）
    # S_sim, V_sim = sv_gen.simulate_heston_paths(
    #     S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler'
    # )

    # # 用几何平均重新计算 Monte Carlo payoff
    # payoffs_geo = sv_gen.geometric_asian_call_payoff(S_sim, K)
    # mc_price_geo, mc_stderr_geo, _ = sv_gen.monte_carlo_estimator(payoffs_geo, r, T)

    # # 几何均值亚洲期权封闭公式 baseline
    # sigma = np.sqrt(V0)
    # bs_price = geometric_asian_call_bs(S0, K, r, sigma, T, N)

    # # 打印结果
    # print(f"Monte Carlo (Geometric Average) Asian Call: {mc_price_geo:.4f} ± {mc_stderr_geo:.4f}")
    # print(f"Closed-form (Geometric Average) Asian Call: {bs_price:.4f}")


    sigma = np.sqrt(V0)
    true_price = geometric_asian_call_bs(S0, K, r, sigma, T, N)

    num_trials = 100
    mc_prices = []
    mc_errors = []

    for _ in range(num_trials):
        # 模拟路径（Heston ξ=0 即为 GBM）
        S_sim, _ = sv_gen.simulate_heston_paths(
            S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler'
        )

        # Monte Carlo 几何平均
        payoffs_geo = sv_gen.geometric_asian_call_payoff(S_sim, K)
        mc_price_geo, _, _ = sv_gen.monte_carlo_estimator(payoffs_geo, r, T)

        mc_prices.append(mc_price_geo)
        mc_errors.append(mc_price_geo - true_price)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(mc_prices, label='Monte Carlo Estimate', marker='o')
    plt.axhline(true_price, color='r', linestyle='--', label='Closed-form Value')
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('Geometric Asian Call Price', fontsize=14)
    plt.title('Monte Carlo vs Closed-form: Geometric Asian Call (BS Model)', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('geometric_asian_comparison.png', dpi=300)
    plt.show()

    print(f"Closed-form Geometric Asian Call Price: {true_price:.4f}")
    print(f"Average Monte Carlo Price: {np.mean(mc_prices):.4f} ± {np.std(mc_prices):.4f}")
    print(f"Average Error: {np.mean(mc_errors):.4f} ± {np.std(mc_errors):.4f}")
    errors = np.array(mc_prices) - true_price
    plt.hist(errors, bins=20, color='gray', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Distribution of MC Estimation Error', fontsize=16)
    plt.xlabel('Error = MC Estimate - Closed-form', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300)
    plt.show()
