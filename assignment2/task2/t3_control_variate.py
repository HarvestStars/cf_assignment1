import numpy as np
from scipy.stats import norm
import t1_1_sv_generator as sv_gen
import t2_validate as t2v

if __name__ == "__main__":
    # 参数设定
    S0 = 100
    V0 = 0.04
    r = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    T = 1.0
    N = 10000
    M = 10000
    K = 100
    scheme = 'euler'  # or 'milstein'

    # Step 1: 同步仿真 Heston
    S_heston, V_heston, Z1_list, Z2_list = sv_gen.simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme=scheme, return_z=True)

    # Step 1: 同步仿真 GBM（使用同样的 Z1, Z2）
    dt = T / N
    S_gbm = np.zeros((M, N + 1))
    S_gbm[:, 0] = S0
    sigma_gbm = np.sqrt(V0)

    for t in range(1, N + 1):
        Z1 = Z1_list[t-1]
        Z2 = Z2_list[t-1]
        Z = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # 与Heston一样的ZS

        S_gbm[:, t] = S_gbm[:, t-1] * np.exp((r - 0.5 * sigma_gbm**2) * dt + sigma_gbm * np.sqrt(dt) * Z)

    # Step 2: 计算 Heston下的算术亚洲payoff 和 GBM下的几何亚洲payoff
    payoff_heston = sv_gen.arithmetic_asian_call_payoff(S_heston, K)
    payoff_gbm = sv_gen.geometric_asian_call_payoff(S_gbm, K)

    # Step 3: 计算封闭解 BS模型的几何亚洲期权价格
    price_gbm_closed = t2v.geometric_asian_call_bs(S0, K, r, sigma_gbm, T, N)

    # Step 3: 估计 plain MC 和 control variate MC
    plain_price, plain_stderr, plain_var = sv_gen.monte_carlo_estimator(payoff_heston, r, T)

    # Control variate 修正 Heston payoff
    control_variate_samples = payoff_heston + (price_gbm_closed - payoff_gbm)
    cv_price, cv_stderr, cv_var = sv_gen.monte_carlo_estimator(control_variate_samples, r, T)

    # Step 4: 打印结果比较
    print(f"GBM Closed-form Geometric Asian Call Price: {price_gbm_closed:.4f}")
    print()
    print(f"Heston Plain Monte Carlo Price: {plain_price:.4f} ± {plain_stderr:.4f}")
    print(f"Plain Monte Carlo Variance: {plain_var:.6f}")
    print()
    print(f"Heston Control Variate Price: {cv_price:.4f} ± {cv_stderr:.4f}")
    print(f"Control Variate Variance: {cv_var:.6f}")
    