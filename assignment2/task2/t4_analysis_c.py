import numpy as np
import t1_1_sv_generator as sv_gen
import t2_analytical_validate as t2v

def run_simulation_optimal_c(S0, V0, r, kappa, theta, xi, rho, T, N, K, scheme, M):
    """单次仿真流程（带最优 c*）"""
    # 1. 模拟 Heston + 保存 Brownian increments
    S_heston, V_heston, Z1_list, Z2_list = sv_gen.simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme=scheme, return_z=True)

    # 2. 模拟 GBM（用相同布朗增量）
    dt = T / N
    S_gbm = np.zeros((M, N + 1))
    S_gbm[:, 0] = S0
    sigma_gbm = np.sqrt(V0)
    for t in range(1, N + 1):
        Z1 = Z1_list[t-1]
        Z2 = Z2_list[t-1]
        ZS = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        S_gbm[:, t] = S_gbm[:, t-1] * np.exp((r - 0.5 * sigma_gbm**2) * dt + sigma_gbm * np.sqrt(dt) * ZS)

    # 3. payoff计算
    payoff_heston = sv_gen.arithmetic_asian_call_payoff(S_heston, K)
    payoff_gbm = sv_gen.geometric_asian_call_payoff(S_gbm, K)

    # 4. 封闭解
    price_gbm_closed = t2v.geometric_asian_call_bs(S0, K, r, sigma_gbm, T, N)

    # 5. plain MC
    plain_price, plain_stderr, plain_var = sv_gen.monte_carlo_estimator(payoff_heston, r, T)

    # 6. control variate MC (c=1)
    control_variate_samples = payoff_heston + (price_gbm_closed - payoff_gbm)
    cv_price, cv_stderr, cv_var = sv_gen.monte_carlo_estimator(control_variate_samples, r, T)

    # 7. control variate MC (optimal c*)
    # --- 重点 ---
    cov_YX = np.cov(payoff_heston, payoff_gbm, ddof=1)[0,1]
    var_X = np.var(payoff_gbm, ddof=1)
    c_star = cov_YX / var_X # 计算最优 c* 解析式

    # 更新修正后的采样
    control_variate_opt_samples = payoff_heston + c_star * (price_gbm_closed - payoff_gbm)
    cv_opt_price, cv_opt_stderr, cv_opt_var = sv_gen.monte_carlo_estimator(control_variate_opt_samples, r, T)

    return {
        'plain_price': plain_price,
        'plain_stderr': plain_stderr,
        'plain_var': plain_var,
        'cv_price': cv_price,
        'cv_stderr': cv_stderr,
        'cv_var': cv_var,
        'cv_opt_price': cv_opt_price,
        'cv_opt_stderr': cv_opt_stderr,
        'cv_opt_var': cv_opt_var,
        'c_star': c_star
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 公共参数
    S0 = 100
    V0 = 0.04
    r = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 1.0
    rho = -0.9
    T = 1.0
    N = 1000
    K = 100
    scheme = 'euler'
    M = 10000

    # 运行一次仿真
    result = run_simulation_optimal_c(S0, V0, r, kappa, theta, xi, rho, T, N, K, scheme, M)

    # 打印结果
    print(f"Plain MC: {result['plain_price']:.4f} ± {result['plain_stderr']:.4f} (Var={result['plain_var']:.6f})")
    print(f"Control Variate (c=1): {result['cv_price']:.4f} ± {result['cv_stderr']:.4f} (Var={result['cv_var']:.6f})")
    print(f"Control Variate (optimal c*): {result['cv_opt_price']:.4f} ± {result['cv_opt_stderr']:.4f} (Var={result['cv_opt_var']:.6f})")
    print(f"Optimal c*: {result['c_star']:.4f}")

    # 绘制 variance 对比图
    methods = ['Plain MC', 'Control Variate (c=1)', 'Control Variate (optimal c*)']
    variances = [result['plain_var'], result['cv_var'], result['cv_opt_var']]

    plt.figure(figsize=(8,6))
    bars = plt.bar(methods, variances, color=['skyblue', 'lightgreen', 'lightcoral'])

    # 给每根柱子顶部加上数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, f'{yval:.6f}', ha='center', va='bottom', fontsize=9)

    plt.ylabel('Variance', fontsize=14)
    plt.title('Variance Comparison: \nPlain MC vs Control Variate (c=1) vs Control Variate (optimal c*)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("var_comparison_task4_100.png")
    plt.show()

    # 绘制 standard error 对比图
    methods = ['Plain MC', 'Control Variate (c=1)', 'Control Variate (optimal c*)']
    stderrs = [result['plain_stderr'], result['cv_stderr'], result['cv_opt_stderr']]

    plt.figure(figsize=(8,6))
    bars = plt.bar(methods, stderrs, color=['deepskyblue', 'lightseagreen', 'tomato'])

    # 给每根柱子顶部加上数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, f'{yval:.6f}', ha='center', va='bottom', fontsize=9)

    plt.ylabel('Standard Error', fontsize=14)
    plt.title('Standard Error Comparison: \nPlain MC vs Control Variate (c=1) vs Control Variate (optimal c*)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("std_err_comparison_task4_100.png")
    plt.show()

