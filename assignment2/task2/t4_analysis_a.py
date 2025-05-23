import numpy as np
from scipy.stats import norm
import t1_1_sv_generator as sv_gen
import t2_analytical_validate as t2v

def run_simulation(S0, V0, r, kappa, theta, xi, rho, T, N, K, scheme, M):
    """单次仿真流程，包括 Heston Plain、Control Variate、GBM MC"""
    # 1. 模拟 Heston + 保存布朗增量
    S_heston, V_heston, Z1_list, Z2_list = sv_gen.simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme=scheme, return_z=True)

    # 2. 模拟 GBM（用相同布朗增量）
    dt = T / N
    S_gbm = np.zeros((M, N + 1))
    S_gbm[:, 0] = S0
    sigma_gbm = np.sqrt(V0)
    for t in range(1, N + 1):
        Z1 = Z1_list[t - 1]
        Z2 = Z2_list[t - 1]
        ZS = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
        S_gbm[:, t] = S_gbm[:, t - 1] * np.exp((r - 0.5 * sigma_gbm**2) * dt + sigma_gbm * np.sqrt(dt) * ZS)

    # 3. 计算payoff
    payoff_heston = sv_gen.arithmetic_asian_call_payoff(S_heston, K)
    payoff_gbm = sv_gen.geometric_asian_call_payoff(S_gbm, K)

    # 4. 封闭解
    price_gbm_closed = t2v.geometric_asian_call_bs(S0, K, r, sigma_gbm, T, N)

    # 5. plain MC
    plain_price, plain_stderr, plain_var = sv_gen.monte_carlo_estimator(payoff_heston, r, T)

    # 6. control variate MC
    control_variate_samples = payoff_heston + (price_gbm_closed - payoff_gbm)
    cv_price, cv_stderr, cv_var = sv_gen.monte_carlo_estimator(control_variate_samples, r, T)

    # 7. GBM 模型 MC（与 control variate 用的是同一个 payoff_gbm）
    gbm_mc_price, gbm_mc_stderr, gbm_mc_var = sv_gen.monte_carlo_estimator(payoff_gbm, r, T)

    return {
        'plain_price': plain_price,
        'plain_stderr': plain_stderr,
        'plain_var': plain_var,
        'cv_price': cv_price,
        'cv_stderr': cv_stderr,
        'cv_var': cv_var,
        'gbm_mc_price': gbm_mc_price,
        'gbm_mc_stderr': gbm_mc_stderr,
        'gbm_mc_var': gbm_mc_var,
    }


# --- 主程序 ---
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
    N = 1000
    K = 100
    scheme = 'euler'

    M_list = [1000, 5000, 10000, 50000]
    results = []

    for M in M_list:
        res = run_simulation(S0, V0, r, kappa, theta, xi, rho, T, N, K, scheme, M)
        results.append((M, res))

    # 打印表格标题（含方差和 GBM 参考）
    print(f"{'M':>7} | {'Plain Price':>12} | {'Plain Stderr':>12} | {'Plain Var':>12} | "
        f"{'CV Price':>12} | {'CV Stderr':>12} | {'CV Var':>12} | {'GBM MC':>10}")
    print('-' * 110)

    # 遍历输出每一行结果
    for M, res in results:
        print(f"{M:7d} | {res['plain_price']:12.4f} | {res['plain_stderr']:12.4f} | {res['plain_var']:12.6f} | "
            f"{res['cv_price']:12.4f} | {res['cv_stderr']:12.4f} | {res['cv_var']:12.6f} | "
            f"{res['gbm_mc_price']:10.4f}")


    import matplotlib.pyplot as plt

    # 提取数据用于绘图
    M_values = [M for M, res in results]
    plain_stderr_values = [res['plain_stderr'] for _, res in results]
    cv_stderr_values = [res['cv_stderr'] for _, res in results]

    # 创建折线图
    plt.figure(figsize=(8,6))
    plt.plot(M_values, plain_stderr_values, marker='o', label='Plain MC Standard Error')
    plt.plot(M_values, cv_stderr_values, marker='s', label='Control Variate Standard Error')

    plt.xscale('log')  # 横坐标用log尺度，更容易看清变化趋势
    plt.xlabel('Number of Paths (M)', fontsize=14)
    plt.ylabel('Standard Error', fontsize=14)
    plt.title('Standard Error vs Number of Simulation Paths', fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig('std_error_vs_paths.png', dpi=300)
    plt.show()
