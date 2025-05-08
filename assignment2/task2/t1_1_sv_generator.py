import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt

# step 0: 显式 GBM 模拟器
def simulate_gbm_paths(S0, sigma, r, T, N, M, Z_list=None, return_z=False):
    """
    Simulate GBM stock price paths with optional external Z_list.

    Parameters:
    - Z_list: optional list of Z_t (each of shape (M,)) for each time step

    Returns:
    - S: simulated stock price paths (M x N+1)
    """
    dt = T / N
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    generated_Z = []

    for t in range(1, N + 1):
        if Z_list is not None:
            Z = Z_list[t - 1]
        else:
            Z = np.random.randn(M)
        generated_Z.append(Z)

        S_prev = S[:, t - 1]
        S[:, t] = S_prev * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if return_z:
        return S, generated_Z
    else:
        return S

# step 1: heston模型的模拟函数
def simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M,
                           scheme='euler', Z1_list=None, Z2_list=None, return_z=False):
    """
    Simulate Heston paths with optional external Z1_list and Z2_list.

    If ξ=0 and ρ=0, then Z1 is unused and ZS = Z2.

    Returns:
    - S: stock paths (M x N+1)
    - V: variance paths (M x N+1)
    """
    dt = T / N
    S = np.zeros((M, N + 1))
    V = np.zeros((M, N + 1))

    generated_Z1 = []
    generated_Z2 = []

    S[:, 0] = S0
    V[:, 0] = V0

    for t in range(1, N + 1):
        Z1 = Z1_list[t - 1] if Z1_list is not None else np.random.randn(M)
        Z2 = Z2_list[t - 1] if Z2_list is not None else np.random.randn(M)

        ZV = Z1
        ZS = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        generated_Z1.append(Z1)
        generated_Z2.append(Z2)

        V_prev = V[:, t - 1]
        S_prev = S[:, t - 1]
        V_pos = np.maximum(V_prev, 0)

        # Update variance
        if scheme == 'euler':
            V_new = V_prev + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * np.sqrt(dt) * ZV
        elif scheme == 'milstein':
            V_new = V_prev + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * np.sqrt(dt) * ZV + 0.25 * xi**2 * dt * (ZV**2 - 1)
        else:
            raise ValueError("Invalid scheme. Use 'euler' or 'milstein'.")

        V[:, t] = np.maximum(V_new, 0)

        # Update stock price
        if scheme == 'euler':
            S[:, t] = S_prev + r * S_prev * dt + np.sqrt(V_pos) * S_prev * np.sqrt(dt) * ZS
        elif scheme == 'milstein':
            S[:, t] = (S_prev + r * S_prev * dt +
                       np.sqrt(V_pos) * S_prev * np.sqrt(dt) * ZS +
                       0.5 * V_pos * S_prev * dt * (ZS**2 - 1))

    if return_z:
        return S, V, generated_Z1, generated_Z2
    else:
        return S, V


# step 2: calculate the average payoff
def arithmetic_asian_call_payoff(S_paths, K):
    """
    Compute the arithmetic-average Asian call payoff from simulated paths.

    Parameters:
    - S_paths: simulated stock price paths (M x N+1)
    - K: strike price

    Returns:
    - payoffs: numpy array of payoffs for each path
    """
    # Exclude initial price (S[:, 0]) to use only future average, or include it depending on contract
    arithmetic_avg = np.mean(S_paths[:, 1:], axis=1)
    payoffs = np.maximum(arithmetic_avg - K, 0)
    return payoffs

def geometric_asian_call_payoff(S_paths, K):
    """
    Compute the geometric-average Asian call payoff from simulated paths.

    Parameters:
    - S_paths: simulated stock price paths (M x N+1)
    - K: strike price

    Returns:
    - payoffs: numpy array of payoffs for each path
    """
    # 取 log 均值后再 exp（避免浮点数乘积下溢）
    log_S = np.log(S_paths[:, 1:])  # 不包括 t=0
    geo_mean_log = np.mean(log_S, axis=1)
    geo_avg = np.exp(geo_mean_log)
    payoffs = np.maximum(geo_avg - K, 0)
    return payoffs

# step 3: Monte Carlo estimator and its standard error
def monte_carlo_estimator(payoffs, r, T):
    """
    Compute the discounted Monte Carlo price estimator and its standard error.

    Parameters:
    - payoffs: array of simulated payoffs (M,)
    - r: risk-free interest rate
    - T: time to maturity

    Returns:
    - price: Monte Carlo price estimate
    - stderr: standard error of the estimate
    """
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(payoffs))
    variance = np.var(discounted_payoffs, ddof=1)  # 注意使用 ddof=1 是无偏估计
    return price, stderr, variance



if __name__ == "__main__":
    # Parameters for Heston model
    S0 = 100
    V0 = 0.04
    r = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.1
    rho = -0.7
    T = 1.0
    N = 1000
    M = 1000
    K = 100

    num_trials = 30
    prices_euler = []
    prices_milstein = []
    stderrs_euler = []
    stderrs_milstein = []

    for _ in range(num_trials):
        Z1_list = [np.random.randn(M) for _ in range(N)]
        Z2_list = [np.random.randn(M) for _ in range(N)]

        # Euler Scheme
        S_euler, _ = simulate_heston_paths(
            S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler', Z1_list=Z1_list, Z2_list=Z2_list
        )
        payoffs_euler = arithmetic_asian_call_payoff(S_euler, K)
        price_euler, stderr_euler, _ = monte_carlo_estimator(payoffs_euler, r, T)
        prices_euler.append(price_euler)
        stderrs_euler.append(stderr_euler)

        # Milstein Scheme
        S_milstein, _ = simulate_heston_paths(
            S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='milstein', Z1_list=Z1_list, Z2_list=Z2_list
        )
        payoffs_milstein = arithmetic_asian_call_payoff(S_milstein, K)
        price_milstein, stderr_milstein, _ = monte_carlo_estimator(payoffs_milstein, r, T)
        prices_milstein.append(price_milstein)
        stderrs_milstein.append(stderr_milstein)

    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.plot(prices_euler, label='Euler Scheme', marker='o')
    plt.plot(prices_milstein, label='Milstein Scheme', marker='x')
    plt.xlabel('Trial')
    plt.ylabel('Estimated Price')
    plt.title('Heston-based Arithmetic Asian Option Estimates\nEuler vs Milstein Schemes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('heston_euler_vs_milstein.png', dpi=300)
    plt.show()

    # ===== 可视化：价格 + stderr =====
    print("Prices and Standard Errors:")
    print("===================================")
    print(f"{'Trial':<6} | {'Euler Price':<15} | {'Euler Std Err':<15} | {'Milstein Price':<15} | {'Milstein Std Err':<15}")
    print("-" * 80)
    for i in range(num_trials):
        print(f"{i+1:<6} | {prices_euler[i]:<15.4f} | {stderrs_euler[i]:<15.4f} | {prices_milstein[i]:<15.4f} | {stderrs_milstein[i]:<15.4f}")

    trials = np.arange(1, num_trials + 1)

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(trials, prices_euler, yerr=stderrs_euler, fmt='o', label='Euler',
    #             capsize=3, elinewidth=1, markeredgewidth=1)

    # #plt.errorbar(trials, prices_milstein, yerr=stderrs_milstein, fmt='s', label='Milstein',
    # #             capsize=3, elinewidth=1, markeredgewidth=1)
    # plt.xlabel('Trial')
    # plt.ylabel('Estimated Price')
    # plt.title('Monte Carlo Pricing of Arithmetic Asian Option\nHeston Model: Euler vs Milstein')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('euler_vs_milstein_with_errorbars.png', dpi=300)
    # plt.show()
    # 将数据转为 numpy 数组，便于偏移
    # 数据
    euler_stderr = np.array(stderrs_euler)
    milstein_stderr = np.array(stderrs_milstein)
    num_trials = len(euler_stderr)

    # 每个 trial 用两个柱子，所以我们把 trial 编号间隔放大
    bar_width = 0.35
    spacing = 1.0
    indices = np.arange(num_trials) * spacing * 2  # 加大 trial 间间隔
    offset = bar_width / 2

    plt.figure(figsize=(14, 6))

    # 画柱子
    plt.bar(indices - offset, euler_stderr, width=bar_width, color='blue', label='Euler StdErr')
    plt.bar(indices + offset, milstein_stderr, width=bar_width, color='orange', label='Milstein StdErr')

    # 设置横轴
    plt.xticks(indices, [str(i + 1) for i in range(num_trials)])
    plt.xlabel('Trial')
    plt.ylabel('Standard Error of Monte Carlo Estimate')
    plt.title('Standard Error per Trial: Euler vs Milstein')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('stderr_comparison_spaced_barplot.png', dpi=300)
    plt.show()

    # 打印均值和标准差对比
    mean_euler = np.mean(prices_euler)
    mean_milstein = np.mean(prices_milstein)
    std_euler = np.std(prices_euler, ddof=1)
    std_milstein = np.std(prices_milstein, ddof=1)

    # ===== 打印均值和平均标准误差 =====
    print(f"Euler:    mean={np.mean(prices_euler):.4f}, avg stderr={np.mean(stderrs_euler):.4f}")
    print(f"Milstein: mean={np.mean(prices_milstein):.4f}, avg stderr={np.mean(stderrs_milstein):.4f}")