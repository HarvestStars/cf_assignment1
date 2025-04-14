import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def compute_step_pnl(C_t, C_tp1, delta_t, S_t, S_tp1, r, dt):
    """
    计算单个时间步的 Delta-Hedging PnL。

    参数:
        C_t: 当前期权价格
        C_tp1: 下一时刻期权价格
        delta_t: 当前时刻 Delta
        S_t: 当前股票价格
        S_tp1: 下一时刻股票价格
        r: 无风险利率
        dt: 时间步长

    返回:
        PnL_t: 单步对冲损益
    """
    dS = S_tp1 - S_t
    pnl = -(C_tp1 - C_t) + r * C_t * dt + delta_t * (dS - r * S_t * dt)
    return pnl

def simulate_hedging_pnl(
    paths, r, sigma_model, K, T, bs_price_fn, bs_delta_fn, hedge_every=1
):
    """
    模拟 Delta-Hedging 过程中的 PnL 路径。

    参数:
        paths: 股票价格路径 (n_paths, N+1)
        r: 无风险利率
        sigma_model: 用于定价和对冲的模型波动率
        K: 执行价
        T: 到期时间
        bs_price_fn: 期权定价函数
        bs_delta_fn: Delta 函数

    返回:
        pnl_paths: 对应每一条路径的 PnL 累积时间序列 (n_paths, N+1)
    """
    n_paths, n_steps = paths.shape
    dt = T / (n_steps - 1)
    time_grid = np.linspace(0, T, n_steps)

    pnl_paths = np.zeros((n_paths, n_steps))  # 初始为 0

    for i in range(n_steps - 1):
        t = time_grid[i]
        t_next = time_grid[i + 1]

        S_t = paths[:, i]
        S_tp1 = paths[:, i + 1]

        C_t = bs_price_fn(S_t, K, r, sigma_model, T, t)
        C_tp1 = bs_price_fn(S_tp1, K, r, sigma_model, T, t_next)
        
        # 只在每 hedge_every 步更新一次 delta
        if i % hedge_every == 0:
            delta_t = bs_delta_fn(S_t, K, r, sigma_model, T, t)

        pnl_step = compute_step_pnl(C_t, C_tp1, delta_t, S_t, S_tp1, r, dt)
        pnl_paths[:, i + 1] = pnl_paths[:, i] + pnl_step  # 累积 PnL

    return pnl_paths

def compare_pnl_across_vols(
    sigma_real_list, sigma_model, S0, K, T, r, N, n_paths
):
    all_pnls = []
    labels = []

    for sigma_real in sigma_real_list:
        paths = mcs.simulate_gbm_paths(S0, r, sigma_real, T, N, n_paths, seed=42)
        pnl_paths = simulate_hedging_pnl(
            paths, r, sigma_model, K, T, black_scholes.bs_call_price, black_scholes.bs_call_delta
        )
        final_pnl = pnl_paths[:, -1]
        all_pnls.append(final_pnl)
        labels.append(f"σ_real = {sigma_real:.2f}")

    plt.figure(figsize=(10, 5))
    plt.boxplot(all_pnls, labels=labels, showfliers=False)
    plt.title("Final PnL Distribution for Different Real Volatilities")
    plt.ylabel("Final PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for sigma_real, pnl in zip(sigma_real_list, all_pnls):
        print(f"σ_real = {sigma_real:.2f} → Mean: {pnl.mean():.4f}, Std: {pnl.std():.4f}")


def plot_single_pnl_path(pnl_paths, path_index=0, T=1.0):
    N = pnl_paths.shape[1] - 1
    time_grid = np.linspace(0, T, N + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(time_grid, pnl_paths[path_index])
    plt.title(f"PnL Path for Simulation #{path_index}")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_final_pnl_distribution(pnl_paths):
    final_pnl = pnl_paths[:, -1]

    plt.figure(figsize=(10, 5))
    plt.hist(final_pnl, bins=100, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Final PnL Across All Paths")
    plt.xlabel("Final PnL")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Mean PnL: {final_pnl.mean():.4f}")
    print(f"Std Dev:  {final_pnl.std():.4f}")

if __name__ == "__main__":
    import monte_carlo_simu as mcs
    import black_scholes as black_scholes

    # 期权参数
    S0 = 100
    K = 99
    T = 1.0
    r = 0.06
    N = 252
    n_paths = 10000

    # 真实波动率 & 模拟路径
    sigma_real = 0.2  # Task 2 可替换为 0.15, 0.2 等
    sigma_model = 0.2  # 用于定价和对冲的模型波动率
    paths = mcs.simulate_gbm_paths(S0, r, sigma_real, T, N, n_paths, seed=123)

    # 模拟 PnL
    pnl_paths = simulate_hedging_pnl(
        paths, r, sigma_model, K, T, black_scholes.bs_call_price, black_scholes.bs_call_delta
    )
    
    # 可视化部分你可以选择绘制：
    # - 某条路径的 PnL 曲线
    # plot_single_pnl_path(pnl_paths, path_index=0, T=T)
    # - 所有路径最终 PnL 的分布（直方图）

    sigma_real_list = [0.15, 0.20, 0.25, 0.30]
    compare_pnl_across_vols(
        sigma_real_list=sigma_real_list,
        sigma_model=0.20,
        S0=100, K=99, T=1.0, r=0.06,
        N=252, n_paths=10000
    )
