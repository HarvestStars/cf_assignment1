import numpy as np

def simulate_gbm_paths(S0, r, sigma, T, N, n_paths, seed=None):
    """
    Simulate stock price paths using the Geometric Brownian Motion (GBM) model.

    Parameters:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (standard deviation, not variance)
        T (float): Total simulation time (in years)
        N (int): Number of time steps
        n_paths (int): Number of simulation paths
        seed (int, optional): Random seed for reproducibility

    Returns:
        paths (ndarray): Simulated path array of shape (n_paths, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N  # Time step size
    time_grid = np.linspace(0, T, N + 1)

    # Initialize path array
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    # Generate standard normal random variables
    Z = np.random.randn(n_paths, N)

    # Generate stock paths step-by-step
    for t in range(1, N + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]  # W_{t+dt} = W_t + Z * sqrt(dt), W_0 = 0
        )

    return paths

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 模拟参数
    S0 = 100
    r = 0.06
    sigma = 0.2
    T = 1.0
    N = 252
    n_paths = 1000  # 总模拟路径数
    n_show = 20     # 展示前 n 条路径

    # 生成路径
    paths = simulate_gbm_paths(S0, r, sigma, T, N, n_paths, seed=42)

    # 构造时间轴
    time_grid = np.linspace(0, T, N + 1)

    # 绘图
    plt.figure(figsize=(10, 6))
    for i in range(n_show):
        plt.plot(time_grid, paths[i], lw=1)

    plt.title(f"Simulated GBM Paths (σ={sigma})")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()