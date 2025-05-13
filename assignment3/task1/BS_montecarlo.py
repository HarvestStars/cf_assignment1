import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ===== 1. Monte Carlo 定价函数 =====
def binary_option_mc_price(S, K, T, t, r, sigma, n_sim=100_000, seed=42):
    np.random.seed(seed)
    dt = T - t
    Z = np.random.randn(n_sim)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoffs = (ST >= K).astype(float)
    discounted_payoff = np.exp(-r * dt) * np.mean(payoffs)
    return discounted_payoff

# ===== 2. 解析式定价函数 =====
def binary_option_analytic_price(S, K, T, t, r, sigma):
    dt = T - t
    d_minus = ((r - 0.5 * sigma ** 2) * dt + np.log(S / K)) / (sigma * np.sqrt(dt))
    price = np.exp(-r * dt) * norm.cdf(d_minus)
    return price

if __name__ == "__main__":
    # ===== 3. 主脚本：计算平面 =====
    S_vals = np.linspace(50, 150, 50)       # 不同股价 S_t
    t_vals = np.linspace(0.001, 0.99, 50)   # 不同 t（接近 T）

    T = 1.0     # 固定到期时间
    K = 100     # 行权价
    r = 0.05    # 无风险利率
    sigma = 0.2 # 波动率

    S_grid, t_grid = np.meshgrid(S_vals, t_vals)

    mc_prices = np.zeros_like(S_grid)
    analytic_prices = np.zeros_like(S_grid)

    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            S_ = S_grid[i, j]
            t_ = t_grid[i, j]
            mc_prices[i, j] = binary_option_mc_price(S_, K, T, t_, r, sigma)
            analytic_prices[i, j] = binary_option_analytic_price(S_, K, T, t_, r, sigma)

    # ===== 4. 可视化 =====
    fig = plt.figure(figsize=(14, 6))

    # Monte Carlo surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(S_grid, t_grid, mc_prices, cmap='viridis', alpha=0.9)
    ax1.set_title('Monte Carlo Binary Option Price', fontsize=14)
    ax1.set_xlabel('Stock Price $S_t$', fontsize=12)
    ax1.set_ylabel('Time $t$', fontsize=12)
    ax1.set_zlabel('Option Price', fontsize=12)

    # Analytic surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(S_grid, t_grid, analytic_prices, cmap='plasma', alpha=0.9)
    ax2.set_title('Analytic Binary Option Price', fontsize=14)
    ax2.set_xlabel('Stock Price $S_t$', fontsize=12)
    ax2.set_ylabel('Time $t$', fontsize=12)
    ax2.set_zlabel('Option Price', fontsize=12)

    plt.tight_layout()
    plt.savefig('figs/binary_option_prices.png', dpi=300)
    plt.show()
    
    # ===== 5. 数值误差统计 =====
    abs_error = np.abs(mc_prices - analytic_prices)
    rel_error = abs_error / np.maximum(analytic_prices, 1e-8)  # 避免除0

    mae = np.mean(abs_error)
    mse = np.mean((mc_prices - analytic_prices)**2)
    max_error = np.max(abs_error)
    mean_rel_error = np.mean(rel_error)

    print("==== Error Statistics ====")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Relative Error: {mean_rel_error * 100:.4f}%")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # ===== 6. 误差平面可视化（absolute error） =====
    fig_err = plt.figure(figsize=(7, 6))
    ax_err = fig_err.add_subplot(111, projection='3d')
    ax_err.plot_surface(S_grid, t_grid, abs_error, cmap='coolwarm', alpha=0.9)
    ax_err.set_title('Absolute Error Surface (|MC - Analytic|)', fontsize=14)
    ax_err.set_xlabel('Stock Price $S_t$', fontsize=12)
    ax_err.set_ylabel('Time $t$', fontsize=12)
    ax_err.set_zlabel('Absolute Error', fontsize=12)
    plt.savefig('figs/abs_error_surface.png', dpi=300)
    plt.tight_layout()
    plt.show()

    # =====（可选）热力图 2D版本 =====
    plt.figure(figsize=(8, 6))
    plt.contourf(S_grid, t_grid, abs_error, levels=50, cmap='coolwarm')
    plt.colorbar(label='|MC - Analytic|')
    plt.title('Absolute Error Heatmap', fontsize=14)
    plt.xlabel('Stock Price $S_t$', fontsize=12)
    plt.ylabel('Time $t$', fontsize=12)
    plt.savefig('figs/abs_error_heatmap.png', dpi=300)
    plt.show()

