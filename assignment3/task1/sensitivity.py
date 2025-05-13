import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from BS_fdm import heat_solver_implicit, heat_solver_cn, recover_C_from_phi
from BS_montecarlo import binary_option_mc_price

def run_fdm_sigma(method, sigma, S_range=(50, 150), N=100, M=100):
    T = 1.0
    K = 100
    r = 0.05
    alpha = sigma**2 / 2
    alpha_c = -(r - 0.5 * sigma**2) / sigma**2
    beta_c = -0.5 * alpha_c**2 * sigma**2

    # 定义 log 价格网格
    S_min, S_max = S_range
    x_min = np.log(S_min / K)
    x_max = np.log(S_max / K)
    dx = (x_max - x_min) / N
    dt = T / M

    x_grid = np.linspace(x_min, x_max, N + 1)
    tau_grid = np.linspace(0, T - 1e-6, M + 1)
    S_grid = K * np.exp(x_grid)
    t_grid = T - tau_grid

    # 初始条件（binary payoff）
    psi0 = np.where(S_grid >= K, 1.0, 0.0)
    phi0 = psi0 * np.exp(-alpha_c * x_grid)

    if method == 'implicit':
        phi = heat_solver_implicit(phi0, dx, dt, M, alpha)
    elif method == 'cn':
        phi = heat_solver_cn(phi0, dx, dt, M, alpha)
    else:
        raise ValueError("Invalid method")

    C_fdm = recover_C_from_phi(phi, x_grid, tau_grid, alpha_c, beta_c, r)
    return S_grid, t_grid, C_fdm

# === Vega sensitivity 工作流 ===
def vega_analysis(method='cn', sigma_vals=[0.1, 0.2, 0.3], t_index=50):
    """
    比较不同 sigma 下的定价结果（固定某个 t 行）
    """
    C_slices = []
    labels = []

    for sigma in sigma_vals:
        S_grid, t_grid, C_fd = run_fdm_sigma(method, sigma)
        C_slice = C_fd[t_index, :]  # 取固定 t 行
        C_slices.append(C_slice)
        labels.append(f"σ = {sigma}")

    # 画图比较
    plt.figure(figsize=(8, 6))
    for C, lbl in zip(C_slices, labels):
        plt.plot(S_grid, C, label=lbl)
    plt.title(f"Digital Option Price vs $S$ at $t = {t_grid[t_index]:.2f}$\n({method} FDM)", fontsize=14)
    plt.xlabel("Stock Price $S$", fontsize=12)
    plt.ylabel("Option Price $C$", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figs/vega_analysis_{}.png'.format(method))
    plt.show()

# === Vega 数值估计图（多组 sigma 绘制）===
def vega_finite_diff_plot_multi(method='cn', sigma_vals=[0.1, 0.2, 0.5, 0.9], d_sigma=0.01, t_index=50):
    plt.figure(figsize=(8, 6))
    
    for sigma_base in sigma_vals:
        sigma1 = sigma_base
        sigma2 = sigma_base + d_sigma

        S_grid, t_grid, C1 = run_fdm_sigma(method, sigma1)
        _, _, C2 = run_fdm_sigma(method, sigma2)

        C1_slice = C1[t_index, :]
        C2_slice = C2[t_index, :]
        vega_est = (C2_slice - C1_slice) / d_sigma

        plt.plot(S_grid, vega_est, label=f"$\\sigma={sigma_base:.2f} \pm {d_sigma:.2f}$")

    # 图形美化
    plt.title(f"Numerical Vega Curves vs $S$ at $t = {t_grid[t_index]:.2f}$ ({method})", fontsize=14)
    plt.xlabel("Stock Price $S$", fontsize=12)
    plt.ylabel("Vega ≈ $\\partial C / \\partial \\sigma$", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/vega_finite_diff_multi_{method}.png')
    plt.show()


# === 主执行入口（按需调用） ===
if __name__ == "__main__":
    vega_analysis(method='implicit', sigma_vals=[0.1, 0.2, 0.5, 0.9])
    vega_finite_diff_plot_multi(method='implicit', sigma_vals=[0.1, 0.2, 0.5, 0.9], d_sigma=0.01)

    vega_analysis(method='cn', sigma_vals=[0.1, 0.2, 0.5, 0.9])
    vega_finite_diff_plot_multi(method='cn', sigma_vals=[0.1, 0.2, 0.5, 0.9], d_sigma=0.01)