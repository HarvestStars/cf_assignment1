import numpy as np
import matplotlib.pyplot as plt

# === 引用已有方法 ===
from Digitial_fdm import heat_solver_cn, heat_solver_implicit, recover_C_from_phi

def run_fdm_for_delta(method='cn', sigma=0.2, N=100, M=100):
    T = 1.0
    K = 100
    r = 0.05
    alpha = sigma**2 / 2
    alpha_c = -(r - 0.5 * sigma**2) / sigma**2
    beta_c = -0.5 * alpha_c**2 * sigma**2

    S_min, S_max = 50, 150
    x_min, x_max = np.log(S_min / K), np.log(S_max / K)
    dx = (x_max - x_min) / N
    dt = T / M

    x_grid = np.linspace(x_min, x_max, N + 1)
    tau_grid = np.linspace(0, T - 1e-6, M + 1)
    S_grid = K * np.exp(x_grid)
    t_grid = T - tau_grid

    psi0 = np.where(S_grid >= K, 1.0, 0.0)
    phi0 = psi0 * np.exp(-alpha_c * x_grid)

    if method == 'cn':
        phi = heat_solver_cn(phi0, dx, dt, M, alpha)
    elif method == 'implicit':
        phi = heat_solver_implicit(phi0, dx, dt, M, alpha)
    else:
        raise ValueError("Invalid method")

    C = recover_C_from_phi(phi, x_grid, tau_grid, alpha_c, beta_c, r)
    return S_grid, t_grid, C

# === 差分方式计算 Delta（对 S 方向） ===
def compute_delta(C_surface, S_grid):
    delta = np.zeros_like(C_surface)
    for i in range(C_surface.shape[0]):
        delta[i, 1:-1] = (C_surface[i, 2:] - C_surface[i, :-2]) / (S_grid[2:] - S_grid[:-2])
        delta[i, 0] = (C_surface[i, 1] - C_surface[i, 0]) / (S_grid[1] - S_grid[0])
        delta[i, -1] = (C_surface[i, -1] - C_surface[i, -2]) / (S_grid[-1] - S_grid[-2])
    return delta

# === 多σ版本：绘制多个 σ 值下的 Delta vs S 曲线 ===
def plot_delta_curves_multi_sigma(method='cn', sigma_vals=[0.1, 0.2, 0.5, 0.9], t_index=50):
    plt.figure(figsize=(8, 6))

    for sigma in sigma_vals:
        S_grid, t_grid, C_surface = run_fdm_for_delta(method, sigma)
        delta_surface = compute_delta(C_surface, S_grid)
        delta_slice = delta_surface[t_index, :]
        t_value = t_grid[t_index]
        plt.plot(S_grid, delta_slice, label=f"$\\sigma$ = {sigma}")

    plt.title(f"Numerical Delta vs $S$ at $t = {t_value:.2f}$ ({method})", fontsize=14)
    plt.xlabel("Stock Price $S$", fontsize=12)
    plt.ylabel("Delta $\\approx \\partial C / \\partial S$", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/delta_curves_{method}.png", dpi=300)
    plt.show()

# === 执行入口 ===
if __name__ == "__main__":
    plot_delta_curves_multi_sigma(method='implicit', sigma_vals=[0.1, 0.2, 0.5, 0.9], t_index=50)
    plot_delta_curves_multi_sigma(method='cn', sigma_vals=[0.1, 0.2, 0.5, 0.9], t_index=50)
