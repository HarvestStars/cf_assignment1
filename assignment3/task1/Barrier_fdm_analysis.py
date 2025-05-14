import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Barrier_montecarlo import closed_form_up_and_out_call
from Barrier_fdm import heat_solver_implicit, recover_C_from_phi

# ==== 单点定价函数 ====
def FDM_price(S0, K, B, sigma):
    T = 1.0
    r = 0.05
    S_min, S_max = 20, 140
    N, M = 200, 200

    x_min = np.log(S_min)
    x_max = np.log(S_max)
    x_grid = np.linspace(x_min, x_max, N + 1)
    dx = x_grid[1] - x_grid[0]

    alpha = 0.5 - r / sigma**2
    beta = r*alpha -0.5 * sigma**2 * alpha + 0.5 * sigma**2 * alpha**2
    alpha_diff = sigma**2 / 2

    tau_max = 0.5 * sigma**2 * T
    tau_grid = np.linspace(0, tau_max, M + 1)
    dt = T / M
    x_barrier = np.log(B)

    def phi0_from_xgrid(x_grid, K, alpha, B=None):
        """
        Construct φ₀(x) = e^{-αx} * max(S - K, 0), with optional barrier cutoff at S ≥ B.
        Inputs:
            x_grid: log-price grid (x = log(S))
            K: strike
            alpha: exponential weight from transformation
            B: (optional) barrier level; if provided, φ₀(x) = 0 for S ≥ B
        Returns:
            φ₀(x_grid) array
        """
        S_grid = np.exp(x_grid)
        payoff = np.maximum(S_grid - K, 0)
        if B is not None:
            payoff[S_grid >= B] = 0.0
        phi0 = np.exp(-alpha * x_grid) * payoff
        return phi0
    
    phi_init = phi0_from_xgrid(x_grid, K, alpha, B)
    phi = heat_solver_implicit(phi_init, dx, dt, M, alpha_diff, x_grid, x_barrier)
    C = recover_C_from_phi(phi, x_grid, tau_grid, alpha, beta, r)

    # 对应 t = 0 ⇔ tau = max ⇒ 最后一行
    x0 = np.log(S0)
    C0 = np.interp(x0, x_grid, C[-1, :])
    return C0

# ==== 参数敏感性绘图 ====
def compare_and_plot(x_values, label, param_name, K_fixed=100, B_fixed=120, sigma_fixed=0.25, S0=100):
    cf_prices, fdm_prices, errors = [], [], []
    for x in x_values:
        if param_name == 'K':
            cf = closed_form_up_and_out_call(S0, x, B_fixed, 1.0, 0.05, sigma_fixed)
            fdm = FDM_price(S0, x, B_fixed, sigma_fixed)
        elif param_name == 'B':
            cf = closed_form_up_and_out_call(S0, K_fixed, x, 1.0, 0.05, sigma_fixed)
            fdm = FDM_price(S0, K_fixed, x, sigma_fixed)
        elif param_name == 'sigma':
            cf = closed_form_up_and_out_call(S0, K_fixed, B_fixed, 1.0, 0.05, x)
            fdm = FDM_price(S0, K_fixed, B_fixed, x)
        cf_prices.append(cf)
        fdm_prices.append(fdm)
        errors.append(abs(cf - fdm))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, cf_prices, 'b-', label='Closed-form')
    plt.plot(x_values, fdm_prices, 'r--', label='FDM (implicit)')
    plt.plot(x_values, errors, 'k-.', label='Absolute Error')
    plt.xlabel(label)
    plt.ylabel('Option Price')
    plt.title(f'Sensitivity of Barrier Option to {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== 执行入口 ====
if __name__ == "__main__":
    K_values = np.linspace(80, 120, 9)
    B_values = np.linspace(105, 140, 8)
    sigma_values = np.linspace(0.1, 0.5, 9)

    compare_and_plot(K_values, label='Strike Price K', param_name='K')
    compare_and_plot(B_values, label='Barrier Level B', param_name='B')
    compare_and_plot(sigma_values, label='Volatility σ', param_name='sigma')
