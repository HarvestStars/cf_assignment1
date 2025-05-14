import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Barrier_montecarlo import closed_form_up_and_out_call

def heat_solver_y_implicit(y0, dx, dtau, M, x_grid, x_barrier):
    """
    Implicit method to solve ∂y/∂τ = ∂²y/∂x² with Dirichlet barrier
    """
    N = len(y0) - 1
    lam = dtau / dx**2
    A = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        A[i, i - 1] = -lam
        A[i, i] = 1 + 2 * lam
        A[i, i + 1] = -lam

    A[0, 0] = A[N, N] = 1  # Dirichlet BC

    # Locate the barrier index
    i_barrier = np.searchsorted(x_grid, x_barrier)

    y = np.zeros((M + 1, N + 1))
    y[0, :] = y0.copy()

    for m in range(1, M + 1):
        b = y[m - 1].copy()
        b[0] = 0.0
        b[-1] = 0.0 # 最大边界敲出了，所以肯定也是0
        b[i_barrier:] = 0.0
        A[i_barrier:, :] = 0.0
        A[i_barrier:, i_barrier:] = np.eye(N + 1 - i_barrier)

        y[m] = np.linalg.solve(A, b)

    return y

def recover_V_from_y(y, x_grid, tau_grid, a, b, K):
    M, N = y.shape
    V = np.zeros_like(y)
    for m in range(M):
        tau = tau_grid[m]
        V[m, :] = K * y[m, :] * np.exp(-a * x_grid - b * tau)
    return V

# === FDM surface 构建 ===
def compute_fdm_surface(S_min, S_max, K, B, T, r, sigma, N=100, M=100):
    x_min = np.log(S_min / K)
    x_max = np.log(S_max / K)
    x_grid = np.linspace(x_min, x_max, N + 1)
    dx = x_grid[1] - x_grid[0]
    tau_grid = np.linspace(0, 0.5 * sigma**2 * T, M + 1)
    dtau = tau_grid[1] - tau_grid[0]
    x_barrier = np.log(B / K)

    # a, b parameters
    q = 2 * r / sigma**2
    a = 0.5 * (q - 1)
    b = 0.25 * (q + 1)**2

    # Initial condition y(x, 0)
    S = K * np.exp(x_grid)
    y0 = np.maximum(S / K - 1, 0) * np.exp(a * x_grid)
    y0[S >= B] = 0.0

    y = heat_solver_y_implicit(y0, dx, dtau, M, x_grid, x_barrier)
    V = recover_V_from_y(y, x_grid, tau_grid, a, b, K)

    S_mesh, t_mesh = np.meshgrid(np.exp(x_grid), T - tau_grid)
    return S_mesh, t_mesh, V

# === Closed-form surface 构建 ===
def compute_closed_form_surface(S_min, S_max, K, B, T, r, sigma, N=100, M=100):
    S_vals = np.linspace(S_min, S_max, N)
    t_vals = np.linspace(0.001, T, M)
    S_mesh, t_mesh = np.meshgrid(S_vals, t_vals)
    price_surface = np.zeros_like(S_mesh)

    for i in range(M):
        for j in range(N):
            S = S_mesh[i, j]
            tau = T - t_mesh[i, j]
            if tau <= 0 or S >= B:
                price_surface[i, j] = 0.0
            else:
                price_surface[i, j] = closed_form_up_and_out_call(S, K, B, tau, r, sigma)

    return S_mesh, t_mesh, price_surface

# === 通用绘图函数 ===
def plot_surface(S_mesh, t_mesh, Z, title, cmap):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, Z, cmap=cmap)
    ax.set_xlabel('Asset Price S')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Option Price')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# === 主程序 ===
if __name__ == "__main__":
    # 通用参数
    S0 = 100
    K = 100
    B = 120
    T = 1.0
    r = 0.05
    sigma = 0.25
    S_min, S_max = 50, 150

    # FDM surface
    S_mesh_fdm, t_mesh_fdm, Z_fdm = compute_fdm_surface(S_min, S_max, K, B, T, r, sigma)
    plot_surface(S_mesh_fdm, t_mesh_fdm, Z_fdm, "FDM Barrier Option Price Surface", "viridis")

    # Closed-form surface
    S_mesh_cf, t_mesh_cf, Z_cf = compute_closed_form_surface(S_min, S_max, K, B, T, r, sigma)
    plot_surface(S_mesh_cf, t_mesh_cf, Z_cf, "Closed-form Barrier Option Price Surface", "viridis") # plasma
