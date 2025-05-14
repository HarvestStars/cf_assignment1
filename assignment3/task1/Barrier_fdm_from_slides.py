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

    S_mesh, t_mesh = np.meshgrid(np.exp(x_grid), T - (2 / sigma**2) * tau_grid)
    return S_mesh, t_mesh, V

def compute_fdm_surface_on_mesh(S_mesh, t_mesh, K, B, T, r, sigma):
    """
    Compute FDM surface on externally provided (S, t) mesh.
    Assumes uniform grid in x = log(S/K), and tau = 0.5 σ² (T - t).
    """
    x_grid = np.log(S_mesh[0] / K)           # horizontal line (shared x)
    tau_grid = 0.5 * sigma**2 * (T - t_mesh[:, 0])  # vertical line (shared tau)
    N = len(x_grid) - 1
    M = len(tau_grid) - 1
    dx = x_grid[1] - x_grid[0]
    dtau = tau_grid[1] - tau_grid[0]

    x_barrier = np.log(B / K)

    # a, b parameters from transformation
    q = 2 * r / sigma**2
    a = 0.5 * (q - 1)
    b = 0.25 * (q + 1)**2

    # Initial condition at tau=0
    S_vals = K * np.exp(x_grid)
    y0 = np.maximum(S_vals / K - 1, 0) * np.exp(a * x_grid)
    y0[S_vals >= B] = 0.0

    y = heat_solver_y_implicit(y0, dx, dtau, M, x_grid, x_barrier)
    V = recover_V_from_y(y, x_grid, tau_grid, a, b, K)

    return V  # same shape as S_mesh, t_mesh

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

def compute_closed_form_surface_on_mesh(S_mesh, t_mesh, K, B, T, r, sigma):
    """
    Compute closed-form surface directly on provided (S, t) mesh.
    """
    M, N = S_mesh.shape
    surface = np.zeros_like(S_mesh)

    for i in range(M):
        for j in range(N):
            S = S_mesh[i, j]
            t = t_mesh[i, j]
            tau = T - t
            if tau <= 1e-8 or S >= B:
                surface[i, j] = 0.0
            else:
                surface[i, j] = closed_form_up_and_out_call(S, K, B, tau, r, sigma)

    return surface

# === 通用绘图函数 ===
def plot_surface(S_mesh, t_mesh, Z, title, cmap):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, Z, cmap=cmap)
    ax.set_xlabel('Asset Price S', fontsize=12)
    ax.set_ylabel('Time t', fontsize=12)
    ax.set_zlabel('Option Price')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figs/barrier/{title}.png', dpi=300)
    plt.show()

# === 误差绘图函数 ===
def plot_error_surface(S_mesh, t_mesh, error_surface, title="Absolute Error Surface"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, error_surface, cmap='magma')
    ax.set_xlabel('Asset Price S', fontsize=12)
    ax.set_ylabel('Time t', fontsize=12)
    ax.set_zlabel('Absolute Error')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figs/barrier/{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def plot_error_heatmap(S_vals, t_vals, error_surface, title="Error Heatmap"):
    plt.figure(figsize=(8, 5))
    plt.imshow(
        error_surface,
        extent=[S_vals[0], S_vals[-1], t_vals[-1], t_vals[0]],
        origin='upper',
        cmap='hot',
        aspect='auto'
    )
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Asset Price $S$')
    plt.ylabel('Time $t$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'figs/barrier/{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def plot_price_surface(S_mesh, t_mesh, Z, title="Option Price Surface", cmap='viridis'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, Z, cmap=cmap)
    ax.set_xlabel("Asset Price $S$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Option Price $V$")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_error_surface(S_mesh, t_mesh, error_surface, title="Absolute Error Surface"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, error_surface, cmap='magma')
    ax.set_xlabel("Asset Price $S$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Absolute Error")
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
    N = 100
    M = 100

    # FDM surface
    S_mesh_fdm, t_mesh_fdm, Z_fdm = compute_fdm_surface(S_min, S_max, K, B, T, r, sigma)
    plot_surface(S_mesh_fdm, t_mesh_fdm, Z_fdm, "Implicit FDM Barrier Option Price Surface", "viridis")

    # Closed-form surface
    S_mesh_cf, t_mesh_cf, Z_cf = compute_closed_form_surface(S_min, S_max, K, B, T, r, sigma)
    plot_surface(S_mesh_cf, t_mesh_cf, Z_cf, "Closed-form Barrier Option Price Surface", "viridis") # plasma
    
    # # === 设置统一 mesh ===
    # S_vals = np.linspace(S_min, S_max, N + 1)
    # t_vals = np.linspace(0, T, M + 1)
    # S_mesh, t_mesh = np.meshgrid(S_vals, t_vals)

    # # === 调用两个定价函数 ===
    # Z_fdm = compute_fdm_surface_on_mesh(S_mesh, t_mesh, K, B, T, r, sigma)
    # Z_cf = compute_closed_form_surface_on_mesh(S_mesh, t_mesh, K, B, T, r, sigma)
    # error_surface = np.abs(Z_fdm - Z_cf)

    # # === 绘图展示 ===
    # plot_price_surface(S_mesh, t_mesh, Z_fdm, title="FDM Barrier Option Price Surface", cmap="viridis")
    # plot_price_surface(S_mesh, t_mesh, Z_cf, title="Closed-form Barrier Option Price Surface", cmap="plasma")
    # # plot_error_surface(S_mesh, t_mesh, error_surface, title="Absolute Error Between FDM and Closed-form")
    # # plot_error_heatmap(S_vals, t_vals, error_surface, "Error Heatmap")