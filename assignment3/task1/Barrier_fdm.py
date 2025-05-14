import numpy as np
import matplotlib.pyplot as plt

# === 隐式差分法（Backward Euler） ===
def heat_solver_implicit(phi0, dx, dt, M, alpha_diff, x_grid, x_barrier):
    N = len(phi0) - 1
    lam = alpha_diff * dt / dx**2
    A = np.zeros((N + 1, N + 1))

    # 构造三对角矩阵
    for i in range(1, N):
        A[i, i - 1] = -lam
        A[i, i] = 1 + 2 * lam
        A[i, i + 1] = -lam
    A[0, 0] = A[N, N] = 1  # Dirichlet 边界

    i_barrier = np.searchsorted(x_grid, x_barrier)

    phi = np.zeros((M + 1, N + 1))
    phi[0, :] = phi0

    for m in range(1, M + 1):
        b = phi[m - 1].copy()
        b[0] = phi0[0]
        b[-1] = 0.0
        b[i_barrier:] = 0.0

        # 屏障点及以右零化矩阵
        A[i_barrier:, :] = 0.0
        A[i_barrier:, i_barrier:] = np.eye(N + 1 - i_barrier)

        phi[m] = np.linalg.solve(A, b)

    return phi


# === 还原到 C(t, S) 的原始价格空间 ===
def recover_C_from_phi(phi, x_grid, tau_grid, alpha, beta, r):
    M, N = phi.shape
    C = np.zeros_like(phi)
    for m in range(M):
        tau = tau_grid[m]
        C[m, :] = phi[m, :] * np.exp(alpha * x_grid + beta * tau - r * tau)
    return C

if __name__ == "__main__":
    # 参数
    S0 = 100
    K = 100
    B = 120
    T = 1.0
    r = 0.05
    sigma = 0.25

    S_min, S_max = 50, 150
    N = 200
    M = 200

    x_min, x_max = np.log(S_min), np.log(S_max)
    x_grid = np.linspace(x_min, x_max, N + 1)
    dx = (x_max - x_min) / N
    tau_grid = np.linspace(0, T, M + 1)
    dt = T / M
    x_barrier = np.log(B)

    alpha = 0.5 - r / sigma**2
    beta = r*alpha -0.5 * sigma**2 * alpha + 0.5 * sigma**2 * alpha**2
    alpha_diff = sigma**2 / 2

    # # 初始条件 φ₀(x)
    # def phi0(x_grid, K, B):
    #     S = np.exp(x_grid)
    #     payoff = np.maximum(S - K, 0)
    #     payoff[S >= B] = 0.0
    #     return np.exp(alpha * x_grid) * payoff # ???


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

    # 可视化定价平面
    from mpl_toolkits.mplot3d import Axes3D

    S_vals = np.exp(x_grid)
    t_vals = T - tau_grid
    S_mesh, t_mesh = np.meshgrid(S_vals, t_vals)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_mesh, t_mesh, C, cmap="viridis")
    ax.set_xlabel('Asset Price S')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Option Price')
    ax.set_title('FDM Barrier Option Price Surface')
    plt.tight_layout()
    plt.show()
