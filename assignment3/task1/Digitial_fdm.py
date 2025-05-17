import numpy as np
from scipy.stats import norm

def heat_solver_implicit(y0, dx, dt, M, alpha):
    """
    Implicit method (Backward Euler)
    """
    N = len(y0) - 1
    lam = alpha * dt / dx**2
    A = np.zeros((N + 1, N + 1))

    # Construct tridiagonal matrix
    for i in range(1, N):
        A[i, i-1] = -lam
        A[i, i] = 1 + 2*lam
        A[i, i+1] = -lam
    A[0, 0] = A[N, N] = 1

    y = np.zeros((M + 1, N + 1))
    y[0, :] = y0

    for m in range(1, M + 1):
        b = y[m-1].copy()
        b[0] = y0[0]
        b[-1] = y0[-1]
        y[m] = np.linalg.solve(A, b)

    return y

def heat_solver_cn(y0, dx, dt, M, alpha):
    """
    Crank-Nicolson method
    """
    N = len(y0) - 1
    lam = alpha * dt / dx**2

    A = np.zeros((N + 1, N + 1))
    B = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        A[i, i-1] = -lam / 2
        A[i, i] = 1 + lam
        A[i, i+1] = -lam / 2

        B[i, i-1] = lam / 2
        B[i, i] = 1 - lam
        B[i, i+1] = lam / 2

    A[0, 0] = B[0, 0] = A[N, N] = B[N, N] = 1

    y = np.zeros((M + 1, N + 1))
    y[0, :] = y0

    for m in range(1, M + 1):
        b = B @ y[m-1]
        b[0] = y0[0]
        b[-1] = y0[-1]
        y[m] = np.linalg.solve(A, b)

    return y

def recover_C_from_phi(phi, x_grid, tau_grid, alpha, beta, r):
    M, N = phi.shape
    C = np.zeros_like(phi)
    for m in range(M):
        tau = tau_grid[m]
        psi = phi[m, :] * np.exp(alpha * x_grid + beta * tau)
        f = psi * np.exp(-r * tau)
        C[m, :] = f
    return C

def binary_option_analytic_price(S, K, T, t, r, sigma):
    dt = T - t
    dt = np.maximum(dt, 1e-10)  # 避免 sqrt(0)
    d_minus = ((r - 0.5 * sigma ** 2) * dt + np.log(S / K)) / (sigma * np.sqrt(dt))
    return np.exp(-r * dt) * norm.cdf(d_minus)

# === 误差统计 ===
def compute_error_metrics(abs_err, label=""):
    mae = np.mean(abs_err)
    max_err = np.max(abs_err)
    mse = np.mean(abs_err**2)
    print(f"=== {label} Error Statistics ===")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Max Absolute Error       : {max_err:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def run_fd_solver():
        # === 参数设定 ===
        T = 1.0
        K = 100
        r = 0.05
        sigma = 0.2
        alpha = sigma**2 / 2
        alpha_c = -(r - 0.5 * sigma**2) / sigma**2
        beta_c = -0.5 * alpha_c**2 * sigma**2

        # === 更统一的 x, S, t 范围（与前面一致） ===
        S_min, S_max = 50, 150
        x_min = np.log(S_min / K)
        x_max = np.log(S_max / K)

        N = 100  # 空间步
        M = 100  # 时间步
        dx = (x_max - x_min) / N
        dt = T / M

        x_grid = np.linspace(x_min, x_max, N + 1)
        tau_grid = np.linspace(0, T - 1e-6, M + 1)
        S_grid = K * np.exp(x_grid)             # 回代股价坐标
        t_grid = T - tau_grid                   # 正向时间坐标

        # === 初始条件（以 log S 为变量） ===
        psi0 = np.where(S_grid >= K, 1.0, 0.0)
        phi0 = psi0 * np.exp(-alpha_c * x_grid)

        phi_implicit = heat_solver_implicit(phi0, dx, dt, M, alpha)
        phi_cn = heat_solver_cn(phi0, dx, dt, M, alpha)

        C_fd_implicit = recover_C_from_phi(phi_implicit, x_grid, tau_grid, alpha_c, beta_c, r)
        C_fd_cn = recover_C_from_phi(phi_cn, x_grid, tau_grid, alpha_c, beta_c, r)

        # === 构造解析解平面 ===
        C_exact = np.zeros_like(C_fd_implicit)
        for i, t in enumerate(t_grid):
            C_exact[i, :] = binary_option_analytic_price(S_grid, K, T, t, r, sigma)

        abs_error_implicit = np.abs(C_fd_implicit - C_exact)
        abs_error_cn = np.abs(C_fd_cn - C_exact)

        # ===== 数值误差统计 =====
        compute_error_metrics(abs_error_implicit, "Implicit")
        compute_error_metrics(abs_error_cn, "Crank-Nicolson")

        plt.figure(figsize=(8, 6))
        T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)
        cp = plt.contourf(S_mesh, T_mesh, (abs_error_implicit.T + 1e-8), levels=50, cmap='coolwarm')
        plt.colorbar(cp, label='|Implicit - Analytic|')
        plt.title('Implicit - Analytic Error Heatmap in $(S, t)$ domain', fontsize=14)
        plt.xlabel('Stock Price $S$', fontsize=12)
        plt.ylabel('Time $t$', fontsize=12)
        plt.savefig('figs/binary_option_Implicit_error_Heatmap.png', dpi=300)
        plt.show()

        plt.figure(figsize=(8, 6))
        T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)
        cp = plt.contourf(S_mesh, T_mesh, (abs_error_cn.T + 1e-8), levels=50, cmap='coolwarm')
        plt.colorbar(cp, label='|CN - Analytic|')
        plt.title('CN - Analytic Error Heatmap in $(S, t)$ domain', fontsize=14)
        plt.xlabel('Stock Price $S$', fontsize=12)
        plt.ylabel('Time $t$', fontsize=12)
        plt.savefig('figs/binary_option_CN_error_cn_Heatmap.png', dpi=300)
        plt.show()

        # 可视化
        T, K = 1.0, 100

        # FDM surface
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)
        ax1.plot_surface(S_mesh, T_mesh, C_fd_implicit.T, cmap='viridis', alpha=0.9)
        ax1.set_title('Implicit FDM Binary Option Price Surface', fontsize=14)
        ax1.set_xlabel('Stock Price $S$', fontsize=12)
        ax1.set_ylabel('Time $t$', fontsize=12)
        ax1.set_zlabel('Price')

        # Crank-Nicolson surface
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(S_mesh, T_mesh, C_fd_cn.T, cmap='plasma', alpha=0.9)
        ax2.set_title('Crank-Nicolson Binary Option Price Surface', fontsize=14)
        ax2.set_xlabel('Stock Price $S$', fontsize=12)
        ax2.set_ylabel('Time $t$', fontsize=12)
        ax2.set_zlabel('Price')

        plt.savefig('figs/binary_option_FDM_prices.png', dpi=300)
        plt.tight_layout()
        plt.show()

        # 3D map of error
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)
        ax1.plot_surface(S_mesh, T_mesh, abs_error_implicit.T, cmap='coolwarm')
        ax1.set_title('Absolute Error Surface |Implicit FDM - Analytic|')
        ax1.set_xlabel('Stock Price $S$', fontsize=12)
        ax1.set_ylabel('Time $t$', fontsize=12)
        ax1.set_zlabel('Absolute Error')

        ax2 = fig.add_subplot(122, projection='3d')
        T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)
        ax2.plot_surface(S_mesh, T_mesh, abs_error_cn.T, cmap='coolwarm')
        ax2.set_title('Absolute Error Surface |Crank-Nicolson - Analytic|')
        ax2.set_xlabel('Stock Price $S$', fontsize=12)
        ax2.set_ylabel('Time $t$', fontsize=12)
        ax2.set_zlabel('Absolute Error')

        plt.savefig('figs/binary_option_FDM_errors.png', dpi=300)
        plt.tight_layout()
        plt.show()

    # === 主程序 ===
    run_fd_solver()



