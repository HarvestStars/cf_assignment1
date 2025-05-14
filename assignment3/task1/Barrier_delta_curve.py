import numpy as np
import matplotlib.pyplot as plt

from Barrier_fdm_from_slides import heat_solver_y_implicit, recover_V_from_y
from Barrier_montecarlo import closed_form_up_and_out_call

# Closed-form Delta by numerical central difference
def closed_form_delta(S_vals, K, B, T, r, sigma, h=1e-2):
    delta_cf = np.zeros_like(S_vals)
    for i, S in enumerate(S_vals):
        if S <= h or S + h >= B:
            delta_cf[i] = 0.0
        else:
            up = closed_form_up_and_out_call(S + h, K, B, T, r, sigma)
            down = closed_form_up_and_out_call(S - h, K, B, T, r, sigma)
            delta_cf[i] = (up - down) / (2 * h)
    return delta_cf

# === 主程序: 绘制 Delta 曲线 ===
if __name__ == "__main__":
    # 参数设定
    K = 100
    B = 120
    T = 1.0
    r = 0.05
    sigma = 0.25
    S_min, S_max = 50, 150
    N, M = 300, 200

    # 空间与时间变换
    x_min = np.log(S_min / K)
    x_max = np.log(S_max / K)
    x_grid = np.linspace(x_min, x_max, N + 1)
    dx = x_grid[1] - x_grid[0]
    S_grid = K * np.exp(x_grid)

    tau_max = 0.5 * sigma**2 * T
    tau_grid = np.linspace(0, tau_max, M + 1)
    dtau = tau_grid[1] - tau_grid[0]

    q = 2 * r / sigma**2
    a = 0.5 * (q - 1)
    b = 0.25 * (q + 1)**2

    x_barrier = np.log(B / K)

    # 初始条件
    y0 = np.maximum(S_grid / K - 1, 0) * np.exp(a * x_grid)
    y0[S_grid >= B] = 0.0

    # 数值求解
    y = heat_solver_y_implicit(y0, dx, dtau, M, x_grid, x_barrier)
    V = recover_V_from_y(y, x_grid, tau_grid, a, b, K)

    # 对应 t = 0 ⇒ tau = max ⇒ 取 V[-1, :]
    V_T = V[-1, :]

    # 数值计算 Delta（中间区域用中心差分）
    dS = S_grid[1] - S_grid[0]
    delta = np.zeros_like(V_T)
    delta[1:-1] = (V_T[2:] - V_T[:-2]) / (S_grid[2:] - S_grid[:-2])
    delta[0] = (V_T[1] - V_T[0]) / (S_grid[1] - S_grid[0])
    delta[-1] = (V_T[-1] - V_T[-2]) / (S_grid[-1] - S_grid[-2])

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, delta, label=r'$\Delta = \frac{\partial V}{\partial S}$')
    plt.axvline(B, color='red', linestyle='--', label='Barrier B')
    plt.xlabel('Asset Price $S$')
    plt.ylabel('Delta')
    plt.title('Delta Curve of Up-and-Out Barrier Call (via FDM)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 取中心的一段 S 值
    S_vals = S_grid
    delta_cf_vals = closed_form_delta(S_vals, K, B, T, r, sigma)

    plt.figure(figsize=(8, 5))
    plt.plot(S_vals, delta, label='FDM Delta', linewidth=2)
    plt.plot(S_vals, delta_cf_vals, 'g--', label='Closed-form Delta (centered diff)', linewidth=1.5)
    plt.axvline(B, color='red', linestyle='--', label='Barrier B')
    plt.xlabel('Asset Price $S$', fontsize=12)
    plt.ylabel('Delta', fontsize=12)
    plt.title('Delta Comparison: Implicit FDM vs Closed-form (Barrier Call)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/barrier/delta_comparison.png", dpi=300)
    plt.show()
