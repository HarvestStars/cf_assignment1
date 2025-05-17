import numpy as np
import matplotlib.pyplot as plt

from Digitial_fdm import heat_solver_implicit, heat_solver_cn, recover_C_from_phi

def run_fdm_full_params(method, K, T, r, sigma, S_range=(50, 150), N=100, M=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def plot_error_dual_axis(param_values, abs_errors, rel_errors, param_name):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左侧轴：绝对误差
    color1 = 'tab:blue'
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Absolute Error', color=color1, fontsize=12)
    ax1.plot(param_values, abs_errors, color=color1, marker='o', label='Absolute Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # 右侧轴：相对误差
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Relative Error', color=color2, fontsize=12)
    ax2.plot(param_values, rel_errors, color=color2, marker='s', linestyle='--', label='Relative Error')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 标题和布局
    fig.suptitle(f"MC vs CF Error w.r.t {param_name}", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"figs/binary/{param_name}_error_comparison.png", dpi=300)
    plt.show()

def sensitivity_analysis_with_error(
    param_name,
    param_values,
    fixed_params,
    S0=100,
    method_names=("implicit", "cn")
):
    # 存储每种方法下 S0 点对应的定价
    fdm_prices = {name: [] for name in method_names}
    index_S0 = None

    for val in param_values:
        # 覆盖某一个参数
        params = fixed_params.copy()
        params[param_name] = val

        for method in method_names:
            S_grid, _, C_fdm = run_fdm_full_params(
                method=method,
                K=params['K'],
                T=params['T'],
                r=params['r'],
                sigma=params['sigma'],
                S_range=(50, 150),
                N=100,
                M=100
            )
            # 提取 S0 位置上的价格（只取 t=0，即 C_fdm[-1, :]）
            if index_S0 is None:
                index_S0 = np.argmin(np.abs(S_grid - S0))

            fdm_prices[method].append(C_fdm[-1, index_S0])  # 最后一个时间层，对应 t=0

    # === 画图 ===
    plt.figure(figsize=(8, 6))
    for method in method_names:
        plt.plot(param_values, fdm_prices[method], marker='o', label=f"{method} FDM")

    plt.title(f"Sensitivity Analysis: effect of ${param_name}$ on option price at $S={S0}$", fontsize=14)
    plt.xlabel(f"{param_name}", fontsize=12)
    plt.ylabel("Option Price at $S_0$", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figs/binary/{param_name}_price_comparison.png", dpi=300)
    plt.show()

    # # 绘图2：误差对比
    # plot_error_dual_axis(param_values, abs_errors, rel_errors, param_name)

if __name__ == "__main__":
    # 设置 baseline 参数
    baseline = {
        'S0': 100,
        'K': 100,
        'B': 120,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
    }

    # option parameters
    # 分析执行价 K 的敏感性
    K_values = np.linspace(80, 120, 9)
    sensitivity_analysis_with_error('K', K_values, fixed_params=baseline)
    # 分析到期时间 T 的敏感性
    T_values = np.linspace(0.1, 2.0, 9)
    sensitivity_analysis_with_error('T', T_values, fixed_params=baseline)
    
    # model’s parameters
    # 分析波动率 sigma 的敏感性
    sigma_values = np.linspace(0.1, 0.5, 9)
    sensitivity_analysis_with_error('sigma', sigma_values, fixed_params=baseline)
    # 分析无风险利率 r 的敏感性
    r_values = np.linspace(0.01, 0.1, 9)
    sensitivity_analysis_with_error('r', r_values, fixed_params=baseline)

