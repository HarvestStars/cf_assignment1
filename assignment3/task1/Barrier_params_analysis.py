import numpy as np
import matplotlib.pyplot as plt

from Barrier_montecarlo import monte_carlo_up_and_out_call
from Barrier_montecarlo import closed_form_up_and_out_call

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
    plt.savefig(f"figs/barrier/{param_name}_error_comparison.png", dpi=300)
    plt.show()

def sensitivity_analysis_with_error(
    param_name,
    param_values,
    fixed_params,
    M=100, N=10000,
    beta_1=0.5826
):
    mc_prices = []
    cf_prices = []
    abs_errors = []
    rel_errors = []

    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val

        mc_price = monte_carlo_up_and_out_call(
            S0=params['S0'], K=params['K'], B=params['B'],
            T=params['T'], r=params['r'], sigma=params['sigma'],
            M=M, N=N
        )

        barrier_adjustment = np.exp(beta_1 * params['sigma'] * np.sqrt(params['T'] / M))
        cf_price = closed_form_up_and_out_call(
            S0=params['S0'], K=params['K'], B=params['B'],
            T=params['T'], r=params['r'], sigma=params['sigma'],
            barrier_adjustment=barrier_adjustment
        )

        mc_prices.append(mc_price)
        cf_prices.append(cf_price)
        abs_errors.append(abs(mc_price - cf_price))
        rel_errors.append(abs(mc_price - cf_price) / mc_price if mc_price != 0 else np.nan)

    # 绘图1：价格对比
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, mc_prices, label='Monte Carlo', marker='o')
    plt.plot(param_values, cf_prices, label='Adjusted Closed-Form', marker='s')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel("Option Price", fontsize=12)
    plt.title(f"Sensitivity of Option Price w.r.t {param_name}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figs/barrier/{param_name}_price_comparison.png", dpi=300)
    plt.show()

    # 绘图2：误差对比
    plot_error_dual_axis(param_values, abs_errors, rel_errors, param_name)

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
    # 分析障碍价 B 的敏感性
    B_values = np.linspace(100, 140, 9)
    sensitivity_analysis_with_error('B', B_values, fixed_params=baseline)
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


