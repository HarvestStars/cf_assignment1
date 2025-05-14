import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from scipy.stats import norm
from Barrier_montecarlo import monte_carlo_up_and_out_call, closed_form_up_and_out_call   

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === 假设这些函数在你的环境中已经定义 ===
# - monte_carlo_up_and_out_call(...)
# - corrected_closed_form_up_and_out_call(...)

def mc_convergence_analysis_avg(
    S0=100, K=100, B=120, T=1.0, r=0.05, sigma=0.2,
    M_values=None,
    N=100000,
    beta_1=0.5826,
    num_repeats=5  # 每个 M 重复次数
):
    if M_values is None:
        M_values = [10, 20, 40, 80, 160, 320, 640]

    avg_errors = []
    std_errors = []

    for M in M_values:
        H_adj = B * np.exp(beta_1 * sigma * np.sqrt(T / M))
        cf_price = closed_form_up_and_out_call(
            S0, K, B, T, r, sigma,
            barrier_adjustment=H_adj / B
        )

        errors = []

        for _ in range(num_repeats):
            mc_price = monte_carlo_up_and_out_call(S0, K, B, T, r, sigma, M, N)
            error = abs(mc_price - cf_price)
            errors.append(error)

        avg_error = np.mean(errors)
        std_error = np.std(errors)

        avg_errors.append(avg_error)
        std_errors.append(std_error)

        print(f"M={M:>4}, mean error={avg_error:.6e}, std={std_error:.2e}")

    # 拟合 log-log 收敛曲线
    log_M = np.log10(M_values)
    log_err = np.log10(avg_errors)
    slope, intercept, _, _, _ = linregress(log_M, log_err)
    fitted = intercept + slope * log_M

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.errorbar(log_M, log_err, yerr=std_errors, fmt='o-', label='Log Avg Error ± StdDev')
    plt.plot(log_M, fitted, '--', label=f'Fit Slope = {slope:.2f}', color='orange')
    plt.xlabel("log10(M)  (number of time steps)", fontsize=12)
    plt.ylabel("log10(Absolute Error)", fontsize=12)
    plt.title("MC Convergence with Averaged Error (log-log)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/barrier/mc_convergence_avg.png", dpi=300)
    plt.show()

    return {
        "M_values": M_values,
        "avg_errors": avg_errors,
        "std_errors": std_errors,
        "log_slope": slope
    }

# 示例调用
if __name__ == "__main__":
    mc_convergence_analysis_avg()
