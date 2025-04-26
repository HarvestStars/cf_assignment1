import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

# step 1: heston模型的模拟函数
def simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler', return_z=False):
    """
    Simulate Heston paths using either Euler or Milstein scheme.
    
    Parameters:
    - S0: initial stock price
    - V0: initial variance
    - r: risk-free rate
    - kappa: mean reversion speed
    - theta: long-term variance
    - xi: volatility of volatility
    - rho: correlation between asset and variance
    - T: time to maturity
    - N: number of time steps
    - M: number of simulation paths
    - scheme: 'euler' or 'milstein'
    
    Returns:
    - S: stock paths (M x N+1)
    - V: variance paths (M x N+1)
    """
    dt = T / N
    S = np.zeros((M, N + 1))
    V = np.zeros((M, N + 1))
    Z1_list = []
    Z2_list = []

    S[:, 0] = S0
    V[:, 0] = V0

    for t in range(1, N + 1):
        Z1 = np.random.randn(M)
        Z2 = np.random.randn(M)
        ZV = Z1
        ZS = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        Z1_list.append(Z1)
        Z2_list.append(Z2)

        V_prev = V[:, t - 1]
        S_prev = S[:, t - 1]
        V_pos = np.maximum(V_prev, 0)

        # Update variance
        if scheme == 'euler':
            V_new = V_prev + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * np.sqrt(dt) * ZV
        elif scheme == 'milstein':
            V_new = V_prev + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * np.sqrt(dt) * ZV + 0.25 * xi**2 * dt * (ZV**2 - 1)

        else:
            raise ValueError("Invalid scheme. Use 'euler' or 'milstein'.")

        V_new = np.maximum(V_new, 0)
        V[:, t] = V_new

        # Update stock price
        if scheme == 'euler':
            S[:, t] = S_prev + r * S_prev * dt + np.sqrt(V_pos) * S_prev * np.sqrt(dt) * ZS
        elif scheme == 'milstein':
            S[:, t] = (S_prev + r * S_prev * dt +
                       np.sqrt(V_pos) * S_prev * np.sqrt(dt) * ZS +
                       0.5 * V_pos * S_prev * dt * (ZS**2 - 1))
    if return_z:
        return S, V, Z1_list, Z2_list
    else:
        return S, V

# step 2: calculate the average payoff
def arithmetic_asian_call_payoff(S_paths, K):
    """
    Compute the arithmetic-average Asian call payoff from simulated paths.

    Parameters:
    - S_paths: simulated stock price paths (M x N+1)
    - K: strike price

    Returns:
    - payoffs: numpy array of payoffs for each path
    """
    # Exclude initial price (S[:, 0]) to use only future average, or include it depending on contract
    arithmetic_avg = np.mean(S_paths[:, 1:], axis=1)
    payoffs = np.maximum(arithmetic_avg - K, 0)
    return payoffs

def geometric_asian_call_payoff(S_paths, K):
    """
    Compute the geometric-average Asian call payoff from simulated paths.

    Parameters:
    - S_paths: simulated stock price paths (M x N+1)
    - K: strike price

    Returns:
    - payoffs: numpy array of payoffs for each path
    """
    # 取 log 均值后再 exp（避免浮点数乘积下溢）
    log_S = np.log(S_paths[:, 1:])  # 不包括 t=0
    geo_mean_log = np.mean(log_S, axis=1)
    geo_avg = np.exp(geo_mean_log)
    payoffs = np.maximum(geo_avg - K, 0)
    return payoffs

# step 3: Monte Carlo estimator and its standard error
def monte_carlo_estimator(payoffs, r, T):
    """
    Compute the discounted Monte Carlo price estimator and its standard error.

    Parameters:
    - payoffs: array of simulated payoffs (M,)
    - r: risk-free interest rate
    - T: time to maturity

    Returns:
    - price: Monte Carlo price estimate
    - stderr: standard error of the estimate
    """
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(payoffs))
    variance = np.var(discounted_payoffs, ddof=1)  # 注意使用 ddof=1 是无偏估计
    return price, stderr, variance


if __name__ == "__main__":
    # Parameters for Heston model
    S0 = 100
    V0 = 0.04
    r = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    T = 1.0
    N = 10000
    M = 10000

    # mote carlo simulate S and V paths
    # Euler
    S_euler, V_euler = simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='euler')
    # Milstein
    S_milstein, V_milstein = simulate_heston_paths(S0, V0, r, kappa, theta, xi, rho, T, N, M, scheme='milstein')

    # Calculate the payoffs for both schemes
    K = 100  # strike price
    payoffs_euler = arithmetic_asian_call_payoff(S_euler, K)
    payoffs_milstein = arithmetic_asian_call_payoff(S_milstein, K)

    print("Payoffs for Euler scheme:")
    print(payoffs_euler[:5])  # Print first 5 payoffs for brevity

    print("Payoffs for Milstein scheme:")
    print(payoffs_milstein[:5])  # Print first 5 payoffs for brevity

    # Calculate the Monte Carlo price estimates and standard errors
    price_euler, stderr_euler, _ = monte_carlo_estimator(payoffs_euler, r, T)
    price_milstein, stderr_milstein, _ = monte_carlo_estimator(payoffs_milstein, r, T)

    print(f"Euler estimate: {price_euler:.4f} ± {stderr_euler:.4f}")
    print(f"Milstein estimate: {price_milstein:.4f} ± {stderr_milstein:.4f}")
