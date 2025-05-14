import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def monte_carlo_up_and_out_call(S0, K, B, T, r, sigma, M, N):
    """
    Monte Carlo pricing of an up-and-out barrier call option.
    Parameters:
        M: number of time steps
        N: number of paths
    """
    dt = T / M
    discount_factor = np.exp(-r * T)

    # Simulate paths
    Z = np.random.randn(N, M)
    S = np.zeros((N, M + 1))
    S[:, 0] = S0

    for t in range(1, M + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    # Barrier condition: max S < B
    max_S = np.max(S, axis=1)
    payoff = np.maximum(S[:, -1] - K, 0) * (max_S < B)
    price = discount_factor * np.mean(payoff)

    return price


def closed_form_up_and_out_call(S0, K, B, tau, r, sigma, barrier_adjustment=1.0):
    """
    Step 4 formula implementation for closed-form up-and-out barrier call option
    with adjusted barrier level.
    """
    H = B * barrier_adjustment
    if S0 >= H or H <= K:
        return 0.0  # Already knocked out or invalid barrier

    mu = r - 0.5 * sigma ** 2
    lambd = 1 + 2 * r / sigma**2
    lambd2 = 1 - 2 * r / sigma**2

    def d_plus(z):
        return (np.log(z) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    def d_minus(z):
        return (np.log(z) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    term1 = S0 * norm.cdf(d_plus(S0 / K)) - S0 * norm.cdf(d_plus(S0 / H))
    term2 = (B / S0)**lambd * S0 * (norm.cdf(d_plus(B**2 / (K * S0))) - norm.cdf(d_plus(B / S0)))
    term3 = -K * np.exp(-r * tau) * (norm.cdf(d_minus(S0 / K)) - norm.cdf(d_minus(S0 / H)))
    term4 = K * np.exp(-r * tau) * (S0 / B)**lambd2 * (norm.cdf(d_minus(B**2 / (K * S0))) - norm.cdf(d_minus(B / S0)))

    price = (term1 - term2 + term3 + term4)
    return price

def closed_form_up_and_out_call_with_tau(St, K, B, T, t, r, sigma, barrier_adjustment=1.0):
    """
    Closed-form pricing of an up-and-out barrier call option with input time-to-maturity tau.
    """
    H = B * barrier_adjustment
    tau = T - t
    if St >= H or H <= K or tau <= 1e-8:
        return 0.0  # Knocked out or no time

    lambd = 1 + 2 * r / sigma**2
    lambd2 = 1 - 2 * r / sigma**2

    def d_plus(z):
        return (np.log(z) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    def d_minus(z):
        return (np.log(z) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    term1 = St * norm.cdf(d_plus(St / K)) - St * norm.cdf(d_plus(St / H))
    term2 = (B / St)**lambd * St * (norm.cdf(d_plus(B**2 / (K * St))) - norm.cdf(d_plus(B / St)))
    term3 = -K * np.exp(-r * tau) * (norm.cdf(d_minus(St / K)) - norm.cdf(d_minus(St / H)))
    term4 = K * np.exp(-r * tau) * (St / B)**lambd2 * (norm.cdf(d_minus(B**2 / (K * St))) - norm.cdf(d_minus(B / St)))

    return term1 - term2 + term3 + term4

if __name__ == "__main__":

    # Parameters (default setup, can be changed per run)
    S0 = 100     # Initial stock price
    K = 100      # Strike price
    B = 120      # Barrier level
    T = 1.0      # Time to maturity
    r = 0.05     # Risk-free rate
    sigma = 0.2  # Volatility

    # Monte Carlo and closed-form prices
    price_mc_example = monte_carlo_up_and_out_call(S0, K, B, T, r, sigma, M=100, N=100000)
    price_cf_example = closed_form_up_and_out_call(S0, K, B, T, r, sigma, barrier_adjustment=np.exp(0.5826 * sigma * np.sqrt(T / 100)))

    print(price_mc_example, price_cf_example)

