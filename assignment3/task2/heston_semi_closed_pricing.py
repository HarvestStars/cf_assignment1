import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

from heston_semi_closed_form import heston_price_call

# Black-Scholes call price
def bs_call_price(S0, K, r, T, sigma):
    if sigma * np.sqrt(T) < 1e-8:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Invert BS to get implied volatility
def implied_vol(C_model, S0, K, r, T):
    try:
        return brentq(lambda sigma: bs_call_price(S0, K, r, T, sigma) - C_model, 1e-6, 3.0)
    except ValueError:
        print(f"Warning: failed to invert BS for C={C_model:.8f}, K={K:.2f}, T={T:.2f}")
        return np.nan

# Surface generation
def generate_heston_surface(S0, r, kappa, theta, sigma, rho, V0,
                            K_vals, T_vals):
    price_surface = np.zeros((len(T_vals), len(K_vals)))
    iv_surface = np.zeros_like(price_surface)

    for i, T in enumerate(T_vals):
        for j, K in enumerate(K_vals):
            C = heston_price_call(S0, K, r, T, kappa, theta, sigma, rho, V0)
            price_surface[i, j] = C
            iv_surface[i, j] = implied_vol(C, S0, K, r, T)
    return price_surface, iv_surface

# 3D plotting
def plot_surface(X, Y, Z, zlabel, title):
    from mpl_toolkits.mplot3d import Axes3D
    K_grid, T_grid = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K_grid, T_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    ax.invert_yaxis()
    
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    S0 = 100
    r = 0.04
    # theta = 0.04      # 长期波动率 ≈ 20%
    # kappa = 1.5       # 中等回归速度
    # sigma = 0.5       # 中高方差波动（增强 smile 弧度）
    # rho = -0.7        # 强负相关 → 放大左侧 OTM 波动率
    # V0 = 0.09         # 初始波动率 ≈ 30%（用于提升短期 T 的 IV）

    # theta = 0.03
    # kappa = 1.0       # 缓慢回归 → term structure 更平坦
    # sigma = 0.7       # 强 vol of vol，smile 更深
    # rho = -0.9        # 极端负相关，偏斜更加严重
    # V0 = 0.12         # 进一步提升前端（T ≈ 0.1）隐含波动率

    theta = 0.05
    kappa = 2.5
    sigma = 0.4
    rho = -0.3        # 较弱负相关 → 微笑趋于对称
    V0 = 0.04

    K_vals = np.linspace(80, 225, 15)
    T_vals = np.linspace(0.1, 2.0, 15)

    price_surf, iv_surf = generate_heston_surface(
        S0, r, kappa, theta, sigma, rho, V0, K_vals, T_vals
    )

    log_moneyness = np.log(K_vals / S0)

    plot_surface(K_vals, T_vals, price_surf, "Call Price", "Heston Call Price Surface")
    plot_surface(log_moneyness, T_vals, iv_surf, "Implied Volatility", "Implied Vol Surface (BS from Heston)")

