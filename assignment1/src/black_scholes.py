from scipy.stats import norm
import numpy as np

def bs_call_price(S, K, r, sigma, T, t):
    """
    欧式看涨期权的 Black-Scholes 定价公式。
    
    参数:
        S (float or ndarray): 当前资产价格
        K (float): 执行价格
        r (float): 无风险利率
        sigma (float): 波动率
        T (float): 到期时间
        t (float): 当前时间

    返回:
        C (float or ndarray): 看涨期权价格
    """
    tau = T - t  # 剩余期限
    if tau <= 0:
        return np.maximum(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_call_delta(S, K, r, sigma, T, t):
    """
    欧式看涨期权的 Delta 值。

    参数同上。
    返回:
        Delta (float or ndarray)
    """
    tau = T - t
    if tau <= 0:
        return 0.0 if np.isscalar(S) else np.zeros_like(S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)
