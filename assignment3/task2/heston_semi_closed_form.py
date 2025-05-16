import numpy as np
from scipy.integrate import simpson

# Heston characteristic function (stable version)
def heston_cf_stable(u, S0, r, T, kappa, theta, sigma, rho, V0):
    """
    Stable version of Heston characteristic function φ(θ; u, t)

    Parameters:
    u : complex or array_like
        Integration variable (complex-valued).
    S0 : float
        Initial asset price.
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-term variance level.
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance.
    V0 : float
        Initial variance.

    Returns:
    phi : complex or array_like
        Characteristic function value(s).
    """
    u = np.array(u, dtype=np.complex128)
    i = 1j
    x = np.log(S0)
    
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * i * u - b)**2 + sigma**2 * (i * u + u**2))
    g2 = (b - rho * sigma * i * u - d) / (b - rho * sigma * i * u + d)

    exp_dt = np.exp(-d * T)
    log_term = np.log((1 - g2 * exp_dt) / (1 - g2))
    
    C = r * i * u * T + (a / sigma**2) * ((b - rho * sigma * i * u - d) * T - 2 * log_term)
    D = ((b - rho * sigma * i * u - d) / sigma**2) * ((1 - exp_dt) / (1 - g2 * exp_dt))
    
    return np.exp(i * u * x + C + D * V0)

# Pricing function using Heston semi-closed formula
def heston_price_call(S0, K, r, T, kappa, theta, sigma, rho, V0, N=1000):
    """
    Compute European vanilla call option price under the Heston model using the semi-closed formula.
    Assume t = 0., S = S0, V = V0.

    Parameters:
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity tau actually, while because of S0, we assume t = 0, so tau = T.
    kappa, theta, sigma, rho, V0 : floats
        Heston model parameters.
    N : int
        Number of integration points.

    Returns:
    C0 : float
        Call option price.
    """
    from scipy.integrate import simpson

    def integrand_P1(u):
        phi_val = heston_cf_stable(u - 1j, S0, r, T, kappa, theta, sigma, rho, V0)
        return np.real(np.exp(-1j * u * np.log(K)) * phi_val / (1j * u * S0 * np.exp(r * T)))

    def integrand_P2(u):
        phi_val = heston_cf_stable(u, S0, r, T, kappa, theta, sigma, rho, V0)
        return np.real(np.exp(-1j * u * np.log(K)) * phi_val / (1j * u))

    u = np.linspace(1e-6, 100, N)
    P1 = 0.5 + (1 / np.pi) * simpson(integrand_P1(u), u)
    P2 = 0.5 + (1 / np.pi) * simpson(integrand_P2(u), u)

    return S0 * P1 - K * np.exp(-r * T) * P2
