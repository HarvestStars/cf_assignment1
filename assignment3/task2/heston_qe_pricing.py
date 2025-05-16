"""
Monte‑Carlo option pricing & implied‑volatility surfaces for the Heston model
using Andersen’s Quadratic‑Exponential (QE) scheme.

Author: Alex Zhang
Date  : 2025‑05‑16

Requirements
------------
* numpy
* scipy
* matplotlib
* A module that exposes ``simulate_heston_qe_paths`` (see README or the
  companion file ``heston_qe_simulation.py``).

Usage (example)
---------------
$ python heston_qe_pricing.py             # runs demo & plots two 3‑D surfaces

or import the helpers in your own code:

>>> from heston_qe_pricing import price_surface_mc, iv_surface_mc
>>> price_grid = price_surface_mc(...)
>>> iv_grid    = iv_surface_mc(...)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
# Import the Monte‑Carlo simulator (edit the module/path to match your project)
# -----------------------------------------------------------------------------
from heston_qe_scheme import simulate_heston_qe_paths  # noqa: E402  (external)

# -----------------------------------------------------------------------------
# Black‑Scholes analytics & helpers
# -----------------------------------------------------------------------------

def bs_call_price(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """Black–Scholes European call price."""
    if sigma * np.sqrt(T) < 1e-8:  # practically zero vol
        return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(C_model, S0, K, r, T,
                left=1e-6, right=5.0,
                abs_tol=1e-8):
    """
    Robust BS implied σ  —— 自动处理零价格、内在价值边界、根区间无根等情况
    """
    # -- 内在 & 上界 --
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper     = S0                     # 理论上界

    # -- ① 全零价：MC 没抽到一次 ITM，直接 NaN --
    if C_model < abs_tol:
        print(f"Warning: C_model={C_model:.4f} < abs_tol={abs_tol:.4f}  →  NaN")
        return np.nan

    # -- ② clip 到合法区间 (intrinsic, upper) --
    C_adj = max(C_model, intrinsic + abs_tol)
    if C_adj >= upper:
        return np.nan

    # -- ③ 求根 --
    try:
        f = lambda s: bs_call_price(S0, K, r, T, s) - C_adj
        # 若端点同号，说明 σ∈[left,right] 内无解  →  NaN
        if f(left)*f(right) > 0:
            return np.nan
        return brentq(f, left, right, maxiter=200)
    except Exception:          # RuntimeError or ZeroDivision
        return np.nan
    
# -----------------------------------------------------------------------------
# 3‑D plotting helper
# -----------------------------------------------------------------------------

def plot_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *,
                 zlabel: str, title: str) -> None:
    """Simple 3‑D surface plot (strike × maturity → Z)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side‑effect import)

    K_grid, T_grid = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(K_grid, T_grid, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.invert_yaxis()

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Monte‑Carlo pricing helpers
# -----------------------------------------------------------------------------

def price_surface_mc(*,
                     S0: float,
                     V0: float,
                     r: float,
                     kappa: float,
                     theta: float,
                     sigma: float,
                     rho: float,
                     K_list: np.ndarray,
                     T_list: np.ndarray,
                     Npaths: int = 200_000,
                     Nsteps_per_year: int = 252,
                     seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate Heston option‑price surface via Monte‑Carlo.

    Parameters
    ----------
    S0, V0 : float
        Initial asset price and variance.
    r : float
        Risk‑free rate (continuous compounding).
    Heston params : kappa, theta, sigma, rho.
    K_list, T_list : 1‑d arrays
        Strikes and maturities for the grid (sorted ascending).
    Npaths : int, default 200k
        Number of Monte‑Carlo scenarios.
    Nsteps_per_year : int, default 252
        Time‑discretisation frequency.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    price_grid : ndarray  shape = (len(T_list), len(K_list))
    S_paths    : ndarray  raw simulated price paths (Npaths × (Nsteps+1)) —
                  returned in case caller wants Greeks / further post‑processing.
    """
    # ------------------------------------------------------------------
    # 1) Simulate once up to the *longest* maturity
    # ------------------------------------------------------------------
    T_max = float(np.max(T_list))
    Nsteps = int(np.ceil(Nsteps_per_year * T_max))

    S_paths, _ = simulate_heston_qe_paths(
        S0, V0,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        T=T_max, Nsteps=Nsteps, Npaths=Npaths,
        r=r,
        return_discounted=False,
        return_full=True,
        seed=seed,
    )

    time_grid = np.linspace(0.0, T_max, Nsteps + 1)

    # ------------------------------------------------------------------
    # 2) Vectorised pricing for each maturity & strike
    # ------------------------------------------------------------------
    K_arr = np.asarray(K_list, dtype=float)
    price_grid = np.empty((len(T_list), len(K_arr)))

    for i, T in enumerate(T_list):
        # locate the closest time index (assumes dt small ⇒ negligible bias)
        idx = np.searchsorted(time_grid, T)
        S_T = S_paths[:, idx]  # shape (Npaths,)

        # Payoff matrix:  Npaths × Nstrikes
        payoff = np.maximum(S_T[:, None] - K_arr[None, :], 0.0)
        price_grid[i] = np.exp(-r * T) * payoff.mean(axis=0)

    return price_grid, S_paths


def iv_surface_mc(price_grid: np.ndarray,
                  S0: float,
                  K_list: np.ndarray,
                  T_list: np.ndarray,
                  r: float) -> np.ndarray:
    """Invert Black‑Scholes to produce an implied‑volatility surface."""
    iv_grid = np.empty_like(price_grid)
    for i, T in enumerate(T_list):
        for j, K in enumerate(K_list):
            iv_grid[i, j] = implied_vol(price_grid[i, j], S0, K, r, T)
    return iv_grid


# -----------------------------------------------------------------------------
# Demo / CLI entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------
    # Grid & model configuration
    # ----------------------------

    S0, V0 = 100.0, 0.04
    r = 0.04

    heston_params = {
        "kappa": 2.5,
        "theta": 0.05,
        "sigma": 0.4,
        "rho": -0.3,
    }

    strikes    = np.linspace(80, 225, 15)        # 80,…,225 (15 points)
    maturities = np.linspace(0.1, 2.0, 15)

    # ----------------------------
    # Monte‑Carlo pricing
    # ----------------------------
    price_surf, _ = price_surface_mc(
        S0=S0,
        V0=V0,
        r=r,
        K_list=strikes,
        T_list=maturities,
        Npaths=25_000,
        seed=2025,
        **heston_params,
    )

    # Plot price surface
    plot_surface(
        X=strikes,
        Y=maturities,
        Z=price_surf,
        zlabel="Call price",
        title="Heston MC Call‑Price Surface (QE scheme)",
    )

    # ----------------------------
    # Implied vol inversion
    # ----------------------------
    iv_surf = iv_surface_mc(price_surf, S0, strikes, maturities, r)

    moneyness = np.log(strikes / S0)
    plot_surface(
        X=strikes,
        Y=maturities,
        Z=iv_surf,
        zlabel="Implied σ",
        title="Black–Scholes Implied‑Vol Surface (from Heston MC)",
    )
