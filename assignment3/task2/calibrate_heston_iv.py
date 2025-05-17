import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from heston_semi_closed_form import heston_price_call
from heston_qe_pricing import implied_vol
import matplotlib.pyplot as plt

# Load raw market IV data
raw_data = np.load("../raw_data/raw_ivol_surfaces.npy", allow_pickle=True).item()
date = "2023 11 03"
market_vols = raw_data[date]["vols"]         # shape (15, N) N = 11
market_strikes = raw_data[date]["strikes"]   # shape (15, N)
market_tenors = raw_data[date]["tenors"]     # shape (N, ) 11个成熟期

data_rows, data_cols = market_strikes.shape
print("shape of market_strikes:", market_strikes.shape)
# print("market_strikes:", market_strikes)

# ==== Step 1: Guess market S0 from shortest maturity ====
eps = 1e-3
short_idx = np.argmin(market_tenors)
short_strikes = market_strikes[:, short_idx]
short_iv = market_vols[:, short_idx]
S0_market_guess = short_strikes[np.argmin(short_iv)]
print(f"[Guess] Market-implied S0 ≈ {S0_market_guess:.2f}")

# ==== Step 2: Compute log-moneyness grid ====
moneyness_matrix = np.log(market_strikes / S0_market_guess)  # shape (15, N)
print("shape of moneyness_matrix:", moneyness_matrix.shape)

moneyness_flat = moneyness_matrix.T.flatten()                # (N, 15) -> (N*15, )
# print("moneyness_flat:", moneyness_flat)
IV_flat = market_vols.T.flatten()
print("shape of IV_flat:", IV_flat.shape)
assert moneyness_matrix.shape == market_vols.shape

T_flat = np.repeat(market_tenors, 15)                        # 1,1,1,1,.... 2,2,2,2,.... 3,3,3,3,3.....  但是注意，如果不对strike或者moneyness进行T转置，那时间就这样 T_flat = np.tile(market_tenors, 15)
print("T_flat:", T_flat)                                     # (N*15, )
assert moneyness_flat.shape == T_flat.shape == IV_flat.shape

# Constants
S0_model = 100
r = 0.04

# === Objective: RMSE between market IVs and Heston-model IVs ===
def heston_objective(params):
    kappa, theta, sigma, rho, V0 = params
    model_ivs = []

    for m, T, iv_mkt in zip(moneyness_flat, T_flat, IV_flat):
        K_model = S0_model * np.exp(m)
        C = heston_price_call(S0_model, K_model, r, T, kappa, theta, sigma, rho, V0)
        iv_model = implied_vol(C, S0_model, K_model, r, T)
        model_ivs.append(iv_model)

    model_ivs = np.array(model_ivs)
    mask = ~np.isnan(model_ivs)
    error = model_ivs[mask] - IV_flat[mask]
    return np.sqrt(np.mean(error ** 2))

# === Initial guess and bounds ===
initial_guess = [2.5, 0.05, 0.4, -0.3, 0.04]
bounds = [
    (0.1, 10),     # kappa
    (0.01, 0.5),   # theta
    (0.05, 1.0),   # sigma
    (-0.99, 0.99), # rho
    (0.001, 0.2)   # V0
]

# === Run calibration ===
result = minimize(heston_objective, initial_guess, bounds=bounds, method='L-BFGS-B')
opt_params = result.x
print("Optimal Heston parameters:", opt_params)
print("Calibration RMSE:", result.fun)
