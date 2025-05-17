import numpy as np
from heston_semi_closed_form import heston_price_call
from heston_qe_pricing import implied_vol
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from calibrate_plot_tools import plot_surface_flat, plot_heatmap_from_matrix

# Load raw market IV data
raw_data = np.load("../raw_data/raw_ivol_surfaces.npy", allow_pickle=True).item()
date = "2023 11 01"
market_vols = raw_data[date]["vols"]         # shape (15, N) N = 11
market_strikes = raw_data[date]["strikes"]   # shape (15, N)
market_tenors = raw_data[date]["tenors"]     # shape (N, ) 11个成熟期

print("shape of market_strikes:", market_strikes.shape)
data_rows, data_cols = market_strikes.shape

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
print("moneyness_flat:", moneyness_flat)
IV_flat = market_vols.T.flatten()
print("shape of IV_flat:", IV_flat.shape)
assert moneyness_matrix.shape == market_vols.shape

T_flat = np.repeat(market_tenors, 15)                        # 1,1,1,1,.... 2,2,2,2,.... 3,3,3,3,3.....  但是注意，如果不对strike或者moneyness进行T转置，那时间就这样 T_flat = np.tile(market_tenors, 15)
print("T_flat:", T_flat)                                     # (N*15, )
assert moneyness_flat.shape == T_flat.shape == IV_flat.shape

# Surface generation
def generate_heston_surface_flat(S0, r, kappa, theta, sigma, rho, V0,
                            moneyness_flat, T_vals_flat):
    price_surface = []
    iv_surface = []

    for M, T in zip(moneyness_flat, T_vals_flat):
        K = S0 * np.exp(M)
        C = heston_price_call(S0, K, r, T, kappa, theta, sigma, rho, V0)
        price_surface.append(C)
        iv_surface.append(implied_vol(C, S0, K, r, T))

    return np.array(price_surface), np.array(iv_surface)

# === Optional parameters from estimation ===
opt_params = [ 2.09426036, 0.04388312, 1, -0.62524922, 0.02800214]
S0_model = 100
r = 0.04

# === start plotting ===
# === Generate fitted IV surface ===
best_kappa, best_theta, best_sigma, best_rho, best_V0 = opt_params
price_surface, iv_surface = generate_heston_surface_flat(
    S0=S0_model,
    r=r,
    kappa=best_kappa,
    theta=best_theta,
    sigma=best_sigma,
    rho=best_rho,
    V0=best_V0,
    moneyness_flat=moneyness_flat,
    T_vals_flat=T_flat,
)

# === Reshape the surfaces for plotting ===
T_grid_reshaped = T_flat.reshape(data_cols, data_rows).T
print("shape of T_grid_reshaped:", T_grid_reshaped.shape)

iv_surface_reshaped = iv_surface.reshape(data_cols, data_rows).T
print("shape of iv_surface_reshaped:", iv_surface_reshaped.shape)

# === Plotting the fitted IV surface ===
plot_surface_flat(
    K_grid=moneyness_matrix,
    T_grid=T_grid_reshaped,
    Z=iv_surface_reshaped,
    zlabel="Implied σ",
    title=f"Fitted Heston Implied Volatility Surface on {date}",
)

# plot_surface_flat(
#     K_grid=moneyness_matrix,
#     T_grid=T_grid_reshaped,
#     Z=market_vols,
#     zlabel="Implied σ",
#     title=f"Market Implied Volatility Surface on {date}",
# )

# 基于 flatten 数据的误差分析
mask = ~np.isnan(iv_surface)
iv_model = iv_surface[mask]
iv_market = IV_flat[mask]

rmse = np.sqrt(mean_squared_error(iv_market, iv_model))
mae = mean_absolute_error(iv_market, iv_model)
max_error = np.max(np.abs(iv_market - iv_model))

print(f"Diagnostics on Implied Volatility Fit:")
print(f"  RMSE       = {rmse:.6f}")
print(f"  MAE        = {mae:.6f}")
print(f"  Max Error  = {max_error:.6f}")

# 重塑 residual surface
residuals = (iv_surface - IV_flat).reshape(data_cols, data_rows).T

plot_surface_flat(
    K_grid=moneyness_matrix,
    T_grid=T_grid_reshaped,
    Z=residuals,
    zlabel="IV Residual (Model - Market)",
    title=f"Heston Model IV Residual Surface on {date}"
)

# 画出残差的热图
residuals = (iv_surface - IV_flat).reshape(data_cols, data_rows).T  # (15, N)
plot_heatmap_from_matrix(
    Z=residuals,
    moneyness_matrix=moneyness_matrix,
    market_tenors=market_tenors,
    title=f"IV Residual Heatmap (Model - Market) on {date}",
)