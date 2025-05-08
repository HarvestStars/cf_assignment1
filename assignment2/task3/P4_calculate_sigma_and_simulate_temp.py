import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# ===== 参数设置 =====
omega = 2 * np.pi / 365.25
I = 2  # 傅里叶阶数
kappa = 0.207  # 由AR(1)得到

# =====step 1: 构造 u_hat(t) =====
daily_df = pd.read_csv('amsterdam_daily_temperature.csv', parse_dates=['date'], index_col='date')
t0 = daily_df.index[0].toordinal()
t = np.array([d.toordinal() - t0 for d in daily_df.index])
y = daily_df['temperature_2m_mean'].values

# 由 P3_deterministic_temp_and_residual_AR.py 中的结果
a = 10.664   # 常数项
b = 0.00152  # 线性趋势项
C = 6.963    # 振幅项
phi = -0.949 # 相位项（弧度）

def u_hat(t):  # 构造解析函数
    return a + b * t + C * np.sin(omega * t + phi)

def d_u_hat_dt(t):  # 构造导数项
    return b + C * omega * np.cos(omega * t + phi)

# ======step 2: 构造 sigma^2(t) ======
residual_e = pd.read_csv('amsterdam_daily_temperature_AR1_residuals.csv', parse_dates=['date'], index_col='date')['residuals'].dropna().values
t_data = np.arange(len(residual_e))
e2_series = pd.Series(residual_e**2)
sigma_data = pd.Series(np.sqrt(e2_series.rolling(window=15, center=True).mean())) # 用平滑值来代替 E[e^2], 从而得到 sigma^2(t)

# 去除 sigma_data 中的 NaN，并同步处理 t_data
valid_idx = ~np.isnan(sigma_data.values)
t_data_clean = t_data[valid_idx]
y_data_clean = sigma_data.values[valid_idx]

# 自定义 sigma^2(t) 结构
def sigma_model(t, V, U, *coeffs):
    I = len(coeffs) // 2
    sin_part = sum([coeffs[i] * np.sin((i+1)*omega*t) for i in range(I)])
    cos_part = sum([coeffs[I+i] * np.cos((i+1)*omega*t) for i in range(I)])
    return V + U*t + sin_part + cos_part

# 初始参数
init_params = [1, 0] + [0.1]*(2*I)

assert not np.isnan(y_data_clean).any(), "y_data contains NaNs"
# 拟合 Fourier 模型
params, _ = curve_fit(sigma_model, t_data_clean, y_data_clean, p0=init_params)
print("=== sigma^2(t) 模型参数 ===")
print("V:", params[0])
print("U:", params[1])
print("Fourier coefficients (sin):", params[2:2+I])
print("Fourier coefficients (cos):", params[2+I:2+2*I])

# ======step 2: 可视化 sigma^2(t) ======
plt.figure(figsize=(12, 4))
plt.plot(t_data, np.abs(residual_e), label='|$e_t$| (AR(1) Residuals)', alpha=0.4)
plt.plot(t_data, sigma_data, label='Smoothed $\sigma(t)$ (15-day MA)', linewidth=2)
plt.xlabel("Time Index", fontsize=14)
plt.ylabel("Magnitude", fontsize=14)
plt.title("Smoothed Estimation of $\sigma(t)$ from AR(1) Residuals", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/sigma_smoothing.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 预测 sigma_model 拟合值
sigma_fitted = sigma_model(t_data_clean, *params)

plt.figure(figsize=(12, 4))
plt.plot(t_data_clean, y_data_clean, label='Smoothed $\sigma(t)$ (Data)', color='blue')
plt.plot(t_data_clean, sigma_fitted, label='Fitted $\sigma(t)$ (Fourier)', color='red', linestyle='--')
plt.xlabel("Time Index", fontsize=14)
plt.ylabel("Volatility", fontsize=14)
plt.title("Fourier Fit of $\sigma(t)$", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/sigma_fitted.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(t_data, e2_series, label='$e_t^2$', alpha=0.4)
plt.plot(t_data, sigma_data**2, label='Rolling Estimate of $\sigma^2(t)$', linewidth=2)
plt.xlabel("Time Index")
plt.ylabel("Variance")
plt.title("Raw $e_t^2$ vs Smoothed $\sigma^2(t)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/sigma_squared_estimation.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 生成解析函数
def sigma(t):
    return np.sqrt(sigma_model(t, *params))

# ===== Step 3: 欧拉法模拟温度路径 =====
dt = 1  # 时间步长
n_days = len(t)
T_sim = np.zeros(n_days)
T_sim[0] = y[0]
np.random.seed(42)

for i in range(1, n_days):
    mu_t = u_hat(t[i])
    dmu_dt = d_u_hat_dt(t[i])
    sigma_t = sigma(t[i])
    z = np.random.normal()
    T_sim[i] = T_sim[i-1] + (dmu_dt + kappa * (mu_t - T_sim[i-1])) * dt + sigma_t * np.sqrt(dt) * z  # dt=1

# 存储或后续使用模拟路径 T_sim
simulated_df = pd.DataFrame({
    'date': daily_df.index,
    'T_sim': T_sim
}).set_index('date')

# ===== Step 4 扩展：模拟多个温度路径 =====
n_paths = 50           # 模拟的路径数量
n_days = len(t)        # 时间长度
T_paths = np.zeros((n_paths, n_days))  # 所有模拟路径集合

np.random.seed(42)  # 固定随机数种子，确保结果可复现
for path in range(n_paths):
    T_sim = np.zeros(n_days)
    T_sim[0] = y[0]  # 起点设为真实起始温度
    for i in range(1, n_days):
        mu_t = u_hat(t[i])
        dmu_dt = d_u_hat_dt(t[i])
        sigma_t = sigma(t[i])
        z = np.random.normal()
        T_sim[i] = T_sim[i-1] + (dmu_dt + kappa * (mu_t - T_sim[i-1])) * dt + sigma_t * np.sqrt(dt) * z  # dt=1, FDM
    T_paths[path] = T_sim

# ===== 储存所有模拟路径 =====
simulated_df_paths = pd.DataFrame(T_paths.T, columns=[f'Path_{i+1}' for i in range(n_paths)], index=daily_df.index)
simulated_df_paths.to_csv('amsterdam_daily_temperature_simulated_paths.csv')

# ===== 可视化多个路径 =====
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(daily_df.index, T_paths[i], lw=0.7, alpha=0.6)
plt.title("Simulated Temperature Paths (Monte Carlo)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Temperature (°C)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/simulated_temperature_paths.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# # ===== 对比实际和模拟路径 =====
# plt.figure(figsize=(12, 4))
# plt.plot(daily_df.index, daily_df['temperature_2m_mean'], label='Observed Temp', alpha=0.5)
# plt.plot(simulated_df.index, simulated_df['T_sim'], label='Simulated Temp', linestyle='--')
# plt.title('Simulated Temperature Path', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Temperature (°C)', fontsize=14)
# plt.legend()
# plt.savefig('figs/simulated_temperature_path.png', dpi=300, bbox_inches='tight')
# plt.grid(True)
# plt.show()