import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设定周期
omega = 2 * np.pi / 365.25

# ---------- STEP 1: 定义确定性模型 ----------
def seasonal_model(t, a, b, a1, b1):
    return a + b * t + a1 * np.cos(omega * t) + b1 * np.sin(omega * t)

# ---------- STEP 2: 准备数据 ----------
# 读取数据
daily_df = pd.read_csv('amsterdam_daily_temperature.csv', parse_dates=['date'], index_col='date')
t0 = daily_df.index[0].toordinal()
t = np.array([d.toordinal() - t0 for d in daily_df.index])  # 自变量 t
y = daily_df['temperature_2m_mean'].values  # 因变量 y

# ---------- STEP 3: 拟合曲线 ----------
params, cov = curve_fit(seasonal_model, t, y)
a, b, a1, b1 = params
daily_df['model'] = seasonal_model(t, *params)
daily_df['residuals'] = daily_df['temperature_2m_mean'] - daily_df['model']
# save the residuals to a CSV file
daily_df['residuals'].to_csv('amsterdam_daily_temperature_residuals.csv')

# ---------- STEP 4: 参数转换 ----------
C = np.sqrt(a1**2 + b1**2)
phi = np.arctan2(b1, a1)  # 注意：arctan2(y, x)
phi_deg = np.degrees(phi)

# ---------- STEP 5: 可视化 ----------
plt.figure(figsize=(12, 4))
plt.plot(daily_df.index, y, label='Observed Temp')
plt.plot(daily_df.index, daily_df['model'], label='Fitted $\mu(t)$', linestyle='--')
plt.legend()
plt.title('Temperature vs Fitted Seasonal Model')
plt.grid(True)
plt.savefig('figs/deterministic_temp_model.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot the residuals
plt.figure(figsize=(12, 4))
plt.plot(daily_df.index, daily_df['residuals'], label='Residuals', color='orange')
plt.savefig('figs/deterministic_temp_residuals.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------- STEP 6: 打印结果 ----------
print("=== 简化模型参数 ===")
print(f"a  (常数项)       = {a:.3f}")
print(f"b  (线性趋势)     = {b:.5f}")
print(f"a1 (cos 系数)    = {a1:.3f}")
print(f"b1 (sin 系数)    = {b1:.3f}")

print("\n=== 原始模型转换结果 ===")
print(f"A  = {a:.3f}   (对应常数项)")
print(f"B  = {b:.5f}   (对应线性项)")
print(f"C  = {C:.3f}   (正弦振幅)")
print(f"phi (弧度)     = {phi:.3f}")
print(f"phi (角度)     = {phi_deg:.2f}°")

# ---------- STEP 7: 拟合 AR 模型并确定最佳阶数 ----------
# 尝试 1~5 阶，选择 AIC 最小的模型
residuals = daily_df['residuals'].dropna()
best_aic = np.inf
best_lag = 1
best_model = None

for lag in range(1, 6):
    model = AutoReg(residuals, lags=lag, old_names=False).fit() # Xt = a0 + a1 * Xt-1 + a2 * Xt-2 + ... + e
    if model.aic < best_aic:
        best_aic = model.aic
        best_lag = lag
        best_model = model

print(f"✅ 最佳 AR 阶数: {best_lag} (AIC = {best_aic:.2f})")
print("\nAR 模型系数:")
print(best_model.params)

# ---------- STEP 8: 若为 AR(1)，估计 κ ----------
model_ar1 = AutoReg(residuals, lags=1, old_names=False).fit()
gamma = model_ar1.params[1]  # AR(1) 系数
kappa = 1 - gamma  # 计算 κ

plt.figure(figsize=(12, 4))
plt.plot(daily_df.index, daily_df['residuals'], label='Residuals', color='orange')
plt.plot(daily_df.index[1:], model_ar1.fittedvalues, label='Fitted AR(1)', linestyle='--')
plt.legend()
plt.show()
plt.savefig('figs/deterministic_temp_AR1.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"κ (AR(1) 系数) = {kappa:.3f}")