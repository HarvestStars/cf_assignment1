import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats

daily_df = pd.read_csv('amsterdam_daily_temperature.csv', parse_dates=['date'], index_col='date')
# ===== Task 1: EDA =====
# 1.1 计算滚动均值和方差
daily_df['rolling_mean_30'] = daily_df['temperature_2m_mean'].rolling(window=30).mean()
daily_df['rolling_var_30'] = daily_df['temperature_2m_mean'].rolling(window=30).var()

# 1.2 时间序列分解（假设一年为 365 天）
decompose_result = seasonal_decompose(daily_df['temperature_2m_mean'], model='additive', period=365)
daily_df['trend'] = decompose_result.trend
daily_df['seasonal'] = decompose_result.seasonal
daily_df['residual'] = decompose_result.resid

# 1.3 绘图代码（滚动统计）
plt.figure(figsize=(12, 5))
plt.plot(daily_df['temperature_2m_mean'], label='Original')
plt.plot(daily_df['rolling_mean_30'], label='Rolling Mean (30 days)', linestyle='--')
plt.plot(daily_df['rolling_var_30'], label='Rolling Variance (30 days)', linestyle=':')
plt.title("Temperature Series with Rolling Mean and Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

# 1.4 绘图代码（趋势+季节+残差分解）
decompose_result.plot()
plt.suptitle("Seasonal Decomposition", fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

# ===== Task 2: Residual Analysis =====
# 2.1 残差直方图
plt.figure(figsize=(8, 6))
plt.hist(daily_df['residual'].dropna(), bins=60)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
stats.probplot(daily_df['residual'].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

# 2.2 ACF 和 PACF 图
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(daily_df['residual'].dropna(), ax=ax[0], lags=50)
plot_pacf(daily_df['residual'].dropna(), ax=ax[1], lags=50)
plt.tight_layout()
plt.show()
plt.close()

# 2.3 ADF 平稳性检验
resid_clean = daily_df['residual'].dropna()
adf_result = adfuller(resid_clean)

adf_summary = {
    "ADF Statistic": adf_result[0],
    "p-value": adf_result[1],
    "Critical Values": adf_result[4]
}
print("ADF Test Summary:")
for key, value in adf_summary.items():
    print(f"{key}: {value}")
