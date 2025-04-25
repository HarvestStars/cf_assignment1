import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 读取模拟路径数据 ===
df = pd.read_csv("amsterdam_daily_temperature_simulated_paths.csv", index_col="date", parse_dates=True)

# === 参数设定 ===
r = 0.01             # 年化贴现率
N = 100              # 合约面值
strike_range = np.arange(100, 400, 20)  # K 执行价区间
window = 90          # HDD 合约期长度
num_paths = df.shape[1]
start_indices = np.arange(0, len(df) - window, 10)  # 起始点每10天滑动

# === 构造定价平面 ===
pricing_surface = []

for t0 in start_indices:
    for K in strike_range:
        temps_window = df.iloc[t0:t0+window].values.T  # 形状: (paths, window_90_days_values)
        HDD = np.maximum(18 - temps_window, 0).sum(axis=1)
        payoff = N * np.maximum(HDD - K, 0)
        discounted_price = np.exp(-r * (window / 365)) * payoff.mean()
        pricing_surface.append((df.index[t0], K, discounted_price))

# === 转为DataFrame 方便画图 ===
surface_df = pd.DataFrame(pricing_surface, columns=["start_date", "strike_K", "C_HDD"])
surface_df["start_date"] = pd.to_datetime(surface_df["start_date"])
X = surface_df["start_date"].map(pd.Timestamp.toordinal)
Y = surface_df["strike_K"].values
Z = surface_df["C_HDD"].values

# === 绘制定价曲面图 ===
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor='none')
ax.set_xlabel("Start Date")
ax.set_ylabel("Strike K")
ax.set_zlabel("Option Price C_HDD")
ax.set_title("Temperature Derivative Pricing Surface")
plt.show()
