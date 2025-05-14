import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Barrier_montecarlo import closed_form_up_and_out_call_with_tau

# 定义参数和网格范围
K = 100
B = 120
T = 1.0
r = 0.05
sigma = 0.25

S_vals = np.linspace(50, 150, 100)    # 标的价格范围
t_vals = np.linspace(0.0, 0.99, 100)  # 当前时刻（不能取T=1，避免tau=0）
S_grid, t_grid = np.meshgrid(S_vals, t_vals)

# 计算价格面
price_grid = np.vectorize(lambda S, t: closed_form_up_and_out_call_with_tau(S, K, B, T, t, r, sigma))(S_grid, t_grid)

# 可视化（2D 热力图）
plt.figure(figsize=(10, 6))
cp = plt.contourf(S_grid, t_grid, price_grid, levels=100, cmap='viridis')
plt.colorbar(cp)
plt.xlabel('Spot Price $S_t$')
plt.ylabel('Time $t$')
plt.title('Up-and-Out Call Option Price Surface (Closed-form)')
plt.show()

# (可选) Step 5: 3D 曲面图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, t_grid, price_grid, cmap='viridis')
ax.set_xlabel('Spot Price $S_t$')
ax.set_ylabel('Time $t$')
ax.set_zlabel('Option Price')
ax.set_title('3D View of Up-and-Out Call Option Price Surface')
plt.show()
