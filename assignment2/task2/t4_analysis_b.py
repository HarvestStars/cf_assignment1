import numpy as np
import t4_analysis_a as t4a
import matplotlib.pyplot as plt

# 公共参数
S0 = 100
V0 = 0.04
r = 0.05
kappa = 2.0
theta = 0.04
T = 1.0
N = 252
M = 10000   # 固定M，做敏感性分析
scheme = 'euler'

# 要变化的参数组合
xi_list = [0.1, 1.0]
rho_list = [-0.9, 0.0, 0.5]
K_list = [90, 100, 110]  # in-the-money, at-the-money, out-of-the-money

# 存储所有实验结果
sensitivity_results = []

# 三重循环跑所有组合
for xi in xi_list:
    for rho in rho_list:
        for K in K_list:
            res = t4a.run_simulation(S0, V0, r, kappa, theta, xi, rho, T, N, K, scheme, M)
            sensitivity_results.append({
                'xi': xi,
                'rho': rho,
                'K': K,
                'plain_price': res['plain_price'],
                'plain_stderr': res['plain_stderr'],
                'plain_var': res['plain_var'],
                'cv_price': res['cv_price'],
                'cv_stderr': res['cv_stderr'],
                'cv_var': res['cv_var']
            })

# 打印结果表格
print(f"{'xi':>5} | {'rho':>5} | {'K':>5} | {'Plain Price':>12} | {'Plain Stderr':>12} | {'Plain Var':>12} | {'CV Price':>12} | {'CV Stderr':>12} | {'CV Var':>12}")
print('-'*120)
for res in sensitivity_results:
    print(f"{res['xi']:5.1f} | {res['rho']:5.1f} | {res['K']:5d} | "
          f"{res['plain_price']:12.4f} | {res['plain_stderr']:12.4f} | {res['plain_var']:12.6f} | "
          f"{res['cv_price']:12.4f} | {res['cv_stderr']:12.4f} | {res['cv_var']:12.6f}")

# 整理数据
labels = []
plain_stderr = []
cv_stderr = []

for res in sensitivity_results:
    label = f"ξ={res['xi']},ρ={res['rho']},K={res['K']}"
    labels.append(label)
    plain_stderr.append(res['plain_stderr'])
    cv_stderr.append(res['cv_stderr'])

x = np.arange(len(labels))  # 横坐标位置
width = 0.35  # 柱子宽度

# 创建图
fig, ax = plt.subplots(figsize=(14,6))

# 画两组柱子
rects1 = ax.bar(x - width/2, plain_stderr, width, label='Plain MC', color='skyblue')
rects2 = ax.bar(x + width/2, cv_stderr, width, label='Control Variate MC', color='lightgreen')

# 加标签和标题
ax.set_xlabel('Parameter Combinations')
ax.set_ylabel('Standard Error')
ax.set_title('Standard Error Comparison under Different Heston Parameters')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.show()
