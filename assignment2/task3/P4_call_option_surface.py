import pandas as pd
import numpy as np

# --------------------
# 参数设定
# --------------------
N = 100      # tick size（单位票面金额）
K = 200      # 执行价（strike level）
r = 0.01     # 无风险利率
t0 = 0       # 当前时间（t）
tn = 90      # 合约长度（天）
temp_threshold = 18  # HDD 基准温度

# --------------------
# 读取模拟数据
# --------------------
df = pd.read_csv("amsterdam_daily_temperature_simulated_paths.csv", parse_dates=['date'])
df.set_index('date', inplace=True)

# 确保只取合同期限内的数据（90天）
df_contract = df.iloc[:tn]  # tn = 合约期的天数，比如 90天

# --------------------
# 计算每条路径的 HDD 值
# --------------------
def compute_HDD(path_temp_series):
    return np.sum(np.maximum(temp_threshold - path_temp_series, 0))

Hn_values = df_contract.apply(compute_HDD, axis=0)  # 对每一列（即每条路径）计算 H_n

# --------------------
# 计算期权支付
# --------------------
payoffs = np.maximum(Hn_values - K, 0)
option_price = np.exp(-r * (tn - t0)) * N * payoffs.mean()

# --------------------
# 输出结果
# --------------------
print(f"HDD Call Option Price (C_HDD): {option_price:.2f}")
