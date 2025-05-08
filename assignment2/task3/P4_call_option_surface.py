import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === 读取模拟路径数据 ===
df = pd.read_csv("amsterdam_daily_temperature_simulated_paths.csv", index_col="date", parse_dates=True)

# === 参数设定 ===
r = 0.01             # 年化贴现率
N = 1                # 合约面值
alpha = 1.0          # payoff 缩放系数
cap = 500            # payoff 上限 CAP
floor = 500          # payoff 下限 FLOOR
strike_range = np.arange(100, 400, 20)  # K 执行价区间
window = 90          # HDD 合约期长度
num_paths = df.shape[1]
start_indices = np.arange(0, len(df) - window, 10)  # 起始点每10天滑动

def price_weather_derivative(
    df: pd.DataFrame,
    payoff_type="call_cap",  # "call_cap" or "put_floor"
    N=100,
    alpha=1.0,
    beta=1.0,
    cap=500,
    floor=500,
    r=0.01,
    window=90,
    strike_range=np.arange(100, 400, 20),
    step_days=10,
    K1=150,
    K2=250
):
    num_paths = df.shape[1]
    start_indices = np.arange(0, len(df) - window, step_days)
    pricing_surface = []

    for t0 in start_indices:
        for K in strike_range:
            temps_window = df.iloc[t0:t0+window].values.T  # shape: (num_paths, window)
            HDD = np.maximum(18 - temps_window, 0).sum(axis=1)

            if payoff_type == "call":
                payoff = N * np.maximum(HDD - K, 0)
            elif payoff_type == "call_cap":
                payoff = np.minimum(alpha * np.maximum(HDD - K, 0), cap)
            elif payoff_type == "put_floor":
                payoff = np.minimum(alpha * np.maximum(K - HDD, 0), floor)
            elif payoff_type == "collar":
                part1 = np.minimum(alpha * np.maximum(HDD - K1, 0), cap)
                part2 = np.minimum(beta * np.maximum(K2 - HDD, 0), floor)
                payoff = part1 - part2

            else:
                raise ValueError("Unsupported payoff_type")

            discounted_price = np.exp(-r * (window / 365)) * payoff.mean()
            pricing_surface.append((df.index[t0], K, discounted_price))

    return pd.DataFrame(pricing_surface, columns=["start_date", "strike_K", "price"])

def plot_pricing_surface(surface_df, title="Option Pricing Surface", zlabel="Price"):
    surface_df["start_date"] = pd.to_datetime(surface_df["start_date"])
    surface_df["date_float"] = mdates.date2num(surface_df["start_date"])
    X = surface_df["date_float"]
    Y = surface_df["strike_K"].values
    Z = surface_df["price"].values

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor='none')

    ax.set_ylabel("Strike K", fontsize=14)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.savefig(f"figs/{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches="tight")
    plt.show()

# === Call Option ===
surface_call = price_weather_derivative(df, payoff_type="call", N=1.0)
plot_pricing_surface(surface_call, title="Call Option", zlabel="C_HDD (call)")

# === Call Option with Cap ===
surface_call_cap = price_weather_derivative(df, payoff_type="call_cap", alpha=1.0, cap=cap)
plot_pricing_surface(surface_call_cap, title="Call Option with Cap", zlabel="C_HDD (call_cap)")

# === Put Option with Floor ===
surface_put_floor = price_weather_derivative(df, payoff_type="put_floor", alpha=1.0, floor=floor)
plot_pricing_surface(surface_put_floor, title="Put Option with Floor", zlabel="C_HDD (put_floor)")

# === Collar Option ===
surface_collar = price_weather_derivative(
    df,
    payoff_type="collar",
    alpha=1.0,
    beta=1.0,
    cap=cap,
    floor=floor,
    K1=150,
    K2=250
)
plot_pricing_surface(surface_collar, title="Collar Option", zlabel="C_HDD (collar)")
