import numpy as np
import matplotlib.pyplot as plt

def plot_raw_iv_surface(vols, strikes, tenors, title="Raw IV Surface"):
    """
    绘制 raw_ivol_surfaces.npy 数据结构的波动率曲面。
    vols: shape (15, N)
    strikes: shape (15, N)
    tenors: shape (N,)
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    vols = vols.T  # shape: (N, 15)
    strikes = strikes.T  # shape: (N, 15)
    maturities = np.tile(tenors[:, np.newaxis], (1, strikes.shape[1]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(strikes, maturities, vols, cmap='viridis', edgecolor='none')

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_interp_iv_surface(vols, strikes, tenors, title="Interpolated IV Surface"):
    """
    绘制 interp_ivol_surfaces.npy 的波动率曲面。
    vols: shape (N, 100)
    strikes: shape (100,)
    tenors: shape (N,)
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    K_grid, T_grid = np.meshgrid(strikes, tenors)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K_grid, T_grid, vols, cmap='viridis', edgecolor='none')

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)
    # ax.invert_yaxis()
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

def plot_iv_surface_moneyness(vol_matrix, strike_matrix, maturities, S0=1.0, title="IV Surface in Moneyness"):
    """
    使用 moneyness（log(K/S0)） 替代 strike 绘制隐含波动率平面。

    Parameters:
    -----------
    vol_matrix : ndarray (shape = [N, M])
        隐含波动率矩阵。
    strike_matrix : ndarray (shape = [N, M])
        原始 strike 矩阵（与 vol_matrix 同 shape）。
    maturities : array_like (shape = [N])
        到期时间（每行一组）。
    S0 : float
        当前资产现价，用于计算 moneyness。
    title : str
        图标题。
    """
    N, M = vol_matrix.shape
    maturity_matrix = np.tile(maturities[:, np.newaxis], (1, M))
    moneyness = np.log(strike_matrix / S0)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(moneyness, maturity_matrix, vol_matrix, cmap='viridis', edgecolor='none')

    ax.set_xlabel("Moneyness (log(K/S0))", fontsize=12)
    ax.set_ylabel("Maturity (T)", fontsize=12)
    ax.set_zlabel("Implied Volatility", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(f"figs/{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    # Reload files after reset
    raw_data = np.load("../raw_data/raw_ivol_surfaces.npy", allow_pickle=True).item()
    interp_data = np.load("../raw_data/interp_ivol_surfaces.npy", allow_pickle=True).item()
    # Choose specific date
    date = "2023 11 01"

    # Extract components from raw data
    raw_tenors = raw_data[date]["tenors"]     # shape (N,)
    raw_strikes = raw_data[date]["strikes"]   # shape (15, N), 15 is number of strikes
    raw_vols = raw_data[date]["vols"]         # shape (15, N), N is number of maturities(years)

    # Extract components from interpolated data
    interp_tenors = interp_data[date]["tenors"]   # shape (N,)
    interp_strikes = interp_data[date]["strikes"] # shape (100,)
    interp_vols = interp_data[date]["vols"]       # shape (N, 100)

    # 对于 maturity 最小的一组 raw 数据
    strike_candidates = raw_strikes[:, 0]  # shape: (15,)
    vols = raw_vols[:, 0]                  # shape: (15,)
    S0_est = strike_candidates[np.argmin(vols)]  # IV 最小的 K 被当作 S0
    print(f"Estimated S0: {S0_est} for {date}")

    central_strike_guess_interp = np.median(interp_strikes)
    print(f"Central strike guess: {central_strike_guess_interp} for {date}")
    # print(raw_vols.shape, raw_strikes.shape, raw_tenors.shape,interp_vols.shape, interp_strikes.shape, interp_tenors.shape)
    plot_iv_surface_moneyness(raw_vols.T, raw_strikes.T, raw_tenors, S0=S0_est, title="Raw SP500 Implied Volatility Surface(Moneyness)")
    # plot_interp_iv_surface(interp_vols, interp_strikes, interp_tenors, title="Interpolated SP500 Implied Volatility Surface")
