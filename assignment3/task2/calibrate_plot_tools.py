import seaborn as sns
import matplotlib.pyplot as plt

# 3D plotting
def plot_surface_flat(K_grid, T_grid, Z, zlabel, title):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K_grid, T_grid, Z, cmap='viridis', edgecolor='none')
    if zlabel == "Implied Volatility":
        ax.set_xlabel("Moneyness log(K/S0)", fontsize=12)
    else:
        ax.set_xlabel("Strike (K)", fontsize=12)
    ax.set_ylabel("Maturity (T)", fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(f'figs/{title.replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_heatmap_from_matrix(Z, moneyness_matrix, market_tenors, title,
                              zlabel="Error", cmap="RdBu_r", fmt=".3f"):
    """Plot heatmap with axis labels directly from moneyness_matrix + tenors."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    # 横轴：strike/moneyness 维度
    xlabels = [f"{x:.2f}" for x in moneyness_matrix[:, 0]]   # 固定 maturity=T0，取所有 moneyness（15 个）
    ylabels = [f"{t:.2f}" for t in market_tenors]            # 所有 11 个 maturity

    sns.heatmap(
        Z.T,  # 注意：transpose 是关键！否则 axis 反了
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=0.0,
        ax=ax
    )

    ax.set_xlabel("Moneyness log(K/S0)", fontsize=12)
    ax.set_ylabel("Maturity (T)", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"figs/{title.replace(' ', '_')}_heatmap.png", dpi=300)
    plt.show()
