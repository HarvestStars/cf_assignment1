import numpy as np
import pandas as pd
from pathlib import Path

def simulate_qe_step(X_t, V_t, *, kappa, theta, sigma, rho,
                     dt, psi_c=1.5, gamma1=0.5,
                     use_martingale_correction=False,
                     rng=None):
    """
    One bias-free QE step for the risk–neutral Heston model.
    
    Parameters
    ----------
    X_t, V_t : float or ndarray
        Log-asset price ln S_t  (discounted)  and variance V_t  at time t.
    kappa, theta, sigma, rho : float
        Heston parameters: mean-reversion speed, long-run mean,
        vol-of-vol, correlation.
    dt : float
        Time-step Δ.
    psi_c : float, optional
        Regime switch threshold (≈1.5 is recommended).
    gamma1 : float, optional
        Weight γ₁ in the trapezoidal rule for ∫V_u du  (0 ≤ γ₁ ≤ 1, default 0.5).
        γ₂ is set to 1−γ₁.
    use_martingale_correction : bool, optional
        If True, applies Andersen’s optional martingale correction to K₀.
    rng : numpy.random.Generator, optional
        Random-number generator (np.random.default_rng() if None).
    
    Returns
    -------
    X_tp, V_tp : same shape as inputs
        Updated ln S and V at time t+Δ.
    """
    if rng is None:
        rng = np.random.default_rng()
    X_t = np.asarray(X_t, dtype=float)
    V_t = np.asarray(V_t, dtype=float)
    gamma2 = 1.0 - gamma1

    # --- exact first two moments of V_{t+Δ} -----------------------
    exp_kdt = np.exp(-kappa * dt)
    m  = theta + (V_t - theta) * exp_kdt                    # eq. (31)
    s2 = (sigma**2 * exp_kdt / kappa) * (                  # eq. (32)
           V_t * (1 - exp_kdt) + theta * 0.5 * (1 - exp_kdt)**2)
    psi = s2 / (m * m)

    # --- sample V_{t+Δ} via QE scheme -----------------------------
    V_tp = np.empty_like(V_t)

    # Regime I  (ψ ≤ ψ_c)
    idx_I = psi <= psi_c
    if np.any(idx_I):
        psi_I = psi[idx_I]
        b2 = 2/psi_I - 1 + np.sqrt(2/psi_I) * np.sqrt(2/psi_I - 1)
        a  = m[idx_I] / (1 + b2)
        b  = np.sqrt(b2)
        ZV = rng.standard_normal(size=b.shape)
        V_tp[idx_I] = a * (b + ZV)**2                       # eq. ( Regime I )

    # Regime II (ψ > ψ_c)
    idx_II = ~idx_I
    if np.any(idx_II):
        psi_II = psi[idx_II]
        p     = (psi_II - 1) / (psi_II + 1)
        beta  = 2 / (m[idx_II] * (1 + psi_II))              # eq. (Regime II)
        U     = rng.random(size=p.shape)
        V_tp[idx_II] = np.where(U <= p,
                                0.0,
                                -np.log((1 - p) / (1 - U)) / beta)

    # Guard against round-off
    V_tp = np.maximum(V_tp, 0.0)

    # --- log-asset update ----------------------------------------
    # coefficients K_i  (B.4 step 4)
    K0 = -rho * kappa * theta * dt / sigma
    K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)

    if use_martingale_correction:
        # Andersen’s optional correction (Algorithm 1 step 5)
        A = rho / sigma**2 * (1 + kappa * gamma2 * dt) - 0.5 * gamma2 * dt * rho**2
        K0_star = np.empty_like(K0 + V_t)  # broadcast helper
        
        # compute K0* separately for the two regimes
        if np.any(idx_I):
            a_I = m[idx_I] / (1 + (2/psi[idx_I] - 1 +
                                   np.sqrt(2/psi[idx_I]) *
                                   np.sqrt(2/psi[idx_I] - 1)))
            b2_I = 2/psi[idx_I] - 1 + np.sqrt(2/psi[idx_I]) * np.sqrt(2/psi[idx_I] - 1)
            K0_star_I = (-A * b2_I * a_I / (1 - 2 * A * a_I)
                         + 0.5 * np.log(1 - 2 * A * a_I)
                         - (K1 + 0.5 * dt * gamma1))
            K0_star[idx_I] = K0_star_I

        if np.any(idx_II):
            p_II    = (psi[idx_II] - 1) / (psi[idx_II] + 1)
            beta_II = 2 / (m[idx_II] * (1 + psi[idx_II]))
            K0_star_II = (-np.log(beta_II * (1 - p_II) / (beta_II - A))
                          - (K1 + 0.5 * dt * gamma1))
            K0_star[idx_II] = K0_star_II

        K0 = K0_star  # replace

    # price increment (B.4 eq.)
    Z = rng.standard_normal(size=V_tp.shape)
    vol_term = K3 * V_t + K4 * V_tp
    vol_term = np.maximum(vol_term, 0.0)    # numeric safety
    X_tp = X_t + K0 + K1 * V_t + K2 * V_tp + np.sqrt(vol_term) * Z

    return X_tp, V_tp


def simulate_heston_qe_paths(
        S0, V0, *,                      # 初始价、初始方差
        kappa, theta, sigma, rho,       # Heston 4 参数
        T, Nsteps, Npaths,              # 终端时刻、时间步、路径数
        r=0.0,                          # 无风险利率 (若要返回未折现价格)
        psi_c=1.5, gamma1=0.5,          # QE 控制参数
        use_martingale_correction=False,
        seed=None,
        return_discounted=True,         # True: 返回 exp(X_t)；False: 乘回 e^{rt}
        return_full=True):              # True: 返回全路径；False: 只要末期
    """
    Monte-Carlo simulation of the Heston model using Andersen's QE scheme.

    Returns
    -------
    S_paths, V_paths : ndarray
        *shape = (Npaths, Nsteps+1)*  if `return_full=True`,
        otherwise 1-d arrays of length *Npaths* with terminal values only.

    Notes
    -----
    The single-step kernel `simulate_qe_step` generates the *discounted*
    log-price X  (i.e. without the rΔ drift).  When `return_discounted=False`
    we multiply back by exp(r t) so that S = exp(X)·e^{rt}.
    """
    rng = np.random.default_rng(seed)
    dt  = T / Nsteps
    time_grid = np.linspace(0.0, T, Nsteps + 1)

    # 预分配
    if return_full:
        X_paths = np.empty((Npaths, Nsteps + 1))
        V_paths = np.empty_like(X_paths)

    # 起点
    X = np.full(Npaths, np.log(S0))
    V = np.full(Npaths, V0)
    if return_full:
        X_paths[:, 0] = X
        V_paths[:, 0] = V

    # 主循环
    for j in range(1, Nsteps + 1):
        X, V = simulate_qe_step(
            X, V,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            dt=dt, psi_c=psi_c, gamma1=gamma1,
            use_martingale_correction=use_martingale_correction,
            rng=rng
        )
        if return_full:
            X_paths[:, j] = X
            V_paths[:, j] = V

    # 折现 ↔ 未折现
    if return_discounted:
        S = np.exp(X) if not return_full else np.exp(X_paths)
    else:
        if return_full:
            S = np.exp(X_paths + r * time_grid)
        else:  # 只有末期
            S = np.exp(X + r * T)

    # 输出
    if return_full:
        return S, V_paths
    else:
        return S, V      # 末期价、末期方差

def save_heston_paths_to_csv(S_paths: np.ndarray,
                             V_paths: np.ndarray,
                             *,
                             file_prefix: str = "heston",
                             out_dir: str | Path = ".",
                             time_grid: np.ndarray | None = None) -> tuple[Path, Path]:
    """
    Save simulated Heston paths to two CSV files.

    Parameters
    ----------
    S_paths, V_paths : ndarray
        Arrays of shape (Npaths, Nsteps+1) containing price and variance paths.
    file_prefix : str, optional
        Prefix for the file names:  <prefix>_S.csv  and  <prefix>_V.csv.
    out_dir : str or Path, optional
        Directory where the files will be written (created if missing).
    time_grid : 1-d ndarray, optional
        Time stamps for the columns;  if None, uses 0,1,2,…  (step indices).

    Returns
    -------
    s_file, v_file : pathlib.Path
        Paths of the written CSV files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if time_grid is None:
        time_grid = np.arange(S_paths.shape[1])

    # 构造列名，例如 ["t0", "t0.004", ..., "t1.0"]
    col_names = [f"t{t:.6g}" for t in time_grid]

    df_S = pd.DataFrame(S_paths, columns=col_names)
    df_V = pd.DataFrame(V_paths, columns=col_names)

    s_file = out_dir / f"{file_prefix}_S.csv"
    v_file = out_dir / f"{file_prefix}_V.csv"

    df_S.to_csv(s_file, index=False)
    df_V.to_csv(v_file, index=False)

    return s_file, v_file



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    # 模型参数
    params = dict(kappa=3.0, theta=0.04, sigma=0.5, rho=-0.7)
    S0, V0  = 100.0, 0.04
    T, Nsteps, Npaths = 1.0, 252, 100_000   # 1 年、日频、10 万条路径

    # 生成终端分布（只要末期，节省内存）
    S_T, V_T = simulate_heston_qe_paths(
        S0, V0, **params, T=T, Nsteps=Nsteps, Npaths=Npaths,
        r=0.02, return_discounted=False, return_full=False, seed=2024
    )

    # 估算平值欧式看涨期权价格
    K = S0
    payoff = np.maximum(S_T - K, 0.0)
    price  = np.exp(-0.02 * T) * payoff.mean()
    print(f"Monte-Carlo price ≈ {price:.4f}")

    # 保存路径
    # 1. 先生成路径
    S_paths, V_paths = simulate_heston_qe_paths(
        S0=100, V0=0.04,
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7,
        T=1.0, Nsteps=252, Npaths=50_000,
        r=0.03,
        return_discounted=False,   # 市场价轨迹
        seed=42,                   # 可复现
        return_full=True)          # 需要全路径

    # 2. 保存为 CSV
    s_csv, v_csv = save_heston_paths_to_csv(
        S_paths, V_paths,
        file_prefix="my_heston",
        out_dir="output")          # 目录自动创建
    print(f"Price paths saved to:    {s_csv}")
    print(f"Variance paths saved to: {v_csv}")