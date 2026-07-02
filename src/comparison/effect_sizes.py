"""
effect_sizes.py — Cliff's delta y efectos omnibus
===================================================

Funciones para calcular tamaños de efecto por pareja (Cliff's delta
con IC bootstrap) y efectos omnibus (η², ε²) para el análisis
comparativo de sentimiento entre voces discursivas.

Ref:
  Cliff, N. (1993). Dominance statistics. Psychological Bulletin, 114, 494-509.
  Romano et al. (2006). Exploring methods for evaluating group differences.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import COMPARISON_DIR


# ─────────────────────────────────────────────────────────────────────
# CLIFF'S DELTA
# ─────────────────────────────────────────────────────────────────────

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cliff's delta effect size.

    Parameters
    ----------
    x, y : np.ndarray
        Two independent samples.

    Returns
    -------
    delta : float
        Cliff's delta in [-1, 1].
    interpretation : str
        'negligible', 'small', 'medium', or 'large'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan, "N/A"

    # Count dominance: how many (x_i, y_j) pairs where x_i > y_j vs x_i < y_j
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)

    delta = (more - less) / (n_x * n_y)

    # Interpretation thresholds (Romano et al., 2006)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interp = "negligible"
    elif abs_delta < 0.33:
        interp = "small"
    elif abs_delta < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return delta, interp


def cliffs_delta_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Cliff's delta with bootstrap confidence interval.

    Parameters
    ----------
    x, y : np.ndarray
        Two independent samples.
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for CI (default 0.05 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'delta', 'ci_lower', 'ci_upper', 'interpretation'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    delta, interp = cliffs_delta(x, y)

    rng = np.random.RandomState(seed)
    boot_deltas = []

    for _ in range(n_boot):
        x_boot = rng.choice(x, size=len(x), replace=True)
        y_boot = rng.choice(y, size=len(y), replace=True)
        d, _ = cliffs_delta(x_boot, y_boot)
        boot_deltas.append(d)

    boot_deltas = np.array(boot_deltas)
    ci_lower = np.percentile(boot_deltas, 100 * alpha / 2)
    ci_upper = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

    return {
        "delta": round(delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────
# OMNIBUS EFFECT SIZES
# ─────────────────────────────────────────────────────────────────────

def omnibus_with_effects(groups: list[np.ndarray]) -> dict:
    """Kruskal-Wallis test with η² and ε² effect sizes.

    Parameters
    ----------
    groups : list[np.ndarray]
        List of samples (one per group).

    Returns
    -------
    dict
        Keys: 'H', 'p', 'eta_sq', 'epsilon_sq', 'N', 'k'.

    Notes
    -----
    η² = H / (N - 1)
    ε² = (H - k + 1) / (N - k)

    Interpretation of η² (Cohen-like):
      < 0.01  negligible
      < 0.06  small
      < 0.14  medium
      ≥ 0.14  large
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(groups)
    N = sum(len(g) for g in groups)

    if k < 2:
        return {"H": np.nan, "p": np.nan, "eta_sq": np.nan,
                "epsilon_sq": np.nan, "N": N, "k": k}

    H, p = stats.kruskal(*groups)

    eta_sq = H / (N - 1) if N > 1 else np.nan
    epsilon_sq = (H - k + 1) / (N - k) if N > k else np.nan

    return {
        "H": round(H, 4),
        "p": p,
        "eta_sq": round(eta_sq, 4),
        "epsilon_sq": round(epsilon_sq, 4),
        "N": N,
        "k": k,
    }


# ─────────────────────────────────────────────────────────────────────
# PAIRWISE EFFECT TABLE
# ─────────────────────────────────────────────────────────────────────

def pairwise_effect_table(
    df: pd.DataFrame,
    val_col: str = "sentiment",
    group_col: str = "source",
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Build a pairwise effect size table with Cliff's δ and Dunn's z.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with val_col and group_col columns.
    val_col : str
        Name of the numeric column.
    group_col : str
        Name of the group column.
    n_boot : int
        Bootstrap iterations for CI.

    Returns
    -------
    pd.DataFrame
        Columns: group_1, group_2, n_1, n_2, mean_1, mean_2,
                 cliffs_delta, ci_lower, ci_upper, interpretation,
                 dunn_z, p_adj.
    """
    from itertools import combinations

    groups = sorted(df[group_col].unique())
    records = []

    # Get Dunn's test z-values
    dunn_results = _dunn_z_values(df, val_col, group_col)

    for g1, g2 in combinations(groups, 2):
        x = df.loc[df[group_col] == g1, val_col].dropna().values
        y = df.loc[df[group_col] == g2, val_col].dropna().values

        ci = cliffs_delta_ci(x, y, n_boot=n_boot)

        # Get Dunn z and p for this pair
        pair_key = (g1, g2)
        dunn_info = dunn_results.get(pair_key, {"z": np.nan, "p_adj": np.nan})

        records.append({
            "group_1": g1,
            "group_2": g2,
            "n_1": len(x),
            "n_2": len(y),
            "mean_1": round(np.mean(x), 4),
            "mean_2": round(np.mean(y), 4),
            "cliffs_delta": ci["delta"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "interpretation": ci["interpretation"],
            "dunn_z": round(dunn_info["z"], 4) if not np.isnan(dunn_info["z"]) else np.nan,
            "p_adj": dunn_info["p_adj"],
        })

    return pd.DataFrame(records)


def _dunn_z_values(df: pd.DataFrame, val_col: str, group_col: str) -> dict:
    """Compute Dunn's test z-statistics and Bonferroni-adjusted p-values.

    Returns dict of {(g1, g2): {'z': float, 'p_adj': float}}.
    """
    from itertools import combinations

    groups = sorted(df[group_col].unique())
    k = len(groups)
    n_comparisons = k * (k - 1) // 2

    # Rank all values
    all_vals = df[val_col].dropna()
    ranks = stats.rankdata(all_vals)
    df_ranked = df.copy()
    df_ranked["_rank"] = np.nan
    df_ranked.loc[all_vals.index, "_rank"] = ranks
    N = len(all_vals)

    # Tied-ranks correction
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(tie_counts ** 3 - tie_counts)
    sigma = np.sqrt((N * (N + 1) / 12 - tie_sum / (12 * (N - 1))))

    results = {}
    for g1, g2 in combinations(groups, 2):
        r1 = df_ranked.loc[df_ranked[group_col] == g1, "_rank"].dropna()
        r2 = df_ranked.loc[df_ranked[group_col] == g2, "_rank"].dropna()
        n1, n2 = len(r1), len(r2)

        mean_diff = r1.mean() - r2.mean()
        se = sigma * np.sqrt(1 / n1 + 1 / n2)
        z = mean_diff / se if se > 0 else np.nan

        p = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
        p_adj = min(p * n_comparisons, 1.0) if not np.isnan(p) else np.nan

        results[(g1, g2)] = {"z": z, "p_adj": p_adj}

    return results


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Effect Sizes Module — Testing")

    # Synthetic test
    np.random.seed(42)
    academic = np.random.normal(0.57, 0.17, 1000)
    institutional = np.random.normal(0.78, 0.14, 32)
    public = np.random.normal(0.53, 0.20, 3000)

    # Omnibus
    omni = omnibus_with_effects([academic, institutional, public])
    print(f"\nOmnibus: H={omni['H']}, p={omni['p']:.2e}")
    print(f"  η² = {omni['eta_sq']:.4f}")
    print(f"  ε² = {omni['epsilon_sq']:.4f}")

    # Pairwise
    df = pd.DataFrame({
        "sentiment": np.concatenate([academic, institutional, public]),
        "source": (["academic"] * len(academic) +
                   ["institutional"] * len(institutional) +
                   ["public"] * len(public)),
    })
    table = pairwise_effect_table(df)
    print(f"\n{table.to_string(index=False)}")
