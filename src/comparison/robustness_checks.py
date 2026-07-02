"""
robustness_checks.py — Controles de robustez para el análisis de sentimiento
=============================================================================

Incluye:
  1. Modelo OLS con control de longitud y lenguaje
  2. Correlación entre scoring original y unificado
  3. Descriptivos por modelo de sentimiento

Estos análisis permiten demostrar que las diferencias entre voces
no son artefactos de la longitud del documento ni del modelo de
sentimiento utilizado.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import COMPARISON_DIR


def length_controlled_model(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    source_col: str = "source",
    token_col: str = "n_tokens",
    lang_col: str = "language",
    reference: str = "academic",
) -> dict:
    """OLS regression controlling for document length and language.

    Model: sentiment ~ C(source) + log(n_tokens) + C(language)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain sentiment_col, source_col, token_col, lang_col.
    sentiment_col : str
        Dependent variable column name.
    source_col : str
        Group variable (academic, institutional, public).
    token_col : str
        Token count column.
    lang_col : str
        Language column.
    reference : str
        Reference category for source dummy coding.

    Returns
    -------
    dict
        Keys: 'summary' (str), 'coefficients' (pd.DataFrame),
              'r_squared', 'adj_r_squared', 'f_stat', 'f_pvalue'.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    # Prepare data
    model_df = df[[sentiment_col, source_col, token_col, lang_col]].dropna().copy()

    # Ensure minimum tokens > 0 for log transform
    model_df[token_col] = model_df[token_col].clip(lower=1)
    model_df["log_tokens"] = np.log(model_df[token_col])

    # Ensure reference is first level
    source_order = [reference] + [s for s in sorted(model_df[source_col].unique()) if s != reference]
    model_df[source_col] = pd.Categorical(model_df[source_col], categories=source_order, ordered=False)

    # Fit OLS
    formula = f"{sentiment_col} ~ C({source_col}, Treatment(reference='{reference}')) + log_tokens + C({lang_col})"
    model = ols(formula, data=model_df).fit()

    # Extract coefficients
    coef_df = pd.DataFrame({
        "coefficient": model.params,
        "std_error": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1],
    })

    return {
        "summary": str(model.summary()),
        "coefficients": coef_df,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_stat": round(model.fvalue, 4),
        "f_pvalue": model.f_pvalue,
        "n_obs": int(model.nobs),
        "model": model,
    }


def sentiment_model_correlation(
    df: pd.DataFrame,
    col_original: str = "sentiment",
    col_unified: str = "sentiment_unified",
    group_col: str = "source",
) -> pd.DataFrame:
    """Compute Spearman correlation between original and unified sentiment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain col_original, col_unified, group_col.

    Returns
    -------
    pd.DataFrame
        Correlation by group: group, n, rho, p_value.
    """
    from scipy.stats import spearmanr

    records = []

    # Overall
    valid = df[[col_original, col_unified]].dropna()
    if len(valid) > 2:
        rho, p = spearmanr(valid[col_original], valid[col_unified])
        records.append({
            "group": "Overall",
            "n": len(valid),
            "spearman_rho": round(rho, 4),
            "p_value": p,
        })

    # By group
    for group in sorted(df[group_col].dropna().unique()):
        subset = df[df[group_col] == group][[col_original, col_unified]].dropna()
        if len(subset) > 2:
            rho, p = spearmanr(subset[col_original], subset[col_unified])
            records.append({
                "group": group,
                "n": len(subset),
                "spearman_rho": round(rho, 4),
                "p_value": p,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Robustness Checks Module -- Testing")

    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "sentiment": np.random.normal(0.57, 0.17, n),
        "sentiment_unified": np.random.normal(0.55, 0.18, n),
        "source": np.random.choice(["academic", "institutional", "public"], n),
        "n_tokens": np.random.lognormal(5, 1, n).astype(int),
        "language": np.random.choice(["en", "es"], n, p=[0.7, 0.3]),
    })

    # Test OLS
    try:
        result = length_controlled_model(df)
        print(f"\nR-squared: {result['r_squared']}")
        print(result["coefficients"].to_string())
    except ImportError as e:
        print(f"Skipping OLS test: {e}")

    # Test correlation
    corr = sentiment_model_correlation(df)
    print(f"\n{corr.to_string(index=False)}")
