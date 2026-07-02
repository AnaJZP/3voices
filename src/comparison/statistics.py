"""
statistics.py — Utilidades de pruebas estadísticas para VOZ_SUS
===============================================================

Funciones auxiliares para análisis comparativo entre los tres grupos
de discurso (académico, institucional, ciudadano). Todas las pruebas
usan scipy.stats y devuelven diccionarios con resultados formateados.
"""

# ── Stdlib ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

# ── Local ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *


# ─────────────────────────────────────────────────────────────────────
# PRUEBAS NO PARAMÉTRICAS
# ─────────────────────────────────────────────────────────────────────

def kruskal_wallis_test(groups: list[np.ndarray]) -> dict:
    """Prueba de Kruskal-Wallis para k grupos independientes.

    Parameters
    ----------
    groups : list[np.ndarray]
        Lista de arrays, uno por grupo. Cada array contiene
        las observaciones numéricas del grupo.

    Returns
    -------
    dict
        Contiene 'H_statistic', 'p_value', 'n_groups', 'n_total',
        y 'significance' (cadena con asteriscos).
    """
    # Filtrar grupos vacíos
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {
            "H_statistic": np.nan,
            "p_value": np.nan,
            "n_groups": len(groups),
            "n_total": sum(len(g) for g in groups),
            "significance": "ns",
        }

    h_stat, p_val = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)

    return {
        "H_statistic": round(h_stat, 4),
        "p_value": p_val,
        "n_groups": len(groups),
        "n_total": n_total,
        "significance": format_significance(p_val),
    }


def dunn_posthoc(
    df: pd.DataFrame,
    val_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Prueba post-hoc de Dunn con corrección de Bonferroni.

    Implementación manual usando rangos medios y la fórmula estándar
    de Dunn, sin necesidad de scikit-posthocs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con al menos las columnas val_col y group_col.
    val_col : str
        Nombre de la columna numérica a comparar.
    group_col : str
        Nombre de la columna categórica (grupo).

    Returns
    -------
    pd.DataFrame
        Tabla con columnas: 'group_1', 'group_2', 'z_statistic',
        'p_value', 'p_adjusted', 'significance'.
    """
    df_clean = df[[val_col, group_col]].dropna()

    # Rangos globales
    df_clean = df_clean.copy()
    df_clean["_rank"] = stats.rankdata(df_clean[val_col])

    groups = df_clean[group_col].unique()
    n_total = len(df_clean)

    # Rangos medios y tamaños por grupo
    group_stats = {}
    for g in groups:
        mask = df_clean[group_col] == g
        group_stats[g] = {
            "mean_rank": df_clean.loc[mask, "_rank"].mean(),
            "n": mask.sum(),
        }

    # Comparaciones por pares
    from itertools import combinations

    pairs = list(combinations(sorted(groups), 2))
    n_comparisons = len(pairs)

    results = []
    # Varianza esperada bajo H0 (con ajuste por empates)
    tied_groups = stats.tiecorrect(df_clean["_rank"].values)
    sigma_sq = (n_total * (n_total + 1) / 12.0) * tied_groups

    for g1, g2 in pairs:
        n1 = group_stats[g1]["n"]
        n2 = group_stats[g2]["n"]
        mean_diff = group_stats[g1]["mean_rank"] - group_stats[g2]["mean_rank"]

        se = np.sqrt(sigma_sq * (1.0 / n1 + 1.0 / n2))
        if se == 0:
            z = 0.0
        else:
            z = mean_diff / se

        p_val = 2.0 * stats.norm.sf(abs(z))
        p_adj = min(p_val * n_comparisons, 1.0)  # Bonferroni

        results.append({
            "group_1": g1,
            "group_2": g2,
            "z_statistic": round(z, 4),
            "p_value": p_val,
            "p_adjusted": p_adj,
            "significance": format_significance(p_adj),
        })

    return pd.DataFrame(results)


def mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
) -> dict:
    """Prueba U de Mann-Whitney para dos grupos independientes.

    Parameters
    ----------
    group1 : np.ndarray
        Observaciones del primer grupo.
    group2 : np.ndarray
        Observaciones del segundo grupo.

    Returns
    -------
    dict
        Contiene 'U_statistic', 'p_value', 'effect_size_r',
        'n1', 'n2', y 'significance'.
        effect_size_r = Z / sqrt(N), interpretado como:
        pequeño ≈ 0.1, mediano ≈ 0.3, grande ≈ 0.5
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    # Eliminar NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return {
            "U_statistic": np.nan,
            "p_value": np.nan,
            "effect_size_r": np.nan,
            "n1": len(group1),
            "n2": len(group2),
            "significance": "ns",
        }

    u_stat, p_val = stats.mannwhitneyu(
        group1, group2, alternative="two-sided"
    )

    # Tamaño del efecto r = Z / sqrt(N)
    n_total = len(group1) + len(group2)
    z_score = stats.norm.ppf(1 - p_val / 2) if p_val < 1.0 else 0.0
    effect_r = abs(z_score) / np.sqrt(n_total)

    return {
        "U_statistic": round(u_stat, 4),
        "p_value": p_val,
        "effect_size_r": round(effect_r, 4),
        "n1": len(group1),
        "n2": len(group2),
        "significance": format_significance(p_val),
    }


def chi_square_test(contingency_table: np.ndarray) -> dict:
    """Prueba Chi-cuadrado de independencia con V de Cramér.

    Parameters
    ----------
    contingency_table : np.ndarray
        Tabla de contingencia (2D array o DataFrame).

    Returns
    -------
    dict
        Contiene 'chi2', 'p_value', 'dof', 'cramers_v',
        y 'significance'.
    """
    contingency_table = np.asarray(contingency_table)

    chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

    # V de Cramér
    n = contingency_table.sum()
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0

    return {
        "chi2": round(chi2, 4),
        "p_value": p_val,
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "significance": format_significance(p_val),
    }


# ─────────────────────────────────────────────────────────────────────
# MEDIDAS DE DIVERGENCIA Y TAMAÑO DEL EFECTO
# ─────────────────────────────────────────────────────────────────────

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Divergencia de Jensen-Shannon entre dos distribuciones.

    Usa la implementación de scipy (raíz cuadrada de JSD),
    devolviendo JSD² para mantener la escala estándar [0, 1].

    Parameters
    ----------
    p : np.ndarray
        Primera distribución de probabilidad (debe sumar 1).
    q : np.ndarray
        Segunda distribución de probabilidad (debe sumar 1).

    Returns
    -------
    float
        Divergencia JSD en [0, 1]. Valores cercanos a 0 indican
        distribuciones similares; cercanos a 1, muy diferentes.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalizar por seguridad
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q

    # scipy devuelve la raíz de JSD; elevamos al cuadrado
    jsd_sqrt = jensenshannon(p, q, base=2)
    return round(jsd_sqrt ** 2, 6)


def effect_size_eta_squared(h_stat: float, n: int) -> float:
    """Eta-cuadrado a partir de la estadística H de Kruskal-Wallis.

    η² = (H - k + 1) / (N - k), donde k es el número de grupos.
    Aproximación simplificada: η² ≈ H / (N - 1).

    Parameters
    ----------
    h_stat : float
        Estadística H de Kruskal-Wallis.
    n : int
        Tamaño total de la muestra.

    Returns
    -------
    float
        Tamaño del efecto eta-cuadrado. Interpretación:
        pequeño ≈ 0.01, mediano ≈ 0.06, grande ≈ 0.14
    """
    if n <= 1:
        return 0.0
    eta_sq = h_stat / (n - 1)
    return round(min(max(eta_sq, 0.0), 1.0), 6)


# ─────────────────────────────────────────────────────────────────────
# FORMATEO DE SIGNIFICANCIA
# ─────────────────────────────────────────────────────────────────────

def format_significance(p_value: float) -> str:
    """Convierte un p-valor en notación de asteriscos para publicación.

    Parameters
    ----------
    p_value : float
        Valor p de una prueba estadística.

    Returns
    -------
    str
        '***' si p < 0.001, '**' si p < 0.01,
        '*' si p < 0.05, 'ns' en otro caso.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA — pruebas rápidas
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  STATISTICS MODULE — Quick Test")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Datos sintéticos para pruebas
    g1 = np.random.normal(0.3, 0.2, 100)
    g2 = np.random.normal(0.5, 0.3, 120)
    g3 = np.random.normal(0.1, 0.15, 80)

    print("\n─── Kruskal-Wallis ───")
    kw = kruskal_wallis_test([g1, g2, g3])
    for k, v in kw.items():
        print(f"  {k}: {v}")

    print("\n─── Dunn Post-hoc ───")
    df_test = pd.DataFrame({
        "value": np.concatenate([g1, g2, g3]),
        "group": (["A"] * 100 + ["B"] * 120 + ["C"] * 80),
    })
    dunn = dunn_posthoc(df_test, "value", "group")
    print(dunn.to_string(index=False))

    print("\n─── Mann-Whitney ───")
    mw = mann_whitney_test(g1, g2)
    for k, v in mw.items():
        print(f"  {k}: {v}")

    print("\n─── Chi-Square ───")
    table = np.array([[50, 30, 20], [35, 45, 20]])
    chi = chi_square_test(table)
    for k, v in chi.items():
        print(f"  {k}: {v}")

    print("\n─── Jensen-Shannon Divergence ───")
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    jsd = jensen_shannon_divergence(p, q)
    print(f"  JSD(p, q) = {jsd}")

    print("\n─── Effect Size (Eta²) ───")
    eta = effect_size_eta_squared(kw["H_statistic"], kw["n_total"])
    print(f"  η² = {eta}")

    print("\n─── Format Significance ───")
    for pv in [0.0001, 0.005, 0.03, 0.15]:
        print(f"  p={pv} → {format_significance(pv)}")

    print("\n✓ All tests passed.")
