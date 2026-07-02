"""
cross_group.py — Análisis comparativo entre los tres grupos de discurso
=======================================================================

Compara distribuciones de temas, keywords y sentimiento entre las voces
académica, política y ciudadana usando divergencia Jensen-Shannon,
similitud Jaccard y pruebas de Kruskal-Wallis / Dunn.
"""

# ── Stdlib ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Local ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *
from src.comparison.statistics import (
    jensen_shannon_divergence,
    kruskal_wallis_test,
    dunn_posthoc,
    effect_size_eta_squared,
    format_significance,
)


# ─────────────────────────────────────────────────────────────────────
# GRUPOS — pares de comparación
# ─────────────────────────────────────────────────────────────────────

SOURCE_GROUPS = ["academic", "political", "public"]
GROUP_PAIRS = [
    ("academic", "political"),
    ("academic", "public"),
    ("political", "public"),
]


# ─────────────────────────────────────────────────────────────────────
# DIVERGENCIA DE TEMAS
# ─────────────────────────────────────────────────────────────────────

def compute_topic_divergence(topic_dists: dict) -> pd.DataFrame:
    """Calcula la Divergencia Jensen-Shannon entre distribuciones de temas.

    Compara cada par de fuentes (académica-política, académica-pública,
    política-pública) para cuantificar qué tan diferentes son sus
    distribuciones temáticas.

    Parameters
    ----------
    topic_dists : dict
        Diccionario {source_name: np.array} donde cada array es la
        distribución de probabilidad sobre temas (debe sumar 1).
        Ejemplo: {"academic": [0.2, 0.3, ...], "political": [...]}

    Returns
    -------
    pd.DataFrame
        Matriz de divergencia simétrica con índices/columnas = nombres
        de fuentes. Diagonal = 0, off-diagonal = JSD.
    """
    print("\n─── Computing Topic Divergence (JSD) ───")

    sources = list(topic_dists.keys())
    n = len(sources)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            p = np.asarray(topic_dists[sources[i]], dtype=float)
            q = np.asarray(topic_dists[sources[j]], dtype=float)

            # Igualar longitudes rellenando con ceros si difieren
            max_len = max(len(p), len(q))
            p_padded = np.zeros(max_len)
            q_padded = np.zeros(max_len)
            p_padded[:len(p)] = p
            q_padded[:len(q)] = q

            jsd = jensen_shannon_divergence(p_padded, q_padded)
            matrix[i, j] = jsd
            matrix[j, i] = jsd

            print(f"  JSD({sources[i]} vs {sources[j]}) = {jsd:.4f}")

    df_divergence = pd.DataFrame(matrix, index=sources, columns=sources)
    return df_divergence


# ─────────────────────────────────────────────────────────────────────
# SOLAPAMIENTO DE KEYWORDS
# ─────────────────────────────────────────────────────────────────────

def compute_keyword_overlap(
    keyword_lists: dict,
    top_n: int = 50,
) -> pd.DataFrame:
    """Calcula similitud Jaccard de keywords entre pares de grupos.

    Mide el solapamiento léxico entre los términos más frecuentes
    de cada fuente de discurso.

    Parameters
    ----------
    keyword_lists : dict
        Diccionario {source_name: list[str]} con las keywords
        ordenadas por frecuencia descendente.
    top_n : int
        Número de keywords top a considerar por grupo.

    Returns
    -------
    pd.DataFrame
        Matriz de similitud Jaccard simétrica. Diagonal = 1.0,
        off-diagonal = Jaccard(top_n keywords).
    """
    print(f"\n─── Computing Keyword Overlap (Jaccard, top {top_n}) ───")

    sources = list(keyword_lists.keys())
    n = len(sources)
    matrix = np.eye(n)  # Diagonal = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(keyword_lists[sources[i]][:top_n])
            set_j = set(keyword_lists[sources[j]][:top_n])

            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 0.0

            matrix[i, j] = jaccard
            matrix[j, i] = jaccard

            print(
                f"  Jaccard({sources[i]} vs {sources[j]}) = {jaccard:.4f}"
                f"  ({intersection} shared / {union} total)"
            )

    df_overlap = pd.DataFrame(matrix, index=sources, columns=sources)
    return df_overlap


# ─────────────────────────────────────────────────────────────────────
# COMPARACIÓN DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def compute_sentiment_comparison(df: pd.DataFrame) -> dict:
    """Compara sentimiento entre los tres grupos de discurso.

    Ejecuta la prueba de Kruskal-Wallis (comparación global) seguida
    de la prueba post-hoc de Dunn (comparaciones por pares) y calcula
    el tamaño del efecto eta-cuadrado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source' (grupo) y 'sentiment_score'
        (puntuación numérica de sentimiento).

    Returns
    -------
    dict
        Diccionario con:
        - 'kruskal_wallis': resultados de la prueba global
        - 'dunn_posthoc': DataFrame con comparaciones por pares
        - 'eta_squared': tamaño del efecto
        - 'group_descriptives': estadísticas descriptivas por grupo
    """
    print("\n─── Computing Sentiment Comparison ───")

    # Validar columnas requeridas
    required_cols = {"source", "sentiment_score"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # Estadísticas descriptivas por grupo
    descriptives = (
        df.groupby("source")["sentiment_score"]
        .agg(["count", "mean", "std", "median"])
        .round(4)
    )
    print("\n  Group descriptives:")
    print(descriptives.to_string(index=True))

    # Kruskal-Wallis (prueba global)
    groups = [
        df.loc[df["source"] == src, "sentiment_score"].dropna().values
        for src in SOURCE_GROUPS
        if src in df["source"].unique()
    ]
    kw_result = kruskal_wallis_test(groups)
    print(f"\n  Kruskal-Wallis H={kw_result['H_statistic']}, "
          f"p={kw_result['p_value']:.6f} {kw_result['significance']}")

    # Tamaño del efecto
    eta_sq = effect_size_eta_squared(
        kw_result["H_statistic"], kw_result["n_total"]
    )
    print(f"  Effect size η² = {eta_sq}")

    # Post-hoc de Dunn
    df_filtered = df[df["source"].isin(SOURCE_GROUPS)].copy()
    dunn_df = dunn_posthoc(df_filtered, "sentiment_score", "source")
    print("\n  Dunn post-hoc results:")
    print(dunn_df.to_string(index=False))

    return {
        "kruskal_wallis": kw_result,
        "dunn_posthoc": dunn_df,
        "eta_squared": eta_sq,
        "group_descriptives": descriptives,
    }


# ─────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────

def run_cross_group_analysis(
    df: pd.DataFrame,
    topic_results: dict,
) -> dict:
    """Ejecuta todos los análisis comparativos entre grupos.

    Pipeline completo que integra divergencia de temas, solapamiento
    de keywords y comparación de sentimiento. Guarda resultados
    intermedios y finales en data/results/.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado con columnas 'source', 'sentiment_score',
        'clean_text', entre otras.
    topic_results : dict
        Resultados del modelado de temas. Debe contener:
        - 'topic_distributions': dict {source: np.array}
        - 'keywords': dict {source: list[str]}

    Returns
    -------
    dict
        Diccionario con todos los resultados:
        - 'topic_divergence': pd.DataFrame (matriz JSD)
        - 'keyword_overlap': pd.DataFrame (matriz Jaccard)
        - 'sentiment_comparison': dict (pruebas estadísticas)
    """
    print("=" * 60)
    print("  CROSS-GROUP COMPARATIVE ANALYSIS")
    print("=" * 60)

    results = {}

    # 1. Divergencia de temas
    if "topic_distributions" in topic_results:
        topic_div = compute_topic_divergence(
            topic_results["topic_distributions"]
        )
        results["topic_divergence"] = topic_div

        out_path = COMPARISON_DIR / "topic_divergence.csv"
        topic_div.to_csv(out_path)
        print(f"\n  ✓ Saved: {out_path}")
    else:
        print("\n  ⚠ Skipping topic divergence — no distributions found")

    # 2. Solapamiento de keywords
    if "keywords" in topic_results:
        kw_overlap = compute_keyword_overlap(topic_results["keywords"])
        results["keyword_overlap"] = kw_overlap

        out_path = COMPARISON_DIR / "keyword_overlap.csv"
        kw_overlap.to_csv(out_path)
        print(f"  ✓ Saved: {out_path}")
    else:
        print("  ⚠ Skipping keyword overlap — no keyword lists found")

    # 3. Comparación de sentimiento
    if "sentiment_score" in df.columns:
        sent_comp = compute_sentiment_comparison(df)
        results["sentiment_comparison"] = sent_comp

        # Guardar post-hoc como CSV
        out_path = COMPARISON_DIR / "sentiment_dunn_posthoc.csv"
        sent_comp["dunn_posthoc"].to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")

        # Guardar descriptivos
        out_path = COMPARISON_DIR / "sentiment_descriptives.csv"
        sent_comp["group_descriptives"].to_csv(out_path)
        print(f"  ✓ Saved: {out_path}")
    else:
        print("  ⚠ Skipping sentiment comparison — column not found")

    print("\n" + "=" * 60)
    print("  ✓ Cross-group analysis complete")
    print("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CROSS-GROUP MODULE — Demo with synthetic data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ── Datos sintéticos ──
    n_topics = 8
    topic_dists = {
        "academic": np.random.dirichlet(np.ones(n_topics)),
        "political": np.random.dirichlet(np.ones(n_topics)),
        "public": np.random.dirichlet(np.ones(n_topics)),
    }

    keyword_lists = {
        "academic": ["sustainability", "climate", "energy", "policy",
                      "emissions", "carbon", "biodiversity", "adaptation",
                      "resilience", "mitigation"],
        "political": ["policy", "climate", "agreement", "nations",
                       "development", "economy", "targets", "emissions",
                       "sustainability", "transition"],
        "public": ["climate", "planet", "future", "change", "pollution",
                    "recycle", "energy", "water", "green", "nature"],
    }

    n_docs = 300
    df_test = pd.DataFrame({
        "source": (["academic"] * 100 + ["political"] * 100
                   + ["public"] * 100),
        "sentiment_score": np.concatenate([
            np.random.normal(0.1, 0.3, 100),
            np.random.normal(0.4, 0.2, 100),
            np.random.normal(-0.1, 0.4, 100),
        ]),
    })

    topic_results = {
        "topic_distributions": topic_dists,
        "keywords": keyword_lists,
    }

    results = run_cross_group_analysis(df_test, topic_results)
    print("\n✓ Demo finished successfully.")
