"""
tables.py — Generación de tablas para el artículo de investigación
==================================================================

Crea tablas resumen formateadas para publicación: estadísticas
descriptivas del corpus, temas, sentimiento, pruebas estadísticas
y métricas de red. Exporta en CSV y LaTeX.
"""

# ── Stdlib ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Local ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *


# ─────────────────────────────────────────────────────────────────────
# TABLA 1: RESUMEN DEL CORPUS
# ─────────────────────────────────────────────────────────────────────

def create_corpus_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de estadísticas descriptivas del corpus.

    Incluye número de documentos, longitud media del texto,
    desviación estándar, mediana, y rango de fechas para cada
    combinación de fuente (source) e idioma (language).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado con columnas 'source', 'language',
        y al menos 'clean_text' o 'text'.

    Returns
    -------
    pd.DataFrame
        Tabla de resumen con una fila por combinación source × language,
        más una fila de totales.
    """
    print("\n─── Creating Corpus Summary Table ───")

    # Determinar columna de texto
    text_col = "clean_text" if "clean_text" in df.columns else "text"

    df = df.copy()
    df["_text_length"] = df[text_col].fillna("").str.len()
    df["_word_count"] = df[text_col].fillna("").str.split().str.len()

    # Agrupar por fuente e idioma
    group_cols = ["source"]
    if "language" in df.columns:
        group_cols.append("language")

    summary = (
        df.groupby(group_cols)
        .agg(
            n_documents=("_text_length", "count"),
            mean_char_length=("_text_length", "mean"),
            std_char_length=("_text_length", "std"),
            median_char_length=("_text_length", "median"),
            mean_word_count=("_word_count", "mean"),
            std_word_count=("_word_count", "std"),
        )
        .round(1)
        .reset_index()
    )

    # Agregar rango de fechas si existe la columna 'year'
    if "year" in df.columns:
        year_range = (
            df.groupby(group_cols)["year"]
            .agg(year_min="min", year_max="max")
            .reset_index()
        )
        summary = summary.merge(year_range, on=group_cols)

    # Fila de totales
    total_row = {
        "source": "TOTAL",
        "n_documents": len(df),
        "mean_char_length": round(df["_text_length"].mean(), 1),
        "std_char_length": round(df["_text_length"].std(), 1),
        "median_char_length": round(df["_text_length"].median(), 1),
        "mean_word_count": round(df["_word_count"].mean(), 1),
        "std_word_count": round(df["_word_count"].std(), 1),
    }
    if "language" in group_cols:
        total_row["language"] = "all"
    if "year" in df.columns:
        total_row["year_min"] = df["year"].min()
        total_row["year_max"] = df["year"].max()

    total_df = pd.DataFrame([total_row])
    summary = pd.concat([summary, total_df], ignore_index=True)

    # Formatear n_documents como entero
    summary["n_documents"] = summary["n_documents"].astype(int)

    print(f"  Rows: {len(summary)} | Columns: {list(summary.columns)}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# TABLA 2: RESUMEN DE TEMAS
# ─────────────────────────────────────────────────────────────────────

def create_topic_summary_table(topic_results: dict) -> pd.DataFrame:
    """Genera tabla resumen de los temas principales por grupo.

    Incluye los temas top con sus keywords representativas y la
    proporción de documentos asignados a cada tema.

    Parameters
    ----------
    topic_results : dict
        Resultados del modelado de temas. Esperado:
        - 'topic_info': dict o pd.DataFrame con info de temas
        - 'topic_keywords': dict {topic_id: list[str]} (top keywords)
        - 'topic_distributions': dict {source: np.array}

    Returns
    -------
    pd.DataFrame
        Tabla con columnas: topic_id, keywords, y una columna de
        prevalencia por cada fuente.
    """
    print("\n─── Creating Topic Summary Table ───")

    rows = []

    # Obtener keywords por tema
    keywords = topic_results.get("topic_keywords", {})
    distributions = topic_results.get("topic_distributions", {})

    if not keywords:
        print("  ⚠ No topic keywords found")
        return pd.DataFrame()

    topic_ids = sorted(keywords.keys())

    for tid in topic_ids:
        row = {
            "topic_id": tid,
            "keywords": ", ".join(keywords[tid][:10]),
        }

        # Prevalencia por fuente
        for source, dist in distributions.items():
            dist_arr = np.asarray(dist)
            if tid < len(dist_arr):
                row[f"prevalence_{source}"] = round(dist_arr[tid], 4)
            else:
                row[f"prevalence_{source}"] = 0.0

        rows.append(row)

    summary = pd.DataFrame(rows)
    print(f"  Topics: {len(summary)} | Sources: {list(distributions.keys())}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# TABLA 3: RESUMEN DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def create_sentiment_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de estadísticas de sentimiento por grupo.

    Incluye media, mediana, desviación estándar, rango intercuartil
    y distribución por categorías (positivo, neutro, negativo).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source' y 'sentiment_score'.

    Returns
    -------
    pd.DataFrame
        Tabla con estadísticas de sentimiento por grupo de discurso.
    """
    print("\n─── Creating Sentiment Summary Table ───")

    required = {"source", "sentiment_score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    summary = (
        df.groupby("source")["sentiment_score"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min_val="min",
            max_val="max",
        )
        .round(4)
        .reset_index()
    )

    # Categorías de sentimiento (umbrales estándar)
    for source in summary["source"]:
        mask = df["source"] == source
        scores = df.loc[mask, "sentiment_score"]
        total = len(scores)

        summary.loc[summary["source"] == source, "pct_positive"] = round(
            (scores > 0.05).sum() / total * 100, 1
        )
        summary.loc[summary["source"] == source, "pct_neutral"] = round(
            ((scores >= -0.05) & (scores <= 0.05)).sum() / total * 100, 1
        )
        summary.loc[summary["source"] == source, "pct_negative"] = round(
            (scores < -0.05).sum() / total * 100, 1
        )

    # Renombrar source para presentación
    summary["source"] = summary["source"].map(
        lambda s: SOURCE_LABELS.get(s, s)
    )

    print(f"  Groups: {len(summary)}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# TABLA 4: RESULTADOS DE PRUEBAS ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────

def create_statistical_tests_table(test_results: dict) -> pd.DataFrame:
    """Genera tabla con todos los resultados de pruebas estadísticas.

    Consolida pruebas de Kruskal-Wallis, Dunn, Mann-Whitney, etc.
    en una tabla unificada con formato para publicación.

    Parameters
    ----------
    test_results : dict
        Diccionario con resultados de las distintas pruebas. Esperado:
        - 'kruskal_wallis': dict con H, p, eta²
        - 'dunn_posthoc': pd.DataFrame (opcional)
        - 'mann_whitney': dict {comparison: result}

    Returns
    -------
    pd.DataFrame
        Tabla con columnas: test_name, comparison, statistic,
        p_value, effect_size, significance.
    """
    print("\n─── Creating Statistical Tests Table ───")

    rows = []

    # Kruskal-Wallis
    kw = test_results.get("kruskal_wallis", {})
    if kw:
        eta_sq = test_results.get("eta_squared", "—")
        rows.append({
            "test_name": "Kruskal-Wallis",
            "comparison": "All groups",
            "statistic": f"H = {kw.get('H_statistic', '—')}",
            "p_value": kw.get("p_value", np.nan),
            "effect_size": f"η² = {eta_sq}",
            "significance": kw.get("significance", "—"),
        })

    # Dunn post-hoc
    dunn = test_results.get("dunn_posthoc", None)
    if dunn is not None and isinstance(dunn, pd.DataFrame):
        for _, row in dunn.iterrows():
            rows.append({
                "test_name": "Dunn (post-hoc)",
                "comparison": f"{row['group_1']} vs {row['group_2']}",
                "statistic": f"Z = {row['z_statistic']}",
                "p_value": row["p_adjusted"],
                "effect_size": "—",
                "significance": row["significance"],
            })

    # Mann-Whitney (bilingüe u otros)
    mw = test_results.get("mann_whitney", {})
    for comparison, result in mw.items():
        rows.append({
            "test_name": "Mann-Whitney U",
            "comparison": comparison,
            "statistic": f"U = {result.get('U_statistic', '—')}",
            "p_value": result.get("p_value", np.nan),
            "effect_size": f"r = {result.get('effect_size_r', '—')}",
            "significance": result.get("significance", "—"),
        })

    # Chi-square
    chi = test_results.get("chi_square", {})
    if chi:
        rows.append({
            "test_name": "Chi-Square",
            "comparison": chi.get("comparison", "—"),
            "statistic": f"χ² = {chi.get('chi2', '—')}",
            "p_value": chi.get("p_value", np.nan),
            "effect_size": f"V = {chi.get('cramers_v', '—')}",
            "significance": chi.get("significance", "—"),
        })

    table = pd.DataFrame(rows)

    # Formatear p-value para presentación
    if len(table) > 0:
        table["p_value_formatted"] = table["p_value"].apply(
            lambda p: f"{p:.6f}" if pd.notna(p) else "—"
        )

    print(f"  Tests included: {len(table)}")
    return table


# ─────────────────────────────────────────────────────────────────────
# TABLA 5: MÉTRICAS DE RED
# ─────────────────────────────────────────────────────────────────────

def create_network_metrics_table(network_metrics: dict) -> pd.DataFrame:
    """Genera tabla de métricas de red por grupo de discurso.

    Incluye métricas a nivel de red: nodos, aristas, densidad,
    clustering, diámetro, centralización, etc.

    Parameters
    ----------
    network_metrics : dict
        Diccionario {source_name: dict_of_metrics}.
        Cada dict_of_metrics puede contener:
        - 'n_nodes', 'n_edges', 'density', 'avg_clustering',
        - 'avg_degree', 'diameter', 'modularity', etc.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por fuente y columnas = métricas.
    """
    print("\n─── Creating Network Metrics Table ───")

    rows = []
    for source, metrics in network_metrics.items():
        row = {"source": SOURCE_LABELS.get(source, source)}
        row.update(metrics)
        rows.append(row)

    table = pd.DataFrame(rows)

    # Renombrar columnas para presentación
    col_rename = {
        "n_nodes": "Nodes",
        "n_edges": "Edges",
        "density": "Density",
        "avg_clustering": "Avg. Clustering",
        "avg_degree": "Avg. Degree",
        "diameter": "Diameter",
        "modularity": "Modularity",
    }
    table = table.rename(columns=col_rename)

    # Redondear columnas numéricas
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(4)

    print(f"  Sources: {len(table)} | Metrics: {len(table.columns) - 1}")
    return table


# ─────────────────────────────────────────────────────────────────────
# EXPORTACIÓN
# ─────────────────────────────────────────────────────────────────────

def save_table_latex(
    df: pd.DataFrame,
    output_path: Path,
    caption: str = "",
    label: str = "",
) -> None:
    """Exporta un DataFrame como tabla LaTeX para publicación.

    Genera código LaTeX con formato booktabs (\\toprule, \\midrule,
    \\bottomrule) listo para incluir en el manuscrito.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla a exportar.
    output_path : Path
        Ruta del archivo .tex de salida.
    caption : str
        Caption de la tabla en LaTeX.
    label : str
        Label para referencia cruzada (\\ref{}).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not label:
        label = f"tab:{output_path.stem}"

    latex_str = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=True,
        column_format="l" + "c" * (len(df.columns) - 1),
    )

    # Agregar booktabs si no están ya
    latex_str = latex_str.replace("\\hline", "")

    output_path.write_text(latex_str, encoding="utf-8")
    print(f"  ✓ LaTeX table saved: {output_path}")


def save_table_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Exporta un DataFrame como CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla a exportar.
    output_path : Path
        Ruta del archivo .csv de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSV table saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TABLES MODULE — Demo with synthetic data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ── Datos sintéticos ──
    n = 300
    df_test = pd.DataFrame({
        "source": (["academic"] * 100 + ["political"] * 100
                   + ["public"] * 100),
        "language": np.random.choice(["en", "es"], n),
        "sentiment_score": np.concatenate([
            np.random.normal(0.15, 0.25, 100),
            np.random.normal(0.35, 0.20, 100),
            np.random.normal(-0.05, 0.35, 100),
        ]),
        "clean_text": [f"word{i} text sample doc" for i in range(n)],
        "year": np.random.choice(range(2015, 2026), n),
    })

    # 1. Corpus summary
    print("\n── Table 1: Corpus Summary ──")
    corpus_tbl = create_corpus_summary_table(df_test)
    print(corpus_tbl.to_string(index=False))
    save_table_csv(corpus_tbl, TABLES_DIR / "corpus_summary.csv")
    save_table_latex(
        corpus_tbl, TABLES_DIR / "corpus_summary.tex",
        caption="Descriptive statistics of the sustainability discourse corpus."
    )

    # 2. Topic summary
    print("\n── Table 2: Topic Summary ──")
    topic_results = {
        "topic_keywords": {
            0: ["sustainability", "climate", "energy", "policy", "global"],
            1: ["development", "social", "economic", "growth", "urban"],
            2: ["biodiversity", "ecosystem", "conservation", "species", "land"],
        },
        "topic_distributions": {
            "academic": [0.5, 0.3, 0.2],
            "political": [0.4, 0.4, 0.2],
            "public": [0.3, 0.2, 0.5],
        },
    }
    topic_tbl = create_topic_summary_table(topic_results)
    print(topic_tbl.to_string(index=False))
    save_table_csv(topic_tbl, TABLES_DIR / "topic_summary.csv")

    # 3. Sentiment summary
    print("\n── Table 3: Sentiment Summary ──")
    sent_tbl = create_sentiment_summary_table(df_test)
    print(sent_tbl.to_string(index=False))
    save_table_csv(sent_tbl, TABLES_DIR / "sentiment_summary.csv")

    # 4. Statistical tests
    print("\n── Table 4: Statistical Tests ──")
    test_results = {
        "kruskal_wallis": {
            "H_statistic": 45.32,
            "p_value": 0.000001,
            "significance": "***",
        },
        "eta_squared": 0.152,
        "dunn_posthoc": pd.DataFrame([
            {"group_1": "academic", "group_2": "political",
             "z_statistic": 3.21, "p_adjusted": 0.004,
             "significance": "**"},
            {"group_1": "academic", "group_2": "public",
             "z_statistic": 5.67, "p_adjusted": 0.00001,
             "significance": "***"},
            {"group_1": "political", "group_2": "public",
             "z_statistic": 4.12, "p_adjusted": 0.0001,
             "significance": "***"},
        ]),
    }
    stat_tbl = create_statistical_tests_table(test_results)
    print(stat_tbl.to_string(index=False))
    save_table_csv(stat_tbl, TABLES_DIR / "statistical_tests.csv")

    # 5. Network metrics
    print("\n── Table 5: Network Metrics ──")
    net_metrics = {
        "academic": {"n_nodes": 150, "n_edges": 420, "density": 0.037,
                     "avg_clustering": 0.45, "avg_degree": 5.6},
        "political": {"n_nodes": 120, "n_edges": 350, "density": 0.049,
                      "avg_clustering": 0.52, "avg_degree": 5.8},
        "public": {"n_nodes": 200, "n_edges": 380, "density": 0.019,
                   "avg_clustering": 0.31, "avg_degree": 3.8},
    }
    net_tbl = create_network_metrics_table(net_metrics)
    print(net_tbl.to_string(index=False))
    save_table_csv(net_tbl, TABLES_DIR / "network_metrics.csv")

    print("\n✓ All demo tables generated successfully.")
