"""
temporal.py — Análisis de evolución temporal del discurso de sustentabilidad
============================================================================

Examina cómo los temas y el sentimiento evolucionan a lo largo del tiempo
en los tres grupos de discurso, detecta puntos de cambio y anota eventos
históricos clave del movimiento de sustentabilidad.
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


# ─────────────────────────────────────────────────────────────────────
# PREVALENCIA DE TEMAS POR AÑO
# ─────────────────────────────────────────────────────────────────────

def compute_yearly_topic_prevalence(
    df: pd.DataFrame,
    topic_col: str = "topic_id",
) -> pd.DataFrame:
    """Calcula la prevalencia de cada tema por año y fuente.

    Para cada combinación (año, fuente), obtiene la proporción
    de documentos asignados a cada tema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'year', 'source', y topic_col.
    topic_col : str
        Nombre de la columna con IDs de temas asignados.

    Returns
    -------
    pd.DataFrame
        DataFrame pivotado con columnas MultiIndex (source, topic_id)
        e índice = year. Valores = proporción del tema en ese año/fuente.
    """
    print("\n─── Computing Yearly Topic Prevalence ───")

    required = {"year", "source", topic_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # Contar documentos por (año, fuente, tema)
    counts = (
        df.groupby(["year", "source", topic_col])
        .size()
        .reset_index(name="count")
    )

    # Total documentos por (año, fuente) para normalizar
    totals = (
        df.groupby(["year", "source"])
        .size()
        .reset_index(name="total")
    )

    merged = counts.merge(totals, on=["year", "source"])
    merged["prevalence"] = merged["count"] / merged["total"]

    # Pivotar para obtener tabla ancha
    pivoted = merged.pivot_table(
        index="year",
        columns=["source", topic_col],
        values="prevalence",
        fill_value=0.0,
    )

    print(f"  Years covered: {pivoted.index.min()} – {pivoted.index.max()}")
    print(f"  Shape: {pivoted.shape}")

    return pivoted


# ─────────────────────────────────────────────────────────────────────
# SENTIMIENTO POR AÑO
# ─────────────────────────────────────────────────────────────────────

def compute_yearly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el sentimiento promedio por año y fuente.

    Genera una serie temporal del sentimiento medio con intervalos
    de confianza (desviación estándar) para cada grupo de discurso.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'year', 'source', 'sentiment_score'.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: year, source, mean_sentiment,
        std_sentiment, median_sentiment, count.
    """
    print("\n─── Computing Yearly Sentiment ───")

    required = {"year", "source", "sentiment_score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    yearly = (
        df.groupby(["year", "source"])["sentiment_score"]
        .agg(
            mean_sentiment="mean",
            std_sentiment="std",
            median_sentiment="median",
            count="count",
        )
        .reset_index()
        .round(4)
    )

    yearly = yearly.sort_values(["source", "year"]).reset_index(drop=True)

    print(f"  Rows generated: {len(yearly)}")
    for src in yearly["source"].unique():
        n = yearly.loc[yearly["source"] == src, "count"].sum()
        print(f"  {src}: {n} documents total")

    return yearly


# ─────────────────────────────────────────────────────────────────────
# DETECCIÓN DE PUNTOS DE CAMBIO
# ─────────────────────────────────────────────────────────────────────

def detect_change_points(
    time_series: pd.Series,
    z_threshold: float = 1.5,
) -> list[tuple]:
    """Detecta puntos de cambio en una serie temporal usando z-scores.

    Identifica años donde el cambio interanual es inusualmente grande
    (supera z_threshold desviaciones estándar de la media de cambios).

    Parameters
    ----------
    time_series : pd.Series
        Serie temporal con índice numérico (años) y valores
        continuos (e.g., sentimiento promedio).
    z_threshold : float
        Umbral de z-score para considerar un cambio significativo.

    Returns
    -------
    list[tuple]
        Lista de tuplas (year, direction) donde direction es
        'increase' o 'decrease'.
    """
    print("\n─── Detecting Change Points ───")

    if len(time_series) < 3:
        print("  ⚠ Series too short for change point detection")
        return []

    # Calcular diferencias interanuales
    diffs = time_series.diff().dropna()

    if diffs.std() == 0:
        print("  ⚠ No variance in differences — no change points")
        return []

    # Z-scores de las diferencias
    z_scores = (diffs - diffs.mean()) / diffs.std()

    change_points = []
    for year, z in z_scores.items():
        if abs(z) > z_threshold:
            direction = "increase" if z > 0 else "decrease"
            change_points.append((year, direction))
            print(f"  Change point: {year} ({direction}, z={z:.2f})")

    if not change_points:
        print("  No significant change points detected")

    return change_points


# ─────────────────────────────────────────────────────────────────────
# ANOTACIÓN DE EVENTOS
# ─────────────────────────────────────────────────────────────────────

def annotate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega anotaciones de eventos clave de sustentabilidad.

    Mapea cada año del DataFrame al evento histórico correspondiente
    definido en KEY_EVENTS de la configuración.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'year'.

    Returns
    -------
    pd.DataFrame
        DataFrame original con columna adicional 'event' que contiene
        la descripción del evento (o NaN si no hay evento ese año).
    """
    print("\n─── Annotating Key Events ───")

    df = df.copy()
    df["event"] = df["year"].map(KEY_EVENTS)

    n_annotated = df["event"].notna().sum()
    print(f"  Rows annotated: {n_annotated} / {len(df)}")

    for year, event in sorted(KEY_EVENTS.items()):
        n = (df["year"] == year).sum()
        if n > 0:
            print(f"  {year}: {event} ({n} rows)")

    return df


# ─────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────

def run_temporal_analysis(df: pd.DataFrame) -> dict:
    """Ejecuta el pipeline completo de análisis temporal.

    Integra prevalencia de temas, evolución de sentimiento,
    detección de puntos de cambio y anotación de eventos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado con columnas 'year', 'source',
        'sentiment_score', y opcionalmente 'topic_id'.

    Returns
    -------
    dict
        Diccionario con:
        - 'yearly_sentiment': pd.DataFrame (series temporales)
        - 'yearly_topics': pd.DataFrame o None
        - 'change_points': dict {source: list[tuple]}
        - 'annotated_sentiment': pd.DataFrame (con eventos)
    """
    print("=" * 60)
    print("  TEMPORAL EVOLUTION ANALYSIS")
    print("=" * 60)

    results = {}

    # 1. Sentimiento por año
    yearly_sent = compute_yearly_sentiment(df)
    results["yearly_sentiment"] = yearly_sent

    out_path = COMPARISON_DIR / "yearly_sentiment.csv"
    yearly_sent.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")

    # 2. Prevalencia de temas (si la columna existe)
    if "topic_id" in df.columns:
        yearly_topics = compute_yearly_topic_prevalence(df)
        results["yearly_topics"] = yearly_topics

        out_path = COMPARISON_DIR / "yearly_topic_prevalence.csv"
        yearly_topics.to_csv(out_path)
        print(f"  ✓ Saved: {out_path}")
    else:
        results["yearly_topics"] = None
        print("\n  ⚠ Skipping topic prevalence — 'topic_id' not found")

    # 3. Detección de puntos de cambio por fuente
    change_points = {}
    for source in sorted(df["source"].unique()):
        print(f"\n  ── Change points for: {source} ──")
        mask = yearly_sent["source"] == source
        ts = yearly_sent.loc[mask].set_index("year")["mean_sentiment"]

        if len(ts) >= 3:
            cp = detect_change_points(ts)
            change_points[source] = cp
        else:
            change_points[source] = []
            print(f"  ⚠ Too few data points for {source}")

    results["change_points"] = change_points

    # 4. Anotación de eventos
    annotated = annotate_events(yearly_sent)
    results["annotated_sentiment"] = annotated

    out_path = COMPARISON_DIR / "yearly_sentiment_annotated.csv"
    annotated.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")

    # Resumen de puntos de cambio
    out_rows = []
    for source, cps in change_points.items():
        for year, direction in cps:
            out_rows.append({
                "source": source,
                "year": year,
                "direction": direction,
            })
    if out_rows:
        cp_df = pd.DataFrame(out_rows)
        out_path = COMPARISON_DIR / "change_points.csv"
        cp_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")

    print("\n" + "=" * 60)
    print("  ✓ Temporal analysis complete")
    print("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TEMPORAL MODULE — Demo with synthetic data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ── Datos sintéticos ──
    years = list(range(2015, 2026))
    rows = []
    for year in years:
        for source in ["academic", "political", "public"]:
            n = np.random.randint(10, 50)
            base_sent = {
                "academic": 0.1,
                "political": 0.3,
                "public": -0.05,
            }[source]
            # Simular tendencia temporal con ruido
            trend = (year - 2015) * 0.02
            for _ in range(n):
                rows.append({
                    "year": year,
                    "source": source,
                    "sentiment_score": np.random.normal(
                        base_sent + trend, 0.3
                    ),
                    "topic_id": np.random.randint(0, 5),
                })

    df_test = pd.DataFrame(rows)

    results = run_temporal_analysis(df_test)

    print(f"\n  Yearly sentiment shape: {results['yearly_sentiment'].shape}")
    print(f"  Change points found: "
          f"{sum(len(v) for v in results['change_points'].values())}")
    print("\n✓ Demo finished successfully.")
