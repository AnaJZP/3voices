"""
sentiment.py — Análisis de sentimiento multi-modelo
=====================================================

Módulo de análisis de sentimiento para el proyecto VOZ_SUS.
Utiliza tres modelos especializados según el idioma:

  - Multilingüe (fallback): nlptown/bert-base-multilingual-uncased-sentiment
  - Inglés:  cardiffnlp/twitter-roberta-base-sentiment-latest
  - Español: pysentimiento (sentiment, lang='es')

Todos los modelos se ejecutan en CPU con truncamiento a 512 tokens.

Autor: VOZ_SUS Project
"""

# ── Forzar CPU ──────────────────────────────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── stdlib ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

# ── local / config ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *  # noqa: E402, F403
from constants import normalize_group_series

# ─────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MÓDULO
# ─────────────────────────────────────────────────────────────────────

SENTIMENT_RESULTS_DIR = RESULTS_DIR / "sentiment"
SENTIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Modelos de sentimiento
MODEL_MULTILINGUAL = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_EN = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Longitud máxima de tokens (evitar OOM en CPU)
MAX_TOKEN_LENGTH = 512

# Tamaño de lote por defecto
DEFAULT_BATCH_SIZE = 32

# Mapeo de estrellas a categorías (para modelo multilingüe)
_STAR_TO_CATEGORY = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────

def _safe_truncate(text: str, max_chars: int = 2000) -> str:
    """Trunca texto largo para evitar exceder el límite de tokens.

    Parameters
    ----------
    text : str
        Texto a truncar.
    max_chars : int
        Máximo de caracteres (aproximación conservadora de 512 tokens).

    Returns
    -------
    str
        Texto truncado.
    """
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


def _process_in_batches(
    pipe,
    texts: list[str],
    batch_size: int,
    desc: str = "Sentiment",
) -> list[dict]:
    """Procesa textos en lotes a través de un pipeline de HuggingFace.

    Parameters
    ----------
    pipe : transformers.Pipeline
        Pipeline de clasificación de texto.
    texts : list[str]
        Lista de textos a clasificar.
    batch_size : int
        Tamaño de lote.
    desc : str
        Descripción para la barra de progreso.

    Returns
    -------
    list[dict]
        Lista de predicciones (label, score).
    """
    all_results = []
    n_total = len(texts)

    for start in tqdm(range(0, n_total, batch_size), desc=f"  {desc}"):
        batch = texts[start : start + batch_size]
        try:
            preds = pipe(batch, truncation=True, max_length=MAX_TOKEN_LENGTH)
            all_results.extend(preds)
        except Exception as e:
            print(f"  ⚠ Batch error at index {start}: {e}")
            # Procesar uno a uno como fallback
            for text in batch:
                try:
                    pred = pipe(text, truncation=True, max_length=MAX_TOKEN_LENGTH)
                    all_results.extend(pred)
                except Exception:
                    all_results.append({"label": "error", "score": 0.0})

    return all_results


# ─────────────────────────────────────────────────────────────────────
# MODELOS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def analyze_sentiment_multilingual(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Análisis de sentimiento multilingüe (1-5 estrellas).

    Usa el modelo nlptown/bert-base-multilingual-uncased-sentiment
    que clasifica en 5 niveles (1-5 estrellas).

    Parameters
    ----------
    texts : list[str]
        Lista de textos a analizar.
    batch_size : int
        Tamaño de lote para procesamiento.

    Returns
    -------
    pd.DataFrame
        Columnas: text, sentiment_label, sentiment_score, sentiment_category
    """
    print(f"\n  Multilingual sentiment analysis | {len(texts):,} texts")
    print(f"  Model: {MODEL_MULTILINGUAL}")

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_MULTILINGUAL,
        device=-1,  # CPU
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    # Truncar textos largos
    safe_texts = [_safe_truncate(t) for t in texts]

    results = _process_in_batches(pipe, safe_texts, batch_size, desc="Multilingual")

    df = pd.DataFrame({
        "text": texts,
        "sentiment_label": [r["label"] for r in results],
        "sentiment_score": [round(r["score"], 4) for r in results],
    })

    # Mapear a categoría positivo/neutro/negativo
    df["sentiment_category"] = df["sentiment_label"].map(_STAR_TO_CATEGORY)
    df["sentiment_category"] = df["sentiment_category"].fillna("neutral")

    print(f"  ✓ Distribution: {df['sentiment_category'].value_counts().to_dict()}")
    return df


def analyze_sentiment_en(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Análisis de sentimiento para textos en inglés.

    Usa el modelo cardiffnlp/twitter-roberta-base-sentiment-latest
    con tres clases: positive, neutral, negative.

    Parameters
    ----------
    texts : list[str]
        Lista de textos en inglés.
    batch_size : int
        Tamaño de lote.

    Returns
    -------
    pd.DataFrame
        Columnas: text, sentiment_label, sentiment_score, sentiment_category
    """
    print(f"\n  English sentiment analysis | {len(texts):,} texts")
    print(f"  Model: {MODEL_EN}")

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_EN,
        device=-1,  # CPU
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    safe_texts = [_safe_truncate(t) for t in texts]
    results = _process_in_batches(pipe, safe_texts, batch_size, desc="English")

    df = pd.DataFrame({
        "text": texts,
        "sentiment_label": [r["label"] for r in results],
        "sentiment_score": [round(r["score"], 4) for r in results],
    })

    # Normalizar etiquetas a categoría
    label_map = {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "POSITIVE": "positive",
        "NEUTRAL": "neutral",
        "NEGATIVE": "negative",
    }
    df["sentiment_category"] = df["sentiment_label"].map(label_map)
    df["sentiment_category"] = df["sentiment_category"].fillna("neutral")

    print(f"  ✓ Distribution: {df['sentiment_category'].value_counts().to_dict()}")
    return df


def analyze_sentiment_es(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Análisis de sentimiento para textos en español.

    Usa el modelo multilingüe nlptown/bert-base-multilingual-uncased-sentiment
    que ya está descargado. Mapea estrellas (1-5) a negative/neutral/positive.

    Parameters
    ----------
    texts : list[str]
        Lista de textos en español.
    batch_size : int
        Tamaño de lote.

    Returns
    -------
    pd.DataFrame
        Columnas: text, sentiment_label, sentiment_score, sentiment_category
    """
    print(f"\n  Spanish sentiment analysis | {len(texts):,} texts")
    print(f"  Model: {MODEL_MULTILINGUAL}")

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_MULTILINGUAL,
        tokenizer=MODEL_MULTILINGUAL,
        device=-1,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    labels = []
    scores = []
    n_total = len(texts)

    for i in tqdm(range(0, n_total, batch_size), desc="  Spanish"):
        batch = texts[i : i + batch_size]
        safe_batch = [_safe_truncate(t) for t in batch]
        try:
            results = pipe(safe_batch, batch_size=batch_size)
            for r in results:
                labels.append(r["label"])
                scores.append(round(r["score"], 4))
        except Exception as e:
            print(f"  ⚠ Error in batch {i}: {e}")
            for _ in batch:
                labels.append("3 stars")
                scores.append(0.0)

    df = pd.DataFrame({
        "text": texts,
        "sentiment_label": labels,
        "sentiment_score": scores,
    })

    # Mapear estrellas a categorías estándar
    star_map = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive",
    }
    df["sentiment_category"] = df["sentiment_label"].map(star_map)
    df["sentiment_category"] = df["sentiment_category"].fillna("neutral")

    # Normalizar labels
    df["sentiment_label"] = df["sentiment_category"]

    print(f"  ✓ Distribution: {df['sentiment_category'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────
# ORQUESTACIÓN
# ─────────────────────────────────────────────────────────────────────

def run_sentiment_analysis(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    lang_col: str = "language",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Ejecuta análisis de sentimiento aplicando el modelo apropiado por idioma.

    Asigna automáticamente el modelo según la columna de idioma:
      - 'en' / 'english' → modelo inglés (Cardiff)
      - 'es' / 'spanish' → modelo español (pysentimiento)
      - otros            → modelo multilingüe (nlptown)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con textos y columna de idioma.
    text_col : str
        Nombre de la columna de texto.
    lang_col : str
        Nombre de la columna de idioma.
    batch_size : int
        Tamaño de lote.

    Returns
    -------
    pd.DataFrame
        DataFrame original con columnas añadidas:
        sentiment_label, sentiment_score, sentiment_category
    """
    print(f"\n{'='*60}")
    print(f"  Sentiment Analysis — {len(df):,} documents")
    print(f"{'='*60}")

    result_df = df.copy()
    result_df["sentiment_label"] = pd.Series("", index=df.index, dtype="object")
    result_df["sentiment_score"] = np.nan
    result_df["sentiment_category"] = pd.Series("", index=df.index, dtype="object")

    # Determinar idiomas presentes
    if lang_col in df.columns:
        lang_values = df[lang_col].str.lower().fillna("unknown")
    else:
        print(f"  ⚠ Column '{lang_col}' not found — using multilingual model for all")
        lang_values = pd.Series(["multilingual"] * len(df))

    # ── Procesar inglés ─────────────────────────────────────────────
    en_mask = lang_values.isin(["en", "english", "eng"])
    if en_mask.sum() > 0:
        en_texts = df.loc[en_mask, text_col].tolist()
        en_results = analyze_sentiment_en(en_texts, batch_size=batch_size)
        result_df.loc[en_mask, "sentiment_label"] = en_results["sentiment_label"].values
        result_df.loc[en_mask, "sentiment_score"] = en_results["sentiment_score"].values
        result_df.loc[en_mask, "sentiment_category"] = en_results["sentiment_category"].values

    # ── Procesar español ────────────────────────────────────────────
    es_mask = lang_values.isin(["es", "spanish", "spa"])
    if es_mask.sum() > 0:
        es_texts = df.loc[es_mask, text_col].tolist()
        es_results = analyze_sentiment_es(es_texts, batch_size=batch_size)
        result_df.loc[es_mask, "sentiment_label"] = es_results["sentiment_label"].values
        result_df.loc[es_mask, "sentiment_score"] = es_results["sentiment_score"].values
        result_df.loc[es_mask, "sentiment_category"] = es_results["sentiment_category"].values

    # ── Procesar otros idiomas con modelo multilingüe ───────────────
    other_mask = ~en_mask & ~es_mask
    if other_mask.sum() > 0:
        other_texts = df.loc[other_mask, text_col].tolist()
        other_results = analyze_sentiment_multilingual(other_texts, batch_size=batch_size)
        result_df.loc[other_mask, "sentiment_label"] = other_results["sentiment_label"].values
        result_df.loc[other_mask, "sentiment_score"] = other_results["sentiment_score"].values
        result_df.loc[other_mask, "sentiment_category"] = other_results["sentiment_category"].values

    # ── Guardar resultados ──────────────────────────────────────────
    output_path = SENTIMENT_RESULTS_DIR / "sentiment_results.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ Sentiment results saved → {output_path.name}")
    print(f"  Overall distribution:")
    print(f"    {result_df['sentiment_category'].value_counts().to_dict()}")

    return result_df


def compute_sentiment_stats(
    df: pd.DataFrame,
    source_col: str = "source",
    lang_col: str = "language",
) -> pd.DataFrame:
    """Calcula estadísticas resumidas de sentimiento por fuente e idioma.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas sentiment_score y sentiment_category.
    source_col : str
        Columna de agrupación por fuente.
    lang_col : str
        Columna de agrupación por idioma.

    Returns
    -------
    pd.DataFrame
        Tabla de estadísticas con: mean, std, median por grupo,
        y distribución de categorías.
    """
    print(f"\n  Computing sentiment statistics ...")

    # Determinar columnas de agrupación disponibles
    group_cols = []
    if source_col in df.columns:
        group_cols.append(source_col)
    if lang_col in df.columns:
        group_cols.append(lang_col)

    if not group_cols:
        group_cols = ["sentiment_category"]

    # ── Estadísticas numéricas ──────────────────────────────────────
    stats = (
        df.groupby(group_cols)["sentiment_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    stats.columns = group_cols + ["n_docs", "mean_score", "std_score", "median_score"]

    # Redondear valores
    for col in ["mean_score", "std_score", "median_score"]:
        stats[col] = stats[col].round(4)

    # ── Distribución de categorías por grupo ────────────────────────
    cat_dist = (
        df.groupby(group_cols)["sentiment_category"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )

    # Renombrar columnas de porcentaje
    for col in ["positive", "neutral", "negative"]:
        if col in cat_dist.columns:
            cat_dist[col] = (cat_dist[col] * 100).round(2)
            cat_dist = cat_dist.rename(columns={col: f"pct_{col}"})

    # Unir estadísticas
    stats = stats.merge(cat_dist, on=group_cols, how="left")

    # Guardar
    output_path = SENTIMENT_RESULTS_DIR / "sentiment_stats.csv"
    stats.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Stats saved → {output_path.name}")
    print(f"\n{stats.to_string(index=False)}")

    return stats


# ─────────────────────────────────────────────────────────────────────
# MAIN — ejecución independiente para pruebas
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VOZ_SUS — Sentiment Analysis Module (standalone test)")
    print("=" * 60)

    # Buscar corpus procesado
    corpus_path = PROCESSED_DIR / "corpus_clean.csv"
    if not corpus_path.exists():
        corpus_path = PROCESSED_DIR / "corpus_clean.parquet"

    if not corpus_path.exists():
        print(f"\n  ✗ Corpus not found at {PROCESSED_DIR}")
        print("  Expected: corpus_clean.csv or corpus_clean.parquet")
        print("  Run the preprocessing pipeline first.")
        sys.exit(1)

    print(f"\n  Loading corpus from: {corpus_path.name}")
    if corpus_path.suffix == ".parquet":
        df = pd.read_parquet(corpus_path)
    else:
        df = pd.read_csv(corpus_path)

    print(f"  Loaded {len(df):,} documents")

    # Normalize legacy 'political' → 'institutional'
    if "source" in df.columns:
        df["source"] = normalize_group_series(df["source"])

    # Determinar columna de texto
    text_col = "text_clean"
    if text_col not in df.columns:
        text_col = "text"

    # Filtrar vacíos
    df = df[df[text_col].notna() & (df[text_col].str.len() > 10)].reset_index(drop=True)
    print(f"  After filtering: {len(df):,} documents")

    # Ejecutar análisis
    result_df = run_sentiment_analysis(df, text_col=text_col)

    # Estadísticas
    stats = compute_sentiment_stats(result_df)

    print("\n  ✓ Sentiment analysis complete!")
