"""
bilingual.py — Comparación bilingüe (EN vs ES) del discurso de sustentabilidad
===============================================================================

Analiza diferencias entre contenido en inglés y español dentro de
cada grupo de discurso (académico, político, ciudadano). Compara
distribuciones de temas, sentimiento y vocabulario clave.
"""

# ── Stdlib ────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Local ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *
from src.comparison.statistics import (
    mann_whitney_test,
    jensen_shannon_divergence,
    format_significance,
)


# ─────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────

LANGUAGES = ["en", "es"]
SOURCE_GROUPS = ["academic", "political", "public"]


# ─────────────────────────────────────────────────────────────────────
# TEMAS POR IDIOMA
# ─────────────────────────────────────────────────────────────────────

def compare_topics_by_language(
    df: pd.DataFrame,
    topic_results: dict,
) -> pd.DataFrame:
    """Compara distribuciones de temas entre EN y ES dentro de cada grupo.

    Para cada fuente de discurso, calcula la distribución de temas
    por separado para documentos en inglés y español, y cuantifica
    la divergencia entre ambas distribuciones.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source', 'language', y 'topic_id'.
    topic_results : dict
        Resultados del modelado de temas (no usado directamente aquí,
        pero disponible para acceder a metadatos de temas).

    Returns
    -------
    pd.DataFrame
        Tabla con columnas: source, language, topic_id, prevalence,
        y una fila por cada combinación fuente × idioma × tema.
    """
    print("\n─── Comparing Topics by Language ───")

    required = {"source", "language", "topic_id"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    results_rows = []

    for source in tqdm(SOURCE_GROUPS, desc="  Sources"):
        mask_src = df["source"] == source

        for lang in LANGUAGES:
            mask_lang = df["language"] == lang
            subset = df[mask_src & mask_lang]

            if len(subset) == 0:
                print(f"  ⚠ No data for {source}/{lang}")
                continue

            # Distribución de temas
            topic_counts = subset["topic_id"].value_counts(normalize=True)

            for topic_id, prevalence in topic_counts.items():
                results_rows.append({
                    "source": source,
                    "language": lang,
                    "topic_id": topic_id,
                    "prevalence": round(prevalence, 4),
                    "n_docs": len(subset),
                })

    results_df = pd.DataFrame(results_rows)

    # Calcular JSD por fuente entre EN y ES
    print("\n  Topic divergence EN vs ES:")
    for source in SOURCE_GROUPS:
        src_data = results_df[results_df["source"] == source]
        en_data = src_data[src_data["language"] == "en"]
        es_data = src_data[src_data["language"] == "es"]

        if len(en_data) == 0 or len(es_data) == 0:
            print(f"  {source}: insufficient data")
            continue

        # Alinear distribuciones por topic_id
        all_topics = sorted(
            set(en_data["topic_id"]) | set(es_data["topic_id"])
        )
        en_dist = np.array([
            en_data.loc[
                en_data["topic_id"] == t, "prevalence"
            ].values[0] if t in en_data["topic_id"].values else 0.0
            for t in all_topics
        ])
        es_dist = np.array([
            es_data.loc[
                es_data["topic_id"] == t, "prevalence"
            ].values[0] if t in es_data["topic_id"].values else 0.0
            for t in all_topics
        ])

        jsd = jensen_shannon_divergence(en_dist, es_dist)
        print(f"  {source}: JSD(EN vs ES) = {jsd:.4f}")

    return results_df


# ─────────────────────────────────────────────────────────────────────
# SENTIMIENTO POR IDIOMA
# ─────────────────────────────────────────────────────────────────────

def compare_sentiment_by_language(df: pd.DataFrame) -> dict:
    """Compara sentimiento EN vs ES dentro de cada grupo de discurso.

    Usa la prueba U de Mann-Whitney para cada fuente por separado,
    evaluando si el sentimiento difiere significativamente entre
    documentos en inglés y español.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source', 'language', 'sentiment_score'.

    Returns
    -------
    dict
        Diccionario {source: test_result_dict} con los resultados
        de Mann-Whitney para cada grupo de discurso.
    """
    print("\n─── Comparing Sentiment by Language ───")

    required = {"source", "language", "sentiment_score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    results = {}

    for source in SOURCE_GROUPS:
        mask_src = df["source"] == source
        en_scores = df.loc[
            mask_src & (df["language"] == "en"), "sentiment_score"
        ].dropna().values
        es_scores = df.loc[
            mask_src & (df["language"] == "es"), "sentiment_score"
        ].dropna().values

        print(f"\n  {source.upper()}:")
        print(f"    EN: n={len(en_scores)}, "
              f"mean={np.mean(en_scores):.4f}" if len(en_scores) > 0
              else f"    EN: n=0")
        print(f"    ES: n={len(es_scores)}, "
              f"mean={np.mean(es_scores):.4f}" if len(es_scores) > 0
              else f"    ES: n=0")

        mw = mann_whitney_test(en_scores, es_scores)
        results[source] = mw

        print(f"    Mann-Whitney U={mw['U_statistic']}, "
              f"p={mw['p_value']:.6f} {mw['significance']}")
        print(f"    Effect size r={mw['effect_size_r']}")

    return results


# ─────────────────────────────────────────────────────────────────────
# KEYWORDS POR IDIOMA
# ─────────────────────────────────────────────────────────────────────

def compare_keywords_by_language(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    top_n: int = 50,
) -> dict:
    """Compara las keywords más frecuentes EN vs ES por grupo.

    Extrae los tokens más frecuentes para cada combinación de
    fuente × idioma y calcula la similitud Jaccard entre pares.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source', 'language', y text_col.
    text_col : str
        Nombre de la columna con texto limpio.
    top_n : int
        Número de keywords top a comparar.

    Returns
    -------
    dict
        Diccionario con:
        - 'keywords': dict {(source, lang): list[str]}
        - 'jaccard_per_source': dict {source: float}
        - 'unique_en': dict {source: list[str]} (solo en EN)
        - 'unique_es': dict {source: list[str]} (solo en ES)
    """
    print(f"\n─── Comparing Keywords by Language (top {top_n}) ───")

    required = {"source", "language", text_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    keywords = {}
    jaccard_per_source = {}
    unique_en = {}
    unique_es = {}

    for source in tqdm(SOURCE_GROUPS, desc="  Sources"):
        for lang in LANGUAGES:
            mask = (df["source"] == source) & (df["language"] == lang)
            texts = df.loc[mask, text_col].dropna()

            if len(texts) == 0:
                keywords[(source, lang)] = []
                continue

            # Tokenizar y contar frecuencias
            all_tokens = " ".join(texts.values).split()
            # Filtrar tokens muy cortos
            all_tokens = [t for t in all_tokens if len(t) > 2]
            counter = Counter(all_tokens)
            top_words = [word for word, _ in counter.most_common(top_n)]
            keywords[(source, lang)] = top_words

        # Calcular Jaccard entre EN y ES para esta fuente
        en_set = set(keywords.get((source, "en"), []))
        es_set = set(keywords.get((source, "es"), []))

        intersection = len(en_set & es_set)
        union = len(en_set | es_set)
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_per_source[source] = round(jaccard, 4)

        # Palabras únicas por idioma
        unique_en[source] = sorted(en_set - es_set)[:20]
        unique_es[source] = sorted(es_set - en_set)[:20]

        print(f"\n  {source}: Jaccard(EN, ES) = {jaccard:.4f}")
        print(f"    Shared: {intersection} | Union: {union}")
        if unique_en[source]:
            print(f"    Only EN: {', '.join(unique_en[source][:10])}")
        if unique_es[source]:
            print(f"    Only ES: {', '.join(unique_es[source][:10])}")

    return {
        "keywords": keywords,
        "jaccard_per_source": jaccard_per_source,
        "unique_en": unique_en,
        "unique_es": unique_es,
    }


# ─────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────

def run_bilingual_analysis(
    df: pd.DataFrame,
    topic_results: dict,
) -> dict:
    """Ejecuta el pipeline completo de análisis bilingüe.

    Compara temas, sentimiento y vocabulario entre documentos en
    inglés y español dentro de cada grupo de discurso.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado con columnas 'source', 'language',
        'sentiment_score', 'clean_text', y opcionalmente 'topic_id'.
    topic_results : dict
        Resultados del modelado de temas.

    Returns
    -------
    dict
        Diccionario con todos los resultados bilingües:
        - 'topics_by_language': pd.DataFrame
        - 'sentiment_by_language': dict
        - 'keywords_by_language': dict
    """
    print("=" * 60)
    print("  BILINGUAL COMPARISON ANALYSIS (EN vs ES)")
    print("=" * 60)

    results = {}

    # 1. Distribución de temas por idioma
    if "topic_id" in df.columns:
        topics_lang = compare_topics_by_language(df, topic_results)
        results["topics_by_language"] = topics_lang

        out_path = COMPARISON_DIR / "topics_by_language.csv"
        topics_lang.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
    else:
        results["topics_by_language"] = None
        print("\n  ⚠ Skipping topic comparison — 'topic_id' not found")

    # 2. Sentimiento por idioma
    if "sentiment_score" in df.columns and "language" in df.columns:
        sent_lang = compare_sentiment_by_language(df)
        results["sentiment_by_language"] = sent_lang

        # Guardar resumen como CSV
        rows = []
        for source, test_res in sent_lang.items():
            row = {"source": source}
            row.update(test_res)
            rows.append(row)
        sent_df = pd.DataFrame(rows)
        out_path = COMPARISON_DIR / "sentiment_by_language.csv"
        sent_df.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
    else:
        results["sentiment_by_language"] = None
        print("\n  ⚠ Skipping sentiment comparison — columns missing")

    # 3. Keywords por idioma
    if "clean_text" in df.columns and "language" in df.columns:
        kw_lang = compare_keywords_by_language(df)
        results["keywords_by_language"] = kw_lang

        # Guardar Jaccard como CSV
        jaccard_df = pd.DataFrame([
            {"source": src, "jaccard_en_es": val}
            for src, val in kw_lang["jaccard_per_source"].items()
        ])
        out_path = COMPARISON_DIR / "keyword_jaccard_by_language.csv"
        jaccard_df.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
    else:
        results["keywords_by_language"] = None
        print("\n  ⚠ Skipping keyword comparison — columns missing")

    print("\n" + "=" * 60)
    print("  ✓ Bilingual analysis complete")
    print("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BILINGUAL MODULE — Demo with synthetic data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ── Datos sintéticos bilingües ──
    rows = []
    for source in SOURCE_GROUPS:
        for lang in LANGUAGES:
            n = np.random.randint(30, 80)
            base_sent = {
                "academic": 0.2,
                "political": 0.4,
                "public": -0.1,
            }[source]
            lang_offset = 0.05 if lang == "en" else -0.05

            for _ in range(n):
                if lang == "en":
                    text = "sustainability climate energy policy carbon"
                else:
                    text = "sustentabilidad clima energía política carbono"
                rows.append({
                    "source": source,
                    "language": lang,
                    "sentiment_score": np.random.normal(
                        base_sent + lang_offset, 0.3
                    ),
                    "clean_text": text,
                    "topic_id": np.random.randint(0, 5),
                })

    df_test = pd.DataFrame(rows)

    results = run_bilingual_analysis(df_test, topic_results={})

    print(f"\n  Topics by language shape: "
          f"{results['topics_by_language'].shape if results['topics_by_language'] is not None else 'N/A'}")
    print("\n✓ Demo finished successfully.")
