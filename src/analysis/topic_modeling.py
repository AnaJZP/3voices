"""
topic_modeling.py — Modelado de tópicos con BERTopic (CPU-optimizado)
=====================================================================

Módulo para descubrir tópicos latentes en el discurso de sustentabilidad
usando BERTopic con embeddings multilingües.  Diseñado para ejecución
en CPU con parámetros conservadores de memoria.

Tres niveles de análisis:
  1. Modelado por grupo (académico, político, ciudadano)
  2. Modelado del corpus combinado
  3. Análisis temporal de tópicos

Autor: VOZ_SUS Project
"""

# ── Forzar ejecución en CPU ─────────────────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── stdlib ──────────────────────────────────────────────────────────
import sys
import pickle
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# ── local / config ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *  # noqa: E402, F403

# Alias: config usa RANDOM_STATE, este módulo usa RANDOM_SEED
try:
    RANDOM_SEED
except NameError:
    RANDOM_SEED = RANDOM_STATE

# ─────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MÓDULO
# ─────────────────────────────────────────────────────────────────────

TOPIC_RESULTS_DIR = RESULTS_DIR / "topics"
TOPIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_CACHE_DIR = PROCESSED_DIR / "embeddings_cache"
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Modelo de embeddings multilingüe liviano
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Tamaño de lote para pre-calcular embeddings (conservar memoria)
EMBEDDING_BATCH_SIZE = 256

# Umbral de documentos para activar procesamiento por lotes
LARGE_CORPUS_THRESHOLD = 50_000

# Stopwords combinadas (español + inglés) para CountVectorizer
_STOPWORDS_ES = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
    "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía",
    "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya",
    "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas",
    "estoy", "estás", "está", "estamos", "estáis", "están", "esté",
    "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará",
    "estaremos", "estaréis", "estarán", "estaría", "estarías",
    "estaríamos", "estaríais", "estarían", "estaba", "estabas",
    "estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo",
    "estuvimos", "estuvisteis", "estuvieron", "ser", "soy", "eres", "es",
    "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean",
    "seré", "serás", "será", "seremos", "seréis", "serán", "sería",
    "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos",
    "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis",
    "fueron", "haber", "he", "has", "ha", "hemos", "habéis", "han",
    "haya", "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás",
    "habrá", "habremos", "habréis", "habrán", "habría", "habrías",
    "habríamos", "habríais", "habrían", "había", "habías", "habíamos",
    "habíais", "habían", "hube", "hubiste", "hubo", "hubimos",
    "hubisteis", "hubieron", "tener", "tengo", "tienes", "tiene",
    "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos",
    "tengáis", "tengan", "tendré", "tendrás", "tendrá", "tendremos",
    "tendréis", "tendrán", "tendría", "tendrías", "tendríamos",
    "tendríais", "tendrían", "tenía", "tenías", "teníamos", "teníais",
    "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis",
    "tuvieron", "hacer", "hago", "haces", "hace", "hacemos", "hacéis",
    "hacen", "haga", "hagas", "hagamos", "hagáis", "hagan",
    "poder", "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "ir", "voy", "vas", "va", "vamos", "vais", "van",
    "decir", "digo", "dices", "dice", "decimos", "decís", "dicen",
    "ver", "veo", "ves", "ve", "vemos", "veis", "ven",
    "dar", "doy", "das", "da", "damos", "dais", "dan",
    "saber", "sé", "sabes", "sabe", "sabemos", "sabéis", "saben",
    "querer", "quiero", "quieres", "quiere", "queremos", "queréis", "quieren",
    "así", "bien", "aquí", "cada", "si", "después", "mejor", "puede",
    "tal", "vez", "ahora", "cual", "mientras", "mismo", "parte",
    "http", "https", "www", "com",
]

_STOPWORDS_EN = [
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "it's", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "being", "having", "doing", "about", "above", "after", "again",
    "against", "all", "any", "because", "before", "below", "between",
    "both", "during", "each", "few", "further", "here", "how", "into",
    "just", "more", "most", "no", "nor", "not", "only", "other", "out",
    "own", "same", "so", "some", "such", "than", "then", "there", "through",
    "too", "under", "until", "up", "very", "when", "where", "while", "why",
    "also", "however", "therefore", "thus", "although", "though",
    "http", "https", "www", "com",
]

COMBINED_STOPWORDS = list(set(_STOPWORDS_ES + _STOPWORDS_EN))


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────

def _get_embedding_model() -> SentenceTransformer:
    """Carga el modelo de embeddings configurado para CPU.

    Returns
    -------
    SentenceTransformer
        Modelo listo para codificar textos.
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return model


def _compute_or_load_embeddings(
    docs: list[str],
    label: str = "corpus",
) -> np.ndarray:
    """Pre-calcula embeddings o los carga de caché en disco.

    Parameters
    ----------
    docs : list[str]
        Lista de textos a codificar.
    label : str
        Etiqueta para el archivo de caché.

    Returns
    -------
    np.ndarray
        Matriz de embeddings (n_docs, dim).
    """
    cache_path = EMBEDDINGS_CACHE_DIR / f"embeddings_{label}.npy"

    if cache_path.exists():
        print(f"  Loading cached embeddings: {cache_path.name}")
        embeddings = np.load(str(cache_path))
        if embeddings.shape[0] == len(docs):
            print(f"  Cache hit — {embeddings.shape[0]} embeddings loaded")
            return embeddings
        print("  Cache mismatch — recomputing embeddings")

    model = _get_embedding_model()
    n_docs = len(docs)
    print(f"  Computing embeddings for {n_docs:,} documents ...")

    if n_docs > LARGE_CORPUS_THRESHOLD:
        # Procesamiento por lotes para corpus grandes
        all_embeddings = []
        for start in tqdm(
            range(0, n_docs, EMBEDDING_BATCH_SIZE),
            desc="  Embedding batches",
        ):
            batch = docs[start : start + EMBEDDING_BATCH_SIZE]
            emb = model.encode(
                batch,
                show_progress_bar=False,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = model.encode(
            docs,
            show_progress_bar=True,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

    np.save(str(cache_path), embeddings)
    print(f"  Embeddings cached → {cache_path.name}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────

def create_bertopic_model(
    docs: list[str],
    source_label: str | None = None,
    nr_topics: int | str = "auto",
) -> tuple[BERTopic, list[int], list[float]]:
    """Crea y entrena un modelo BERTopic optimizado para CPU.

    Parameters
    ----------
    docs : list[str]
        Documentos de texto limpio.
    source_label : str | None
        Etiqueta para identificar el subconjunto (ej. 'academic').
        Se usa para el nombre del archivo de caché de embeddings.
    nr_topics : int | str
        Número de tópicos objetivo o 'auto' para selección automática.

    Returns
    -------
    tuple[BERTopic, list[int], list[float]]
        (modelo BERTopic, lista de asignaciones de tópico, probabilidades)
    """
    label = source_label or "combined"
    print(f"\n{'='*60}")
    print(f"  BERTopic — source: {label} | docs: {len(docs):,}")
    print(f"{'='*60}")

    # ── Pre-calcular embeddings ─────────────────────────────────────
    embeddings = _compute_or_load_embeddings(docs, label=label)

    # ── Componentes del pipeline ────────────────────────────────────
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=COMBINED_STOPWORDS,
        ngram_range=(1, 2),
        min_df=10,
    )

    representation_model = KeyBERTInspired()

    # ── Construir BERTopic ──────────────────────────────────────────
    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL_NAME,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    print("  Fitting BERTopic model ...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    print(f"  ✓ Found {n_topics} topics | {n_outliers:,} outliers (-1)")

    return topic_model, topics, probs


def run_dynamic_topics(
    model: BERTopic,
    docs: list[str],
    timestamps: list[str],
) -> dict:
    """Ejecuta análisis temporal de tópicos (topics over time).

    Parameters
    ----------
    model : BERTopic
        Modelo BERTopic ya entrenado.
    docs : list[str]
        Los mismos documentos usados en el entrenamiento.
    timestamps : list[str]
        Marcas temporales por documento (formato 'YYYY' o 'YYYY-MM').

    Returns
    -------
    dict
        Diccionario con:
        - 'topics_over_time': DataFrame de BERTopic con tópicos por periodo
        - 'n_periods': número de periodos únicos
    """
    print(f"\n  Running topics-over-time | {len(set(timestamps))} unique periods")

    topics_over_time = model.topics_over_time(
        docs,
        timestamps,
        nr_bins=20,
    )

    n_periods = topics_over_time["Timestamp"].nunique()
    print(f"  ✓ Temporal analysis complete — {n_periods} time bins")

    return {
        "topics_over_time": topics_over_time,
        "n_periods": n_periods,
    }


def run_topic_modeling_per_group(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    source_col: str = "source",
) -> dict:
    """Ejecuta BERTopic por separado para cada fuente y para el corpus completo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con al menos las columnas de texto y fuente.
    text_col : str
        Nombre de la columna con texto limpio.
    source_col : str
        Nombre de la columna que identifica la fuente
        (valores esperados: 'academic', 'political', 'public').

    Returns
    -------
    dict
        Diccionario con estructura:
        {
            'academic':  {'model': BERTopic, 'topics': list, 'probs': list, 'docs': list},
            'political': { ... },
            'public':    { ... },
            'combined':  { ... },
        }
    """
    results = {}
    sources = sorted(df[source_col].unique())
    total = len(sources) + 1  # +1 para el corpus combinado

    # ── Modelado por grupo ──────────────────────────────────────────
    for i, source in enumerate(sources, 1):
        print(f"\n  Processing group: {source} | {i}/{total}")
        subset = df[df[source_col] == source]
        docs = subset[text_col].tolist()

        if len(docs) < 100:
            print(f"  ⚠ Skipping '{source}' — only {len(docs)} docs (need ≥100)")
            continue

        model, topics, probs = create_bertopic_model(
            docs, source_label=source
        )

        # Guardar resultados
        output_dir = TOPIC_RESULTS_DIR / source
        save_topic_results(model, topics, docs, output_dir, label=source)

        results[source] = {
            "model": model,
            "topics": topics,
            "probs": probs,
            "docs": docs,
        }

    # ── Modelado del corpus combinado ───────────────────────────────
    print(f"\n  Processing group: combined | {total}/{total}")
    all_docs = df[text_col].tolist()
    model, topics, probs = create_bertopic_model(
        all_docs, source_label="combined"
    )

    output_dir = TOPIC_RESULTS_DIR / "combined"
    save_topic_results(model, topics, all_docs, output_dir, label="combined")

    results["combined"] = {
        "model": model,
        "topics": topics,
        "probs": probs,
        "docs": all_docs,
    }

    print(f"\n  ✓ Topic modeling complete for {len(results)} groups")
    return results


def compare_topic_distributions(models: dict) -> pd.DataFrame:
    """Compara distribuciones de tópicos entre grupos.

    Parameters
    ----------
    models : dict
        Diccionario devuelto por `run_topic_modeling_per_group`.
        Cada valor debe tener claves 'model' y 'topics'.

    Returns
    -------
    pd.DataFrame
        Tabla con prevalencia (%) de cada tópico por grupo.
    """
    print("\n  Comparing topic distributions across groups ...")
    rows = []

    for group_name, data in models.items():
        topics = data["topics"]
        model = data["model"]
        total = len(topics)

        topic_info = model.get_topic_info()

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Omitir outliers

            count = row["Count"]
            top_words = row.get("Name", f"Topic_{topic_id}")
            prevalence = (count / total) * 100

            rows.append({
                "group": group_name,
                "topic_id": topic_id,
                "topic_name": top_words,
                "count": count,
                "total_docs": total,
                "prevalence_pct": round(prevalence, 2),
            })

    comparison_df = pd.DataFrame(rows)

    # Guardar comparación
    output_path = TOPIC_RESULTS_DIR / "topic_distribution_comparison.csv"
    comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Comparison saved → {output_path.name}")

    return comparison_df


def save_topic_results(
    model: BERTopic,
    topics: list[int],
    docs: list[str],
    output_dir: Path,
    label: str,
) -> None:
    """Guarda resultados del modelo de tópicos en disco.

    Parameters
    ----------
    model : BERTopic
        Modelo BERTopic entrenado.
    topics : list[int]
        Asignaciones de tópico por documento.
    docs : list[str]
        Documentos originales.
    output_dir : Path
        Directorio de salida.
    label : str
        Etiqueta descriptiva (ej. 'academic', 'combined').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving topic results for '{label}' → {output_dir}")

    # ── 1. Topic info CSV ───────────────────────────────────────────
    topic_info = model.get_topic_info()
    topic_info.to_csv(
        output_dir / f"topic_info_{label}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  ✓ topic_info_{label}.csv")

    # ── 2. Document-topic assignments ───────────────────────────────
    doc_topics_df = pd.DataFrame({
        "document": docs,
        "topic": topics,
    })
    doc_topics_df.to_csv(
        output_dir / f"doc_topics_{label}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  ✓ doc_topics_{label}.csv")

    # ── 3. Topic words (top palabras por tópico) ────────────────────
    topic_words_rows = []
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words_scores = model.get_topic(topic_id)
        if words_scores:
            for word, score in words_scores:
                topic_words_rows.append({
                    "topic_id": topic_id,
                    "word": word,
                    "score": round(score, 4),
                })

    if topic_words_rows:
        pd.DataFrame(topic_words_rows).to_csv(
            output_dir / f"topic_words_{label}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"  ✓ topic_words_{label}.csv")

    # ── 4. Visualizaciones HTML ─────────────────────────────────────
    try:
        fig = model.visualize_topics()
        fig.write_html(str(output_dir / f"viz_topics_{label}.html"))
        print(f"  ✓ viz_topics_{label}.html")
    except Exception as e:
        print(f"  ⚠ Could not generate topic visualization: {e}")

    try:
        fig = model.visualize_barchart(top_n_topics=15)
        fig.write_html(str(output_dir / f"viz_barchart_{label}.html"))
        print(f"  ✓ viz_barchart_{label}.html")
    except Exception as e:
        print(f"  ⚠ Could not generate barchart: {e}")

    try:
        fig = model.visualize_heatmap()
        fig.write_html(str(output_dir / f"viz_heatmap_{label}.html"))
        print(f"  ✓ viz_heatmap_{label}.html")
    except Exception as e:
        print(f"  ⚠ Could not generate heatmap: {e}")

    try:
        fig = model.visualize_hierarchy()
        fig.write_html(str(output_dir / f"viz_hierarchy_{label}.html"))
        print(f"  ✓ viz_hierarchy_{label}.html")
    except Exception as e:
        print(f"  ⚠ Could not generate hierarchy: {e}")

    # ── 5. Guardar modelo serializado ───────────────────────────────
    model_path = output_dir / f"bertopic_model_{label}"
    model.save(str(model_path), serialization="pickle")
    print(f"  ✓ Model saved → bertopic_model_{label}/")

    print(f"  ✓ All results saved for '{label}'")


# ─────────────────────────────────────────────────────────────────────
# MAIN — ejecución independiente para pruebas
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VOZ_SUS — Topic Modeling Module (standalone test)")
    print("=" * 60)

    # Buscar corpus procesado
    corpus_path = PROCESSED_DIR / "corpus_clean.csv"
    if not corpus_path.exists():
        # Intentar con parquet
        corpus_path = PROCESSED_DIR / "corpus_clean.parquet"

    if not corpus_path.exists():
        print(f"\n  ✗ Corpus not found at {PROCESSED_DIR}")
        print("  Expected: corpus_clean.csv or corpus_clean.parquet")
        print("  Run the preprocessing pipeline first.")
        sys.exit(1)

    # Cargar datos
    print(f"\n  Loading corpus from: {corpus_path.name}")
    if corpus_path.suffix == ".parquet":
        df = pd.read_parquet(corpus_path)
    else:
        df = pd.read_csv(corpus_path)

    print(f"  Loaded {len(df):,} documents")
    print(f"  Columns: {list(df.columns)}")

    # Determinar columna de texto
    text_col = "text_clean"
    if text_col not in df.columns:
        text_col = "text"
        if text_col not in df.columns:
            print(f"  ✗ No suitable text column found")
            sys.exit(1)

    # Filtrar documentos vacíos
    df = df[df[text_col].notna() & (df[text_col].str.len() > 30)].reset_index(drop=True)
    print(f"  After filtering: {len(df):,} documents")

    # Ejecutar modelado por grupo si existe columna 'source'
    if "source" in df.columns:
        results = run_topic_modeling_per_group(df, text_col=text_col)
        comparison = compare_topic_distributions(results)
        print(f"\n  Topic distribution comparison:")
        print(comparison.to_string(index=False))
    else:
        # Modelado simple del corpus completo
        docs = df[text_col].tolist()
        model, topics, probs = create_bertopic_model(docs, source_label="full")
        save_topic_results(model, topics, docs, TOPIC_RESULTS_DIR / "full", label="full")

    print("\n  ✓ Topic modeling complete!")
