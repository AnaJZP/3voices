"""
academic_collector.py — Recolector de artículos académicos sobre sustentabilidad
================================================================================

Utiliza la API de OpenAlex (vía pyalex) para recopilar artículos de revistas
científicas que contengan términos clave de sustentabilidad en título/resumen.

Estrategia de búsqueda:
  - Consultas en inglés y español
  - Filtro por tipo: journal-article
  - Rango temporal: 2015-01-01 a 2026-12-31
  - Paginación con cursor (pyalex lo maneja nativamente)

Salida: academic_corpus.csv en data/raw/
"""

# ── stdlib ──────────────────────────────────────────────────────────
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Configuración del proyecto ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    ACADEMIC_CHECKPOINT_EVERY,
    ACADEMIC_CSV,
    CONTACT_EMAIL,
    OPENALEX_ARTICLE_TYPE,
    OPENALEX_DATE_FROM,
    OPENALEX_DATE_TO,
    OPENALEX_MAX_RESULTS,
    OPENALEX_PER_PAGE,
    OPENALEX_SEARCH_QUERY,
    RAW_DIR,
)

# ── third-party ─────────────────────────────────────────────────────
import pandas as pd
import pyalex
from pyalex import Works
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────


def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruye el texto del abstract a partir del abstract_inverted_index.

    OpenAlex almacena los abstracts como un índice invertido
    {palabra: [posiciones]}. Esta función lo convierte de nuevo a texto plano.

    Args:
        inverted_index: Diccionario {token: [pos1, pos2, ...]} o None.

    Returns:
        Texto reconstruido del abstract, o cadena vacía si no hay datos.
    """
    if not inverted_index:
        return ""

    # Crear lista de (posición, palabra)
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Ordenar por posición y unir
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_concepts(work: dict) -> str:
    """Extrae los conceptos (topics/concepts) de un trabajo de OpenAlex.

    Args:
        work: Diccionario con los datos del artículo.

    Returns:
        Cadena con conceptos separados por " | ".
    """
    concepts = work.get("concepts") or []
    return " | ".join(
        c.get("display_name", "") for c in concepts if c.get("display_name")
    )


def extract_keywords(work: dict) -> str:
    """Extrae las keywords de un trabajo de OpenAlex.

    Args:
        work: Diccionario con los datos del artículo.

    Returns:
        Cadena con keywords separadas por " | ".
    """
    keywords = work.get("keywords") or []
    return " | ".join(
        kw.get("keyword", "") if isinstance(kw, dict) else str(kw)
        for kw in keywords
        if kw
    )


def extract_authors(work: dict) -> str:
    """Extrae los nombres de los autores de un trabajo.

    Args:
        work: Diccionario con los datos del artículo.

    Returns:
        Cadena con autores separados por " ; ".
    """
    authorships = work.get("authorships") or []
    names = []
    for authorship in authorships:
        author = authorship.get("author", {}) or {}
        name = author.get("display_name", "")
        if name:
            names.append(name)
    return " ; ".join(names)


def extract_journal(work: dict) -> str:
    """Extrae el nombre de la revista (source) del primary_location.

    Args:
        work: Diccionario con los datos del artículo.

    Returns:
        Nombre de la revista o cadena vacía.
    """
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name", "")


def parse_work(work: dict) -> dict[str, Any]:
    """Extrae los campos relevantes de un artículo de OpenAlex.

    Args:
        work: Diccionario completo retornado por la API.

    Returns:
        Diccionario con las columnas del corpus académico.
    """
    return {
        "id": work.get("id", ""),
        "title": work.get("title", ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "journal": extract_journal(work),
        "citations": work.get("cited_by_count", 0),
        "language": work.get("language", ""),
        "keywords": extract_keywords(work),
        "concepts": extract_concepts(work),
        "authors": extract_authors(work),
    }


def save_checkpoint(records: list[dict], path: Path) -> None:
    """Guarda un checkpoint parcial del corpus a CSV.

    Args:
        records: Lista de diccionarios con los registros recolectados.
        path: Ruta del archivo CSV de destino.
    """
    df = pd.DataFrame(records)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  💾 Checkpoint guardado: {len(df)} registros → {path.name}")


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE RECOLECCIÓN
# ─────────────────────────────────────────────────────────────────────


def collect_academic_corpus(
    max_results: int = OPENALEX_MAX_RESULTS,
    output_path: Path = ACADEMIC_CSV,
) -> pd.DataFrame:
    """Recolecta artículos académicos sobre sustentabilidad desde OpenAlex.

    Itera sobre cada consulta de búsqueda (EN + ES) individualmente,
    usando paginación por cursor. Los resultados se deduplican por ID.

    Args:
        max_results: Número máximo total de artículos a recolectar.
        output_path: Ruta del CSV de salida.

    Returns:
        DataFrame con el corpus académico completo.
    """
    # Importar las queries desde config
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import OPENALEX_SEARCH

    all_queries = (
        OPENALEX_SEARCH["queries_en"] + OPENALEX_SEARCH["queries_es"]
    )
    per_query_limit = max(500, max_results // len(all_queries))

    print("=" * 70)
    print("📚 ACADEMIC COLLECTOR — OpenAlex")
    print("=" * 70)
    print(f"  🔍 Queries: {len(all_queries)} ({len(OPENALEX_SEARCH['queries_en'])} EN + {len(OPENALEX_SEARCH['queries_es'])} ES)")
    print(f"  📅 Rango:   {OPENALEX_DATE_FROM} -> {OPENALEX_DATE_TO}")
    print(f"  📄 Tipo:    {OPENALEX_ARTICLE_TYPE}")
    print(f"  🎯 Target:  {max_results:,} articulos ({per_query_limit} por query)")
    print(f"  💾 Output:  {output_path}")
    print("-" * 70)

    # Configurar pyalex para entrar al polite pool (10x mas rapido)
    pyalex.config.email = CONTACT_EMAIL
    pyalex.config.max_retries = 5
    pyalex.config.retry_backoff_factor = 0.5

    records: list[dict] = []
    seen_ids: set[str] = set()
    start_time = time.time()

    for qi, q in enumerate(all_queries, 1):
        if len(records) >= max_results:
            break

        print(f"\n  [{qi}/{len(all_queries)}] Buscando: \"{q}\"")

        try:
            query = (
                Works()
                .search(q)
                .filter(
                    type=OPENALEX_ARTICLE_TYPE,
                    from_publication_date=OPENALEX_DATE_FROM,
                    to_publication_date=OPENALEX_DATE_TO,
                )
                .sort(cited_by_count="desc")
            )

            query_count = 0
            for page in query.paginate(per_page=OPENALEX_PER_PAGE, n_max=per_query_limit):
                for work in page:
                    work_id = work.get("id", "")

                    # Evitar duplicados
                    if work_id in seen_ids:
                        continue
                    seen_ids.add(work_id)

                    record = parse_work(work)

                    # Solo agregar si tiene titulo o abstract
                    if record["title"] or record["abstract"]:
                        records.append(record)
                        query_count += 1

                    # Checkpoint periodico
                    if len(records) % ACADEMIC_CHECKPOINT_EVERY == 0 and len(records) > 0:
                        save_checkpoint(records, output_path)

                    if len(records) >= max_results:
                        break

                if len(records) >= max_results:
                    break

            print(f"    -> {query_count} nuevos articulos (total acumulado: {len(records):,})")

        except Exception as e:
            print(f"    ✗ Error en query \"{q}\": {e}")
            continue

    elapsed = time.time() - start_time

    # ── Guardar resultado final ──────────────────────────────────────
    df = pd.DataFrame(records)

    if not df.empty:
        # Limpieza basica
        df["title"] = df["title"].fillna("").str.strip()
        df["abstract"] = df["abstract"].fillna("").str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n{'=' * 70}")
        print(f"  ✓ Recoleccion completada en {elapsed:.1f}s")
        print(f"  📊 Total articulos: {len(df):,}")
        print(f"  📅 Rango de anos:   {df['year'].min()} - {df['year'].max()}")
        print(f"  📰 Revistas unicas: {df['journal'].nunique():,}")
        print(f"  🌐 Idiomas:         {df['language'].value_counts().to_dict()}")
        print(f"  💾 Guardado en:     {output_path}")
        print(f"{'=' * 70}")
    else:
        print("\n  ✗ No se recolectaron articulos.")

    return df


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN INDEPENDIENTE
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Ejecutando academic_collector.py de forma independiente\n")
    df = collect_academic_corpus()
    print(f"\n🏁 Finalizado. Shape del DataFrame: {df.shape}")
