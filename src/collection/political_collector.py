"""
political_collector.py — Recolector de discursos políticos sobre sustentabilidad
================================================================================

Utiliza la API de YouTube (v3) para buscar videos de discursos políticos
sobre sustentabilidad, y la librería youtube_transcript_api para extraer
las transcripciones (subtítulos) de cada video.

Flujo:
  1. Buscar videos con queries de sustentabilidad (EN + ES)
  2. Filtrar por canales políticos oficiales (desde config)
  3. Obtener metadata completa de cada video
  4. Extraer transcripción (subtítulos automáticos o manuales)

Salida: political_corpus.csv en data/raw/
"""

# ── stdlib ──────────────────────────────────────────────────────────
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Configuración del proyecto ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    POLITICAL_CHANNELS,
    POLITICAL_CHECKPOINT_EVERY,
    POLITICAL_CSV,
    POLITICAL_MAX_RESULTS_PER_QUERY,
    POLITICAL_MAX_VIDEOS,
    RAW_DIR,
    SEARCH_QUERIES_ALL,
    YOUTUBE_API_KEY,
    YOUTUBE_REQUEST_DELAY,
    YOUTUBE_SEARCH_ORDER,
)

# ── third-party ─────────────────────────────────────────────────────
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────


def build_youtube_client():
    """Construye el cliente de YouTube Data API v3.

    Returns:
        Resource de la API de YouTube autenticado con la API key.
    """
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos_by_query(
    youtube,
    query: str,
    max_results: int = POLITICAL_MAX_RESULTS_PER_QUERY,
    channel_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Busca videos en YouTube con una query específica.

    Args:
        youtube: Cliente de la API de YouTube.
        query: Texto de búsqueda.
        max_results: Número máximo de resultados.
        channel_id: Opcional, filtrar por canal específico.

    Returns:
        Lista de diccionarios con video_id, title, channel, channel_id, published_at.
    """
    results: list[dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while len(results) < max_results:
        try:
            params: dict[str, Any] = {
                "q": query,
                "part": "snippet",
                "type": "video",
                "maxResults": min(50, max_results - len(results)),
                "order": YOUTUBE_SEARCH_ORDER,
                "relevanceLanguage": "en",
                "videoDuration": "long",  # Discursos suelen ser largos
            }

            if channel_id:
                params["channelId"] = channel_id
            if next_page_token:
                params["pageToken"] = next_page_token

            request = youtube.search().list(**params)
            response = request.execute()

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                vid = item.get("id", {}).get("videoId")
                if vid:
                    results.append({
                        "video_id": vid,
                        "title": snippet.get("title", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "channel_id": snippet.get("channelId", ""),
                        "published_at": snippet.get("publishedAt", ""),
                    })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(YOUTUBE_REQUEST_DELAY)

        except HttpError as e:
            print(f"  ✗ Error en búsqueda '{query}': {e}")
            break

    return results


def get_video_metadata(
    youtube, video_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Obtiene metadata detallada de una lista de videos.

    Utiliza batch requests (hasta 50 IDs por llamada) para minimizar
    el consumo de quota.

    Args:
        youtube: Cliente de la API de YouTube.
        video_ids: Lista de IDs de videos.

    Returns:
        Diccionario {video_id: {views, likes, duration, ...}}.
    """
    metadata: dict[str, dict[str, Any]] = {}

    # Procesar en lotes de 50 (máximo de la API)
    batch_size = 50
    for i in range(0, len(video_ids), batch_size):
        batch = video_ids[i : i + batch_size]
        try:
            request = youtube.videos().list(
                part="statistics,contentDetails,snippet",
                id=",".join(batch),
            )
            response = request.execute()

            for item in response.get("items", []):
                vid = item["id"]
                stats = item.get("statistics", {})
                details = item.get("contentDetails", {})
                snippet = item.get("snippet", {})

                metadata[vid] = {
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "duration": details.get("duration", ""),
                    "description": snippet.get("description", ""),
                    "tags": " | ".join(snippet.get("tags", [])),
                }

            time.sleep(YOUTUBE_REQUEST_DELAY)

        except HttpError as e:
            print(f"  ✗ Error obteniendo metadata (batch {i}): {e}")

    return metadata


def get_transcript(video_id: str) -> tuple[str, str]:
    """Extrae la transcripción de un video de YouTube.

    Usa youtube-transcript-api v1.2+ (instance API con .fetch / .list).

    Args:
        video_id: ID del video de YouTube.

    Returns:
        Tupla (transcript_text, language) donde language es 'en', 'es', o ''.
    """
    api = YouTubeTranscriptApi()

    # Intentar primero español, luego inglés
    language_priorities = [
        ["es", "es-419", "es-MX", "es-ES"],
        ["en", "en-US", "en-GB"],
    ]

    for lang_list in language_priorities:
        try:
            result = api.fetch(video_id, languages=lang_list)
            full_text = " ".join(
                snippet.text.strip()
                for snippet in result.snippets
                if snippet.text.strip()
            )
            if full_text:
                detected_lang = lang_list[0]
                return full_text, detected_lang
        except Exception:
            continue

    # Fallback: intentar cualquier idioma disponible
    try:
        result = api.fetch(video_id)
        full_text = " ".join(
            snippet.text.strip()
            for snippet in result.snippets
            if snippet.text.strip()
        )
        if full_text:
            return full_text, result.language[:2].lower() if result.language else "unknown"
    except Exception:
        pass

    return "", ""


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


def collect_political_corpus(
    max_videos: int = POLITICAL_MAX_VIDEOS,
    output_path: Path = POLITICAL_CSV,
) -> pd.DataFrame:
    """Recolecta discursos políticos sobre sustentabilidad desde YouTube.

    Combina dos estrategias de búsqueda:
      A) Búsqueda libre con queries de sustentabilidad + filtro de canales
      B) Búsqueda directa dentro de canales políticos oficiales

    Para cada video encontrado, extrae la transcripción completa.

    Args:
        max_videos: Número máximo de videos a recolectar.
        output_path: Ruta del CSV de salida.

    Returns:
        DataFrame con el corpus político completo.
    """
    print("=" * 70)
    print("🎤 POLITICAL COLLECTOR — YouTube Speeches")
    print("=" * 70)
    print(f"  🔍 Queries:   {len(SEARCH_QUERIES_ALL)} consultas")
    print(f"  📺 Canales:   {len(POLITICAL_CHANNELS)} canales políticos")
    print(f"  🎯 Target:    {max_videos:,} videos")
    print(f"  💾 Output:    {output_path}")
    print("-" * 70)

    youtube = build_youtube_client()
    seen_ids: set[str] = set()
    video_candidates: list[dict[str, Any]] = []

    # ── Estrategia A: Búsqueda directa en canales oficiales ──────────
    print("\n  ⏳ Fase 1: Buscando en canales políticos oficiales...")

    for channel_name, channel_id in tqdm(
        POLITICAL_CHANNELS.items(),
        desc="  📺 Canales",
        unit="canal",
        ncols=90,
    ):
        for query in SEARCH_QUERIES_ALL:
            results = search_videos_by_query(
                youtube,
                query=query,
                max_results=POLITICAL_MAX_RESULTS_PER_QUERY,
                channel_id=channel_id,
            )
            for r in results:
                if r["video_id"] not in seen_ids:
                    seen_ids.add(r["video_id"])
                    video_candidates.append(r)

            time.sleep(YOUTUBE_REQUEST_DELAY)

    print(f"  ✓ Fase 1 completada: {len(video_candidates)} videos de canales oficiales")

    # ── Estrategia B: Búsqueda abierta (filtrar canales después) ─────
    print("\n  ⏳ Fase 2: Búsqueda abierta con queries de sustentabilidad...")

    # Mapeo inverso channel_id → channel_name para filtro rápido
    official_channel_ids = set(POLITICAL_CHANNELS.values())

    for query in tqdm(
        SEARCH_QUERIES_ALL,
        desc="  🔍 Queries",
        unit="query",
        ncols=90,
    ):
        results = search_videos_by_query(
            youtube,
            query=f"{query} speech OR discourse OR address OR summit",
            max_results=POLITICAL_MAX_RESULTS_PER_QUERY,
        )
        for r in results:
            if r["video_id"] not in seen_ids:
                seen_ids.add(r["video_id"])
                video_candidates.append(r)

        time.sleep(YOUTUBE_REQUEST_DELAY)

        if len(video_candidates) >= max_videos:
            break

    # Limitar al máximo configurado
    video_candidates = video_candidates[:max_videos]
    print(f"  ✓ Fase 2 completada: {len(video_candidates)} videos candidatos totales")

    # ── Obtener metadata en batch ────────────────────────────────────
    print("\n  ⏳ Fase 3: Obteniendo metadata de videos...")
    all_video_ids = [v["video_id"] for v in video_candidates]
    metadata = get_video_metadata(youtube, all_video_ids)
    print(f"  ✓ Metadata obtenida para {len(metadata)} videos")

    # ── Extraer transcripciones ──────────────────────────────────────
    print("\n  ⏳ Fase 4: Extrayendo transcripciones...")
    records: list[dict] = []
    transcripts_found = 0
    transcripts_missing = 0

    for video in tqdm(
        video_candidates,
        desc="  📝 Transcripts",
        unit="vid",
        ncols=90,
    ):
        vid = video["video_id"]
        meta = metadata.get(vid, {})

        transcript_text, lang = get_transcript(vid)

        if transcript_text:
            transcripts_found += 1
        else:
            transcripts_missing += 1

        record = {
            "video_id": vid,
            "title": video["title"],
            "channel": video["channel"],
            "channel_id": video["channel_id"],
            "published_at": video["published_at"],
            "views": meta.get("views", 0),
            "likes": meta.get("likes", 0),
            "transcript": transcript_text,
            "language": lang,
            "duration": meta.get("duration", ""),
        }
        records.append(record)

        # Checkpoint periódico
        if len(records) % POLITICAL_CHECKPOINT_EVERY == 0 and len(records) > 0:
            save_checkpoint(records, output_path)

        time.sleep(YOUTUBE_REQUEST_DELAY)

    # ── Guardar resultado final ──────────────────────────────────────
    df = pd.DataFrame(records)

    if not df.empty:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n{'=' * 70}")
        print(f"  ✓ Recolección de discursos completada")
        print(f"  📊 Total videos:          {len(df):,}")
        print(f"  📝 Con transcripción:      {transcripts_found:,}")
        print(f"  ❌ Sin transcripción:       {transcripts_missing:,}")
        print(f"  📺 Canales únicos:         {df['channel'].nunique():,}")
        print(f"  🌐 Idiomas:               {df['language'].value_counts().to_dict()}")
        print(f"  👀 Promedio de vistas:     {df['views'].mean():,.0f}")
        print(f"  💾 Guardado en:            {output_path}")
        print(f"{'=' * 70}")
    else:
        print("\n  ✗ No se recolectaron videos.")

    return df


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN INDEPENDIENTE
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Ejecutando political_collector.py de forma independiente\n")
    df = collect_political_corpus()
    print(f"\n🏁 Finalizado. Shape del DataFrame: {df.shape}")
