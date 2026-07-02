"""
public_collector.py — Recolector de comentarios ciudadanos sobre sustentabilidad
================================================================================

Utiliza la API de YouTube (v3) para recopilar comentarios de ciudadanos
(voz pública) en videos relacionados con sustentabilidad.

Estrategia dual:
  A) Recolectar comentarios de los videos políticos ya recopilados
     (carga political_corpus.csv para obtener los video_ids)
  B) Buscar videos populares de sustentabilidad y recolectar sus comentarios

Filtros:
  - Solo comentarios de primer nivel (top-level)
  - Mínimo de likes configurable (PUBLIC_MIN_LIKES)

Salida: public_corpus.csv en data/raw/
"""

# ── stdlib ──────────────────────────────────────────────────────────
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Configuración del proyecto ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    POLITICAL_CSV,
    PUBLIC_CHECKPOINT_EVERY,
    PUBLIC_CSV,
    PUBLIC_EXTRA_SEARCH_QUERIES,
    PUBLIC_EXTRA_VIDEOS,
    PUBLIC_MAX_COMMENTS_PER_VIDEO,
    PUBLIC_MIN_LIKES,
    RAW_DIR,
    YOUTUBE_API_KEY,
    YOUTUBE_REQUEST_DELAY,
)

# ── third-party ─────────────────────────────────────────────────────
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────


def build_youtube_client():
    """Construye el cliente de YouTube Data API v3.

    Returns:
        Resource de la API de YouTube autenticado con la API key.
    """
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def load_political_video_ids(political_csv: Path = POLITICAL_CSV) -> list[str]:
    """Carga los video_ids del corpus político previamente recolectado.

    Args:
        political_csv: Ruta al CSV del corpus político.

    Returns:
        Lista de video_ids únicos.
    """
    if not political_csv.exists():
        print(f"  ⚠️  No se encontró {political_csv.name}, se omite Estrategia A")
        return []

    df = pd.read_csv(political_csv, usecols=["video_id"])
    ids = df["video_id"].dropna().unique().tolist()
    print(f"  📋 Cargados {len(ids)} video_ids del corpus político")
    return ids


def search_popular_sustainability_videos(
    youtube,
    queries: list[str],
    max_videos: int = PUBLIC_EXTRA_VIDEOS,
) -> list[dict[str, Any]]:
    """Busca videos populares de sustentabilidad para recolectar comentarios.

    Ordena por viewCount para obtener videos con alta interacción
    donde es más probable encontrar comentarios de calidad.

    Args:
        youtube: Cliente de la API de YouTube.
        queries: Lista de consultas de búsqueda.
        max_videos: Número máximo de videos a retornar.

    Returns:
        Lista de diccionarios con video_id y title.
    """
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in queries:
        if len(results) >= max_videos:
            break

        try:
            request = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=min(25, max_videos - len(results)),
                order="viewCount",
                relevanceLanguage="en",
            )
            response = request.execute()

            for item in response.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    results.append({
                        "video_id": vid,
                        "title": item["snippet"].get("title", ""),
                    })

            time.sleep(YOUTUBE_REQUEST_DELAY)

        except HttpError as e:
            print(f"  ✗ Error en búsqueda '{query}': {e}")
            break

    return results[:max_videos]


def get_video_titles(youtube, video_ids: list[str]) -> dict[str, str]:
    """Obtiene los títulos de una lista de videos en batch.

    Args:
        youtube: Cliente de la API de YouTube.
        video_ids: Lista de IDs de videos.

    Returns:
        Diccionario {video_id: title}.
    """
    titles: dict[str, str] = {}
    batch_size = 50

    for i in range(0, len(video_ids), batch_size):
        batch = video_ids[i : i + batch_size]
        try:
            request = youtube.videos().list(
                part="snippet",
                id=",".join(batch),
            )
            response = request.execute()

            for item in response.get("items", []):
                titles[item["id"]] = item["snippet"].get("title", "")

            time.sleep(YOUTUBE_REQUEST_DELAY)

        except HttpError as e:
            print(f"  ✗ Error obteniendo títulos (batch {i}): {e}")

    return titles


def collect_comments_for_video(
    youtube,
    video_id: str,
    max_comments: int = PUBLIC_MAX_COMMENTS_PER_VIDEO,
    min_likes: int = PUBLIC_MIN_LIKES,
) -> list[dict[str, Any]]:
    """Recolecta comentarios de primer nivel de un video de YouTube.

    Pagina a través de los resultados usando nextPageToken hasta alcanzar
    el máximo configurado o agotar los comentarios disponibles.

    Args:
        youtube: Cliente de la API de YouTube.
        video_id: ID del video de YouTube.
        max_comments: Número máximo de comentarios a recolectar.
        min_likes: Mínimo de likes para incluir un comentario.

    Returns:
        Lista de diccionarios con los datos de cada comentario.
    """
    comments: list[dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while len(comments) < max_comments:
        try:
            params: dict[str, Any] = {
                "videoId": video_id,
                "part": "snippet",
                "maxResults": min(100, max_comments - len(comments)),
                "order": "relevance",
                "textFormat": "plainText",
            }

            if next_page_token:
                params["pageToken"] = next_page_token

            request = youtube.commentThreads().list(**params)
            response = request.execute()

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                top_comment = snippet.get("topLevelComment", {})
                comment_snippet = top_comment.get("snippet", {})

                like_count = int(comment_snippet.get("likeCount", 0))

                # Filtrar por mínimo de likes
                if like_count < min_likes:
                    continue

                comments.append({
                    "comment_id": top_comment.get("id", ""),
                    "video_id": video_id,
                    "author_display_name": comment_snippet.get(
                        "authorDisplayName", ""
                    ),
                    "text": comment_snippet.get("textDisplay", ""),
                    "published_at": comment_snippet.get("publishedAt", ""),
                    "like_count": like_count,
                    "reply_count": int(snippet.get("totalReplyCount", 0)),
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(YOUTUBE_REQUEST_DELAY)

        except HttpError as e:
            error_reason = ""
            if hasattr(e, "error_details") and e.error_details:
                error_reason = str(e.error_details)
            elif "commentsDisabled" in str(e):
                error_reason = "comments disabled"
            elif "quotaExceeded" in str(e):
                error_reason = "quota exceeded"
                print(f"\n  🛑 Quota de API agotada. Deteniendo recolección.")
                return comments  # Retornar lo que se tiene

            # Silenciar errores comunes (comentarios deshabilitados)
            if "commentsDisabled" not in str(e):
                print(
                    f"  ✗ Error en comentarios de {video_id}: "
                    f"{error_reason or e}"
                )
            break

        except Exception as e:
            print(f"  ✗ Error inesperado en {video_id}: {e}")
            break

    return comments


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


def collect_public_corpus(
    output_path: Path = PUBLIC_CSV,
) -> pd.DataFrame:
    """Recolecta comentarios ciudadanos sobre sustentabilidad desde YouTube.

    Estrategia dual:
      A) Comentarios de los videos políticos ya recopilados
      B) Comentarios de videos populares de sustentabilidad

    Args:
        output_path: Ruta del CSV de salida.

    Returns:
        DataFrame con el corpus de comentarios ciudadanos.
    """
    print("=" * 70)
    print("💬 PUBLIC COLLECTOR — YouTube Comments (Citizen Voice)")
    print("=" * 70)
    print(f"  👍 Min likes:          {PUBLIC_MIN_LIKES}")
    print(f"  💬 Max comments/video: {PUBLIC_MAX_COMMENTS_PER_VIDEO}")
    print(f"  💾 Output:             {output_path}")
    print("-" * 70)

    youtube = build_youtube_client()
    all_records: list[dict] = []
    quota_exhausted = False

    # ── Estrategia A: Comentarios de videos políticos ────────────────
    print("\n  ⏳ Estrategia A: Comentarios de videos políticos...")

    political_ids = load_political_video_ids()

    if political_ids:
        # Obtener títulos para enriquecer los datos
        titles_map = get_video_titles(youtube, political_ids)

        for i, vid in enumerate(
            tqdm(
                political_ids,
                desc="  💬 Videos políticos",
                unit="vid",
                ncols=90,
            )
        ):
            comments = collect_comments_for_video(youtube, vid)

            # Agregar título del video a cada comentario
            video_title = titles_map.get(vid, "")
            for c in comments:
                c["video_title"] = video_title

            all_records.extend(comments)

            # Checkpoint periódico
            if (
                len(all_records) % PUBLIC_CHECKPOINT_EVERY == 0
                and len(all_records) > 0
            ):
                save_checkpoint(all_records, output_path)

            # Detectar si la quota se agotó
            if any("quota" in str(c.get("text", "")).lower() for c in comments):
                pass  # Los comentarios de texto no indican quota
            # La detección real se hace dentro de collect_comments_for_video

            time.sleep(YOUTUBE_REQUEST_DELAY)

        print(
            f"  ✓ Estrategia A completada: {len(all_records):,} "
            f"comentarios de {len(political_ids)} videos"
        )

    # ── Estrategia B: Videos populares de sustentabilidad ────────────
    print("\n  ⏳ Estrategia B: Videos populares de sustentabilidad...")

    extra_videos = search_popular_sustainability_videos(
        youtube,
        queries=PUBLIC_EXTRA_SEARCH_QUERIES,
        max_videos=PUBLIC_EXTRA_VIDEOS,
    )
    print(f"  📺 Encontrados {len(extra_videos)} videos populares adicionales")

    records_before = len(all_records)
    seen_video_ids = set(political_ids)

    for video in tqdm(
        extra_videos,
        desc="  💬 Videos populares",
        unit="vid",
        ncols=90,
    ):
        vid = video["video_id"]
        if vid in seen_video_ids:
            continue
        seen_video_ids.add(vid)

        comments = collect_comments_for_video(youtube, vid)

        for c in comments:
            c["video_title"] = video.get("title", "")

        all_records.extend(comments)

        # Checkpoint periódico
        if (
            len(all_records) % PUBLIC_CHECKPOINT_EVERY == 0
            and len(all_records) > 0
        ):
            save_checkpoint(all_records, output_path)

        time.sleep(YOUTUBE_REQUEST_DELAY)

    extra_count = len(all_records) - records_before
    print(f"  ✓ Estrategia B completada: {extra_count:,} comentarios adicionales")

    # ── Guardar resultado final ──────────────────────────────────────
    df = pd.DataFrame(all_records)

    if not df.empty:
        # Eliminar duplicados por comment_id
        df = df.drop_duplicates(subset=["comment_id"], keep="first")

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 70}")
        print(f"  ✓ Recolección de comentarios completada")
        print(f"  📊 Total comentarios:   {len(df):,}")
        print(f"  📺 Videos con datos:    {df['video_id'].nunique():,}")
        print(f"  👍 Promedio de likes:   {df['like_count'].mean():.1f}")
        print(f"  💬 Promedio de replies: {df['reply_count'].mean():.1f}")
        print(
            f"  📏 Longitud media texto: "
            f"{df['text'].str.len().mean():.0f} caracteres"
        )
        print(f"  💾 Guardado en:         {output_path}")
        print(f"{'=' * 70}")
    else:
        print("\n  ✗ No se recolectaron comentarios.")

    return df


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN INDEPENDIENTE
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Ejecutando public_collector.py de forma independiente\n")
    df = collect_public_corpus()
    print(f"\n🏁 Finalizado. Shape del DataFrame: {df.shape}")
