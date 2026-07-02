"""
main_preprocess.py – Orquestador del pipeline de preprocesamiento
==================================================================
Carga CSVs crudos de las tres fuentes (academic, institutional, public),
aplica limpieza por fuente, detecta idioma, filtra y guarda los
resultados procesados en data/processed/.

Uso:
    python main_preprocess.py
"""

# stdlib
import sys
from pathlib import Path

# Insertar raíz del proyecto para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent))

# third-party
import pandas as pd

# local
from config import *  # noqa: E402, F403
from constants import normalize_group_series

# Constantes de fuente (si no están en config)
SOURCE_ACADEMIC      = "academic"
SOURCE_INSTITUTIONAL = "institutional"
SOURCE_PUBLIC        = "public"
ALL_SOURCES          = [SOURCE_ACADEMIC, SOURCE_INSTITUTIONAL, SOURCE_PUBLIC]

from src.preprocessing.cleaner import clean_corpus
from src.preprocessing.language_detector import filter_bilingual


# ─────────────────────────────────────────────────────────────
#  FUNCIONES DE CARGA / Loading functions
# ─────────────────────────────────────────────────────────────

def load_raw_csvs(directory: Path, source: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from a raw data directory.

    Each CSV is expected to have at least a ``text`` column.
    A ``source`` column is added with the given source label.

    Parameters
    ----------
    directory : Path
        Path to the directory containing CSV files.
    source : str
        Source label to add (e.g., 'academic', 'political', 'public').

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all rows from all CSVs,
        or an empty DataFrame if no CSVs are found.
    """
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"  ⚠ No CSV files found in {directory}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        print(f"  Loading: {f.name}")
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")
        df["_source_file"] = f.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["source"] = source
    print(f"  Total rows loaded [{source}]: {len(combined)}")
    return combined


def ensure_columns(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Ensure the DataFrame has the standard output columns.

    Adds missing columns with defaults so the final output
    always has: id, text_original, text_clean, source, lang,
    year, plus any source-specific metadata already present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to standardize.
    source : str
        Source label for this corpus.

    Returns
    -------
    pd.DataFrame
        DataFrame with guaranteed standard columns.
    """
    df = df.copy()

    # Columna de texto original
    text_col_candidates = ["text", "abstract", "content", "comment", "transcript"]
    text_col = None
    for col in text_col_candidates:
        if col in df.columns:
            text_col = col
            break

    if text_col is None and len(df.columns) > 0:
        # Tomar la primera columna de tipo string como texto
        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols) > 0:
            text_col = str_cols[0]
            print(f"  ⚠ No standard text column found, using: '{text_col}'")

    if text_col is not None:
        df["text_original"] = df[text_col]
        if text_col != "text":
            df["text"] = df[text_col]
    else:
        df["text_original"] = ""
        df["text"] = ""

    # ID: usar existente o generar
    if "id" not in df.columns:
        df["id"] = [f"{source}_{i:06d}" for i in range(len(df))]

    # Año: usar existente o dejar vacío
    if "year" not in df.columns:
        # Intentar extraer de columnas de fecha
        date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
        if date_cols:
            try:
                df["year"] = pd.to_datetime(df[date_cols[0]], errors="coerce").dt.year
            except Exception:
                df["year"] = pd.NA
        else:
            df["year"] = pd.NA

    # Source ya debería estar presente
    df["source"] = source

    return df


# ─────────────────────────────────────────────────────────────
#  ESTADÍSTICAS / Descriptive statistics
# ─────────────────────────────────────────────────────────────

def print_corpus_stats(df: pd.DataFrame, label: str = "Corpus") -> None:
    """Print descriptive statistics about the processed corpus.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with columns: text_clean, source, lang.
    label : str
        Label for the statistics header.
    """
    print(f"\n{'='*60}")
    print(f"  📊 DESCRIPTIVE STATISTICS – {label}")
    print(f"{'='*60}")

    total = len(df)
    print(f"\n  Total documents: {total:,}")

    if total == 0:
        print("  (empty corpus)")
        return

    # Documentos por idioma
    if "lang" in df.columns:
        print(f"\n  Documents by language:")
        for lang, count in df["lang"].value_counts().items():
            pct = count / total * 100
            print(f"    {lang:>10s}: {count:>6,d}  ({pct:.1f}%)")

    # Documentos por fuente
    if "source" in df.columns:
        print(f"\n  Documents by source:")
        for src, count in df["source"].value_counts().items():
            pct = count / total * 100
            print(f"    {src:>10s}: {count:>6,d}  ({pct:.1f}%)")

    # Longitud promedio del texto limpio
    if "text_clean" in df.columns:
        word_counts = df["text_clean"].str.split().str.len()
        print(f"\n  Text length (words):")
        print(f"    Mean:   {word_counts.mean():.1f}")
        print(f"    Median: {word_counts.median():.1f}")
        print(f"    Min:    {word_counts.min():.0f}")
        print(f"    Max:    {word_counts.max():.0f}")
        print(f"    Std:    {word_counts.std():.1f}")

    # Distribución por año (si existe)
    if "year" in df.columns:
        valid_years = df["year"].dropna()
        if len(valid_years) > 0:
            print(f"\n  Year range: {int(valid_years.min())} – {int(valid_years.max())}")
            print(f"  Documents with year: {len(valid_years):,}")


# ─────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL / Main pipeline
# ─────────────────────────────────────────────────────────────

def process_source(
    raw_dir: Path,
    source: str,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline for a single source.

    Steps:
        1. Load raw CSVs
        2. Standardize columns
        3. Apply source-specific text cleaning
        4. Detect and filter languages

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw CSV files.
    source : str
        Source label ('academic', 'political', 'public').

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame ready for analysis.
    """
    print(f"\n{'#'*60}")
    print(f"  PROCESSING SOURCE: {source.upper()}")
    print(f"{'#'*60}")

    # 1. Cargar datos crudos
    df = load_raw_csvs(raw_dir, source)
    if df.empty:
        print(f"  ⚠ Skipping {source} – no data found.")
        return pd.DataFrame()

    # 2. Estandarizar columnas
    df = ensure_columns(df, source)

    # 3. Limpiar texto
    df = clean_corpus(df, text_col="text", source=source)

    # 4. Detectar idioma y filtrar
    df = filter_bilingual(df, text_col="text_clean")

    return df


def main() -> None:
    """Execute the full preprocessing pipeline for all sources."""
    print("\n" + "█" * 60)
    print("  VOZ_SUS – PREPROCESSING PIPELINE")
    print("  " + "─" * 40)
    print("  Cleaning, language detection & filtering")
    print("█" * 60)

    # Mapeo de fuentes a directorios
    source_dirs = {
        SOURCE_ACADEMIC:      RAW_ACADEMIC,
        SOURCE_INSTITUTIONAL: RAW_INSTITUTIONAL,
        SOURCE_PUBLIC:        RAW_PUBLIC,
    }

    # Procesar cada fuente
    processed_frames: list[pd.DataFrame] = []
    for source, raw_dir in source_dirs.items():
        df = process_source(raw_dir, source)
        if not df.empty:
            # Guardar CSV procesado individual
            out_path = PROCESSED_DIR / f"{source}_processed.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"\n  ✓ Saved: {out_path.name} ({len(df):,} rows)")

            print_corpus_stats(df, label=source.upper())
            processed_frames.append(df)

    # Unificar todas las fuentes
    if processed_frames:
        print(f"\n{'#'*60}")
        print(f"  CREATING UNIFIED CORPUS")
        print(f"{'#'*60}")

        # Columnas estándar para el corpus unificado
        standard_cols = ["id", "text_original", "text_clean", "source", "lang", "year"]

        unified_frames = []
        for df in processed_frames:
            # Asegurar columnas estándar
            for col in standard_cols:
                if col not in df.columns:
                    df[col] = pd.NA
            # Conservar columnas extra como metadata
            extra_cols = [c for c in df.columns if c not in standard_cols]
            unified_frames.append(df[standard_cols + extra_cols])

        corpus = pd.concat(unified_frames, ignore_index=True)

        # Normalize legacy 'political' → 'institutional' in source column
        if "source" in corpus.columns:
            corpus["source"] = normalize_group_series(corpus["source"])

        # Guardar corpus unificado
        unified_path = PROCESSED_DIR / "corpus_unified.csv"
        corpus.to_csv(unified_path, index=False, encoding="utf-8-sig")
        print(f"\n  ✓ Saved: {unified_path.name} ({len(corpus):,} rows)")

        print_corpus_stats(corpus, label="UNIFIED CORPUS")
    else:
        print("\n  ⚠ No data processed. Place CSV files in data/raw/ subdirectories.")
        print(f"    Expected directories:")
        for source, raw_dir in source_dirs.items():
            print(f"      {source}: {raw_dir}")

    print(f"\n{'█'*60}")
    print("  PREPROCESSING COMPLETE")
    print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
