"""
language_detector.py – Detección de idioma y filtrado bilingüe
==============================================================
Detecta idioma (EN/ES) de cada documento, filtra el corpus para
conservar solo los idiomas deseados, y detecta cambio de código
(code-switching) entre inglés y español.

Uso:
    from src.preprocessing.language_detector import filter_bilingual
"""

# stdlib
import sys
from pathlib import Path

# third-party
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm

# local – importar configuración del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *  # noqa: E402, F403

# Fallback si no está en config
try:
    ALLOWED_LANGS
except NameError:
    ALLOWED_LANGS = ["en", "es"]

# Semilla para reproducibilidad de langdetect
DetectorFactory.seed = 0


# ─────────────────────────────────────────────────────────────
#  DETECCIÓN DE IDIOMA / Language detection
# ─────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect the language of a single text.

    Uses the ``langdetect`` library and returns an ISO 639-1
    language code (e.g., ``'en'``, ``'es'``).

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    str
        ISO 639-1 language code, or ``'unknown'`` if detection
        fails or text is too short / empty.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_language_batch(texts: list[str]) -> list[str]:
    """Detect language for a list of texts with a progress bar.

    Parameters
    ----------
    texts : list[str]
        List of texts to analyze.

    Returns
    -------
    list[str]
        List of ISO 639-1 codes (same length as input).
    """
    results: list[str] = []
    for text in tqdm(texts, desc="  Detecting language"):
        results.append(detect_language(text))
    return results


# ─────────────────────────────────────────────────────────────
#  FILTRADO BILINGÜE / Bilingual filtering
# ─────────────────────────────────────────────────────────────

def filter_bilingual(
    df: pd.DataFrame,
    text_col: str = "text_clean",
) -> pd.DataFrame:
    """Detect language for each row and keep only allowed languages.

    Adds a ``lang`` column with the detected ISO 639-1 code, then
    filters the DataFrame to keep only rows whose language is in
    ``ALLOWED_LANGS`` (default: ``['en', 'es']``).

    Prints statistics about language distribution before and after
    filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a text column.
    text_col : str
        Name of the column containing the text to analyze.
        Default: ``'text_clean'``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with ``lang`` column added.
    """
    total = len(df)
    print(f"\n{'='*60}")
    print(f"  Language detection | {total} documents")
    print(f"{'='*60}")

    df = df.copy()
    df["lang"] = detect_language_batch(df[text_col].tolist())

    # Estadísticas de idiomas detectados
    lang_counts = df["lang"].value_counts()
    print(f"\n  Language distribution (all):")
    for lang, count in lang_counts.items():
        pct = count / total * 100
        print(f"    {lang:>10s}: {count:>6d}  ({pct:.1f}%)")

    # Filtrar solo idiomas permitidos
    before = len(df)
    df_filtered = df[df["lang"].isin(ALLOWED_LANGS)].copy()
    after = len(df_filtered)

    removed_langs = df[~df["lang"].isin(ALLOWED_LANGS)]["lang"].value_counts()
    if not removed_langs.empty:
        print(f"\n  Removed languages ({before - after} docs):")
        for lang, count in removed_langs.items():
            print(f"    {lang:>10s}: {count:>6d}")

    print(f"\n  Kept: {after} documents ({ALLOWED_LANGS})")
    print(f"  Removed: {before - after} documents")

    return df_filtered.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  DETECCIÓN DE CODE-SWITCHING / Code-switching detection
# ─────────────────────────────────────────────────────────────

def detect_code_switching(text: str) -> bool:
    """Detect if a text mixes English and Spanish (code-switching).

    Uses a simple heuristic: splits the text into sentences and
    detects the language of each. If both ``'en'`` and ``'es'``
    are found across different sentences, the text is flagged
    as code-switching.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    bool
        ``True`` if the text contains both English and Spanish
        segments, ``False`` otherwise.
    """
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False

    import re
    # Dividir por oraciones (punto, signo de interrogación / exclamación)
    sentences = re.split(r"[.!?]+", text)
    # Filtrar oraciones muy cortas que dan falsos positivos
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if len(sentences) < 2:
        return False

    detected_langs: set[str] = set()
    for sent in sentences:
        lang = detect_language(sent)
        if lang in ("en", "es"):
            detected_langs.add(lang)
        # Salida temprana si ya encontramos ambos idiomas
        if detected_langs == {"en", "es"}:
            return True

    return len(detected_langs) == 2


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pruebas rápidas
    print("\n--- Single detection ---")
    samples = [
        ("This is a test about sustainability.", "en"),
        ("Esta es una prueba sobre sostenibilidad.", "es"),
        ("短い", "unknown"),
        ("", "unknown"),
    ]
    for text, expected in samples:
        result = detect_language(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:40]}' → {result} (expected: {expected})")

    print("\n--- Batch detection ---")
    texts = [s[0] for s in samples]
    results = detect_language_batch(texts)
    print(f"  Results: {results}")

    print("\n--- Code-switching detection ---")
    mixed = (
        "Climate change is a global emergency. "
        "El cambio climático es una emergencia global. "
        "We need immediate action on renewable energy."
    )
    pure_en = "This is only in English. We discuss sustainability here."
    print(f"  Mixed text: {detect_code_switching(mixed)}")
    print(f"  Pure EN:    {detect_code_switching(pure_en)}")

    print("\n--- Bilingual filter ---")
    test_df = pd.DataFrame({
        "text_clean": [
            "This is a long enough English text about climate change.",
            "Este es un texto suficientemente largo sobre cambio climático.",
            "Ceci est un texte en français sur le changement climatique.",
            "Dies ist ein deutscher Text über den Klimawandel und Nachhaltigkeit.",
        ]
    })
    filtered = filter_bilingual(test_df)
    print(f"\n  Filtered DataFrame:\n{filtered.to_string()}")
