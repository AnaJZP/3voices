"""
cleaner.py – Pipeline de limpieza de texto con estrategias por fuente
=====================================================================
Limpia textos académicos, políticos (transcripciones) y públicos
(comentarios de YouTube/redes) con reglas específicas por tipo de fuente.

Uso:
    from src.preprocessing.cleaner import clean_text, clean_corpus
"""

# stdlib
import re
import sys
import unicodedata
from pathlib import Path

# third-party
import pandas as pd
from tqdm import tqdm

# local – importar configuración del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MIN_TOKENS  # noqa: E402


# ─────────────────────────────────────────────────────────────
#  CONSTANTES / Constants
# ─────────────────────────────────────────────────────────────

# Acrónimos comunes en discurso de sostenibilidad
ACRONYM_MAP: dict[str, str] = {
    r"\bSDGs?\b":  "Sustainable Development Goals",
    r"\bESG\b":    "Environmental Social and Governance",
    r"\bCSR\b":    "Corporate Social Responsibility",
    r"\bGHG\b":    "greenhouse gas",
    r"\bLCA\b":    "life cycle assessment",
    r"\bNDC\b":    "nationally determined contribution",
    r"\bIPCC\b":   "Intergovernmental Panel on Climate Change",
    r"\bUNFCCC\b": "United Nations Framework Convention on Climate Change",
    r"\bCOP\b":    "Conference of the Parties",
    r"\bGRI\b":    "Global Reporting Initiative",
}

# Patrones de DOI / URL
DOI_PATTERN     = re.compile(r"https?://doi\.org/\S+|doi:\s*\S+", re.IGNORECASE)
URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Marcadores de transcripción (timestamps, etiquetas de hablante)
TIMESTAMP_PATTERN   = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")
SPEAKER_PATTERN     = re.compile(r"^[\w\s]{2,30}:\s", re.MULTILINE)
AUTOGEN_ARTIFACTS   = re.compile(
    r"\[(?:music|applause|laughter|inaudible|crosstalk)\]",
    re.IGNORECASE,
)

# Muletillas / filler words (EN + ES)
FILLER_WORDS = re.compile(
    r"\b(?:um+|uh+|erm+|hmm+|ah+|oh+|like|you know|I mean"
    r"|este|bueno|pues|o sea|eh+|ajá|mmm+|verdad)\b",
    re.IGNORECASE,
)

# Emojis – mapeo manual de los más frecuentes (alternativa ligera a la
# librería `emoji` para no agregar dependencia pesada)
EMOJI_MAP: dict[str, str] = {
    "😀": "happy face", "😂": "laughing", "❤️": "heart",
    "👍": "thumbs up", "👎": "thumbs down", "🔥": "fire",
    "🌍": "earth globe", "🌱": "seedling", "♻️": "recycling",
    "⚡": "lightning", "💧": "water drop", "🌊": "ocean wave",
    "🏭": "factory", "🌳": "tree", "☀️": "sun",
    "💚": "green heart", "🙏": "folded hands", "😡": "angry",
    "😢": "crying", "🤔": "thinking", "💡": "light bulb",
    "📊": "chart", "🗳️": "ballot box", "🏛️": "parliament",
}

# Patrón genérico para emojis Unicode fuera del mapeo
EMOJI_GENERIC_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

# Spam / caracteres repetidos
REPEATED_CHARS = re.compile(r"(.)\1{3,}")          # 4+ repeticiones → 2
EXCESSIVE_PUNCT = re.compile(r"([!?.]){2,}")        # !!!! → !
EXCESSIVE_CAPS  = re.compile(r"\b[A-Z]{4,}\b")     # gritar en mayúsculas

# Hashtag camelCase splitter
CAMELCASE_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Menciones (@user)
MENTION_PATTERN = re.compile(r"@\w+")


# ─────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES / Helper functions
# ─────────────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form and strip accidental control chars."""
    text = unicodedata.normalize("NFC", text)
    # Eliminar caracteres de control excepto newline y tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_hashtag(match: re.Match) -> str:
    """Convert #ClimateChange → climate change."""
    tag = match.group(0).lstrip("#")
    words = CAMELCASE_SPLIT.sub(" ", tag)
    return words.lower()


def _replace_emojis(text: str) -> str:
    """Replace known emojis with text descriptions, remove the rest."""
    # Intentar importar la librería emoji; si no existe, usar mapeo manual
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
        # Limpiar los delimitadores de la librería (:thumbs_up: → thumbs up)
        text = re.sub(r":(\w[\w_]*):", lambda m: m.group(1).replace("_", " "), text)
    except ImportError:
        for em, desc in EMOJI_MAP.items():
            text = text.replace(em, f" {desc} ")
        # Eliminar emojis restantes no mapeados
        text = EMOJI_GENERIC_PATTERN.sub(" ", text)
    return text


# ─────────────────────────────────────────────────────────────
#  LIMPIADORES POR FUENTE / Source-specific cleaners
# ─────────────────────────────────────────────────────────────

def clean_academic(text: str) -> str:
    """Clean academic / journal text.

    Removes metadata artifacts (DOIs, URLs), normalizes Unicode,
    expands common sustainability acronyms, and normalizes whitespace.

    Parameters
    ----------
    text : str
        Raw academic text (abstract, full-text paragraph, etc.).

    Returns
    -------
    str
        Cleaned text ready for NLP analysis.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _normalize_unicode(text)

    # Eliminar DOIs y URLs
    text = DOI_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    # Eliminar artefactos de metadata comunes en PDFs/Scopus
    # e.g., "© 2023 Elsevier", números de página, headers repetidos
    text = re.sub(r"©.*?\d{4}.*?(?:\.|$)", " ", text)
    text = re.sub(r"\bpp?\.\s*\d+[-–]\d+\b", " ", text)
    text = re.sub(r"\b(?:vol|volume|issue|no)\.\s*\d+", " ", text, flags=re.IGNORECASE)

    # Expandir acrónimos de sostenibilidad
    for pattern, expansion in ACRONYM_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

    # Eliminar referencias numéricas entre corchetes [1], [2,3]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)

    text = _normalize_whitespace(text)
    return text


def clean_political(text: str) -> str:
    """Clean political speech transcripts.

    Removes timestamps, auto-generated transcript artifacts,
    filler words, joins fragmented lines, and normalizes speaker
    change markers.

    Parameters
    ----------
    text : str
        Raw transcript text (e.g., from UN debates, parliament
        sessions, political speeches).

    Returns
    -------
    str
        Cleaned transcript text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _normalize_unicode(text)

    # Eliminar timestamps [00:01:23] y variantes
    text = TIMESTAMP_PATTERN.sub(" ", text)

    # Eliminar artefactos de transcripción automática [Music], [Applause]
    text = AUTOGEN_ARTIFACTS.sub(" ", text)

    # Normalizar cambios de hablante → separar con punto
    text = SPEAKER_PATTERN.sub(". ", text)

    # Unir líneas fragmentadas (saltos de línea dentro de una oración)
    text = re.sub(r"\n(?![A-Z\"\'\(])", " ", text)

    # Eliminar muletillas
    text = FILLER_WORDS.sub(" ", text)

    # Eliminar URLs que puedan aparecer en transcripciones
    text = URL_PATTERN.sub(" ", text)

    # Normalizar puntuación duplicada resultante
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*\.\s*\.\s*", ". ", text)

    text = _normalize_whitespace(text)
    return text


def clean_public(text: str) -> str:
    """Clean public discourse text (YouTube comments, social media).

    Removes URLs, converts emojis to text, removes @mentions,
    handles hashtags (splits camelCase), reduces spam patterns
    (repeated chars, excessive punctuation/caps), and normalizes slang.

    Parameters
    ----------
    text : str
        Raw comment or social media post.

    Returns
    -------
    str
        Cleaned text suitable for NLP analysis.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _normalize_unicode(text)

    # Eliminar URLs
    text = URL_PATTERN.sub(" ", text)

    # Convertir emojis a descripciones textuales
    text = _replace_emojis(text)

    # Eliminar @menciones
    text = MENTION_PATTERN.sub(" ", text)

    # Procesar hashtags: #ClimateChange → climate change
    text = re.sub(r"#\w+", _split_hashtag, text)

    # Reducir spam de caracteres repetidos: jajajaja → jaja
    text = REPEATED_CHARS.sub(r"\1\1", text)

    # Reducir puntuación excesiva: !!!! → !
    text = EXCESSIVE_PUNCT.sub(r"\1", text)

    # Convertir GRITOS a minúsculas (solo palabras de 4+ letras mayúsculas)
    text = EXCESSIVE_CAPS.sub(lambda m: m.group(0).lower(), text)

    # Slang / normalizaciones básicas
    slang_map = {
        r"\bplz\b": "please", r"\bpls\b": "please",
        r"\bthx\b": "thanks", r"\bthnx\b": "thanks",
        r"\bu\b": "you", r"\br\b": "are",
        r"\bur\b": "your", r"\bw/\b": "with",
        r"\bw/o\b": "without", r"\bimo\b": "in my opinion",
        r"\bimho\b": "in my humble opinion",
        r"\btbh\b": "to be honest", r"\bsmh\b": "shaking my head",
        r"\bfyi\b": "for your information",
        r"\bbtw\b": "by the way", r"\bidk\b": "I don't know",
    }
    for pattern, replacement in slang_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = _normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────────────────────
#  DISPATCHER / CORPUS CLEANER
# ─────────────────────────────────────────────────────────────

def clean_text(text: str, source: str) -> str:
    """Dispatch text to the appropriate source-specific cleaner.

    Parameters
    ----------
    text : str
        Raw text to clean.
    source : str
        One of 'academic', 'institutional' (or legacy 'political'), or 'public'.

    Returns
    -------
    str
        Cleaned text.

    Raises
    ------
    ValueError
        If ``source`` is not recognized.
    """
    cleaners = {
        "academic":      clean_academic,
        "institutional": clean_political,
        "political":     clean_political,   # backward compat
        "public":        clean_public,
    }
    if source not in cleaners:
        raise ValueError(
            f"Unknown source '{source}'. Expected one of {list(cleaners.keys())}"
        )
    return cleaners[source](text)


def clean_corpus(
    df: pd.DataFrame,
    text_col: str = "text",
    source: str = "academic",
) -> pd.DataFrame:
    """Apply source-specific cleaning to an entire DataFrame.

    Adds a ``text_clean`` column with the cleaned text and drops
    rows where the cleaned text has fewer than ``MIN_TOKENS`` words.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw texts.
    text_col : str
        Name of the column with raw text.  Default: ``'text'``.
    source : str
        Source type ('academic', 'institutional', 'public').

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``text_clean`` column added.
        Short documents (< MIN_TOKENS words) are removed.
    """
    total = len(df)
    print(f"\n{'='*60}")
    print(f"  Cleaning corpus | source={source} | {total} documents")
    print(f"{'='*60}")

    tqdm.pandas(desc=f"  Cleaning [{source}]")
    df = df.copy()
    df["text_clean"] = df[text_col].fillna("").progress_apply(
        lambda t: clean_text(t, source)
    )

    # Filtrar documentos demasiado cortos
    before = len(df)
    df["_word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    df = df[df["_word_count"] >= MIN_TOKENS].drop(columns=["_word_count"])
    after = len(df)

    print(f"  Removed {before - after} short docs (< {MIN_TOKENS} tokens)")
    print(f"  Remaining: {after} documents")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pruebas rápidas de cada limpiador
    print("\n--- Academic cleaner ---")
    sample_acad = (
        "The SDGs framework [1] emphasizes ESG metrics (doi: 10.1234/abc). "
        "© 2023 Elsevier Ltd. pp. 123-145. vol. 12 issue no. 3. "
        "See https://example.com/paper for details."
    )
    print(f"  IN:  {sample_acad}")
    print(f"  OUT: {clean_academic(sample_acad)}")

    print("\n--- Political cleaner ---")
    sample_pol = (
        "[00:01:23] President Smith: Um, well, you know, este, "
        "we need to address [Music] climate change. Uh, the UNFCCC "
        "has established... bueno, the targets are clear.\n"
        "Secretary Jones: I agree completely."
    )
    print(f"  IN:  {sample_pol}")
    print(f"  OUT: {clean_political(sample_pol)}")

    print("\n--- Public cleaner ---")
    sample_pub = (
        "@user123 This is SOOOOO important!!! 🌍💚 #ClimateChange "
        "#GreenEnergy Check https://example.com plz share tbh 😂😂"
    )
    print(f"  IN:  {sample_pub}")
    print(f"  OUT: {clean_public(sample_pub)}")

    # Prueba de corpus
    print("\n--- Corpus cleaner ---")
    test_df = pd.DataFrame({
        "text": [
            sample_acad,
            "Too short.",
            "Another long enough academic text about sustainability and ESG reporting frameworks.",
        ]
    })
    result = clean_corpus(test_df, text_col="text", source="academic")
    print(result[["text", "text_clean"]].to_string())
