"""
config.py — Configuración central del proyecto VOZ_SUS
======================================================

"Tres Voces de la Sustentabilidad"
Análisis comparativo del discurso de sustentabilidad en tres esferas:
  • Académica      (OpenAlex)
  • Institucional  (YouTube — canales institucionales)
  • Pública        (YouTube — comentarios ciudadanos)

Todas las rutas, parámetros de búsqueda, modelos y constantes se
definen aquí para mantener un único punto de verdad en todo el pipeline.

Autor: Ana J.
Última actualización: 2026-06-14
"""

from pathlib import Path
from constants import normalize_group

# ─────────────────────────────────────────────────────────────────────
# 1. RUTAS DEL PROYECTO
# ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

# Datos crudos — separados por fuente
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
RAW_ACADEMIC    = RAW_DIR  / "academic"      # Artículos científicos (OpenAlex)
RAW_INSTITUTIONAL = RAW_DIR / "political"     # Discurso institucional (YouTube transcripts)
RAW_PUBLIC      = RAW_DIR  / "public"        # Opinión pública (YouTube comments)

# Datos procesados y finales
PROCESSED_DIR   = DATA_DIR / "processed"     # Textos limpios y normalizados
FINAL_DIR       = DATA_DIR / "final"         # Datasets listos para análisis

# Resultados
RESULTS_DIR     = BASE_DIR / "results"
FIGURES_DIR     = RESULTS_DIR / "figures"
TABLES_DIR      = RESULTS_DIR / "tables"
MODELS_DIR      = RESULTS_DIR / "models"     # Modelos BERTopic serializados
REPORTS_DIR     = RESULTS_DIR / "reports"
COMPARISON_DIR  = RESULTS_DIR / "comparison"

# Logs
LOGS_DIR        = BASE_DIR / "logs"

# Crear todas las carpetas al importar config
for _d in [
    RAW_ACADEMIC, RAW_INSTITUTIONAL, RAW_PUBLIC,
    PROCESSED_DIR, FINAL_DIR,
    FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR, COMPARISON_DIR,
    LOGS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 2. CLAVES DE API
# ─────────────────────────────────────────────────────────────────────

# YouTube Data API v3 — reemplazar con tu clave real
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

# OpenAlex — correo para acceso cortés (polite pool)
OPENALEX_EMAIL = "your_email@example.com"


# ─────────────────────────────────────────────────────────────────────
# 3. PARÁMETROS DE BÚSQUEDA — OpenAlex (Voz Académica)
# ─────────────────────────────────────────────────────────────────────

OPENALEX_SEARCH = {
    # Consultas en inglés y español
    "queries_en": [
        "sustainability discourse",
        "sustainable development goals",
        "sustainability transition",
        "climate change discourse",
        "environmental sustainability",
        "green economy",
        "sustainability policy",
        "sustainability governance",
        "circular economy",
        "energy transition",
    ],
    "queries_es": [
        "discurso de sustentabilidad",
        "objetivos de desarrollo sustentable",
        "transición sustentable",
        "discurso cambio climático",
        "sustentabilidad ambiental",
        "economía verde",
        "política de sustentabilidad",
        "gobernanza sustentable",
        "economía circular",
        "transición energética",
    ],

    # Rango temporal del estudio
    "date_from": "2015-01-01",
    "date_to":   "2026-12-31",

    # Filtros adicionales
    "type": "article",               # Solo artículos revisados por pares
    "is_oa": None,                   # None = todos, True = solo open access
    "per_page": 200,                 # Resultados por página
    "max_results_per_query": 2000,   # Límite por consulta
    "sort": "cited_by_count:desc",   # Ordenar por citas
}


# ─────────────────────────────────────────────────────────────────────
# 4. PARÁMETROS DE BÚSQUEDA — YouTube (Voz Institucional + Pública)
# ─────────────────────────────────────────────────────────────────────

# ── 4a. Canales institucionales (discurso institucional) ─────────────
YOUTUBE_CHANNELS = {
    # Organismos internacionales
    "UNFCCC":               "UCMGEyz3MRin4MxRAkuWxQBQ",  # UN Climate Change
    "UN":                   "UCvDtRhHjEOPjk3C9umhljg",   # United Nations
    "World_Economic_Forum": "UCw-kH-Od73XDAt7qtH9uBYA",  # WEF
    "UNEP":                 "UCqJLnl-GXXMwU3IFGHmOCA",   # UN Environment Programme

    # Parlamentos y congresos
    "European_Parliament":  "UC44WswjLPrdFEB2hiaOuXng",  # European Parliament
    "Congreso_Mexico":      "UCsnSTlA-PYpTLjw4ev2WNRQ",  # Canal del Congreso MX
    "Senado_Mexico":        "UCrKCAghLJvwW-0dF_2HsBuQ",  # Senado de la República MX

    # Gobiernos
    "EU_Commission":        "UCcEq0IkNJjCajHKIiLNmJTg",  # European Commission
    "Gobierno_Mexico":      "UCIqxD3jfg7jGqAOAGw2SzMA",  # Gobierno de México
    "UK_Parliament":        "UCQqJlrWhSBSwpSMXkkM09Lw",  # UK Parliament
}

# ── 4b. Búsquedas temáticas en YouTube ───────────────────────────────
YOUTUBE_SEARCH = {
    "queries_en": [
        "sustainability speech",
        "climate change policy",
        "sustainable development goals",
        "COP climate summit",
        "green new deal",
        "net zero emissions",
        "Paris Agreement",
        "energy transition policy",
    ],
    "queries_es": [
        "discurso sustentabilidad",
        "política cambio climático",
        "objetivos desarrollo sustentable",
        "cumbre clima COP",
        "pacto verde",
        "emisiones netas cero",
        "Acuerdo de París",
        "transición energética política",
    ],

    # Filtros de búsqueda
    "date_from": "2015-01-01",
    "date_to":   "2026-12-31",
    "max_results_per_query": 50,      # Máximo videos por consulta
    "order": "relevance",             # relevance | date | viewCount
    "video_duration": "medium",       # short | medium | long
    "relevance_language": "en",       # Idioma preferido para relevancia
    "caption": "closedCaption",       # Solo videos con subtítulos
}

# ── 4c. Comentarios de YouTube (opinión pública) ─────────────────────
YOUTUBE_COMMENTS = {
    "max_comments_per_video": 500,    # Límite de comentarios por video
    "min_likes": 2,                   # Filtro mínimo de likes
    "order": "relevance",             # relevance | time
    "include_replies": False,         # Incluir respuestas a comentarios
    "max_total_comments": 50_000,     # Límite global del corpus
}

# ── 4d. Transcripciones de YouTube ───────────────────────────────────
YOUTUBE_TRANSCRIPTS = {
    "preferred_languages": ["en", "es"],  # Idiomas de transcripción preferidos
    "fallback_auto_generated": True,      # Usar auto-generadas si no hay manual
}


# ─────────────────────────────────────────────────────────────────────
# 5. PREPROCESAMIENTO DE TEXTO
# ─────────────────────────────────────────────────────────────────────

# Longitud mínima de documento (en tokens) para incluir en el análisis
MIN_TOKENS = 15
MAX_TOKENS = 10_000

# Modelos de spaCy para lematización y POS tagging
SPACY_MODELS = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
}

# Detección de idioma
LANG_DETECT_THRESHOLD = 0.85  # Confianza mínima para asignar idioma

# ── Stopwords personalizadas por fuente y idioma ─────────────────────
# Estas complementan las stopwords por defecto de spaCy

CUSTOM_STOPWORDS = {
    # ── Académico ────────────────────────────────────────────────────
    "academic": {
        "en": {
            "abstract", "introduction", "conclusion", "methodology",
            "literature", "review", "paper", "study", "research",
            "findings", "results", "analysis", "framework", "approach",
            "however", "therefore", "moreover", "furthermore",
            "et", "al", "fig", "figure", "table", "appendix",
            "doi", "journal", "volume", "issue", "pp", "pages",
            "university", "department", "author", "corresponding",
            "https", "http", "www", "copyright", "elsevier",
            "springer", "wiley", "rights", "reserved",
        },
        "es": {
            "resumen", "introducción", "conclusión", "metodología",
            "literatura", "revisión", "artículo", "estudio", "investigación",
            "hallazgos", "resultados", "análisis", "marco", "enfoque",
            "sin embargo", "por lo tanto", "además", "asimismo",
            "et", "al", "fig", "figura", "tabla", "apéndice",
            "doi", "revista", "volumen", "número", "pp", "páginas",
            "universidad", "departamento", "autor", "correspondencia",
            "https", "http", "www", "derechos", "reservados",
        },
    },

    # ── Institucional ────────────────────────────────────────────────
    "institutional": {
        "en": {
            "thank", "thanks", "mr", "mrs", "president", "chairman",
            "chairwoman", "honourable", "distinguished", "colleagues",
            "minister", "secretary", "general", "assembly",
            "committee", "commission", "parliament", "council",
            "ladies", "gentlemen", "applause", "laughter",
            "please", "would", "like", "say", "think",
            "speaker", "floor", "yield", "time", "minute",
        },
        "es": {
            "gracias", "señor", "señora", "presidente", "presidenta",
            "diputado", "diputada", "senador", "senadora",
            "honorable", "distinguido", "distinguida", "compañeros",
            "ministro", "ministra", "secretario", "secretaria",
            "comisión", "comité", "parlamento", "congreso", "cámara",
            "señoras", "señores", "aplausos", "risas",
            "favor", "quisiera", "decir", "creo", "pensar",
            "orador", "tribuna", "turno", "tiempo", "minuto",
        },
    },

    # ── Público (comentarios) ────────────────────────────────────────
    "public": {
        "en": {
            "like", "video", "subscribe", "channel", "watch",
            "comment", "share", "click", "link", "description",
            "lol", "lmao", "omg", "wow", "yeah", "yep", "nope",
            "guys", "gonna", "wanna", "gotta", "kinda",
            "stuff", "thing", "things", "something", "anything",
            "really", "actually", "literally", "basically",
            "great", "good", "nice", "awesome", "amazing",
            "https", "http", "www", "com",
        },
        "es": {
            "like", "video", "suscribir", "suscríbete", "canal",
            "ver", "comentario", "compartir", "clic", "enlace",
            "descripción", "jaja", "jajaja", "xd", "xdd",
            "bueno", "pues", "osea", "ósea", "neta", "wey",
            "cosas", "cosa", "algo", "nada",
            "verdad", "realmente", "literal", "básicamente",
            "genial", "bueno", "bonito", "increíble", "excelente",
            "https", "http", "www", "com",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────
# 6. BERTopic — MODELADO DE TÓPICOS
# ─────────────────────────────────────────────────────────────────────

# Modelo de embeddings multilingüe (ligero, apto para CPU)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# UMAP — reducción de dimensionalidad (optimizado para CPU)
UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": 42,
    "low_memory": True,        # Crítico para ejecución en CPU
    "n_jobs": 1,               # Un solo hilo para estabilidad en CPU
}

# HDBSCAN — clustering (optimizado para CPU)
HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eef",   # Excess of mass
    "prediction_data": True,
}

# Parámetros generales de BERTopic
BERTOPIC_PARAMS = {
    "language": "multilingual",
    "top_n_words": 10,           # Palabras representativas por tópico
    "nr_topics": "auto",         # 'auto' o un int para reducir tópicos
    "calculate_probabilities": False,  # False ahorra memoria en CPU
    "verbose": True,
}

# Vectorizer para la representación c-TF-IDF
VECTORIZER_PARAMS = {
    "min_df": 5,                # Frecuencia mínima de documento
    "max_df": 0.95,             # Frecuencia máxima (elimina términos muy comunes)
    "ngram_range": (1, 3),      # Unigramas a trigramas
    "stop_words": None,         # Se aplican stopwords personalizadas antes
}

# CountVectorizer para BERTopic
CTFIDF_PARAMS = {
    "reduce_frequent_words": True,
    "bm25_weighting": False,
}


# ─────────────────────────────────────────────────────────────────────
# 7. ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

# Modelos de Hugging Face para análisis de sentimiento
SENTIMENT_MODELS = {
    # pysentimiento — entrenado específicamente para español
    "es": "pysentimiento/robertuito-sentiment-analysis",
    # Modelo multilingüe para inglés y fallback
    "en": "nlptown/bert-base-multilingual-uncased-sentiment",
}

# Modelo alternativo multilingüe (más ligero)
SENTIMENT_MODEL_MULTILINGUAL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

# Tamaño de batch para inferencia en CPU
SENTIMENT_BATCH_SIZE = 16
SENTIMENT_MAX_LENGTH = 512  # Longitud máxima de tokens por texto


# ─────────────────────────────────────────────────────────────────────
# 8. ANÁLISIS DE REDES
# ─────────────────────────────────────────────────────────────────────

NETWORK_PARAMS = {
    "min_cooccurrence": 3,       # Co-ocurrencia mínima para crear arista
    "top_n_nodes": 100,          # Máximo de nodos en la red
    "layout": "spring",          # spring | kamada_kawai | circular
    "node_size_metric": "degree", # degree | betweenness | eigenvector
}


# ─────────────────────────────────────────────────────────────────────
# 9. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────

VIZ_CONFIG = {
    "dpi": 300,
    "fig_size_single": (10, 6),
    "fig_size_wide": (16, 8),
    "fig_size_square": (10, 10),
    "fig_size_tall": (10, 14),

    # Paleta de colores del proyecto — tonos neutros tipo académico
    # Inspirada en paletas tipo viridis/Blues, fondo claro
    "colormap": "Blues",
    "source_colors": {
        "academic":      "#1B4F72",  # Azul oscuro (profundo, serio)
        "institutional": "#2E86C1",  # Azul medio (institucional)
        "public":        "#85C1E9",  # Azul claro (ciudadano, accesible)
    },
    "font_family": "serif",
    "style": "seaborn-v0_8-whitegrid",

    # Formatos de exportación
    "save_formats": ["png", "pdf"],
}

# Configuración de WordCloud
WORDCLOUD_CONFIG = {
    "width": 1600,
    "height": 800,
    "max_words": 200,
    "background_color": "white",
    "colormap": "Blues",
    "contour_width": 1,
    "contour_color": "#2E86C1",
    "prefer_horizontal": 0.7,
}

# ─────────────────────────────────────────────────────────────────────
# Variables de conveniencia a nivel top (usadas por plots.py)
# ─────────────────────────────────────────────────────────────────────
SOURCE_COLORS = VIZ_CONFIG["source_colors"]
FIGURE_DPI    = VIZ_CONFIG["dpi"]
FIGURE_FORMAT = VIZ_CONFIG["save_formats"]


# ─────────────────────────────────────────────────────────────────────
# ALIASES PLANOS — usados por los módulos de recolección
# ─────────────────────────────────────────────────────────────────────
# Estos aliases mapean las variables que los collectors importan
# directamente desde las estructuras dict de arriba.

# — OpenAlex (academic_collector.py) —
CONTACT_EMAIL               = OPENALEX_EMAIL
OPENALEX_SEARCH_QUERY       = "|".join(
    OPENALEX_SEARCH["queries_en"][:3] + OPENALEX_SEARCH["queries_es"][:3]
)
OPENALEX_DATE_FROM           = OPENALEX_SEARCH["date_from"]
OPENALEX_DATE_TO             = OPENALEX_SEARCH["date_to"]
OPENALEX_ARTICLE_TYPE        = OPENALEX_SEARCH["type"]
OPENALEX_PER_PAGE            = OPENALEX_SEARCH["per_page"]
OPENALEX_MAX_RESULTS         = OPENALEX_SEARCH.get("max_results_per_query", 2000) * len(
    OPENALEX_SEARCH["queries_en"]
)
ACADEMIC_CSV                 = RAW_ACADEMIC / "academic_corpus.csv"
ACADEMIC_CHECKPOINT_EVERY    = 100

# — YouTube institutional (political_collector.py) —
INSTITUTIONAL_CHANNELS            = YOUTUBE_CHANNELS
INSTITUTIONAL_CSV                 = RAW_INSTITUTIONAL / "political_corpus.csv"
INSTITUTIONAL_CHECKPOINT_EVERY    = 50
INSTITUTIONAL_MAX_RESULTS_PER_QUERY = YOUTUBE_SEARCH["max_results_per_query"]
INSTITUTIONAL_MAX_VIDEOS          = 1000
SEARCH_QUERIES_ALL            = YOUTUBE_SEARCH["queries_en"] + YOUTUBE_SEARCH["queries_es"]
YOUTUBE_REQUEST_DELAY         = 0.5   # segundos entre requests
YOUTUBE_SEARCH_ORDER          = YOUTUBE_SEARCH["order"]

# Backward-compat aliases (deprecated — use INSTITUTIONAL_* instead)
POLITICAL_CHANNELS              = INSTITUTIONAL_CHANNELS
POLITICAL_CSV                   = INSTITUTIONAL_CSV
POLITICAL_CHECKPOINT_EVERY      = INSTITUTIONAL_CHECKPOINT_EVERY
POLITICAL_MAX_RESULTS_PER_QUERY = INSTITUTIONAL_MAX_RESULTS_PER_QUERY
POLITICAL_MAX_VIDEOS            = INSTITUTIONAL_MAX_VIDEOS
RAW_POLITICAL                   = RAW_INSTITUTIONAL

# — YouTube public (public_collector.py) —
PUBLIC_CSV                    = RAW_PUBLIC / "public_corpus.csv"
PUBLIC_CHECKPOINT_EVERY       = 100
PUBLIC_MAX_COMMENTS_PER_VIDEO = YOUTUBE_COMMENTS["max_comments_per_video"]
PUBLIC_MIN_LIKES              = YOUTUBE_COMMENTS["min_likes"]
PUBLIC_EXTRA_SEARCH_QUERIES   = YOUTUBE_SEARCH["queries_en"][:4]
PUBLIC_EXTRA_VIDEOS           = 50


# ─────────────────────────────────────────────────────────────────────
# 10. LÍNEA TEMPORAL — EVENTOS CLAVE DE SUSTENTABILIDAD
# ─────────────────────────────────────────────────────────────────────
# Estos hitos se usan para contextualizar el análisis temporal

KEY_EVENTS = {
    "2015-09": {
        "name": "Agenda 2030 / ODS",
        "name_en": "2030 Agenda / SDGs",
        "description": "Adopción de los 17 Objetivos de Desarrollo Sustentable por la ONU",
    },
    "2015-12": {
        "name": "Acuerdo de París",
        "name_en": "Paris Agreement",
        "description": "COP21 — Acuerdo climático vinculante para limitar calentamiento a 1.5 C",
    },
    "2018-10": {
        "name": "Informe IPCC 1.5 C",
        "name_en": "IPCC 1.5 C Report",
        "description": "Informe especial del IPCC sobre los impactos de 1.5 C de calentamiento",
    },
    "2019-09": {
        "name": "Cumbre Acción Climática ONU",
        "name_en": "UN Climate Action Summit",
        "description": "Cumbre en Nueva York — Discurso de Greta Thunberg ante la ONU",
    },
    "2019-12": {
        "name": "Pacto Verde Europeo",
        "name_en": "European Green Deal",
        "description": "Presentación del European Green Deal por la Comisión Europea",
    },
    "2020-03": {
        "name": "Pandemia COVID-19",
        "name_en": "COVID-19 Pandemic",
        "description": "Inicio de la pandemia — impacto en agendas de sustentabilidad",
    },
    "2021-11": {
        "name": "COP26 Glasgow",
        "name_en": "COP26 Glasgow",
        "description": "Pacto Climático de Glasgow — compromiso de reducir carbón",
    },
    "2022-11": {
        "name": "COP27 Sharm el-Sheikh",
        "name_en": "COP27 Sharm el-Sheikh",
        "description": "Fondo de pérdidas y daños para países vulnerables",
    },
    "2022-12": {
        "name": "COP15 Biodiversidad",
        "name_en": "COP15 Biodiversity",
        "description": "Marco Global de Biodiversidad Kunming-Montreal",
    },
    "2023-12": {
        "name": "COP28 Dubái",
        "name_en": "COP28 Dubai",
        "description": "Primer balance mundial — compromiso de transición fuera de combustibles fósiles",
    },
    "2024-11": {
        "name": "COP29 Bakú",
        "name_en": "COP29 Baku",
        "description": "Meta de financiamiento climático — 300 mil millones USD anuales",
    },
    "2025-11": {
        "name": "COP30 Belém",
        "name_en": "COP30 Belém",
        "description": "COP en la Amazonía — nuevas NDC alineadas a 1.5 C",
    },
}


# ─────────────────────────────────────────────────────────────────────
# 11. ETIQUETAS DE FUENTES (para DataFrames y gráficos)
# ─────────────────────────────────────────────────────────────────────

# Labels for figures and tables (English — publication language)
SOURCE_LABELS = {
    "academic":      "Academic Voice",
    "institutional": "Institutional Voice",
    "public":        "Public Voice",
}

# Aliases
SOURCE_LABELS_EN = SOURCE_LABELS  # backward compat

SOURCE_LABELS_ES = {
    "academic":      "Voz Académica",
    "institutional": "Voz Institucional",
    "public":        "Voz Pública",
}


# ─────────────────────────────────────────────────────────────────────
# 12. CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
RANDOM_SEED  = RANDOM_STATE  # Alias usado por algunos módulos
LOG_LEVEL = "INFO"
N_JOBS = 1          # Número de procesos paralelos (1 = serial, seguro en CPU)


# ─────────────────────────────────────────────────────────────────────
# 13. VALIDACIÓN AL IMPORTAR
# ─────────────────────────────────────────────────────────────────────

def validate_config() -> None:
    """Verifica que la configuración mínima esté definida correctamente."""
    warnings = []

    if YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
        warnings.append(
            "  YOUTUBE_API_KEY no configurada. "
            "Edita config.py y reemplaza 'YOUR_API_KEY_HERE' con tu clave real."
        )

    if OPENALEX_EMAIL == "your_email@example.com":
        warnings.append(
            "  OPENALEX_EMAIL no configurado. "
            "Añade tu correo para acceso cortés al API de OpenAlex."
        )

    for w in warnings:
        print(w)


if __name__ == "__main__":
    print("=" * 70)
    print("  VOZ_SUS — Configuración del proyecto")
    print("=" * 70)
    print(f"\n  BASE_DIR:      {BASE_DIR}")
    print(f"  DATA_DIR:      {DATA_DIR}")
    print(f"  RESULTS_DIR:   {RESULTS_DIR}")
    print(f"  LOGS_DIR:      {LOGS_DIR}")
    print(f"\n  Embedding:     {EMBEDDING_MODEL}")
    print(f"  RANDOM_STATE:  {RANDOM_STATE}")
    print(f"  LOG_LEVEL:     {LOG_LEVEL}")
    print(f"\n  Eventos clave: {len(KEY_EVENTS)}")
    print(f"  Canales YT:    {len(YOUTUBE_CHANNELS)}")
    print()
    validate_config()
    print("\n  Configuración cargada correctamente.")
