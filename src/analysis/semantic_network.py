"""
semantic_network.py — Redes semánticas de co-ocurrencia
========================================================

Módulo para construir y analizar redes semánticas a partir del
discurso de sustentabilidad.  Utiliza co-ocurrencia de palabras
en ventanas deslizantes para construir grafos ponderados.

Métricas de red:
  - Centralidad de grado, intermediación, cercanía y eigenvector
  - Detección de comunidades (Louvain)
  - Métricas globales: densidad, clustering, modularidad

Autor: VOZ_SUS Project
"""

# ── stdlib ──────────────────────────────────────────────────────────
import sys
import re
from pathlib import Path
from collections import Counter
from itertools import combinations

# ── third-party ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from tqdm import tqdm

# ── local / config ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *  # noqa: E402, F403
from constants import normalize_group_series

# ─────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MÓDULO
# ─────────────────────────────────────────────────────────────────────

NETWORK_RESULTS_DIR = RESULTS_DIR / "networks"
NETWORK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Stopwords combinadas para filtrar tokens (español + inglés)
_STOPWORDS = set([
    # Español
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
    "ellas", "nosotras", "vosotros", "vosotras", "os", "ser", "es",
    "son", "fue", "era", "han", "ha", "haber", "hacer", "tiene",
    "puede", "poder", "ir", "ver", "dar", "saber", "querer", "decir",
    "así", "bien", "aquí", "cada", "si", "después", "mejor", "vez",
    "ahora", "mientras", "mismo", "parte", "más", "tan", "solo",
    # Inglés
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "also", "not", "no", "nor", "so", "if", "then", "there", "here",
    "about", "up", "out", "into", "over", "after", "before", "between",
    "under", "through", "during", "been", "being", "having", "doing",
    "only", "own", "same", "however", "therefore", "thus", "although",
    # Ruido web
    "http", "https", "www", "com",
])

# Longitud mínima de token para considerar como palabra válida
MIN_TOKEN_LENGTH = 3


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Tokeniza un texto en palabras limpias.

    Parameters
    ----------
    text : str
        Texto a tokenizar.

    Returns
    -------
    list[str]
        Lista de tokens (minúsculas, sin puntuación, sin stopwords).
    """
    if not isinstance(text, str):
        return []
    # Solo letras (incluye acentos y ñ)
    tokens = re.findall(r"[a-záéíóúüñ]+", text.lower())
    return [
        t for t in tokens
        if len(t) >= MIN_TOKEN_LENGTH and t not in _STOPWORDS
    ]


def _get_top_words(
    texts: list[str],
    top_n: int = 100,
) -> list[str]:
    """Extrae las N palabras más frecuentes del corpus.

    Parameters
    ----------
    texts : list[str]
        Lista de textos.
    top_n : int
        Número de palabras top a extraer.

    Returns
    -------
    list[str]
        Lista de las top_n palabras más frecuentes.
    """
    counter = Counter()
    for text in texts:
        tokens = _tokenize(text)
        counter.update(tokens)

    top_words = [word for word, _ in counter.most_common(top_n)]
    return top_words


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────

def build_cooccurrence_matrix(
    texts: list[str],
    top_n: int = 100,
    window_size: int = 5,
) -> pd.DataFrame:
    """Construye una matriz de co-ocurrencia usando ventana deslizante.

    Para cada documento, se utiliza una ventana deslizante sobre los
    tokens filtrados.  Dos palabras co-ocurren si aparecen dentro de
    la misma ventana.

    Parameters
    ----------
    texts : list[str]
        Lista de textos del corpus.
    top_n : int
        Número de palabras top a incluir en la matriz.
    window_size : int
        Tamaño de la ventana deslizante (en tokens).

    Returns
    -------
    pd.DataFrame
        Matriz de adyacencia (simétrica) con co-ocurrencias.
    """
    print(f"\n  Building co-occurrence matrix | top_n={top_n}, window={window_size}")

    # 1. Obtener vocabulario top
    top_words = _get_top_words(texts, top_n)
    top_set = set(top_words)
    print(f"  Vocabulary: {len(top_words)} words")

    # 2. Contar co-ocurrencias con ventana deslizante
    cooc_counts = Counter()

    for text in tqdm(texts, desc="  Co-occurrence"):
        tokens = _tokenize(text)
        # Filtrar solo tokens del vocabulario top
        filtered = [t for t in tokens if t in top_set]

        # Ventana deslizante
        for i in range(len(filtered)):
            window_end = min(i + window_size, len(filtered))
            for j in range(i + 1, window_end):
                pair = tuple(sorted([filtered[i], filtered[j]]))
                if pair[0] != pair[1]:  # Evitar auto-co-ocurrencia
                    cooc_counts[pair] += 1

    # 3. Construir matriz de adyacencia
    matrix = pd.DataFrame(0, index=top_words, columns=top_words, dtype=int)

    for (w1, w2), count in cooc_counts.items():
        if w1 in matrix.index and w2 in matrix.columns:
            matrix.loc[w1, w2] = count
            matrix.loc[w2, w1] = count

    total_pairs = len(cooc_counts)
    print(f"  ✓ Matrix built — {total_pairs:,} unique pairs with co-occurrences")

    return matrix


def create_network(
    cooc_matrix: pd.DataFrame,
    min_weight: int = 5,
) -> nx.Graph:
    """Crea un grafo de NetworkX a partir de la matriz de co-ocurrencia.

    Parameters
    ----------
    cooc_matrix : pd.DataFrame
        Matriz de adyacencia (simétrica) de co-ocurrencias.
    min_weight : int
        Peso mínimo de arista para incluir en el grafo.

    Returns
    -------
    nx.Graph
        Grafo no dirigido ponderado.
    """
    print(f"\n  Creating network graph | min_weight={min_weight}")

    G = nx.Graph()
    words = cooc_matrix.index.tolist()

    # Agregar nodos
    G.add_nodes_from(words)

    # Agregar aristas (solo triángulo superior para evitar duplicados)
    edges_added = 0
    for i, w1 in enumerate(words):
        for j in range(i + 1, len(words)):
            w2 = words[j]
            weight = cooc_matrix.loc[w1, w2]
            if weight >= min_weight:
                G.add_edge(w1, w2, weight=int(weight))
                edges_added += 1

    # Remover nodos aislados (sin conexiones)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    print(f"  ✓ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if isolated:
        print(f"    Removed {len(isolated)} isolated nodes")

    return G


def compute_network_metrics(G: nx.Graph) -> pd.DataFrame:
    """Calcula métricas de centralidad y detecta comunidades.

    Parameters
    ----------
    G : nx.Graph
        Grafo de co-ocurrencia.

    Returns
    -------
    pd.DataFrame
        DataFrame con métricas por nodo: degree_centrality,
        betweenness_centrality, closeness_centrality,
        eigenvector_centrality, community.
    """
    print(f"\n  Computing network metrics for {G.number_of_nodes()} nodes ...")

    if G.number_of_nodes() == 0:
        print("  ⚠ Empty graph — returning empty DataFrame")
        return pd.DataFrame()

    # ── Centralidades ───────────────────────────────────────────────
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight="weight")
    closeness_c = nx.closeness_centrality(G)

    try:
        eigenvector_c = nx.eigenvector_centrality(
            G, max_iter=500, weight="weight"
        )
    except nx.PowerIterationFailedConvergence:
        print("  ⚠ Eigenvector centrality did not converge — using zeros")
        eigenvector_c = {node: 0.0 for node in G.nodes()}

    # ── Detección de comunidades (Louvain) ──────────────────────────
    print("  Detecting communities (Louvain) ...")
    communities = louvain_communities(G, weight="weight", seed=42)
    node_community = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_community[node] = comm_id

    n_communities = len(communities)
    print(f"  ✓ Found {n_communities} communities")

    # ── Construir DataFrame ─────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "word": list(G.nodes()),
        "degree_centrality": [round(degree_c[n], 6) for n in G.nodes()],
        "betweenness_centrality": [round(betweenness_c[n], 6) for n in G.nodes()],
        "closeness_centrality": [round(closeness_c[n], 6) for n in G.nodes()],
        "eigenvector_centrality": [round(eigenvector_c[n], 6) for n in G.nodes()],
        "community": [node_community.get(n, -1) for n in G.nodes()],
        "degree": [G.degree(n) for n in G.nodes()],
        "weighted_degree": [
            sum(d["weight"] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        ],
    })

    metrics_df = metrics_df.sort_values(
        "degree_centrality", ascending=False
    ).reset_index(drop=True)

    return metrics_df


def build_networks_per_group(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    source_col: str = "source",
    top_n: int = 100,
    window_size: int = 5,
    min_weight: int = 5,
) -> dict:
    """Construye redes semánticas separadas para cada grupo de fuente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con textos y columna de fuente.
    text_col : str
        Columna de texto.
    source_col : str
        Columna de fuente.
    top_n : int
        Número de palabras top por red.
    window_size : int
        Tamaño de ventana deslizante.
    min_weight : int
        Peso mínimo de arista.

    Returns
    -------
    dict
        Estructura:
        {
            'academic':      {'graph': nx.Graph, 'metrics': pd.DataFrame, 'matrix': pd.DataFrame},
            'institutional': { ... },
            'public':        { ... },
        }
    """
    print(f"\n{'='*60}")
    print(f"  Building Semantic Networks per Group")
    print(f"{'='*60}")

    results = {}
    sources = sorted(df[source_col].unique())

    for i, source in enumerate(sources, 1):
        print(f"\n  Processing: {source} | {i}/{len(sources)}")
        subset = df[df[source_col] == source]
        texts = subset[text_col].dropna().tolist()

        if len(texts) < 50:
            print(f"  ⚠ Skipping '{source}' — only {len(texts)} texts (need ≥50)")
            continue

        # Construir red
        matrix = build_cooccurrence_matrix(texts, top_n=top_n, window_size=window_size)
        G = create_network(matrix, min_weight=min_weight)
        metrics = compute_network_metrics(G)

        # Guardar resultados
        group_dir = NETWORK_RESULTS_DIR / source
        group_dir.mkdir(parents=True, exist_ok=True)

        matrix.to_csv(group_dir / f"cooc_matrix_{source}.csv", encoding="utf-8-sig")
        metrics.to_csv(group_dir / f"network_metrics_{source}.csv", index=False, encoding="utf-8-sig")

        # Exportar HTML interactivo
        export_network_html(
            G,
            output_path=group_dir / f"network_{source}.html",
            title=f"Semantic Network — {source.title()}",
        )

        # Guardar grafo como GraphML para análisis externo (ej. Gephi)
        nx.write_graphml(G, str(group_dir / f"network_{source}.graphml"))

        results[source] = {
            "graph": G,
            "metrics": metrics,
            "matrix": matrix,
        }

        print(f"  ✓ {source}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return results


def compare_networks(networks: dict) -> pd.DataFrame:
    """Compara métricas globales de red entre grupos.

    Parameters
    ----------
    networks : dict
        Diccionario devuelto por `build_networks_per_group`.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa con métricas a nivel de red y
        top-10 nodos centrales.
    """
    print(f"\n  Comparing networks across {len(networks)} groups ...")

    rows = []
    top_nodes_per_group = {}

    for group_name, data in networks.items():
        G = data["graph"]
        metrics = data["metrics"]

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Métricas globales
        density = round(nx.density(G), 6) if n_nodes > 0 else 0
        avg_clustering = round(nx.average_clustering(G, weight="weight"), 6) if n_nodes > 0 else 0

        # Modularidad (usando comunidades Louvain)
        if n_nodes > 0 and "community" in metrics.columns:
            communities_dict = metrics.groupby("community")["word"].apply(set).tolist()
            modularity = round(nx.algorithms.community.modularity(G, communities_dict), 6)
        else:
            modularity = 0

        # Diámetro (solo para grafos conexos)
        if n_nodes > 0 and nx.is_connected(G):
            diameter = nx.diameter(G)
        elif n_nodes > 0:
            # Usar el componente conexo más grande
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
        else:
            diameter = 0

        # Grado promedio
        avg_degree = round(np.mean([d for _, d in G.degree()]), 2) if n_nodes > 0 else 0

        rows.append({
            "group": group_name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_clustering": avg_clustering,
            "modularity": modularity,
            "diameter": diameter,
            "avg_degree": avg_degree,
        })

        # Top-10 nodos centrales
        if not metrics.empty:
            top10 = metrics.head(10)["word"].tolist()
            top_nodes_per_group[group_name] = top10

    comparison_df = pd.DataFrame(rows)

    # Agregar top-10 nodos como columna
    comparison_df["top_10_central_nodes"] = comparison_df["group"].map(
        lambda g: ", ".join(top_nodes_per_group.get(g, []))
    )

    # Guardar
    output_path = NETWORK_RESULTS_DIR / "network_comparison.csv"
    comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Comparison saved → {output_path.name}")

    print(f"\n{comparison_df.to_string(index=False)}")

    return comparison_df


def export_network_html(
    G: nx.Graph,
    output_path: Path,
    title: str = "Semantic Network",
    height: str = "750px",
    width: str = "100%",
) -> None:
    """Exporta el grafo como visualización HTML interactiva con PyVis.

    Parameters
    ----------
    G : nx.Graph
        Grafo de co-ocurrencia.
    output_path : Path
        Ruta del archivo HTML de salida.
    title : str
        Título de la visualización.
    height : str
        Altura del canvas.
    width : str
        Ancho del canvas.
    """
    from pyvis.network import Network

    print(f"  Exporting interactive HTML → {output_path.name}")

    if G.number_of_nodes() == 0:
        print("  ⚠ Empty graph — skipping HTML export")
        return

    # Crear red PyVis
    net = Network(
        height=height,
        width=width,
        bgcolor="#222222",
        font_color="white",
        heading=title,
        notebook=False,
    )

    # Configurar física para layout agradable
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
    )

    # Calcular grados para escalar tamaño de nodos
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # Calcular comunidades para colorear
    try:
        communities = louvain_communities(G, weight="weight", seed=42)
        node_comm = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                node_comm[node] = comm_id
    except Exception:
        node_comm = {n: 0 for n in G.nodes()}

    # Paleta de colores para comunidades
    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
        "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    ]

    # Agregar nodos
    for node in G.nodes():
        size = 10 + (degrees[node] / max_degree) * 40
        comm_id = node_comm.get(node, 0)
        color = colors[comm_id % len(colors)]
        net.add_node(
            node,
            label=node,
            size=size,
            color=color,
            title=f"{node}\nDegree: {degrees[node]}\nCommunity: {comm_id}",
        )

    # Agregar aristas
    max_weight = max(
        (d.get("weight", 1) for _, _, d in G.edges(data=True)),
        default=1,
    )
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        # Escalar grosor de arista
        width = 0.5 + (weight / max_weight) * 5
        net.add_edge(u, v, value=weight, width=width, title=f"Weight: {weight}")

    # Habilitar botones de configuración
    net.show_buttons(filter_=["physics"])

    # Guardar HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    print(f"  ✓ HTML saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# MAIN — ejecución independiente para pruebas
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VOZ_SUS — Semantic Network Module (standalone test)")
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
    df = df[df[text_col].notna() & (df[text_col].str.len() > 30)].reset_index(drop=True)
    print(f"  After filtering: {len(df):,} documents")

    # Ejecutar por grupo si existe columna 'source'
    if "source" in df.columns:
        networks = build_networks_per_group(df, text_col=text_col)
        comparison = compare_networks(networks)
    else:
        # Red completa
        texts = df[text_col].tolist()
        matrix = build_cooccurrence_matrix(texts, top_n=100, window_size=5)
        G = create_network(matrix, min_weight=5)
        metrics = compute_network_metrics(G)

        # Guardar
        matrix.to_csv(NETWORK_RESULTS_DIR / "cooc_matrix_full.csv", encoding="utf-8-sig")
        metrics.to_csv(NETWORK_RESULTS_DIR / "network_metrics_full.csv", index=False, encoding="utf-8-sig")
        export_network_html(G, NETWORK_RESULTS_DIR / "network_full.html", title="Full Corpus")

        print(f"\n  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"\n  Top-10 central nodes:")
        print(metrics.head(10).to_string(index=False))

    print("\n  ✓ Semantic network analysis complete!")
