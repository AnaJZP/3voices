"""
plots.py — Visualizaciones de calidad para publicación (VOZ_SUS)
================================================================

Genera figuras para el artículo de investigación sobre discurso de
sustentabilidad. Todas las figuras se guardan en 300 DPI como PNG y PDF
con estilo académico (fuente serif, ejes limpios, paleta consistente).
"""

# ── Stdlib ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Local ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *


# ─────────────────────────────────────────────────────────────────────
# ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────

def setup_plot_style() -> None:
    """Configura el estilo global de matplotlib para publicación académica.

    Establece fuente serif, tamaños apropiados para figuras de
    artículo (ancho simple y doble columna), y estilo seaborn
    'whitegrid' para claridad.
    """
    sns.set_style("whitegrid")

    plt.rcParams.update({
        # Fuentes
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Figuras
        "figure.figsize": (8, 5),
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Líneas y ejes
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    })
    print("  ✓ Plot style configured (serif, 300 DPI, whitegrid)")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Guarda una figura en todos los formatos configurados (PNG + PDF).

    Parameters
    ----------
    fig : plt.Figure
        Figura de matplotlib a guardar.
    output_path : Path
        Ruta base sin extensión. Se generarán archivos .png y .pdf.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in FIGURE_FORMAT:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")

    plt.close(fig)


def _get_color_palette() -> list[str]:
    """Devuelve la paleta de colores ordenada para los 3 grupos."""
    return [
        SOURCE_COLORS["academic"],
        SOURCE_COLORS["institutional"],
        SOURCE_COLORS["public"],
    ]


def _prepare_events(events: dict | None = None) -> dict[int, str]:
    """Convert KEY_EVENTS to a flat {year_int: label_str} dict for plots.

    KEY_EVENTS has string keys like '2015-09' and dict values with
    'name', 'name_en', 'description'.  This helper extracts the English
    name and uses the numeric year for easy comparison with DataFrame
    year columns.
    """
    if events is None:
        events = KEY_EVENTS

    flat: dict[int, str] = {}
    for key, val in events.items():
        try:
            year_int = int(str(key)[:4])
        except (ValueError, TypeError):
            continue
        if isinstance(val, dict):
            label = val.get("name_en", val.get("name", str(val)))
        else:
            label = str(val)
        # Keep only one event per year (the latest month wins)
        flat[year_int] = label
    return flat


# ─────────────────────────────────────────────────────────────────────
# HEATMAP DE DISTRIBUCIÓN DE TEMAS
# ─────────────────────────────────────────────────────────────────────

def plot_topic_distribution_heatmap(
    topic_data: dict,
    output_path: Path,
) -> None:
    """Heatmap de distribución de temas por grupo de discurso.

    Muestra la prevalencia de cada tema (filas) en cada fuente
    (columnas) con escala de colores que facilita la comparación.

    Parameters
    ----------
    topic_data : dict
        Diccionario {source_name: np.array} con distribuciones de temas.
    output_path : Path
        Ruta para guardar la figura (sin extensión).
    """
    setup_plot_style()

    # Construir DataFrame para heatmap
    sources = list(topic_data.keys())
    max_topics = max(len(v) for v in topic_data.values())
    data = np.zeros((max_topics, len(sources)))
    for j, src in enumerate(sources):
        dist = np.asarray(topic_data[src])
        data[:len(dist), j] = dist

    labels = [SOURCE_LABELS.get(s, s) for s in sources]
    topic_labels = [f"Topic {i}" for i in range(max_topics)]

    df_heat = pd.DataFrame(data, index=topic_labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, max(5, max_topics * 0.5)))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Topic Prevalence"},
    )
    ax.set_title("Topic Distribution by Discourse Group")
    ax.set_ylabel("Topics")
    ax.set_xlabel("Discourse Group")

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# EVOLUCIÓN TEMPORAL DE TEMAS
# ─────────────────────────────────────────────────────────────────────

def plot_topic_evolution(
    temporal_data: pd.DataFrame,
    output_path: Path,
    events: dict | None = None,
) -> None:
    """Gráfica de líneas de evolución temporal con anotaciones de eventos.

    Muestra la evolución de una métrica (e.g., sentimiento o prevalencia
    de un tema) a lo largo del tiempo para cada fuente de discurso.

    Parameters
    ----------
    temporal_data : pd.DataFrame
        DataFrame con columnas: year, source, mean_sentiment (o similar).
    output_path : Path
        Ruta para guardar la figura.
    events : dict, optional
        Diccionario {year: event_label} para anotar en la gráfica.
        Si None, usa KEY_EVENTS.
    """
    setup_plot_style()

    flat_events = _prepare_events(events)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = SOURCE_COLORS

    for source in sorted(temporal_data["source"].unique()):
        mask = temporal_data["source"] == source
        subset = temporal_data[mask].sort_values("year")
        color = colors.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        ax.plot(
            subset["year"],
            subset["mean_sentiment"],
            marker="o",
            color=color,
            label=label,
            linewidth=2,
            markersize=5,
        )

        # Confidence band (±1 std)
        if "std_sentiment" in subset.columns:
            ax.fill_between(
                subset["year"],
                subset["mean_sentiment"] - subset["std_sentiment"],
                subset["mean_sentiment"] + subset["std_sentiment"],
                color=color,
                alpha=0.1,
            )

    # Event annotations
    y_min, y_max = ax.get_ylim()
    for year_val, event_label in flat_events.items():
        if (temporal_data["year"].min() <= year_val
                <= temporal_data["year"].max()):
            ax.axvline(
                x=year_val, color="#B0BEC5", linestyle="--",
                linewidth=0.8, alpha=0.7,
            )
            ax.text(
                year_val, y_max * 0.95, event_label,
                rotation=45, fontsize=7, ha="left", va="top",
                color="#1B4F72",
            )

    ax.set_title("Temporal Evolution of Sentiment by Discourse Group")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Sentiment Score")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_xticks(sorted(temporal_data["year"].unique()))

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# VIOLIN PLOT DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def plot_sentiment_violin(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Violin plots de sentimiento por grupo de discurso.

    Muestra la distribución completa del sentimiento para cada fuente,
    con caja interior para mediana e IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source' y 'sentiment_score'.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    palette = {
        src: SOURCE_COLORS[src]
        for src in df["source"].unique()
        if src in SOURCE_COLORS
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    order = [s for s in ["academic", "institutional", "public"]
             if s in df["source"].unique()]

    sns.violinplot(
        data=df,
        x="source",
        y="sentiment_score",
        order=order,
        palette=palette,
        inner="box",
        linewidth=1,
        ax=ax,
    )

    # Renombrar etiquetas del eje x
    ax.set_xticklabels([SOURCE_LABELS.get(s, s) for s in order])
    ax.set_title("Sentiment Distribution by Discourse Group")
    ax.set_xlabel("Discourse Group")
    ax.set_ylabel("Sentiment Score")
    ax.axhline(y=0, color="#B0BEC5", linestyle="--", linewidth=0.8)

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# RIDGE PLOT DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def plot_sentiment_ridge(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Ridge plot (joy plot) de sentimiento por grupo de discurso.

    Alternativa visual al violin plot que apila distribuciones de
    densidad superpuestas para facilitar la comparación.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'source' y 'sentiment_score'.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    order = [s for s in ["academic", "institutional", "public"]
             if s in df["source"].unique()]

    df_plot = df[df["source"].isin(order)].copy()
    df_plot["source_label"] = df_plot["source"].map(SOURCE_LABELS)

    label_order = [SOURCE_LABELS[s] for s in order]

    g = sns.FacetGrid(
        df_plot,
        row="source_label",
        row_order=label_order,
        hue="source",
        hue_order=order,
        palette=SOURCE_COLORS,
        height=2,
        aspect=5,
    )

    g.map(
        sns.kdeplot,
        "sentiment_score",
        fill=True,
        alpha=0.6,
        linewidth=1.5,
    )

    g.set_titles("{row_name}")
    g.set_xlabels("Sentiment Score")
    g.fig.suptitle(
        "Sentiment Distribution by Discourse Group (Ridge Plot)",
        y=1.02,
        fontsize=13,
    )

    _save_figure(g.fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# COMPARACIÓN DE REDES (3 PANELES)
# ─────────────────────────────────────────────────────────────────────

def plot_network_comparison(
    networks: dict,
    output_path: Path,
) -> None:
    """Visualización de 3 paneles comparando redes por grupo de discurso.

    Cada panel muestra la red semántica de un grupo usando un layout
    spring con nodos proporcionales al grado.

    Parameters
    ----------
    networks : dict
        Diccionario {source_name: networkx.Graph}.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    try:
        import networkx as nx
    except ImportError:
        print("  ⚠ networkx not installed — skipping network plot")
        return

    sources = [s for s in ["academic", "institutional", "public"]
               if s in networks]
    n_panels = len(sources)

    if n_panels == 0:
        print("  ⚠ No networks to plot")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        G = networks[source]
        color = SOURCE_COLORS.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        if len(G.nodes()) == 0:
            ax.set_title(f"{label}\n(empty network)")
            ax.axis("off")
            continue

        # Layout
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5)

        # Tamaño de nodos proporcional al grado
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [
            300 * (degrees[n] / max_deg) + 50 for n in G.nodes()
        ]

        # Ancho de aristas proporcional al peso
        edge_weights = [
            G[u][v].get("weight", 1) for u, v in G.edges()
        ]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [2 * (w / max_w) + 0.3 for w in edge_weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color="#B0BEC5",
            alpha=0.5,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=7,
            font_family="serif",
        )

        ax.set_title(f"{label}\n(N={G.number_of_nodes()}, "
                     f"E={G.number_of_edges()})")
        ax.axis("off")

    fig.suptitle(
        "Semantic Network Comparison",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# HEATMAP DE DIVERGENCIA
# ─────────────────────────────────────────────────────────────────────

def plot_divergence_heatmap(
    divergence_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap triangular de divergencia entre grupos de discurso.

    Muestra la matriz de Jensen-Shannon Divergence (o Jaccard, etc.)
    como un triángulo inferior anotado.

    Parameters
    ----------
    divergence_matrix : pd.DataFrame
        Matriz cuadrada simétrica con valores de divergencia.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    # Crear máscara triangular superior
    mask = np.triu(np.ones_like(divergence_matrix, dtype=bool), k=1)

    # Renombrar índices para el plot
    labels = [
        SOURCE_LABELS.get(s, s) for s in divergence_matrix.index
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        divergence_matrix.values,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap="RdYlBu_r",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        ax=ax,
        cbar_kws={"label": "Jensen-Shannon Divergence"},
        vmin=0,
        square=True,
    )

    ax.set_title("Topic Distribution Divergence Between Groups")

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# NUBES DE PALABRAS
# ─────────────────────────────────────────────────────────────────────

def plot_wordclouds(
    keyword_data: dict,
    output_path: Path,
) -> None:
    """Tres nubes de palabras lado a lado, una por grupo de discurso.

    Parameters
    ----------
    keyword_data : dict
        Diccionario {source_name: dict {word: frequency}}.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  ⚠ wordcloud not installed — skipping word cloud plot")
        return

    sources = [s for s in ["academic", "institutional", "public"]
               if s in keyword_data]
    n_panels = len(sources)

    if n_panels == 0:
        print("  ⚠ No keyword data to plot")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        color = SOURCE_COLORS.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        # Función de color personalizada que usa el color del grupo
        def color_func(word, font_size, position, orientation,
                       random_state=None, **kwargs):
            # Oscurecer/aclarar según tamaño
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            factor = 0.4 + 0.6 * (font_size / 100)
            return (
                int(r * factor),
                int(g * factor),
                int(b * factor),
            )

        wc = WordCloud(
            width=800,
            height=600,
            background_color="white",
            max_words=80,
            prefer_horizontal=0.7,
            color_func=color_func,
            random_state=RANDOM_SEED,
        )

        frequencies = keyword_data[source]
        if isinstance(frequencies, list):
            # Convertir lista de palabras a frecuencias
            from collections import Counter
            frequencies = dict(Counter(frequencies))

        wc.generate_from_frequencies(frequencies)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Keyword Word Clouds by Discourse Group", fontsize=14)
    plt.tight_layout()

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# COMPARACIÓN BILINGÜE
# ─────────────────────────────────────────────────────────────────────

def plot_bilingual_comparison(
    bilingual_data: dict,
    output_path: Path,
) -> None:
    """Gráfica de barras agrupadas para comparación bilingüe EN vs ES.

    Muestra una métrica (e.g., Jaccard overlap o sentimiento) para
    cada fuente de discurso, agrupada por idioma.

    Parameters
    ----------
    bilingual_data : dict
        Diccionario con datos bilingües. Esperado:
        - 'sentiment_by_language': dict con resultados por fuente
        - 'keywords_by_language': dict con jaccard_per_source
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Jaccard de keywords ──
    ax1 = axes[0]
    kw_data = bilingual_data.get("keywords_by_language", {})
    jaccard = kw_data.get("jaccard_per_source", {})

    if jaccard:
        sources_present = [s for s in ["academic", "institutional", "public"]
                           if s in jaccard]
        values = [jaccard[s] for s in sources_present]
        colors = [SOURCE_COLORS[s] for s in sources_present]
        labels = [SOURCE_LABELS[s] for s in sources_present]

        bars = ax1.bar(labels, values, color=colors, edgecolor="white",
                       linewidth=0.8, alpha=0.85)

        # Anotar valores
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
            )

        ax1.set_title("Keyword Overlap (EN vs ES)")
        ax1.set_ylabel("Jaccard Similarity")
        ax1.set_ylim(0, max(values) * 1.2 if values else 1)
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title("Keyword Overlap (EN vs ES)")

    # ── Panel 2: Sentimiento por idioma y fuente ──
    ax2 = axes[1]
    sent_data = bilingual_data.get("sentiment_by_language", {})

    if sent_data:
        # Necesitamos datos originales para este panel
        # Mostrar U-statistic y significancia
        rows = []
        for source, res in sent_data.items():
            rows.append({
                "Source": SOURCE_LABELS.get(source, source),
                "U_statistic": res.get("U_statistic", 0),
                "significance": res.get("significance", "ns"),
                "effect_r": res.get("effect_size_r", 0),
            })
        sent_df = pd.DataFrame(rows)

        x = range(len(sent_df))
        colors = [
            SOURCE_COLORS.get(s, "#999")
            for s in sent_data.keys()
        ]
        bars = ax2.bar(
            sent_df["Source"], sent_df["effect_r"],
            color=colors, edgecolor="white", linewidth=0.8, alpha=0.85,
        )

        # Anotar significancia
        for bar, (_, row) in zip(bars, sent_df.iterrows()):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                row["significance"],
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax2.set_title("Sentiment Difference (EN vs ES)")
        ax2.set_ylabel("Effect Size (r)")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Sentiment Difference (EN vs ES)")

    fig.suptitle("Bilingual Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# TIMELINE DE CONVERGENCIA
# ─────────────────────────────────────────────────────────────────────

def plot_convergence_timeline(
    temporal_data: pd.DataFrame,
    events: dict | None = None,
    output_path: Path = None,
) -> None:
    """Timeline anotada de convergencia/divergencia entre grupos.

    Muestra cómo evoluciona la distancia (o similitud) entre los
    tres grupos a lo largo del tiempo, con anotaciones de eventos.

    Parameters
    ----------
    temporal_data : pd.DataFrame
        DataFrame con columnas: year, source, mean_sentiment.
    events : dict, optional
        Diccionario {year: event_label}. Si None, usa KEY_EVENTS.
    output_path : Path
        Ruta para guardar la figura.
    """
    setup_plot_style()

    flat_events = _prepare_events(events)

    # Inter-group standard deviation per year
    # (measure of divergence/convergence)
    pivot = temporal_data.pivot_table(
        index="year", columns="source", values="mean_sentiment",
    )

    yearly_spread = pivot.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )

    # -- Upper panel: sentiment lines --
    for source in sorted(temporal_data["source"].unique()):
        mask = temporal_data["source"] == source
        subset = temporal_data[mask].sort_values("year")
        color = SOURCE_COLORS.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        ax1.plot(
            subset["year"], subset["mean_sentiment"],
            marker="o", color=color, label=label,
            linewidth=2, markersize=5,
        )

    ax1.set_ylabel("Mean Sentiment")
    ax1.set_title("Discourse Convergence Timeline")
    ax1.legend(loc="best", frameon=True)

    # -- Lower panel: inter-group spread --
    ax2.fill_between(
        yearly_spread.index, 0, yearly_spread.values,
        color="#B0BEC5", alpha=0.3,
    )
    ax2.plot(
        yearly_spread.index, yearly_spread.values,
        color="#1B4F72", linewidth=2, marker="s", markersize=4,
    )
    ax2.set_ylabel("Inter-group\nStd. Dev.")
    ax2.set_xlabel("Year")

    # -- Event annotations on both panels --
    for year_val, event_label in flat_events.items():
        for ax in [ax1, ax2]:
            if (temporal_data["year"].min() <= year_val
                    <= temporal_data["year"].max()):
                ax.axvline(
                    x=year_val, color="#B0BEC5", linestyle="--",
                    linewidth=0.7, alpha=0.6,
                )

        # Labels only on upper panel
        if (temporal_data["year"].min() <= year_val
                <= temporal_data["year"].max()):
            y_max = ax1.get_ylim()[1]
            ax1.text(
                year_val, y_max * 0.95, event_label,
                rotation=45, fontsize=7, ha="left", va="top",
                color="#1B4F72",
            )

    ax2.set_xticks(sorted(temporal_data["year"].unique()))

    plt.tight_layout()

    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PLOTS MODULE — Demo with synthetic data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    setup_plot_style()

    # ── Datos sintéticos ──
    n = 300
    df_test = pd.DataFrame({
        "source": (["academic"] * 100 + ["institutional"] * 100
                   + ["public"] * 100),
        "sentiment_score": np.concatenate([
            np.random.normal(0.15, 0.25, 100),
            np.random.normal(0.35, 0.20, 100),
            np.random.normal(-0.05, 0.35, 100),
        ]),
    })

    # Violin plot demo
    out = FIGURES_DIR / "demo_sentiment_violin"
    plot_sentiment_violin(df_test, out)
    print(f"  Demo violin plot saved.")

    # Ridge plot demo
    out = FIGURES_DIR / "demo_sentiment_ridge"
    plot_sentiment_ridge(df_test, out)
    print(f"  Demo ridge plot saved.")

    # Topic distribution heatmap demo
    topic_data = {
        "academic": np.random.dirichlet(np.ones(6)),
        "institutional": np.random.dirichlet(np.ones(6)),
        "public": np.random.dirichlet(np.ones(6)),
    }
    out = FIGURES_DIR / "demo_topic_heatmap"
    plot_topic_distribution_heatmap(topic_data, out)
    print(f"  Demo heatmap saved.")

    # Divergence heatmap demo
    div_matrix = pd.DataFrame(
        [[0.0, 0.15, 0.22], [0.15, 0.0, 0.18], [0.22, 0.18, 0.0]],
        index=["academic", "institutional", "public"],
        columns=["academic", "institutional", "public"],
    )
    out = FIGURES_DIR / "demo_divergence_heatmap"
    plot_divergence_heatmap(div_matrix, out)
    print(f"  Demo divergence heatmap saved.")

    # Temporal evolution demo
    years = list(range(2015, 2026))
    rows = []
    for yr in years:
        for src in ["academic", "institutional", "public"]:
            rows.append({
                "year": yr,
                "source": src,
                "mean_sentiment": np.random.normal(
                    {"academic": 0.1, "institutional": 0.3,
                     "public": -0.05}[src]
                    + (yr - 2015) * 0.02,
                    0.05,
                ),
                "std_sentiment": np.random.uniform(0.1, 0.3),
            })
    temporal_df = pd.DataFrame(rows)

    out = FIGURES_DIR / "demo_topic_evolution"
    plot_topic_evolution(temporal_df, out)
    print(f"  Demo temporal evolution saved.")

    out = FIGURES_DIR / "demo_convergence_timeline"
    plot_convergence_timeline(temporal_df, output_path=out)
    print(f"  Demo convergence timeline saved.")

    print("\n✓ All demo plots generated successfully.")
