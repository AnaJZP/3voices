"""
fix_figures.py — Load saved results and generate all missing figures
====================================================================
The main pipeline ran topics, sentiment, and networks successfully but
some figures failed because the data structures weren't passed correctly
to the plotting functions.  This script loads the persisted results
from disk and regenerates every figure.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from constants import normalize_group_series

import numpy as np
import pandas as pd
import networkx as nx


# ─────────────────────────────────────────────────────────────────────
# 1. LOAD ALL SAVED DATA
# ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  LOADING SAVED RESULTS")
print("=" * 70)

# ── Corpus with sentiment ──
df = pd.read_csv(PROCESSED_DIR / "corpus_with_sentiment.csv", encoding="utf-8-sig")
if "source" in df.columns:
    df["source"] = normalize_group_series(df["source"])
print(f"  Corpus: {len(df):,} docs | sources: {df['source'].value_counts().to_dict()}")

# ── Topic info from saved CSVs ──
topic_info = {}
topic_words = {}
for group in ["academic", "combined"]:
    info_path = RESULTS_DIR / "topics" / group / f"topic_info_{group}.csv"
    words_path = RESULTS_DIR / "topics" / group / f"topic_words_{group}.csv"
    if info_path.exists():
        topic_info[group] = pd.read_csv(info_path)
        print(f"  Topic info [{group}]: {len(topic_info[group])} topics")
    if words_path.exists():
        topic_words[group] = pd.read_csv(words_path)
        print(f"  Topic words [{group}]: {len(topic_words[group])} word-topic pairs")

# ── Topic distribution comparison ──
topic_comparison = pd.read_csv(RESULTS_DIR / "topics" / "topic_distribution_comparison.csv")
print(f"  Topic comparison: {len(topic_comparison)} rows")

# ── Networks — rebuild from saved edges ──
network_graphs = {}
for group in ["academic", "public"]:
    edges_path = RESULTS_DIR / "networks" / group / f"edge_list_{group}.csv"
    if edges_path.exists():
        edges_df = pd.read_csv(edges_path)
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row["source_word"], row["target_word"], weight=row["weight"])
        network_graphs[group] = G
        print(f"  Network [{group}]: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Temporal data ──
yearly_sent_path = RESULTS_DIR / "comparison" / "yearly_sentiment.csv"
if yearly_sent_path.exists():
    yearly_sentiment = pd.read_csv(yearly_sent_path)
    print(f"  Yearly sentiment: {len(yearly_sentiment)} rows")
else:
    yearly_sentiment = None

# ── Sentiment comparison ──
sent_desc_path = RESULTS_DIR / "comparison" / "sentiment_descriptives.csv"
if sent_desc_path.exists():
    sent_descriptives = pd.read_csv(sent_desc_path)
    print(f"  Sentiment descriptives loaded")


# ─────────────────────────────────────────────────────────────────────
# 2. SETUP PLOTTING
# ─────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.plots import (
    setup_plot_style,
    _save_figure,
    _prepare_events,
)

setup_plot_style()

fig_dir = RESULTS_DIR / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# FIG 2: TOPIC DISTRIBUTION HEATMAP
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 2: Topic Distribution Heatmap")
print("-" * 60)

# Build distribution from topic_comparison CSV
groups_with_topics = [g for g in ["academic", "combined"] if g in topic_info]

if groups_with_topics:
    # Get top 15 topics per group by prevalence
    top_topics = (
        topic_comparison[topic_comparison["topic_id"] >= 0]
        .sort_values("prevalence_pct", ascending=False)
        .groupby("group")
        .head(15)
    )

    # Create a pivot table for heatmap
    # Use short topic names (first few words)
    top_topics = top_topics.copy()
    top_topics["short_name"] = top_topics["topic_name"].apply(
        lambda x: "_".join(str(x).split("_")[1:4]) if "_" in str(x) else str(x)[:30]
    )

    # For academic vs combined heatmap
    for group in groups_with_topics:
        grp_data = top_topics[top_topics["group"] == group].head(15)
        if grp_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        data_matrix = grp_data[["short_name", "prevalence_pct"]].set_index("short_name")

        colors_list = [SOURCE_COLORS.get(group, "#2E86C1")]
        cmap = sns.light_palette(colors_list[0], as_cmap=True)

        sns.heatmap(
            data_matrix,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Prevalence (%)"},
        )
        label = SOURCE_LABELS.get(group, group)
        ax.set_title(f"Top 15 Topic Distribution — {label}", fontsize=14)
        ax.set_ylabel("Topic")
        ax.set_xlabel("")
        plt.tight_layout()
        _save_figure(fig, fig_dir / f"fig2_topic_heatmap_{group}")

    print("  OK Fig 2")


# ─────────────────────────────────────────────────────────────────────
# FIG 3: TEMPORAL SENTIMENT EVOLUTION
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 3: Temporal Sentiment Evolution")
print("-" * 60)

if yearly_sentiment is not None and not yearly_sentiment.empty:
    flat_events = _prepare_events(KEY_EVENTS)

    fig, ax = plt.subplots(figsize=(12, 6))

    for source in sorted(yearly_sentiment["source"].unique()):
        mask = yearly_sentiment["source"] == source
        subset = yearly_sentiment[mask].sort_values("year")
        color = SOURCE_COLORS.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        ax.plot(
            subset["year"], subset["mean_sentiment"],
            marker="o", color=color, label=label,
            linewidth=2, markersize=6,
        )
        if "std_sentiment" in subset.columns:
            ax.fill_between(
                subset["year"],
                subset["mean_sentiment"] - subset["std_sentiment"],
                subset["mean_sentiment"] + subset["std_sentiment"],
                color=color, alpha=0.1,
            )

    # Event annotations
    y_min, y_max = ax.get_ylim()
    for year_val, event_label in flat_events.items():
        if yearly_sentiment["year"].min() <= year_val <= yearly_sentiment["year"].max():
            ax.axvline(x=year_val, color="#B0BEC5", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(
                year_val + 0.1, y_max * 0.97, event_label,
                rotation=45, fontsize=7, ha="left", va="top", color="#1B4F72",
            )

    ax.set_title("Temporal Evolution of Sentiment by Discourse Group", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Sentiment Score")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_xticks(sorted(yearly_sentiment["year"].unique()))
    plt.tight_layout()
    _save_figure(fig, fig_dir / "fig3_topic_evolution")
    print("  OK Fig 3")
else:
    print("  SKIP — no temporal data")


# ─────────────────────────────────────────────────────────────────────
# FIG 4: SENTIMENT VIOLIN (re-gen with 3 voices)
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 4: Sentiment Violin Plot")
print("-" * 60)

if "sentiment_score" in df.columns:
    from src.visualization.plots import plot_sentiment_violin
    plot_sentiment_violin(df, fig_dir / "fig4_sentiment_violin")
    print("  OK Fig 4")


# ─────────────────────────────────────────────────────────────────────
# FIG 5: SEMANTIC NETWORK COMPARISON
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 5: Semantic Network Comparison")
print("-" * 60)

if network_graphs:
    sources = [s for s in ["academic", "institutional", "public"] if s in network_graphs]
    n_panels = len(sources)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        G = network_graphs[source]
        color = SOURCE_COLORS.get(source, "#999999")
        label = SOURCE_LABELS.get(source, source)

        # Use top 50 nodes by degree for readability
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:50]
        top_node_names = [n for n, d in top_nodes]
        G_sub = G.subgraph(top_node_names).copy()

        pos = nx.spring_layout(G_sub, seed=42, k=2.0)

        degrees = dict(G_sub.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [400 * (degrees[n] / max_deg) + 30 for n in G_sub.nodes()]

        edge_weights = [G_sub[u][v].get("weight", 1) for u, v in G_sub.edges()]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [2.5 * (w / max_w) + 0.2 for w in edge_weights]

        nx.draw_networkx_edges(G_sub, pos, ax=ax, width=edge_widths, edge_color="#B0BEC5", alpha=0.4)
        nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=node_sizes, node_color=color, alpha=0.85, edgecolors="white", linewidths=0.5)
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=7, font_family="serif")

        ax.set_title(f"{label}\n(N={G.number_of_nodes()}, E={G.number_of_edges()})", fontsize=13)
        ax.axis("off")

    fig.suptitle("Semantic Network Comparison (Top 50 Nodes)", fontsize=15, y=1.02)
    plt.tight_layout()
    _save_figure(fig, fig_dir / "fig5_networks")
    print("  OK Fig 5")
else:
    print("  SKIP — no network data")


# ─────────────────────────────────────────────────────────────────────
# FIG 7: WORD CLOUDS
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 7: Word Clouds")
print("-" * 60)

try:
    from wordcloud import WordCloud

    # Build word frequencies from topic_words CSVs and from the corpus itself
    wc_data = {}

    # For groups with topic words
    for group in ["academic"]:
        if group in topic_words:
            tw = topic_words[group]
            # Aggregate word scores across all topics
            word_freq = tw.groupby("word")["score"].sum().to_dict()
            wc_data[group] = word_freq

    # For public — use word frequencies from the corpus directly
    for source in ["public"]:
        subset = df[df["source"] == source]["text_clean"].dropna()
        if len(subset) > 0:
            from collections import Counter
            import re
            all_words = []
            stopwords = set([
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "not", "no", "it", "its", "this",
                "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
                "he", "she", "they", "them", "their", "what", "which", "who", "how",
                "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
                "por", "un", "para", "con", "una", "su", "al", "lo", "como",
                "http", "https", "www", "com", "don", "just", "like", "get",
                "really", "going", "one", "much", "also", "very", "more", "than",
            ])
            for text in subset:
                words = re.findall(r"[a-z]+", str(text).lower())
                all_words.extend([w for w in words if len(w) >= 3 and w not in stopwords])
            word_freq = dict(Counter(all_words).most_common(300))
            wc_data[source] = word_freq

    if wc_data:
        sources_wc = [s for s in ["academic", "institutional", "public"] if s in wc_data]
        n_panels = len(sources_wc)

        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        for ax, source in zip(axes, sources_wc):
            color = SOURCE_COLORS.get(source, "#999999")
            label = SOURCE_LABELS.get(source, source)

            def make_color_func(base_color):
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    factor = 0.4 + 0.6 * (font_size / 100)
                    return (int(r * factor), int(g * factor), int(b * factor))
                return color_func

            wc = WordCloud(
                width=900, height=600,
                background_color="white",
                max_words=100,
                prefer_horizontal=0.7,
                color_func=make_color_func(color),
                random_state=42,
            )
            wc.generate_from_frequencies(wc_data[source])
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.axis("off")

        fig.suptitle("Keyword Word Clouds by Discourse Group", fontsize=15, y=1.02)
        plt.tight_layout()
        _save_figure(fig, fig_dir / "fig7_wordclouds")
        print("  OK Fig 7")

except ImportError:
    print("  SKIP — wordcloud not installed")


# ─────────────────────────────────────────────────────────────────────
# FIG 8: CONVERGENCE TIMELINE (re-gen with better events)
# ─────────────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("  Fig 8: Convergence Timeline")
print("-" * 60)

if yearly_sentiment is not None and not yearly_sentiment.empty:
    from src.visualization.plots import plot_convergence_timeline
    plot_convergence_timeline(yearly_sentiment, events=KEY_EVENTS, output_path=fig_dir / "fig8_timeline")
    print("  OK Fig 8")


# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FIGURE GENERATION COMPLETE")
print("=" * 70)

import os as _os
for f in sorted(fig_dir.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:45s}  {size_kb:>8.1f} KB")

print()
