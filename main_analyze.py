"""
main_analyze.py — Orquestador del pipeline de análisis completo
================================================================

Ejecuta todas las fases de análisis NLP, comparación y visualización
sobre el corpus unificado preprocesado.

Uso:
    python main_analyze.py --phase all
    python main_analyze.py --phase topics
    python main_analyze.py --phase sentiment
    python main_analyze.py --phase networks
    python main_analyze.py --phase comparison
    python main_analyze.py --phase visualize
"""

import sys
import argparse
import time
from pathlib import Path

# ── Configuración de rutas ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from constants import normalize_group_series

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Carga el corpus unificado preprocesado."""
    corpus_path = PROCESSED_DIR / "corpus_unified.csv"
    if not corpus_path.exists():
        print("✗ No se encontró el corpus unificado en:")
        print(f"  {corpus_path}")
        print("  Ejecuta primero: python main_preprocess.py")
        sys.exit(1)

    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    # Normalize legacy 'political' → 'institutional'
    if "source" in df.columns:
        df["source"] = normalize_group_series(df["source"])
    print(f"✓ Corpus cargado: {len(df):,} documentos")
    print(f"  Fuentes: {df['source'].value_counts().to_dict()}")
    print(f"  Idiomas: {df['lang'].value_counts().to_dict()}")
    return df


def print_phase_header(phase_name: str):
    """Imprime encabezado de fase."""
    print()
    print("=" * 70)
    print(f"  FASE: {phase_name}")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────
# FASE 1: TOPIC MODELING
# ─────────────────────────────────────────────────────────────────────

def run_topics(df: pd.DataFrame) -> dict:
    """Ejecuta BERTopic sobre el corpus."""
    print_phase_header("TOPIC MODELING (BERTopic)")

    from src.analysis.topic_modeling import (
        run_topic_modeling_per_group,
        compare_topic_distributions,
    )

    print("⏳ Ejecutando BERTopic por grupo y combinado...")
    print("   (Esto puede tomar 15-60 min en CPU dependiendo del corpus)")
    print()

    topic_results = run_topic_modeling_per_group(df, text_col="text_clean")

    print()
    print("⏳ Comparando distribuciones de tópicos entre grupos...")
    comparison = compare_topic_distributions(topic_results)
    comparison.to_csv(RESULTS_DIR / "topic_distribution_comparison.csv", index=False)
    print(f"✓ Comparación guardada en: {RESULTS_DIR / 'topic_distribution_comparison.csv'}")

    # Guardar resumen
    print()
    print("─" * 50)
    print("RESUMEN DE TÓPICOS POR GRUPO:")
    print("─" * 50)
    for group, data in topic_results.items():
        if "topic_info" in data:
            n_topics = len(data["topic_info"]) - 1  # excluir -1 (outliers)
            print(f"  {group}: {n_topics} tópicos encontrados")

    return topic_results


# ─────────────────────────────────────────────────────────────────────
# FASE 2: ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────────────

def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta análisis de sentimiento multilingüe."""
    print_phase_header("ANÁLISIS DE SENTIMIENTO")

    from src.analysis.sentiment import (
        run_sentiment_analysis,
        compute_sentiment_stats,
    )

    print("⏳ Analizando sentimiento (modelos multilingües en CPU)...")
    print("   Modelos: nlptown/bert-multilingual (EN+ES)")
    print("            cardiffnlp/twitter-roberta (EN)")
    print("            pysentimiento/robertuito (ES)")
    print()

    df_sentiment = run_sentiment_analysis(df, text_col="text_clean")

    # Guardar resultados
    output_path = PROCESSED_DIR / "corpus_with_sentiment.csv"
    df_sentiment.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✓ Corpus con sentimiento guardado: {output_path}")

    # Estadísticas resumidas
    stats = compute_sentiment_stats(df_sentiment)
    stats.to_csv(RESULTS_DIR / "tables" / "sentiment_stats.csv", index=False)
    print(f"✓ Estadísticas de sentimiento guardadas")

    print()
    print("─" * 50)
    print("RESUMEN DE SENTIMIENTO:")
    print("─" * 50)
    print(stats.to_string(index=False))

    return df_sentiment


# ─────────────────────────────────────────────────────────────────────
# FASE 3: REDES SEMÁNTICAS
# ─────────────────────────────────────────────────────────────────────

def run_networks(df: pd.DataFrame) -> dict:
    """Construye y compara redes semánticas por grupo."""
    print_phase_header("REDES SEMÁNTICAS")

    from src.analysis.semantic_network import (
        build_networks_per_group,
        compare_networks,
    )

    print("⏳ Construyendo redes de co-ocurrencia por grupo...")
    network_results = build_networks_per_group(df, text_col="text_clean")

    print()
    print("⏳ Comparando métricas de redes entre grupos...")
    comparison = compare_networks(network_results)
    comparison.to_csv(RESULTS_DIR / "tables" / "network_comparison.csv", index=False)
    print(f"✓ Comparación de redes guardada")

    print()
    print("─" * 50)
    print("MÉTRICAS DE REDES POR GRUPO:")
    print("─" * 50)
    print(comparison.to_string(index=False))

    return network_results


# ─────────────────────────────────────────────────────────────────────
# FASE 4: ANÁLISIS COMPARATIVO
# ─────────────────────────────────────────────────────────────────────

def run_comparison(df: pd.DataFrame, topic_results: dict) -> dict:
    """Ejecuta análisis comparativo cross-group, temporal y bilingüe."""
    print_phase_header("ANÁLISIS COMPARATIVO")

    results = {}

    # 4a. Cross-group
    print("⏳ [4a] Comparación entre grupos...")
    try:
        from src.comparison.cross_group import run_cross_group_analysis
        results["cross_group"] = run_cross_group_analysis(df, topic_results)
        print("✓ Comparación cross-group completada")
    except Exception as e:
        print(f"✗ Error en cross-group: {e}")

    # 4b. Temporal
    print()
    print("⏳ [4b] Análisis de evolución temporal...")
    try:
        from src.comparison.temporal import run_temporal_analysis
        results["temporal"] = run_temporal_analysis(df)
        print("✓ Análisis temporal completado")
    except Exception as e:
        print(f"✗ Error en temporal: {e}")

    # 4c. Bilingüe
    print()
    print("⏳ [4c] Comparación bilingüe (EN vs ES)...")
    try:
        from src.comparison.bilingual import run_bilingual_analysis
        results["bilingual"] = run_bilingual_analysis(df, topic_results)
        print("✓ Análisis bilingüe completado")
    except Exception as e:
        print(f"✗ Error en bilingüe: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────
# FASE 5: VISUALIZACIONES Y TABLAS
# ─────────────────────────────────────────────────────────────────────

def run_visualizations(
    df: pd.DataFrame,
    topic_results: dict,
    network_results: dict,
    comparison_results: dict,
):
    """Genera todas las figuras y tablas para el paper."""
    print_phase_header("VISUALIZACIONES Y TABLAS")

    fig_dir = RESULTS_DIR / "figures"
    tab_dir = RESULTS_DIR / "tables"

    # ── Figuras ──
    print("⏳ Generando figuras para el paper...")
    try:
        from src.visualization.plots import (
            setup_plot_style,
            plot_topic_distribution_heatmap,
            plot_topic_evolution,
            plot_sentiment_violin,
            plot_divergence_heatmap,
            plot_wordclouds,
            plot_convergence_timeline,
            plot_network_comparison,
        )

        setup_plot_style()

        # Fig 2: Distribución de tópicos por grupo
        if topic_results:
            print("  📊 Fig 2: Distribución de tópicos...")
            try:
                plot_topic_distribution_heatmap(topic_results, fig_dir / "fig2_topic_heatmap")
                print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 3: Evolución temporal de tópicos
        if comparison_results.get("temporal"):
            print("  📊 Fig 3: Evolución temporal...")
            try:
                temporal_data = comparison_results["temporal"]
                if "yearly_sentiment" in temporal_data:
                    plot_topic_evolution(
                        temporal_data.get("yearly_topics", pd.DataFrame()),
                        fig_dir / "fig3_topic_evolution"
                    )
                    print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 4: Violin plot de sentimiento
        if "sentiment_score" in df.columns:
            print("  📊 Fig 4: Sentimiento por grupo...")
            try:
                plot_sentiment_violin(df, fig_dir / "fig4_sentiment_violin")
                print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 5: Redes semánticas
        if network_results:
            print("  📊 Fig 5: Redes semánticas comparativas...")
            try:
                plot_network_comparison(network_results, fig_dir / "fig5_networks")
                print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 6: Divergence heatmap
        if comparison_results.get("cross_group"):
            print("  📊 Fig 6: Divergencia entre grupos...")
            try:
                cg = comparison_results["cross_group"]
                if "topic_divergence" in cg:
                    plot_divergence_heatmap(
                        cg["topic_divergence"],
                        fig_dir / "fig6_divergence"
                    )
                    print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 7: Word clouds
        if topic_results:
            print("  📊 Fig 7: Word clouds...")
            try:
                plot_wordclouds(topic_results, fig_dir / "fig7_wordclouds")
                print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Fig 8: Timeline de convergencia
        if comparison_results.get("temporal"):
            print("  📊 Fig 8: Timeline convergencia/divergencia...")
            try:
                plot_convergence_timeline(
                    comparison_results["temporal"].get("yearly_sentiment", pd.DataFrame()),
                    KEY_EVENTS,
                    fig_dir / "fig8_timeline"
                )
                print("     ✓ Guardada")
            except Exception as e:
                print(f"     ✗ Error: {e}")

    except ImportError as e:
        print(f"✗ Error importando módulos de visualización: {e}")

    # ── Tablas ──
    print()
    print("⏳ Generando tablas para el paper...")
    try:
        from src.visualization.tables import (
            create_corpus_summary_table,
            create_sentiment_summary_table,
            create_network_metrics_table,
            save_table_csv,
            save_table_latex,
        )

        # Tabla 1: Resumen del corpus
        print("  📋 Tabla 1: Resumen descriptivo del corpus...")
        try:
            t1 = create_corpus_summary_table(df)
            save_table_csv(t1, tab_dir / "table1_corpus_summary.csv")
            save_table_latex(t1, tab_dir / "table1_corpus_summary.tex",
                          caption="Descriptive statistics of the tri-partite corpus")
            print("     ✓ Guardada (CSV + LaTeX)")
        except Exception as e:
            print(f"     ✗ Error: {e}")

        # Tabla 3: Sentimiento
        if "sentiment_score" in df.columns:
            print("  📋 Tabla 3: Resumen de sentimiento...")
            try:
                t3 = create_sentiment_summary_table(df)
                save_table_csv(t3, tab_dir / "table3_sentiment_summary.csv")
                save_table_latex(t3, tab_dir / "table3_sentiment_summary.tex",
                              caption="Sentiment analysis results by source and language")
                print("     ✓ Guardada (CSV + LaTeX)")
            except Exception as e:
                print(f"     ✗ Error: {e}")

        # Tabla 5: Métricas de redes
        if network_results:
            print("  📋 Tabla 5: Métricas de redes...")
            try:
                t5 = create_network_metrics_table(network_results)
                save_table_csv(t5, tab_dir / "table5_network_metrics.csv")
                save_table_latex(t5, tab_dir / "table5_network_metrics.tex",
                              caption="Semantic network metrics comparison")
                print("     ✓ Guardada (CSV + LaTeX)")
            except Exception as e:
                print(f"     ✗ Error: {e}")

    except ImportError as e:
        print(f"✗ Error importando módulos de tablas: {e}")

    print()
    print(f"✓ Figuras guardadas en: {fig_dir}")
    print(f"✓ Tablas guardadas en:  {tab_dir}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VOZ_SUS — Pipeline de análisis NLP completo"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "topics", "sentiment", "networks", "comparison", "visualize"],
        default="all",
        help="Fase a ejecutar (default: all)"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TRES VOCES DE LA SUSTENTABILIDAD — Análisis NLP      ║")
    print("║   Pipeline de análisis completo                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    t_start = time.time()

    # Cargar corpus
    df = load_corpus()

    # Variables para resultados intermedios
    topic_results = {}
    network_results = {}
    comparison_results = {}

    # ── Ejecutar fases ──
    if args.phase in ("all", "topics"):
        try:
            topic_results = run_topics(df)
        except Exception as e:
            print(f"\n✗ ERROR en Topic Modeling: {e}")
            import traceback; traceback.print_exc()

    if args.phase in ("all", "sentiment"):
        try:
            df = run_sentiment(df)
        except Exception as e:
            print(f"\n✗ ERROR en Sentimiento: {e}")
            import traceback; traceback.print_exc()

    if args.phase in ("all", "networks"):
        try:
            network_results = run_networks(df)
        except Exception as e:
            print(f"\n✗ ERROR en Redes Semánticas: {e}")
            import traceback; traceback.print_exc()

    if args.phase in ("all", "comparison"):
        try:
            comparison_results = run_comparison(df, topic_results)
        except Exception as e:
            print(f"\n✗ ERROR en Comparación: {e}")
            import traceback; traceback.print_exc()

    if args.phase in ("all", "visualize"):
        try:
            run_visualizations(df, topic_results, network_results, comparison_results)
        except Exception as e:
            print(f"\n✗ ERROR en Visualizaciones: {e}")
            import traceback; traceback.print_exc()

    # ── Resumen final ──
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("═" * 70)
    print(f"  PIPELINE COMPLETADO en {minutes}m {seconds}s")
    print("═" * 70)
    print()
    print("  Resultados guardados en:")
    print(f"    📁 Datos procesados: {PROCESSED_DIR}")
    print(f"    📁 Figuras:          {RESULTS_DIR / 'figures'}")
    print(f"    📁 Tablas:           {RESULTS_DIR / 'tables'}")
    print(f"    📁 Modelos:          {RESULTS_DIR / 'models'}")
    print()


if __name__ == "__main__":
    main()
