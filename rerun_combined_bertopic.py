"""
rerun_combined_bertopic.py — Re-entrena solo el modelo combinado
================================================================

Re-corre BERTopic sobre el corpus completo actualizado (15,232 docs)
sin re-entrenar el modelo académico (que no cambió).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from constants import normalize_group_series

import pandas as pd


def main():
    t_start = time.time()

    print()
    print("=" * 60)
    print("  RE-RUN: BERTopic Combined Model (Updated Corpus)")
    print("=" * 60)

    # ── Cargar corpus ────────────────────────────────────────────
    corpus_path = PROCESSED_DIR / "corpus_unified.csv"
    if not corpus_path.exists():
        print(f"  ✗ Corpus not found: {corpus_path}")
        sys.exit(1)

    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    if "source" in df.columns:
        df["source"] = normalize_group_series(df["source"])

    print(f"  ✓ Corpus loaded: {len(df):,} documents")
    print(f"    Sources: {df['source'].value_counts().to_dict()}")

    text_col = "text_clean"
    if text_col not in df.columns:
        text_col = "text"
    df = df[df[text_col].notna() & (df[text_col].str.len() > 30)].reset_index(drop=True)
    print(f"    After filtering: {len(df):,} documents")

    # ── Import topic modeling ────────────────────────────────────
    from src.analysis.topic_modeling import (
        create_bertopic_model,
        save_topic_results,
        compare_topic_distributions,
    )

    results = {}

    # ── Run institutional model (new group, was <100 before) ─────
    inst_df = df[df["source"] == "institutional"]
    if len(inst_df) >= 100:
        print(f"\n  Running BERTopic on institutional ({len(inst_df)} docs)...")
        inst_docs = inst_df[text_col].tolist()
        model, topics, probs = create_bertopic_model(
            inst_docs, source_label="institutional"
        )
        output_dir = RESULTS_DIR / "topics" / "institutional"
        save_topic_results(model, topics, inst_docs, output_dir, label="institutional")
        results["institutional"] = {
            "model": model, "topics": topics, "probs": probs, "docs": inst_docs,
        }
    else:
        print(f"  Institutional has only {len(inst_df)} docs, skipping standalone model")

    # ── Run combined model (full corpus) ────────────────────────
    print(f"\n  Running BERTopic COMBINED on full corpus ({len(df)} docs)...")
    all_docs = df[text_col].tolist()
    model, topics, probs = create_bertopic_model(
        all_docs, source_label="combined"
    )

    output_dir = RESULTS_DIR / "topics" / "combined"
    save_topic_results(model, topics, all_docs, output_dir, label="combined")
    results["combined"] = {
        "model": model, "topics": topics, "probs": probs, "docs": all_docs,
    }

    # ── Also load academic results for comparison ────────────────
    academic_topic_info_path = RESULTS_DIR / "topics" / "academic" / "topic_info_academic.csv"
    if academic_topic_info_path.exists():
        print("\n  Loading existing academic model for comparison...")
        from bertopic import BERTopic as BT
        academic_model_path = RESULTS_DIR / "topics" / "academic" / "bertopic_model_academic"
        try:
            academic_model = BT.load(str(academic_model_path))
            academic_doc_topics = pd.read_csv(
                RESULTS_DIR / "topics" / "academic" / "doc_topics_academic.csv",
                encoding="utf-8-sig",
            )
            results["academic"] = {
                "model": academic_model,
                "topics": academic_doc_topics["topic"].tolist(),
                "probs": None,
                "docs": academic_doc_topics["document"].tolist(),
            }
            print("  Academic model loaded from cache")
        except Exception as e:
            print(f"  Could not load academic model: {e}")

    # ── Compare distributions ────────────────────────────────────
    if len(results) > 1:
        print("\n  Comparing topic distributions...")
        comparison = compare_topic_distributions(results)
        comparison.to_csv(
            RESULTS_DIR / "topic_distribution_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"  Comparison saved")
        print(f"\n  Combined model summary:")
        combined_topics = comparison[comparison["group"] == "combined"]
        print(f"    Total topics (excl. outliers): {len(combined_topics)}")
        if len(combined_topics) > 0:
            print(f"    Total docs: {combined_topics['total_docs'].iloc[0]}")
        print(f"\n  Top 10 combined topics:")
        print(combined_topics.head(10).to_string(index=False))

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 60)
    print(f"  RE-RUN COMPLETE in {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
