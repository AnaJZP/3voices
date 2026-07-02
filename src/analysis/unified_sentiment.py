"""
unified_sentiment.py — Sentimiento con modelo unico multilingue
================================================================

Re-scores the entire corpus using a single multilingual model to
eliminate calibration confounds between language-specific models.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
  - 3-class: negative, neutral, positive
  - Supports: English, Spanish + 6 more languages
  - Normalized to [0, 1] scale

For long documents (transcripts), the text is chunked into segments
of max_tokens and scores are averaged per document.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

UNIFIED_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
MAX_CHUNK_CHARS = 1800  # ~512 tokens, conservative
BATCH_SIZE = 16

# Map 3-class labels to [0, 1]
LABEL_MAP = {
    "negative": 0.0,
    "neutral": 0.5,
    "positive": 1.0,
}


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of approximately max_chars."""
    if not isinstance(text, str) or len(text) == 0:
        return [""]

    if len(text) <= max_chars:
        return [text]

    chunks = []
    words = text.split()
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1
        if current_len + word_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [""]


def _score_to_continuous(result: dict) -> float:
    """Convert 3-class result to continuous [0, 1] score.

    Uses weighted average of all class probabilities.
    """
    if isinstance(result, list):
        # Pipeline returns list of dicts with all scores
        total = 0.0
        for item in result:
            label = item.get("label", "neutral").lower()
            score = item.get("score", 0.0)
            total += LABEL_MAP.get(label, 0.5) * score
        return total
    else:
        # Single top-1 result
        label = result.get("label", "neutral").lower()
        score = result.get("score", 0.5)
        mapped = LABEL_MAP.get(label, 0.5)
        # Interpolate: if positive with 0.7 confidence, score = 0.5 + 0.5*0.7 = 0.85
        if mapped == 1.0:
            return 0.5 + 0.5 * score
        elif mapped == 0.0:
            return 0.5 - 0.5 * score
        else:
            return 0.5


def build_unified_scorer():
    """Build the unified sentiment pipeline."""
    print(f"\n  Loading model: {UNIFIED_MODEL}")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=UNIFIED_MODEL,
        device=-1,
        truncation=True,
        max_length=512,
        top_k=3,  # Get all 3 class probabilities
    )
    print("  Model loaded.")
    return pipe


def score_corpus(
    df: pd.DataFrame,
    text_col: str = "text",
    pipe=None,
) -> pd.Series:
    """Score an entire corpus with the unified model.

    For long documents, chunks text and averages scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain text_col.
    text_col : str
        Column with text to score.
    pipe : pipeline, optional
        Pre-loaded pipeline. If None, loads model.

    Returns
    -------
    pd.Series
        Unified sentiment scores [0, 1].
    """
    if pipe is None:
        pipe = build_unified_scorer()

    scores = []
    total = len(df)

    for idx, text in tqdm(enumerate(df[text_col]), total=total, desc="  Unified scoring"):
        if not isinstance(text, str) or len(text.strip()) == 0:
            scores.append(np.nan)
            continue

        chunks = _chunk_text(text)
        chunk_scores = []

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                result = pipe(chunk[:MAX_CHUNK_CHARS])
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # top_k returns list of lists
                        s = _score_to_continuous(result[0])
                    else:
                        s = _score_to_continuous(result)
                    chunk_scores.append(s)
            except Exception as e:
                if idx < 5:
                    print(f"  [WARN] Error scoring chunk {idx}: {e}")

        if chunk_scores:
            scores.append(round(np.mean(chunk_scores), 4))
        else:
            scores.append(np.nan)

    return pd.Series(scores, index=df.index, name="sentiment_unified")


def run_unified_scoring(
    corpus_path: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """Run unified scoring on the full corpus.

    Parameters
    ----------
    corpus_path : str, optional
        Path to corpus CSV. If None, uses default.
    output_path : str, optional
        Path for output. If None, uses default.

    Returns
    -------
    pd.DataFrame
        Corpus with sentiment_unified column added.
    """
    if corpus_path is None:
        # Try common locations
        candidates = [
            FINAL_DIR / "unified_corpus.csv",
            PROCESSED_DIR / "unified_corpus_clean.csv",
            PROCESSED_DIR / "corpus_processed.csv",
        ]
        for c in candidates:
            if c.exists():
                corpus_path = c
                break
        if corpus_path is None:
            raise FileNotFoundError("No corpus file found. Pass corpus_path explicitly.")

    if output_path is None:
        output_path = COMPARISON_DIR / "corpus_unified_sentiment.csv"

    print(f"\n  Loading corpus: {corpus_path}")
    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    print(f"  Corpus size: {len(df):,} documents")

    # Normalize source column
    from constants import normalize_group_series
    if "source" in df.columns:
        df["source"] = normalize_group_series(df["source"])

    # Build scorer and run
    pipe = build_unified_scorer()
    df["sentiment_unified"] = score_corpus(df, pipe=pipe)

    # Compute correlation with original if available
    if "sentiment" in df.columns:
        from scipy.stats import spearmanr
        valid = df[["sentiment", "sentiment_unified"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["sentiment"], valid["sentiment_unified"])
            print(f"\n  Spearman correlation (original vs unified): rho={rho:.4f}, p={p:.2e}")

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to: {output_path}")

    # Summary
    if "source" in df.columns:
        print("\n  Unified sentiment by source:")
        for source, group in df.groupby("source"):
            vals = group["sentiment_unified"].dropna()
            print(f"    {source}: M={vals.mean():.4f}, SD={vals.std():.4f}, N={len(vals)}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_unified_scoring(corpus_path=args.corpus, output_path=args.output)
