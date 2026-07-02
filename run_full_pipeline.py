"""
run_full_pipeline.py — Execute all pipeline stages in sequence
===============================================================
1. Preprocess (integrate public voice into unified corpus)
2. Full analysis (topics, sentiment, networks, comparison, figures)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    t0 = time.time()

    print()
    print("=" * 70)
    print("  VOZ_SUS — FULL PIPELINE (preprocess + analyze)")
    print("=" * 70)
    print()

    # ── Step 1: Preprocess ──────────────────────────────────────────
    print("STEP 1/2: PREPROCESSING")
    print("-" * 70)
    from main_preprocess import main as preprocess_main
    preprocess_main()

    # ── Step 2: Full analysis ───────────────────────────────────────
    print()
    print("STEP 2/2: ANALYSIS + VISUALIZATION")
    print("-" * 70)

    # Override sys.argv so main_analyze runs with --phase all
    sys.argv = ["main_analyze.py", "--phase", "all"]
    from main_analyze import main as analyze_main
    analyze_main()

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print()
    print("=" * 70)
    print(f"  FULL PIPELINE COMPLETE in {mins}m {secs}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
