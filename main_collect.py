"""
main_collect.py — Orquestador principal del pipeline de recolección VOZ_SUS
===========================================================================

Ejecuta los tres módulos de recolección de datos en secuencia:
  1. Academic  → Artículos científicos (OpenAlex)
  2. Institutional → Discursos institucionales (YouTube + transcripciones)
  3. Public         → Comentarios ciudadanos (YouTube)

Uso:
  python main_collect.py                  # Ejecutar todo
  python main_collect.py --phase academic      # Solo académico
  python main_collect.py --phase institutional
  python main_collect.py --phase public
  python main_collect.py --phase all             # Equivale a sin argumento
"""

# ── stdlib ──────────────────────────────────────────────────────────
import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Configuración del proyecto ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: E402, F403


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES DE CADA FASE
# ─────────────────────────────────────────────────────────────────────


def run_academic_phase() -> dict:
    """Ejecuta la fase de recolección académica.

    Returns:
        Diccionario con estadísticas de la fase.
    """
    from src.collection.academic_collector import collect_academic_corpus

    print("\n" + "█" * 70)
    print("█  FASE 1: RECOLECCIÓN ACADÉMICA (OpenAlex)")
    print("█" * 70)

    start = time.time()
    df = collect_academic_corpus()
    elapsed = time.time() - start

    return {
        "phase": "academic",
        "status": "✓ completada",
        "records": len(df),
        "time_seconds": round(elapsed, 1),
        "output": str(ACADEMIC_CSV),
    }


def run_institutional_phase() -> dict:
    """Ejecuta la fase de recolección de discursos institucionales.

    Returns:
        Diccionario con estadísticas de la fase.
    """
    from src.collection.political_collector import collect_political_corpus

    print("\n" + "█" * 70)
    print("█  FASE 2: RECOLECCIÓN INSTITUCIONAL (YouTube Speeches)")
    print("█" * 70)

    start = time.time()
    df = collect_political_corpus()
    elapsed = time.time() - start

    return {
        "phase": "institutional",
        "status": "✓ completada",
        "records": len(df),
        "time_seconds": round(elapsed, 1),
        "output": str(INSTITUTIONAL_CSV),
    }


def run_public_phase() -> dict:
    """Ejecuta la fase de recolección de comentarios ciudadanos.

    Returns:
        Diccionario con estadísticas de la fase.
    """
    from src.collection.public_collector import collect_public_corpus

    print("\n" + "█" * 70)
    print("█  FASE 3: RECOLECCIÓN PÚBLICA (YouTube Comments)")
    print("█" * 70)

    start = time.time()
    df = collect_public_corpus()
    elapsed = time.time() - start

    return {
        "phase": "public",
        "status": "✓ completada",
        "records": len(df),
        "time_seconds": round(elapsed, 1),
        "output": str(PUBLIC_CSV),
    }


# ─────────────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Punto de entrada principal del orquestador de recolección.

    Parsea argumentos de línea de comandos y ejecuta las fases
    solicitadas. Si una fase falla, continúa con la siguiente.
    """
    parser = argparse.ArgumentParser(
        description="VOZ_SUS — Pipeline de recolección de datos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fases disponibles:
  academic       Artículos científicos desde OpenAlex
  institutional  Discursos institucionales desde YouTube
  public         Comentarios ciudadanos desde YouTube
  all            Ejecutar las tres fases en secuencia
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["academic", "institutional", "political", "public", "all"],
        default="all",
        help="Fase a ejecutar (default: all)",
    )
    args = parser.parse_args()

    # Banner de inicio
    print("\n" + "═" * 70)
    print("  🔬 VOZ_SUS — Pipeline de Recolección de Datos")
    print("  📋 Análisis del discurso de sustentabilidad")
    print("  🗂️  Tres voces: académica · institucional · ciudadana")
    print("═" * 70)
    print(f"  📁 Proyecto:  {BASE_DIR}")
    print(f"  📂 Datos:     {RAW_DIR}")
    print(f"  ⚙️  Fase:      {args.phase}")
    print("═" * 70)

    total_start = time.time()

    # Definir qué fases ejecutar
    phase_map = {
        "academic": run_academic_phase,
        "institutional": run_institutional_phase,
        "political": run_institutional_phase,  # backward compat alias
        "public": run_public_phase,
    }

    if args.phase == "all":
        phases_to_run = ["academic", "institutional", "public"]
    else:
        phases_to_run = [args.phase]

    # Ejecutar fases
    results: list[dict] = []

    for phase_name in phases_to_run:
        phase_func = phase_map[phase_name]

        try:
            result = phase_func()
            results.append(result)

        except Exception as e:
            print(f"\n  ✗ ERROR en fase '{phase_name}': {e}")
            traceback.print_exc()
            results.append({
                "phase": phase_name,
                "status": f"✗ error: {str(e)[:60]}",
                "records": 0,
                "time_seconds": 0,
                "output": "N/A",
            })
            print(f"  ⏩ Continuando con la siguiente fase...\n")

    total_elapsed = time.time() - total_start

    # ── Resumen final ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  📊 RESUMEN DE RECOLECCIÓN")
    print("═" * 70)

    total_records = 0
    for r in results:
        total_records += r["records"]
        print(
            f"  {r['status']:20s} │ {r['phase']:10s} │ "
            f"{r['records']:>8,} registros │ {r['time_seconds']:>7.1f}s"
        )
        if r["output"] != "N/A":
            print(f"  {'':20s} │ {'':10s} │ 💾 {r['output']}")

    print("-" * 70)
    print(f"  📦 Total registros: {total_records:,}")
    print(f"  ⏱️  Tiempo total:   {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("═" * 70)
    print("  🏁 Pipeline de recolección finalizado")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
