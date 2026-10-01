"""Pipeline Kakuro: imagen/JSON -> modelo CP -> solución.

Por ahora acepta un JSON (salida de la Fase 1). Cuando esté lista la parte de
visión, se añadirá la entrada por imagen:  python main.py foto.jpg

Uso:
    python main.py data/ground_truth/gen_7x7_00.json
    python main.py data/ground_truth/gen_7x7_00.json --model table --unique
"""
import argparse
import sys
from pathlib import Path

from core.puzzle import KakuroPuzzle, PuzzleError
from solver import cp_model, cp_table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="JSON del tablero (más adelante: imagen)")
    ap.add_argument("--model", choices=["global", "table"], default="global",
                    help="global = AllDifferent + suma; table = AllowedAssignments")
    ap.add_argument("--unique", action="store_true", help="verificar que la solución sea única")
    args = ap.parse_args()

    if args.input.suffix.lower() != ".json":
        print("La entrada por imagen aún no está implementada (Fase 1).", file=sys.stderr)
        return 2

    try:
        puzzle = KakuroPuzzle.load(args.input)
        print(puzzle.to_text(), "\n")
        if args.model == "table":
            result = cp_table.solve(puzzle)
        else:
            result = cp_model.solve(puzzle, check_unique=args.unique)
    except PuzzleError as e:
        print(f"Tablero inválido: {e}", file=sys.stderr)
        return 1

    print(f"Estado: {result.status}")
    if result.num_solutions is not None:
        print("Solución única" if result.num_solutions == 1 else "¡Hay más de una solución!")
    for k, v in result.stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if result.solution:
        print()
        print(puzzle.to_text(result.solution))
        assert cp_model.check_solution(puzzle, result.solution)
    return 0 if result.solution else 1


if __name__ == "__main__":
    sys.exit(main())
