"""Pipeline Kakuro end-to-end: imagen (o JSON) -> visión -> modelo CP -> solución.

Uso:
    python main.py data/images/ejemplo.jpg                 # imagen -> solución
    python main.py data/images/ejemplo.jpg --only-extract  # sólo Fase 1 (JSON)
    python main.py foto.jpg --engine tesseract --debug     # OCR alternativo + imágenes intermedias
    python main.py foto.jpg --rows 9 --cols 9              # forzar tamaño si la detección falla
    python main.py data/ground_truth/gen_7x7_00.json --model table --unique

Salidas (en data/outputs/<nombre>/):
    puzzle.json     estado inicial extraído (contrato Fase 1 -> Fase 2)
    solution.png    solución dibujada sobre la foto original
    debug/*.png     etapas intermedias de visión (con --debug)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

from core.puzzle import KakuroPuzzle, PuzzleError
from solver import cp_model, cp_table

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="imagen del Kakuro o JSON del tablero")
    ap.add_argument("--model", choices=["global", "table"], default="global",
                    help="global = AllDifferent + suma; table = AllowedAssignments")
    ap.add_argument("--unique", action="store_true", help="verificar que la solución sea única")
    ap.add_argument("--engine", choices=["auto", "cnn", "tesseract"], default="auto", help="motor OCR")
    ap.add_argument("--rows", type=int, help="número de filas (si se conoce)")
    ap.add_argument("--cols", type=int, help="número de columnas (si se conoce)")
    ap.add_argument("--only-extract", action="store_true", help="sólo Fase 1: guardar el JSON")
    ap.add_argument("--debug", action="store_true", help="guardar imágenes intermedias")
    ap.add_argument("--out", type=Path, default=Path("data/outputs"))
    args = ap.parse_args()

    ext = args.input.suffix.lower()
    extraction = None
    try:
        if ext == ".json":
            puzzle = KakuroPuzzle.load(args.input)
        elif ext in IMAGE_EXT:
            from vision.pipeline import extract, save_debug

            out_dir = args.out / args.input.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            extraction = extract(args.input, args.engine, args.rows, args.cols)
            (out_dir / "puzzle.json").write_text(json.dumps(extraction.data, indent=2), encoding="utf-8")
            if args.debug:
                save_debug(extraction, out_dir / "debug")
            print(f"Fase 1: {extraction.data['rows']}x{extraction.data['cols']} "
                  f"(OCR: {extraction.data['ocr_engine']}) -> {out_dir / 'puzzle.json'}")
            for w in extraction.warnings:
                print(f"  aviso: {w}")
            puzzle = extraction.puzzle
        else:
            print(f"Formato no soportado: {ext}", file=sys.stderr)
            return 2

        print(puzzle.to_text(), "\n")
        if args.only_extract:
            return 0
        if args.model == "table":
            result = cp_table.solve(puzzle)
        else:
            result = cp_model.solve(puzzle, check_unique=args.unique)
    except PuzzleError as e:
        print(f"Tablero inválido (¿error de lectura?): {e}", file=sys.stderr)
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
        if extraction is not None:
            from render.overlay import draw_solution
            from vision.preprocess import load_image

            img = draw_solution(load_image(args.input), extraction.H, extraction.warped_shape,
                                extraction.grid_lines, result.solution)
            path = args.out / args.input.stem / "solution.png"
            cv2.imwrite(str(path), img)
            print(f"\nSolución dibujada en {path}")
    return 0 if result.solution else 1


if __name__ == "__main__":
    sys.exit(main())
