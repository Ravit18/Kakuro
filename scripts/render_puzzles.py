"""Exporta imágenes sintéticas de Kakuro con su etiqueta JSON.

Sirve para armar un conjunto de prueba reproducible, para imprimir tableros y
fotografiarlos, o para ilustrar el informe.

Uso:
    python -m scripts.render_puzzles --count 20 --out data/synthetic            # con aumentación
    python -m scripts.render_puzzles --count 5 --clean --out data/printables   # limpias (para imprimir)
    python -m scripts.render_puzzles --from-json data/ground_truth --clean --out data/printables
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.puzzle import KakuroPuzzle  # noqa: E402
from render.synth import make_sample, render_board  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=900000, help="semillas distintas a las de entrenamiento")
    ap.add_argument("--clean", action="store_true", help="sin aumentación (para imprimir)")
    ap.add_argument("--from-json", type=Path, help="renderizar los JSON de esta carpeta")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        for i, js in enumerate(sorted(args.from_json.glob("*.json"))):
            data = json.loads(js.read_text(encoding="utf-8"))
            puzzle = KakuroPuzzle.from_grid(data["grid"])
            rng = random.Random(args.seed + i)
            img = cv2.cvtColor(render_board(puzzle, rng, cell=90), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.out / f"{js.stem}.png"), img)
            puzzle.save(args.out / f"{js.stem}.json", **({"solution": data["solution"]} if "solution" in data else {}))
            print(args.out / f"{js.stem}.png")
        return

    for i in range(args.count):
        rng = random.Random(args.seed + i)
        img, puzzle = make_sample(rng, augmented=not args.clean)
        name = f"synth_{args.seed + i}"
        cv2.imwrite(str(args.out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        puzzle.save(args.out / f"{name}.json")
        print(args.out / f"{name}.jpg", f"{puzzle.rows}x{puzzle.cols}")


if __name__ == "__main__":
    main()
