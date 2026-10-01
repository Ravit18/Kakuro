"""Evaluación cuantitativa (para el informe).

    python -m scripts.benchmark vision --synthetic 100                 # CNN sobre sintéticos nuevos
    python -m scripts.benchmark vision --synthetic 100 --engine tesseract
    python -m scripts.benchmark vision --dir data/images               # fotos reales etiquetadas
    python -m scripts.benchmark solver                                 # global vs. tabla

Métricas de visión:
    grid_ok      % de imágenes con filas x columnas correctas
    cells_acc    % de celdas bien clasificadas (blanca / no blanca)
    clues_acc    % de pistas (celda, dirección) leídas con el valor exacto
    exact        % de tableros extraídos sin ningún error
    solved       % de tableros cuya solución (con el JSON extraído) es la correcta
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.puzzle import KakuroPuzzle, PuzzleError  # noqa: E402
from solver import cp_model, cp_table  # noqa: E402


# ------------------------------------------------------------------ visión
def compare(gt: KakuroPuzzle, pred_grid: list[list[str]]) -> dict:
    gt_grid = gt.to_grid()
    dims = len(pred_grid) == gt.rows and all(len(r) == gt.cols for r in pred_grid)
    res = {"grid_ok": dims, "cells": 0, "cells_ok": 0, "clues": 0, "clues_ok": 0, "exact": pred_grid == gt_grid}
    gt_clues = {(k, "d"): v for k, v in gt.down.items()} | {(k, "a"): v for k, v in gt.across.items()}
    res["clues"] = len(gt_clues)
    if not dims:
        res["cells"] = gt.rows * gt.cols
        return res
    pred = KakuroPuzzle.from_grid(pred_grid)
    pc = {(k, "d"): v for k, v in pred.down.items()} | {(k, "a"): v for k, v in pred.across.items()}
    res["cells"] = gt.rows * gt.cols
    res["cells_ok"] = sum(((r, c) in gt.white) == ((r, c) in pred.white)
                          for r in range(gt.rows) for c in range(gt.cols))
    res["clues_ok"] = sum(pc.get(k) == v for k, v in gt_clues.items())
    return res


def solved_ok(gt: KakuroPuzzle, pred_grid) -> bool:
    try:
        pred = KakuroPuzzle.from_grid(pred_grid)
        r = cp_model.solve(pred, time_limit=10)
    except (PuzzleError, ValueError):
        return False
    return bool(r.solution) and pred.white == gt.white and cp_model.check_solution(gt, r.solution)


def bench_vision(samples, engine: str, rows_hint: bool = False):
    from vision.ocr import get_reader
    from vision.pipeline import extract

    reader = get_reader(engine)
    tot = {"n": 0, "grid_ok": 0, "cells": 0, "cells_ok": 0, "clues": 0, "clues_ok": 0, "exact": 0, "solved": 0}
    t_total, failures = 0.0, []
    for name, img, gt in samples:
        t0 = time.perf_counter()
        try:
            ex = extract(img, reader=reader, rows=gt.rows if rows_hint else None,
                         cols=gt.cols if rows_hint else None)
            grid = ex.data["grid"]
        except Exception as e:  # noqa: BLE001
            grid, _ = [[]], failures.append(f"{name}: {e}")
        t_total += time.perf_counter() - t0
        m = compare(gt, grid)
        tot["n"] += 1
        for k in ("grid_ok", "exact"):
            tot[k] += int(m[k])
        for k in ("cells", "cells_ok", "clues", "clues_ok"):
            tot[k] += m[k]
        tot["solved"] += int(solved_ok(gt, grid)) if m["grid_ok"] else 0
        if not m["exact"]:
            failures.append(f"{name}: grid_ok={m['grid_ok']} pistas {m['clues_ok']}/{m['clues']}")
    n = max(1, tot["n"])
    print(f"\nMotor OCR: {reader.name}   imágenes: {tot['n']}   tiempo medio: {t_total / n:.3f}s")
    print(f"  grid_ok   {100 * tot['grid_ok'] / n:6.2f} %")
    print(f"  cells_acc {100 * tot['cells_ok'] / max(1, tot['cells']):6.2f} %")
    print(f"  clues_acc {100 * tot['clues_ok'] / max(1, tot['clues']):6.2f} %")
    print(f"  exact     {100 * tot['exact'] / n:6.2f} %")
    print(f"  solved    {100 * tot['solved'] / n:6.2f} %")
    if failures:
        print("\nCasos con error:")
        for f in failures[:25]:
            print("  ", f)


def synthetic_samples(n: int, seed: int):
    from render.synth import make_sample

    for i in range(n):
        img, puzzle = make_sample(random.Random(seed + i), augmented=True)
        yield f"synth_{seed + i}", img, puzzle


def folder_samples(folder: Path):
    from vision.preprocess import load_image

    for js in sorted(folder.glob("*.json")):
        imgs = [p for p in folder.glob(js.stem + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs:
            gt = KakuroPuzzle.from_grid(json.loads(js.read_text(encoding="utf-8"))["grid"])
            yield imgs[0].name, load_image(imgs[0]), gt


# ------------------------------------------------------------------ solver
def bench_solver(seed: int, count: int):
    from scripts.generate_puzzles import PATTERNS, puzzle_from_solution, random_fill

    print(f"{'patrón':8s} {'modelo':7s} {'tiempo_s':>9s} {'ramas':>8s} {'conflictos':>10s}")
    for name, mask in PATTERNS.items():
        for model_name, fn in (("global", cp_model.solve), ("table", cp_table.solve)):
            ts, br, cf = [], [], []
            for i in range(count):
                # Tablero válido (no necesariamente de solución única): basta para medir tiempos.
                fill = random_fill(mask, random.Random(seed + i))
                r = fn(puzzle_from_solution(mask, fill))
                ts.append(r.stats["solve_s"])
                br.append(r.stats["branches"])
                cf.append(r.stats["conflicts"])
            k = len(ts)
            print(f"{name:8s} {model_name:7s} {sum(ts) / k:9.4f} {sum(br) / k:8.1f} {sum(cf) / k:10.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vision")
    v.add_argument("--synthetic", type=int, default=0, help="nº de imágenes sintéticas nuevas")
    v.add_argument("--dir", type=Path, help="carpeta con imágenes + JSON etiquetados")
    v.add_argument("--engine", choices=["auto", "cnn", "tesseract"], default="auto")
    v.add_argument("--seed", type=int, default=500000, help="semillas distintas a las de entrenamiento")
    v.add_argument("--size-hint", action="store_true", help="pasar filas/columnas reales")
    s = sub.add_parser("solver")
    s.add_argument("--count", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "solver":
        bench_solver(args.seed, args.count)
        return
    if args.synthetic:
        bench_vision(synthetic_samples(args.synthetic, args.seed), args.engine, args.size_hint)
    if args.dir:
        bench_vision(folder_samples(args.dir), args.engine, args.size_hint)
    if not args.synthetic and not args.dir:
        ap.error("indica --synthetic N y/o --dir carpeta")


if __name__ == "__main__":
    main()
