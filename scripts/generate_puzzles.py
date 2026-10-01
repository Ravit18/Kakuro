"""Genera puzzles Kakuro con solución única a partir de patrones de celdas.

Procedimiento:
  1. Se rellena el patrón con dígitos aleatorios sin repetir en cada corrida.
  2. Se derivan las pistas (sumas) de ese relleno.
  3. Se usa el solver CP para verificar que la solución sea única; si no, se reintenta.

Uso:
    python -m scripts.generate_puzzles --count 5 --out data/ground_truth
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.puzzle import Cell, KakuroPuzzle  # noqa: E402
from solver.cp_model import solve  # noqa: E402

# "#" = negra, "." = blanca. La primera fila y la primera columna son negras.
PATTERNS = {
    "5x5": [
        "#####",
        "#..##",
        "#...#",
        "#...#",
        "##..#",
    ],
    "7x7": [
        "#######",
        "#..#..#",
        "#.....#",
        "##..###",
        "###..##",
        "#.....#",
        "#..#..#",
    ],
    "8x8": [
        "########",
        "#..##..#",
        "#...#...",
        "##......",
        "#..##..#",
        "#......#",
        "#..#...#",
        "#..##..#",
    ],
    "10x10": [
        "##########",
        "#..##...##",
        "#...#....#",
        "##...#...#",
        "#...##..##",
        "#.....#..#",
        "##..#.....",
        "#...##...#",
        "#....#...#",
        "##...##..#",
    ],
}


def mask_runs(mask: list[str]) -> list[list[Cell]]:
    """Corridas de celdas blancas (horizontales y verticales) de un patrón."""
    rows, cols = len(mask), len(mask[0])
    runs = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(rows):
            for c in range(cols):
                if mask[r][c] != "#":
                    continue
                cells, rr, cc = [], r + dr, c + dc
                while rr < rows and cc < cols and mask[rr][cc] == ".":
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                if cells:
                    runs.append(cells)
    return runs


def random_fill(mask: list[str], rng: random.Random) -> dict[Cell, int] | None:
    """Backtracking con orden aleatorio de dígitos; sin repetición por corrida."""
    runs = mask_runs(mask)
    cell_runs: dict[Cell, list[int]] = {}
    for i, run in enumerate(runs):
        for c in run:
            cell_runs.setdefault(c, []).append(i)
    cells = sorted(cell_runs)
    used = [set() for _ in runs]
    fill: dict[Cell, int] = {}

    def bt(k: int) -> bool:
        if k == len(cells):
            return True
        cell = cells[k]
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for d in digits:
            if any(d in used[i] for i in cell_runs[cell]):
                continue
            fill[cell] = d
            for i in cell_runs[cell]:
                used[i].add(d)
            if bt(k + 1):
                return True
            for i in cell_runs[cell]:
                used[i].discard(d)
        fill.pop(cell, None)
        return False

    return fill if bt(0) else None


def puzzle_from_solution(mask: list[str], solution: dict[Cell, int]) -> KakuroPuzzle:
    rows, cols = len(mask), len(mask[0])
    white = frozenset((r, c) for r in range(rows) for c in range(cols) if mask[r][c] == ".")
    down, across = {}, {}
    for r in range(rows):
        for c in range(cols):
            if mask[r][c] != "#":
                continue
            for (dr, dc), clues in (((0, 1), across), ((1, 0), down)):
                total, rr, cc = 0, r + dr, c + dc
                while (rr, cc) in white:
                    total += solution[(rr, cc)]
                    rr, cc = rr + dr, cc + dc
                if total:
                    clues[(r, c)] = total
    return KakuroPuzzle(rows, cols, white, down, across)


def generate(mask: list[str], rng: random.Random, max_tries: int = 2000):
    for attempt in range(1, max_tries + 1):
        fill = random_fill(mask, rng)
        if fill is None:
            raise ValueError("el patrón no admite ningún relleno válido")
        puzzle = puzzle_from_solution(mask, fill)
        if solve(puzzle, check_unique=True).num_solutions == 1:
            return puzzle, fill, attempt
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", choices=PATTERNS, nargs="*", default=list(PATTERNS))
    ap.add_argument("--count", type=int, default=1, help="puzzles por patrón")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/ground_truth"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.pattern:
        for i in range(args.count):
            result = generate(PATTERNS[name], rng)
            if result is None:
                print(f"[{name}] no se encontró puzzle único")
                continue
            puzzle, fill, tries = result
            path = args.out / f"gen_{name}_{i:02d}.json"
            sol = [[fill.get((r, c), 0) for c in range(puzzle.cols)] for r in range(puzzle.rows)]
            puzzle.save(path, solution=sol)
            print(f"[{name}] {path} (intentos: {tries})")


if __name__ == "__main__":
    main()
