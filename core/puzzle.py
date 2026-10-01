"""Representación de un tablero Kakuro.

Es el contrato entre la Fase 1 (visión) y la Fase 2 (CP): la visión produce un
JSON con este formato y el solver lo consume sin intervención manual.

Formato de la grilla (una lista de filas, cada fila una lista de tokens):
    "."      celda blanca (variable a resolver)
    "#"      celda negra sin pistas
    "D\\R"   celda negra con pistas: D = suma hacia abajo, R = suma hacia la
             derecha. Cualquiera de los dos lados puede ir vacío: "16\\", "\\24".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Cell = tuple[int, int]

WHITE = "."
BLACK = "#"


@dataclass(frozen=True)
class Run:
    """Secuencia maximal de celdas blancas contiguas que debe sumar `total`."""

    direction: str  # "across" | "down"
    clue_cell: Cell
    cells: tuple[Cell, ...]
    total: int


class PuzzleError(ValueError):
    """El tablero es inconsistente (típicamente por un error de OCR)."""


@dataclass
class KakuroPuzzle:
    rows: int
    cols: int
    white: frozenset[Cell]
    down: dict[Cell, int]    # celda pista -> suma vertical
    across: dict[Cell, int]  # celda pista -> suma horizontal

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_grid(cls, grid: list[list[str]]) -> "KakuroPuzzle":
        rows, cols = len(grid), len(grid[0])
        white, down, across = set(), {}, {}
        for r, row in enumerate(grid):
            if len(row) != cols:
                raise PuzzleError(f"la fila {r} tiene {len(row)} celdas, se esperaban {cols}")
            for c, token in enumerate(row):
                token = token.strip()
                if token == WHITE:
                    white.add((r, c))
                elif token == BLACK:
                    continue
                elif "\\" in token:
                    d, a = token.split("\\")
                    if d:
                        down[(r, c)] = int(d)
                    if a:
                        across[(r, c)] = int(a)
                else:
                    raise PuzzleError(f"token inválido {token!r} en ({r}, {c})")
        return cls(rows, cols, frozenset(white), down, across)

    @classmethod
    def load(cls, path: str | Path) -> "KakuroPuzzle":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_grid(data["grid"])

    def to_grid(self) -> list[list[str]]:
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.white:
                    row.append(WHITE)
                elif (r, c) in self.down or (r, c) in self.across:
                    d = self.down.get((r, c), "")
                    a = self.across.get((r, c), "")
                    row.append(f"{d}\\{a}")
                else:
                    row.append(BLACK)
            grid.append(row)
        return grid

    def save(self, path: str | Path, **extra) -> None:
        data = {"rows": self.rows, "cols": self.cols, "grid": self.to_grid(), **extra}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- runs
    def runs(self) -> list[Run]:
        """Extrae todas las corridas (horizontales y verticales) del tablero."""
        runs = []
        for clues, (dr, dc), name in ((self.across, (0, 1), "across"), (self.down, (1, 0), "down")):
            for (r, c), total in clues.items():
                cells = []
                rr, cc = r + dr, c + dc
                while (rr, cc) in self.white:
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                runs.append(Run(name, (r, c), tuple(cells), total))
        return runs

    def validate(self) -> list[Run]:
        """Comprueba que el tablero sea coherente y devuelve sus corridas.

        Útil para detectar errores de la Fase 1 antes de llamar al solver.
        """
        runs = self.runs()
        covered = {"across": set(), "down": set()}
        for run in runs:
            n = len(run.cells)
            if not 1 <= n <= 9:
                raise PuzzleError(f"corrida {run.direction} en {run.clue_cell} tiene {n} celdas")
            lo, hi = n * (n + 1) // 2, sum(range(10 - n, 10))
            if not lo <= run.total <= hi:
                raise PuzzleError(
                    f"suma {run.total} imposible para {n} celdas "
                    f"({run.direction} en {run.clue_cell}); rango válido [{lo}, {hi}]"
                )
            covered[run.direction].update(run.cells)
        for direction, cells in covered.items():
            missing = self.white - cells
            if missing:
                raise PuzzleError(f"celdas blancas sin pista {direction}: {sorted(missing)}")
        return runs

    # ------------------------------------------------------------- display
    def to_text(self, solution: dict[Cell, int] | None = None) -> str:
        """Dibuja el tablero en consola (con la solución si se proporciona)."""
        width = 6
        lines = []
        for r in range(self.rows):
            parts = []
            for c in range(self.cols):
                if (r, c) in self.white:
                    v = solution.get((r, c)) if solution else None
                    parts.append(str(v if v is not None else ".").center(width))
                elif (r, c) in self.down or (r, c) in self.across:
                    d = self.down.get((r, c), "")
                    a = self.across.get((r, c), "")
                    parts.append(f"{d}\\{a}".center(width))
                else:
                    parts.append("#".center(width))
            lines.append("|".join(parts))
        return "\n".join(lines)
