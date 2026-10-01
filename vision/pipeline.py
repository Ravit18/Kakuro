"""Fase 1 completa: imagen -> JSON del tablero (contrato con la Fase 2).

    from vision.pipeline import extract
    result = extract("data/images/foto.jpg")
    result.data      # dict listo para json.dump / KakuroPuzzle.from_grid
    result.puzzle    # KakuroPuzzle

Después del OCR se aplica una reparación guiada por las reglas del Kakuro:
si una suma leída es imposible para la longitud de su corrida (p. ej. 47 en 2
celdas), se reemplaza por la alternativa más probable del clasificador que sí
sea factible. Así el conocimiento del dominio corrige errores de la red.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.puzzle import KakuroPuzzle
from vision import cells as C
from vision.grid import detect_grid, draw_grid
from vision.ocr import Reading, get_reader
from vision.preprocess import load_image, preprocess


@dataclass
class Extraction:
    data: dict
    puzzle: KakuroPuzzle
    H: np.ndarray                     # homografía original -> rectificada
    warped_shape: tuple[int, int]
    grid_lines: tuple[np.ndarray, np.ndarray]
    warnings: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def _run_length(white: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    n, rr, cc = 0, r + dr, c + dc
    while rr < white.shape[0] and cc < white.shape[1] and white[rr, cc]:
        n, rr, cc = n + 1, rr + dr, cc + dc
    return n


def _feasible(total: int, n: int) -> bool:
    return 1 <= n <= 9 and n * (n + 1) // 2 <= total <= sum(range(10 - n, 10))


def _resolve(reading: Reading, n: int, where: str, warnings: list[str]) -> int | None:
    if n == 0:
        if reading.value is not None:
            warnings.append(f"{where}: se leyó {reading.value} pero no hay corrida; se ignora")
        return None
    if reading.value is None:
        warnings.append(f"{where}: corrida de {n} celdas sin pista legible")
        return None
    if _feasible(reading.value, n):
        return reading.value
    for v, _ in reading.alternatives:
        if _feasible(v, n):
            warnings.append(f"{where}: {reading.value} es imposible para {n} celdas; corregido a {v}")
            return v
    warnings.append(f"{where}: {reading.value} es imposible para {n} celdas y no hay alternativa")
    return reading.value


def extract(image, engine: str = "auto", rows: int | None = None, cols: int | None = None,
            reader=None) -> Extraction:
    img = load_image(image) if isinstance(image, (str, Path)) else image
    rect = preprocess(img)
    grid = detect_grid(rect.gray, rows, cols)
    info = C.analyze_cells(rect.gray, grid)
    reader = reader or get_reader(engine)

    white = np.array([[ci.kind == C.WHITE for ci in row] for row in info])
    warnings: list[str] = []
    tokens, confidences = [], {}
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            ci = info[r][c]
            if ci.kind == C.WHITE:
                row.append(".")
                continue
            nd, na = _run_length(white, r, c, 1, 0), _run_length(white, r, c, 0, 1)
            if ci.kind == C.BLACK and nd == 0 and na == 0:
                row.append("#")
                continue
            rd = reader.read(ci.down_digits, ci.down_mask) if ci.kind == C.CLUE else Reading(None, 1.0, [])
            ra = reader.read(ci.across_digits, ci.across_mask) if ci.kind == C.CLUE else Reading(None, 1.0, [])
            d = _resolve(rd, nd, f"({r},{c}) vertical", warnings)
            a = _resolve(ra, na, f"({r},{c}) horizontal", warnings)
            if d is None and a is None:
                row.append("#")
            else:
                row.append(f"{'' if d is None else d}\\{'' if a is None else a}")
                confidences[f"{r},{c}"] = round(min(rd.confidence, ra.confidence), 4)
        tokens.append(row)

    puzzle = KakuroPuzzle.from_grid(tokens)
    data = {
        "rows": grid.rows, "cols": grid.cols, "grid": tokens,
        "source": str(image) if isinstance(image, (str, Path)) else None,
        "ocr_engine": reader.name,
        "clue_confidence": confidences,
        "warnings": warnings,
    }
    debug = dict(rect.debug)
    debug["5_grid"] = draw_grid(rect.color, grid)
    debug["6_cells"] = C.draw_cells(rect.color, grid, info)
    debug["7_digits"] = _digit_mosaic(info)
    return Extraction(data, puzzle, rect.H, rect.gray.shape, (grid.ys, grid.xs), warnings, debug)


def _digit_mosaic(info) -> np.ndarray:
    crops = [d for row in info for ci in row for d in ci.down_digits + ci.across_digits]
    if not crops:
        return np.zeros((28, 28), np.uint8)
    per = 20
    rows = [crops[i: i + per] for i in range(0, len(crops), per)]
    rows[-1] += [np.zeros_like(crops[0])] * (per - len(rows[-1]))
    mosaic = np.vstack([np.hstack(r) for r in rows])
    return cv2.resize(mosaic, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def save_debug(ex: Extraction, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in ex.debug.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), img)
