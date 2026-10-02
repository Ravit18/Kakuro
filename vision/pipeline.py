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

from core.puzzle import KakuroPuzzle, PuzzleError
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


def _rotation(k: int, w: int, h: int) -> np.ndarray:
    """Homografía que lleva coordenadas del tablero rectificado (w x h) a np.rot90(img, k)."""
    if k == 0:
        return np.eye(3)
    if k == 1:   # 90° antihorario
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], float)
    if k == 2:   # 180°
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float)
    return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float)   # 90° horario


@dataclass
class _BoardReading:
    """Lectura completa del tablero en una orientación concreta (k giros de 90°)."""
    k: int
    gray: np.ndarray
    color: np.ndarray
    grid: object
    info: list
    tokens: list
    warnings: list
    confidences: dict
    clues: list            # (fila, col, "vertical"/"horizontal", Reading, n_celdas, valor)


def _read_board(gray, color, k, rows, cols, reader) -> _BoardReading:
    grid = detect_grid(gray, rows, cols)
    info = C.analyze_cells(gray, grid)
    white = np.array([[ci.kind == C.WHITE for ci in row] for row in info])
    warnings: list[str] = []
    tokens, confidences, clues = [], {}, []
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
                if d is not None:
                    clues.append((r, c, "vertical", rd, nd, d))
                if a is not None:
                    clues.append((r, c, "horizontal", ra, na, a))
        tokens.append(row)
    return _BoardReading(k, gray, color, grid, info, tokens, warnings, confidences, clues)


def _check(tokens) -> tuple[bool, int]:
    """(tablero válido, nº de soluciones hasta 2) usando el modelo CP de la Fase 2."""
    from solver.cp_model import solve  # import diferido: la Fase 2 valida a la Fase 1

    try:
        puzzle = KakuroPuzzle.from_grid(tokens)
        puzzle.validate()
    except (PuzzleError, ValueError):
        return False, 0
    # un tablero sin celdas blancas (o casi) es "válido" pero no es un Kakuro: suele
    # salir al leer en una orientación equivocada (colores invertidos)
    if len(puzzle.white) < max(4, 0.15 * puzzle.rows * puzzle.cols):
        return False, 0
    res = solve(puzzle, check_unique=True, time_limit=5)
    return True, int(res.num_solutions or 0)


def _quality(rd: _BoardReading) -> tuple:
    """Puntaje de una orientación (mayor es mejor).

    1. Kakuro válido con solución única: prácticamente imposible si la orientación
       es incorrecta.
    2. Menos avisos de lectura (pistas sin corrida, corridas sin pista, sumas
       imposibles): en una orientación equivocada aparecen muchos.
    3. Estructura válida y, por último, confianza media del OCR.
    No se exige que el tablero tenga solución: un solo dígito mal leído en la
    orientación correcta lo vuelve infactible (eso lo arregla repair_with_solver).
    """
    valid, n = _check(rd.tokens)
    conf = float(np.mean(list(rd.confidences.values()))) if rd.confidences else 0.0
    return (valid and n == 1, -len(rd.warnings), valid, conf)


def _set_clue(tokens, r, c, direction, value):
    d, a = tokens[r][c].split("\\")
    if direction == "vertical":
        d = str(value)
    else:
        a = str(value)
    tokens[r][c] = f"{d}\\{a}"


def repair_with_solver(rd: _BoardReading, max_clues: int = 10, max_pairs: int = 6) -> list[str]:
    """Si el tablero leído NO tiene solución (señal segura de un dígito mal leído),
    prueba las lecturas alternativas de la CNN en las pistas menos seguras (primero de
    a una, luego de a dos) y se queda con la que vuelve el tablero resoluble,
    prefiriendo la que da solución única y, a igualdad, la más probable según la CNN.
    Así el modelo CP corrige errores del OCR (p. ej. un 16 leído como 11).
    No se toca un tablero que ya tiene solución (aunque no sea única): hay Kakuros
    válidos con varias soluciones y "arreglarlos" introduciría errores."""
    valid, n = _check(rd.tokens)
    if not valid or n >= 1:
        return []
    cands = []
    for r, c, direction, reading, ncell, value in rd.clues:
        alts = [(v, p) for v, p in reading.alternatives if v != value and _feasible(v, ncell)][:3]
        if alts:
            cands.append((reading.confidence, r, c, direction, value, alts))
    cands.sort(key=lambda t: t[0])

    def best_of(options):
        """Entre las correcciones que dejan el tablero resoluble: la de solución única y,
        a igualdad, la más probable según la CNN."""
        ok = []
        for tokens, prob, msg in options:
            valid_t, n_sol = _check(tokens)
            if valid_t and n_sol >= 1:
                ok.append((n_sol == 1, prob, tokens, msg))
        return max(ok, key=lambda o: (o[0], o[1])) if ok else None

    singles = []
    for _, r, c, direction, value, alts in cands[:max_clues]:
        for v, p in alts:
            t = [row[:] for row in rd.tokens]
            _set_clue(t, r, c, direction, v)
            singles.append((t, p, f"({r},{c}) {direction}: {value} corregido a {v}"))
    found = best_of(singles)
    if found is None:
        pairs = []
        top = cands[:max_pairs]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                (_, r1, c1, d1, v1, a1), (_, r2, c2, d2, v2, a2) = top[i], top[j]
                for x, p in a1:
                    for y, q in a2:
                        t = [row[:] for row in rd.tokens]
                        _set_clue(t, r1, c1, d1, x)
                        _set_clue(t, r2, c2, d2, y)
                        pairs.append((t, p * q, f"({r1},{c1}) {d1}: {v1} corregido a {x} y "
                                                 f"({r2},{c2}) {d2}: {v2} corregido a {y}"))
        found = best_of(pairs)
    if found is None:
        return ["el tablero leído no tiene solución y no se encontró una corrección"]
    rd.tokens = found[2]
    return [found[3] + " (el tablero leído no tenía solución; el solver valida la corrección)"]


def extract(image, engine: str = "auto", rows: int | None = None, cols: int | None = None,
            reader=None) -> Extraction:
    """Imagen -> tablero.

    Orientación: si la foto está girada (90°/180°/270°, p. ej. hoja volteada o
    tablero muy inclinado), se prueban las 4 rotaciones del tablero rectificado y
    se elige la lectura que da un Kakuro válido con solución única (y, a igualdad,
    menos avisos y más confianza del OCR). Si la orientación original ya es
    perfecta no se prueban las demás.
    """
    img = load_image(image) if isinstance(image, (str, Path)) else image
    rect = preprocess(img)
    reader = reader or get_reader(engine)
    h, w = rect.gray.shape

    best, best_q = None, None
    for k in range(4):
        g, col = np.ascontiguousarray(np.rot90(rect.gray, k)), np.ascontiguousarray(np.rot90(rect.color, k))
        rr, cc = (rows, cols) if k % 2 == 0 else (cols, rows)
        try:
            rd = _read_board(g, col, k, rr, cc, reader)
        except Exception:  # noqa: BLE001  (una orientación absurda puede no tener grilla)
            continue
        q = _quality(rd)
        if best is None or q > best_q:
            best, best_q = rd, q
        if k == 0 and q[0] and not rd.warnings:
            break  # orientación original perfecta: no hace falta probar las demás
    if best is None:
        raise ValueError("no se pudo leer el tablero en ninguna orientación")

    rd = best
    warnings = list(rd.warnings)
    if rd.k:
        warnings.append(f"tablero girado: se leyó rotado {90 * rd.k}° (antihorario)")
    warnings += repair_with_solver(rd)
    tokens, grid = rd.tokens, rd.grid
    H = _rotation(rd.k, w, h) @ rect.H

    puzzle = KakuroPuzzle.from_grid(tokens)
    data = {
        "rows": grid.rows, "cols": grid.cols, "grid": tokens,
        "source": str(image) if isinstance(image, (str, Path)) else None,
        "ocr_engine": reader.name,
        "clue_confidence": rd.confidences,
        "warnings": warnings,
    }
    debug = dict(rect.debug)
    debug["4_warped"] = rd.color
    debug["5_grid"] = draw_grid(rd.color, grid)
    debug["6_cells"] = C.draw_cells(rd.color, grid, rd.info)
    debug["7_digits"] = _digit_mosaic(rd.info)
    return Extraction(data, puzzle, H, rd.gray.shape, (grid.ys, grid.xs), warnings, debug)


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
