"""Generador de imágenes sintéticas de Kakuro (datos de entrenamiento y prueba).

No existe un dataset público de fotos de Kakuro etiquetadas, así que se generan:
  1. Tableros aleatorios (patrón + relleno válido -> sumas) con etiqueta exacta.
  2. Renderizado con distintas fuentes, tamaños, grosores y dos estilos
     (celdas negras con texto blanco / celdas grises con texto negro).
  3. Aumentación "fotográfica": hoja sobre un fondo, perspectiva, rotación,
     iluminación no uniforme, desenfoque, ruido y compresión JPEG.
Esto es "domain randomization" [Tobin et al., 2017]: si el modelo ve suficiente
variación sintética, generaliza a fotos reales.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.puzzle import KakuroPuzzle
from scripts.generate_puzzles import puzzle_from_solution, random_fill

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
_SYSTEM_FONTS = [  # se usan además si existen (Windows / macOS / Linux)
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
]


def font_paths() -> list[str]:
    fonts = sorted(str(p) for p in FONT_DIR.glob("*.tt[fc]"))
    fonts += [f for f in _SYSTEM_FONTS if Path(f).exists()]
    if not fonts:
        raise FileNotFoundError(f"no hay fuentes en {FONT_DIR}")
    return fonts


# ------------------------------------------------------------ tableros
def random_mask(rows: int, cols: int, rng: random.Random, p_white: float = 0.72) -> list[str]:
    """Patrón aleatorio: fila y columna 0 negras, corridas de 2..9 celdas."""
    for _ in range(200):
        m = [["#"] * cols for _ in range(rows)]
        for r in range(1, rows):
            for c in range(1, cols):
                if rng.random() < p_white:
                    m[r][c] = "."
        # eliminar corridas de longitud 1 (no válidas en Kakuro) iterativamente
        changed = True
        while changed:
            changed = False
            for r in range(1, rows):
                for c in range(1, cols):
                    if m[r][c] != ".":
                        continue
                    h = (m[r][c - 1] == ".") or (c + 1 < cols and m[r][c + 1] == ".")
                    v = (m[r - 1][c] == ".") or (r + 1 < rows and m[r + 1][c] == ".")
                    if not (h and v):
                        m[r][c] = "#"
                        changed = True
        mask = ["".join(row) for row in m]
        whites = sum(row.count(".") for row in mask)
        if whites >= 0.45 * (rows - 1) * (cols - 1):
            return mask
    raise RuntimeError("no se pudo generar un patrón")


def random_puzzle(rng: random.Random, min_side: int = 5, max_side: int = 10) -> KakuroPuzzle:
    while True:
        rows, cols = rng.randint(min_side, max_side), rng.randint(min_side, max_side)
        mask = random_mask(rows, cols, rng)
        fill = random_fill(mask, rng)
        if fill:
            return puzzle_from_solution(mask, fill)


# ---------------------------------------------------------- renderizado
STYLES = ["dark", "gray", "inverted", "half", "color"]
STYLE_WEIGHTS = [3, 2, 2, 2, 2]


def _dashed(d, p0, p1, fill, width, dash):
    (x0, y0), (x1, y1) = p0, p1
    L = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
    n = int(L // (2 * dash))
    for k in range(n + 1):
        t0, t1 = 2 * k * dash / L, min(1.0, (2 * k + 1) * dash / L)
        d.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
               fill=fill, width=width)


def render_board(puzzle: KakuroPuzzle, rng: random.Random, cell: int | None = None,
                 style: str | None = None, font_path: str | None = None) -> np.ndarray:
    """Dibuja el tablero limpio (vista cenital). Devuelve imagen RGB uint8.

    Estilos (variantes reales de Kakuro impresos y de apps):
      dark      celdas pista negras, texto blanco (clásico)
      gray      celdas pista grises, texto negro
      inverted  celdas a rellenar NEGRAS, pistas grises con texto negro (apps móviles)
      half      la mitad con pista es blanca (texto negro) y la mitad sin pista es negra
      color     tonos de color (p. ej. amarillo claro / amarillo oscuro)
    Además, al azar: diagonal punteada o ausente en celdas vacías, sombra
    desplazada, grosor de línea y recorte sin margen.
    """
    cell = cell or rng.randint(36, 90)
    lw = rng.choice([1, 1, 2, 2, 3])
    border = lw + rng.randint(0, 3)
    style = style or rng.choices(STYLES, STYLE_WEIGHTS)[0]
    g = lambda v: (v, v, v)  # noqa: E731
    fill_c, half_dark = g(rng.randint(235, 255)), None
    if style == "dark":
        clue_c, ink = g(rng.randint(0, 45)), g(rng.randint(225, 255))
    elif style == "gray":
        clue_c, ink = g(rng.randint(140, 200)), g(rng.randint(0, 40))
    elif style == "inverted":
        fill_c, clue_c, ink = g(rng.randint(0, 30)), g(rng.randint(160, 200)), g(rng.randint(0, 30))
    elif style == "half":
        clue_c, ink = g(rng.randint(0, 30)), g(rng.randint(0, 30))
        half_dark = clue_c
    else:  # color
        hue = np.array([rng.uniform(0.3, 1.0) for _ in range(3)])
        hue /= hue.max()
        hue[rng.randrange(3)] = rng.uniform(0.15, 0.55)  # color saturado (contraste realista)
        fill_c = tuple(int(255 - (255 - 255 * h) * rng.uniform(0.1, 0.25)) for h in hue)
        clue_c = tuple(int(255 - (255 - 255 * h) * rng.uniform(0.45, 0.7)) for h in hue)
        ink = tuple(int(v * rng.uniform(0.15, 0.4)) for v in clue_c)
    line_c = g(rng.randint(0, 60)) if style != "inverted" else g(rng.randint(90, 150))
    diag_c = ink if style != "inverted" else g(rng.randint(60, 120))
    dashed = rng.random() < 0.2
    empty_diag = rng.random() < 0.5
    font = ImageFont.truetype(font_path or rng.choice(font_paths()), int(cell * rng.uniform(0.24, 0.36)))
    W, H = puzzle.cols * cell, puzzle.rows * cell
    paper = g(rng.randint(235, 255))

    img = Image.new("RGB", (W + 1, H + 1), fill_c)
    d = ImageDraw.Draw(img)
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            x0, y0 = c * cell, r * cell
            if (r, c) in puzzle.white:
                continue
            has_d, has_a = (r, c) in puzzle.down, (r, c) in puzzle.across
            box = [x0, y0, x0 + cell, y0 + cell]
            if half_dark is not None and (has_d or has_a):
                d.rectangle(box, fill=fill_c)
                if not has_d:   # triángulo inferior-izquierdo sin pista -> negro
                    d.polygon([(x0, y0), (x0, y0 + cell), (x0 + cell, y0 + cell)], fill=half_dark)
                if not has_a:   # triángulo superior-derecho sin pista -> negro
                    d.polygon([(x0, y0), (x0 + cell, y0), (x0 + cell, y0 + cell)], fill=half_dark)
            else:
                d.rectangle(box, fill=clue_c)
            if has_d or has_a or empty_diag:
                if dashed:
                    _dashed(d, (x0, y0), (x0 + cell, y0 + cell), diag_c, max(1, lw), max(3, cell // 14))
                else:
                    d.line([x0, y0, x0 + cell, y0 + cell], fill=diag_c, width=max(1, lw))
            for has, total, down in ((has_d, puzzle.down.get((r, c)), True), (has_a, puzzle.across.get((r, c)), False)):
                if not has:
                    continue
                t = str(total)
                bx = d.textbbox((0, 0), t, font=font)
                if down:
                    tx = x0 + cell * rng.uniform(0.06, 0.14) - bx[0]
                    ty = y0 + cell * rng.uniform(0.86, 0.93) - bx[3]
                else:
                    tx = x0 + cell * rng.uniform(0.86, 0.94) - bx[2]
                    ty = y0 + cell * rng.uniform(0.07, 0.14) - bx[1]
                d.text((tx, ty), t, fill=ink, font=font)
    for k in range(puzzle.rows + 1):
        d.line([0, k * cell, W, k * cell], fill=line_c, width=lw)
    for k in range(puzzle.cols + 1):
        d.line([k * cell, 0, k * cell, H], fill=line_c, width=lw)
    if border:
        d.rectangle([0, 0, W, H], outline=line_c if style != "inverted" else g(0), width=border)
    arr = np.array(img)
    if rng.random() < 0.15:  # sombra desplazada (estilo ilustración)
        off = max(3, cell // 8)
        sh = np.full((arr.shape[0] + off, arr.shape[1] + off, 3), paper, np.uint8)
        sh[off:, off:] = rng.randint(0, 40)
        sh[:arr.shape[0], :arr.shape[1]] = arr
        arr = sh
    if rng.random() < 0.2:  # captura recortada justo al tablero (sin margen)
        return arr
    margin = rng.randint(int(cell * 0.3), int(cell * 1.5))
    return cv2.copyMakeBorder(arr, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=paper)


def photo(img: np.ndarray, rng: random.Random, *, tint=(1.0, 1.0, 1.0), pad: float = 0.25,
          bg_color=(120, 110, 100), persp: float = 0.0, rotation: float = 0.0,
          gradient: float = 0.0, grad_dir=(1.0, 0.5), brightness: float = 1.0, blur: float = 0.0,
          noise: float = 3.0, scale: float = 1.0, jpeg: int = 90) -> np.ndarray:
    """Simula una foto con parámetros explícitos (ver `augment` para valores aleatorios).

    persp    desplazamiento máximo de cada esquina (fracción del lado) -> ángulo de cámara
    rotation grados de rotación en el plano
    gradient intensidad de la iluminación no uniforme (0 = uniforme)
    """
    nprng = np.random.default_rng(rng.randrange(2**32))
    h, w = img.shape[:2]
    img = np.clip(img * np.array(tint), 0, 255).astype(np.uint8)

    p = int(max(h, w) * pad)
    canvas_size = (w + 2 * p, h + 2 * p)
    bg = np.full((canvas_size[1], canvas_size[0], 3), bg_color, np.uint8)
    tex = nprng.normal(0, 8, bg.shape)
    bg = np.clip(bg + cv2.GaussianBlur(tex, (0, 0), 3), 0, 255).astype(np.uint8)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = persp * max(h, w)
    dst = src + p + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    ang = np.deg2rad(rotation)
    cx, cy = canvas_size[0] / 2, canvas_size[1] / 2
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = ((dst - [cx, cy]) @ R.T + [cx, cy]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, canvas_size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, canvas_size)
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)

    yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]].astype(np.float32)
    gx, gy = grad_dir
    grad = 1 + gradient * ((xx / out.shape[1] - 0.5) * gx + (yy / out.shape[0] - 0.5) * gy)
    out = np.clip(out * grad[..., None] * brightness, 0, 255)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out = np.clip(out + nprng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def augment(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Foto simulada con parámetros aleatorios (domain randomization)."""
    return photo(
        img, rng,
        tint=(rng.uniform(0.85, 1.0), rng.uniform(0.88, 1.0), rng.uniform(0.85, 1.0)),
        pad=rng.uniform(0.15, 0.4),
        bg_color=[rng.randint(20, 200) for _ in range(3)],
        persp=0.08 * strength,
        rotation=rng.uniform(-12, 12) * strength,
        gradient=strength * rng.uniform(0.1, 0.35),
        grad_dir=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
        brightness=rng.uniform(0.75, 1.1),
        blur=rng.choice([0, 0, 0.7, 1.0, 1.4]) * strength,
        noise=rng.uniform(1, 8) * strength,
        scale=rng.uniform(0.7, 1.0),
        jpeg=rng.randint(35, 95),
    )


def make_sample(rng: random.Random, augmented: bool = True, puzzle: KakuroPuzzle | None = None):
    """Devuelve (imagen BGR, puzzle) listo para el pipeline."""
    puzzle = puzzle or random_puzzle(rng)
    img = cv2.cvtColor(render_board(puzzle, rng), cv2.COLOR_RGB2BGR)
    if augmented:
        img = augment(img, rng)
    return img, puzzle
