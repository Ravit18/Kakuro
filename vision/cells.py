"""Fase 1 - Paso 3: clasificación de celdas y segmentación de dígitos.

Tipos de celda:
    WHITE  celda blanca (variable)
    BLACK  celda negra sin pista
    CLUE   celda negra con al menos una pista

Una celda pista se divide por su diagonal (arriba-izq -> abajo-der):
    triángulo inferior-izquierdo  -> suma vertical   (down)
    triángulo superior-derecho    -> suma horizontal (across)
En cada triángulo se buscan componentes conexas de "tinta" (los dígitos), se
ordenan de izquierda a derecha y se normalizan a 28x28 (estilo MNIST) para el
clasificador.

La polaridad de la tinta (texto blanco sobre negro o negro sobre gris) se
decide por celda, por lo que funciona con distintos estilos de impresión.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.grid import Grid

WHITE, BLACK, CLUE = "white", "black", "clue"
DIGIT_SIZE = 28


@dataclass
class CellInfo:
    kind: str
    down_digits: list[np.ndarray] = field(default_factory=list)    # crops 28x28
    across_digits: list[np.ndarray] = field(default_factory=list)
    down_mask: np.ndarray | None = None     # tinta del triángulo (para Tesseract)
    across_mask: np.ndarray | None = None


def _otsu_threshold(values: np.ndarray) -> float:
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def crop_cell(gray: np.ndarray, grid: Grid, r: int, c: int, inset: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = grid.cell_box(r, c)
    dx, dy = int((x1 - x0) * inset), int((y1 - y0) * inset)
    return gray[y0 + dy: y1 - dy, x0 + dx: x1 - dx]


def _cell_medians(gray: np.ndarray, grid: Grid) -> np.ndarray:
    return np.array([[np.median(crop_cell(gray, grid, r, c, 0.15)) for c in range(grid.cols)]
                     for r in range(grid.rows)])


def white_threshold(means: np.ndarray) -> float:
    """Umbral de intensidad entre celdas claras y oscuras (Otsu + punto medio)."""
    if means.max() - means.min() < 12:  # todas iguales: no hay contraste suficiente
        return 127.0
    t = _otsu_threshold(means)
    lo_cls, hi_cls = means[means <= t], means[means > t]
    if len(lo_cls) and len(hi_cls):
        t = (lo_cls.mean() + hi_cls.mean()) / 2  # punto medio entre ambas clases
    return float(t)


def has_content(cell: np.ndarray) -> bool:
    """¿La celda contiene trazos (dígitos o diagonal) sobre su fondo, de cualquier color?"""
    h, w = cell.shape
    inner = cell[int(h * 0.14): int(h * 0.86), int(w * 0.14): int(w * 0.86)].astype(np.float32)
    bg = float(np.median(inner))
    return (np.abs(inner - bg) > 50).mean() > 0.015


def classify_cells(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Devuelve una matriz booleana: True = celda a rellenar ("blanca").

    Las celdas se agrupan en dos colores (claro / oscuro). Para saber cuál es el
    color "a rellenar" se usa una regla estructural del Kakuro: la primera fila y
    la primera columna nunca son celdas a rellenar, así que el color de sus celdas
    lisas (sin dígitos ni diagonal) es el de las celdas pista/negras. Funciona en
    el estilo clásico (blancas a rellenar) y en el invertido de algunas apps
    (negras a rellenar, pistas en gris). Si no hay celdas lisas en el borde, se
    usa el grupo que casi nunca tiene contenido.
    """
    means = _cell_medians(gray, grid)
    light = means > white_threshold(means)
    content = np.array([[has_content(crop_cell(gray, grid, r, c)) for c in range(grid.cols)]
                        for r in range(grid.rows)])
    edge = np.zeros_like(content)
    edge[0, :] = edge[:, 0] = True
    plain_edge = edge & ~content
    if plain_edge.sum() >= 2 and abs(light[plain_edge].mean() - 0.5) > 0.2:
        fill = ~light if light[plain_edge].mean() > 0.5 else light
    else:
        rate = lambda m: content[m].mean() if m.any() else 1.0  # noqa: E731
        fill = light if rate(light) <= rate(~light) else ~light
    return fill & ~content


def _ink_percentile(cell: np.ndarray) -> np.ndarray | None:
    """Respaldo para celdas de bajo contraste.

    El fondo es la mediana de la celda. La tinta es el lado (más claro o más
    oscuro) con el extremo más alejado del fondo (percentil 1 / 99), así se
    soportan texto blanco sobre negro y texto negro sobre gris. El umbral es el
    punto medio entre fondo y extremo.
    """
    blur = cv2.GaussianBlur(cell, (3, 3), 0).astype(np.float32)
    h, w = blur.shape
    inner = blur[int(h * 0.08): int(h * 0.92), int(w * 0.08): int(w * 0.92)]  # sin las líneas del borde
    bg = float(np.median(inner))
    light, dark = float(np.percentile(inner, 99.5)) - bg, bg - float(np.percentile(inner, 0.5))
    contrast = max(light, dark)
    if contrast < 25:
        return None
    ink = blur > bg + contrast / 2 if light >= dark else blur < bg - contrast / 2
    return ink.astype(np.uint8) * 255


def ink_mask(cell: np.ndarray) -> np.ndarray | None:
    """Máscara de tinta (255) dentro de una celda oscura, o None si está vacía.

    1) Otsu dentro de la celda: la tinta es la clase minoritaria.
    2) Si el contraste resulta bajo (foto oscura, texto gris), se usa el método
       por percentiles (`_ink_percentile`).
    """
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    bg = float(np.median(blur))
    t = _otsu_threshold(blur)
    hi, lo = blur > t, blur <= t
    ink = hi if hi.sum() < lo.sum() else lo
    if ink.sum() and (~ink).sum():
        contrast = abs(float(blur[ink].mean()) - float(blur[~ink].mean()))
        if contrast >= 40 and abs(bg - t) >= 15:
            return ink.astype(np.uint8) * 255
    return _ink_percentile(cell)


def _components(mask: np.ndarray, cell_h: int) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.10 * cell_h or area < 0.004 * cell_h * cell_h:
            continue
        if h > 0.6 * cell_h or w > 0.6 * cell_h:  # restos de líneas/bordes
            continue
        if w > 0.12 * cell_h and h > 0.12 * cell_h and area < 0.22 * w * h:
            continue  # trazo fino y oblicuo: fragmento de la diagonal, no un dígito
        boxes.append([x, y, x + w, y + h])
    # Unir fragmentos del mismo dígito (solapamiento horizontal grande).
    boxes.sort(key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - max(m[0], b[0])
            if overlap > 0.5 * min(m[2] - m[0], b[2] - b[0]):
                merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                continue
        merged.append(b)
    # Separar dígitos pegados (caja demasiado ancha para un solo dígito).
    out = []
    for x0, y0, x1, y1 in merged:
        w, h = x1 - x0, y1 - y0
        if w > 1.0 * h:
            cols = mask[y0:y1, x0:x1].sum(axis=0)
            a, b = int(w * 0.3), int(w * 0.7)
            cut = x0 + a + int(np.argmin(cols[a:b]))
            out += [(x0, y0, cut, y1), (cut, y0, x1, y1)]
        else:
            out.append((x0, y0, x1, y1))
    return out


def normalize_digit(mask: np.ndarray) -> np.ndarray:
    """Recorta la tinta, la centra y la escala a 28x28 (dígito en 20x20)."""
    ys, xs = np.nonzero(mask > 40)
    if len(xs) == 0:
        return np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8)
    d = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = d.shape
    s = 20.0 / max(h, w)
    d = cv2.resize(d, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8)
    y0, x0 = (DIGIT_SIZE - d.shape[0]) // 2, (DIGIT_SIZE - d.shape[1]) // 2
    out[y0: y0 + d.shape[0], x0: x0 + d.shape[1]] = d
    return out


def remove_diagonal(ink: np.ndarray) -> np.ndarray:
    """Borra la línea diagonal: componentes que cruzan gran parte de la celda.

    No basta con una banda fija alrededor de la diagonal teórica, porque en fotos
    la línea real queda algo desplazada; se elimina la componente completa.
    """
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = ink.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw > 0.45 * w and bh > 0.45 * h:
            out[labels == i] = 0
    return out


def soft_ink(cell: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Intensidad de tinta en escala de grises (0 = fondo, 255 = tinta plena).

    Conserva más información que la máscara binaria cuando la foto está borrosa.
    """
    blur = cv2.GaussianBlur(cell, (3, 3), 0).astype(np.float32)
    bg = float(np.median(blur[ink == 0]))
    fg = float(np.median(blur[ink > 0]))
    soft = (blur - bg) / (fg - bg + 1e-6)
    return (np.clip(soft, 0, 1) * 255).astype(np.uint8)


def split_clue(cell: np.ndarray, ink: np.ndarray, soft: np.ndarray | None = None):
    """Separa los dos triángulos y devuelve (crops_down, crops_across, mask_down, mask_across)."""
    h, w = ink.shape
    soft = soft_ink(cell, ink) if soft is None else soft
    yy, xx = np.mgrid[0:h, 0:w]
    # Distancia (normalizada) a la diagonal y = x·h/w.
    dist = (yy - xx * h / w) / h
    band = 0.075 + 2.0 / h
    border = (yy < 0.035 * h) | (yy > 0.965 * h) | (xx < 0.035 * w) | (xx > 0.965 * w)
    clean = ink.copy()
    clean[border] = 0
    # Quitar segmentos rectos largos (líneas de la grilla): ningún dígito tiene un
    # trazo horizontal o vertical de más de ~media celda.
    k = max(5, int(0.45 * min(h, w)))
    lines = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    lines |= cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    clean[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 0
    clean = remove_diagonal(clean)
    clean[np.abs(dist) < band] = 0
    down = np.where(dist > 0, clean, 0).astype(np.uint8)
    across = np.where(dist < 0, clean, 0).astype(np.uint8)

    def digits(mask):
        crops = []
        for x0, y0, x1, y1 in _components(mask, h):
            # Se recorta la intensidad suave, pero sólo donde hay tinta del dígito (dilatada).
            m = cv2.dilate(mask[y0:y1, x0:x1], np.ones((3, 3), np.uint8))
            crops.append(normalize_digit(np.where(m > 0, soft[y0:y1, x0:x1], 0).astype(np.uint8)))
        return crops

    return digits(down), digits(across), down, across


def _tri_geometry(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = (yy - xx * h / w) / h       # > 0: triángulo inferior-izq (vertical)
    band = 0.075 + 2.0 / h
    border = (yy < 0.035 * h) | (yy > 0.965 * h) | (xx < 0.035 * w) | (xx > 0.965 * w)
    return dist, band, border


def triangle_ink(cell: np.ndarray, t_white: float):
    """Tinta por triángulo, cada uno con su propio fondo y polaridad.

    Cubre el estilo en que la pista está escrita en negro sobre un triángulo
    blanco y el otro triángulo es negro (o la celda entera es blanca con diagonal).
    Devuelve (máscara de tinta, intensidad suave) o None si un triángulo no es claro.
    """
    h, w = cell.shape
    dist, band, border = _tri_geometry(h, w)
    blur = cv2.GaussianBlur(cell, (3, 3), 0).astype(np.float32)
    ink = np.zeros((h, w), np.uint8)
    soft = np.zeros((h, w), np.float32)
    any_bright = False
    for region in (dist > band, dist < -band):
        region = region & ~border
        vals = blur[region]
        if vals.size == 0:
            continue
        bg = float(np.median(vals))
        if bg > t_white:
            any_bright = True
        light = float(np.percentile(vals, 99.5)) - bg
        dark = bg - float(np.percentile(vals, 0.5))
        contrast = max(light, dark)
        if contrast < 40:
            continue  # triángulo sin texto
        sign = 1.0 if light >= dark else -1.0
        s_ = np.clip(sign * (blur - bg) / contrast, 0, 1)
        ink[region & (s_ > 0.5)] = 255
        soft[region] = s_[region]
    if not any_bright:
        return None
    return ink, (soft * 255).astype(np.uint8)


def analyze_cells(gray: np.ndarray, grid: Grid) -> list[list[CellInfo]]:
    white = classify_cells(gray, grid)
    t_white = white_threshold(_cell_medians(gray, grid))
    out = []
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            if white[r, c]:
                row.append(CellInfo(WHITE))
                continue
            cell = crop_cell(gray, grid, r, c, 0.0)
            tri = triangle_ink(cell, t_white)
            if tri is not None:          # algún triángulo claro: polaridad por triángulo
                ink, soft = tri
            else:                        # celda oscura (estilo clásico)
                ink, soft = ink_mask(cell), None
            if ink is None or not ink.any():
                row.append(CellInfo(BLACK))
                continue
            dd, aa, dm, am = split_clue(cell, ink, soft)
            kind = CLUE if (dd or aa) else BLACK
            row.append(CellInfo(kind, dd, aa, dm, am))
        out.append(row)
    return out


def draw_cells(color: np.ndarray, grid: Grid, cells: list[list[CellInfo]]) -> np.ndarray:
    out = color.copy()
    overlay = out.copy()
    colors = {WHITE: (80, 200, 80), BLACK: (60, 60, 60), CLUE: (40, 140, 255)}
    for r in range(grid.rows):
        for c in range(grid.cols):
            x0, y0, x1, y1 = grid.cell_box(r, c)
            cv2.rectangle(overlay, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), colors[cells[r][c].kind], -1)
    return cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
