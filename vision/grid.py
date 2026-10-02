"""Fase 1 - Paso 2: segmentación de la grilla (filas, columnas y celdas).

Sobre el tablero ya rectificado:
  1. Perfiles de líneas: se combinan dos evidencias, (a) trazos oscuros finos
     (binarización adaptativa) y (b) bordes de intensidad (Sobel), ambos
     filtrados con una apertura morfológica alargada que elimina dígitos y
     diagonales. Los bordes hacen que funcione también cuando las celdas se
     separan por cambio de color y no por una línea dibujada (p. ej. celdas
     negras contiguas a celdas grises).
  2. Se proyecta cada mapa (suma por filas / columnas) -> perfil con picos en
     cada línea de la grilla.
  3. Se busca conjuntamente el inicio a, el fin b y el número de celdas n que
     maximizan   contraste = media(perfil en las líneas) - media(perfil en los
     centros de celda).  Con n correcto las líneas caen en picos y los centros
     en valles; con 2n la mitad de las "líneas" caen en centros; con n/2 los
     "centros" caen en líneas. Probar varios a y b descarta márgenes, sombras y
     marcos alrededor del tablero.
  4. Cada línea se ajusta al pico local más cercano (tolera distorsión residual).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_CELLS, MAX_CELLS = 3, 30
MIN_CELL_PX = 12


@dataclass
class Grid:
    rows: int
    cols: int
    ys: np.ndarray   # rows + 1 posiciones de líneas horizontales
    xs: np.ndarray   # cols + 1 posiciones de líneas verticales
    # Posición local de cada línea (papel curvado / rectificación imperfecta):
    # ys_local[i, c] = y de la línea horizontal i a la altura de la columna c;
    # xs_local[j, r] = x de la línea vertical j a la altura de la fila r.
    ys_local: np.ndarray | None = None
    xs_local: np.ndarray | None = None

    def cell_box(self, r: int, c: int) -> tuple[int, int, int, int]:
        if self.ys_local is None or self.xs_local is None:
            return int(self.xs[c]), int(self.ys[r]), int(self.xs[c + 1]), int(self.ys[r + 1])
        return (int(self.xs_local[c, r]), int(self.ys_local[r, c]),
                int(self.xs_local[c + 1, r]), int(self.ys_local[r + 1, c]))


def _norm(p: np.ndarray) -> np.ndarray:
    p = p - np.percentile(p, 10)
    return np.clip(p / (p.max() + 1e-9), 0, 1)


def line_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perfiles (vertical py, horizontal px) con picos en las líneas de la grilla."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(11, (min(h, w) // 25) | 1)
    ink = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 5)
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    thr = max(30.0, float(np.percentile(np.maximum(gx, gy), 90)) * 0.5)
    ey = (gy > thr).astype(np.uint8) * 255
    ex = (gx > thr).astype(np.uint8) * 255

    k = max(7, min(h, w) // (MAX_CELLS + 10))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk).sum(1) + cv2.morphologyEx(ey, cv2.MORPH_OPEN, hk).sum(1)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk).sum(0) + cv2.morphologyEx(ex, cv2.MORPH_OPEN, vk).sum(0)
    smooth = lambda p: np.convolve(p.astype(np.float64), np.ones(3) / 3, mode="same")  # noqa: E731
    return _norm(smooth(horiz)), _norm(smooth(vert))


def _peaks(p: np.ndarray, min_dist: int, rel: float = 0.25) -> list[int]:
    out = []
    for i in np.argsort(-p):
        if p[i] < rel:
            break
        if all(abs(i - j) >= min_dist for j in out):
            out.append(int(i))
    return sorted(out)


def _score(p: np.ndarray, a: float, b: float, n: int, edge_lines: bool = False) -> float:
    """media(perfil en las líneas) - media(máximo del perfil en el interior de cada celda).

    Si n es demasiado pequeño, el interior de las celdas contiene líneas reales
    (máximo alto); si es demasiado grande, la mitad de las "líneas" caen en
    centros de celda (perfil bajo). Sólo el n correcto da líneas altas e
    interiores vacíos.
    """
    cell = (b - a) / n
    win = max(1, int(cell * 0.07))
    L = len(p)
    lines = [p[max(0, int(round(a + k * cell)) - win): min(L, int(round(a + k * cell)) + win + 1)].max()
             for k in range(n + 1)]
    # El borde de la imagen cuenta como línea (capturas recortadas justo al tablero).
    if edge_lines and a <= 0:
        lines[0] = 1.0
    if edge_lines and b >= L - 1:
        lines[-1] = 1.0
    inner = []
    for k in range(n):
        lo, hi = int(a + (k + 0.2) * cell), int(a + (k + 0.8) * cell)
        inner.append(p[max(0, lo): min(L, hi + 1)].max() if hi > lo else 0.0)
    return float(np.mean(lines) - np.mean(inner))


def fit_lines(p: np.ndarray, n_fixed: int | None = None, edge_lines: bool = False,
              busy: np.ndarray | None = None) -> tuple[float, float, int]:
    """Busca (inicio, fin, n) que mejor explican el perfil.

    busy (opcional, sólo si el tablero toca el borde): 1 donde la imagen tiene
    contenido. Dejar fuera del tablero una franja con contenido se penaliza:
    cuando el tablero llega al borde, no debería haber "margen" con dibujo.
    """
    L = len(p)
    pk = _peaks(p, max(3, L // 80))
    if len(pk) < 2:
        pk = [0, L - 1]
    starts = sorted(set([0] + pk[:5]))         # el tablero puede empezar en el borde
    ends = sorted(set(pk[-5:] + [L - 1]))
    best = (-1e9, 0.0, float(L - 1), MIN_CELLS)
    for a in starts:
        for b in ends:
            span = b - a
            if span < 0.4 * L:
                continue
            ns = [n_fixed] if n_fixed else range(MIN_CELLS, min(MAX_CELLS, span // MIN_CELL_PX) + 1)
            for n in ns:
                # Penaliza picos fuertes que queden fuera del tablero propuesto
                # (evita elegir sólo una parte de la grilla).
                tol = 0.35 * span / n
                outside = sum(p[q] for q in pk if q < a - tol or q > b + tol)
                s = _score(p, a, b, n, edge_lines) - 0.3 * outside
                if busy is not None:
                    cell = span / n
                    left = busy[:max(0, int(a))].sum() / cell
                    right = busy[int(b) + 1:].sum() / cell
                    s -= 0.3 * (min(1.0, left) + min(1.0, right))
                if s > best[0]:
                    best = (s, float(a), float(b), n)
    return best[1], best[2], best[3]


def refine_lines(p: np.ndarray, a: float, b: float, n: int) -> np.ndarray:
    cell = (b - a) / n
    win = max(2, int(cell * 0.18))
    pos = []
    for k in range(n + 1):
        q = int(round(a + k * cell))
        lo, hi = max(0, q - win), min(len(p), q + win + 1)
        seg = p[lo:hi]
        pos.append(lo + int(seg.argmax()) if seg.max() > 0.2 else q)
    return np.array(pos)


def active_range(gray: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ex = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)) > 40   # bordes verticales
    ey = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)) > 40   # bordes horizontales
    h, w = gray.shape

    bright_r = np.median(blur, axis=1) > 170   # filas claras (papel)
    bright_c = np.median(blur, axis=0) > 170

    def rng_(counts, n, need, bright):
        # activa = tiene bordes o no es clara (una fila de celdas negras no es margen)
        idx = np.nonzero((counts >= need) | ~bright)[0]
        if len(idx) == 0:
            return 0, n
        lo, hi = max(0, idx[0] - 3), min(n, idx[-1] + 4)
        return (lo, hi) if hi - lo > 0.4 * n else (0, n)

    return (rng_(ex.sum(1), h, max(3, 0.01 * w), bright_r),
            rng_(ey.sum(0), w, max(3, 0.01 * h), bright_c))


def detect_grid(gray: np.ndarray, rows: int | None = None, cols: int | None = None,
                edge_lines: bool = True) -> Grid:
    """Tras rectificar, el borde de la imagen suele ser el borde del tablero, por lo
    que cuenta como línea (edge_lines). Las franjas con contenido que quedarían fuera
    del tablero se penalizan."""
    # Recortar márgenes claros y lisos (papel): toda fila del tablero cruza líneas verticales
    # de la grilla y toda columna cruza líneas horizontales; una franja sin bordes
    # en el extremo de la imagen es margen, no tablero.
    full_gray = gray
    oy, ox = active_range(gray)
    gray = gray[oy[0]: oy[1], ox[0]: ox[1]]
    py, px = line_profiles(gray)
    by_ = bx_ = None
    if edge_lines:
        by_ = (gray.std(axis=1) > 30).astype(float)
        bx_ = (gray.std(axis=0) > 30).astype(float)
    ay, by, rows = fit_lines(py, rows, edge_lines, by_)
    ax, bx, cols = fit_lines(px, cols, edge_lines, bx_)
    grid = Grid(rows, cols, refine_lines(py, ay, by, rows) + oy[0], refine_lines(px, ax, bx, cols) + ox[0])
    return refine_local(full_gray, grid)


def _line_maps(gray: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mapas 2D de evidencia de líneas horizontales y verticales (tinta + bordes)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    block = max(11, (min(h, w) // 25) | 1)
    ink = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 5)
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    thr = max(30.0, float(np.percentile(np.maximum(gx, gy), 90)) * 0.5)
    ey = (gy > thr).astype(np.uint8) * 255
    ex = (gx > thr).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    hmap = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk).astype(np.float32) + cv2.morphologyEx(ey, cv2.MORPH_OPEN, hk)
    vmap = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk).astype(np.float32) + cv2.morphologyEx(ex, cv2.MORPH_OPEN, vk)
    return hmap, vmap


def refine_local(gray: np.ndarray, grid: Grid, search: float = 0.22) -> Grid:
    """Ajusta cada línea de la grilla por tramos (una posición por celda).

    Si el papel no está plano (hoja curvada o doblada) la homografía no deja las
    líneas perfectamente rectas y una sola posición por línea corta celdas por la
    mitad. Aquí, para cada línea y cada tramo de celda, se busca el pico de
    evidencia de línea cerca de la posición global (±search·celda) y luego se suaviza
    con la mediana de los tramos vecinos para ignorar dígitos o diagonales."""
    ch = (grid.ys[-1] - grid.ys[0]) / grid.rows
    cw = (grid.xs[-1] - grid.xs[0]) / grid.cols
    k = max(7, int(0.5 * min(ch, cw)))
    hmap, vmap = _line_maps(gray, k)

    def track(evidence, glob, other, n_seg, size, axis):
        out = np.zeros((len(glob), n_seg))
        win = max(2, int(search * size))
        for i, q in enumerate(glob):
            for sgm in range(n_seg):
                a, b = int(other[sgm]), int(other[sgm + 1])
                a, b = a + (b - a) // 6, b - (b - a) // 6          # centro del tramo
                lo, hi = max(0, int(q) - win), min(evidence.shape[axis], int(q) + win + 1)
                if b <= a or hi <= lo:
                    out[i, sgm] = q
                    continue
                prof = evidence[lo:hi, a:b].sum(1) if axis == 0 else evidence[a:b, lo:hi].sum(0)
                j = int(q) - lo
                at_glob = prof[max(0, j - 1): j + 2].max() if 0 <= j < len(prof) else 0.0
                strong = prof.max() > 0.35 * 255 * (b - a)
                # sólo se mueve si en la posición global casi no hay línea y cerca sí:
                # así una sombra o un dígito no "atraen" una línea que ya estaba bien
                moved = strong and prof.max() > 1.5 * at_glob + 0.1 * 255 * (b - a)
                out[i, sgm] = lo + int(np.argmax(prof)) if moved else q
            # suavizado: mediana de 3 tramos (un dígito o una diagonal no mueve la línea)
            row = out[i].copy()
            for sgm in range(n_seg):
                out[i, sgm] = np.median(row[max(0, sgm - 1): sgm + 2])
        return out

    ys_local = track(hmap, grid.ys, grid.xs, grid.cols, ch, 0)
    xs_local = track(vmap, grid.xs, grid.ys, grid.rows, cw, 1)
    return Grid(grid.rows, grid.cols, grid.ys, grid.xs, ys_local, xs_local)


def draw_grid(color: np.ndarray, grid: Grid) -> np.ndarray:
    out = color.copy()
    x0, x1 = int(grid.xs[0]), int(grid.xs[-1])
    y0, y1 = int(grid.ys[0]), int(grid.ys[-1])
    for y in grid.ys:
        cv2.line(out, (x0, int(y)), (x1, int(y)), (0, 0, 255), 2)
    for x in grid.xs:
        cv2.line(out, (int(x), y0), (int(x), y1), (255, 0, 0), 2)
    return out
