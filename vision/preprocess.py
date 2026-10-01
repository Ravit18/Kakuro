"""Fase 1 - Paso 1: preprocesamiento y corrección de perspectiva.

Flujo:
    imagen BGR -> escala de grises -> desenfoque -> binarización adaptativa
    -> componente conexa más grande (la grilla) -> 4 esquinas
    -> transformación de perspectiva (vista cenital del tablero).

La grilla se localiza como la componente conexa con más píxeles de "tinta"
(líneas + bordes de las celdas negras), no como el contorno más grande: así no
se confunde con el borde de la hoja cuando la foto se toma sobre una mesa.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_SIDE = 1600      # se reduce la imagen de entrada a este tamaño máximo
WARP_SIDE = 900      # lado mayor del tablero rectificado


@dataclass
class Rectified:
    gray: np.ndarray          # tablero rectificado (escala de grises)
    color: np.ndarray         # tablero rectificado (BGR)
    H: np.ndarray             # homografía imagen_original -> rectificada
    corners: np.ndarray       # 4 esquinas en la imagen original (tl, tr, br, bl)
    scale: float              # factor aplicado al redimensionar la entrada
    debug: dict               # imágenes intermedias (para el informe)
    at_edge: bool = False     # el tablero llega al borde de la imagen


def load_image(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"no se pudo leer la imagen {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def contrast_gray(bgr: np.ndarray) -> np.ndarray:
    """Escala de grises de máximo contraste (primera componente principal del color).

    La conversión estándar a grises puede dejar casi iguales dos colores distintos
    (p. ej. amarillo claro y amarillo oscuro). Se proyecta cada píxel sobre la
    dirección de color con mayor varianza del tablero, orientada como la
    luminancia, y se estira a 0-255. Para tableros en blanco y negro equivale a
    la escala de grises usual.
    """
    if bgr.ndim == 2:
        return bgr
    X = bgr.reshape(-1, 3).astype(np.float32)
    sample = X[:: max(1, len(X) // 50000)]
    mu = sample.mean(0)
    _, vecs = np.linalg.eigh(np.cov((sample - mu).T))
    v = vecs[:, -1]
    lum = np.array([0.114, 0.587, 0.299], np.float32)  # BGR
    if v @ lum < 0:
        v = -v
    g = (X - mu) @ v
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
    g = np.clip((g - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    return g.reshape(bgr.shape[:2]).astype(np.uint8)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarización adaptativa invertida: tinta (oscuro) = 255."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(11, (min(gray.shape) // 30) | 1)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 7)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como (arriba-izq, arriba-der, abajo-der, abajo-izq)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def _quad_from_points(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(points)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def find_board(binary: np.ndarray) -> np.ndarray:
    """Devuelve las 4 esquinas del tablero en la imagen binaria."""
    # Cerrar pequeñas discontinuidades de las líneas de la grilla.
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        raise ValueError("no se encontró ninguna grilla en la imagen")
    h, w = binary.shape
    best, best_score = None, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.15 * w or bh < 0.15 * h:
            continue
        # Penaliza componentes que tocan el borde de la imagen (sombras, mesa).
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        score = area * (0.3 if touches else 1.0)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        raise ValueError("no se encontró ninguna grilla suficientemente grande")
    ys, xs = np.nonzero(labels == best)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    return _quad_from_points(pts)


def rectify(img: np.ndarray, corners: np.ndarray, side: int = WARP_SIDE):
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    k = side / max(width, height)
    W, Hh = int(round(width * k)), int(round(height * k))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (W, Hh), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE), H


def full_frame(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def preprocess(img: np.ndarray) -> Rectified:
    scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = to_gray(img)
    binary = binarize(gray)
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Si el tablero llega al borde de la imagen (captura digital recortada justo al
    # tablero), su contorno no es detectable: se usa la imagen completa.
    try:
        corners = find_board(binary)
        h, w = gray.shape
        m = 0.02 * max(h, w)
        at_edge = sum(x < m or y < m or x > w - 1 - m or y > h - 1 - m for x, y in corners)
        full = at_edge >= 2
    except ValueError:
        full = True
    if full:
        corners = full_frame(gray.shape)
    color_w, H = rectify(bgr, corners)
    gray_w = contrast_gray(color_w)

    contour = bgr.copy()
    cv2.polylines(contour, [corners.astype(np.int32)], True, (0, 0, 255), 3)
    for p in corners:
        cv2.circle(contour, tuple(int(v) for v in p), 8, (0, 200, 0), -1)

    # Homografía respecto a la imagen ORIGINAL (antes de redimensionar).
    S = np.diag([scale, scale, 1.0])
    return Rectified(gray_w, color_w, H @ S, corners / scale, scale,
                     {"1_gray": gray, "2_binary": binary, "3_board": contour, "4_warped": color_w}, full)
