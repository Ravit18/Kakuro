"""Dibuja la solución sobre la imagen ORIGINAL (realidad aumentada simple).

Los dígitos se dibujan en el espacio rectificado (celdas alineadas) y luego se
proyectan a la foto original con la homografía inversa.
"""
from __future__ import annotations

import cv2
import numpy as np


def draw_solution(original: np.ndarray, H: np.ndarray, warped_shape: tuple[int, int],
                  lines: tuple[np.ndarray, np.ndarray], solution: dict,
                  color=(0, 170, 0)) -> np.ndarray:
    ys, xs = lines
    h, w = warped_shape[:2]
    layer = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for (r, c), v in solution.items():
        x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
        size = min(x1 - x0, y1 - y0)
        scale = size / 45.0
        thick = max(1, int(size / 18))
        (tw, th), _ = cv2.getTextSize(str(v), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int((x0 + x1 - tw) / 2), int((y0 + y1 + th) / 2))
        cv2.putText(layer, str(v), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        cv2.putText(mask, str(v), org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    Hinv = np.linalg.inv(H)
    size = (original.shape[1], original.shape[0])
    layer_o = cv2.warpPerspective(layer, Hinv, size)
    mask_o = cv2.warpPerspective(mask, Hinv, size).astype(np.float32)[..., None] / 255.0
    out = original.astype(np.float32) * (1 - mask_o) + layer_o.astype(np.float32) * mask_o
    return out.astype(np.uint8)
