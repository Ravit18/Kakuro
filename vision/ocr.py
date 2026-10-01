"""Fase 1 - Paso 4: reconocimiento de dígitos (OCR).

Dos motores intercambiables:
  * "cnn"        Red convolucional propia (PyTorch), entrenada con
                 scripts/train_digits.py sobre dígitos extraídos de tableros
                 sintéticos renderizados con varias fuentes y aumentación
                 fotográfica. Arquitectura tipo LeNet-5 [LeCun et al., 1998].
  * "tesseract"  Tesseract OCR [Smith, 2007] preentrenado, modo línea (--psm 7)
                 y lista blanca 0-9. Sirve como línea base de comparación.

Ambos devuelven, para cada triángulo de pista, el número leído y una confianza.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "digits_cnn.pt"


@dataclass
class Reading:
    value: int | None
    confidence: float
    alternatives: list[tuple[int, float]]  # (valor, prob) ordenadas, para reparación


# --------------------------------------------------------------------- CNN
def build_net():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 7x7
        nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 10),
    )


class CNNReader:
    name = "cnn"

    def __init__(self, path: Path = MODEL_PATH):
        import torch

        if not Path(path).exists():
            raise FileNotFoundError(
                f"no existe el modelo {path}. Entrénalo con: python -m scripts.train_digits")
        self.torch = torch
        self.net = build_net()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        self.net.eval()

    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        x = np.stack(crops).astype(np.float32)[:, None] / 255.0
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(x))
            return self.torch.softmax(logits, 1).numpy()

    def read(self, crops: list[np.ndarray], mask: np.ndarray | None = None) -> Reading:
        if not crops:
            return Reading(None, 1.0, [])
        if len(crops) > 2:  # una suma de Kakuro tiene a lo sumo 2 dígitos (máx. 45)
            keep = sorted(sorted(range(len(crops)), key=lambda i: -int(crops[i].sum()))[:2])
            crops = [crops[i] for i in keep]  # se conserva el orden izquierda -> derecha
        p = self.probs(crops)
        # Enumerar las lecturas más probables (top-3 por dígito) para reparación posterior.
        tops = [np.argsort(-pi)[:3] for pi in p]
        alts: list[tuple[int, float]] = []
        if len(crops) == 1:
            alts = [(int(d), float(p[0][d])) for d in tops[0]]
        else:
            alts = [(int(a) * 10 + int(b), float(p[0][a] * p[1][b])) for a in tops[0] for b in tops[1]]
        alts = sorted({v: q for v, q in alts if v > 0}.items(), key=lambda t: -t[1])
        if not alts:
            return Reading(None, 0.0, [])
        return Reading(alts[0][0], alts[0][1], alts)


# --------------------------------------------------------------- Tesseract
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


class TesseractReader:
    name = "tesseract"

    def __init__(self):
        if not tesseract_available():
            raise RuntimeError("Tesseract no está instalado o no está en el PATH")

    def read(self, crops: list[np.ndarray], mask: np.ndarray | None = None) -> Reading:
        if not crops or mask is None:
            return Reading(None, 1.0, [])
        ys, xs = np.nonzero(mask)
        img = 255 - mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]  # negro sobre blanco
        h = img.shape[0]
        img = cv2.resize(img, None, fx=48 / h, fy=48 / h, interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "clue.png"
            cv2.imwrite(str(f), img)
            res = subprocess.run(
                ["tesseract", str(f), "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789"],
                capture_output=True, text=True)
        txt = "".join(ch for ch in res.stdout if ch.isdigit())
        if not txt:
            return Reading(None, 0.0, [])
        v = int(txt[:2])
        return Reading(v, 0.9, [(v, 0.9)])


def get_reader(engine: str = "auto"):
    if engine == "cnn":
        return CNNReader()
    if engine == "tesseract":
        return TesseractReader()
    if MODEL_PATH.exists():
        return CNNReader()
    return TesseractReader()
