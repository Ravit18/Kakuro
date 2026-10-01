"""Entrena la CNN de dígitos (vision/models/digits_cnn.pt).

Datos:
  * Sintéticos: se renderizan tableros aleatorios con aumentación fotográfica
    (render/synth.py) y se pasan por EL MISMO pipeline de visión que se usa en
    inferencia (preprocesamiento -> grilla -> celdas -> recorte de dígitos).
    Como la etiqueta de cada pista se conoce, cada recorte queda etiquetado sin
    trabajo manual. Así el modelo aprende exactamente la distribución de
    recortes que verá después.
  * Reales (opcional): fotos propias en --real-dir con un JSON del mismo nombre
    que contenga "grid" (ver README). Sus recortes se añaden al entrenamiento.

Uso:
    python -m scripts.train_digits                       # ~2000 tableros, 12 épocas
    python -m scripts.train_digits --puzzles 4000 --epochs 20
    python -m scripts.train_digits --real-dir data/images/train
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.puzzle import KakuroPuzzle  # noqa: E402
from render.synth import make_sample  # noqa: E402
from vision import cells as C  # noqa: E402
from vision.grid import detect_grid  # noqa: E402
from vision.ocr import MODEL_PATH, build_net  # noqa: E402
from vision.preprocess import load_image, preprocess  # noqa: E402


def crops_from_image(img, puzzle: KakuroPuzzle):
    """Recortes etiquetados de una imagen con tablero conocido."""
    try:
        rect = preprocess(img)
        grid = detect_grid(rect.gray, puzzle.rows, puzzle.cols)
        info = C.analyze_cells(rect.gray, grid)
    except Exception:
        return [], []
    X, y = [], []
    for side, clues in (("down", puzzle.down), ("across", puzzle.across)):
        for (r, c), total in clues.items():
            crops = getattr(info[r][c], f"{side}_digits")
            label = str(total)
            if len(crops) != len(label):
                continue  # segmentación dudosa: no se usa para entrenar
            X += crops
            y += [int(ch) for ch in label]
    return X, y


def _worker(seed: int):
    rng = random.Random(seed)
    img, puzzle = make_sample(rng, augmented=rng.random() < 0.85)
    return crops_from_image(img, puzzle)


def build_dataset(n_puzzles: int, seed: int, workers: int):
    X, y = [], []
    with Pool(workers) as pool:
        for i, (xs, ys) in enumerate(pool.imap_unordered(_worker, range(seed, seed + n_puzzles), 8)):
            X += xs
            y += ys
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{n_puzzles} tableros, {len(y)} dígitos")
    return np.array(X, np.uint8), np.array(y, np.int64)


def real_dataset(folder: Path):
    X, y = [], []
    for js in sorted(folder.glob("*.json")):
        imgs = [p for p in folder.glob(js.stem + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not imgs:
            continue
        puzzle = KakuroPuzzle.from_grid(json.loads(js.read_text(encoding="utf-8"))["grid"])
        xs, ys = crops_from_image(load_image(imgs[0]), puzzle)
        X += xs
        y += ys
    return np.array(X, np.uint8).reshape(-1, 28, 28), np.array(y, np.int64)


def train(X, y, epochs: int, out: Path, seed: int = 0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).permutation(len(y))
    n_val = max(1, int(0.1 * len(y)))
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X.astype(np.float32)[:, None] / 255.0)
    yt = torch.from_numpy(y)

    net = build_net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) // 128 + 1))

    def jitter(xb):
        # Aumentación en línea: pequeña traslación/escala/rotación.
        n = xb.shape[0]
        ang = (torch.rand(n) - 0.5) * 0.2
        sc = 1 + (torch.rand(n) - 0.5) * 0.15
        th = torch.zeros(n, 2, 3)
        th[:, 0, 0] = sc * torch.cos(ang)
        th[:, 0, 1] = -sc * torch.sin(ang)
        th[:, 1, 0] = sc * torch.sin(ang)
        th[:, 1, 1] = sc * torch.cos(ang)
        th[:, :, 2] = (torch.rand(n, 2) - 0.5) * 0.15
        grid = F.affine_grid(th, xb.shape, align_corners=False)
        return F.grid_sample(xb, grid, align_corners=False)

    best = 0.0
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr[np.random.default_rng(seed + ep).permutation(len(tr))]
        t0, loss_sum = time.time(), 0.0
        for k in range(0, len(perm), 128):
            b = perm[k: k + 128]
            xb, yb = jitter(Xt[b]), yt[b]
            loss = nn.functional.cross_entropy(net(xb), yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pred = net(Xt[va]).argmax(1).numpy()
        acc = float((pred == y[va]).mean())
        print(f"época {ep:2d}  loss={loss_sum / len(tr):.4f}  val_acc={acc:.4f}  ({time.time() - t0:.1f}s)")
        if acc >= best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "val_acc": acc, "n_train": len(tr)}, out)

    # matriz de confusión en validación
    net.load_state_dict(torch.load(out, weights_only=True)["state_dict"])
    net.eval()
    with torch.no_grad():
        pred = net(Xt[va]).argmax(1).numpy()
    cm = np.zeros((10, 10), int)
    for t, p in zip(y[va], pred):
        cm[t, p] += 1
    print("\nMatriz de confusión (fila = real, columna = predicho):")
    print("     " + " ".join(f"{d:4d}" for d in range(10)))
    for d in range(10):
        print(f"{d:4d} " + " ".join(f"{v:4d}" for v in cm[d]))
    print(f"\nMejor val_acc = {best:.4f}  ->  {out}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", type=int, default=2000, help="tableros sintéticos a generar")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--real-dir", type=Path, help="carpeta con fotos reales etiquetadas (opcional)")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "digits" / "digits.npz")
    ap.add_argument("--regen", action="store_true", help="regenerar aunque exista la caché")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        d = np.load(args.cache)
        X, y = d["X"], d["y"]
        print(f"Dataset cargado de {args.cache}: {len(y)} dígitos")
    else:
        print(f"Generando dataset sintético con {args.puzzles} tableros...")
        X, y = build_dataset(args.puzzles, args.seed, args.workers)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, X=X, y=y)
        print(f"Guardado en {args.cache}: {len(y)} dígitos")

    if args.real_dir:
        Xr, yr = real_dataset(args.real_dir)
        print(f"Recortes reales: {len(yr)} (se repiten x5 para darles más peso)")
        if len(yr):
            X = np.concatenate([X] + [Xr] * 5)
            y = np.concatenate([y] + [yr] * 5)

    print("Distribución por clase:", np.bincount(y, minlength=10).tolist())
    train(X, y, args.epochs, args.out, args.seed)


if __name__ == "__main__":
    main()
