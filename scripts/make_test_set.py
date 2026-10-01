"""Construye el dataset de prueba (requisito: >= 10 imágenes en distintas condiciones).

Genera en data/test_set/:
  * digital/   versiones digitales (render limpio, captura escalada, estilo gris).
  * simulated/ fotos simuladas con UNA condición controlada cada una
               (iluminación baja, sombra, reflejo, ángulo, rotación, desenfoque,
               fondo oscuro, JPEG fuerte), para medir robustez por condición.
  * printables/kakuro_para_imprimir.pdf + JSON: tableros para imprimir y
               fotografiar. Las fotos reales van en data/test_set/real/ con el
               mismo nombre que su JSON (ver README).
Se busca solución única (CP-SAT); si no se encuentra rápido, se usa un tablero válido.

Uso:
    python -m scripts.make_test_set
    python -m scripts.benchmark vision --dir data/test_set/digital
    python -m scripts.benchmark vision --dir data/test_set/simulated
    python -m scripts.benchmark vision --dir data/test_set/real
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
from PIL import Image, JpegImagePlugin  # noqa: F401  (necesario para guardar PDF)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.synth import FONT_DIR, photo, render_board  # noqa: E402
from scripts.generate_puzzles import PATTERNS, generate, puzzle_from_solution, random_fill  # noqa: E402


def make_puzzle(pattern: str, rng: random.Random):
    """Tablero con solución única si se encuentra rápido; si no, uno válido cualquiera."""
    res = generate(PATTERNS[pattern], rng, max_tries=40)
    if res:
        return res[0], res[1]
    fill = random_fill(PATTERNS[pattern], rng)
    return puzzle_from_solution(PATTERNS[pattern], fill), fill

FONT = str(FONT_DIR / "LiberationSans-Bold.ttf")

# nombre -> (patrón, estilo, parámetros de photo() o None si es digital)
DIGITAL = {
    "d01_digital_limpio_5x5": ("5x5", "dark", None),
    "d02_digital_limpio_10x10": ("10x10", "dark", None),
    "d03_digital_gris_8x8": ("8x8", "gray", None),
    "d04_captura_reducida_7x7": ("7x7", "dark", {"scale": 0.55, "jpeg": 70, "noise": 0}),
    "d05_app_invertido_8x8": ("8x8", "inverted", None),
    "d06_triangulo_blanco_7x7": ("7x7", "half", None),
    "d07_color_10x10": ("10x10", "color", None),
}
SIMULATED = {
    "s01_luz_baja": ("8x8", "dark", {"brightness": 0.45, "noise": 9, "jpeg": 75}),
    "s02_sombra_lateral": ("7x7", "dark", {"gradient": 1.1, "grad_dir": (1.0, 0.2)}),
    "s03_reflejo_brillante": ("10x10", "gray", {"brightness": 1.25, "gradient": 0.5, "grad_dir": (-0.6, -1)}),
    "s04_angulo_leve": ("8x8", "dark", {"persp": 0.06}),
    "s05_angulo_fuerte": ("7x7", "dark", {"persp": 0.11}),
    "s06_rotado_10": ("10x10", "dark", {"rotation": 10, "persp": 0.02}),
    "s07_desenfoque": ("8x8", "dark", {"blur": 1.6, "noise": 4}),
    "s08_fondo_oscuro": ("7x7", "gray", {"bg_color": (25, 25, 30), "pad": 0.5, "persp": 0.04}),
    "s09_jpeg_fuerte": ("10x10", "dark", {"jpeg": 25, "scale": 0.8}),
    "s10_luz_calida": ("5x5", "dark", {"tint": (0.75, 0.9, 1.0), "gradient": 0.4, "persp": 0.05}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "test_set")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--printables", type=int, default=8, help="tableros para imprimir")
    args = ap.parse_args()

    for sub in ("digital", "simulated", "printables", "real"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    k = 0
    for folder, spec in (("digital", DIGITAL), ("simulated", SIMULATED)):
        for name, (pat, style, params) in spec.items():
            k += 1
            rng = random.Random(args.seed + k)
            puzzle, fill = make_puzzle(pat, rng)
            img = cv2.cvtColor(render_board(puzzle, rng, cell=rng.randint(60, 80), style=style,
                                            font_path=FONT if k % 2 else None), cv2.COLOR_RGB2BGR)
            if params is not None:
                img = photo(img, rng, **params)
            cv2.imwrite(str(args.out / folder / f"{name}.png"), img)
            sol = [[fill.get((r, c), 0) for c in range(puzzle.cols)] for r in range(puzzle.rows)]
            puzzle.save(args.out / folder / f"{name}.json", solution=sol, condition=name.split("_", 1)[1])
            print(f"{folder}/{name}.png  ({pat}, {style})")

    # Tableros para imprimir (uno por página A4, con su nombre)
    pages = []
    pdir = args.out / "printables"
    pats = list(PATTERNS)
    for i in range(args.printables):
        rng = random.Random(args.seed + 100 + i)
        pat = pats[i % len(pats)]
        puzzle, fill = make_puzzle(pat, rng)
        name = f"p{i + 1:02d}_impreso_{pat}"
        board = render_board(puzzle, rng, cell=110, style="dark" if i % 3 else "gray", font_path=FONT)
        page = Image.new("RGB", (2480, 3508), "white")  # A4 a 300 dpi
        b = Image.fromarray(board)
        s = min(2000 / b.width, 2600 / b.height)
        b = b.resize((int(b.width * s), int(b.height * s)), Image.LANCZOS)
        page.paste(b, ((2480 - b.width) // 2, 450))
        from PIL import ImageDraw, ImageFont

        ImageDraw.Draw(page).text((200, 200), name, fill="black", font=ImageFont.truetype(FONT, 70))
        pages.append(page)
        page.save(pdir / f"{name}.png")
        sol = [[fill.get((r, c), 0) for c in range(puzzle.cols)] for r in range(puzzle.rows)]
        puzzle.save(pdir / f"{name}.json", solution=sol)
    pages[0].save(pdir / "kakuro_para_imprimir.pdf", save_all=True, append_images=pages[1:], resolution=300)
    print(f"{pdir / 'kakuro_para_imprimir.pdf'}  ({len(pages)} páginas)")
    (args.out / "real" / "LEEME.txt").write_text(
        "Coloca aquí las fotos de los tableros impresos.\n"
        "Nombra cada foto como el tablero + la condición, p. ej. p01_impreso_5x5__luz_natural.jpg\n"
        "y copia el JSON correspondiente de ../printables con el MISMO nombre que la foto\n"
        "(p01_impreso_5x5__luz_natural.json). Sugerencias de condiciones: luz natural, lámpara,\n"
        "sombra parcial, ángulo leve, rotada, desde lejos, foto de la pantalla del celular.\n"
        "Evalúa con:  python -m scripts.benchmark vision --dir data/test_set/real\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
