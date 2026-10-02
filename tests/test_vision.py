import random

import numpy as np
import pytest

from core.puzzle import KakuroPuzzle
from render.synth import make_sample, random_puzzle, render_board
from vision import cells as C
from vision.grid import detect_grid
from vision.ocr import MODEL_PATH
from vision.preprocess import order_corners, preprocess

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="falta entrenar la CNN")


def test_order_corners():
    pts = np.array([[10, 90], [90, 90], [10, 10], [90, 10]])
    tl, tr, br, bl = order_corners(pts)
    assert tuple(tl) == (10, 10) and tuple(br) == (90, 90)
    assert tuple(tr) == (90, 10) and tuple(bl) == (10, 90)


def test_random_puzzle_is_valid():
    p = random_puzzle(random.Random(3))
    p.validate()  # no lanza excepción


@pytest.mark.parametrize("seed", range(5))
def test_grid_and_cells_clean(seed):
    img, puzzle = make_sample(random.Random(seed), augmented=False)
    rect = preprocess(img)
    grid = detect_grid(rect.gray)
    assert (grid.rows, grid.cols) == (puzzle.rows, puzzle.cols)
    white = C.classify_cells(rect.gray, grid)
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            assert white[r, c] == ((r, c) in puzzle.white)


@pytest.mark.parametrize("seed", range(5))
def test_grid_augmented(seed):
    img, puzzle = make_sample(random.Random(100 + seed), augmented=True)
    grid = detect_grid(preprocess(img).gray)
    assert (grid.rows, grid.cols) == (puzzle.rows, puzzle.cols)


@needs_model
@pytest.mark.parametrize("seed", range(5))
def test_end_to_end_clean(seed):
    from vision.pipeline import extract

    img, puzzle = make_sample(random.Random(200 + seed), augmented=False)
    ex = extract(img, engine="cnn")
    assert ex.data["grid"] == puzzle.to_grid()


@needs_model
def test_json_contract():
    from vision.pipeline import extract

    rng = random.Random(7)
    puzzle = random_puzzle(rng)
    import cv2

    img = cv2.cvtColor(render_board(puzzle, rng), cv2.COLOR_RGB2BGR)
    data = extract(img, engine="cnn").data
    assert {"rows", "cols", "grid"} <= set(data)
    KakuroPuzzle.from_grid(data["grid"]).validate()


@pytest.mark.parametrize("style", ["dark", "gray", "inverted", "half", "color"])
def test_grid_and_cells_all_styles(style):
    import cv2

    rng = random.Random(42)
    puzzle = random_puzzle(rng, 6, 9)
    img = cv2.cvtColor(render_board(puzzle, rng, cell=70, style=style), cv2.COLOR_RGB2BGR)
    rect = preprocess(img)
    grid = detect_grid(rect.gray)
    assert (grid.rows, grid.cols) == (puzzle.rows, puzzle.cols)
    white = C.classify_cells(rect.gray, grid)
    assert {(r, c) for r in range(grid.rows) for c in range(grid.cols) if white[r, c]} == set(puzzle.white)


@pytest.mark.parametrize("deg", [0, 30, 44, 45, 46, 60, 135])
def test_order_corners_rotated(deg):
    """Con el tablero muy girado en la foto (~45°) las 4 esquinas no deben repetirse."""
    a = np.deg2rad(deg)
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * 100
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (base @ rot.T + 500).astype(np.float32)[[2, 0, 3, 1]]   # desordenadas
    q = order_corners(pts)
    assert len({tuple(p) for p in q.round(3)}) == 4
    x, y = q[:, 0], q[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0   # sentido horario


@pytest.mark.parametrize("seed", range(3))
def test_light_cells_with_shadow(seed):
    """Una franja de sombra (mano/celular) no convierte celdas blancas en negras."""
    rng = np.random.default_rng(seed)
    truth = rng.random((9, 9)) < 0.6
    truth[0, :] = truth[:, 0] = False
    yy, xx = np.mgrid[0:9, 0:9]
    shade = 1 - 0.65 * np.exp(-((xx - yy) / 2.0) ** 2)
    means = np.where(truth, 215.0, 35.0) * shade
    assert (C.light_cells(means) == truth).all()
    assert not ((means > C.white_threshold(means)) == truth).all()   # el umbral global sí falla


@needs_model
@pytest.mark.parametrize("rotation", [None, "ROTATE_90_CLOCKWISE", "ROTATE_180"])
def test_extract_rotated(rotation):
    """Foto girada 90° o volteada 180°: se prueba cada orientación y se lee igual."""
    import cv2
    from vision.pipeline import extract

    rng = random.Random(7)
    puzzle = random_puzzle(rng)
    img = cv2.cvtColor(render_board(puzzle, rng), cv2.COLOR_RGB2BGR)
    if rotation:
        img = cv2.rotate(img, getattr(cv2, rotation))
    assert extract(img, engine="cnn").data["grid"] == puzzle.to_grid()
