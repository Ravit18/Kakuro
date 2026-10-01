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
