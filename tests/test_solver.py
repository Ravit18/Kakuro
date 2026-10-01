import random

import pytest

from core.puzzle import KakuroPuzzle, PuzzleError
from scripts.generate_puzzles import PATTERNS, generate
from solver import cp_model, cp_table
from solver.combos import candidates, combos

# 2x2 mínimo con solución única: [[1, 3], [2, 1]]
TINY = [
    ["#", "3\\", "4\\"],
    ["\\4", ".", "."],
    ["\\3", ".", "."],
]


def test_combos():
    assert combos(3, 2) == (frozenset({1, 2}),)
    assert candidates(17, 2) == {8, 9}
    assert candidates(45, 9) == set(range(1, 10))


def test_roundtrip_grid():
    p = KakuroPuzzle.from_grid(TINY)
    assert p.to_grid() == TINY


def test_tiny_unique():
    p = KakuroPuzzle.from_grid(TINY)
    res = cp_model.solve(p, check_unique=True)
    assert res.num_solutions == 1
    assert res.solution == {(1, 1): 1, (1, 2): 3, (2, 1): 2, (2, 2): 1}


def test_invalid_sum_detected():
    bad = [row[:] for row in TINY]
    bad[1][0] = "\\40"  # suma imposible para 2 celdas (típico error de OCR)
    with pytest.raises(PuzzleError):
        KakuroPuzzle.from_grid(bad).validate()


@pytest.mark.parametrize("name", ["5x5", "7x7"])
def test_generated_both_models(name):
    puzzle, fill, _ = generate(PATTERNS[name], random.Random(42))
    for solve in (cp_model.solve, cp_table.solve):
        res = solve(puzzle)
        assert res.solution == fill
        assert cp_model.check_solution(puzzle, res.solution)
