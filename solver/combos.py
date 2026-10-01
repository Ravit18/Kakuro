"""Combinaciones válidas de dígitos para una suma de Kakuro.

Una corrida de `n` celdas que suma `total` sólo puede usar ciertos conjuntos de
dígitos distintos del 1 al 9. Precalcularlos permite:
  * reducir los dominios de las variables antes de resolver, y
  * construir restricciones de tabla (ver cp_table.py).
"""
from functools import lru_cache
from itertools import combinations, permutations

DIGITS = range(1, 10)


@lru_cache(maxsize=None)
def combos(total: int, length: int) -> tuple[frozenset[int], ...]:
    """Conjuntos de `length` dígitos distintos (1-9) cuya suma es `total`."""
    return tuple(frozenset(c) for c in combinations(DIGITS, length) if sum(c) == total)


@lru_cache(maxsize=None)
def candidates(total: int, length: int) -> frozenset[int]:
    """Dígitos que aparecen en al menos una combinación válida."""
    return frozenset().union(*combos(total, length))


def tuples(total: int, length: int) -> list[tuple[int, ...]]:
    """Todas las asignaciones ordenadas válidas (permutaciones de cada combinación)."""
    return [p for s in combos(total, length) for p in permutations(sorted(s))]


def count_tuples(total: int, length: int) -> int:
    from math import factorial

    return len(combos(total, length)) * factorial(length)
