"""Variante del modelo usando restricciones de tabla (AllowedAssignments).

Cada corrida se modela como una única restricción global de tabla cuyas tuplas
son todas las permutaciones de las combinaciones válidas para su suma. Esto
captura AllDifferent + suma en una sola restricción con propagación más fuerte,
a costa de un tamaño de tabla que crece factorialmente con la longitud.

Para corridas largas (tabla > `max_tuples`) se usa AllDifferent + suma.
Sirve para comparar ambos enfoques en el informe (scripts/benchmark.py).
"""
from __future__ import annotations

import time

from ortools.sat.python import cp_model

from core.puzzle import KakuroPuzzle
from solver.combos import count_tuples, tuples
from solver.cp_model import SolveResult, cell_domains


def build_model(puzzle: KakuroPuzzle, max_tuples: int = 5000):
    runs = puzzle.validate()
    model = cp_model.CpModel()
    x = {}
    for cell, dom in cell_domains(puzzle, runs).items():
        r, c = cell
        x[cell] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(dom), f"x_{r}_{c}")

    n_table = 0
    for run in runs:
        vars_ = [x[c] for c in run.cells]
        if count_tuples(run.total, len(vars_)) <= max_tuples:
            model.AddAllowedAssignments(vars_, tuples(run.total, len(vars_)))
            n_table += 1
        else:
            model.AddAllDifferent(vars_)
            model.Add(sum(vars_) == run.total)
    return model, x, runs, n_table


def solve(puzzle: KakuroPuzzle, max_tuples: int = 5000, time_limit: float = 30.0, workers: int = 8) -> SolveResult:
    t0 = time.perf_counter()
    model, x, runs, n_table = build_model(puzzle, max_tuples)
    t_build = time.perf_counter() - t0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution = {c: solver.Value(v) for c, v in x.items()} if ok else None

    stats = {
        "variables": len(x),
        "runs": len(runs),
        "table_runs": n_table,
        "build_s": t_build,
        "solve_s": solver.WallTime(),
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
    }
    return SolveResult(solver.StatusName(status), solution, None, stats)
