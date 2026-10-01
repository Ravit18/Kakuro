"""Modelo CP de Kakuro con OR-Tools CP-SAT (restricciones globales).

Modelo formal
-------------
Variables:    x_c  para cada celda blanca c.
Dominios:     D(x_c) = {1..9}  ∩  cand(run_across(c))  ∩  cand(run_down(c))
              (cand = dígitos que aparecen en alguna combinación válida de la suma).
Restricciones, para cada corrida R con suma objetivo s_R:
    AllDifferent({x_c : c ∈ R})       (global)
    Σ_{c ∈ R} x_c = s_R               (global / lineal)

No se necesitan restricciones reificadas: todas las reglas del Kakuro se
expresan directamente como AllDifferent + suma lineal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from core.puzzle import Cell, KakuroPuzzle, Run
from solver.combos import candidates


@dataclass
class SolveResult:
    status: str
    solution: dict[Cell, int] | None
    num_solutions: int | None = None  # sólo si se pidió comprobar unicidad
    stats: dict = field(default_factory=dict)


def cell_domains(puzzle: KakuroPuzzle, runs: list[Run], prune: bool = True) -> dict[Cell, list[int]]:
    """Dominio de cada celda; con `prune` se intersecta con los candidatos de sus corridas."""
    domains = {c: set(range(1, 10)) for c in puzzle.white}
    if prune:
        for run in runs:
            cand = candidates(run.total, len(run.cells))
            for c in run.cells:
                domains[c] &= cand
    return {c: sorted(d) for c, d in domains.items()}


def build_model(puzzle: KakuroPuzzle, prune: bool = True):
    """Construye el modelo CP-SAT. Devuelve (model, variables, runs)."""
    runs = puzzle.validate()
    model = cp_model.CpModel()

    x = {}
    for cell, dom in cell_domains(puzzle, runs, prune).items():
        r, c = cell
        x[cell] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(dom), f"x_{r}_{c}")

    for run in runs:
        vars_ = [x[c] for c in run.cells]
        model.AddAllDifferent(vars_)
        model.Add(sum(vars_) == run.total)

    return model, x, runs


class _Counter(cp_model.CpSolverSolutionCallback):
    """Cuenta soluciones y se detiene al llegar a `limit`."""

    def __init__(self, x, limit: int):
        super().__init__()
        self._x, self._limit = x, limit
        self.count, self.first = 0, None

    def on_solution_callback(self):
        self.count += 1
        if self.first is None:
            self.first = {c: self.Value(v) for c, v in self._x.items()}
        if self.count >= self._limit:
            self.StopSearch()


def solve(
    puzzle: KakuroPuzzle,
    prune: bool = True,
    check_unique: bool = False,
    time_limit: float = 30.0,
    workers: int = 8,
) -> SolveResult:
    t0 = time.perf_counter()
    model, x, runs = build_model(puzzle, prune)
    t_build = time.perf_counter() - t0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    if check_unique:
        # Enumerar todas las soluciones requiere un solo worker en CP-SAT.
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.num_workers = 1
        counter = _Counter(x, limit=2)
        status = solver.Solve(model, counter)
        solution, n = counter.first, counter.count
    else:
        solver.parameters.num_workers = workers
        status = solver.Solve(model)
        ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        solution = {c: solver.Value(v) for c, v in x.items()} if ok else None
        n = None

    stats = {
        "variables": len(x),
        "runs": len(runs),
        "build_s": t_build,
        "solve_s": solver.WallTime(),
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
    }
    return SolveResult(solver.StatusName(status), solution, n, stats)


def check_solution(puzzle: KakuroPuzzle, solution: dict[Cell, int]) -> bool:
    """Verificador independiente del solver (para tests y para la Fase 3)."""
    if set(solution) != set(puzzle.white):
        return False
    for run in puzzle.runs():
        values = [solution[c] for c in run.cells]
        if sum(values) != run.total or len(set(values)) != len(values):
            return False
        if not all(1 <= v <= 9 for v in values):
            return False
    return True
