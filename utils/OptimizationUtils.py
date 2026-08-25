#!/usr/bin/env python3
"""Generic optimizer, finite-difference, and CSV utilities for DiffDose."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class BenchmarkResult:
    """Optimization result and convergence history for one method."""

    method: str
    optimizer: str
    status: str
    history: pd.DataFrame
    controls: np.ndarray
    objective: float
    iterations: int
    evaluations: int
    elapsed_seconds: float
    forward_solves: int
    gradient_calls: int


def _vector(values: Sequence[float] | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def _history_row(
    method: str,
    iteration: int,
    evaluation: int,
    objective: float,
    best_objective: float,
    elapsed_seconds: float,
    controls: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "method": method,
        "iteration": iteration,
        "evaluation": evaluation,
        "objective": objective,
        "best_objective": best_objective,
        "elapsed_seconds": elapsed_seconds,
    }
    row.update({f"u_{index}": float(value) for index, value in enumerate(controls)})
    return row


def bound_aware_finite_difference(
    loss_fn: Callable[[np.ndarray], float],
    controls: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    relative_step: float = 1e-4,
) -> np.ndarray:
    """Use central differences internally and one-sided differences at bounds."""
    controls = _vector(controls)
    gradient = np.zeros_like(controls)
    base_value: float | None = None
    for index, value in enumerate(controls):
        lower, upper = bounds[index]
        step = relative_step * max(1.0, abs(value))
        can_decrease = value - step >= lower
        can_increase = value + step <= upper
        if can_decrease and can_increase:
            plus = controls.copy()
            minus = controls.copy()
            plus[index] += step
            minus[index] -= step
            gradient[index] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * step)
        else:
            if base_value is None:
                base_value = float(loss_fn(controls))
            shifted = controls.copy()
            if can_increase:
                shifted[index] += step
                gradient[index] = (loss_fn(shifted) - base_value) / step
            elif can_decrease:
                shifted[index] -= step
                gradient[index] = (base_value - loss_fn(shifted)) / step
            else:
                raise ValueError(f"No finite-difference step is valid for control {index}.")
    return gradient


def scalar_bound_aware_finite_difference(
    loss_fn: Callable[[float], float],
    value: float,
    bounds: tuple[float, float],
    *,
    relative_step: float = 1e-4,
) -> float:
    gradient = bound_aware_finite_difference(
        lambda vector: loss_fn(float(vector[0])),
        [value],
        [bounds],
        relative_step=relative_step,
    )
    return float(gradient[0])


def run_lbfgsb(
    *,
    method: str,
    loss_fn: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    initial_controls: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    max_iterations: int,
    timing: str = "warm",
    objective_scale: float = 1.0,
    forward_solves_per_gradient: int | Callable[[np.ndarray], int] = 1,
) -> BenchmarkResult:
    """Run L-BFGS-B and record the unscaled objective at accepted iterates."""
    controls0 = _vector(initial_controls)
    bounds = tuple(bounds)
    if timing == "warm":
        float(loss_fn(controls0.copy()))
        np.asarray(gradient_fn(controls0.copy()), dtype=float)

    evaluations = 0
    gradient_calls = 0
    forward_solves = 0
    history: list[dict[str, float | int | str]] = []
    cache_x: np.ndarray | None = None
    cache_f = np.nan
    cache_g_x: np.ndarray | None = None
    cache_g: np.ndarray | None = None
    start = time.perf_counter()

    def loss(controls: np.ndarray) -> float:
        nonlocal cache_x, cache_f, evaluations, forward_solves
        vector = _vector(controls)
        if cache_x is None or not np.array_equal(cache_x, vector):
            cache_f = float(loss_fn(vector.copy()))
            cache_x = vector.copy()
            evaluations += 1
            forward_solves += 1
        return cache_f

    def gradient(controls: np.ndarray) -> np.ndarray:
        nonlocal cache_g_x, cache_g, gradient_calls, forward_solves
        vector = _vector(controls)
        if cache_g_x is None or not np.array_equal(cache_g_x, vector):
            cache_g = _vector(gradient_fn(vector.copy()))
            cache_g_x = vector.copy()
            gradient_calls += 1
            count = (
                forward_solves_per_gradient(vector)
                if callable(forward_solves_per_gradient)
                else forward_solves_per_gradient
            )
            forward_solves += int(count)
        return np.asarray(cache_g, dtype=float)

    initial_value = loss(controls0)
    best_value = initial_value
    history.append(
        _history_row(method, 0, evaluations, initial_value, best_value, time.perf_counter() - start, controls0)
    )
    accepted = 0

    def callback(controls: np.ndarray) -> None:
        nonlocal accepted, best_value
        accepted += 1
        value = loss(controls)
        gradient(controls)
        best_value = min(best_value, value)
        history.append(
            _history_row(
                method,
                accepted,
                evaluations,
                value,
                best_value,
                time.perf_counter() - start,
                _vector(controls),
            )
        )

    result = optimize.minimize(
        lambda controls: objective_scale * loss(controls),
        controls0,
        jac=lambda controls: objective_scale * gradient(controls),
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-9},
    )
    final_controls = _vector(result.x)
    final_value = loss(final_controls)
    status = "converged" if result.success else "maxiter_reached" if result.nit >= max_iterations else "failed"
    return BenchmarkResult(
        method=method,
        optimizer="L-BFGS-B",
        status=status,
        history=pd.DataFrame(history),
        controls=final_controls,
        objective=final_value,
        iterations=int(result.nit),
        evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - start,
        forward_solves=forward_solves,
        gradient_calls=gradient_calls,
    )


class _BudgetExpired(RuntimeError):
    pass


def run_nelder_mead(
    *,
    loss_fn: Callable[[np.ndarray], float],
    initial_controls: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    budget_seconds: float,
) -> BenchmarkResult:
    controls0 = _vector(initial_controls)
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    history: list[dict[str, float | int | str]] = []
    best_controls = controls0.copy()
    best_value = np.inf
    evaluations = 0
    start = time.perf_counter()

    def objective(controls: np.ndarray) -> float:
        nonlocal best_controls, best_value, evaluations
        if evaluations and time.perf_counter() - start >= budget_seconds:
            raise _BudgetExpired
        controls = np.clip(_vector(controls), lower, upper)
        value = float(loss_fn(controls))
        evaluations += 1
        if value < best_value:
            best_value = value
            best_controls = controls.copy()
        history.append(
            _history_row(
                "Nelder-Mead",
                evaluations - 1,
                evaluations,
                value,
                best_value,
                time.perf_counter() - start,
                controls,
            )
        )
        return value

    status = "budget_exhausted"
    try:
        result = optimize.minimize(
            objective,
            controls0,
            method="Nelder-Mead",
            options={"maxiter": 1_000_000, "xatol": 1e-10, "fatol": 1e-12},
        )
        status = "converged" if result.success else "failed"
    except _BudgetExpired:
        pass
    return BenchmarkResult(
        "Nelder-Mead",
        "Nelder-Mead",
        status,
        pd.DataFrame(history),
        best_controls,
        float(best_value),
        max(evaluations - 1, 0),
        evaluations,
        time.perf_counter() - start,
        evaluations,
        0,
    )


def run_random_walk(
    *,
    loss_fn: Callable[[np.ndarray], float],
    initial_controls: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    budget_seconds: float,
    step_scale: float | Sequence[float],
    seed: int = 2021,
) -> BenchmarkResult:
    controls = _vector(initial_controls)
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    scale = np.broadcast_to(np.asarray(step_scale, dtype=float), controls.shape)
    random = np.random.default_rng(seed)
    start = time.perf_counter()
    current_value = float(loss_fn(controls))
    best_value = current_value
    best_controls = controls.copy()
    evaluations = 1
    history = [
        _history_row("Random Walk", 0, 1, current_value, best_value, 0.0, controls)
    ]
    while time.perf_counter() - start < budget_seconds:
        proposal = np.clip(controls + random.normal(size=controls.shape) * scale, lower, upper)
        proposal_value = float(loss_fn(proposal))
        evaluations += 1
        if np.log(random.random()) < min(0.0, current_value - proposal_value):
            controls = proposal
            current_value = proposal_value
        if current_value < best_value:
            best_value = current_value
            best_controls = controls.copy()
        history.append(
            _history_row(
                "Random Walk",
                evaluations - 1,
                evaluations,
                current_value,
                best_value,
                time.perf_counter() - start,
                controls,
            )
        )
    return BenchmarkResult(
        "Random Walk",
        "random-walk Metropolis",
        "budget_exhausted",
        pd.DataFrame(history),
        best_controls,
        best_value,
        evaluations - 1,
        evaluations,
        time.perf_counter() - start,
        evaluations,
        0,
    )


def save_results(results: Sequence[BenchmarkResult], output: Path, model: str) -> None:
    """Write one summary, one control table, and one convergence table."""
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    controls = []
    histories = []
    for result in results:
        summaries.append(
            {
                "method": result.method,
                "optimizer": result.optimizer,
                "status": result.status,
                "objective": result.objective,
                "iterations": result.iterations,
                "evaluations": result.evaluations,
                "elapsed_seconds": result.elapsed_seconds,
                "forward_solves": result.forward_solves,
                "gradient_calls": result.gradient_calls,
            }
        )
        controls.append(
            {"method": result.method, **{f"u_{i}": value for i, value in enumerate(result.controls)}}
        )
        histories.append(result.history)
    pd.DataFrame(summaries).to_csv(output / f"{model}_summary.csv", index=False)
    pd.DataFrame(controls).to_csv(output / f"{model}_controls.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / f"{model}_convergence.csv", index=False)
