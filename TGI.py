#!/usr/bin/env python3
"""Tumor-growth-inhibition (TGI) dose-amplitude benchmark.

This script keeps the OptiDose short-window dose representation fixed while
comparing analytic, continuous-sensitivity, automatic-differentiation, finite-
difference, and derivative-free optimization routes.

Quick smoke test::

    python TGI.py --preset quick --output output

Manuscript comparators::

    python TGI.py --preset paper --methods all --output output

Model and benchmark source: Bachmann et al., 2021,
https://doi.org/10.1007/s10957-021-01819-w
"""

from __future__ import annotations
from dataclasses import dataclass

import jax
from jax import config as jax_config
import jax.numpy as jnp
import diffrax as dx

jax_config.update("jax_enable_x64", True)


# =========================
#  Parameters and schedule
# =========================

@jax.tree_util.register_pytree_node_class
@dataclass
class TumorParams:
    V: float
    ka: float
    kel: float
    lam0: float
    lam1: float
    kt: float
    kpot: float
    P0: float
    alpha: float      # regularisation weight
    eps: float        # width for dose smoothing
    T: float          # final time
    t_cost_start: float  # time when cost starts (12)
    t_doses: jnp.ndarray  # shape (m,) for daily doses at 12..28

    def tree_flatten(self):
        children = (
            self.V,
            self.ka,
            self.kel,
            self.lam0,
            self.lam1,
            self.kt,
            self.kpot,
            self.P0,
            self.alpha,
            self.eps,
            self.T,
            self.t_cost_start,
            self.t_doses,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def make_params():
    """Create parameter object with OptiDose tumor settings."""
    # daily doses from day 12 to 28 inclusive -> 17 doses
    t_doses = jnp.arange(12.0, 29.0)  # [12,13,...,28] shape (17,)

    return TumorParams(
        V=2.79,
        ka=5.0,
        kel=2.53,
        lam0=0.194,
        lam1=0.246,
        kt=0.666,
        kpot=0.0077,
        P0=0.0098,
        alpha=1e-7,
        eps=0.1,
        T=30.0,
        t_cost_start=12.0,
        t_doses=t_doses,
    )


# =========================
#  Reference trajectory
# =========================

def W_ref(t, params: TumorParams):
    """Reference total tumor burden W_ref(t)."""
    num = 0.25 * (jnp.exp(2.0) - jnp.exp(-2.0))
    den = 0.5 * (jnp.exp(2.0) - 3.0 * jnp.exp(-2.0)) + jnp.exp(0.5 * t - 8.0)
    return num / den


# =========================
#  Regularised oral doses
# =========================

def dose_profile(t, u, params: TumorParams):
    """
    Regularised oral input into Abs compartment:
        In(t,u) = sum_i u_i/eps * 1_[t_i, t_i+eps](t)
    """
    t_i = params.t_doses  # (m,)
    in_pulse = (t >= t_i) & (t < t_i + params.eps)  # (m,)
    in_pulse = in_pulse.astype(jnp.float64)
    return jnp.sum(u * in_pulse) / params.eps


# =========================
#  Tumor ODE (state equation)
# =========================

def tumor_rhs(t, y, args):
    """
    y = [Abs, C, P, D1, D2, D3].
    """
    params, u = args
    Abs, C, P, D1, D2, D3 = y

    In = dose_profile(t, u, params)

    dAbs = In - params.ka * Abs
    dC = params.ka / params.V * Abs - params.kel * C

    W = P + D1 + D2 + D3
    f = 2.0 * params.lam0 * params.lam1 * P**2
    g = params.lam1 + 2.0 * params.lam0 * P
    # Avoid division by zero with tiny epsilon; W should be >0 in practice
    H = g * W + 1e-16
    Ap = f / H

    dP = Ap - params.kpot * C * P
    dD1 = params.kpot * C * P - params.kt * D1
    dD2 = params.kt * (D1 - D2)
    dD3 = params.kt * (D2 - D3)

    return jnp.stack([dAbs, dC, dP, dD1, dD2, dD3])


def initial_state(params: TumorParams):
    Abs0 = 0.0
    C0 = 0.0
    P0 = params.P0
    D10 = 0.0
    D20 = 0.0
    D30 = 0.0
    return jnp.array([Abs0, C0, P0, D10, D20, D30])


def forward_solve(u, params: TumorParams, N_t: int = 600):
    """Forward solve used for analytic adjoint and FD."""
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(tumor_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=500_000,
    )

    return ts, sol.ys


def forward_solve_backsolve(u, params: TumorParams, N_t: int = 600):
    """Forward solve configured for reverse-mode AD (BacksolveAdjoint)."""
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(tumor_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=500_000,
        adjoint=dx.BacksolveAdjoint(),  # reverse-mode compatible
    )

    return ts, sol.ys


def forward_solve_forwardmode(u, params: TumorParams, N_t: int = 600):
    """Forward solve configured for forward-mode AD (ForwardMode adjoint)."""
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(tumor_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=500_000,
        adjoint=dx.ForwardMode(),
    )

    return ts, sol.ys


# =========================
#  Utilities, cost J(u)
# =========================

def _trapz(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Simple trapezoidal rule without jnp.trapz."""
    dx = x[1:] - x[:-1]
    y0 = y[:-1]
    y1 = y[1:]
    return jnp.sum(0.5 * (y0 + y1) * dx)


def J_from_trajectory(ts, ys, params: TumorParams):
    # ys: (N_t, 6)
    Abs, C, P, D1, D2, D3 = ys.T
    W = P + D1 + D2 + D3
    Wref = W_ref(ts, params)

    mask = (ts >= params.t_cost_start).astype(ts.dtype)
    diff = (W - Wref) * mask
    integrand = 0.5 * diff**2
    return _trapz(integrand, ts)


def J_reduced(u, params: TumorParams):
    ts, ys = forward_solve(u, params)
    return J_from_trajectory(ts, ys, params) + params.alpha * jnp.sum(u)


def J_reduced_backsolve(u, params: TumorParams):
    ts, ys = forward_solve_backsolve(u, params)
    return J_from_trajectory(ts, ys, params) + params.alpha * jnp.sum(u)


def J_reduced_forwardmode(u, params: TumorParams):
    ts, ys = forward_solve_forwardmode(u, params)
    return J_from_trajectory(ts, ys, params) + params.alpha * jnp.sum(u)


# =========================
#  Jacobian d g / d y
# =========================

def dg_dy(t, y, params: TumorParams):
    Abs, C, P, D1, D2, D3 = y

    W = P + D1 + D2 + D3
    f = 2.0 * params.lam0 * params.lam1 * P**2
    g = params.lam1 + 2.0 * params.lam0 * P
    H = g * W + 1e-16

    f_prime = 4.0 * params.lam0 * params.lam1 * P
    H_prime_P = 2.0 * params.lam0 * W + g
    dAp_dP = (f_prime * H - f * H_prime_P) / (H**2)

    # dAp/dD1, dAp/dD2, dAp/dD3:
    # = -f * g / H^2 = - f / (g * W^2)
    dAp_dD = -f * g / (H**2)

    J = jnp.zeros((6, 6), dtype=jnp.float64)

    # Abs equation
    J = J.at[0, 0].set(-params.ka)

    # C equation
    J = J.at[1, 0].set(params.ka / params.V)
    J = J.at[1, 1].set(-params.kel)

    # P equation
    J = J.at[2, 1].set(-params.kpot * P)          # d/dC
    J = J.at[2, 2].set(dAp_dP - params.kpot * C)  # d/dP
    J = J.at[2, 3].set(dAp_dD)                    # d/dD1
    J = J.at[2, 4].set(dAp_dD)                    # d/dD2
    J = J.at[2, 5].set(dAp_dD)                    # d/dD3

    # D1 equation
    J = J.at[3, 1].set(params.kpot * P)           # d/dC
    J = J.at[3, 2].set(params.kpot * C)           # d/dP
    J = J.at[3, 3].set(-params.kt)                # d/dD1

    # D2 equation
    J = J.at[4, 3].set(params.kt)                 # d/dD1
    J = J.at[4, 4].set(-params.kt)                # d/dD2

    # D3 equation
    J = J.at[5, 4].set(params.kt)                 # d/dD2
    J = J.at[5, 5].set(-params.kt)                # d/dD3

    return J


# =========================
#  Adjoint equation
# =========================

def make_y_interp(ts, ys):
    return dx.LinearInterpolation(ts=ts, ys=ys)


def adjoint_rhs(t, p, args):
    params, y_interp = args
    y = y_interp.evaluate(t)  # (6,)
    Abs, C, P, D1, D2, D3 = y

    Jg = dg_dy(t, y, params)         # (6,6)
    W = P + D1 + D2 + D3
    Wref = W_ref(t, params)

    mask = jnp.where(t >= params.t_cost_start, 1.0, 0.0)
    diff = (W - Wref) * mask
    dh_dy = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    obs_term = diff * dh_dy

    return - Jg.T @ p + obs_term


def solve_adjoint(ts, ys, params):
    """
    Solve adjoint backward in time from T to 0, sampling at the same ts grid.
    Returns p_ts aligned with ts (ascending).
    """
    y_interp = make_y_interp(ts, ys)
    solver = dx.Tsit5()
    t0 = params.T
    t1 = 0.0
    pT = jnp.zeros(6, dtype=jnp.float64)

    sol = dx.diffeqsolve(
        dx.ODETerm(adjoint_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=-0.01,
        y0=pT,
        args=(params, y_interp),
        saveat=dx.SaveAt(ts=ts[::-1]),
        max_steps=50_000,
    )

    p_backwards = sol.ys
    p_ts = p_backwards[::-1]
    return p_ts


# =========================
#  Analytic gradient in u
# =========================

def grad_component_for_i(i, ts, pAbs, params: TumorParams):
    eps = params.eps
    ti = params.t_doses[i]  # scalar
    in_pulse = (ts >= ti) & (ts < ti + eps)
    mask = in_pulse.astype(ts.dtype)

    integral = _trapz(pAbs * mask, ts)
    # alpha term + adjoint contribution
    return params.alpha - integral / eps


def grad_reduced(u, params: TumorParams):
    ts, ys = forward_solve(u, params)
    p_ts = solve_adjoint(ts, ys, params)
    pAbs = p_ts[:, 0]  # Abs component of adjoint

    m = params.t_doses.shape[0]
    grad_components = jax.vmap(
        lambda i: grad_component_for_i(i, ts, pAbs, params)
    )(jnp.arange(m))

    return grad_components


def J_and_grad(u, params: TumorParams):
    ts, ys = forward_solve(u, params)
    J_val = J_from_trajectory(ts, ys, params) + params.alpha * jnp.sum(u)
    p_ts = solve_adjoint(ts, ys, params)
    pAbs = p_ts[:, 0]

    m = params.t_doses.shape[0]
    grad_u = jax.vmap(
        lambda i: grad_component_for_i(i, ts, pAbs, params)
    )(jnp.arange(m))

    return J_val, grad_u


# =========================
#  AD gradient & FD gradient
# =========================

def grad_backsolve(u, params: TumorParams):
    """Gradient via reverse-mode AD through diffrax (BacksolveAdjoint)."""
    return jax.grad(J_reduced_backsolve, argnums=0)(u, params)


def grad_forwardmode(u, params: TumorParams):
    """Gradient via forward-mode AD (JVP) through ForwardMode adjoint."""
    return jax.jacfwd(J_reduced_forwardmode, argnums=0)(u, params)


# =========================
# Optimization and output
# =========================

def _parse_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Optimize the OptiDose tumor-growth-inhibition benchmark with DiffDose."
    )
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--methods", choices=("diffdose", "all"), default="diffdose")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--timing", choices=("warm", "cold"), default="warm")
    return parser.parse_args()


def run_analysis(*, preset="quick", methods="diffdose", output=None, timing="warm"):
    """Run the TGI optimization benchmark and return method-level results."""
    from pathlib import Path

    import numpy as np

    from utils.OptimizationUtils import (
        bound_aware_finite_difference,
        run_lbfgsb,
        run_nelder_mead,
        run_random_walk,
        save_results,
    )

    params = make_params()
    n_controls = int(params.t_doses.shape[0])
    initial_controls = np.zeros(n_controls, dtype=float)
    bounds = tuple((0.0, 4000.0) for _ in range(n_controls))
    max_iterations = 2 if preset == "quick" else 40
    fd_iterations = 1 if preset == "quick" else 30

    def loss(controls):
        return float(J_reduced(jnp.asarray(controls), params))

    def finite_difference_calls(controls):
        at_bound = sum(
            np.isclose(value, lower) or np.isclose(value, upper)
            for value, (lower, upper) in zip(controls, bounds)
        )
        return (1 if at_bound else 0) + at_bound + 2 * (n_controls - at_bound)

    routes = [
        ("Forward AD", lambda controls: np.asarray(grad_forwardmode(jnp.asarray(controls), params)), 1, max_iterations),
    ]
    if methods == "all":
        routes = [
            ("OptiDose", lambda controls: np.asarray(grad_reduced(jnp.asarray(controls), params)), 1, max_iterations),
            ("Adjoint Sensitivity", lambda controls: np.asarray(grad_backsolve(jnp.asarray(controls), params)), 1, max_iterations),
            ("Forward AD", lambda controls: np.asarray(grad_forwardmode(jnp.asarray(controls), params)), 1, max_iterations),
            ("Reverse AD", lambda controls: np.asarray(jax.grad(J_reduced, argnums=0)(jnp.asarray(controls), params)), 1, max_iterations),
            (
                "Finite Difference",
                lambda controls: bound_aware_finite_difference(
                    loss, controls, bounds, relative_step=1e-4
                ),
                finite_difference_calls,
                fd_iterations,
            ),
        ]

    results = [
        run_lbfgsb(
            method=name,
            loss_fn=loss,
            gradient_fn=gradient,
            initial_controls=initial_controls,
            bounds=bounds,
            max_iterations=iterations,
            timing=timing,
            objective_scale=100.0,
            forward_solves_per_gradient=solve_count,
        )
        for name, gradient, solve_count, iterations in routes
    ]
    if methods == "all":
        budget = max(result.elapsed_seconds for result in results)
        results.extend(
            [
                run_nelder_mead(
                    loss_fn=loss,
                    initial_controls=initial_controls,
                    bounds=bounds,
                    budget_seconds=budget,
                ),
                run_random_walk(
                    loss_fn=loss,
                    initial_controls=initial_controls,
                    bounds=bounds,
                    budget_seconds=budget,
                    step_scale=2000.0,
                ),
            ]
        )
    if output is not None:
        save_results(results, Path(output), "TGI")
    return results


def main():
    args = _parse_args()
    results = run_analysis(
        preset=args.preset,
        methods=args.methods,
        output=args.output,
        timing=args.timing,
    )
    for result in results:
        print(
            f"{result.method}: L={result.objective:.8g}, "
            f"controls={result.controls.tolist()}, status={result.status}"
        )


if __name__ == "__main__":
    main()
