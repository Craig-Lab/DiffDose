#!/usr/bin/env python3
"""Bispecific T-cell-engager (BiTE) dose-amplitude benchmark.

This script keeps the OptiDose short-window dose representation fixed while
comparing analytic, automatic-differentiation, finite-difference, and
derivative-free optimization routes.

Quick smoke test::

    python BiTE.py --preset quick --output output

Manuscript comparators::

    python BiTE.py --preset paper --methods all --output output

Model and benchmark source: Bachmann et al., 2021,
https://doi.org/10.1007/s10957-021-01819-w
"""

from __future__ import annotations
from dataclasses import dataclass

import jax
from jax import config as jax_config
import jax.numpy as jnp
import diffrax as dx

# Enable float64
jax_config.update("jax_enable_x64", True)


# =======================
# Parameters + utilities
# =======================

@jax.tree_util.register_pytree_node_class
@dataclass
class BiTEParams:
    kel: float
    kon1: float
    koff1: float
    kon2: float
    koff2: float
    kon3: float
    koff3: float
    kon4: float
    koff4: float
    ksynA: float
    kdegA: float
    ksynB: float
    kdegB: float
    kintA: float
    kintB: float
    kintAB: float
    k12: float
    k21: float
    ka: float
    V: float
    eps: float
    T: float
    dose_times: jnp.ndarray
    R_ref: float

    def tree_flatten(self):
        children = (
            self.kel, self.kon1, self.koff1, self.kon2, self.koff2,
            self.kon3, self.koff3, self.kon4, self.koff4,
            self.ksynA, self.kdegA, self.ksynB, self.kdegB,
            self.kintA, self.kintB, self.kintAB,
            self.k12, self.k21, self.ka, self.V,
            self.eps, self.T, self.dose_times, self.R_ref,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def make_params() -> BiTEParams:
    dose_times = jnp.asarray([0.0, 48.0, 96.0], dtype=jnp.float64)
    ksynA = 1.0
    kdegA = 0.1
    ksynB = 10.0
    kdegB = 0.1
    # R_ref = min(ksynA/kdegA, ksynB/kdegB) = 10
    R_ref = min(ksynA / kdegA, ksynB / kdegB)
    return BiTEParams(
        kel=0.1,
        kon1=10.0,
        koff1=0.01,
        kon2=1.0,
        koff2=0.01,
        kon3=1.0,
        koff3=0.01,
        kon4=10.0,
        koff4=0.01,
        ksynA=ksynA,
        kdegA=kdegA,
        ksynB=ksynB,
        kdegB=kdegB,
        kintA=0.05,
        kintB=0.05,
        kintAB=0.1,
        k12=0.0,
        k21=0.03,
        ka=0.2,
        V=3.0,
        eps=0.01,
        T=140.0,
        dose_times=dose_times,
        R_ref=R_ref,
    )


def _trapz(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    y0 = y[:-1]
    y1 = y[1:]
    return jnp.sum(0.5 * (y0 + y1) * dx)


# ============
# Dosing term
# ============

def dose_profile(t: float, u: float, params: BiTEParams) -> jnp.ndarray:
    """
    Regularised bolus into Abs over window of length eps at each dose_time:
        In(t,u) = u/eps * sum_l 1_[t_l, t_l+eps](t)
    Units: amount/time into Abs.
    """
    in_window = (t >= params.dose_times) & (t < params.dose_times + params.eps)
    n_active = in_window.astype(jnp.float64).sum()
    return (u / params.eps) * n_active


def dose_breakpoints(params: BiTEParams) -> jnp.ndarray:
    """Known vector-field discontinuities at infusion-window boundaries."""
    boundaries = jnp.concatenate([params.dose_times, params.dose_times + params.eps])
    return jnp.sort(boundaries[(boundaries > 0.0) & (boundaries < params.T)])


def step_controller(params: BiTEParams, rtol: float, atol: float):
    return dx.ClipStepSizeController(
        dx.PIDController(rtol=rtol, atol=atol),
        jump_ts=dose_breakpoints(params),
    )


# ============
# ODE system
# ============

def bite_rhs(t, y, args):
    """
    y = [C, RA, RB, RCA, RCB, RCAB, AP, Abs]
    """
    params, u = args
    C, RA, RB, RCA, RCB, RCAB, AP, Abs = y

    In = dose_profile(t, u, params)

    p = params
    dC   = (-p.kel * C
            - p.kon1 * C * RA + p.koff1 * RCA
            - p.kon2 * C * RB + p.koff2 * RCB
            - p.k12 * C
            + p.k21 * AP / p.V
            + p.ka * Abs / p.V)

    dRA  = (p.ksynA
            - p.kdegA * RA
            - p.kon1 * C * RA + p.koff1 * RCA
            - p.kon4 * RA * RCB + p.koff4 * RCAB)

    dRB  = (p.ksynB
            - p.kdegB * RB
            - p.kon2 * C * RB + p.koff2 * RCB
            - p.kon3 * RB * RCA + p.koff3 * RCAB)

    dRCA = (p.kon1 * C * RA
            - (p.koff1 + p.kintA) * RCA
            - p.kon3 * RB * RCA + p.koff3 * RCAB)

    dRCB = (p.kon2 * C * RB
            - (p.koff2 + p.kintB) * RCB
            - p.kon4 * RA * RCB + p.koff4 * RCAB)

    dRCAB = (p.kon4 * RA * RCB + p.kon3 * RB * RCA
             - (p.koff3 + p.koff4 + p.kintAB) * RCAB)

    dAP  = p.k12 * C * p.V - p.k21 * AP
    dAbs = -p.ka * Abs + In

    return jnp.stack([dC, dRA, dRB, dRCA, dRCB, dRCAB, dAP, dAbs])


def initial_state(params: BiTEParams):
    return jnp.array([
        0.0,
        params.ksynA / params.kdegA,
        params.ksynB / params.kdegB,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ])


# ==========================
# Forward solve (baseline)
# ==========================

def forward_solve(u: float, params: BiTEParams, N_t: int = 4000):
    """Forward solve used for analytic adjoint + finite differences."""
    t0, t1 = 0.0, params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()
    controller = step_controller(params, rtol=1e-7, atol=1e-9)
    ts = jnp.linspace(t0, t1, N_t)
    sol = dx.diffeqsolve(
        dx.ODETerm(bite_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-4,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=5_000_000,
        stepsize_controller=controller,
    )
    return ts, sol.ys


# ===================
# Cost functional J
# ===================

def J_from_trajectory(ts, ys, params: BiTEParams):
    RCAB = ys[:, 5]
    diff = RCAB - params.R_ref
    integrand = 0.5 * diff**2
    return _trapz(integrand, ts)


def J_reduced(u: float, params: BiTEParams):
    ts, ys = forward_solve(u, params)
    return J_from_trajectory(ts, ys, params)


# ==========================
# Jacobian g_y(t, y; u)
# ==========================

def dg_dy(t, y, params: BiTEParams):
    C, RA, RB, RCA, RCB, RCAB, AP, Abs = y
    p = params

    kel, kon1, koff1 = p.kel, p.kon1, p.koff1
    kon2, koff2      = p.kon2, p.koff2
    kon3, koff3      = p.kon3, p.koff3
    kon4, koff4      = p.kon4, p.koff4
    ksynA, kdegA     = p.ksynA, p.kdegA
    ksynB, kdegB     = p.ksynB, p.kdegB
    kintA, kintB, kintAB = p.kintA, p.kintB, p.kintAB
    k12, k21, ka, V  = p.k12, p.k21, p.ka, p.V

    J = jnp.zeros((8, 8), dtype=y.dtype)

    # row 0: dC/dt
    J = J.at[0, 0].set(-(kel + kon1 * RA + kon2 * RB + k12))
    J = J.at[0, 1].set(-kon1 * C)
    J = J.at[0, 2].set(-kon2 * C)
    J = J.at[0, 3].set(koff1)
    J = J.at[0, 4].set(koff2)
    J = J.at[0, 6].set(k21 / V)
    J = J.at[0, 7].set(ka / V)

    # row 1: dRA/dt
    J = J.at[1, 0].set(-kon1 * RA)
    J = J.at[1, 1].set(-(kdegA + kon1 * C + kon4 * RCB))
    J = J.at[1, 3].set(koff1)
    J = J.at[1, 4].set(-kon4 * RA)
    J = J.at[1, 5].set(koff4)

    # row 2: dRB/dt
    J = J.at[2, 0].set(-kon2 * RB)
    J = J.at[2, 2].set(-(kdegB + kon2 * C + kon3 * RCA))
    J = J.at[2, 3].set(-kon3 * RB)
    J = J.at[2, 4].set(koff2)
    J = J.at[2, 5].set(koff3)

    # row 3: dRCA/dt
    J = J.at[3, 0].set(kon1 * RA)
    J = J.at[3, 1].set(kon1 * C)
    J = J.at[3, 2].set(-kon3 * RCA)
    J = J.at[3, 3].set(-(koff1 + kintA) - kon3 * RB)
    J = J.at[3, 5].set(koff3)

    # row 4: dRCB/dt
    J = J.at[4, 0].set(kon2 * RB)
    J = J.at[4, 1].set(-kon4 * RCB)
    J = J.at[4, 2].set(kon2 * C)
    J = J.at[4, 4].set(-(koff2 + kintB) - kon4 * RA)
    J = J.at[4, 5].set(koff4)

    # row 5: dRCAB/dt
    J = J.at[5, 1].set(kon4 * RCB)
    J = J.at[5, 2].set(kon3 * RCA)
    J = J.at[5, 3].set(kon3 * RB)
    J = J.at[5, 4].set(kon4 * RA)
    J = J.at[5, 5].set(-(koff3 + koff4 + kintAB))

    # row 6: dAP/dt
    J = J.at[6, 0].set(k12 * V)
    J = J.at[6, 6].set(-k21)

    # row 7: dAbs/dt
    J = J.at[7, 7].set(-ka)

    return J


# =========================
# Analytic adjoint (p(t))
# =========================

def make_y_interp(ts, ys):
    return dx.LinearInterpolation(ts=ts, ys=ys)


def adjoint_rhs(t, p, args):
    params, y_interp = args
    y = y_interp.evaluate(t)
    Jg = dg_dy(t, y, params)
    RCAB = y[5]
    # Observation term: (RCAB - R_ref)*e_RCAB
    obs = jnp.zeros_like(p)
    obs = obs.at[5].set(RCAB - params.R_ref)
    return -Jg.T @ p + obs


def solve_adjoint(ts, ys, params: BiTEParams):
    """
    Solve p'(t) = -g_y^T p + (RCAB - R_ref) e_RCAB,   p(T) = 0,
    backward from T to 0, sampled on same ts grid.
    """
    y_interp = make_y_interp(ts, ys)
    solver = dx.Tsit5()
    t0, t1 = params.T, 0.0
    pT = jnp.zeros(8)

    controller = step_controller(params, rtol=1e-7, atol=1e-9)
    sol = dx.diffeqsolve(
        dx.ODETerm(adjoint_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=-1e-2,
        y0=pT,
        args=(params, y_interp),
        saveat=dx.SaveAt(ts=ts[::-1]),
        max_steps=5_000_000,
        stepsize_controller=controller,
    )

    p_backwards = sol.ys
    p_ts = p_backwards[::-1]
    return p_ts


def grad_analytic(u: float, params: BiTEParams, N_adjoint: int = 12_000):
    """
    dJ/du = - (1/eps) sum_l ∫_{t_l}^{t_l+eps} p_Abs(t) dt,
    where Abs is the last state (index 7).
    """
    ts, ys = forward_solve(u, params, N_t=N_adjoint)
    p_ts = solve_adjoint(ts, ys, params)
    p_abs_interp = dx.LinearInterpolation(ts=ts, ys=p_ts[:, 7])

    # The paper configuration uses eps=0.01 days, which is shorter than the
    # fixed objective-output spacing. Integrating a Boolean mask on that coarse
    # grid can therefore miss most of a dose window. Evaluate the already-solved
    # adjoint interpolation on a dedicated grid inside every window instead.
    local_times = jnp.linspace(0.0, params.eps, 33)
    integral = jnp.asarray(0.0, dtype=ts.dtype)
    for dose_time in params.dose_times:
        window_times = dose_time + local_times
        window_values = jax.vmap(p_abs_interp.evaluate)(window_times)
        integral = integral + _trapz(window_values, window_times)

    return -integral / params.eps


# ============================
# AD gradients via diffrax
# ============================

# --- Forward-mode AD (via ForwardMode adjoint + jvp) ---

def forward_solve_forwardmode(u: float, params: BiTEParams, N_t: int = 4000):
    t0, t1 = 0.0, params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()
    controller = step_controller(params, rtol=1e-7, atol=1e-9)
    ts = jnp.linspace(t0, t1, N_t)
    sol = dx.diffeqsolve(
        dx.ODETerm(bite_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-4,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=5_000_000,
        stepsize_controller=controller,
        adjoint=dx.ForwardMode(),  # recommended for forward-mode AD
    )
    return ts, sol.ys


def J_reduced_forwardmode(u: float, params: BiTEParams):
    ts, ys = forward_solve_forwardmode(u, params)
    return J_from_trajectory(ts, ys, params)


def grad_forwardmode(u: float, params: BiTEParams):
    """Forward-mode AD gradient dJ/du via jvp."""
    _, dJ = jax.jvp(
        lambda uu: J_reduced_forwardmode(uu, params),
        (u,),
        (1.0,),
    )
    return dJ


# --- BacksolveAdjoint AD (continuous adjoint inside diffrax) ---

def forward_solve_backsolve(u: float, params: BiTEParams, N_t: int = 4000):
    """
    Forward solve configured so that jax.grad(J_reduced_backsolve)
    uses diffrax.BacksolveAdjoint under the hood.

    We follow the pattern from the Diffrax FAQ:
    diffeqsolve(..., solver=Tsit5(), adjoint=BacksolveAdjoint(), max_steps=None).
    """
    t0, t1 = 0.0, params.T
    y0 = initial_state(params)

    # Use Tsit5 here; docs recommend Tsit5 + BacksolveAdjoint as the "odeint-like" combo.
    solver = dx.Tsit5()
    controller = step_controller(params, rtol=1e-6, atol=1e-8)

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(bite_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=None,                       # let the controller choose
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        stepsize_controller=controller,
        adjoint=dx.BacksolveAdjoint(),  # use default Backsolve config
        max_steps=None,                 # <-- important for Backsolve, see Diffrax FAQ
    )

    return ts, sol.ys



def J_reduced_backsolve(u: float, params: BiTEParams):
    ts, ys = forward_solve_backsolve(u, params)
    return J_from_trajectory(ts, ys, params)


grad_backsolve = jax.grad(J_reduced_backsolve, argnums=0)


# =========================
# Optimization and output
# =========================

def _parse_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Optimize the OptiDose bispecific T-cell engager benchmark with DiffDose."
    )
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--methods", choices=("diffdose", "all"), default="diffdose")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--timing", choices=("warm", "cold"), default="warm")
    parser.add_argument("--_backsolve-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_analysis(*, preset="quick", methods="diffdose", output=None, timing="warm"):
    """Run the BiTE optimization benchmark and return method-level results."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from utils.OptimizationUtils import (
        run_lbfgsb,
        run_nelder_mead,
        run_random_walk,
        save_results,
        scalar_bound_aware_finite_difference,
    )

    params = make_params()
    initial_controls = np.asarray([800.0])
    bounds = ((0.0, 1000.0),)
    max_iterations = 1 if preset == "quick" else 20
    fd_iterations = 1 if preset == "quick" else 12

    def loss(controls):
        return float(J_reduced(float(np.asarray(controls)[0]), params))

    def finite_difference(controls):
        derivative = scalar_bound_aware_finite_difference(
            lambda value: loss(np.asarray([value])),
            float(np.asarray(controls)[0]),
            bounds[0],
            relative_step=1e-3,
        )
        return np.asarray([derivative])

    routes = [
        (
            "Forward AD",
            lambda controls: np.asarray(
                [float(grad_forwardmode(jnp.asarray(float(np.asarray(controls)[0])), params))]
            ),
            1,
            max_iterations,
        )
    ]
    if methods == "all":
        routes = [
            (
                "OptiDose",
                lambda controls: np.asarray(
                    [float(grad_analytic(float(np.asarray(controls)[0]), params))]
                ),
                1,
                max_iterations,
            ),
            (
                "Adjoint Sensitivity",
                lambda controls: np.asarray(
                    [float(grad_backsolve(jnp.asarray(float(np.asarray(controls)[0])), params))]
                ),
                1,
                max_iterations,
            ),
            (
                "Forward AD",
                lambda controls: np.asarray(
                    [float(grad_forwardmode(jnp.asarray(float(np.asarray(controls)[0])), params))]
                ),
                1,
                max_iterations,
            ),
            (
                "Reverse AD",
                lambda controls: np.asarray(
                    [
                        float(
                            jax.grad(J_reduced, argnums=0)(
                                jnp.asarray(float(np.asarray(controls)[0])), params
                            )
                        )
                    ]
                ),
                1,
                max_iterations,
            ),
            ("Finite Difference", finite_difference, 2, fd_iterations),
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
            forward_solves_per_gradient=solve_count,
        )
        for name, gradient, solve_count, iterations in routes
    ]
    if methods == "all":
        reference = float(finite_difference(initial_controls)[0])
        validation = []
        for name, gradient, _, _ in routes:
            value = float(gradient(initial_controls)[0])
            validation.append(
                {
                    "method": name,
                    "gradient": value,
                    "finite_difference_reference": reference,
                    "relative_error": abs(value - reference) / max(abs(reference), 1e-12),
                }
            )
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
                    step_scale=20.0,
                ),
            ]
        )
        if output is not None:
            Path(output).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(validation).to_csv(
                Path(output) / "BiTE_gradient_validation.csv", index=False
            )
    if output is not None:
        save_results(results, Path(output), "BiTE")
    return results


def main():
    import json

    args = _parse_args()
    if args._backsolve_worker:
        params = make_params()
        print(json.dumps({"gradient": float(grad_backsolve(jnp.asarray(800.0), params))}))
        return
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
