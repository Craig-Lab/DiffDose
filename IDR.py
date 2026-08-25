#!/usr/bin/env python3
"""Indirect-response (IDR) dose-amplitude benchmark.

The primary benchmark reproduces the short ``epsilon``-window dose input used
by OptiDose so gradient routes are compared on the same numerical problem.  A
separate supplementary workflow uses instantaneous fixed-time state jumps and
differentiates the complete segmented solve directly; see
``jump_event_loss_*`` and ``run_dosing_representation_comparison``.

Examples
--------
Run the quick DiffDose route::

    python IDR.py --preset quick --output output

Run every manuscript comparator::

    python IDR.py --preset paper --methods all --output output

Regenerate the smooth-versus-jump optimization data::

    python IDR.py --preset paper --compare-dose-representations --output output

Model and benchmark source: Bachmann et al., 2021,
https://doi.org/10.1007/s10957-021-01819-w
"""

from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import diffrax as dx

jax.config.update("jax_enable_x64", True)


# =========================
#  Parameters and schedule
# =========================

@jax.tree_util.register_pytree_node_class
@dataclass
class IndirectResponseParams:
    V: float
    B0: float
    kout: float
    kin: float
    kel: float
    Emax: float
    EC50: float
    B_tar: float
    m1: float
    eps: float
    T: float
    t_doses: jnp.ndarray  # shape (m, n_i)

    def tree_flatten(self):
        children = (
            self.V,
            self.B0,
            self.kout,
            self.kin,
            self.kel,
            self.Emax,
            self.EC50,
            self.B_tar,
            self.m1,
            self.eps,
            self.T,
            self.t_doses,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def make_params():
    """Create parameter object with OptiDose IDR test settings."""
    m = 6
    n_i = 7
    week = 7.0

    # dose times: week i (0-based) & day l (0..6)
    t_doses = jnp.stack(
        [(i * week) + jnp.arange(n_i) for i in range(m)],
        axis=0
    )  # (m, n_i)

    return IndirectResponseParams(
        V=3.0,
        B0=46.0,
        kout=0.02,
        kin=0.92,
        kel=0.49,
        Emax=8.8,
        EC50=0.81,
        B_tar=10.0,
        m1=2.0,
        eps=0.1,       # smoothing window for bolus (days)
        T=42.0,        # 6 weeks
        t_doses=t_doses
    )


# =========================
#  Reference trajectory
# =========================

def B_ref(t, params: IndirectResponseParams):
    """Quadratic down to B_tar by 14 days, then constant."""
    B0 = params.B0
    Btar = params.B_tar
    m1 = params.m1
    tsw = 7.0 * m1  # 14 days

    quad = ((B0 - Btar) / (tsw ** 2)) * t**2 \
           - (2.0 * (B0 - Btar) / tsw) * t + B0
    return jnp.where(t <= tsw, quad, Btar)


# =========================
#  Regularised IV bolus
# =========================

def dose_profile(t, u, params: IndirectResponseParams):
    """
    Regularised IV-bolus input:
        \tilde r(t,u) = sum_i u_i/(V eps) sum_l 1_[t_{i,l}, t_{i,l}+eps](t)
    """
    # t_doses: (m, n_i), u: (m,)
    in_pulse = (t >= params.t_doses) & (t < params.t_doses + params.eps)
    in_pulse = in_pulse.astype(jnp.result_type(t, u))  # 0/1 mask

    u_broadcast = u[:, None]  # (m, 1)
    total = jnp.sum(u_broadcast * in_pulse)  # sum over i,l

    return total / (params.V * params.eps)


# =========================
#  IDR ODE (state equation)
# =========================

def idr_rhs(t, y, args):
    """
    y = [C, B].
    """
    params, u = args
    C, B = y

    In = dose_profile(t, u, params)

    dCdt = In - params.kel * C

    frac = params.Emax * C / (params.EC50 + C)
    dBdt = params.kin - params.kout * (1.0 + frac) * B

    return jnp.stack([dCdt, dBdt])


def initial_state(params: IndirectResponseParams):
    C0 = 0.0
    B0 = params.B0
    return jnp.array([C0, B0])


# ==========================================
#  Instantaneous fixed-time jump formulation
# ==========================================

def idr_between_doses_rhs(t, y, params):
    """Continuous IDR dynamics between instantaneous dose events."""
    del t
    C, B = y
    dCdt = -params.kel * C
    frac = params.Emax * C / (params.EC50 + C)
    dBdt = params.kin - params.kout * (1.0 + frac) * B
    return jnp.stack([dCdt, dBdt])


def _jump_event_loss(u, params, *, adjoint, samples_per_segment=20):
    """Evaluate the IDR objective with exact, fixed-time dose jumps.

    Each event applies ``C(t_i+) = C(t_i-) + u_i / V`` and the ODE is solved
    to the next event. Event times are fixed constants, while jump amplitudes
    retain their JAX tracers. Consequently AD differentiates both the jump map
    and every subsequent solver segment without replacing the jump by a pulse.
    """
    daily_controls = jnp.repeat(u, params.t_doses.shape[1])
    event_times = params.t_doses.reshape(-1)
    state = initial_state(params)
    total_loss = jnp.asarray(0.0, dtype=jnp.result_type(u))
    term = dx.ODETerm(idr_between_doses_rhs)

    for event_index in range(event_times.shape[0]):
        # Exact jump map. The dose amplitude remains a differentiable tracer.
        state = state.at[0].add(daily_controls[event_index] / params.V)
        segment_start = event_times[event_index]
        segment_end = (
            event_times[event_index + 1]
            if event_index + 1 < event_times.shape[0]
            else params.T
        )
        segment_times = jnp.linspace(
            segment_start, segment_end, samples_per_segment
        )
        solution = dx.diffeqsolve(
            term,
            dx.Tsit5(),
            t0=segment_start,
            t1=segment_end,
            dt0=0.01,
            y0=state,
            args=params,
            saveat=dx.SaveAt(ts=segment_times),
            max_steps=20_000,
            adjoint=adjoint,
        )
        total_loss = total_loss + J_from_trajectory(
            segment_times, solution.ys, params
        )
        state = solution.ys[-1]

    return total_loss


def jump_event_loss_forwardmode(u, params):
    """Exact-jump objective configured for forward-mode differentiation."""
    return _jump_event_loss(u, params, adjoint=dx.ForwardMode())


def jump_event_loss_reverse(u, params):
    """Exact-jump objective configured for reverse-mode differentiation."""
    return _jump_event_loss(u, params, adjoint=dx.RecursiveCheckpointAdjoint())


def jump_event_gradient_forward(u, params):
    """Forward-mode AD gradient through exact jumps and solver segments."""
    return jax.jacfwd(jump_event_loss_forwardmode, argnums=0)(u, params)


def jump_event_gradient_reverse(u, params):
    """Reverse-mode AD gradient through exact jumps and solver segments."""
    return jax.grad(jump_event_loss_reverse, argnums=0)(u, params)


def forward_solve(u, params: IndirectResponseParams, N_t: int = 200):
    """Solve the IDR model forward in time for a given control u."""
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(idr_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=200_000,
    )

    return ts, sol.ys  # ys shape: (N_t, 2)


def forward_solve_backsolve(u, params: IndirectResponseParams, N_t: int = 200):
    """Forward solve using BacksolveAdjoint (continuous adjoint)."""
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(idr_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=200_000,
        adjoint=dx.BacksolveAdjoint(),
    )

    return ts, sol.ys


def forward_solve_forwardmode(u, params: IndirectResponseParams, N_t: int = 200):
    """
    Forward solve using adjoint=ForwardMode, so that JAX forward-mode
    (jax.jvp / jax.jacfwd) works through the solver.
    """
    t0 = 0.0
    t1 = params.T
    y0 = initial_state(params)
    solver = dx.Tsit5()

    ts = jnp.linspace(t0, t1, N_t)

    sol = dx.diffeqsolve(
        dx.ODETerm(idr_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=200_000,
        adjoint=dx.ForwardMode(),
    )

    return ts, sol.ys


# =========================
#  Cost functional J(u)
# =========================

def J_from_trajectory(ts, ys, params):
    B = ys[:, 1]
    Bref = B_ref(ts, params)
    diff = B - Bref
    integrand = 0.5 * diff**2
    return _trapz(integrand, ts)


def J_reduced(u, params):
    ts, ys = forward_solve(u, params)
    return J_from_trajectory(ts, ys, params)


def J_reduced_forwardmode(u, params):
    ts, ys = forward_solve_forwardmode(u, params)
    return J_from_trajectory(ts, ys, params)


def J_reduced_backsolve(u, params):
    ts, ys = forward_solve_backsolve(u, params)
    return J_from_trajectory(ts, ys, params)


def _trapz(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Simple trapezoidal rule (scalar or vector y) without jnp.trapz."""
    dx = x[1:] - x[:-1]
    y0 = y[:-1]
    y1 = y[1:]
    return jnp.sum(0.5 * (y0 + y1) * dx)


# =========================
#  Jacobian d g / d y
# =========================

def dg_dy(t, y, params):
    C, B = y
    J11 = -params.kel
    J12 = 0.0
    num = params.Emax * params.EC50
    den = (params.EC50 + C)**2
    J21 = -params.kout * (num / den) * B
    frac = params.Emax * C / (params.EC50 + C)
    J22 = -params.kout * (1.0 + frac)
    return jnp.array([[J11, J12],
                      [J21, J22]])


# =========================
#  Adjoint equation
# =========================

def make_y_interp(ts, ys):
    return dx.LinearInterpolation(ts=ts, ys=ys)


def adjoint_rhs(t, p, args):
    params, y_interp = args
    y = y_interp.evaluate(t)  # (2,)
    C, B = y

    Jg = dg_dy(t, y, params)         # (2,2)
    Bref = B_ref(t, params)

    # term (B - Bref) * [0,1]^T
    obs_term = jnp.array([0.0, B - Bref])

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
    pT = jnp.array([0.0, 0.0])

    sol = dx.diffeqsolve(
        dx.ODETerm(adjoint_rhs),
        solver,
        t0=t0,
        t1=t1,
        dt0=-0.01,               # negative for backward integration
        y0=pT,
        args=(params, y_interp),
        saveat=dx.SaveAt(ts=ts[::-1]),  # integrate backward, sample backward
        max_steps=200_000,
    )

    # sol.ys is aligned with ts[::-1]; flip back to match ts ascending
    p_backwards = sol.ys       # (N_t, 2)
    p_ts = p_backwards[::-1]   # now aligned with ts ascending
    return p_ts


# =========================
#  Gradient formula
# =========================

def grad_component_for_i(i, ts, pC, params):
    p_c_interp = dx.LinearInterpolation(ts=ts, ys=pC)
    local_times = jnp.linspace(0.0, params.eps, 17)
    window_times = params.t_doses[i, :, None] + local_times[None, :]
    window_values = jax.vmap(jax.vmap(p_c_interp.evaluate))(window_times)
    window_integrals = jax.vmap(lambda values, times: _trapz(values, times))(
        window_values, window_times
    )
    return -jnp.sum(window_integrals) / (params.V * params.eps)


def grad_reduced(u, params, N_adjoint: int = 500):
    ts, ys = forward_solve(u, params, N_t=N_adjoint)
    p_ts = solve_adjoint(ts, ys, params)
    pC = p_ts[:, 0]

    m = params.t_doses.shape[0]
    grad_components = jax.vmap(
        lambda i: grad_component_for_i(i, ts, pC, params)
    )(jnp.arange(m))

    return grad_components  # shape (m,)


def grad_reduced_matched(u, params):
    ts, ys = forward_solve(u, params)
    p_ts = solve_adjoint_matched(ts, ys, params)
    pC = p_ts[:, 0]

    m = params.t_doses.shape[0]
    grad_components = jax.vmap(
        lambda i: grad_component_for_i(i, ts, pC, params)
    )(jnp.arange(m))

    return grad_components


def grad_forwardmode(u, params):
    """∂J/∂u via forward-mode AD (JVP) through diffrax.ForwardMode solve."""
    return jax.jacfwd(J_reduced_forwardmode, argnums=0)(u, params)


def grad_backsolve_auto(u, params):
    """∂J/∂u via reverse-mode AD through diffrax.BacksolveAdjoint solve."""
    return jax.grad(J_reduced_backsolve, argnums=0)(u, params)


def grad_forward_sensitivity(u, params, N_t: int = 200):
    """Continuous forward sensitivities integrated with the IDR state."""
    u = jnp.asarray(u, dtype=jnp.float64)
    n_controls = u.shape[0]
    ts = jnp.linspace(0.0, params.T, N_t)
    state0 = initial_state(params)
    sensitivity0 = jnp.zeros((n_controls, 2), dtype=jnp.float64)
    augmented0 = jnp.concatenate([state0, sensitivity0.reshape(-1)])

    def augmented_rhs(t, augmented, args):
        local_params, local_u = args
        state = augmented[:2]
        sensitivity = augmented[2:].reshape((n_controls, 2))
        state_rate = idr_rhs(t, state, (local_params, local_u))
        state_jacobian = dg_dy(t, state, local_params)
        active = (t >= local_params.t_doses) & (
            t < local_params.t_doses + local_params.eps
        )
        pulse_counts = active.astype(jnp.float64).sum(axis=1)
        input_derivative = pulse_counts / (local_params.V * local_params.eps)
        control_jacobian = jnp.stack(
            [input_derivative, jnp.zeros_like(input_derivative)], axis=1
        )
        sensitivity_rate = sensitivity @ state_jacobian.T + control_jacobian
        return jnp.concatenate([state_rate, sensitivity_rate.reshape(-1)])

    solution = dx.diffeqsolve(
        dx.ODETerm(augmented_rhs),
        dx.Tsit5(),
        t0=0.0,
        t1=params.T,
        dt0=0.01,
        y0=augmented0,
        args=(params, u),
        saveat=dx.SaveAt(ts=ts),
        max_steps=200_000,
    )
    states = solution.ys
    difference = states[:, 1] - B_ref(ts, params)
    biomarker_sensitivity = states[:, 2:].reshape((N_t, n_controls, 2))[:, :, 1]
    dt = ts[1:] - ts[:-1]
    integrand = 0.5 * (
        difference[:-1, None] * biomarker_sensitivity[:-1]
        + difference[1:, None] * biomarker_sensitivity[1:]
    )
    return jnp.sum(integrand * dt[:, None], axis=0)


def _rk4_step_adj(t_hi, p, h, params, ts, ys):
    """One RK4 step for adjoint using forward states on the same grid."""
    # Linear interpolation between grid points t_hi and t_hi+h for y
    def y_of(t):
        # assume t in [t_hi+h, t_hi], h negative
        t0 = t_hi + h
        t1 = t_hi
        w = (t - t0) / (t1 - t0 + 1e-12)
        idx_hi = jnp.searchsorted(ts, t1, side="left")
        idx_lo = jnp.maximum(0, idx_hi - 1)
        y0 = ys[idx_lo]
        y1 = ys[idx_hi]
        return (1 - w) * y0 + w * y1

    def f_adj(t, p_vec):
        y = y_of(t)
        Jg = dg_dy(t, y, params)
        Bref = B_ref(t, params)
        obs_term = jnp.array([0.0, y[1] - Bref])
        return - Jg.T @ p_vec + obs_term

    k1 = f_adj(t_hi, p)
    k2 = f_adj(t_hi + 0.5 * h, p + 0.5 * h * k1)
    k3 = f_adj(t_hi + 0.5 * h, p + 0.5 * h * k2)
    k4 = f_adj(t_hi + h, p + h * k3)
    return p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_adjoint_matched(ts, ys, params):
    """Backward adjoint using the same grid as forward (no interpolation object)."""
    p = jnp.array([0.0, 0.0])
    p_traj = []
    for k in range(len(ts) - 1, 0, -1):
        t_hi = ts[k]
        t_lo = ts[k - 1]
        h = t_lo - t_hi  # negative
        p = _rk4_step_adj(t_hi, p, h, params, ts, ys)
        p_traj.append(p)
    p_traj = p_traj[::-1]
    p_traj = jnp.vstack([p_traj, jnp.array([0.0, 0.0])])
    return p_traj


def J_and_grad(u, params):
    ts, ys = forward_solve(u, params)
    J_val  = J_from_trajectory(ts, ys, params)
    p_ts   = solve_adjoint(ts, ys, params)
    pC     = p_ts[:, 0]

    m = params.t_doses.shape[0]
    grad_u = jax.vmap(
        lambda i: grad_component_for_i(i, ts, pC, params)
    )(jnp.arange(m))

    return J_val, grad_u


# =========================
# Optimization and output
# =========================

def _parse_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Optimize the OptiDose indirect-response benchmark with DiffDose."
    )
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--methods", choices=("diffdose", "all"), default="diffdose")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--timing", choices=("warm", "cold"), default="warm")
    parser.add_argument(
        "--compare-dose-representations",
        action="store_true",
        help=(
            "run the supplementary OptiDose-window versus exact-jump "
            "comparison instead of the primary benchmark"
        ),
    )
    return parser.parse_args()


def run_analysis(*, preset="quick", methods="diffdose", output=None, timing="warm"):
    """Run the IDR optimization benchmark and return method-level results."""
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
    initial_controls = np.ones(n_controls, dtype=float)
    bounds = tuple((0.0, 10.0) for _ in range(n_controls))
    max_iterations = 2 if preset == "quick" else 30
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
            ("Adjoint Sensitivity", lambda controls: np.asarray(grad_backsolve_auto(jnp.asarray(controls), params)), 1, max_iterations),
            ("Forward AD", lambda controls: np.asarray(grad_forwardmode(jnp.asarray(controls), params)), 1, max_iterations),
            ("Reverse AD", lambda controls: np.asarray(jax.grad(J_reduced, argnums=0)(jnp.asarray(controls), params)), 1, max_iterations),
            ("Forward Sensitivity", lambda controls: np.asarray(grad_forward_sensitivity(jnp.asarray(controls), params)), 1, max_iterations),
            (
                "Finite Difference",
                lambda controls: bound_aware_finite_difference(
                    loss, controls, bounds, relative_step=1e-2
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
            objective_scale=1e-3,
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
                    step_scale=0.2,
                ),
            ]
        )
    if output is not None:
        save_results(results, Path(output), "IDR")
    return results


def run_dosing_representation_comparison(
    *, preset="quick", output=None, timing="warm"
):
    """Compare regularized and exact-jump IDR dose representations.

    This supplementary analysis is intentionally separate from ``run_analysis``:
    the latter holds the OptiDose dose representation fixed while comparing
    gradient routes, whereas this function changes the dose representation.
    """
    from pathlib import Path

    import numpy as np

    from utils.OptimizationUtils import run_lbfgsb, save_results

    params = make_params()
    initial_controls = np.ones(int(params.t_doses.shape[0]), dtype=float)
    bounds = tuple((0.0, 10.0) for _ in initial_controls)
    max_iterations = 2 if preset == "quick" else 30

    def smooth_loss(controls):
        return float(J_reduced(jnp.asarray(controls), params))

    def jump_forward_loss(controls):
        return float(jump_event_loss_forwardmode(jnp.asarray(controls), params))

    def jump_reverse_loss(controls):
        return float(jump_event_loss_reverse(jnp.asarray(controls), params))

    routes = [
        (
            "Analytic adjoint (OptiDose)",
            smooth_loss,
            lambda controls: np.asarray(
                grad_reduced(jnp.asarray(controls), params)
            ),
        ),
        (
            "Forward-mode AD (jump)",
            jump_forward_loss,
            lambda controls: np.asarray(
                jump_event_gradient_forward(jnp.asarray(controls), params)
            ),
        ),
        (
            "Forward-mode AD (smooth)",
            smooth_loss,
            lambda controls: np.asarray(
                grad_forwardmode(jnp.asarray(controls), params)
            ),
        ),
        (
            "Reverse-mode AD (jump)",
            jump_reverse_loss,
            lambda controls: np.asarray(
                jump_event_gradient_reverse(jnp.asarray(controls), params)
            ),
        ),
        (
            "Reverse-mode AD (smooth)",
            smooth_loss,
            lambda controls: np.asarray(
                jax.grad(J_reduced, argnums=0)(jnp.asarray(controls), params)
            ),
        ),
    ]
    results = [
        run_lbfgsb(
            method=name,
            loss_fn=loss,
            gradient_fn=gradient,
            initial_controls=initial_controls,
            bounds=bounds,
            max_iterations=max_iterations,
            timing=timing,
            objective_scale=1e-3,
        )
        for name, loss, gradient in routes
    ]
    if output is not None:
        save_results(
            results, Path(output), "IDR_dose_representation_comparison"
        )
    return results


def main():
    args = _parse_args()
    if args.compare_dose_representations:
        results = run_dosing_representation_comparison(
            preset=args.preset,
            output=args.output,
            timing=args.timing,
        )
    else:
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
