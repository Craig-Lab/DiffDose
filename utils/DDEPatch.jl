#!/usr/bin/env julia

using ForwardDiff

"""
Return the primal scalar required by DelayDiffEq's discontinuity bracketing.

This is the local compatibility change tested with DelayDiffEq 5.61.0 when a
declared-delay tracker received a `ForwardDiff.Dual` and attempted to place it
in a `Float64` root-finding queue.
"""
primal_scalar(value) = value isa ForwardDiff.Dual ? ForwardDiff.value(value) : value

function patched_discontinuity_value(
    discontinuity_function,
    integrator,
    lag,
    scalar_type,
    time,
    theta,
    step,
)
    value = discontinuity_function(
        integrator,
        lag,
        scalar_type,
        time + theta * step,
    )
    return primal_scalar(value)
end

# This snippet is a solver-compatibility patch, not a complete event-time
# sensitivity rule: extracting the primal value removes derivatives of the
# internally located discontinuity time. The manuscript timing results use the
# undeclared-lag solver with a smooth dosing representation, implemented in
# `utils/NeutropeniaDDEModel.jl`; `NeutropeniaDDE.jl` constructs the seven timing
# controls and differentiates that smoothed objective.
