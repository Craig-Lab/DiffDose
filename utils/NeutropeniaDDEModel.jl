"""
Reusable 12-state state-dependent DDE model of granulopoiesis, chemotherapy,
and exogenous G-CSF.

Load from another Julia script with:

    include("utils/NeutropeniaDDEModel.jl")
    using .NeutropeniaDDEModel.Parameters
    using .NeutropeniaDDEModel.Dosing
    using .NeutropeniaDDEModel.SolverUndeclared

    p = default_parameters()
    p.AdminChemo = 1
    p.AdminGCSF = 1
    chemo = [ChemoInfusion(0.0, p.TotalDose, p.DeltaC)]
    gcsf = [GCSFInjection(8.0, p.Dose)]
    sol = solve_scenario_undeclared(p, (0.0, 14.0);
        chemo_schedule=chemo, gcsf_schedule=gcsf,
        smooth_chemo=false, smooth_gcsf=false, saveat=0.1)

The example above uses exact fixed-time callbacks. See `NeutropeniaDDE.jl` for
the differentiable seven-cycle timing objective and optimization workflow, and
`utils/NeutropeniaDDEParity.jl` for comparison with MATLAB reference output.

Scientific sources:

* Craig et al. (2015), https://doi.org/10.1016/j.jtbi.2015.08.015
* Craig, Humphries & Mackey (2016),
  https://doi.org/10.1007/s11538-016-0179-8
* Craig (2017), https://doi.org/10.1002/psp4.12191
* Gonzalez-Sales et al. (2012),
  https://doi.org/10.1007/s40262-012-0011-z

The biological equations and source parameterization are translated from the
published MATLAB implementation. The dosing wrappers, AD-safe numerical
guards, optional sharp sigmoid timing map, and undeclared-delay solve are
DiffDose implementation choices rather than components of the Craig model.
Code-aligned values and provenance are in
`data/NeutropeniaDDEParameters.csv`.
"""
module NeutropeniaDDEModel

module Parameters

using Parameters: @with_kw

export ModelParameters, default_parameters, with_parameters, initial_conditions,
       recompute_derived!, find_root

"""
ModelParameters holds the full set of neutrophil DDE parameters translated from MATLAB.
Mutable so scenario adjustments and virtual-population scaling can reuse the same instance.
"""
@with_kw mutable struct ModelParameters{T<:Real}
    # Stem cell parameters
    Qstar::T = T(1.1)
    gamma_s::T = T(0.1)
    tau_s::T = T(2.8)
    AQstar::T = T(2) * exp(-T(0.1) * T(2.8))
    fQ::T = T(8.0)
    s2::T = T(2.0)
    BetaQstar::T = T(0.043)
    theta2::T = ((T(1.1)^T(2.0) * T(0.043)) / (T(8.0) - T(0.043)))^(T(1.0) / T(2.0))

    # Neutrophil population parameters
    Np::T = T(0.93)
    Nm::T = T(4.51)
    Nrstar::T = T(2.26)
    Nstar::T = T(0.22) / T(0.585)

    # Neutrophil kinetics
    gamma_N::T = T(35.0) / T(16.0)
    tauNC::T = T(16.0) / T(35.0)
    trans_homeo::T = (T(35.0) / T(16.0) * (T(0.22) / T(0.585))) / T(2.26)
    aNM::T = T(3.9)
    tauNR::T = T(2.7)
    gammaNr::T = T(1.0) / T(2.7) - (T(35.0) / T(16.0) * (T(0.22) / T(0.585))) / T(2.26)

    # Derived neutrophil parameters (computed)
    gammaNM::T = zero(T)
    kappaNstar::T = zero(T)
    kappa_delta::T = zero(T)
    kappa_delta1::T = zero(T)
    ANstar::T = zero(T)
    etaNPTAU::T = zero(T)
    etaNP_h::T = zero(T)
    tauNP::T = zero(T)

    # G-CSF pharmacokinetics
    kren::T = T(0.16139)
    kint::T = T(462.4209)
    k12G::T = T(2.2423)
    k21G::T = T(184.8658)
    Gstar::T = T(0.025)
    V::T = T(0.525)
    pow::T = T(1.4608)
    G2_h::T = zero(T)
    Gprod::T = zero(T)

    # Chemotherapy pharmacokinetics (Zalypsis)
    Cl::T = T(43.7) * T(24.0)
    V1::T = T(32.7)
    Q2::T = T(123.0) * T(24.0)
    V2::T = T(162.0)
    Q3::T = T(11.3) * T(24.0)
    V3::T = T(388.0)
    Q4::T = T(62.3) * T(24.0)
    V4::T = T(23.9)
    BSA::T = T(1.723)

    # Computed PK rates
    k10::T = (T(43.7) * T(24.0)) / T(32.7)
    k12::T = (T(123.0) * T(24.0)) / T(32.7)
    k21::T = (T(123.0) * T(24.0)) / T(162.0)
    k13::T = (T(11.3) * T(24.0)) / T(32.7)
    k31::T = (T(11.3) * T(24.0)) / T(388.0)
    k24::T = (T(62.3) * T(24.0)) / T(162.0)
    k42::T = (T(62.3) * T(24.0)) / T(23.9)

    # Fitted neutrophil parameters
    Ntot::T = T(4.2009)
    muval::T = T(0.844575731873359)
    bNP::T = T(0.022867963747985)
    Vmax::T = T(7.866997674083609)
    trans_ratio::T = T(11.355614184371277)
    ftrans0::T = T(0.020055920452792)
    bvtilde::T = T(0.031283425611887)
    bv::T = T(0.031283425611887) * T(7.866997674083609)

    # Additional computed parameters
    s1::T = T(1.5)
    etaNP_min::T = zero(T)
    kappaN_min::T = zero(T)
    bG::T = zero(T)
    trans_max::T = zero(T)

    # Chemotherapy pharmacodynamics
    EC50::T = T(0.75390)
    sc::T = T(0.89816)
    hQ::T = T(0.0079657)
    eta_NPneg::T = zero(T)
    hS::T = zero(T)

    # Treatment parameters
    AdminChemo::Int = 1
    DoseChemo::T = T(4000.0)
    TotalDose::T = T(4000.0) * T(1.723)
    DeltaC::T = T(1.0) / T(24.0)

    AdminGCSF::Int = 0
    Dose::T = T(300000.0)
    F::T = T(0.64466)
    ka::T = T(8.0236)
    Vd::T = T(2322.9)

    Period::T = T(21.0)
    DayAdminChemo::T = zero(T)
    NumAdminsChemo::Int = 1
    GCSFPeriod::T = T(1.0)
    AdminsGCSF::Int = 1
    AdminDay::T = T(15.0)

end

"""
Solve the maturation apoptosis rate and update all derived parameters to ensure
homeostasis matches the MATLAB baseline.
"""
function recompute_derived!(p::ModelParameters{T}) where T
    function fgam(x)
        return p.Nrstar * (exp(x * p.aNM) - one(T)) - x * p.tauNR * p.Nm
    end

    gamapprox = T(2) * ((p.tauNR * p.Nm) / (p.aNM * p.Nrstar) - one(T)) / p.aNM
    gammaNM = find_root(fgam, gamapprox, T)

    kappaNstar = (T(1) / T(3)) * (p.AQstar - one(T)) * p.BetaQstar
    kappa_delta = (-(kappaNstar + p.BetaQstar) * p.Qstar + (p.AQstar * p.BetaQstar * p.Qstar)) / p.Qstar
    kappa_delta1 = (T(2) / T(3)) * (p.AQstar - one(T)) * p.BetaQstar

    ANstar = p.Nrstar / (kappaNstar * p.Qstar * T(1e-3) * p.tauNR)
    etaNPTAU = ANstar * exp(gammaNM * p.aNM)
    etaNP_h = kappaNstar * p.Qstar * T(1e-3) * ((etaNPTAU - one(T)) / p.Np)
    tauNP = (one(T) / etaNP_h) * log(ANstar * exp(gammaNM * p.aNM))

    G2_h = (p.k12G * p.Gstar^p.pow * p.V * (p.Nstar + p.Nrstar)) /
           (p.kint + p.k12G * p.Gstar^p.pow + p.k21G)
    Gprod = p.kren * p.Gstar + p.k12G * p.Gstar^p.pow * p.V * (p.Nstar + p.Nrstar) -
            p.k12G * p.Gstar^p.pow * G2_h - p.k21G * G2_h

    etaNP_min = p.muval * etaNP_h
    kappaN_min = calculate_kappaN_min(p, etaNP_h, etaNP_min, ANstar, gammaNM, kappaNstar, tauNP)
    GBFstar = G2_h / (p.V * (p.Nstar + p.Nrstar))
    bG = GBFstar * ((p.trans_ratio * p.trans_homeo - p.ftrans0) / (p.trans_homeo - p.ftrans0))
    trans_max = p.trans_ratio * p.trans_homeo

    p.gammaNM = gammaNM
    p.kappaNstar = kappaNstar
    p.kappa_delta = kappa_delta
    p.kappa_delta1 = kappa_delta1
    p.ANstar = ANstar
    p.etaNPTAU = etaNPTAU
    p.etaNP_h = etaNP_h
    p.tauNP = tauNP
    p.G2_h = G2_h
    p.Gprod = Gprod
    p.etaNP_min = etaNP_min
    p.kappaN_min = kappaN_min
    p.bG = bG
    p.trans_max = trans_max

    return p
end

"""
Root finder (Newton's method) used to compute gammaNM, matching MATLAB's fzero usage.
"""
function find_root(f, x0, ::Type{T}) where T
    x = T(x0)
    tol = T(1e-12)
    maxiter = 100

    for _ in 1:maxiter
        fx = f(x)
        if abs(fx) < tol
            return x
        end
        h = T(1e-8)
        fpx = (f(x + h) - fx) / h
        x -= fx / fpx
    end

    return x
end

"""
Calculate minimal neutrophil differentiation rate based on knockout study parameters.
"""
function calculate_kappaN_min(p, etaNP_h, etaNP_min, ANstar, gammaNM, kappaNstar, tauNP)
    T = typeof(p.Qstar)
    Cko = T(0.25)
    VN0 = one(T) + (p.Vmax - one(T)) * (-p.Gstar / (p.bv - p.Gstar))
    theta = (Cko * (p.ftrans0 + p.gammaNr) / (p.trans_homeo + p.gammaNr)) *
            exp(p.aNM * (gammaNM / VN0 - gammaNM))
    return theta * kappaNstar * exp(tauNP * etaNP_h * (one(T) - p.muval))
end

"""
Construct default model parameters and ensure derived quantities are populated.
"""
function default_parameters(::Type{T}=Float64; kwargs...) where T
    p = ModelParameters{T}()
    for (k, v) in kwargs
        setfield!(p, k, T(v))
    end
    recompute_derived!(p)
    return p
end

"""
Return a fresh copy of parameters with keyword overrides and derived fields updated.
Useful for scenario adjustments and virtual population samples.
"""
function with_parameters(p::ModelParameters{T}; kwargs...) where T
    newp = ModelParameters{T}()
    for field in fieldnames(ModelParameters{T})
        setfield!(newp, field, getfield(p, field))
    end
    for (k, v) in kwargs
        field_T = fieldtype(ModelParameters{T}, k)
        setfield!(newp, k, convert(field_T, v))
    end
    return recompute_derived!(newp)
end

"""
Initial conditions consistent with MATLAB's history vector.
"""
function initial_conditions(p::ModelParameters{T}) where T
    base = [
        p.Qstar,
        p.Nstar,
        p.Gstar,
        p.aNM,
        zero(T),
        p.AQstar,
        p.ANstar,
        zero(T),
        zero(T),
        p.Nrstar,
        zero(T),
        p.G2_h
    ]

    return base
end

end # module

module Dosing

import SciMLBase
using SciMLBase: CallbackSet
using DiffEqCallbacks: PresetTimeCallback
import ForwardDiff

using ..Parameters: ModelParameters

export ChemoInfusion, GCSFInjection, DosingContext,
       normalize_chemo_schedule, normalize_gcsf_schedule,
       prepare_dosing

struct ChemoInfusion{T1<:Number,T2<:Number,T3<:Number}
    start::T1
    amount::T2
    duration::T3
end

struct GCSFInjection{T1,T2}
    time::T1
    amount::T2
end

mutable struct DosingContext{T<:Number}
    model_params::ModelParameters{T}
    chemo_rate::T
    chemo_smooth_shape::Symbol
    chemo_smooth_width::T
    use_chemo_smooth::Bool
    chemo_smooth_decay::T
    chemo_smooth_start::Vector{T}      # T-typed: carries Dual partials for timing AD
    chemo_smooth_amount::Vector{T}
    chemo_smooth_duration::Vector{T}
    gcsf_times::Vector{Float64}        # Float64: discrete callback path only
    gcsf_amounts::Vector{T}
    # Smooth pulse parameters (optional)
    gcsf_smooth_shape::Symbol
    gcsf_smooth_width::T
    use_smooth::Bool
    smooth_decay::T
    smooth_start::Vector{T}            # T-typed: carries Dual partials for timing AD
    smooth_amounts::Vector{T}
end

function DosingContext(params::ModelParameters{T};
        chemo_smooth_shape::Symbol = :gaussian,
        chemo_smooth_width::T = T(1.0/24.0),
        use_chemo_smooth::Bool = false,
        chemo_smooth_decay::T = T(100.0),
        chemo_smooth_start::Vector{T} = T[],
        chemo_smooth_amount::Vector{T} = T[],
        chemo_smooth_duration::Vector{T} = T[],
        gcsf_smooth_shape::Symbol = :gaussian,
        gcsf_smooth_width::T = T(0.25),
        use_smooth::Bool = false,
        smooth_decay::T = T(50.0),
        smooth_start::Vector{T} = T[],
        smooth_amounts::Vector{T} = T[]) where {T<:Number}
    return DosingContext{T}(params, zero(T),
        chemo_smooth_shape, chemo_smooth_width,
        use_chemo_smooth, chemo_smooth_decay, chemo_smooth_start, chemo_smooth_amount, chemo_smooth_duration,
        Float64[], T[],
        gcsf_smooth_shape, gcsf_smooth_width,
        use_smooth, smooth_decay, smooth_start, smooth_amounts)
end

function normalize_chemo_schedule(params::ModelParameters{T}, schedule) where {T}
    events = ChemoInfusion{T,T,T}[]
    if schedule === nothing
        if params.AdminChemo == 1
            for k in 0:(params.NumAdminsChemo - 1)
                start = T(params.DayAdminChemo + params.Period * T(k))
                push!(events, ChemoInfusion(start, T(params.TotalDose), T(params.DeltaC)))
            end
        end
    elseif schedule isa AbstractVector{<:Real}
        for t in schedule
            push!(events, ChemoInfusion(T(t), T(params.TotalDose), T(params.DeltaC)))
        end
    elseif schedule isa AbstractVector{<:ChemoInfusion}
        events = [ChemoInfusion(T(ev.start), T(ev.amount), T(ev.duration)) for ev in schedule]
    else
        throw(ArgumentError("Unsupported chemo schedule type: $(typeof(schedule))"))
    end
    sort!(events, by = ev -> ev.start)
    return events
end

function normalize_gcsf_schedule(params::ModelParameters{T}, schedule) where {T}
    events = GCSFInjection{T,T}[]
    if schedule === nothing
        if params.AdminGCSF == 1
            if params.AdminChemo == 1
                for cycle in 0:(params.NumAdminsChemo - 1)
                    base_time = params.DayAdminChemo + params.Period * T(cycle)
                    for dose_idx in 0:(params.AdminsGCSF - 1)
                        inj_time = T(base_time + params.AdminDay + params.GCSFPeriod * T(dose_idx))
                        push!(events, GCSFInjection(inj_time, T(params.Dose)))
                    end
                end
            else
                for dose_idx in 0:(params.AdminsGCSF - 1)
                    inj_time = T(params.AdminDay + params.GCSFPeriod * T(dose_idx))
                    push!(events, GCSFInjection(inj_time, T(params.Dose)))
                end
            end
        end
    elseif schedule isa AbstractVector{<:Real}
        for t in schedule
            push!(events, GCSFInjection(T(t), T(params.Dose)))
        end
    elseif schedule isa AbstractVector{<:GCSFInjection}
        events = [GCSFInjection(promote_type(T, typeof(ev.time))(ev.time),
                                promote_type(T, typeof(ev.amount))(ev.amount)) for ev in schedule]
    else
        throw(ArgumentError("Unsupported G-CSF schedule type: $(typeof(schedule))"))
    end
    sort!(events, by = ev -> ev.time)
    return events
end

"""
    prepare_dosing(params, tspan; smooth_chemo=false, smooth_gcsf=false, ...)

Construct treatment inputs in one of two explicitly separated forms:

* With `smooth_* = false`, `PresetTimeCallback` objects register fixed event
  times. Chemotherapy callbacks switch an infusion rate on/off; G-CSF
  callbacks register the onset of the causal subcutaneous absorption profile.
* With `smooth_* = true`, event times retain scalar type `T` and enter smooth
  right-hand-side functions. This is the timing-relaxation path used by the
  manuscript because `T` becomes `ForwardDiff.Dual` during gradient calls.

The callback path is useful for fixed schedules but does not differentiate an
event time. The smooth path is therefore a deliberate numerical approximation,
not an exact derivative through a moving discontinuity.
"""
function prepare_dosing(params::ModelParameters{T}, tspan::Tuple{R,R};
        chemo_schedule = nothing, gcsf_schedule = nothing,
        smooth_chemo = false, chemo_smooth_shape::Symbol = :gaussian,
        chemo_smooth_width = T(1.0 / 24.0), smooth_chemo_decay = T(100.0),
        smooth_gcsf = false, gcsf_smooth_shape::Symbol = :gaussian,
        gcsf_smooth_width = T(0.25), smooth_decay = T(50.0)) where {T<:Number,R<:Real}
    # Ensure decay parameters are of type T even if passed as Float64/Dual
    smooth_chemo_decay = T(smooth_chemo_decay)
    smooth_decay = T(smooth_decay)
    chemo_smooth_width = T(chemo_smooth_width)
    gcsf_smooth_width = T(gcsf_smooth_width)
    gcsf_events = normalize_gcsf_schedule(params, gcsf_schedule)
    # Keep event times as T (not stripped to Float64) so Dual partials propagate
    # through sigmoid smoothing for timing-gradient AD.
    smooth_start = smooth_gcsf ? T[ev.time for ev in gcsf_events] : T[]
    smooth_amounts = smooth_gcsf ? [ev.amount for ev in gcsf_events] : T[]
    ctx = DosingContext(params;
        use_chemo_smooth = smooth_chemo,
        chemo_smooth_shape = chemo_smooth_shape,
        chemo_smooth_width = chemo_smooth_width,
        chemo_smooth_decay = smooth_chemo_decay,
        use_smooth = smooth_gcsf,
        gcsf_smooth_shape = gcsf_smooth_shape,
        gcsf_smooth_width = gcsf_smooth_width,
        smooth_decay = smooth_decay,
        smooth_start = smooth_start,
        smooth_amounts = smooth_amounts)

    chemo_events = normalize_chemo_schedule(params, chemo_schedule)

    t0 = T(tspan[1])
    tol = T(eps(Float64))
    future_callbacks = Any[]
    tstops = Float64[]

    if !isempty(chemo_events)
        if smooth_chemo
            for ev in chemo_events
                push!(ctx.chemo_smooth_start, ev.start)  # T-typed, preserves Dual
                push!(ctx.chemo_smooth_amount, T(ev.amount))
                push!(ctx.chemo_smooth_duration, T(ev.duration))
                # Float64 tstops near transitions help solver accuracy
                push!(tstops, Float64(ForwardDiff.value(ev.start)))
                push!(tstops, Float64(ForwardDiff.value(ev.start)) + Float64(ForwardDiff.value(ev.duration)))
            end
        else
            start_idx = Ref(1)
            for (idx, ev) in enumerate(chemo_events)
                if ev.start <= t0 + tol
                    ctx.chemo_rate = ev.amount / ev.duration
                    start_idx[] = idx + 1
                else
                    break
                end
            end
            start_times = [Float64(ForwardDiff.value(ev.start)) for ev in chemo_events if ev.start > t0 + tol]
            stop_times = [Float64(ForwardDiff.value(ev.start + ev.duration)) for ev in chemo_events if ev.start + ev.duration > t0 + tol]
            if !isempty(start_times)
                start_cb = PresetTimeCallback(
                    start_times,
                    (integrator) -> begin
                        ev = chemo_events[start_idx[]]
                        ctx.chemo_rate = ev.amount / ev.duration
                        start_idx[] += 1
                    end;
                    save_positions = (false, false)
                )
                push!(future_callbacks, start_cb)
                append!(tstops, start_times)
            end
            if !isempty(stop_times)
                stop_idx = Ref(count(ev -> ev.start + ev.duration <= t0 + tol, chemo_events) + 1)
                stop_cb = PresetTimeCallback(
                    stop_times,
                    (_) -> begin
                        ctx.chemo_rate = zero(T)
                        stop_idx[] += 1
                    end;
                    save_positions = (false, false)
                )
                push!(future_callbacks, stop_cb)
                append!(tstops, stop_times)
            end
        end
    end

    if !isempty(gcsf_events) && !ctx.use_smooth
        inj_idx = Ref(1)
        for (idx, ev) in enumerate(gcsf_events)
            if ev.time <= t0 + tol
                push!(ctx.gcsf_times, Float64(ForwardDiff.value(ev.time)))
                push!(ctx.gcsf_amounts, T(ev.amount))
                inj_idx[] = idx + 1
            else
                break
            end
        end
        inj_times = [Float64(ForwardDiff.value(ev.time)) for ev in gcsf_events if ev.time > t0 + tol]
        if !isempty(inj_times)
            inj_cb = PresetTimeCallback(
                inj_times,
                (integrator) -> begin
                    ev = gcsf_events[inj_idx[]]
                    push!(ctx.gcsf_times, Float64(ForwardDiff.value(ev.time)))
                    push!(ctx.gcsf_amounts, T(ev.amount))
                    inj_idx[] += 1
                end;
                save_positions = (false, false)
            )
            push!(future_callbacks, inj_cb)
            append!(tstops, inj_times)
        end
    end

    callback = isempty(future_callbacks) ? nothing : CallbackSet(future_callbacks...)
    sort!(tstops)

    # Solver expects real tstops; times are not parameters so strip duals.
    tstopsF = map(t -> ForwardDiff.value(t), tstops)

    return ctx, callback, tstopsF
end

end # module

module DDESystem

import ForwardDiff
using ..Parameters: ModelParameters, initial_conditions
using ..Dosing: DosingContext

# AD-safe regularization for fractional powers near zero.
# ForwardDiff computes x^a via exp(a·log(x)); at x=0 the partial
# becomes 0·Inf = NaN.  Adding a tiny ε to the base prevents this
# while being negligible for any physically meaningful concentration.
const _AD_EPS = 1e-30

"""    _safe_base(x)
Return `abs(x) + ε` – a guaranteed-positive base for fractional powers.
"""
@inline _safe_base(x) = abs(x) + _AD_EPS

"""
    _safe_sigmoid(x)
Numerically stable sigmoid that never evaluates `exp(large_positive)`.
Prevents `Inf * 0 = NaN` in ForwardDiff Dual partials.

  σ(x) = x ≥ 0 ? 1/(1+exp(-x))          — exp arg ≤ 0, underflows to 0
        : exp(x)/(1+exp(x))              — exp arg ≤ 0, underflows to 0
"""
@inline function _safe_sigmoid(x)
    if ForwardDiff.value(x) >= 0
        return inv(1 + exp(-x))
    else
        ex = exp(x)
        return ex / (1 + ex)
    end
end

"""
Hill-type stem cell self-renewal rate (Beta) as defined in MATLAB Beta.m.
"""
@inline function beta_rate(y, p::ModelParameters)
    return p.fQ * (p.theta2^p.s2) / (p.theta2^p.s2 + _safe_base(y)^p.s2)
end

"""
Differentiation rate from stem cells to neutrophil lineage (kappaN.m).
"""
@inline function kappa_n_rate(y, p::ModelParameters)
    y_safe = _safe_base(y)
    return p.kappaN_min + (2.0 * (p.kappaNstar - p.kappaN_min) * y_safe^p.s1) /
           (y_safe^p.s1 + p.Gstar^p.s1)
end

"""
Baseline neutrophil proliferation rate (eta_NP.m).
"""
@inline function eta_np_rate(y, p::ModelParameters)
    return p.etaNP_h + p.bNP * ((p.etaNP_h - p.etaNP_min) * (abs(y)/p.Gstar - 1)) /
           (abs(y) + p.bNP)
end

"""
Chemotherapy-modified neutrophil proliferation (eta_NP_chemo.m).
Guard the power at C≈0 to stay Dual/AD safe (pow on 0^noninteger → NaN via log).
"""
@inline function eta_np_chemo_rate(y, C, p::ModelParameters)
    eta_np_base = eta_np_rate(y, p)
    Cabs = abs(C) / p.V1 + _AD_EPS   # regularise → safe for Cabs^sc when sc < 1
    dose_term = Cabs^p.sc
    chemo_effect = (p.EC50^p.sc) / (p.EC50^p.sc + dose_term)
    return p.eta_NPneg + (eta_np_base - p.eta_NPneg) * chemo_effect
end

"""
Reservoir-to-circulation transition rate (f_trans.m).
"""
@inline function f_trans_rate(G2, N, NR, p::ModelParameters)
    fact1 = G2 / (p.V * (N + NR))
    fact2 = p.G2_h / (p.V * (p.Nstar + p.Nrstar))
    return p.trans_homeo * (p.trans_ratio * (fact1 - fact2) + p.bG) /
           (fact1 - fact2 + p.bG)
end

@inline tau_n_delay(tauNM, p::ModelParameters) = p.tauNP + tauNM

@inline function maturation_velocity(G, p::ModelParameters)
    return ((p.Vmax - 1) * (G - p.Gstar) / (G - p.Gstar + p.bv)) + 1
end

@inline function gaussian_pulse(t, t0, area, width)
    # area * normalized Gaussian centered at t0 with std=width
    norm = one(width) / (width * sqrt(2π))
    return area * norm * exp(-0.5 * ((t - t0)/width)^2)
end

@inline function logistic_window(t, t0, dur, width, area)
    rise = 1 / (1 + exp(-(t - t0)/width))
    fall = 1 / (1 + exp((t - (t0 + dur))/width))
    norm = area / dur  # approximate normalization
    return norm * rise * fall
end

@inline function chemo_infusion_rate(ctx::DosingContext, t)
    if ctx.use_chemo_smooth
        total = zero(ctx.chemo_rate)
        for (t0, amt, dur) in zip(ctx.chemo_smooth_start, ctx.chemo_smooth_amount, ctx.chemo_smooth_duration)
            if ctx.chemo_smooth_shape == :gaussian
                total += gaussian_pulse(t, t0, amt, ctx.chemo_smooth_width)
            elseif ctx.chemo_smooth_shape == :logistic
                total += logistic_window(t, t0, dur, ctx.chemo_smooth_width, amt)
            elseif ctx.chemo_smooth_shape == :sigmoid
                # Sigmoid-edge rectangle: preserves rectangular IV shape with C∞ edges.
                # width parameter = ε (transition scale in days); sharpness = 1/ε.
                # ε is a hyperparameter (not differentiated); t0 and dur keep Dual partials.
                # Uses _safe_sigmoid to prevent exp overflow → Inf*0=NaN in ForwardDiff.
                ε = ForwardDiff.value(ctx.chemo_smooth_width)
                sharpness = inv(ε)
                σ_on  = _safe_sigmoid( sharpness * (t - t0))
                σ_off = _safe_sigmoid( sharpness * ((t0 + dur) - t))
                total += (amt / dur) * σ_on * σ_off
            else
                total += amt * ctx.chemo_smooth_decay * exp(-ctx.chemo_smooth_decay * (t - t0))
            end
        end
        return total
    else
        return ctx.chemo_rate
    end
end

@inline function unpack_gcsf_payload(payload, p::ModelParameters)
    if payload isa Number
        return payload, p.F, p.ka, p.Vd
    elseif payload isa NamedTuple
        dose =
            hasproperty(payload, :dose) ? getproperty(payload, :dose) :
            hasproperty(payload, :amount) ? getproperty(payload, :amount) :
            throw(ArgumentError("G-CSF payload missing :dose or :amount field"))
        F = hasproperty(payload, :F) ? getproperty(payload, :F) : p.F
        ka = hasproperty(payload, :ka) ? getproperty(payload, :ka) : p.ka
        Vd = hasproperty(payload, :Vd) ? getproperty(payload, :Vd) : p.Vd
        return dose, F, ka, Vd
    else
        throw(ArgumentError("Unsupported G-CSF payload type $(typeof(payload))"))
    end
end

"""Evaluate exogenous G-CSF input for callback or relaxed timing events.

For the manuscript path, the Heaviside onset in the first-order absorption
profile is replaced by `sigmoid((t - t0) / epsilon)`. Keeping `t0` dual-valued
is the exact code location where administration-time derivatives enter the DDE.
"""
function gcsf_input(ctx::DosingContext, t)
    p = ctx.model_params
    total = zero(typeof(p.Dose))
    initialized = false
    if ctx.model_params.AdminGCSF == 0
        return total
    end
    if ctx.use_smooth
        for (t0, payload) in zip(ctx.smooth_start, ctx.smooth_amounts)
            # Shape-aware time filter: sigmoid needs pre-injection onset,
            # other shapes use hard t >= t0 cutoff for past injections only.
            if ctx.gcsf_smooth_shape == :sigmoid
                ε_gcsf = ForwardDiff.value(ctx.gcsf_smooth_width)
                ForwardDiff.value(t - t0) < -30.0 * ε_gcsf && continue
            else
                t < t0 && continue
            end
            dose, F, ka, Vd = unpack_gcsf_payload(payload, p)
            iszero(dose) && continue
            contribution = if ctx.gcsf_smooth_shape == :sigmoid
                # Sigmoid-onset PK absorption: exact ka·F·(D/Vd)·exp(-ka·dt)
                # with Heaviside H(dt) replaced by σ(dt/ε).  C∞ smooth,
                # preserves exponential PK tail and total AUC to O(ε·ka).
                # Uses _safe_sigmoid to prevent exp overflow in ForwardDiff.
                dt = t - t0
                sharpness = inv(ε_gcsf)
                onset = _safe_sigmoid(sharpness * dt)
                ka * F * (dose / Vd) * exp(-ka * dt) * onset
            elseif ctx.gcsf_smooth_shape == :gaussian
                gaussian_pulse(t, t0, F * (dose / Vd), ctx.gcsf_smooth_width)
            elseif ctx.gcsf_smooth_shape == :logistic
                logistic_window(t, t0, 1e-3, ctx.gcsf_smooth_width, F * (dose / Vd))
            else
                dt = t - t0
                ctx.smooth_decay * F * (dose / Vd) * exp(-ctx.smooth_decay * dt)
            end
            if !initialized
                total = zero(typeof(contribution))
                initialized = true
            end
            total += contribution
        end
    else
        for (t0, payload) in zip(ctx.gcsf_times, ctx.gcsf_amounts)
            if t >= t0
                dose, F, ka, Vd = unpack_gcsf_payload(payload, p)
                iszero(dose) && continue
                dt = t - t0
                contribution = ka * F * (dose / Vd) * exp(-ka * dt)
                if !initialized
                    total = zero(typeof(contribution))
                    initialized = true
                end
                total += contribution
            end
        end
    end
    return initialized ? total : zero(promote_type(typeof(p.Dose), Float64))
end

"""
12-variable delay differential system equivalent to MATLAB Chemo4.m.
"""
function neutrophil_dde!(du, u, h, p_params::DosingContext, t)
    p = p_params.model_params

    tauNM = u[4]
    tau_N = tau_n_delay(tauNM, p)
    delays = (
        t - p.tau_s,
        t - tau_N,
        t - tau_N + p.tauNP,
        t - tauNM
    )

    ylag1 = h(p_params, delays[1])
    ylag2 = h(p_params, delays[2])
    ylag3 = h(p_params, delays[3])
    ylag4 = h(p_params, delays[4])

    infusion = chemo_infusion_rate(p_params, t)

    gcsf_dose = gcsf_input(p_params, t)

    V_n = maturation_velocity(u[3], p)
    V_n_lag = maturation_velocity(ylag4[3], p)

    eta_NPlag2 = eta_np_chemo_rate(ylag2[3], ylag2[5], p)
    eta_NPlag3 = eta_np_chemo_rate(ylag3[3], ylag3[5], p)

    deriv_AN = u[7] * ((V_n / V_n_lag) * (eta_NPlag3 - eta_NPlag2) -
                       (1 - (V_n / V_n_lag)) * p.gammaNM)

    du[1] = -(beta_rate(u[1], p) + kappa_n_rate(u[3], p) + p.kappa_delta) * u[1] +
            (u[6] * beta_rate(ylag1[1], p)) * ylag1[1]

    du[2] = f_trans_rate(u[12], u[2], u[10], p) * u[10] - p.gamma_N * u[2]

    u3_safe = _safe_base(u[3])
    u3pow = u[3] >= 0 ? u3_safe^p.pow : -(u3_safe^p.pow)
    du[3] = gcsf_dose + p.Gprod - p.kren * u[3] -
            p.k12G * u3pow * p.V * (u[2] + u[10]) +
            p.k12G * u3pow * u[12] + p.k21G * u[12]

    du[4] = 1 - (V_n / V_n_lag)

    du[5] = infusion - (p.k10 + p.k12 + p.k13) * u[5] + p.k21 * u[8] + p.k31 * u[9]

    du[6] = u[6] * (p.hQ * (ylag1[5] / p.V1 - u[5] / p.V1))

    du[7] = deriv_AN

    du[8] = -(p.k21 + p.k24) * u[8] + p.k12 * u[5] + p.k42 * u[11]

    du[9] = -p.k31 * u[9] + p.k13 * u[5]

    du[10] = (u[7] * 1e-3 * kappa_n_rate(ylag2[3], p)) * ylag2[1] * (V_n / V_n_lag) -
             u[10] * (f_trans_rate(u[12], u[2], u[10], p) + p.gammaNr)

    du[11] = -p.k42 * u[11] + p.k24 * u[8]

    du[12] = -p.kint * u[12] + p.k12G * ((u[2] + u[10]) * p.V - u[12]) * u3pow - p.k21G * u[12]

end

"""
History function replicates MATLAB initial conditions.
"""
function history_function(p_params, t)
    p = p_params.model_params
    return initial_conditions(p)
end

"""
State-dependent lag functions for DelayDiffEq `dependent_lags`.

**Julia convention**: return the delay DURATION τ (positive scalar),
NOT the delayed time point t−τ (that was the MATLAB `ddesd` convention).

The four delays mirror MATLAB's `delayP.m`:
  1. τ_s           — stem cell cycling delay (constant)
  2. τ_NP + τ_NM   — total neutrophil transit (τ_N = tauNP + u[4])
  3. τ_NM          — maturation time only (= u[4])
  4. τ_NM          — same, used for V_n_lag computation

These declarations are used by the non-AD parity solver. The manuscript timing
gradient uses undeclared lags and evaluates the same delays inline, avoiding
derivatives through DelayDiffEq discontinuity bookkeeping.
"""
function lag_functions()
    return [
        (u, p, t) -> p.model_params.tau_s,
        (u, p, t) -> p.model_params.tauNP + max(u[4], zero(eltype(u))),
        (u, p, t) -> max(u[4], zero(eltype(u))),
        (u, p, t) -> max(u[4], zero(eltype(u)))
    ]
end

end # module

"""
Reference solver with declared state-dependent lags.

This route is intended for forward-simulation parity checks against the
published MATLAB implementation. It is not used for the manuscript timing
gradients because DelayDiffEq's discontinuity bookkeeping is not an event-time
sensitivity implementation.
"""
module SolverDeclared

using DifferentialEquations
using DelayDiffEq

using ..Parameters: ModelParameters, initial_conditions
using ..DDESystem: neutrophil_dde!, history_function, lag_functions
using ..Dosing: prepare_dosing

export build_problem_declared, solve_scenario_declared

function build_problem_declared(params::ModelParameters, tspan::Tuple{T,T};
        chemo_schedule=nothing, gcsf_schedule=nothing,
        smooth_chemo::Bool=false, smooth_chemo_decay::T=T(100.0),
        chemo_smooth_shape::Symbol=:gaussian, chemo_smooth_width=1.0/24.0,
        smooth_gcsf::Bool=false, smooth_decay=T(50.0),
        gcsf_smooth_shape::Symbol=:gaussian, gcsf_smooth_width=0.25) where T
    u0 = initial_conditions(params)
    dosing_ctx, callback, tstops = prepare_dosing(params, tspan;
        chemo_schedule=chemo_schedule, gcsf_schedule=gcsf_schedule,
        smooth_chemo=smooth_chemo, chemo_smooth_shape=chemo_smooth_shape,
        chemo_smooth_width=chemo_smooth_width,
        smooth_chemo_decay=smooth_chemo_decay,
        smooth_gcsf=smooth_gcsf, gcsf_smooth_shape=gcsf_smooth_shape,
        gcsf_smooth_width=gcsf_smooth_width, smooth_decay=smooth_decay)
    problem = DDEProblem(
        neutrophil_dde!, u0, history_function, tspan, dosing_ctx;
        dependent_lags=lag_functions(),
    )
    return problem, callback, tstops
end

function solve_scenario_declared(params::ModelParameters, tspan::Tuple{T,T};
        alg=MethodOfSteps(Tsit5()), reltol=1e-6, abstol=1e-6,
        dtmax=1e-2, chemo_schedule=nothing, gcsf_schedule=nothing,
        smooth_chemo::Bool=false, smooth_chemo_decay::T=T(100.0),
        chemo_smooth_shape::Symbol=:gaussian, chemo_smooth_width=1.0/24.0,
        smooth_gcsf::Bool=false, smooth_decay=T(50.0),
        gcsf_smooth_shape::Symbol=:gaussian, gcsf_smooth_width=0.25,
        kwargs...) where T
    problem, callback, tstops = build_problem_declared(params, tspan;
        chemo_schedule=chemo_schedule, gcsf_schedule=gcsf_schedule,
        smooth_chemo=smooth_chemo, smooth_chemo_decay=smooth_chemo_decay,
        chemo_smooth_shape=chemo_smooth_shape,
        chemo_smooth_width=chemo_smooth_width,
        smooth_gcsf=smooth_gcsf, smooth_decay=smooth_decay,
        gcsf_smooth_shape=gcsf_smooth_shape,
        gcsf_smooth_width=gcsf_smooth_width)
    solve_kwargs = (; reltol, abstol, dtmax)
    callback !== nothing && (solve_kwargs = merge(solve_kwargs, (; callback)))
    !isempty(tstops) && (solve_kwargs = merge(solve_kwargs, (; tstops)))
    return solve(problem, alg; solve_kwargs..., kwargs...)
end

end # module SolverDeclared

"""
AD-Compatible Solver Module (No Discontinuity Tracking)

This module provides DDEProblem construction WITHOUT `dependent_lags`.
By computing state-dependent delays inline (undeclared), we enable:
- ForwardDiff.gradient
- ReverseDiffAdjoint
- TrackerAdjoint
- All other AD methods

Trade-off: Less accurate integration (use MethodOfSteps(RK4()) with tight tolerances)
"""
module SolverUndeclared

using DifferentialEquations
using DelayDiffEq

using ..Parameters: ModelParameters, initial_conditions
using ..DDESystem: neutrophil_dde!, history_function
using ..Dosing: prepare_dosing

export build_problem_undeclared, solve_scenario_undeclared

"""
Construct a DDEProblem WITHOUT dependent_lags for AD compatibility.

This allows ForwardDiff and other AD tools to work by not triggering
discontinuity tracking (which can't handle Dual numbers).
"""
function build_problem_undeclared(params::ModelParameters, tspan::Tuple{T,T};
        chemo_schedule=nothing, gcsf_schedule=nothing,
        smooth_chemo::Bool=true,  # Default to smooth for AD compatibility
        smooth_chemo_decay::T=T(100.0),
        chemo_smooth_shape::Symbol=:gaussian, chemo_smooth_width=1.0/24.0,
        smooth_gcsf::Bool=true,   # Default to smooth for AD compatibility
        smooth_decay=T(50.0),
        gcsf_smooth_shape::Symbol=:gaussian, gcsf_smooth_width=0.25) where T

    u0 = initial_conditions(params)
    dosing_ctx, callback, tstops = prepare_dosing(params, tspan;
        chemo_schedule=chemo_schedule, gcsf_schedule=gcsf_schedule,
        smooth_chemo=smooth_chemo, chemo_smooth_shape=chemo_smooth_shape,
        chemo_smooth_width=chemo_smooth_width, smooth_chemo_decay=smooth_chemo_decay,
        smooth_gcsf=smooth_gcsf, gcsf_smooth_shape=gcsf_smooth_shape,
        gcsf_smooth_width=gcsf_smooth_width, smooth_decay=smooth_decay)

    # KEY CHANGE: No dependent_lags!
    # State-dependent delays are computed inline in neutrophil_dde!
    # This is called "undeclared delays" and requires RK4 with residual control
    prob = DDEProblem(neutrophil_dde!, u0, history_function, tspan, dosing_ctx)

    return prob, callback, tstops
end

"""
Solve the neutrophil model with AD-compatible settings.

Uses MethodOfSteps(RK4()) for residual control of undeclared delays.
Tighter tolerances recommended since discontinuity tracking is disabled.
"""
function solve_scenario_undeclared(params::ModelParameters, tspan::Tuple{T,T};
        alg = MethodOfSteps(RK4()),  # RK4 for undeclared delays
        reltol = 1e-10,  # Tighter tolerances needed
        abstol = 1e-10,
        dtmax = 1e-2,
        chemo_schedule=nothing,
        gcsf_schedule=nothing,
        smooth_chemo::Bool=true,
        smooth_chemo_decay::T=T(100.0),
        chemo_smooth_shape::Symbol=:gaussian,
        chemo_smooth_width=1.0/24.0,
        smooth_gcsf::Bool=true,
        smooth_decay=T(50.0),
        gcsf_smooth_shape::Symbol=:gaussian,
        gcsf_smooth_width=0.25,
        kwargs...) where T

    prob, callback, tstops = build_problem_undeclared(params, tspan;
        chemo_schedule=chemo_schedule, gcsf_schedule=gcsf_schedule,
        smooth_chemo=smooth_chemo, smooth_chemo_decay=smooth_chemo_decay,
        chemo_smooth_shape=chemo_smooth_shape, chemo_smooth_width=chemo_smooth_width,
        smooth_gcsf=smooth_gcsf, smooth_decay=smooth_decay,
        gcsf_smooth_shape=gcsf_smooth_shape, gcsf_smooth_width=gcsf_smooth_width)

    solve_kwargs = (; reltol=reltol, abstol=abstol, dtmax=dtmax)
    if callback !== nothing
        solve_kwargs = merge(solve_kwargs, (; callback))
    end
    if !isempty(tstops)
        solve_kwargs = merge(solve_kwargs, (; tstops))
    end

    return solve(prob, alg; solve_kwargs..., kwargs...)
end

end # module

end # module NeutropeniaDDEModel
