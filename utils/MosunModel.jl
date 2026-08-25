"""
Reusable Julia translation of the 36-state mosunetuzumab QSP model.

The original SimBiology project and supporting MATLAB files are distributed in
the official source-code archive accompanying Hosseini et al. (2020):
https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-020-00145-7/MediaObjects/41540_2020_145_MOESM3_ESM.zip
The virtual-patient extension is described by Susilo et al. (2023;
https://doi.org/10.1111/cts.13501). `MosunParity.jl` compares this translation
with a trajectory exported from that SimBiology project.

Minimal fixed-regimen simulation:

    include("utils/MosunModel.jl")
    using .MosunModel

    p = default_params()
    mg_to_ug_per_kg(dose) = dose * 1000 / 70
    doses = Dict(0.0 => mg_to_ug_per_kg(1.6),
                 7.0 => mg_to_ug_per_kg(10.0),
                 14.0 => mg_to_ug_per_kg(10.0),
                 21.0 => mg_to_ug_per_kg(20.0))
    regimen = bolus_regimen(:TDBc_ugperkg, doses)
    built, sol = solve_regimen(regimen, p, make_solver_alg("rodas4p");
        saveat=0.0:1.0:84.0)

See `Mosun.jl` for VPop loading, the six-dose clinical objective, gradients,
and patient-level optimization.

Dose events are instantaneous fixed-time state updates, not smoothed inputs.
`MosunRegimenEvent.amount` retains the numeric type of the dose control. During
ForwardDiff evaluation that type is a dual number, so `apply_event_deltas!`
propagates amplitude derivatives through the jump map and subsequent ODE solve.
Administration times remain fixed `Float64` values in this case study.
"""
module MosunModel

using SciMLBase
using DifferentialEquations
using DiffEqCallbacks
using ForwardDiff
using ReverseDiff
using Sundials

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SOLVER_ABSTOL = parse(Float64, get(ENV, "TCE_ABSTOL", "1e-8"))
const SOLVER_RELTOL = parse(Float64, get(ENV, "TCE_RELTOL", "1e-5"))
const SMOOTH_POSITIVE_EPS = parse(Float64, get(ENV, "TCE_SMOOTH_POSITIVE_EPS", "1e-8"))
const TDBC_CUTOFF_UGPML = parse(Float64, get(ENV, "TCE_TDBC_CUTOFF_UGPML", "1e-5"))
const TDBC_GATE_EPS = parse(Float64, get(ENV, "TCE_TDBC_GATE_EPS", "1e-6"))

function make_solver_alg(solver_name::AbstractString)
    solver = lowercase(String(solver_name))
    if solver == "cvode_bdf"
        return CVODE_BDF()
    elseif solver == "qndf"
        return QNDF(autodiff = false)
    elseif solver == "rodas4p"
        return Rodas4P(autodiff = false)
    elseif solver == "kencarp4"
        return KenCarp4(autodiff = false)
    elseif solver == "vcabm"
        return VCABM()
    elseif solver == "tsit5"
        return Tsit5()
    else
        error("Unsupported solver=$solver. Use cvode_bdf, qndf, rodas4p, kencarp4, vcabm, or tsit5.")
    end
end

function make_solver_alg()
    solver = lowercase(get(ENV, "TCE_SOLVER", "cvode_bdf"))
    return make_solver_alg(solver)
end

pow_safe(a::Float64, b::Float64) = Float64(real((complex(a) ^ b)))
pow_safe(a::Float64, b::Integer) = Float64(real((complex(a) ^ b)))
pow_safe(a::Real, b::Real) = real((complex(a) ^ b))
pow_safe(a, b) = a ^ b
@inline function pow_safe(a::ReverseDiff.TrackedReal, b::Real)
    ReverseDiff.value(a) <= 0 ? zero(a) : exp(b * log(a))
end
@inline function pow_safe(a::Real, b::ReverseDiff.TrackedReal)
    a <= 0 ? zero(b) : exp(b * log(a))
end
@inline function pow_safe(a::ReverseDiff.TrackedReal, b::ReverseDiff.TrackedReal)
    ReverseDiff.value(a) <= 0 ? zero(a + b) : exp(b * log(a))
end
pow_safe_symbolic(a::Float64, b::Float64) = pow_safe(a, b)
pow_safe_symbolic(a::Float64, b::Integer) = pow_safe(a, b)
pow_safe_symbolic(a::Real, b::Real) = pow_safe(a, b)
pow_safe_symbolic(a, b) = pow_safe(a, b)

@inline function smooth_positive_part(x, eps::Real = SMOOTH_POSITIVE_EPS)
    epsx = oftype(x, eps)
    return (x + sqrt(x * x + epsx * epsx)) / 2
end

@inline function smooth_positive_part_deriv(x, eps::Real = SMOOTH_POSITIVE_EPS)
    epsx = oftype(x, eps)
    return (one(x) + x / sqrt(x * x + epsx * epsx)) / 2
end

@inline function smooth_heaviside(x, eps::Real = TDBC_GATE_EPS)
    epsx = oftype(x, eps)
    return (one(x) + x / sqrt(x * x + epsx * epsx)) / 2
end

@inline function smooth_heaviside_deriv(x, eps::Real = TDBC_GATE_EPS)
    epsx = oftype(x, eps)
    return (epsx * epsx) / (2 * (x * x + epsx * epsx)^(3 / 2))
end

@inline function smooth_tdbc_cutoff(val, threshold::Real = TDBC_CUTOFF_UGPML; pos_eps::Real = SMOOTH_POSITIVE_EPS, gate_eps::Real = TDBC_GATE_EPS)
    pos_val = smooth_positive_part(val, pos_eps)
    z = val - oftype(val, threshold)
    return pos_val * smooth_heaviside(z, gate_eps)
end

@inline function smooth_tdbc_cutoff_deriv(val, threshold::Real = TDBC_CUTOFF_UGPML; pos_eps::Real = SMOOTH_POSITIVE_EPS, gate_eps::Real = TDBC_GATE_EPS)
    pos_val = smooth_positive_part(val, pos_eps)
    pos_grad = smooth_positive_part_deriv(val, pos_eps)
    z = val - oftype(val, threshold)
    gate = smooth_heaviside(z, gate_eps)
    gate_grad = smooth_heaviside_deriv(z, gate_eps)
    return pos_grad * gate + pos_val * gate_grad
end

@inline function tdb_central_concentration(TDBc_ugperkg, Vc_tdb, PKflag, VPid, t, end_time, fvalidation)
    val = TDBc_ugperkg / Vc_tdb
    return smooth_tdbc_cutoff(val)
end
@inline tdb_central_concentration_symbolic(TDBc_ugperkg, Vc_tdb, PKflag, VPid, t, end_time, fvalidation) = TDBc_ugperkg / Vc_tdb

@inline sanitize_ad_value(x) = x

@inline function sanitize_ad_value(x::ForwardDiff.Dual{Tag,V,N}) where {Tag,V,N}
    primal = ForwardDiff.value(x)
    isfinite(primal) || return x
    parts = ForwardDiff.partials(x)
    all(isfinite, Tuple(parts)) && return x
    vals = ntuple(i -> begin
        pi = parts[i]
        isfinite(pi) ? pi : zero(V)
    end, N)
    return ForwardDiff.Dual{Tag}(primal, vals...)
end

@inline sanitize_ad_vector!(du::AbstractVector) = du

function sanitize_ad_vector!(du::AbstractVector{<:ForwardDiff.Dual})
    @inbounds for i in eachindex(du)
        du[i] = sanitize_ad_value(du[i])
    end
    return du
end

@inline function event_scalar_value(x::Real)
    try
        return Float64(real(x))
    catch
        if :value in fieldnames(typeof(x))
            return Float64(getfield(x, :value))
        end
        rethrow()
    end
end
@inline event_scalar_value(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))

const DYNAMIC_STATE_NAMES = (
    :actTtiss,
    :Btiss,
    :TDBc_ugperkg,
    :actTpb,
    :restTtiss,
    :TDBp_ugperkg,
    :restTpb,
    :Bpb,
    :BAFF,
    :act0Tpb,
    :act0Ttiss,
    :injection_effect,
    :Btiss2,
    :act0Ttiss2,
    :restTtiss2,
    :actTtiss2,
    :RTXc_ugperkg,
    :RTXp_ugperkg,
    :drug_effect,
    :B1920tiss3,
    :B19no20tiss3,
    :restTtiss3,
    :act0Ttiss3,
    :actTtiss3,
    :Blinc_ug,
    :restTtumor,
    :actTtumor,
    :Btumor,
    :act0Ttumor,
    :TDBsc_ugperkg,
    :TDBc_ugperml_AUC,
    :IL6pb,
    :IL6tiss,
    :IL6tiss2,
    :IL6tiss3,
    :IL6tumor,
)

const OBSERVABLE_NAMES = (
    :TDBc_ugperml,
    :RTXc_ugperml,
    :Blinc_ngperml,
    :restTpb_perml,
    :restTtiss_perml,
    :restTtiss2_perml,
    :restTtiss3_perml,
    :restTtumor_perml,
    :act0Tpb_perml,
    :act0Ttiss_perml,
    :act0Ttiss2_perml,
    :act0Ttiss3_perml,
    :act0Ttumor_perml,
    :actTpb_perml,
    :actTtiss_perml,
    :actTtiss2_perml,
    :actTtiss3_perml,
    :actTtumor_perml,
    :B19tiss3,
    :Bpb_perml,
    :Btiss_perml,
    :Btiss2_perml,
    :B1920tiss3_perml,
    :B19no20tiss3_perml,
    :Btumor_perml,
    :BTrratio_pb,
    :BTrratio_tiss,
    :BTrratio_tiss2,
    :B1920Trratio_tiss3,
    :BTrratio_tumor,
    :TaBratio_pb,
    :TaBratio_tiss,
    :TaBratio_tiss2,
    :TaB1920ratio_tiss3,
    :TaBratio_tumor,
    :totTtiss,
    :totTtiss2,
    :totTtiss3,
    :totTtumor,
    :Baffconsumption,
    :IL6combo,
    :TDBt_ugperml,
    :TDBt2_ugperml,
    :TDBt3_ugperml,
    :TDBtumor_ugperml,
    :RTXt_ugperml,
    :RTXt2_ugperml,
    :RTXt3_ugperml,
    :RTXtumor_ugperml,
    :Blint_ngperml,
    :Blint2_ngperml,
    :Blint3_ngperml,
    :Blintumor_ngperml,
    :totTpb_perml,
    :totTtiss_perml,
    :totTtiss2_perml,
    :totTtiss3_perml,
    :totTtumor_perml,
    :B19tiss3_perml,
    :B19Trratio_tiss3,
    :TaB19ratio_tiss3,
    :Bpb_norm,
    :drugTpbact,
    :BlinTpbact,
    :drugBpbkill,
    :BlinBpbkill,
    :drugTtissact,
    :drugBtisskill,
    :drugTtissact2,
    :drugBtisskill2,
    :drugTtissact3,
    :drugBtisskill3,
    :drugTtumoract,
    :drugBtumorkill,
    :BlinTtissact,
    :BlinBtisskill,
    :BlinTtissact2,
    :BlinBtisskill2,
    :BlinTtumoract,
    :BlinBtumorkill,
    :Tafraction_pb,
    :Tafraction_tiss,
    :BTtotRatio_tiss,
    :Tafraction_tiss2,
    :BTtotRatio_tiss2,
    :Tafraction_tiss3,
    :Tafraction_tumor,
    :BlinTtissact3,
    :BlinBtisskill3,
)

const RHS_OBSERVABLE_NAMES = (
    :TDBc_ugperml,
    :RTXc_ugperml,
    :Blinc_ngperml,
    :restTpb_perml,
    :act0Tpb_perml,
    :actTpb_perml,
    :B19tiss3,
    :Bpb_perml,
    :Btiss_perml,
    :Btiss2_perml,
    :B1920tiss3_perml,
    :Btumor_perml,
    :BTrratio_pb,
    :BTrratio_tiss,
    :BTrratio_tiss2,
    :B1920Trratio_tiss3,
    :BTrratio_tumor,
    :TaBratio_pb,
    :TaBratio_tiss,
    :TaBratio_tiss2,
    :TaB1920ratio_tiss3,
    :TaBratio_tumor,
    :TDBt_ugperml,
    :TDBt2_ugperml,
    :TDBt3_ugperml,
    :TDBtumor_ugperml,
    :RTXt_ugperml,
    :RTXt2_ugperml,
    :RTXt3_ugperml,
    :RTXtumor_ugperml,
    :Blint_ngperml,
    :Blint2_ngperml,
    :Blint3_ngperml,
    :Blintumor_ngperml,
    :totTpb_perml,
    :B19tiss3_perml,
    :B19Trratio_tiss3,
    :TaB19ratio_tiss3,
    :drugTpbact,
    :BlinTpbact,
    :drugBpbkill,
    :BlinBpbkill,
    :drugTtissact,
    :drugBtisskill,
    :drugTtissact2,
    :drugBtisskill2,
    :drugTtissact3,
    :drugBtisskill3,
    :drugTtumoract,
    :drugBtumorkill,
    :BlinTtissact,
    :BlinBtisskill,
    :BlinTtissact2,
    :BlinBtisskill2,
    :BlinTtumoract,
    :BlinBtumorkill,
    :BlinTtissact3,
    :BlinBtisskill3,
)

const PARAMETER_NAMES = (
    :VmB,
    :KmTB_kill,
    :KdrugB,
    :kBapop,
    :kBprolif,
    :kTprolif,
    :kBkill,
    :KdrugactT,
    :VmT,
    :KmBT_act,
    :kTaexit,
    :kTact,
    :fTadeact,
    :fTap,
    :Cl_tdb,
    :Cld_tdb,
    :Vc_tdb,
    :Vp_tdb,
    :kTgen,
    :kTaapop,
    :KTrp,
    :Trpbo_perml,
    :fTaprolif,
    :fBprolif,
    :Bpbo_perml,
    :fBexit,
    :KBp,
    :Kp,
    :Vpb,
    :Vtissue,
    :fTrapop,
    :Trpbref_perml,
    :nkill,
    :KTrp2,
    :Vm_tdb,
    :Km_tdb,
    :BW,
    :act0on,
    :kIL6prod,
    :fa0,
    :fAICD,
    :kBAFFprod,
    :Bpbref_perml,
    :thBAFF,
    :thalfIL6,
    :fTa0deact,
    :fTa0apop,
    :BAFFo,
    :fBAFFo,
    :tinjhalf,
    :finj,
    :KBp2,
    :Vtissue2,
    :Kp2,
    :tissue2on,
    :Vc_rtx,
    :Vp_rtx,
    :Cl_rtx,
    :Cld_rtx,
    :Vm_rtx,
    :Km_rtx,
    :Kmkill_rtx,
    :fTgenbl,
    :depleteTpb,
    :depleteBpb,
    :kTrexit,
    :PKflag,
    :VPid,
    :end_time,
    :fdrug,
    :Vtissue3,
    :KTrp3,
    :tissue3on,
    :Kp3,
    :kBtiss3exit,
    :KBp3,
    :B19no20_B1920_ratio,
    :fvalidation,
    :fapop_v24,
    :fBtissue3_v1,
    :kBmat_kBapop_ratio,
    :ndrugactT,
    :Cl_blin,
    :Vz_blin,
    :KdrugactT_blin,
    :ndrugactT_blin,
    :KmTB_kill_blin,
    :KdrugB_blin,
    :nkill_blin,
    :KmBT_act_blin,
    :kBapop_cll,
    :kBgen_cll,
    :fBkill,
    :fTact,
    :fKmTB_kill,
    :Kptumor,
    :Vtumor,
    :KBptumor,
    :KTrptumor,
    :tumor_on,
    :IL6_tiss_contribution,
    :kBtumorprolif,
    :S,
    :tissue1on,
    :kabs_TDB,
    :fbio_TDB,
    :Bcell_tumor_trafficking_on,
)

const DEAD_LEGACY_NAMES = (
    :act0Tpb_1,
    :Bpb_1,
    :restTpb_1,
    :actTpb_1,
    :B19TtotRatio_tiss3,
    :Tafraction_pb_init,
)

const DYNAMIC_STATE_COUNT = 36
const OBSERVABLE_COUNT = 89
const RHS_OBSERVABLE_COUNT = 58

Base.@kwdef mutable struct MosunParams
    VmB::Float64 = 0.95
    KmTB_kill::Float64 = 0.75
    KdrugB::Float64 = 10.0
    kBapop::Float64 = 0.03
    kBprolif::Float64 = 0.7
    kTprolif::Float64 = 0.7
    kBkill::Float64 = 1.0
    KdrugactT::Float64 = 50.0
    VmT::Float64 = 0.9
    KmBT_act::Float64 = 0.1
    kTaexit::Float64 = 1.0
    kTact::Float64 = 1.0
    fTadeact::Float64 = 0.0
    fTap::Float64 = 1.0
    Cl_tdb::Float64 = 3.0
    Cld_tdb::Float64 = 25.0
    Vc_tdb::Float64 = 35.0
    Vp_tdb::Float64 = 110.0
    kTgen::Float64 = 1.0
    kTaapop::Float64 = 0.5
    KTrp::Float64 = 1.0
    Trpbo_perml::Float64 = 2.0e6
    fTaprolif::Float64 = 2.0
    fBprolif::Float64 = 0.0
    Bpbo_perml::Float64 = 1.0e6
    fBexit::Float64 = 1.0
    KBp::Float64 = 1.0
    Kp::Float64 = 0.14
    Vpb::Float64 = 40.0
    Vtissue::Float64 = 40.0
    fTrapop::Float64 = 0.1
    Trpbref_perml::Float64 = 2.0e6
    nkill::Float64 = 2.0
    KTrp2::Float64 = 0.02
    Vm_tdb::Float64 = 212.0
    Km_tdb::Float64 = 4.74
    BW::Float64 = 3.4
    act0on::Float64 = 1.0
    kIL6prod::Float64 = 1.0
    fa0::Float64 = 0.5
    fAICD::Float64 = 1.0
    kBAFFprod::Float64 = 1.0
    Bpbref_perml::Float64 = 1.0e6
    thBAFF::Float64 = 30.0
    thalfIL6::Float64 = 20.0
    fTa0deact::Float64 = 1.0
    fTa0apop::Float64 = 1.0
    BAFFo::Float64 = 1.0
    fBAFFo::Float64 = 0.1
    tinjhalf::Float64 = 1.0
    finj::Float64 = 0.1
    KBp2::Float64 = 1.0
    Vtissue2::Float64 = 1.0
    Kp2::Float64 = 0.07
    tissue2on::Float64 = 0.0
    Vc_rtx::Float64 = 28.0
    Vp_rtx::Float64 = 9.0
    Cl_rtx::Float64 = 8.0
    Cld_rtx::Float64 = 23.0
    Vm_rtx::Float64 = 0.0
    Km_rtx::Float64 = 1.0
    Kmkill_rtx::Float64 = 25.0
    fTgenbl::Float64 = 0.0
    depleteTpb::Float64 = 0.0
    depleteBpb::Float64 = 0.0
    kTrexit::Float64 = 1.0
    PKflag::Float64 = 1.0
    VPid::Float64 = 1.0
    end_time::Float64 = 1.0
    fdrug::Float64 = 1.0
    Vtissue3::Float64 = 1.0
    KTrp3::Float64 = 500.0
    tissue3on::Float64 = 0.0
    Kp3::Float64 = 0.07
    kBtiss3exit::Float64 = 0.03
    KBp3::Float64 = 600.0
    B19no20_B1920_ratio::Float64 = 0.25
    fvalidation::Float64 = 0.0
    fapop_v24::Float64 = 0.0
    fBtissue3_v1::Float64 = 1.0
    kBmat_kBapop_ratio::Float64 = 0.25
    ndrugactT::Float64 = 1.0
    Cl_blin::Float64 = 22300.0
    Vz_blin::Float64 = 1610.0
    KdrugactT_blin::Float64 = 0.1
    ndrugactT_blin::Float64 = 1.0
    KmTB_kill_blin::Float64 = 0.75
    KdrugB_blin::Float64 = 0.015
    nkill_blin::Float64 = 1.027
    KmBT_act_blin::Float64 = 0.716
    kBapop_cll::Float64 = 1.0
    kBgen_cll::Float64 = 1.0
    fBkill::Float64 = 1.0
    fTact::Float64 = 1.0
    fKmTB_kill::Float64 = 1.0
    Kptumor::Float64 = 1.0
    Vtumor::Float64 = 1.0
    KBptumor::Float64 = 1.0
    KTrptumor::Float64 = 1.0
    tumor_on::Float64 = 0.0
    IL6_tiss_contribution::Float64 = 0.02
    kBtumorprolif::Float64 = 0.025
    S::Float64 = 1.0
    tissue1on::Float64 = 1.0
    kabs_TDB::Float64 = 1.4
    fbio_TDB::Float64 = 0.6
    Bcell_tumor_trafficking_on::Float64 = 1.0
end

Base.@kwdef struct MosunDynamicState
    actTtiss::Float64 = 0.0
    Btiss::Float64 = 0.0
    TDBc_ugperkg::Float64 = 0.0
    actTpb::Float64 = 0.0
    restTtiss::Float64 = 0.0
    TDBp_ugperkg::Float64 = 0.0
    restTpb::Float64 = 0.0
    Bpb::Float64 = 0.0
    BAFF::Float64 = 0.0
    act0Tpb::Float64 = 0.0
    act0Ttiss::Float64 = 0.0
    injection_effect::Float64 = 0.0
    Btiss2::Float64 = 0.0
    act0Ttiss2::Float64 = 0.0
    restTtiss2::Float64 = 0.0
    actTtiss2::Float64 = 0.0
    RTXc_ugperkg::Float64 = 0.0
    RTXp_ugperkg::Float64 = 0.0
    drug_effect::Float64 = 0.0
    B1920tiss3::Float64 = 0.0
    B19no20tiss3::Float64 = 0.0
    restTtiss3::Float64 = 0.0
    act0Ttiss3::Float64 = 0.0
    actTtiss3::Float64 = 0.0
    Blinc_ug::Float64 = 0.0
    restTtumor::Float64 = 0.0
    actTtumor::Float64 = 0.0
    Btumor::Float64 = 0.0
    act0Ttumor::Float64 = 0.0
    TDBsc_ugperkg::Float64 = 0.0
    TDBc_ugperml_AUC::Float64 = 0.0
    IL6pb::Float64 = 0.0
    IL6tiss::Float64 = 0.0
    IL6tiss2::Float64 = 0.0
    IL6tiss3::Float64 = 0.0
    IL6tumor::Float64 = 0.0
end

Base.@kwdef mutable struct MosunObservablesCache
    TDBc_ugperml::Float64 = 0.0
    RTXc_ugperml::Float64 = 0.0
    Blinc_ngperml::Float64 = 0.0
    restTpb_perml::Float64 = 0.0
    restTtiss_perml::Float64 = 0.0
    restTtiss2_perml::Float64 = 0.0
    restTtiss3_perml::Float64 = 0.0
    restTtumor_perml::Float64 = 0.0
    act0Tpb_perml::Float64 = 0.0
    act0Ttiss_perml::Float64 = 0.0
    act0Ttiss2_perml::Float64 = 0.0
    act0Ttiss3_perml::Float64 = 0.0
    act0Ttumor_perml::Float64 = 0.0
    actTpb_perml::Float64 = 0.0
    actTtiss_perml::Float64 = 0.0
    actTtiss2_perml::Float64 = 0.0
    actTtiss3_perml::Float64 = 0.0
    actTtumor_perml::Float64 = 0.0
    B19tiss3::Float64 = 0.0
    Bpb_perml::Float64 = 0.0
    Btiss_perml::Float64 = 0.0
    Btiss2_perml::Float64 = 0.0
    B1920tiss3_perml::Float64 = 0.0
    B19no20tiss3_perml::Float64 = 0.0
    Btumor_perml::Float64 = 0.0
    BTrratio_pb::Float64 = 0.0
    BTrratio_tiss::Float64 = 0.0
    BTrratio_tiss2::Float64 = 0.0
    B1920Trratio_tiss3::Float64 = 0.0
    BTrratio_tumor::Float64 = 0.0
    TaBratio_pb::Float64 = 0.0
    TaBratio_tiss::Float64 = 0.0
    TaBratio_tiss2::Float64 = 0.0
    TaB1920ratio_tiss3::Float64 = 0.0
    TaBratio_tumor::Float64 = 0.0
    totTtiss::Float64 = 0.0
    totTtiss2::Float64 = 0.0
    totTtiss3::Float64 = 0.0
    totTtumor::Float64 = 0.0
    Baffconsumption::Float64 = 0.0
    IL6combo::Float64 = 0.0
    TDBt_ugperml::Float64 = 0.0
    TDBt2_ugperml::Float64 = 0.0
    TDBt3_ugperml::Float64 = 0.0
    TDBtumor_ugperml::Float64 = 0.0
    RTXt_ugperml::Float64 = 0.0
    RTXt2_ugperml::Float64 = 0.0
    RTXt3_ugperml::Float64 = 0.0
    RTXtumor_ugperml::Float64 = 0.0
    Blint_ngperml::Float64 = 0.0
    Blint2_ngperml::Float64 = 0.0
    Blint3_ngperml::Float64 = 0.0
    Blintumor_ngperml::Float64 = 0.0
    totTpb_perml::Float64 = 0.0
    totTtiss_perml::Float64 = 0.0
    totTtiss2_perml::Float64 = 0.0
    totTtiss3_perml::Float64 = 0.0
    totTtumor_perml::Float64 = 0.0
    B19tiss3_perml::Float64 = 0.0
    B19Trratio_tiss3::Float64 = 0.0
    TaB19ratio_tiss3::Float64 = 0.0
    Bpb_norm::Float64 = 0.0
    drugTpbact::Float64 = 0.0
    BlinTpbact::Float64 = 0.0
    drugBpbkill::Float64 = 0.0
    BlinBpbkill::Float64 = 0.0
    drugTtissact::Float64 = 0.0
    drugBtisskill::Float64 = 0.0
    drugTtissact2::Float64 = 0.0
    drugBtisskill2::Float64 = 0.0
    drugTtissact3::Float64 = 0.0
    drugBtisskill3::Float64 = 0.0
    drugTtumoract::Float64 = 0.0
    drugBtumorkill::Float64 = 0.0
    BlinTtissact::Float64 = 0.0
    BlinBtisskill::Float64 = 0.0
    BlinTtissact2::Float64 = 0.0
    BlinBtisskill2::Float64 = 0.0
    BlinTtumoract::Float64 = 0.0
    BlinBtumorkill::Float64 = 0.0
    Tafraction_pb::Float64 = 0.0
    Tafraction_tiss::Float64 = 0.0
    BTtotRatio_tiss::Float64 = 0.0
    Tafraction_tiss2::Float64 = 0.0
    BTtotRatio_tiss2::Float64 = 0.0
    Tafraction_tiss3::Float64 = 0.0
    Tafraction_tumor::Float64 = 0.0
    BlinTtissact3::Float64 = 0.0
    BlinBtisskill3::Float64 = 0.0
end

struct MosunRegimenEvent{T<:Real}
    target::Symbol
    time::Float64
    amount::T
    rate::T
end

function MosunRegimenEvent(; target::Symbol, time::Real, amount::Real = 0.0, rate::Real = 0.0)
    T = promote_type(typeof(amount), typeof(rate))
    return MosunRegimenEvent{T}(target, Float64(time), convert(T, amount), convert(T, rate))
end

struct MosunRegimen{T<:Real}
    events::Vector{MosunRegimenEvent{T}}
end

MosunRegimen() = MosunRegimen{Float64}(MosunRegimenEvent{Float64}[])
MosunRegimen(events::Vector{MosunRegimenEvent{T}}) where {T<:Real} = MosunRegimen{T}(events)
MosunRegimen(; events::Vector{MosunRegimenEvent{T}}) where {T<:Real} = MosunRegimen{T}(events)
function MosunRegimen(events::AbstractVector{<:MosunRegimenEvent})
    isempty(events) && return MosunRegimen()
    T = promote_type(map(ev -> typeof(ev.amount), events)..., map(ev -> typeof(ev.rate), events)...)
    typed_events = MosunRegimenEvent{T}[MosunRegimenEvent{T}(ev.target, ev.time, convert(T, ev.amount), convert(T, ev.rate)) for ev in events]
    return MosunRegimen{T}(typed_events)
end
MosunRegimen(; events::AbstractVector{<:MosunRegimenEvent}) = MosunRegimen(events)

Base.@kwdef mutable struct MosunProblemContext
    params::MosunParams
    cache::MosunObservablesCache = MosunObservablesCache()
    active_rates = zeros(Float64, DYNAMIC_STATE_COUNT)
end

Base.@kwdef struct MosunBuiltProblem
    prob
    ctx::MosunProblemContext
    initial_u
    callback
    tstops::Vector{Float64}
    d_discontinuities::Vector{Float64}
    event_map
    saveat
    regimen
    callback_mode::Symbol
end

Base.@kwdef struct MosunVectorBuiltProblem
    prob
    initial_u
    callback
    tstops::Vector{Float64}
    d_discontinuities::Vector{Float64}
    event_map
    saveat
    regimen
    callback_mode::Symbol
end

default_params() = MosunParams()
zero_observables_cache() = MosunObservablesCache()
has_parameter(name) = Symbol(name) in PARAMETER_NAMES

function parameter_index(name)
    sym = Symbol(name)
    idx = findfirst(==(sym), PARAMETER_NAMES)
    idx === nothing && throw(ArgumentError("Unknown parameter: $(name)"))
    return idx
end

default_param_vector(::Type{T} = Float64) where {T<:Real} = T.(pack_params(default_params()))

function set_param!(p::MosunParams, name, value)
    sym = Symbol(name)
    sym in PARAMETER_NAMES || throw(ArgumentError("Unknown parameter: $(name)"))
    setproperty!(p, sym, Float64(value))
    return p
end

function params_from_named_values(names, values; base::MosunParams = default_params(), ignore_unknown::Bool = false)
    length(names) == length(values) || throw(ArgumentError("names and values must have equal length"))
    p = deepcopy(base)
    for (name, value) in zip(names, values)
        if has_parameter(name)
            set_param!(p, name, value)
        elseif !ignore_unknown
            throw(ArgumentError("Unknown parameter: $(name)"))
        end
    end
    return p
end

function params_from_dict(values; base::MosunParams = default_params(), ignore_unknown::Bool = false)
    p = deepcopy(base)
    for (name, value) in pairs(values)
        if has_parameter(name)
            set_param!(p, name, value)
        elseif !ignore_unknown
            throw(ArgumentError("Unknown parameter: $(name)"))
        end
    end
    return p
end

function bolus_regimen(target::Symbol, dose_map)
    T = Float64
    times = sort(collect(keys(dose_map)))
    for t in times
        T = promote_type(T, typeof(dose_map[t]))
    end
    events = MosunRegimenEvent{T}[]
    for t in times
        amt = convert(T, dose_map[t])
        iszero(amt) && continue
        push!(events, MosunRegimenEvent{T}(target, Float64(t), amt, zero(T)))
    end
    return MosunRegimen(events = events)
end

function initial_state(p::MosunParams)
    actTtiss = 0.0
    Btiss = Float64(real(p.KBp * p.Bpbo_perml * p.Vtissue))
    TDBc_ugperkg = 0.0
    actTpb = 0.0
    restTtiss = Float64(real(p.KTrp * p.Trpbo_perml * p.Vtissue))
    TDBp_ugperkg = 0.0
    restTpb = Float64(real((1 - p.depleteTpb) * p.Trpbo_perml * p.Vpb))
    Bpb = Float64(real((1 - p.depleteBpb) * p.Bpbo_perml * p.Vpb))
    BAFF = Float64(real(p.BAFFo))
    act0Tpb = 0.0
    act0Ttiss = 0.0
    injection_effect = 0.0
    Btiss2 = Float64(real(p.KBp2 * p.Bpbo_perml * p.Vtissue2))
    act0Ttiss2 = 0.0
    restTtiss2 = Float64(real(p.KTrp2 * p.Trpbo_perml * p.Vtissue2))
    actTtiss2 = 0.0
    RTXc_ugperkg = 0.0
    RTXp_ugperkg = 0.0
    drug_effect = 0.0
    B1920tiss3 = Float64(real(p.Bpbref_perml * p.KBp3 * p.Vtissue3))
    B19no20tiss3 = Float64(real(p.Bpbref_perml * p.KBp3 * p.B19no20_B1920_ratio * p.Vtissue3))
    restTtiss3 = Float64(real(p.KTrp3 * p.Trpbo_perml * p.Vtissue3))
    act0Ttiss3 = 0.0
    actTtiss3 = 0.0
    Blinc_ug = 0.0
    restTtumor = Float64(real(p.KTrptumor * p.Trpbo_perml * p.Vtumor))
    actTtumor = 0.0
    Btumor = Float64(real(p.KBptumor * p.Bpbo_perml * p.Vtumor))
    act0Ttumor = 0.0
    TDBsc_ugperkg = 0.0
    TDBc_ugperml_AUC = 0.0
    IL6pb = 0.0
    IL6tiss = 0.0
    IL6tiss2 = 0.0
    IL6tiss3 = 0.0
    IL6tumor = 0.0
    return MosunDynamicState(
        actTtiss = actTtiss,
        Btiss = Btiss,
        TDBc_ugperkg = TDBc_ugperkg,
        actTpb = actTpb,
        restTtiss = restTtiss,
        TDBp_ugperkg = TDBp_ugperkg,
        restTpb = restTpb,
        Bpb = Bpb,
        BAFF = BAFF,
        act0Tpb = act0Tpb,
        act0Ttiss = act0Ttiss,
        injection_effect = injection_effect,
        Btiss2 = Btiss2,
        act0Ttiss2 = act0Ttiss2,
        restTtiss2 = restTtiss2,
        actTtiss2 = actTtiss2,
        RTXc_ugperkg = RTXc_ugperkg,
        RTXp_ugperkg = RTXp_ugperkg,
        drug_effect = drug_effect,
        B1920tiss3 = B1920tiss3,
        B19no20tiss3 = B19no20tiss3,
        restTtiss3 = restTtiss3,
        act0Ttiss3 = act0Ttiss3,
        actTtiss3 = actTtiss3,
        Blinc_ug = Blinc_ug,
        restTtumor = restTtumor,
        actTtumor = actTtumor,
        Btumor = Btumor,
        act0Ttumor = act0Ttumor,
        TDBsc_ugperkg = TDBsc_ugperkg,
        TDBc_ugperml_AUC = TDBc_ugperml_AUC,
        IL6pb = IL6pb,
        IL6tiss = IL6tiss,
        IL6tiss2 = IL6tiss2,
        IL6tiss3 = IL6tiss3,
        IL6tumor = IL6tumor,
    )
end

function initial_state_vector(p::AbstractVector{<:Real})
    KBp = p[27]
    Bpbo_perml = p[25]
    Vtissue = p[30]
    KTrp = p[21]
    Trpbo_perml = p[22]
    depleteTpb = p[64]
    Vpb = p[29]
    depleteBpb = p[65]
    BAFFo = p[48]
    KBp2 = p[52]
    Vtissue2 = p[53]
    KTrp2 = p[34]
    Bpbref_perml = p[43]
    KBp3 = p[76]
    Vtissue3 = p[71]
    B19no20_B1920_ratio = p[77]
    KTrp3 = p[72]
    KTrptumor = p[99]
    Vtumor = p[97]
    KBptumor = p[98]
    T = eltype(p)
    return T[
        zero(T),
        real(KBp * Bpbo_perml * Vtissue),
        zero(T),
        zero(T),
        real(KTrp * Trpbo_perml * Vtissue),
        zero(T),
        real((one(T) - depleteTpb) * Trpbo_perml * Vpb),
        real((one(T) - depleteBpb) * Bpbo_perml * Vpb),
        real(BAFFo),
        zero(T),
        zero(T),
        zero(T),
        real(KBp2 * Bpbo_perml * Vtissue2),
        zero(T),
        real(KTrp2 * Trpbo_perml * Vtissue2),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        real(Bpbref_perml * KBp3 * Vtissue3),
        real(Bpbref_perml * KBp3 * B19no20_B1920_ratio * Vtissue3),
        real(KTrp3 * Trpbo_perml * Vtissue3),
        zero(T),
        zero(T),
        zero(T),
        real(KTrptumor * Trpbo_perml * Vtumor),
        zero(T),
        real(KBptumor * Bpbo_perml * Vtumor),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
    ]
end

function pack_state(s::MosunDynamicState)
    return Float64[
        s.actTtiss,
        s.Btiss,
        s.TDBc_ugperkg,
        s.actTpb,
        s.restTtiss,
        s.TDBp_ugperkg,
        s.restTpb,
        s.Bpb,
        s.BAFF,
        s.act0Tpb,
        s.act0Ttiss,
        s.injection_effect,
        s.Btiss2,
        s.act0Ttiss2,
        s.restTtiss2,
        s.actTtiss2,
        s.RTXc_ugperkg,
        s.RTXp_ugperkg,
        s.drug_effect,
        s.B1920tiss3,
        s.B19no20tiss3,
        s.restTtiss3,
        s.act0Ttiss3,
        s.actTtiss3,
        s.Blinc_ug,
        s.restTtumor,
        s.actTtumor,
        s.Btumor,
        s.act0Ttumor,
        s.TDBsc_ugperkg,
        s.TDBc_ugperml_AUC,
        s.IL6pb,
        s.IL6tiss,
        s.IL6tiss2,
        s.IL6tiss3,
        s.IL6tumor,
    ]
end

function pack_params(p::MosunParams)
    return Float64[
        p.VmB,
        p.KmTB_kill,
        p.KdrugB,
        p.kBapop,
        p.kBprolif,
        p.kTprolif,
        p.kBkill,
        p.KdrugactT,
        p.VmT,
        p.KmBT_act,
        p.kTaexit,
        p.kTact,
        p.fTadeact,
        p.fTap,
        p.Cl_tdb,
        p.Cld_tdb,
        p.Vc_tdb,
        p.Vp_tdb,
        p.kTgen,
        p.kTaapop,
        p.KTrp,
        p.Trpbo_perml,
        p.fTaprolif,
        p.fBprolif,
        p.Bpbo_perml,
        p.fBexit,
        p.KBp,
        p.Kp,
        p.Vpb,
        p.Vtissue,
        p.fTrapop,
        p.Trpbref_perml,
        p.nkill,
        p.KTrp2,
        p.Vm_tdb,
        p.Km_tdb,
        p.BW,
        p.act0on,
        p.kIL6prod,
        p.fa0,
        p.fAICD,
        p.kBAFFprod,
        p.Bpbref_perml,
        p.thBAFF,
        p.thalfIL6,
        p.fTa0deact,
        p.fTa0apop,
        p.BAFFo,
        p.fBAFFo,
        p.tinjhalf,
        p.finj,
        p.KBp2,
        p.Vtissue2,
        p.Kp2,
        p.tissue2on,
        p.Vc_rtx,
        p.Vp_rtx,
        p.Cl_rtx,
        p.Cld_rtx,
        p.Vm_rtx,
        p.Km_rtx,
        p.Kmkill_rtx,
        p.fTgenbl,
        p.depleteTpb,
        p.depleteBpb,
        p.kTrexit,
        p.PKflag,
        p.VPid,
        p.end_time,
        p.fdrug,
        p.Vtissue3,
        p.KTrp3,
        p.tissue3on,
        p.Kp3,
        p.kBtiss3exit,
        p.KBp3,
        p.B19no20_B1920_ratio,
        p.fvalidation,
        p.fapop_v24,
        p.fBtissue3_v1,
        p.kBmat_kBapop_ratio,
        p.ndrugactT,
        p.Cl_blin,
        p.Vz_blin,
        p.KdrugactT_blin,
        p.ndrugactT_blin,
        p.KmTB_kill_blin,
        p.KdrugB_blin,
        p.nkill_blin,
        p.KmBT_act_blin,
        p.kBapop_cll,
        p.kBgen_cll,
        p.fBkill,
        p.fTact,
        p.fKmTB_kill,
        p.Kptumor,
        p.Vtumor,
        p.KBptumor,
        p.KTrptumor,
        p.tumor_on,
        p.IL6_tiss_contribution,
        p.kBtumorprolif,
        p.S,
        p.tissue1on,
        p.kabs_TDB,
        p.fbio_TDB,
        p.Bcell_tumor_trafficking_on,
    ]
end

function unpack_state(u::AbstractVector{<:Real})
    return MosunDynamicState(
        actTtiss = Float64(u[1]),
        Btiss = Float64(u[2]),
        TDBc_ugperkg = Float64(u[3]),
        actTpb = Float64(u[4]),
        restTtiss = Float64(u[5]),
        TDBp_ugperkg = Float64(u[6]),
        restTpb = Float64(u[7]),
        Bpb = Float64(u[8]),
        BAFF = Float64(u[9]),
        act0Tpb = Float64(u[10]),
        act0Ttiss = Float64(u[11]),
        injection_effect = Float64(u[12]),
        Btiss2 = Float64(u[13]),
        act0Ttiss2 = Float64(u[14]),
        restTtiss2 = Float64(u[15]),
        actTtiss2 = Float64(u[16]),
        RTXc_ugperkg = Float64(u[17]),
        RTXp_ugperkg = Float64(u[18]),
        drug_effect = Float64(u[19]),
        B1920tiss3 = Float64(u[20]),
        B19no20tiss3 = Float64(u[21]),
        restTtiss3 = Float64(u[22]),
        act0Ttiss3 = Float64(u[23]),
        actTtiss3 = Float64(u[24]),
        Blinc_ug = Float64(u[25]),
        restTtumor = Float64(u[26]),
        actTtumor = Float64(u[27]),
        Btumor = Float64(u[28]),
        act0Ttumor = Float64(u[29]),
        TDBsc_ugperkg = Float64(u[30]),
        TDBc_ugperml_AUC = Float64(u[31]),
        IL6pb = Float64(u[32]),
        IL6tiss = Float64(u[33]),
        IL6tiss2 = Float64(u[34]),
        IL6tiss3 = Float64(u[35]),
        IL6tumor = Float64(u[36]),
    )
end

function dynamic_state_index(name::Symbol)
    if name === :actTtiss
        return 1
    elseif name === :Btiss
        return 2
    elseif name === :TDBc_ugperkg
        return 3
    elseif name === :actTpb
        return 4
    elseif name === :restTtiss
        return 5
    elseif name === :TDBp_ugperkg
        return 6
    elseif name === :restTpb
        return 7
    elseif name === :Bpb
        return 8
    elseif name === :BAFF
        return 9
    elseif name === :act0Tpb
        return 10
    elseif name === :act0Ttiss
        return 11
    elseif name === :injection_effect
        return 12
    elseif name === :Btiss2
        return 13
    elseif name === :act0Ttiss2
        return 14
    elseif name === :restTtiss2
        return 15
    elseif name === :actTtiss2
        return 16
    elseif name === :RTXc_ugperkg
        return 17
    elseif name === :RTXp_ugperkg
        return 18
    elseif name === :drug_effect
        return 19
    elseif name === :B1920tiss3
        return 20
    elseif name === :B19no20tiss3
        return 21
    elseif name === :restTtiss3
        return 22
    elseif name === :act0Ttiss3
        return 23
    elseif name === :actTtiss3
        return 24
    elseif name === :Blinc_ug
        return 25
    elseif name === :restTtumor
        return 26
    elseif name === :actTtumor
        return 27
    elseif name === :Btumor
        return 28
    elseif name === :act0Ttumor
        return 29
    elseif name === :TDBsc_ugperkg
        return 30
    elseif name === :TDBc_ugperml_AUC
        return 31
    elseif name === :IL6pb
        return 32
    elseif name === :IL6tiss
        return 33
    elseif name === :IL6tiss2
        return 34
    elseif name === :IL6tiss3
        return 35
    elseif name === :IL6tumor
        return 36
    end
    throw(ArgumentError("Unknown name: $(name)"))
end

function dynamic_state_value(u::AbstractVector{<:Real}, name::Symbol)
    if name === :actTtiss
        return Float64(u[1])
    elseif name === :Btiss
        return Float64(u[2])
    elseif name === :TDBc_ugperkg
        return Float64(u[3])
    elseif name === :actTpb
        return Float64(u[4])
    elseif name === :restTtiss
        return Float64(u[5])
    elseif name === :TDBp_ugperkg
        return Float64(u[6])
    elseif name === :restTpb
        return Float64(u[7])
    elseif name === :Bpb
        return Float64(u[8])
    elseif name === :BAFF
        return Float64(u[9])
    elseif name === :act0Tpb
        return Float64(u[10])
    elseif name === :act0Ttiss
        return Float64(u[11])
    elseif name === :injection_effect
        return Float64(u[12])
    elseif name === :Btiss2
        return Float64(u[13])
    elseif name === :act0Ttiss2
        return Float64(u[14])
    elseif name === :restTtiss2
        return Float64(u[15])
    elseif name === :actTtiss2
        return Float64(u[16])
    elseif name === :RTXc_ugperkg
        return Float64(u[17])
    elseif name === :RTXp_ugperkg
        return Float64(u[18])
    elseif name === :drug_effect
        return Float64(u[19])
    elseif name === :B1920tiss3
        return Float64(u[20])
    elseif name === :B19no20tiss3
        return Float64(u[21])
    elseif name === :restTtiss3
        return Float64(u[22])
    elseif name === :act0Ttiss3
        return Float64(u[23])
    elseif name === :actTtiss3
        return Float64(u[24])
    elseif name === :Blinc_ug
        return Float64(u[25])
    elseif name === :restTtumor
        return Float64(u[26])
    elseif name === :actTtumor
        return Float64(u[27])
    elseif name === :Btumor
        return Float64(u[28])
    elseif name === :act0Ttumor
        return Float64(u[29])
    elseif name === :TDBsc_ugperkg
        return Float64(u[30])
    elseif name === :TDBc_ugperml_AUC
        return Float64(u[31])
    elseif name === :IL6pb
        return Float64(u[32])
    elseif name === :IL6tiss
        return Float64(u[33])
    elseif name === :IL6tiss2
        return Float64(u[34])
    elseif name === :IL6tiss3
        return Float64(u[35])
    elseif name === :IL6tumor
        return Float64(u[36])
    end
    throw(ArgumentError("Unknown name: $(name)"))
end

function observable(cache::MosunObservablesCache, name::Symbol)
    if name === :TDBc_ugperml
        return cache.TDBc_ugperml
    elseif name === :RTXc_ugperml
        return cache.RTXc_ugperml
    elseif name === :Blinc_ngperml
        return cache.Blinc_ngperml
    elseif name === :restTpb_perml
        return cache.restTpb_perml
    elseif name === :restTtiss_perml
        return cache.restTtiss_perml
    elseif name === :restTtiss2_perml
        return cache.restTtiss2_perml
    elseif name === :restTtiss3_perml
        return cache.restTtiss3_perml
    elseif name === :restTtumor_perml
        return cache.restTtumor_perml
    elseif name === :act0Tpb_perml
        return cache.act0Tpb_perml
    elseif name === :act0Ttiss_perml
        return cache.act0Ttiss_perml
    elseif name === :act0Ttiss2_perml
        return cache.act0Ttiss2_perml
    elseif name === :act0Ttiss3_perml
        return cache.act0Ttiss3_perml
    elseif name === :act0Ttumor_perml
        return cache.act0Ttumor_perml
    elseif name === :actTpb_perml
        return cache.actTpb_perml
    elseif name === :actTtiss_perml
        return cache.actTtiss_perml
    elseif name === :actTtiss2_perml
        return cache.actTtiss2_perml
    elseif name === :actTtiss3_perml
        return cache.actTtiss3_perml
    elseif name === :actTtumor_perml
        return cache.actTtumor_perml
    elseif name === :B19tiss3
        return cache.B19tiss3
    elseif name === :Bpb_perml
        return cache.Bpb_perml
    elseif name === :Btiss_perml
        return cache.Btiss_perml
    elseif name === :Btiss2_perml
        return cache.Btiss2_perml
    elseif name === :B1920tiss3_perml
        return cache.B1920tiss3_perml
    elseif name === :B19no20tiss3_perml
        return cache.B19no20tiss3_perml
    elseif name === :Btumor_perml
        return cache.Btumor_perml
    elseif name === :BTrratio_pb
        return cache.BTrratio_pb
    elseif name === :BTrratio_tiss
        return cache.BTrratio_tiss
    elseif name === :BTrratio_tiss2
        return cache.BTrratio_tiss2
    elseif name === :B1920Trratio_tiss3
        return cache.B1920Trratio_tiss3
    elseif name === :BTrratio_tumor
        return cache.BTrratio_tumor
    elseif name === :TaBratio_pb
        return cache.TaBratio_pb
    elseif name === :TaBratio_tiss
        return cache.TaBratio_tiss
    elseif name === :TaBratio_tiss2
        return cache.TaBratio_tiss2
    elseif name === :TaB1920ratio_tiss3
        return cache.TaB1920ratio_tiss3
    elseif name === :TaBratio_tumor
        return cache.TaBratio_tumor
    elseif name === :totTtiss
        return cache.totTtiss
    elseif name === :totTtiss2
        return cache.totTtiss2
    elseif name === :totTtiss3
        return cache.totTtiss3
    elseif name === :totTtumor
        return cache.totTtumor
    elseif name === :Baffconsumption
        return cache.Baffconsumption
    elseif name === :IL6combo
        return cache.IL6combo
    elseif name === :TDBt_ugperml
        return cache.TDBt_ugperml
    elseif name === :TDBt2_ugperml
        return cache.TDBt2_ugperml
    elseif name === :TDBt3_ugperml
        return cache.TDBt3_ugperml
    elseif name === :TDBtumor_ugperml
        return cache.TDBtumor_ugperml
    elseif name === :RTXt_ugperml
        return cache.RTXt_ugperml
    elseif name === :RTXt2_ugperml
        return cache.RTXt2_ugperml
    elseif name === :RTXt3_ugperml
        return cache.RTXt3_ugperml
    elseif name === :RTXtumor_ugperml
        return cache.RTXtumor_ugperml
    elseif name === :Blint_ngperml
        return cache.Blint_ngperml
    elseif name === :Blint2_ngperml
        return cache.Blint2_ngperml
    elseif name === :Blint3_ngperml
        return cache.Blint3_ngperml
    elseif name === :Blintumor_ngperml
        return cache.Blintumor_ngperml
    elseif name === :totTpb_perml
        return cache.totTpb_perml
    elseif name === :totTtiss_perml
        return cache.totTtiss_perml
    elseif name === :totTtiss2_perml
        return cache.totTtiss2_perml
    elseif name === :totTtiss3_perml
        return cache.totTtiss3_perml
    elseif name === :totTtumor_perml
        return cache.totTtumor_perml
    elseif name === :B19tiss3_perml
        return cache.B19tiss3_perml
    elseif name === :B19Trratio_tiss3
        return cache.B19Trratio_tiss3
    elseif name === :TaB19ratio_tiss3
        return cache.TaB19ratio_tiss3
    elseif name === :Bpb_norm
        return cache.Bpb_norm
    elseif name === :drugTpbact
        return cache.drugTpbact
    elseif name === :BlinTpbact
        return cache.BlinTpbact
    elseif name === :drugBpbkill
        return cache.drugBpbkill
    elseif name === :BlinBpbkill
        return cache.BlinBpbkill
    elseif name === :drugTtissact
        return cache.drugTtissact
    elseif name === :drugBtisskill
        return cache.drugBtisskill
    elseif name === :drugTtissact2
        return cache.drugTtissact2
    elseif name === :drugBtisskill2
        return cache.drugBtisskill2
    elseif name === :drugTtissact3
        return cache.drugTtissact3
    elseif name === :drugBtisskill3
        return cache.drugBtisskill3
    elseif name === :drugTtumoract
        return cache.drugTtumoract
    elseif name === :drugBtumorkill
        return cache.drugBtumorkill
    elseif name === :BlinTtissact
        return cache.BlinTtissact
    elseif name === :BlinBtisskill
        return cache.BlinBtisskill
    elseif name === :BlinTtissact2
        return cache.BlinTtissact2
    elseif name === :BlinBtisskill2
        return cache.BlinBtisskill2
    elseif name === :BlinTtumoract
        return cache.BlinTtumoract
    elseif name === :BlinBtumorkill
        return cache.BlinBtumorkill
    elseif name === :Tafraction_pb
        return cache.Tafraction_pb
    elseif name === :Tafraction_tiss
        return cache.Tafraction_tiss
    elseif name === :BTtotRatio_tiss
        return cache.BTtotRatio_tiss
    elseif name === :Tafraction_tiss2
        return cache.Tafraction_tiss2
    elseif name === :BTtotRatio_tiss2
        return cache.BTtotRatio_tiss2
    elseif name === :Tafraction_tiss3
        return cache.Tafraction_tiss3
    elseif name === :Tafraction_tumor
        return cache.Tafraction_tumor
    elseif name === :BlinTtissact3
        return cache.BlinTtissact3
    elseif name === :BlinBtisskill3
        return cache.BlinBtisskill3
    end
    throw(ArgumentError("Unknown name: $(name)"))
end

observable(cache::MosunObservablesCache, ::Val{:TDBc_ugperml}) = cache.TDBc_ugperml
observable(cache::MosunObservablesCache, ::Val{:RTXc_ugperml}) = cache.RTXc_ugperml
observable(cache::MosunObservablesCache, ::Val{:Blinc_ngperml}) = cache.Blinc_ngperml
observable(cache::MosunObservablesCache, ::Val{:restTpb_perml}) = cache.restTpb_perml
observable(cache::MosunObservablesCache, ::Val{:restTtiss_perml}) = cache.restTtiss_perml
observable(cache::MosunObservablesCache, ::Val{:restTtiss2_perml}) = cache.restTtiss2_perml
observable(cache::MosunObservablesCache, ::Val{:restTtiss3_perml}) = cache.restTtiss3_perml
observable(cache::MosunObservablesCache, ::Val{:restTtumor_perml}) = cache.restTtumor_perml
observable(cache::MosunObservablesCache, ::Val{:act0Tpb_perml}) = cache.act0Tpb_perml
observable(cache::MosunObservablesCache, ::Val{:act0Ttiss_perml}) = cache.act0Ttiss_perml
observable(cache::MosunObservablesCache, ::Val{:act0Ttiss2_perml}) = cache.act0Ttiss2_perml
observable(cache::MosunObservablesCache, ::Val{:act0Ttiss3_perml}) = cache.act0Ttiss3_perml
observable(cache::MosunObservablesCache, ::Val{:act0Ttumor_perml}) = cache.act0Ttumor_perml
observable(cache::MosunObservablesCache, ::Val{:actTpb_perml}) = cache.actTpb_perml
observable(cache::MosunObservablesCache, ::Val{:actTtiss_perml}) = cache.actTtiss_perml
observable(cache::MosunObservablesCache, ::Val{:actTtiss2_perml}) = cache.actTtiss2_perml
observable(cache::MosunObservablesCache, ::Val{:actTtiss3_perml}) = cache.actTtiss3_perml
observable(cache::MosunObservablesCache, ::Val{:actTtumor_perml}) = cache.actTtumor_perml
observable(cache::MosunObservablesCache, ::Val{:B19tiss3}) = cache.B19tiss3
observable(cache::MosunObservablesCache, ::Val{:Bpb_perml}) = cache.Bpb_perml
observable(cache::MosunObservablesCache, ::Val{:Btiss_perml}) = cache.Btiss_perml
observable(cache::MosunObservablesCache, ::Val{:Btiss2_perml}) = cache.Btiss2_perml
observable(cache::MosunObservablesCache, ::Val{:B1920tiss3_perml}) = cache.B1920tiss3_perml
observable(cache::MosunObservablesCache, ::Val{:B19no20tiss3_perml}) = cache.B19no20tiss3_perml
observable(cache::MosunObservablesCache, ::Val{:Btumor_perml}) = cache.Btumor_perml
observable(cache::MosunObservablesCache, ::Val{:BTrratio_pb}) = cache.BTrratio_pb
observable(cache::MosunObservablesCache, ::Val{:BTrratio_tiss}) = cache.BTrratio_tiss
observable(cache::MosunObservablesCache, ::Val{:BTrratio_tiss2}) = cache.BTrratio_tiss2
observable(cache::MosunObservablesCache, ::Val{:B1920Trratio_tiss3}) = cache.B1920Trratio_tiss3
observable(cache::MosunObservablesCache, ::Val{:BTrratio_tumor}) = cache.BTrratio_tumor
observable(cache::MosunObservablesCache, ::Val{:TaBratio_pb}) = cache.TaBratio_pb
observable(cache::MosunObservablesCache, ::Val{:TaBratio_tiss}) = cache.TaBratio_tiss
observable(cache::MosunObservablesCache, ::Val{:TaBratio_tiss2}) = cache.TaBratio_tiss2
observable(cache::MosunObservablesCache, ::Val{:TaB1920ratio_tiss3}) = cache.TaB1920ratio_tiss3
observable(cache::MosunObservablesCache, ::Val{:TaBratio_tumor}) = cache.TaBratio_tumor
observable(cache::MosunObservablesCache, ::Val{:totTtiss}) = cache.totTtiss
observable(cache::MosunObservablesCache, ::Val{:totTtiss2}) = cache.totTtiss2
observable(cache::MosunObservablesCache, ::Val{:totTtiss3}) = cache.totTtiss3
observable(cache::MosunObservablesCache, ::Val{:totTtumor}) = cache.totTtumor
observable(cache::MosunObservablesCache, ::Val{:Baffconsumption}) = cache.Baffconsumption
observable(cache::MosunObservablesCache, ::Val{:IL6combo}) = cache.IL6combo
observable(cache::MosunObservablesCache, ::Val{:TDBt_ugperml}) = cache.TDBt_ugperml
observable(cache::MosunObservablesCache, ::Val{:TDBt2_ugperml}) = cache.TDBt2_ugperml
observable(cache::MosunObservablesCache, ::Val{:TDBt3_ugperml}) = cache.TDBt3_ugperml
observable(cache::MosunObservablesCache, ::Val{:TDBtumor_ugperml}) = cache.TDBtumor_ugperml
observable(cache::MosunObservablesCache, ::Val{:RTXt_ugperml}) = cache.RTXt_ugperml
observable(cache::MosunObservablesCache, ::Val{:RTXt2_ugperml}) = cache.RTXt2_ugperml
observable(cache::MosunObservablesCache, ::Val{:RTXt3_ugperml}) = cache.RTXt3_ugperml
observable(cache::MosunObservablesCache, ::Val{:RTXtumor_ugperml}) = cache.RTXtumor_ugperml
observable(cache::MosunObservablesCache, ::Val{:Blint_ngperml}) = cache.Blint_ngperml
observable(cache::MosunObservablesCache, ::Val{:Blint2_ngperml}) = cache.Blint2_ngperml
observable(cache::MosunObservablesCache, ::Val{:Blint3_ngperml}) = cache.Blint3_ngperml
observable(cache::MosunObservablesCache, ::Val{:Blintumor_ngperml}) = cache.Blintumor_ngperml
observable(cache::MosunObservablesCache, ::Val{:totTpb_perml}) = cache.totTpb_perml
observable(cache::MosunObservablesCache, ::Val{:totTtiss_perml}) = cache.totTtiss_perml
observable(cache::MosunObservablesCache, ::Val{:totTtiss2_perml}) = cache.totTtiss2_perml
observable(cache::MosunObservablesCache, ::Val{:totTtiss3_perml}) = cache.totTtiss3_perml
observable(cache::MosunObservablesCache, ::Val{:totTtumor_perml}) = cache.totTtumor_perml
observable(cache::MosunObservablesCache, ::Val{:B19tiss3_perml}) = cache.B19tiss3_perml
observable(cache::MosunObservablesCache, ::Val{:B19Trratio_tiss3}) = cache.B19Trratio_tiss3
observable(cache::MosunObservablesCache, ::Val{:TaB19ratio_tiss3}) = cache.TaB19ratio_tiss3
observable(cache::MosunObservablesCache, ::Val{:Bpb_norm}) = cache.Bpb_norm
observable(cache::MosunObservablesCache, ::Val{:drugTpbact}) = cache.drugTpbact
observable(cache::MosunObservablesCache, ::Val{:BlinTpbact}) = cache.BlinTpbact
observable(cache::MosunObservablesCache, ::Val{:drugBpbkill}) = cache.drugBpbkill
observable(cache::MosunObservablesCache, ::Val{:BlinBpbkill}) = cache.BlinBpbkill
observable(cache::MosunObservablesCache, ::Val{:drugTtissact}) = cache.drugTtissact
observable(cache::MosunObservablesCache, ::Val{:drugBtisskill}) = cache.drugBtisskill
observable(cache::MosunObservablesCache, ::Val{:drugTtissact2}) = cache.drugTtissact2
observable(cache::MosunObservablesCache, ::Val{:drugBtisskill2}) = cache.drugBtisskill2
observable(cache::MosunObservablesCache, ::Val{:drugTtissact3}) = cache.drugTtissact3
observable(cache::MosunObservablesCache, ::Val{:drugBtisskill3}) = cache.drugBtisskill3
observable(cache::MosunObservablesCache, ::Val{:drugTtumoract}) = cache.drugTtumoract
observable(cache::MosunObservablesCache, ::Val{:drugBtumorkill}) = cache.drugBtumorkill
observable(cache::MosunObservablesCache, ::Val{:BlinTtissact}) = cache.BlinTtissact
observable(cache::MosunObservablesCache, ::Val{:BlinBtisskill}) = cache.BlinBtisskill
observable(cache::MosunObservablesCache, ::Val{:BlinTtissact2}) = cache.BlinTtissact2
observable(cache::MosunObservablesCache, ::Val{:BlinBtisskill2}) = cache.BlinBtisskill2
observable(cache::MosunObservablesCache, ::Val{:BlinTtumoract}) = cache.BlinTtumoract
observable(cache::MosunObservablesCache, ::Val{:BlinBtumorkill}) = cache.BlinBtumorkill
observable(cache::MosunObservablesCache, ::Val{:Tafraction_pb}) = cache.Tafraction_pb
observable(cache::MosunObservablesCache, ::Val{:Tafraction_tiss}) = cache.Tafraction_tiss
observable(cache::MosunObservablesCache, ::Val{:BTtotRatio_tiss}) = cache.BTtotRatio_tiss
observable(cache::MosunObservablesCache, ::Val{:Tafraction_tiss2}) = cache.Tafraction_tiss2
observable(cache::MosunObservablesCache, ::Val{:BTtotRatio_tiss2}) = cache.BTtotRatio_tiss2
observable(cache::MosunObservablesCache, ::Val{:Tafraction_tiss3}) = cache.Tafraction_tiss3
observable(cache::MosunObservablesCache, ::Val{:Tafraction_tumor}) = cache.Tafraction_tumor
observable(cache::MosunObservablesCache, ::Val{:BlinTtissact3}) = cache.BlinTtissact3
observable(cache::MosunObservablesCache, ::Val{:BlinBtisskill3}) = cache.BlinBtisskill3

dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:actTtiss}) = Float64(u[1])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:Btiss}) = Float64(u[2])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:TDBc_ugperkg}) = Float64(u[3])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:actTpb}) = Float64(u[4])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:restTtiss}) = Float64(u[5])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:TDBp_ugperkg}) = Float64(u[6])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:restTpb}) = Float64(u[7])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:Bpb}) = Float64(u[8])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:BAFF}) = Float64(u[9])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:act0Tpb}) = Float64(u[10])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:act0Ttiss}) = Float64(u[11])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:injection_effect}) = Float64(u[12])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:Btiss2}) = Float64(u[13])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:act0Ttiss2}) = Float64(u[14])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:restTtiss2}) = Float64(u[15])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:actTtiss2}) = Float64(u[16])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:RTXc_ugperkg}) = Float64(u[17])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:RTXp_ugperkg}) = Float64(u[18])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:drug_effect}) = Float64(u[19])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:B1920tiss3}) = Float64(u[20])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:B19no20tiss3}) = Float64(u[21])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:restTtiss3}) = Float64(u[22])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:act0Ttiss3}) = Float64(u[23])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:actTtiss3}) = Float64(u[24])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:Blinc_ug}) = Float64(u[25])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:restTtumor}) = Float64(u[26])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:actTtumor}) = Float64(u[27])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:Btumor}) = Float64(u[28])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:act0Ttumor}) = Float64(u[29])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:TDBsc_ugperkg}) = Float64(u[30])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:TDBc_ugperml_AUC}) = Float64(u[31])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:IL6pb}) = Float64(u[32])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:IL6tiss}) = Float64(u[33])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:IL6tiss2}) = Float64(u[34])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:IL6tiss3}) = Float64(u[35])
dynamic_state_value(u::AbstractVector{<:Real}, ::Val{:IL6tumor}) = Float64(u[36])

function state_or_observable(u::AbstractVector{<:Real}, cache::MosunObservablesCache, name::Symbol)
    if name in DYNAMIC_STATE_NAMES
        return dynamic_state_value(u, name)
    elseif name in OBSERVABLE_NAMES
        return observable(cache, name)
    end
    throw(ArgumentError("Unknown state or observable: $(name)"))
end

function update_observables!(cache::MosunObservablesCache, u::AbstractVector{<:Real}, p::MosunParams, t)
    actTtiss = u[1]
    Btiss = u[2]
    TDBc_ugperkg = u[3]
    actTpb = u[4]
    restTtiss = u[5]
    TDBp_ugperkg = u[6]
    restTpb = u[7]
    Bpb = u[8]
    BAFF = u[9]
    act0Tpb = u[10]
    act0Ttiss = u[11]
    injection_effect = u[12]
    Btiss2 = u[13]
    act0Ttiss2 = u[14]
    restTtiss2 = u[15]
    actTtiss2 = u[16]
    RTXc_ugperkg = u[17]
    RTXp_ugperkg = u[18]
    drug_effect = u[19]
    B1920tiss3 = u[20]
    B19no20tiss3 = u[21]
    restTtiss3 = u[22]
    act0Ttiss3 = u[23]
    actTtiss3 = u[24]
    Blinc_ug = u[25]
    restTtumor = u[26]
    actTtumor = u[27]
    Btumor = u[28]
    act0Ttumor = u[29]
    TDBsc_ugperkg = u[30]
    TDBc_ugperml_AUC = u[31]
    IL6pb = u[32]
    IL6tiss = u[33]
    IL6tiss2 = u[34]
    IL6tiss3 = u[35]
    IL6tumor = u[36]
    cache.TDBc_ugperml = Float64(real(tdb_central_concentration(TDBc_ugperkg, p.Vc_tdb, p.PKflag, p.VPid, t, p.end_time, p.fvalidation)))
    cache.RTXc_ugperml = Float64(real(((RTXc_ugperkg / p.Vc_rtx > 1.0e-5) * RTXc_ugperkg) / p.Vc_rtx))
    cache.Blinc_ngperml = Float64(real((Blinc_ug / p.Vz_blin) * 1000))
    cache.restTpb_perml = Float64(real(restTpb / p.Vpb))
    cache.restTtiss_perml = Float64(real(restTtiss / p.Vtissue))
    cache.restTtiss2_perml = Float64(real(restTtiss2 / p.Vtissue2))
    cache.restTtiss3_perml = Float64(real(restTtiss3 / p.Vtissue3))
    cache.restTtumor_perml = Float64(real(restTtumor / p.Vtumor))
    cache.act0Tpb_perml = Float64(real(act0Tpb / p.Vpb))
    cache.act0Ttiss_perml = Float64(real(act0Ttiss / p.Vtissue))
    cache.act0Ttiss2_perml = Float64(real(act0Ttiss2 / p.Vtissue2))
    cache.act0Ttiss3_perml = Float64(real(act0Ttiss3 / p.Vtissue3))
    cache.act0Ttumor_perml = Float64(real(act0Ttumor / p.Vtumor))
    cache.actTpb_perml = Float64(real(actTpb / p.Vpb))
    cache.actTtiss_perml = Float64(real(actTtiss / p.Vtissue))
    cache.actTtiss2_perml = Float64(real(actTtiss2 / p.Vtissue2))
    cache.actTtiss3_perml = Float64(real(actTtiss3 / p.Vtissue3))
    cache.actTtumor_perml = Float64(real(actTtumor / p.Vtumor))
    cache.B19tiss3 = Float64(real(B19no20tiss3 + B1920tiss3))
    cache.Bpb_perml = Float64(real(Bpb / p.Vpb))
    cache.Btiss_perml = Float64(real(Btiss / p.Vtissue))
    cache.Btiss2_perml = Float64(real(Btiss2 / p.Vtissue2))
    cache.B1920tiss3_perml = Float64(real(B1920tiss3 / p.Vtissue3))
    cache.B19no20tiss3_perml = Float64(real(B19no20tiss3 / p.Vtissue3))
    cache.Btumor_perml = Float64(real(Btumor / p.Vtumor))
    cache.BTrratio_pb = Float64(real(Bpb / max(restTpb + act0Tpb, 1)))
    cache.BTrratio_tiss = Float64(real(Btiss / max(restTtiss + act0Ttiss, 1)))
    cache.BTrratio_tiss2 = Float64(real(Btiss2 / max(restTtiss2 + act0Ttiss2, 1)))
    cache.B1920Trratio_tiss3 = Float64(real(B1920tiss3 / max(restTtiss3 + act0Ttiss3, 1)))
    cache.BTrratio_tumor = Float64(real(Btumor / max(restTtumor + act0Ttumor, 1)))
    cache.TaBratio_pb = Float64(real(actTpb / max(Bpb, 1)))
    cache.TaBratio_tiss = Float64(real(actTtiss / max(Btiss, 1)))
    cache.TaBratio_tiss2 = Float64(real(actTtiss2 / max(Btiss2, 1)))
    cache.TaB1920ratio_tiss3 = Float64(real(actTtiss3 / max(B1920tiss3, 1)))
    cache.TaBratio_tumor = Float64(real(actTtumor / max(Btumor, 1)))
    cache.totTtiss = Float64(real(restTtiss + act0Ttiss + actTtiss))
    cache.totTtiss2 = Float64(real(restTtiss2 + act0Ttiss2 + actTtiss2))
    cache.totTtiss3 = Float64(real(restTtiss3 + act0Ttiss3 + actTtiss3))
    cache.totTtumor = Float64(real(restTtumor + act0Ttumor + actTtumor))
    cache.Baffconsumption = Float64(real(((log(2) / p.thBAFF) * BAFF * (Btiss + Bpb + Btiss2)) / (p.Bpbref_perml * (p.Vpb + p.KBp * p.Vtissue + p.KBp2 * p.Vtissue2))))
    cache.IL6combo = Float64(real(IL6pb + (p.IL6_tiss_contribution * (IL6tiss * p.Vtissue + IL6tiss2 * p.Vtissue2 + IL6tiss3 * p.Vtissue3 + IL6tumor * p.Vtumor)) / p.Vpb))
    cache.TDBt_ugperml = Float64(real(p.Kp * cache.TDBc_ugperml))
    cache.TDBt2_ugperml = Float64(real(p.Kp2 * cache.TDBc_ugperml))
    cache.TDBt3_ugperml = Float64(real(p.tissue3on * p.Kp3 * cache.TDBc_ugperml))
    cache.TDBtumor_ugperml = Float64(real(p.tumor_on * p.Kptumor * cache.TDBc_ugperml))
    cache.RTXt_ugperml = Float64(real(cache.RTXc_ugperml * p.Kp))
    cache.RTXt2_ugperml = Float64(real(cache.RTXc_ugperml * p.Kp2))
    cache.RTXt3_ugperml = Float64(real(p.tissue3on * cache.RTXc_ugperml * p.Kp3))
    cache.RTXtumor_ugperml = Float64(real(p.tumor_on * cache.RTXc_ugperml * p.Kptumor))
    cache.Blint_ngperml = Float64(real(p.Kp * cache.Blinc_ngperml))
    cache.Blint2_ngperml = Float64(real(p.Kp2 * cache.Blinc_ngperml))
    cache.Blint3_ngperml = Float64(real(p.tissue3on * p.Kp3 * cache.Blinc_ngperml))
    cache.Blintumor_ngperml = Float64(real(p.tumor_on * p.Kptumor * cache.Blinc_ngperml))
    cache.totTpb_perml = Float64(real(cache.restTpb_perml + cache.act0Tpb_perml + cache.actTpb_perml))
    cache.totTtiss_perml = Float64(real(cache.restTtiss_perml + cache.act0Ttiss_perml + cache.actTtiss_perml))
    cache.totTtiss2_perml = Float64(real(cache.restTtiss2_perml + cache.act0Ttiss2_perml + cache.actTtiss2_perml))
    cache.totTtiss3_perml = Float64(real(cache.restTtiss3_perml + cache.act0Ttiss3_perml + cache.actTtiss3_perml))
    cache.totTtumor_perml = Float64(real(cache.restTtumor_perml + cache.act0Ttumor_perml + cache.actTtumor_perml))
    cache.B19tiss3_perml = Float64(real(cache.B19tiss3 / p.Vtissue3))
    cache.B19Trratio_tiss3 = Float64(real(cache.B19tiss3 / max(restTtiss3 + act0Ttiss3, 1)))
    cache.TaB19ratio_tiss3 = Float64(real(actTtiss3 / max(cache.B19tiss3, 1)))
    cache.Bpb_norm = Float64(real(cache.Bpb_perml / p.Bpbref_perml))
    cache.drugTpbact = Float64(real(p.VmT * (pow_safe(cache.BTrratio_pb, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(cache.BTrratio_pb, p.S))) * (pow_safe(cache.TDBc_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(cache.TDBc_ugperml * 1000, p.ndrugactT)))))
    cache.BlinTpbact = Float64(real(p.VmT * (pow_safe(cache.BTrratio_pb, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(cache.BTrratio_pb, p.S))) * (pow_safe(cache.Blinc_ngperml, p.ndrugactT_blin) / (pow_safe(cache.Blinc_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    cache.drugBpbkill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_pb), p.nkill)) / (pow_safe(max(0, p.KmTB_kill), p.nkill) + pow_safe(max(0, cache.TaBratio_pb), p.nkill))) * ((cache.TDBc_ugperml * 1000) / (p.KdrugB + cache.TDBc_ugperml * 1000))))
    cache.BlinBpbkill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_pb), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin), p.nkill_blin) + pow_safe(max(0, cache.TaBratio_pb), p.nkill_blin))) * (cache.Blinc_ngperml / (p.KdrugB_blin + cache.Blinc_ngperml))))
    cache.drugTtissact = Float64(real(p.fTact * p.VmT * (pow_safe(cache.BTrratio_tiss, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(cache.BTrratio_tiss, p.S))) * (pow_safe(cache.TDBt_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(cache.TDBt_ugperml * 1000, p.ndrugactT)))))
    cache.drugBtisskill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tiss), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, cache.TaBratio_tiss), p.nkill))) * ((cache.TDBt_ugperml * 1000) / (p.KdrugB + cache.TDBt_ugperml * 1000))))
    cache.drugTtissact2 = Float64(real(p.fTact * p.VmT * (pow_safe(cache.BTrratio_tiss2, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(cache.BTrratio_tiss2, p.S))) * (pow_safe(cache.TDBt2_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(cache.TDBt2_ugperml * 1000, p.ndrugactT)))))
    cache.drugBtisskill2 = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tiss2), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, cache.TaBratio_tiss2), p.nkill))) * ((cache.TDBt2_ugperml * 1000) / (p.KdrugB + cache.TDBt2_ugperml * 1000))))
    cache.drugTtissact3 = Float64(real(p.fTact * p.VmT * (pow_safe(cache.B1920Trratio_tiss3, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(cache.B1920Trratio_tiss3, p.S))) * (pow_safe(cache.TDBt3_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(cache.TDBt3_ugperml * 1000, p.ndrugactT)))))
    cache.drugBtisskill3 = Float64(real(((p.VmB * pow_safe(max(0, cache.TaB1920ratio_tiss3), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, cache.TaB1920ratio_tiss3), p.nkill))) * ((cache.TDBt3_ugperml * 1000) / (p.KdrugB + cache.TDBt3_ugperml * 1000))))
    cache.drugTtumoract = Float64(real(p.fTact * p.VmT * (pow_safe(max(cache.BTrratio_tumor, 0), p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(max(cache.BTrratio_tumor, 0), p.S))) * (pow_safe(cache.TDBtumor_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(cache.TDBtumor_ugperml * 1000, p.ndrugactT)))))
    cache.drugBtumorkill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tumor), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, cache.TaBratio_tumor), p.nkill))) * ((cache.TDBtumor_ugperml * 1000) / (p.KdrugB + cache.TDBtumor_ugperml * 1000))))
    cache.BlinTtissact = Float64(real(p.fTact * p.VmT * (pow_safe(cache.BTrratio_tiss, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(cache.BTrratio_tiss, p.S))) * (pow_safe(cache.Blint_ngperml, p.ndrugactT_blin) / (pow_safe(cache.Blint_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    cache.BlinBtisskill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tiss), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, cache.TaBratio_tiss), p.nkill_blin))) * (cache.Blint_ngperml / (p.KdrugB_blin + cache.Blint_ngperml))))
    cache.BlinTtissact2 = Float64(real(p.fTact * p.VmT * (pow_safe(cache.BTrratio_tiss2, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(cache.BTrratio_tiss2, p.S))) * (pow_safe(cache.Blint2_ngperml, p.ndrugactT_blin) / (pow_safe(cache.Blint2_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    cache.BlinBtisskill2 = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tiss2), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, cache.TaBratio_tiss2), p.nkill_blin))) * (cache.Blint2_ngperml / (p.KdrugB_blin + cache.Blint2_ngperml))))
    cache.BlinTtumoract = Float64(real(p.fTact * p.VmT * (pow_safe(max(cache.BTrratio_tumor, 0), p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(max(cache.BTrratio_tumor, 0), p.S))) * (pow_safe(cache.Blintumor_ngperml, p.ndrugactT_blin) / (pow_safe(cache.Blintumor_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    cache.BlinBtumorkill = Float64(real(((p.VmB * pow_safe(max(0, cache.TaBratio_tumor), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, cache.TaBratio_tumor), p.nkill_blin))) * (cache.Blintumor_ngperml / (p.KdrugB_blin + cache.Blintumor_ngperml))))
    cache.Tafraction_pb = Float64(real(cache.actTpb_perml / max(cache.totTpb_perml, 1)))
    cache.Tafraction_tiss = Float64(real(cache.actTtiss_perml / max(cache.totTtiss_perml, 1)))
    cache.BTtotRatio_tiss = Float64(real(cache.Btiss_perml / cache.totTtiss_perml))
    cache.Tafraction_tiss2 = Float64(real(cache.actTtiss2_perml / max(cache.totTtiss2_perml, 1)))
    cache.BTtotRatio_tiss2 = Float64(real(cache.Btiss2_perml / cache.totTtiss2_perml))
    cache.Tafraction_tiss3 = Float64(real(cache.actTtiss3_perml / max(cache.totTtiss3_perml, 1)))
    cache.Tafraction_tumor = Float64(real(cache.actTtumor_perml / max(cache.totTtumor_perml, 1)))
    cache.BlinTtissact3 = Float64(real(p.fTact * p.VmT * (pow_safe(cache.B19Trratio_tiss3, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(cache.B19Trratio_tiss3, p.S))) * (pow_safe(cache.Blint3_ngperml, p.ndrugactT_blin) / (pow_safe(cache.Blint3_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    cache.BlinBtisskill3 = Float64(real(((p.VmB * pow_safe(max(0, cache.TaB19ratio_tiss3), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, cache.TaB19ratio_tiss3), p.nkill_blin))) * (cache.Blint3_ngperml / (p.KdrugB_blin + cache.Blint3_ngperml))))
    return cache
end

function value_at(u::AbstractVector{<:Real}, p::MosunParams, t, name::Symbol, cache::MosunObservablesCache = zero_observables_cache())
    update_observables!(cache, u, p, t)
    return state_or_observable(u, cache, name)
end

function mosun_rhs!(du, u::AbstractVector{<:Real}, p::MosunParams, t, cache::MosunObservablesCache; active_rates = nothing, infusions = nothing)
    actTtiss = u[1]
    Btiss = u[2]
    TDBc_ugperkg = u[3]
    actTpb = u[4]
    restTtiss = u[5]
    TDBp_ugperkg = u[6]
    restTpb = u[7]
    Bpb = u[8]
    BAFF = u[9]
    act0Tpb = u[10]
    act0Ttiss = u[11]
    injection_effect = u[12]
    Btiss2 = u[13]
    act0Ttiss2 = u[14]
    restTtiss2 = u[15]
    actTtiss2 = u[16]
    RTXc_ugperkg = u[17]
    RTXp_ugperkg = u[18]
    drug_effect = u[19]
    B1920tiss3 = u[20]
    B19no20tiss3 = u[21]
    restTtiss3 = u[22]
    act0Ttiss3 = u[23]
    actTtiss3 = u[24]
    Blinc_ug = u[25]
    restTtumor = u[26]
    actTtumor = u[27]
    Btumor = u[28]
    act0Ttumor = u[29]
    TDBsc_ugperkg = u[30]
    TDBc_ugperml_AUC = u[31]
    IL6pb = u[32]
    IL6tiss = u[33]
    IL6tiss2 = u[34]
    IL6tiss3 = u[35]
    IL6tumor = u[36]
    TDBc_ugperml = Float64(real(tdb_central_concentration(TDBc_ugperkg, p.Vc_tdb, p.PKflag, p.VPid, t, p.end_time, p.fvalidation)))
    RTXc_ugperml = Float64(real(((RTXc_ugperkg / p.Vc_rtx > 1.0e-5) * RTXc_ugperkg) / p.Vc_rtx))
    Blinc_ngperml = Float64(real((Blinc_ug / p.Vz_blin) * 1000))
    restTpb_perml = Float64(real(restTpb / p.Vpb))
    act0Tpb_perml = Float64(real(act0Tpb / p.Vpb))
    actTpb_perml = Float64(real(actTpb / p.Vpb))
    B19tiss3 = Float64(real(B19no20tiss3 + B1920tiss3))
    Bpb_perml = Float64(real(Bpb / p.Vpb))
    Btiss_perml = Float64(real(Btiss / p.Vtissue))
    Btiss2_perml = Float64(real(Btiss2 / p.Vtissue2))
    B1920tiss3_perml = Float64(real(B1920tiss3 / p.Vtissue3))
    Btumor_perml = Float64(real(Btumor / p.Vtumor))
    BTrratio_pb = Float64(real(Bpb / max(restTpb + act0Tpb, 1)))
    BTrratio_tiss = Float64(real(Btiss / max(restTtiss + act0Ttiss, 1)))
    BTrratio_tiss2 = Float64(real(Btiss2 / max(restTtiss2 + act0Ttiss2, 1)))
    B1920Trratio_tiss3 = Float64(real(B1920tiss3 / max(restTtiss3 + act0Ttiss3, 1)))
    BTrratio_tumor = Float64(real(Btumor / max(restTtumor + act0Ttumor, 1)))
    TaBratio_pb = Float64(real(actTpb / max(Bpb, 1)))
    TaBratio_tiss = Float64(real(actTtiss / max(Btiss, 1)))
    TaBratio_tiss2 = Float64(real(actTtiss2 / max(Btiss2, 1)))
    TaB1920ratio_tiss3 = Float64(real(actTtiss3 / max(B1920tiss3, 1)))
    TaBratio_tumor = Float64(real(actTtumor / max(Btumor, 1)))
    TDBt_ugperml = Float64(real(p.Kp * TDBc_ugperml))
    TDBt2_ugperml = Float64(real(p.Kp2 * TDBc_ugperml))
    TDBt3_ugperml = Float64(real(p.tissue3on * p.Kp3 * TDBc_ugperml))
    TDBtumor_ugperml = Float64(real(p.tumor_on * p.Kptumor * TDBc_ugperml))
    RTXt_ugperml = Float64(real(RTXc_ugperml * p.Kp))
    RTXt2_ugperml = Float64(real(RTXc_ugperml * p.Kp2))
    RTXt3_ugperml = Float64(real(p.tissue3on * RTXc_ugperml * p.Kp3))
    RTXtumor_ugperml = Float64(real(p.tumor_on * RTXc_ugperml * p.Kptumor))
    Blint_ngperml = Float64(real(p.Kp * Blinc_ngperml))
    Blint2_ngperml = Float64(real(p.Kp2 * Blinc_ngperml))
    Blint3_ngperml = Float64(real(p.tissue3on * p.Kp3 * Blinc_ngperml))
    Blintumor_ngperml = Float64(real(p.tumor_on * p.Kptumor * Blinc_ngperml))
    totTpb_perml = Float64(real(restTpb_perml + act0Tpb_perml + actTpb_perml))
    B19tiss3_perml = Float64(real(B19tiss3 / p.Vtissue3))
    B19Trratio_tiss3 = Float64(real(B19tiss3 / max(restTtiss3 + act0Ttiss3, 1)))
    TaB19ratio_tiss3 = Float64(real(actTtiss3 / max(B19tiss3, 1)))
    drugTpbact = Float64(real(p.VmT * (pow_safe(BTrratio_pb, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(BTrratio_pb, p.S))) * (pow_safe(TDBc_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBc_ugperml * 1000, p.ndrugactT)))))
    BlinTpbact = Float64(real(p.VmT * (pow_safe(BTrratio_pb, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(BTrratio_pb, p.S))) * (pow_safe(Blinc_ngperml, p.ndrugactT_blin) / (pow_safe(Blinc_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    drugBpbkill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_pb), p.nkill)) / (pow_safe(max(0, p.KmTB_kill), p.nkill) + pow_safe(max(0, TaBratio_pb), p.nkill))) * ((TDBc_ugperml * 1000) / (p.KdrugB + TDBc_ugperml * 1000))))
    BlinBpbkill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_pb), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin), p.nkill_blin) + pow_safe(max(0, TaBratio_pb), p.nkill_blin))) * (Blinc_ngperml / (p.KdrugB_blin + Blinc_ngperml))))
    drugTtissact = Float64(real(p.fTact * p.VmT * (pow_safe(BTrratio_tiss, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(BTrratio_tiss, p.S))) * (pow_safe(TDBt_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt_ugperml * 1000, p.ndrugactT)))))
    drugBtisskill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tiss), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, TaBratio_tiss), p.nkill))) * ((TDBt_ugperml * 1000) / (p.KdrugB + TDBt_ugperml * 1000))))
    drugTtissact2 = Float64(real(p.fTact * p.VmT * (pow_safe(BTrratio_tiss2, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(BTrratio_tiss2, p.S))) * (pow_safe(TDBt2_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt2_ugperml * 1000, p.ndrugactT)))))
    drugBtisskill2 = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tiss2), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, TaBratio_tiss2), p.nkill))) * ((TDBt2_ugperml * 1000) / (p.KdrugB + TDBt2_ugperml * 1000))))
    drugTtissact3 = Float64(real(p.fTact * p.VmT * (pow_safe(B1920Trratio_tiss3, p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(B1920Trratio_tiss3, p.S))) * (pow_safe(TDBt3_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt3_ugperml * 1000, p.ndrugactT)))))
    drugBtisskill3 = Float64(real(((p.VmB * pow_safe(max(0, TaB1920ratio_tiss3), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, TaB1920ratio_tiss3), p.nkill))) * ((TDBt3_ugperml * 1000) / (p.KdrugB + TDBt3_ugperml * 1000))))
    drugTtumoract = Float64(real(p.fTact * p.VmT * (pow_safe(max(BTrratio_tumor, 0), p.S) / (pow_safe(p.KmBT_act, p.S) + pow_safe(max(BTrratio_tumor, 0), p.S))) * (pow_safe(TDBtumor_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBtumor_ugperml * 1000, p.ndrugactT)))))
    drugBtumorkill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tumor), p.nkill)) / (pow_safe(max(0, p.KmTB_kill * p.fKmTB_kill), p.nkill) + pow_safe(max(0, TaBratio_tumor), p.nkill))) * ((TDBtumor_ugperml * 1000) / (p.KdrugB + TDBtumor_ugperml * 1000))))
    BlinTtissact = Float64(real(p.fTact * p.VmT * (pow_safe(BTrratio_tiss, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(BTrratio_tiss, p.S))) * (pow_safe(Blint_ngperml, p.ndrugactT_blin) / (pow_safe(Blint_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    BlinBtisskill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tiss), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, TaBratio_tiss), p.nkill_blin))) * (Blint_ngperml / (p.KdrugB_blin + Blint_ngperml))))
    BlinTtissact2 = Float64(real(p.fTact * p.VmT * (pow_safe(BTrratio_tiss2, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(BTrratio_tiss2, p.S))) * (pow_safe(Blint2_ngperml, p.ndrugactT_blin) / (pow_safe(Blint2_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    BlinBtisskill2 = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tiss2), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, TaBratio_tiss2), p.nkill_blin))) * (Blint2_ngperml / (p.KdrugB_blin + Blint2_ngperml))))
    BlinTtumoract = Float64(real(p.fTact * p.VmT * (pow_safe(max(BTrratio_tumor, 0), p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(max(BTrratio_tumor, 0), p.S))) * (pow_safe(Blintumor_ngperml, p.ndrugactT_blin) / (pow_safe(Blintumor_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    BlinBtumorkill = Float64(real(((p.VmB * pow_safe(max(0, TaBratio_tumor), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, TaBratio_tumor), p.nkill_blin))) * (Blintumor_ngperml / (p.KdrugB_blin + Blintumor_ngperml))))
    BlinTtissact3 = Float64(real(p.fTact * p.VmT * (pow_safe(B19Trratio_tiss3, p.S) / (pow_safe(p.KmBT_act_blin, p.S) + pow_safe(B19Trratio_tiss3, p.S))) * (pow_safe(Blint3_ngperml, p.ndrugactT_blin) / (pow_safe(Blint3_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    BlinBtisskill3 = Float64(real(((p.VmB * pow_safe(max(0, TaB19ratio_tiss3), p.nkill_blin)) / (pow_safe(max(0, p.KmTB_kill_blin * p.fKmTB_kill), p.nkill_blin) + pow_safe(max(0, TaB19ratio_tiss3), p.nkill_blin))) * (Blint3_ngperml / (p.KdrugB_blin + Blint3_ngperml))))
    fill!(du, 0.0)
    if active_rates !== nothing
        @inbounds for i in eachindex(du, active_rates)
            du[i] += active_rates[i]
        end
    elseif infusions !== nothing
        for inf in infusions
            if t >= inf.t_start && t <= inf.t_end
                du[inf.target_idx] += inf.rate
            end
        end
    end
    rate_1 = Float64(real(p.kBprolif * p.KBp * p.Bpbref_perml * p.Vtissue * pow_safe(max(0, 1 - Btiss / (p.KBp * p.Bpbref_perml * p.Vtissue)), 1)))
    du[2] += 1.0 * rate_1
    rate_2 = Float64(real(p.kTact * ((drugTtissact + BlinTtissact) * restTtiss - p.fTadeact * actTtiss)))
    du[1] += 1.0 * rate_2
    du[5] += -1.0 * rate_2
    rate_3 = Float64(real(0 * p.fBprolif * p.kBprolif * Btiss * p.kBapop_cll))
    du[2] += -1.0 * rate_3
    rate_4 = Float64(real(((p.Cl_tdb + (p.Vm_tdb / (p.Km_tdb + TDBc_ugperkg / p.Vc_tdb)) / p.BW) / p.Vc_tdb) * TDBc_ugperkg))
    du[3] += -1.0 * rate_4
    rate_5 = Float64(real(p.Cld_tdb * (TDBc_ugperkg / p.Vc_tdb - TDBp_ugperkg / p.Vp_tdb)))
    du[3] += -1.0 * rate_5
    du[6] += 1.0 * rate_5
    rate_6 = Float64(real(0))
    du[5] += 1.0 * rate_6
    rate_7 = Float64(real(0))
    rate_8 = Float64(real(p.tissue1on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * restTpb) / p.Vpb) * p.KTrp - restTtiss / p.Vtissue)))
    du[5] += 1.0 * rate_8
    du[7] += -1.0 * rate_8
    rate_9 = Float64(real(p.kTgen * p.Vpb * p.Trpbref_perml * (p.fTgenbl + pow_safe(max(0, 1 - totTpb_perml / p.Trpbref_perml), 2))))
    du[7] += 1.0 * rate_9
    rate_10 = Float64(real(p.tissue1on * p.fBexit * p.kTaexit * p.Vpb * max(0, (Bpb / p.Vpb) * p.KBp - Btiss / p.Vtissue)))
    du[2] += 1.0 * rate_10
    du[8] += -1.0 * rate_10
    rate_11 = Float64(real(p.kTact * ((drugTpbact + BlinTpbact) * restTpb - p.fTadeact * actTpb)))
    du[4] += 1.0 * rate_11
    du[7] += -1.0 * rate_11
    rate_12 = Float64(real(p.tissue3on * p.fBtissue3_v1 * p.kBapop * Bpb * p.kBapop_cll))
    du[8] += 1.0 * rate_12
    rate_13 = Float64(real(p.kBapop * Bpb * p.kBapop_cll))
    du[8] += -1.0 * rate_13
    rate_14 = Float64(real(0))
    rate_15 = Float64(real(0))
    rate_16 = Float64(real(p.kBkill * (drugBpbkill + BlinBpbkill + RTXc_ugperml / (p.Kmkill_rtx / 1000 + RTXc_ugperml)) * Bpb))
    du[8] += -1.0 * rate_16
    rate_17 = Float64(real(p.fBkill * p.kBkill * (drugBtisskill + BlinBtisskill + RTXt_ugperml / (p.Kmkill_rtx / 1000 + RTXt_ugperml)) * Btiss))
    du[2] += -1.0 * rate_17
    rate_18 = Float64(real(p.kTgen * p.fTgenbl * restTpb + p.fTrapop * p.kTaapop * max(0, restTpb - p.Trpbref_perml * p.Vpb)))
    du[7] += -1.0 * rate_18
    rate_19 = Float64(real(p.tissue1on * p.kTaexit * p.Vpb * ((((1 + p.fa0 * (p.finj * injection_effect + p.fdrug * drug_effect)) * actTpb) / p.Vpb) * p.KTrp * p.fTap - actTtiss / p.Vtissue)))
    du[1] += 1.0 * rate_19
    du[4] += -1.0 * rate_19
    rate_20 = Float64(real(p.kTaapop * (actTpb + (p.fAICD * pow_safe(actTpb, 2)) / (p.Vpb * p.Trpbref_perml))))
    du[4] += -1.0 * rate_20
    rate_21 = Float64(real((log(2) / ((p.thBAFF / 24) / 60)) * ((p.fBAFFo * BAFF) / p.BAFFo + ((1 - p.fBAFFo) * (Btiss + Bpb + Btiss2)) / (p.Bpbref_perml * (p.Vpb + p.KBp * p.Vtissue + p.KBp2 * p.Vtissue2)))))
    du[9] += -1.0 * rate_21
    rate_22 = Float64(real((log(2) / ((p.thBAFF / 24) / 60)) * (p.fBAFFo + ((1 - p.fBAFFo) * p.Bpbo_perml) / p.Bpbref_perml)))
    du[9] += 1.0 * rate_22
    rate_23 = Float64(real(p.act0on * p.fTa0deact * act0Tpb))
    du[7] += 1.0 * rate_23
    du[10] += -1.0 * rate_23
    rate_24 = Float64(real(p.act0on * p.kTact * ((drugTpbact + BlinTpbact) * act0Tpb - p.fTadeact * actTpb)))
    du[4] += 1.0 * rate_24
    du[10] += -1.0 * rate_24
    rate_25 = Float64(real(p.fTa0apop * p.kTaapop * act0Tpb))
    du[10] += -1.0 * rate_25
    rate_26 = Float64(real(p.fTa0apop * p.kTaapop * act0Ttiss))
    du[11] += -1.0 * rate_26
    rate_27 = Float64(real(p.act0on * p.kTact * ((drugTtissact + BlinTtissact) * act0Ttiss - p.fTadeact * actTtiss)))
    du[1] += 1.0 * rate_27
    du[11] += -1.0 * rate_27
    rate_28 = Float64(real(p.act0on * p.fTa0deact * act0Ttiss))
    du[5] += 1.0 * rate_28
    du[11] += -1.0 * rate_28
    rate_29 = Float64(real(p.tissue1on * p.act0on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * act0Tpb) / p.Vpb) * p.KTrp - act0Ttiss / p.Vtissue)))
    du[10] += -1.0 * rate_29
    du[11] += 1.0 * rate_29
    rate_30 = Float64(real(p.act0on * p.fTaprolif * p.kTprolif * actTpb))
    du[10] += 1.0 * rate_30
    rate_31 = Float64(real(p.act0on * p.fTaprolif * p.kTprolif * actTtiss))
    du[11] += 1.0 * rate_31
    rate_32 = Float64(real((log(2) / p.tinjhalf) * injection_effect))
    du[12] += -1.0 * rate_32
    rate_33 = Float64(real(p.act0on * p.fTaprolif * p.kTprolif * actTtiss2))
    du[14] += 1.0 * rate_33
    rate_34 = Float64(real(p.tissue2on * p.act0on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * act0Tpb) / p.Vpb) * p.KTrp2 - act0Ttiss2 / p.Vtissue2)))
    du[10] += -1.0 * rate_34
    du[14] += 1.0 * rate_34
    rate_35 = Float64(real(p.act0on * p.fTa0deact * act0Ttiss2))
    du[14] += -1.0 * rate_35
    du[15] += 1.0 * rate_35
    rate_36 = Float64(real(p.act0on * p.kTact * ((drugTtissact2 + BlinTtissact2) * act0Ttiss2 - p.fTadeact * actTtiss2)))
    du[14] += -1.0 * rate_36
    du[16] += 1.0 * rate_36
    rate_37 = Float64(real(p.fTa0apop * p.kTaapop * act0Ttiss2))
    du[14] += -1.0 * rate_37
    rate_38 = Float64(real(p.tissue2on * p.kTaexit * p.Vpb * ((((1 + p.fa0 * (p.finj * injection_effect + p.fdrug * drug_effect)) * actTpb) / p.Vpb) * p.KTrp2 * p.fTap - actTtiss2 / p.Vtissue2)))
    du[4] += -1.0 * rate_38
    du[16] += 1.0 * rate_38
    rate_39 = Float64(real(p.fBkill * p.kBkill * (drugBtisskill2 + BlinBtisskill2 + RTXt2_ugperml / (p.Kmkill_rtx / 1000 + RTXt2_ugperml)) * Btiss2))
    du[13] += -1.0 * rate_39
    rate_40 = Float64(real(p.tissue2on * p.fBexit * p.kTaexit * p.Vpb * max(0, (Bpb / p.Vpb) * p.KBp2 - Btiss2 / p.Vtissue2)))
    du[8] += -1.0 * rate_40
    du[13] += 1.0 * rate_40
    rate_41 = Float64(real(p.tissue2on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * restTpb) / p.Vpb) * p.KTrp2 - restTtiss2 / p.Vtissue2)))
    du[7] += -1.0 * rate_41
    du[15] += 1.0 * rate_41
    rate_42 = Float64(real(0))
    rate_43 = Float64(real(0))
    rate_44 = Float64(real(0 * p.fBprolif * p.kBprolif * Btiss2 * p.kBapop_cll))
    du[13] += -1.0 * rate_44
    rate_45 = Float64(real(p.kTact * ((drugTtissact2 + BlinTtissact2) * restTtiss2 - p.fTadeact * actTtiss2)))
    du[15] += -1.0 * rate_45
    du[16] += 1.0 * rate_45
    rate_46 = Float64(real(p.kBprolif * p.KBp2 * p.Bpbref_perml * p.Vtissue2 * pow_safe(max(0, 1 - Btiss2 / (p.KBp2 * p.Bpbref_perml * p.Vtissue2)), 1)))
    du[13] += 1.0 * rate_46
    rate_47 = Float64(real(p.Cld_rtx * (RTXc_ugperkg / p.Vc_rtx - RTXp_ugperkg / p.Vp_rtx)))
    du[17] += -1.0 * rate_47
    du[18] += 1.0 * rate_47
    rate_48 = Float64(real(((p.Cl_rtx + (p.Vm_rtx / (p.Km_rtx + RTXc_ugperkg / p.Vc_rtx)) / p.BW) / p.Vc_rtx) * RTXc_ugperkg))
    du[17] += -1.0 * rate_48
    rate_49 = Float64(real((log(2) / p.tinjhalf) * drug_effect))
    du[19] += -1.0 * rate_49
    rate_50 = Float64(real(p.Bpbref_perml * p.KBp3 * p.B19no20_B1920_ratio * p.Vtissue3 * (p.kBapop + p.kBapop / p.kBmat_kBapop_ratio) * (1 + 5 * pow_safe(max(0, 1 - (Bpb + Btiss + Btiss2) / (p.Bpbref_perml * p.Vpb + p.KBp * p.Bpbo_perml * p.Vtissue + p.KBp2 * p.Bpbo_perml * p.Vtissue2)), 1))))
    du[21] += 1.0 * rate_50
    rate_51 = Float64(real((p.kBapop / p.kBmat_kBapop_ratio) * B19no20tiss3))
    du[20] += 1.0 * rate_51
    du[21] += -1.0 * rate_51
    rate_52 = Float64(real(p.tissue3on * p.fBtissue3_v1 * p.kBtiss3exit * p.Vpb * max(0, B1920tiss3 / p.Vtissue3 - (Bpb / p.Vpb) * p.KBp3)))
    du[8] += 1.0 * rate_52
    du[20] += -1.0 * rate_52
    rate_53 = Float64(real(p.kBapop * B19no20tiss3 * p.kBapop_cll))
    du[21] += -1.0 * rate_53
    rate_54 = Float64(real(p.kBapop * B1920tiss3 * p.kBapop_cll))
    du[20] += -1.0 * rate_54
    rate_55 = Float64(real(p.tissue3on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * restTpb) / p.Vpb) * p.KTrp3 - restTtiss3 / p.Vtissue3)))
    du[7] += -1.0 * rate_55
    du[22] += 1.0 * rate_55
    rate_56 = Float64(real(p.fBkill * p.kBkill * (drugBtisskill3 + BlinBtisskill3 + RTXt3_ugperml / (p.Kmkill_rtx / 1000 + RTXt3_ugperml)) * B1920tiss3))
    du[20] += -1.0 * rate_56
    rate_57 = Float64(real(p.tissue3on * p.act0on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * act0Tpb) / p.Vpb) * p.KTrp3 - act0Ttiss3 / p.Vtissue3)))
    du[10] += -1.0 * rate_57
    du[23] += 1.0 * rate_57
    rate_58 = Float64(real(p.tissue3on * p.kTaexit * p.Vpb * ((((1 + p.fa0 * (p.finj * injection_effect + p.fdrug * drug_effect)) * actTpb) / p.Vpb) * p.KTrp3 * p.fTap - actTtiss3 / p.Vtissue3)))
    du[4] += -1.0 * rate_58
    du[24] += 1.0 * rate_58
    rate_59 = Float64(real(p.fTa0apop * p.kTaapop * act0Ttiss3))
    du[23] += -1.0 * rate_59
    rate_60 = Float64(real(p.act0on * p.kTact * ((drugTtissact3 + BlinTtissact3) * act0Ttiss3 - p.fTadeact * actTtiss3)))
    du[23] += -1.0 * rate_60
    du[24] += 1.0 * rate_60
    rate_61 = Float64(real(p.act0on * p.fTa0deact * act0Ttiss3))
    du[22] += 1.0 * rate_61
    du[23] += -1.0 * rate_61
    rate_62 = Float64(real(p.kTact * ((drugTtissact3 + BlinTtissact3) * restTtiss3 - p.fTadeact * actTtiss3)))
    du[22] += -1.0 * rate_62
    du[24] += 1.0 * rate_62
    rate_63 = Float64(real(p.act0on * p.fTaprolif * p.kTprolif * actTtiss3))
    du[23] += 1.0 * rate_63
    rate_64 = Float64(real(0))
    rate_65 = Float64(real(p.fBkill * p.kBkill * BlinBtisskill3 * B19no20tiss3))
    du[21] += -1.0 * rate_65
    rate_66 = Float64(real(0))
    rate_67 = Float64(real(0))
    rate_68 = Float64(real(p.fapop_v24 * p.fTrapop * p.kTaapop * (p.Trpbref_perml * p.Vtissue * p.KTrp) * max(0, restTtiss / (p.Trpbref_perml * p.Vtissue * p.KTrp) - 1)))
    du[5] += -1.0 * rate_68
    rate_69 = Float64(real(p.fapop_v24 * p.fTrapop * p.kTaapop * (p.Trpbref_perml * p.Vtissue2 * p.KTrp2) * max(0, restTtiss2 / (p.Trpbref_perml * p.Vtissue2 * p.KTrp2) - 1)))
    du[15] += -1.0 * rate_69
    rate_70 = Float64(real(p.fapop_v24 * p.kTaapop * (actTtiss + (p.fAICD * pow_safe(actTtiss, 2)) / (p.KTrp * p.Vtissue * p.Trpbref_perml))))
    du[1] += -1.0 * rate_70
    rate_71 = Float64(real(p.fapop_v24 * p.kTaapop * (actTtiss2 + (p.fAICD * pow_safe(actTtiss2, 2)) / (p.Vtissue2 * p.KTrp2 * p.Trpbref_perml))))
    du[16] += -1.0 * rate_71
    rate_72 = Float64(real(p.fapop_v24 * p.fTrapop * p.kTaapop * (p.Trpbref_perml * p.Vtissue3 * p.KTrp3) * max(0, restTtiss3 / (p.Trpbref_perml * p.Vtissue3 * p.KTrp3) - 1)))
    du[22] += -1.0 * rate_72
    rate_73 = Float64(real(p.fapop_v24 * p.kTaapop * (actTtiss3 + (p.fAICD * pow_safe(actTtiss3, 2)) / (p.Vtissue3 * p.KTrp3 * p.Trpbref_perml))))
    du[24] += -1.0 * rate_73
    rate_74 = Float64(real(p.tissue3on * p.fBtissue3_v1 * p.kBprolif * p.kBapop_cll * p.Bpbref_perml * p.Vpb * pow_safe(max(0, 1 - Bpb / (p.Bpbref_perml * p.Vpb)), 1)))
    du[8] += 1.0 * rate_74
    rate_75 = Float64(real((p.Cl_blin * Blinc_ug) / p.Vz_blin))
    du[25] += -1.0 * rate_75
    rate_76 = Float64(real(0))
    rate_77 = Float64(real(0))
    rate_78 = Float64(real(0))
    rate_79 = Float64(real(0))
    rate_80 = Float64(real(0))
    rate_81 = Float64(real(0))
    rate_82 = Float64(real(0))
    rate_83 = Float64(real(0))
    rate_84 = Float64(real(p.kBprolif * p.KBp3 * p.Bpbref_perml * p.Vtissue3 * pow_safe(max(0, 1 - B1920tiss3 / (p.KBp3 * p.Bpbref_perml * p.Vtissue3)), 1)))
    du[20] += 1.0 * rate_84
    rate_85 = Float64(real(p.kBprolif * p.KBp3 * p.Bpbref_perml * p.B19no20_B1920_ratio * p.Vtissue3 * pow_safe(max(0, 1 - B19no20tiss3 / (p.KBp3 * p.Bpbref_perml * p.B19no20_B1920_ratio * p.Vtissue3)), 1)))
    du[21] += 1.0 * rate_85
    rate_86 = Float64(real(p.act0on * p.fTa0deact * act0Ttumor))
    du[26] += 1.0 * rate_86
    du[29] += -1.0 * rate_86
    rate_87 = Float64(real(p.act0on * p.kTact * ((drugTtumoract + BlinTtumoract) * act0Ttumor - p.fTadeact * actTtumor)))
    du[27] += 1.0 * rate_87
    du[29] += -1.0 * rate_87
    rate_88 = Float64(real(p.act0on * p.fTaprolif * p.kTprolif * actTtumor))
    du[29] += 1.0 * rate_88
    rate_89 = Float64(real(p.kTact * ((drugTtumoract + BlinTtumoract) * restTtumor - p.fTadeact * actTtumor)))
    du[26] += -1.0 * rate_89
    du[27] += 1.0 * rate_89
    rate_90 = Float64(real(0))
    rate_91 = Float64(real(0))
    rate_92 = Float64(real(0))
    rate_93 = Float64(real(0))
    rate_94 = Float64(real(p.kBtumorprolif * Btumor + 0 * p.kBprolif * p.KBptumor * p.Bpbref_perml * p.Vtumor * pow_safe(max(0, 1 - Btumor / (p.KBptumor * p.Bpbref_perml * p.Vtumor)), 1)))
    du[28] += 1.0 * rate_94
    rate_95 = Float64(real(0 * p.fBprolif * p.kBprolif * Btumor))
    du[28] += -1.0 * rate_95
    rate_96 = Float64(real(p.fBkill * p.kBkill * (drugBtumorkill + BlinBtumorkill + RTXtumor_ugperml / (p.Kmkill_rtx / 1000 + RTXtumor_ugperml)) * Btumor))
    du[28] += -1.0 * rate_96
    rate_97 = Float64(real(p.fapop_v24 * p.kTaapop * (actTtumor + (p.fAICD * pow_safe(actTtumor, 2)) / (p.Vtumor * p.KTrptumor * p.Trpbref_perml))))
    du[27] += -1.0 * rate_97
    rate_98 = Float64(real(p.Bcell_tumor_trafficking_on * p.tumor_on * p.fBexit * p.kTaexit * p.Vpb * max(0, (Bpb / p.Vpb) * p.KBptumor - Btumor / p.Vtumor)))
    du[8] += -1.0 * rate_98
    du[28] += 1.0 * rate_98
    rate_99 = Float64(real(p.tumor_on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * restTpb) / p.Vpb) * p.KTrptumor - restTtumor / p.Vtumor)))
    du[7] += -1.0 * rate_99
    du[26] += 1.0 * rate_99
    rate_100 = Float64(real(p.tumor_on * p.kTaexit * p.Vpb * ((((1 + p.fa0 * (p.finj * injection_effect + p.fdrug * drug_effect)) * actTpb) / p.Vpb) * p.KTrptumor * p.fTap - actTtumor / p.Vtumor)))
    du[4] += -1.0 * rate_100
    du[27] += 1.0 * rate_100
    rate_101 = Float64(real(p.tumor_on * p.act0on * p.kTrexit * p.Vpb * ((((1 + p.finj * injection_effect + p.fdrug * drug_effect) * act0Tpb) / p.Vpb) * p.KTrptumor - act0Ttumor / p.Vtumor)))
    du[10] += -1.0 * rate_101
    du[29] += 1.0 * rate_101
    rate_102 = Float64(real(p.fTa0apop * p.kTaapop * act0Ttumor))
    du[29] += -1.0 * rate_102
    rate_103 = Float64(real(p.fapop_v24 * p.fTrapop * p.kTaapop * (p.Trpbref_perml * p.Vtumor * p.KTrptumor) * max(0, restTtumor / (p.Trpbref_perml * p.Vtumor * p.KTrptumor) - 1)))
    du[26] += -1.0 * rate_103
    rate_104 = Float64(real(p.kabs_TDB * p.fbio_TDB * TDBsc_ugperkg))
    du[3] += 1.0 * rate_104
    du[30] += -1.0 * rate_104
    rate_105 = Float64(real(p.kabs_TDB * (1 - p.fbio_TDB) * TDBsc_ugperkg))
    du[30] += -1.0 * rate_105
    rate_106 = Float64(real(TDBc_ugperml))
    du[31] += 1.0 * rate_106
    rate_107 = Float64(real(((p.kIL6prod * actTpb) / p.Vpb) * (Bpb_perml / p.Bpbref_perml) * (pow_safe(TDBc_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBc_ugperml * 1000, p.ndrugactT)) + pow_safe(Blinc_ngperml, p.ndrugactT_blin) / (pow_safe(Blinc_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    du[32] += 1.0 * rate_107
    rate_108 = Float64(real((log(2) / ((p.thalfIL6 / 60) / 24)) * IL6pb))
    du[32] += -1.0 * rate_108
    rate_109 = Float64(real(((p.kIL6prod * actTtiss) / p.Vtissue) * (Btiss_perml / (p.Bpbref_perml * p.KBp)) * (pow_safe(TDBt_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt_ugperml * 1000, p.ndrugactT)) + pow_safe(Blint_ngperml, p.ndrugactT_blin) / (pow_safe(Blint_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    du[33] += 1.0 * rate_109
    rate_110 = Float64(real((log(2) / ((p.thalfIL6 / 60) / 24)) * IL6tiss))
    du[33] += -1.0 * rate_110
    rate_111 = Float64(real(((p.kIL6prod * actTtiss2) / p.Vtissue2) * (Btiss2_perml / (p.Bpbref_perml * p.KBp2)) * (pow_safe(TDBt2_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt2_ugperml * 1000, p.ndrugactT)) + pow_safe(Blint2_ngperml, p.ndrugactT_blin) / (pow_safe(Blint2_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    du[34] += 1.0 * rate_111
    rate_112 = Float64(real((log(2) / ((p.thalfIL6 / 60) / 24)) * IL6tiss2))
    du[34] += -1.0 * rate_112
    rate_113 = Float64(real(((p.kIL6prod * actTtiss3) / p.Vtissue3) * ((B1920tiss3_perml / (p.Bpbref_perml * p.KBp3)) * (pow_safe(TDBt3_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBt3_ugperml * 1000, p.ndrugactT))) + (B19tiss3_perml / (p.Bpbref_perml * p.KBp3 * (1 + p.B19no20_B1920_ratio))) * (pow_safe(Blint3_ngperml, p.ndrugactT_blin) / (pow_safe(Blint3_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin))))))
    du[35] += 1.0 * rate_113
    rate_114 = Float64(real((log(2) / ((p.thalfIL6 / 60) / 24)) * IL6tiss3))
    du[35] += -1.0 * rate_114
    rate_115 = Float64(real(((p.kIL6prod * actTtumor) / p.Vtumor) * (Btumor_perml / (p.Bpbref_perml * p.KBptumor)) * (pow_safe(TDBtumor_ugperml * 1000, p.ndrugactT) / (pow_safe(p.KdrugactT, p.ndrugactT) + pow_safe(TDBtumor_ugperml * 1000, p.ndrugactT)) + pow_safe(Blintumor_ngperml, p.ndrugactT_blin) / (pow_safe(Blintumor_ngperml, p.ndrugactT_blin) + pow_safe(p.KdrugactT_blin, p.ndrugactT_blin)))))
    du[36] += 1.0 * rate_115
    rate_116 = Float64(real((log(2) / ((p.thalfIL6 / 60) / 24)) * IL6tumor))
    du[36] += -1.0 * rate_116
    return nothing
end

function mosun_rhs_vector!(du, u::AbstractVector, p::AbstractVector, t)
    actTtiss = u[1]
    Btiss = u[2]
    TDBc_ugperkg = u[3]
    actTpb = u[4]
    restTtiss = u[5]
    TDBp_ugperkg = u[6]
    restTpb = u[7]
    Bpb = u[8]
    BAFF = u[9]
    act0Tpb = u[10]
    act0Ttiss = u[11]
    injection_effect = u[12]
    Btiss2 = u[13]
    act0Ttiss2 = u[14]
    restTtiss2 = u[15]
    actTtiss2 = u[16]
    RTXc_ugperkg = u[17]
    RTXp_ugperkg = u[18]
    drug_effect = u[19]
    B1920tiss3 = u[20]
    B19no20tiss3 = u[21]
    restTtiss3 = u[22]
    act0Ttiss3 = u[23]
    actTtiss3 = u[24]
    Blinc_ug = u[25]
    restTtumor = u[26]
    actTtumor = u[27]
    Btumor = u[28]
    act0Ttumor = u[29]
    TDBsc_ugperkg = u[30]
    TDBc_ugperml_AUC = u[31]
    IL6pb = u[32]
    IL6tiss = u[33]
    IL6tiss2 = u[34]
    IL6tiss3 = u[35]
    IL6tumor = u[36]
    VmB = p[1]
    KmTB_kill = p[2]
    KdrugB = p[3]
    kBapop = p[4]
    kBprolif = p[5]
    kTprolif = p[6]
    kBkill = p[7]
    KdrugactT = p[8]
    VmT = p[9]
    KmBT_act = p[10]
    kTaexit = p[11]
    kTact = p[12]
    fTadeact = p[13]
    fTap = p[14]
    Cl_tdb = p[15]
    Cld_tdb = p[16]
    Vc_tdb = p[17]
    Vp_tdb = p[18]
    kTgen = p[19]
    kTaapop = p[20]
    KTrp = p[21]
    Trpbo_perml = p[22]
    fTaprolif = p[23]
    fBprolif = p[24]
    Bpbo_perml = p[25]
    fBexit = p[26]
    KBp = p[27]
    Kp = p[28]
    Vpb = p[29]
    Vtissue = p[30]
    fTrapop = p[31]
    Trpbref_perml = p[32]
    nkill = p[33]
    KTrp2 = p[34]
    Vm_tdb = p[35]
    Km_tdb = p[36]
    BW = p[37]
    act0on = p[38]
    kIL6prod = p[39]
    fa0 = p[40]
    fAICD = p[41]
    kBAFFprod = p[42]
    Bpbref_perml = p[43]
    thBAFF = p[44]
    thalfIL6 = p[45]
    fTa0deact = p[46]
    fTa0apop = p[47]
    BAFFo = p[48]
    fBAFFo = p[49]
    tinjhalf = p[50]
    finj = p[51]
    KBp2 = p[52]
    Vtissue2 = p[53]
    Kp2 = p[54]
    tissue2on = p[55]
    Vc_rtx = p[56]
    Vp_rtx = p[57]
    Cl_rtx = p[58]
    Cld_rtx = p[59]
    Vm_rtx = p[60]
    Km_rtx = p[61]
    Kmkill_rtx = p[62]
    fTgenbl = p[63]
    depleteTpb = p[64]
    depleteBpb = p[65]
    kTrexit = p[66]
    PKflag = p[67]
    VPid = p[68]
    end_time = p[69]
    fdrug = p[70]
    Vtissue3 = p[71]
    KTrp3 = p[72]
    tissue3on = p[73]
    Kp3 = p[74]
    kBtiss3exit = p[75]
    KBp3 = p[76]
    B19no20_B1920_ratio = p[77]
    fvalidation = p[78]
    fapop_v24 = p[79]
    fBtissue3_v1 = p[80]
    kBmat_kBapop_ratio = p[81]
    ndrugactT = p[82]
    Cl_blin = p[83]
    Vz_blin = p[84]
    KdrugactT_blin = p[85]
    ndrugactT_blin = p[86]
    KmTB_kill_blin = p[87]
    KdrugB_blin = p[88]
    nkill_blin = p[89]
    KmBT_act_blin = p[90]
    kBapop_cll = p[91]
    kBgen_cll = p[92]
    fBkill = p[93]
    fTact = p[94]
    fKmTB_kill = p[95]
    Kptumor = p[96]
    Vtumor = p[97]
    KBptumor = p[98]
    KTrptumor = p[99]
    tumor_on = p[100]
    IL6_tiss_contribution = p[101]
    kBtumorprolif = p[102]
    S = p[103]
    tissue1on = p[104]
    kabs_TDB = p[105]
    fbio_TDB = p[106]
    Bcell_tumor_trafficking_on = p[107]
    TDBc_ugperml = real(tdb_central_concentration_symbolic(TDBc_ugperkg, Vc_tdb, PKflag, VPid, t, end_time, fvalidation))
    RTXc_ugperml = real(((RTXc_ugperkg / Vc_rtx > 1.0e-5) * RTXc_ugperkg) / Vc_rtx)
    Blinc_ngperml = real((Blinc_ug / Vz_blin) * 1000)
    restTpb_perml = real(restTpb / Vpb)
    act0Tpb_perml = real(act0Tpb / Vpb)
    actTpb_perml = real(actTpb / Vpb)
    B19tiss3 = real(B19no20tiss3 + B1920tiss3)
    Bpb_perml = real(Bpb / Vpb)
    Btiss_perml = real(Btiss / Vtissue)
    Btiss2_perml = real(Btiss2 / Vtissue2)
    B1920tiss3_perml = real(B1920tiss3 / Vtissue3)
    Btumor_perml = real(Btumor / Vtumor)
    BTrratio_pb = real(Bpb / max(restTpb + act0Tpb, 1))
    BTrratio_tiss = real(Btiss / max(restTtiss + act0Ttiss, 1))
    BTrratio_tiss2 = real(Btiss2 / max(restTtiss2 + act0Ttiss2, 1))
    B1920Trratio_tiss3 = real(B1920tiss3 / max(restTtiss3 + act0Ttiss3, 1))
    BTrratio_tumor = real(Btumor / max(restTtumor + act0Ttumor, 1))
    TaBratio_pb = real(actTpb / max(Bpb, 1))
    TaBratio_tiss = real(actTtiss / max(Btiss, 1))
    TaBratio_tiss2 = real(actTtiss2 / max(Btiss2, 1))
    TaB1920ratio_tiss3 = real(actTtiss3 / max(B1920tiss3, 1))
    TaBratio_tumor = real(actTtumor / max(Btumor, 1))
    TDBt_ugperml = real(Kp * TDBc_ugperml)
    TDBt2_ugperml = real(Kp2 * TDBc_ugperml)
    TDBt3_ugperml = real(tissue3on * Kp3 * TDBc_ugperml)
    TDBtumor_ugperml = real(tumor_on * Kptumor * TDBc_ugperml)
    RTXt_ugperml = real(RTXc_ugperml * Kp)
    RTXt2_ugperml = real(RTXc_ugperml * Kp2)
    RTXt3_ugperml = real(tissue3on * RTXc_ugperml * Kp3)
    RTXtumor_ugperml = real(tumor_on * RTXc_ugperml * Kptumor)
    Blint_ngperml = real(Kp * Blinc_ngperml)
    Blint2_ngperml = real(Kp2 * Blinc_ngperml)
    Blint3_ngperml = real(tissue3on * Kp3 * Blinc_ngperml)
    Blintumor_ngperml = real(tumor_on * Kptumor * Blinc_ngperml)
    totTpb_perml = real(restTpb_perml + act0Tpb_perml + actTpb_perml)
    B19tiss3_perml = real(B19tiss3 / Vtissue3)
    B19Trratio_tiss3 = real(B19tiss3 / max(restTtiss3 + act0Ttiss3, 1))
    TaB19ratio_tiss3 = real(actTtiss3 / max(B19tiss3, 1))
    drugTpbact = real(VmT * (pow_safe_symbolic(BTrratio_pb, S) / (pow_safe_symbolic(KmBT_act, S) + pow_safe_symbolic(BTrratio_pb, S))) * (pow_safe_symbolic(TDBc_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBc_ugperml * 1000, ndrugactT))))
    BlinTpbact = real(VmT * (pow_safe_symbolic(BTrratio_pb, S) / (pow_safe_symbolic(KmBT_act_blin, S) + pow_safe_symbolic(BTrratio_pb, S))) * (pow_safe_symbolic(Blinc_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blinc_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    drugBpbkill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_pb), nkill)) / (pow_safe_symbolic(max(0, KmTB_kill), nkill) + pow_safe_symbolic(max(0, TaBratio_pb), nkill))) * ((TDBc_ugperml * 1000) / (KdrugB + TDBc_ugperml * 1000)))
    BlinBpbkill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_pb), nkill_blin)) / (pow_safe_symbolic(max(0, KmTB_kill_blin), nkill_blin) + pow_safe_symbolic(max(0, TaBratio_pb), nkill_blin))) * (Blinc_ngperml / (KdrugB_blin + Blinc_ngperml)))
    drugTtissact = real(fTact * VmT * (pow_safe_symbolic(BTrratio_tiss, S) / (pow_safe_symbolic(KmBT_act, S) + pow_safe_symbolic(BTrratio_tiss, S))) * (pow_safe_symbolic(TDBt_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt_ugperml * 1000, ndrugactT))))
    drugBtisskill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tiss), nkill)) / (pow_safe_symbolic(max(0, KmTB_kill * fKmTB_kill), nkill) + pow_safe_symbolic(max(0, TaBratio_tiss), nkill))) * ((TDBt_ugperml * 1000) / (KdrugB + TDBt_ugperml * 1000)))
    drugTtissact2 = real(fTact * VmT * (pow_safe_symbolic(BTrratio_tiss2, S) / (pow_safe_symbolic(KmBT_act, S) + pow_safe_symbolic(BTrratio_tiss2, S))) * (pow_safe_symbolic(TDBt2_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt2_ugperml * 1000, ndrugactT))))
    drugBtisskill2 = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tiss2), nkill)) / (pow_safe_symbolic(max(0, KmTB_kill * fKmTB_kill), nkill) + pow_safe_symbolic(max(0, TaBratio_tiss2), nkill))) * ((TDBt2_ugperml * 1000) / (KdrugB + TDBt2_ugperml * 1000)))
    drugTtissact3 = real(fTact * VmT * (pow_safe_symbolic(B1920Trratio_tiss3, S) / (pow_safe_symbolic(KmBT_act, S) + pow_safe_symbolic(B1920Trratio_tiss3, S))) * (pow_safe_symbolic(TDBt3_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt3_ugperml * 1000, ndrugactT))))
    drugBtisskill3 = real(((VmB * pow_safe_symbolic(max(0, TaB1920ratio_tiss3), nkill)) / (pow_safe_symbolic(max(0, KmTB_kill * fKmTB_kill), nkill) + pow_safe_symbolic(max(0, TaB1920ratio_tiss3), nkill))) * ((TDBt3_ugperml * 1000) / (KdrugB + TDBt3_ugperml * 1000)))
    drugTtumoract = real(fTact * VmT * (pow_safe_symbolic(max(BTrratio_tumor, 0), S) / (pow_safe_symbolic(KmBT_act, S) + pow_safe_symbolic(max(BTrratio_tumor, 0), S))) * (pow_safe_symbolic(TDBtumor_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBtumor_ugperml * 1000, ndrugactT))))
    drugBtumorkill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tumor), nkill)) / (pow_safe_symbolic(max(0, KmTB_kill * fKmTB_kill), nkill) + pow_safe_symbolic(max(0, TaBratio_tumor), nkill))) * ((TDBtumor_ugperml * 1000) / (KdrugB + TDBtumor_ugperml * 1000)))
    BlinTtissact = real(fTact * VmT * (pow_safe_symbolic(BTrratio_tiss, S) / (pow_safe_symbolic(KmBT_act_blin, S) + pow_safe_symbolic(BTrratio_tiss, S))) * (pow_safe_symbolic(Blint_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    BlinBtisskill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tiss), nkill_blin)) / (pow_safe_symbolic(max(0, KmTB_kill_blin * fKmTB_kill), nkill_blin) + pow_safe_symbolic(max(0, TaBratio_tiss), nkill_blin))) * (Blint_ngperml / (KdrugB_blin + Blint_ngperml)))
    BlinTtissact2 = real(fTact * VmT * (pow_safe_symbolic(BTrratio_tiss2, S) / (pow_safe_symbolic(KmBT_act_blin, S) + pow_safe_symbolic(BTrratio_tiss2, S))) * (pow_safe_symbolic(Blint2_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint2_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    BlinBtisskill2 = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tiss2), nkill_blin)) / (pow_safe_symbolic(max(0, KmTB_kill_blin * fKmTB_kill), nkill_blin) + pow_safe_symbolic(max(0, TaBratio_tiss2), nkill_blin))) * (Blint2_ngperml / (KdrugB_blin + Blint2_ngperml)))
    BlinTtumoract = real(fTact * VmT * (pow_safe_symbolic(max(BTrratio_tumor, 0), S) / (pow_safe_symbolic(KmBT_act_blin, S) + pow_safe_symbolic(max(BTrratio_tumor, 0), S))) * (pow_safe_symbolic(Blintumor_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blintumor_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    BlinBtumorkill = real(((VmB * pow_safe_symbolic(max(0, TaBratio_tumor), nkill_blin)) / (pow_safe_symbolic(max(0, KmTB_kill_blin * fKmTB_kill), nkill_blin) + pow_safe_symbolic(max(0, TaBratio_tumor), nkill_blin))) * (Blintumor_ngperml / (KdrugB_blin + Blintumor_ngperml)))
    BlinTtissact3 = real(fTact * VmT * (pow_safe_symbolic(B19Trratio_tiss3, S) / (pow_safe_symbolic(KmBT_act_blin, S) + pow_safe_symbolic(B19Trratio_tiss3, S))) * (pow_safe_symbolic(Blint3_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint3_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    BlinBtisskill3 = real(((VmB * pow_safe_symbolic(max(0, TaB19ratio_tiss3), nkill_blin)) / (pow_safe_symbolic(max(0, KmTB_kill_blin * fKmTB_kill), nkill_blin) + pow_safe_symbolic(max(0, TaB19ratio_tiss3), nkill_blin))) * (Blint3_ngperml / (KdrugB_blin + Blint3_ngperml)))
    fill!(du, zero(eltype(u)))
    rate_1 = real(kBprolif * KBp * Bpbref_perml * Vtissue * pow_safe_symbolic(max(0, 1 - Btiss / (KBp * Bpbref_perml * Vtissue)), 1))
    du[2] += 1.0 * rate_1
    rate_2 = real(kTact * ((drugTtissact + BlinTtissact) * restTtiss - fTadeact * actTtiss))
    du[1] += 1.0 * rate_2
    du[5] += -1.0 * rate_2
    rate_3 = real(0 * fBprolif * kBprolif * Btiss * kBapop_cll)
    du[2] += -1.0 * rate_3
    rate_4 = real(((Cl_tdb + (Vm_tdb / (Km_tdb + TDBc_ugperkg / Vc_tdb)) / BW) / Vc_tdb) * TDBc_ugperkg)
    du[3] += -1.0 * rate_4
    rate_5 = real(Cld_tdb * (TDBc_ugperkg / Vc_tdb - TDBp_ugperkg / Vp_tdb))
    du[3] += -1.0 * rate_5
    du[6] += 1.0 * rate_5
    rate_6 = real(0)
    du[5] += 1.0 * rate_6
    rate_7 = real(0)
    rate_8 = real(tissue1on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * restTpb) / Vpb) * KTrp - restTtiss / Vtissue))
    du[5] += 1.0 * rate_8
    du[7] += -1.0 * rate_8
    rate_9 = real(kTgen * Vpb * Trpbref_perml * (fTgenbl + pow_safe_symbolic(max(0, 1 - totTpb_perml / Trpbref_perml), 2)))
    du[7] += 1.0 * rate_9
    rate_10 = real(tissue1on * fBexit * kTaexit * Vpb * max(0, (Bpb / Vpb) * KBp - Btiss / Vtissue))
    du[2] += 1.0 * rate_10
    du[8] += -1.0 * rate_10
    rate_11 = real(kTact * ((drugTpbact + BlinTpbact) * restTpb - fTadeact * actTpb))
    du[4] += 1.0 * rate_11
    du[7] += -1.0 * rate_11
    rate_12 = real(tissue3on * fBtissue3_v1 * kBapop * Bpb * kBapop_cll)
    du[8] += 1.0 * rate_12
    rate_13 = real(kBapop * Bpb * kBapop_cll)
    du[8] += -1.0 * rate_13
    rate_14 = real(0)
    rate_15 = real(0)
    rate_16 = real(kBkill * (drugBpbkill + BlinBpbkill + RTXc_ugperml / (Kmkill_rtx / 1000 + RTXc_ugperml)) * Bpb)
    du[8] += -1.0 * rate_16
    rate_17 = real(fBkill * kBkill * (drugBtisskill + BlinBtisskill + RTXt_ugperml / (Kmkill_rtx / 1000 + RTXt_ugperml)) * Btiss)
    du[2] += -1.0 * rate_17
    rate_18 = real(kTgen * fTgenbl * restTpb + fTrapop * kTaapop * max(0, restTpb - Trpbref_perml * Vpb))
    du[7] += -1.0 * rate_18
    rate_19 = real(tissue1on * kTaexit * Vpb * ((((1 + fa0 * (finj * injection_effect + fdrug * drug_effect)) * actTpb) / Vpb) * KTrp * fTap - actTtiss / Vtissue))
    du[1] += 1.0 * rate_19
    du[4] += -1.0 * rate_19
    rate_20 = real(kTaapop * (actTpb + (fAICD * pow_safe_symbolic(actTpb, 2)) / (Vpb * Trpbref_perml)))
    du[4] += -1.0 * rate_20
    rate_21 = real((log(2) / ((thBAFF / 24) / 60)) * ((fBAFFo * BAFF) / BAFFo + ((1 - fBAFFo) * (Btiss + Bpb + Btiss2)) / (Bpbref_perml * (Vpb + KBp * Vtissue + KBp2 * Vtissue2))))
    du[9] += -1.0 * rate_21
    rate_22 = real((log(2) / ((thBAFF / 24) / 60)) * (fBAFFo + ((1 - fBAFFo) * Bpbo_perml) / Bpbref_perml))
    du[9] += 1.0 * rate_22
    rate_23 = real(act0on * fTa0deact * act0Tpb)
    du[7] += 1.0 * rate_23
    du[10] += -1.0 * rate_23
    rate_24 = real(act0on * kTact * ((drugTpbact + BlinTpbact) * act0Tpb - fTadeact * actTpb))
    du[4] += 1.0 * rate_24
    du[10] += -1.0 * rate_24
    rate_25 = real(fTa0apop * kTaapop * act0Tpb)
    du[10] += -1.0 * rate_25
    rate_26 = real(fTa0apop * kTaapop * act0Ttiss)
    du[11] += -1.0 * rate_26
    rate_27 = real(act0on * kTact * ((drugTtissact + BlinTtissact) * act0Ttiss - fTadeact * actTtiss))
    du[1] += 1.0 * rate_27
    du[11] += -1.0 * rate_27
    rate_28 = real(act0on * fTa0deact * act0Ttiss)
    du[5] += 1.0 * rate_28
    du[11] += -1.0 * rate_28
    rate_29 = real(tissue1on * act0on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * act0Tpb) / Vpb) * KTrp - act0Ttiss / Vtissue))
    du[10] += -1.0 * rate_29
    du[11] += 1.0 * rate_29
    rate_30 = real(act0on * fTaprolif * kTprolif * actTpb)
    du[10] += 1.0 * rate_30
    rate_31 = real(act0on * fTaprolif * kTprolif * actTtiss)
    du[11] += 1.0 * rate_31
    rate_32 = real((log(2) / tinjhalf) * injection_effect)
    du[12] += -1.0 * rate_32
    rate_33 = real(act0on * fTaprolif * kTprolif * actTtiss2)
    du[14] += 1.0 * rate_33
    rate_34 = real(tissue2on * act0on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * act0Tpb) / Vpb) * KTrp2 - act0Ttiss2 / Vtissue2))
    du[10] += -1.0 * rate_34
    du[14] += 1.0 * rate_34
    rate_35 = real(act0on * fTa0deact * act0Ttiss2)
    du[14] += -1.0 * rate_35
    du[15] += 1.0 * rate_35
    rate_36 = real(act0on * kTact * ((drugTtissact2 + BlinTtissact2) * act0Ttiss2 - fTadeact * actTtiss2))
    du[14] += -1.0 * rate_36
    du[16] += 1.0 * rate_36
    rate_37 = real(fTa0apop * kTaapop * act0Ttiss2)
    du[14] += -1.0 * rate_37
    rate_38 = real(tissue2on * kTaexit * Vpb * ((((1 + fa0 * (finj * injection_effect + fdrug * drug_effect)) * actTpb) / Vpb) * KTrp2 * fTap - actTtiss2 / Vtissue2))
    du[4] += -1.0 * rate_38
    du[16] += 1.0 * rate_38
    rate_39 = real(fBkill * kBkill * (drugBtisskill2 + BlinBtisskill2 + RTXt2_ugperml / (Kmkill_rtx / 1000 + RTXt2_ugperml)) * Btiss2)
    du[13] += -1.0 * rate_39
    rate_40 = real(tissue2on * fBexit * kTaexit * Vpb * max(0, (Bpb / Vpb) * KBp2 - Btiss2 / Vtissue2))
    du[8] += -1.0 * rate_40
    du[13] += 1.0 * rate_40
    rate_41 = real(tissue2on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * restTpb) / Vpb) * KTrp2 - restTtiss2 / Vtissue2))
    du[7] += -1.0 * rate_41
    du[15] += 1.0 * rate_41
    rate_42 = real(0)
    rate_43 = real(0)
    rate_44 = real(0 * fBprolif * kBprolif * Btiss2 * kBapop_cll)
    du[13] += -1.0 * rate_44
    rate_45 = real(kTact * ((drugTtissact2 + BlinTtissact2) * restTtiss2 - fTadeact * actTtiss2))
    du[15] += -1.0 * rate_45
    du[16] += 1.0 * rate_45
    rate_46 = real(kBprolif * KBp2 * Bpbref_perml * Vtissue2 * pow_safe_symbolic(max(0, 1 - Btiss2 / (KBp2 * Bpbref_perml * Vtissue2)), 1))
    du[13] += 1.0 * rate_46
    rate_47 = real(Cld_rtx * (RTXc_ugperkg / Vc_rtx - RTXp_ugperkg / Vp_rtx))
    du[17] += -1.0 * rate_47
    du[18] += 1.0 * rate_47
    rate_48 = real(((Cl_rtx + (Vm_rtx / (Km_rtx + RTXc_ugperkg / Vc_rtx)) / BW) / Vc_rtx) * RTXc_ugperkg)
    du[17] += -1.0 * rate_48
    rate_49 = real((log(2) / tinjhalf) * drug_effect)
    du[19] += -1.0 * rate_49
    rate_50 = real(Bpbref_perml * KBp3 * B19no20_B1920_ratio * Vtissue3 * (kBapop + kBapop / kBmat_kBapop_ratio) * (1 + 5 * pow_safe_symbolic(max(0, 1 - (Bpb + Btiss + Btiss2) / (Bpbref_perml * Vpb + KBp * Bpbo_perml * Vtissue + KBp2 * Bpbo_perml * Vtissue2)), 1)))
    du[21] += 1.0 * rate_50
    rate_51 = real((kBapop / kBmat_kBapop_ratio) * B19no20tiss3)
    du[20] += 1.0 * rate_51
    du[21] += -1.0 * rate_51
    rate_52 = real(tissue3on * fBtissue3_v1 * kBtiss3exit * Vpb * max(0, B1920tiss3 / Vtissue3 - (Bpb / Vpb) * KBp3))
    du[8] += 1.0 * rate_52
    du[20] += -1.0 * rate_52
    rate_53 = real(kBapop * B19no20tiss3 * kBapop_cll)
    du[21] += -1.0 * rate_53
    rate_54 = real(kBapop * B1920tiss3 * kBapop_cll)
    du[20] += -1.0 * rate_54
    rate_55 = real(tissue3on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * restTpb) / Vpb) * KTrp3 - restTtiss3 / Vtissue3))
    du[7] += -1.0 * rate_55
    du[22] += 1.0 * rate_55
    rate_56 = real(fBkill * kBkill * (drugBtisskill3 + BlinBtisskill3 + RTXt3_ugperml / (Kmkill_rtx / 1000 + RTXt3_ugperml)) * B1920tiss3)
    du[20] += -1.0 * rate_56
    rate_57 = real(tissue3on * act0on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * act0Tpb) / Vpb) * KTrp3 - act0Ttiss3 / Vtissue3))
    du[10] += -1.0 * rate_57
    du[23] += 1.0 * rate_57
    rate_58 = real(tissue3on * kTaexit * Vpb * ((((1 + fa0 * (finj * injection_effect + fdrug * drug_effect)) * actTpb) / Vpb) * KTrp3 * fTap - actTtiss3 / Vtissue3))
    du[4] += -1.0 * rate_58
    du[24] += 1.0 * rate_58
    rate_59 = real(fTa0apop * kTaapop * act0Ttiss3)
    du[23] += -1.0 * rate_59
    rate_60 = real(act0on * kTact * ((drugTtissact3 + BlinTtissact3) * act0Ttiss3 - fTadeact * actTtiss3))
    du[23] += -1.0 * rate_60
    du[24] += 1.0 * rate_60
    rate_61 = real(act0on * fTa0deact * act0Ttiss3)
    du[22] += 1.0 * rate_61
    du[23] += -1.0 * rate_61
    rate_62 = real(kTact * ((drugTtissact3 + BlinTtissact3) * restTtiss3 - fTadeact * actTtiss3))
    du[22] += -1.0 * rate_62
    du[24] += 1.0 * rate_62
    rate_63 = real(act0on * fTaprolif * kTprolif * actTtiss3)
    du[23] += 1.0 * rate_63
    rate_64 = real(0)
    rate_65 = real(fBkill * kBkill * BlinBtisskill3 * B19no20tiss3)
    du[21] += -1.0 * rate_65
    rate_66 = real(0)
    rate_67 = real(0)
    rate_68 = real(fapop_v24 * fTrapop * kTaapop * (Trpbref_perml * Vtissue * KTrp) * max(0, restTtiss / (Trpbref_perml * Vtissue * KTrp) - 1))
    du[5] += -1.0 * rate_68
    rate_69 = real(fapop_v24 * fTrapop * kTaapop * (Trpbref_perml * Vtissue2 * KTrp2) * max(0, restTtiss2 / (Trpbref_perml * Vtissue2 * KTrp2) - 1))
    du[15] += -1.0 * rate_69
    rate_70 = real(fapop_v24 * kTaapop * (actTtiss + (fAICD * pow_safe_symbolic(actTtiss, 2)) / (KTrp * Vtissue * Trpbref_perml)))
    du[1] += -1.0 * rate_70
    rate_71 = real(fapop_v24 * kTaapop * (actTtiss2 + (fAICD * pow_safe_symbolic(actTtiss2, 2)) / (Vtissue2 * KTrp2 * Trpbref_perml)))
    du[16] += -1.0 * rate_71
    rate_72 = real(fapop_v24 * fTrapop * kTaapop * (Trpbref_perml * Vtissue3 * KTrp3) * max(0, restTtiss3 / (Trpbref_perml * Vtissue3 * KTrp3) - 1))
    du[22] += -1.0 * rate_72
    rate_73 = real(fapop_v24 * kTaapop * (actTtiss3 + (fAICD * pow_safe_symbolic(actTtiss3, 2)) / (Vtissue3 * KTrp3 * Trpbref_perml)))
    du[24] += -1.0 * rate_73
    rate_74 = real(tissue3on * fBtissue3_v1 * kBprolif * kBapop_cll * Bpbref_perml * Vpb * pow_safe_symbolic(max(0, 1 - Bpb / (Bpbref_perml * Vpb)), 1))
    du[8] += 1.0 * rate_74
    rate_75 = real((Cl_blin * Blinc_ug) / Vz_blin)
    du[25] += -1.0 * rate_75
    rate_76 = real(0)
    rate_77 = real(0)
    rate_78 = real(0)
    rate_79 = real(0)
    rate_80 = real(0)
    rate_81 = real(0)
    rate_82 = real(0)
    rate_83 = real(0)
    rate_84 = real(kBprolif * KBp3 * Bpbref_perml * Vtissue3 * pow_safe_symbolic(max(0, 1 - B1920tiss3 / (KBp3 * Bpbref_perml * Vtissue3)), 1))
    du[20] += 1.0 * rate_84
    rate_85 = real(kBprolif * KBp3 * Bpbref_perml * B19no20_B1920_ratio * Vtissue3 * pow_safe_symbolic(max(0, 1 - B19no20tiss3 / (KBp3 * Bpbref_perml * B19no20_B1920_ratio * Vtissue3)), 1))
    du[21] += 1.0 * rate_85
    rate_86 = real(act0on * fTa0deact * act0Ttumor)
    du[26] += 1.0 * rate_86
    du[29] += -1.0 * rate_86
    rate_87 = real(act0on * kTact * ((drugTtumoract + BlinTtumoract) * act0Ttumor - fTadeact * actTtumor))
    du[27] += 1.0 * rate_87
    du[29] += -1.0 * rate_87
    rate_88 = real(act0on * fTaprolif * kTprolif * actTtumor)
    du[29] += 1.0 * rate_88
    rate_89 = real(kTact * ((drugTtumoract + BlinTtumoract) * restTtumor - fTadeact * actTtumor))
    du[26] += -1.0 * rate_89
    du[27] += 1.0 * rate_89
    rate_90 = real(0)
    rate_91 = real(0)
    rate_92 = real(0)
    rate_93 = real(0)
    rate_94 = real(kBtumorprolif * Btumor + 0 * kBprolif * KBptumor * Bpbref_perml * Vtumor * pow_safe_symbolic(max(0, 1 - Btumor / (KBptumor * Bpbref_perml * Vtumor)), 1))
    du[28] += 1.0 * rate_94
    rate_95 = real(0 * fBprolif * kBprolif * Btumor)
    du[28] += -1.0 * rate_95
    rate_96 = real(fBkill * kBkill * (drugBtumorkill + BlinBtumorkill + RTXtumor_ugperml / (Kmkill_rtx / 1000 + RTXtumor_ugperml)) * Btumor)
    du[28] += -1.0 * rate_96
    rate_97 = real(fapop_v24 * kTaapop * (actTtumor + (fAICD * pow_safe_symbolic(actTtumor, 2)) / (Vtumor * KTrptumor * Trpbref_perml)))
    du[27] += -1.0 * rate_97
    rate_98 = real(Bcell_tumor_trafficking_on * tumor_on * fBexit * kTaexit * Vpb * max(0, (Bpb / Vpb) * KBptumor - Btumor / Vtumor))
    du[8] += -1.0 * rate_98
    du[28] += 1.0 * rate_98
    rate_99 = real(tumor_on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * restTpb) / Vpb) * KTrptumor - restTtumor / Vtumor))
    du[7] += -1.0 * rate_99
    du[26] += 1.0 * rate_99
    rate_100 = real(tumor_on * kTaexit * Vpb * ((((1 + fa0 * (finj * injection_effect + fdrug * drug_effect)) * actTpb) / Vpb) * KTrptumor * fTap - actTtumor / Vtumor))
    du[4] += -1.0 * rate_100
    du[27] += 1.0 * rate_100
    rate_101 = real(tumor_on * act0on * kTrexit * Vpb * ((((1 + finj * injection_effect + fdrug * drug_effect) * act0Tpb) / Vpb) * KTrptumor - act0Ttumor / Vtumor))
    du[10] += -1.0 * rate_101
    du[29] += 1.0 * rate_101
    rate_102 = real(fTa0apop * kTaapop * act0Ttumor)
    du[29] += -1.0 * rate_102
    rate_103 = real(fapop_v24 * fTrapop * kTaapop * (Trpbref_perml * Vtumor * KTrptumor) * max(0, restTtumor / (Trpbref_perml * Vtumor * KTrptumor) - 1))
    du[26] += -1.0 * rate_103
    rate_104 = real(kabs_TDB * fbio_TDB * TDBsc_ugperkg)
    du[3] += 1.0 * rate_104
    du[30] += -1.0 * rate_104
    rate_105 = real(kabs_TDB * (1 - fbio_TDB) * TDBsc_ugperkg)
    du[30] += -1.0 * rate_105
    rate_106 = real(TDBc_ugperml)
    du[31] += 1.0 * rate_106
    rate_107 = real(((kIL6prod * actTpb) / Vpb) * (Bpb_perml / Bpbref_perml) * (pow_safe_symbolic(TDBc_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBc_ugperml * 1000, ndrugactT)) + pow_safe_symbolic(Blinc_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blinc_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    du[32] += 1.0 * rate_107
    rate_108 = real((log(2) / ((thalfIL6 / 60) / 24)) * IL6pb)
    du[32] += -1.0 * rate_108
    rate_109 = real(((kIL6prod * actTtiss) / Vtissue) * (Btiss_perml / (Bpbref_perml * KBp)) * (pow_safe_symbolic(TDBt_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt_ugperml * 1000, ndrugactT)) + pow_safe_symbolic(Blint_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    du[33] += 1.0 * rate_109
    rate_110 = real((log(2) / ((thalfIL6 / 60) / 24)) * IL6tiss)
    du[33] += -1.0 * rate_110
    rate_111 = real(((kIL6prod * actTtiss2) / Vtissue2) * (Btiss2_perml / (Bpbref_perml * KBp2)) * (pow_safe_symbolic(TDBt2_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt2_ugperml * 1000, ndrugactT)) + pow_safe_symbolic(Blint2_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint2_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    du[34] += 1.0 * rate_111
    rate_112 = real((log(2) / ((thalfIL6 / 60) / 24)) * IL6tiss2)
    du[34] += -1.0 * rate_112
    rate_113 = real(((kIL6prod * actTtiss3) / Vtissue3) * ((B1920tiss3_perml / (Bpbref_perml * KBp3)) * (pow_safe_symbolic(TDBt3_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBt3_ugperml * 1000, ndrugactT))) + (B19tiss3_perml / (Bpbref_perml * KBp3 * (1 + B19no20_B1920_ratio))) * (pow_safe_symbolic(Blint3_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blint3_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin)))))
    du[35] += 1.0 * rate_113
    rate_114 = real((log(2) / ((thalfIL6 / 60) / 24)) * IL6tiss3)
    du[35] += -1.0 * rate_114
    rate_115 = real(((kIL6prod * actTtumor) / Vtumor) * (Btumor_perml / (Bpbref_perml * KBptumor)) * (pow_safe_symbolic(TDBtumor_ugperml * 1000, ndrugactT) / (pow_safe_symbolic(KdrugactT, ndrugactT) + pow_safe_symbolic(TDBtumor_ugperml * 1000, ndrugactT)) + pow_safe_symbolic(Blintumor_ngperml, ndrugactT_blin) / (pow_safe_symbolic(Blintumor_ngperml, ndrugactT_blin) + pow_safe_symbolic(KdrugactT_blin, ndrugactT_blin))))
    du[36] += 1.0 * rate_115
    rate_116 = real((log(2) / ((thalfIL6 / 60) / 24)) * IL6tumor)
    du[36] += -1.0 * rate_116
    sanitize_ad_vector!(du)
    return nothing
end

function mosun_rhs!(du, u::AbstractVector{<:Real}, ctx::MosunProblemContext, t)
    return mosun_rhs!(du, u, ctx.params, t, ctx.cache; active_rates = ctx.active_rates)
end

function push_event_delta!(event_map::Dict{Float64, Vector{Tuple{Int,T,T}}}, t::Float64, entry::Tuple{Int,T,T}) where {T<:Real}
    if !haskey(event_map, t)
        event_map[t] = Tuple{Int,T,T}[]
    end
    push!(event_map[t], entry)
    return event_map
end

function event_items_at(event_map::Dict{Float64, Vector{Tuple{Int,T,T}}}, t::Real) where {T<:Real}
    if haskey(event_map, t)
        return event_map[t]
    end
    for (tt, items) in event_map
        if isapprox(t, tt; atol = 1e-8, rtol = 0.0)
            return items
        end
    end
    return nothing
end

function regimen_to_event_map(regimen::MosunRegimen{T}) where {T<:Real}
    event_map = Dict{Float64, Vector{Tuple{Int,T,T}}}()
    for ev in regimen.events
        idx = dynamic_state_index(ev.target)
        rate_val = event_scalar_value(ev.rate)
        if iszero(rate_val)
            push_event_delta!(event_map, ev.time, (idx, ev.amount, zero(T)))
        else
            rate_val > 0.0 || throw(ArgumentError("Infusion rates must be positive"))
            duration = event_scalar_value(ev.amount / ev.rate)
            duration >= 0.0 || throw(ArgumentError("Infusion amount/rate produced negative duration"))
            t_end = ev.time + duration
            push_event_delta!(event_map, ev.time, (idx, zero(T), ev.rate))
            push_event_delta!(event_map, t_end, (idx, zero(T), -ev.rate))
        end
    end
    return event_map
end

"""Apply the differentiable jump map `u⁺ = u⁻ + amount` at a fixed time."""
function apply_event_deltas!(u::AbstractVector, active_rates::AbstractVector, event_map::Dict{Float64, Vector{Tuple{Int,T,T}}}, t::Real) where {T<:Real}
    items = event_items_at(event_map, t)
    if items !== nothing
        for (idx, amt_delta, rate_delta) in items
            u[idx] += oftype(u[idx], amt_delta)
            active_rates[idx] += oftype(active_rates[idx], rate_delta)
        end
    end
    return u, active_rates
end

function build_problem(regimen::MosunRegimen{Treg}, p::MosunParams; tspan::Tuple{Float64,Float64} = (0.0, 84.0), saveat = nothing, callback_mode::Symbol = :segmented, post_event_proposed_dt = nothing) where {Treg<:Real}
    callback_mode in (:segmented, :callback) || throw(ArgumentError("Unsupported callback_mode=$(callback_mode)"))
    event_map = regimen_to_event_map(regimen)
    Tactive = promote_type(Float64, Treg)
    u0 = Tactive.(pack_state(initial_state(p)))
    active_rates = zeros(Tactive, DYNAMIC_STATE_COUNT)
    t0 = tspan[1]
    if event_items_at(event_map, t0) !== nothing
        apply_event_deltas!(u0, active_rates, event_map, t0)
        delete!(event_map, t0)
    end
    ctx = MosunProblemContext(params = p, cache = zero_observables_cache(), active_rates = active_rates)
    prob = ODEProblem(mosun_rhs!, u0, tspan, ctx)
    cb_times = sort(collect(keys(event_map)))
    callback = nothing
    tstops = copy(cb_times)
    d_discontinuities = copy(cb_times)
    if !isempty(cb_times)
        function affect!(integrator)
            apply_event_deltas!(integrator.u, integrator.p.active_rates, event_map, Float64(integrator.t))
            if !isnothing(post_event_proposed_dt)
                SciMLBase.set_proposed_dt!(integrator, post_event_proposed_dt)
            end
        end
        callback = PresetTimeCallback(cb_times, affect!; save_positions = (false, false))
    end
    return MosunBuiltProblem(prob = prob, ctx = ctx, initial_u = u0, callback = callback, tstops = tstops, d_discontinuities = d_discontinuities, event_map = event_map, saveat = saveat, regimen = regimen, callback_mode = callback_mode == :segmented ? :callback : callback_mode)
end

function build_problem_vector(regimen::MosunRegimen{Treg}, p::AbstractVector; tspan::Tuple{Float64,Float64} = (0.0, 84.0), saveat = nothing, callback_mode::Symbol = :segmented, post_event_proposed_dt = nothing, u0 = initial_state_vector(p)) where {Treg<:Real}
    callback_mode in (:segmented, :callback) || throw(ArgumentError("Unsupported callback_mode=$(callback_mode)"))
    event_map = regimen_to_event_map(regimen)
    Tactive = promote_type(eltype(u0), eltype(p), Treg)
    u0v = Tactive.(copy(u0))
    active_rates = zeros(Tactive, DYNAMIC_STATE_COUNT)
    t0 = tspan[1]
    if event_items_at(event_map, t0) !== nothing
        apply_event_deltas!(u0v, active_rates, event_map, t0)
        delete!(event_map, t0)
    end
    rhs = let active_rates_ref = active_rates
        (du, u, pvec, t) -> begin
            mosun_rhs_vector!(du, u, pvec, t)
            @inbounds for i in eachindex(du, active_rates_ref)
                du[i] += active_rates_ref[i]
            end
            sanitize_ad_vector!(du)
            return nothing
        end
    end
    prob = ODEProblem(rhs, u0v, tspan, copy(p))
    cb_times = sort(collect(keys(event_map)))
    callback = nothing
    tstops = copy(cb_times)
    d_discontinuities = copy(cb_times)
    if !isempty(cb_times)
        function affect!(integrator)
            apply_event_deltas!(integrator.u, active_rates, event_map, integrator.t)
            if !isnothing(post_event_proposed_dt)
                SciMLBase.set_proposed_dt!(integrator, post_event_proposed_dt)
            end
        end
        callback = PresetTimeCallback(cb_times, affect!; save_positions = (false, false))
    end
    return MosunVectorBuiltProblem(prob = prob, initial_u = u0v, callback = callback, tstops = tstops, d_discontinuities = d_discontinuities, event_map = event_map, saveat = saveat, regimen = regimen, callback_mode = callback_mode == :segmented ? :callback : callback_mode)
end

function solve_problem(built::MosunBuiltProblem, alg; abstol = 1e-8, reltol = 1e-5, kwargs...)
    if isnothing(built.saveat)
        return solve(built.prob, alg; abstol = abstol, reltol = reltol, callback = built.callback, tstops = built.tstops, d_discontinuities = built.d_discontinuities, kwargs...)
    end
    return solve(built.prob, alg; abstol = abstol, reltol = reltol, callback = built.callback, tstops = built.tstops, d_discontinuities = built.d_discontinuities, saveat = built.saveat, kwargs...)
end

function solve_problem(built::MosunVectorBuiltProblem, alg; abstol = 1e-8, reltol = 1e-5, kwargs...)
    if isnothing(built.saveat)
        return solve(built.prob, alg; abstol = abstol, reltol = reltol, callback = built.callback, tstops = built.tstops, d_discontinuities = built.d_discontinuities, kwargs...)
    end
    return solve(built.prob, alg; abstol = abstol, reltol = reltol, callback = built.callback, tstops = built.tstops, d_discontinuities = built.d_discontinuities, saveat = built.saveat, kwargs...)
end

function solve_regimen(regimen::MosunRegimen, p::MosunParams, alg; tspan::Tuple{Float64,Float64} = (0.0, 84.0), saveat = nothing, callback_mode::Symbol = :callback, post_event_proposed_dt = nothing, abstol = 1e-8, reltol = 1e-5, kwargs...)
    built = build_problem(regimen, p; tspan = tspan, saveat = saveat, callback_mode = callback_mode, post_event_proposed_dt = post_event_proposed_dt)
    sol = solve_problem(built, alg; abstol = abstol, reltol = reltol, kwargs...)
    return built, sol
end

function solve_regimen_vector(regimen::MosunRegimen, p::AbstractVector, alg; tspan::Tuple{Float64,Float64} = (0.0, 84.0), saveat = nothing, callback_mode::Symbol = :callback, post_event_proposed_dt = nothing, abstol = 1e-8, reltol = 1e-5, kwargs...)
    built = build_problem_vector(regimen, p; tspan = tspan, saveat = saveat, callback_mode = callback_mode, post_event_proposed_dt = post_event_proposed_dt)
    sol = solve_problem(built, alg; abstol = abstol, reltol = reltol, kwargs...)
    return built, sol
end

export MosunParams, MosunDynamicState, MosunObservablesCache, MosunRegimenEvent, MosunRegimen, MosunProblemContext, MosunBuiltProblem, MosunVectorBuiltProblem,
    DYNAMIC_STATE_NAMES, OBSERVABLE_NAMES, RHS_OBSERVABLE_NAMES, PARAMETER_NAMES, DEAD_LEGACY_NAMES, DYNAMIC_STATE_COUNT, OBSERVABLE_COUNT, RHS_OBSERVABLE_COUNT,
    REPO_ROOT, SOLVER_ABSTOL, SOLVER_RELTOL, make_solver_alg,
    default_params, zero_observables_cache, has_parameter, set_param!, params_from_named_values, params_from_dict, bolus_regimen,
    parameter_index, default_param_vector, initial_state, initial_state_vector, pack_state, pack_params, unpack_state, update_observables!, value_at, mosun_rhs!, mosun_rhs_vector!, build_problem, build_problem_vector, solve_problem, solve_regimen, solve_regimen_vector,
    observable, dynamic_state_value, state_or_observable, dynamic_state_index, regimen_to_event_map, apply_event_deltas!

end # module
