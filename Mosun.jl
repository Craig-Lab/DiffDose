#!/usr/bin/env julia

"""
Individualized mosunetuzumab dose-amplitude optimization.

This entry point loads the committed VPop, translates each six-dose vector into
fixed-time bolus events, solves the 36-state QSP model, evaluates the tumor,
IL-6, and dose-target objective, obtains its gradient with ForwardDiff, and
optimizes each virtual patient with `Fminbox(BFGS())`. The final output also
reports the six post hoc endpoints used in the manuscript's endpoint-win
analysis: peak IL-6, IL-6 AUC, day-84 tumor, tumor AUC, total dose, and drug
exposure.

Quick smoke test:

    julia +1.12.5 --project=environments/mosun Mosun.jl --preset quick --patients 1 --output output

Manuscript configuration:

    julia +1.12.5 --project=environments/mosun Mosun.jl --preset paper --patients all --output output

Original MATLAB/SimBiology model source:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-020-00145-7/MediaObjects/41540_2020_145_MOESM3_ESM.zip

Scientific model sources:
Hosseini et al. (2020), https://doi.org/10.1038/s41540-020-00145-7
Susilo et al. (2023), https://doi.org/10.1111/cts.13501
"""

using CSV
using DataFrames
using DifferentialEquations
using ForwardDiff
using Optim
using Random
using SciMLBase
using Statistics

include(joinpath(@__DIR__, "utils", "MosunModel.jl"))
using .MosunModel

const MMC = MosunModel
const DEFAULT_VPOP = joinpath(@__DIR__, "data", "MosunVPop250.csv")
const DEFAULT_PARAMETER_RANGES = joinpath(@__DIR__, "data", "MosunParameterRanges.csv")
const DEFAULT_VARIANTS = joinpath(@__DIR__, "data", "MosunVariantOverrides.tsv")

const DOSE_DAYS = Float64[0, 7, 14, 21, 42, 63]
const LOW_DOSES = fill(1.0, 6)
const HIGH_DOSES = fill(60.0, 6)
const HOSSEINI_DOSES = Float64[1.6, 10.0, 10.0, 20.0, 20.0, 20.0]
const LOWER = zeros(6)
const UPPER = fill(60.0, 6)
const SAVEAT = collect(0.0:0.5:84.0)
const BODY_WEIGHT_KG = 70.0
const SEED = 20260519
const RHO = 1e-8
const EPSILON = 1e-12
const VARIANT_IDS = Set([5, 9, 14, 20, 24, 25, 27, 28])
const DUMMY_PARAMETERS = Set(["dummy_null_1", "dummy_null_2", "dummy_null_3"])
const LATENT_PARAMETERS = Set(["tumor_burden_factor", "BT_ratio_tumor_init"])

const ALG = Rodas4P(autodiff=false)
const DTO = SensitivityADPassThrough()
const PARAMETER_INDEX = Dict(name => i for (i, name) in enumerate(MMC.PARAMETER_NAMES))
const STATE_INDEX = Dict(name => i for (i, name) in enumerate(MMC.DYNAMIC_STATE_NAMES))

struct Patient
    id::Int
    parameters::Vector{Float64}
    tumor_target::Float64
    il6_target::Float64
    tumor_scale::Float64
    il6_scale::Float64
end

smooth_positive(x) = (x + sqrt(x^2 + oftype(x, RHO)^2)) / 2

function apply_published_variants!(parameters, variants_file)
    variants = CSV.read(variants_file, DataFrame; delim='\t')
    for row in eachrow(variants)
        Int(row.idx) in VARIANT_IDS || continue
        String(row.action) == "parameter" || continue
        String(row.name) == "Value" || continue
        value = tryparse(Float64, strip(string(row.value)))
        value === nothing && continue
        name = String(row.class)
        MMC.has_parameter(name) && MMC.set_param!(parameters, name, value)
    end
    return parameters
end

function base_parameters(variants_file=DEFAULT_VARIANTS)
    parameters = apply_published_variants!(deepcopy(MMC.default_params()), variants_file)
    MMC.has_parameter("PKflag") && MMC.set_param!(parameters, "PKflag", 1.0)
    MMC.has_parameter("VPid") && MMC.set_param!(parameters, "VPid", 1.0)
    MMC.has_parameter("fvalidation") && MMC.set_param!(parameters, "fvalidation", 0.0)
    MMC.has_parameter("end_time") && MMC.set_param!(parameters, "end_time", 84.0)
    return parameters
end

boolish(value) = value isa Bool ? value : lowercase(strip(String(value))) in ("1", "true", "yes")

function apply_vpop_row(base, universe::DataFrame, row)
    parameters = deepcopy(base)
    latent = Dict{String, Float64}()
    for specification in eachrow(universe)
        name = String(specification.parameter)
        value = Float64(row[Symbol(name)])
        if name in DUMMY_PARAMETERS
            continue
        elseif name in LATENT_PARAMETERS
            latent[name] = value
        elseif boolish(specification.applied_to_model) && MMC.has_parameter(name)
            MMC.set_param!(parameters, name, value)
        end
    end

    bpbo = Float64(getproperty(base, :Bpbo_perml))
    trpbo = Float64(getproperty(base, :Trpbo_perml))
    kbp = Float64(getproperty(base, :KBptumor))
    burden_factor = get(latent, "tumor_burden_factor", 1.0)
    bt_ratio = get(latent, "BT_ratio_tumor_init", 20.0)
    btumor_perml = kbp * max(bpbo, 1e-12) * burden_factor
    MMC.set_param!(parameters, "KBptumor", btumor_perml / max(bpbo, 1e-12))
    MMC.set_param!(parameters, "KTrptumor", btumor_perml / max(bt_ratio * trpbo, 1e-12))
    return parameters
end

"""Map six differentiable dose amplitudes to fixed-time bolus events."""
function regimen(doses_mg)
    events = map(zip(DOSE_DAYS, doses_mg)) do (time, dose)
        amount = dose * (1000.0 / BODY_WEIGHT_KG)
        MMC.MosunRegimenEvent(
            target=:TDBc_ugperkg,
            time=time,
            amount=amount,
            rate=zero(amount),
        )
    end
    return MMC.MosunRegimen(events=collect(events))
end

function il6_total(state, parameters)
    return state[STATE_INDEX[:IL6pb]] +
        parameters[PARAMETER_INDEX[:IL6_tiss_contribution]] * (
            state[STATE_INDEX[:IL6tiss]] * parameters[PARAMETER_INDEX[:Vtissue]] +
            state[STATE_INDEX[:IL6tiss2]] * parameters[PARAMETER_INDEX[:Vtissue2]] +
            state[STATE_INDEX[:IL6tiss3]] * parameters[PARAMETER_INDEX[:Vtissue3]] +
            state[STATE_INDEX[:IL6tumor]] * parameters[PARAMETER_INDEX[:Vtumor]]
        ) / parameters[PARAMETER_INDEX[:Vpb]]
end

function trapezoid(times, values)
    length(times) == length(values) || throw(DimensionMismatch("times and values must have equal length"))
    length(times) < 2 && return zero(first(values))
    integral = zero(first(values))
    for index in 1:(length(times) - 1)
        integral += (times[index + 1] - times[index]) * (values[index + 1] + values[index]) / 2
    end
    return integral
end

function endpoints(parameters, doses_mg; sensealg=nothing, posthoc=false)
    built = MMC.build_problem_vector(
        regimen(doses_mg),
        parameters;
        tspan=(0.0, 84.0),
        saveat=SAVEAT,
        callback_mode=:callback,
        post_event_proposed_dt=0.01,
    )
    kwargs = (
        abstol=1e-8,
        reltol=1e-5,
        callback=built.callback,
        tstops=built.tstops,
        d_discontinuities=built.d_discontinuities,
        saveat=built.saveat,
        save_everystep=false,
        maxiters=1_000_000,
    )
    solution = sensealg === nothing ?
        solve(built.prob, ALG; kwargs...) :
        solve(built.prob, ALG; kwargs..., sensealg=sensealg)
    solution.retcode == SciMLBase.ReturnCode.Success || error("ODE solve failed: $(solution.retcode)")

    tumor_index = STATE_INDEX[:Btumor]
    tumor_initial = built.initial_u[tumor_index]
    tumor_ratio = smooth_positive(solution.u[end][tumor_index] / (tumor_initial + EPSILON))
    il6_values = [smooth_positive(il6_total(state, parameters)) for state in solution.u]
    il6_peak = maximum(il6_values)
    primary = (; tumor_ratio, il6_peak)
    posthoc || return primary

    tumor_values = [smooth_positive(state[tumor_index]) for state in solution.u]
    return (
        ;
        primary...,
        day84_tumor_change_pct=100 * (tumor_ratio - one(tumor_ratio)),
        il6_auc=trapezoid(solution.t, il6_values),
        tumor_auc=trapezoid(solution.t, tumor_values),
        total_dose_mg=sum(doses_mg),
        drug_exposure=solution.u[end][STATE_INDEX[:TDBc_ugperml_AUC]],
    )
end

function patient_from_row(base, universe, row)
    parameters = MMC.pack_params(apply_vpop_row(base, universe, row))
    low = endpoints(parameters, LOW_DOSES)
    high = endpoints(parameters, HIGH_DOSES)
    tumor_gap = smooth_positive(log((low.tumor_ratio + EPSILON) / (high.tumor_ratio + EPSILON)))
    il6_gap = smooth_positive(log((high.il6_peak + EPSILON) / (low.il6_peak + EPSILON)))
    return Patient(
        Int(row.vpop_id),
        parameters,
        Float64(high.tumor_ratio),
        Float64(low.il6_peak),
        max(Float64(tumor_gap^2), 1e-9),
        max(Float64(il6_gap^2), 1e-9),
    )
end

function load_patients(vpop_file, universe_file, variants_file, patient_count)
    rows = CSV.read(vpop_file, DataFrame)
    universe = CSV.read(universe_file, DataFrame)
    base = base_parameters(variants_file)
    n = patient_count === :all ? nrow(rows) : min(Int(patient_count), nrow(rows))
    n > 0 || throw(ArgumentError("No virtual patients selected from $vpop_file"))
    return [patient_from_row(base, universe, rows[i, :]) for i in 1:n]
end

function loss_components_from_endpoints(patient::Patient, doses, result)
    tumor_gap = smooth_positive(log((result.tumor_ratio + EPSILON) / (patient.tumor_target + EPSILON)))
    il6_gap = smooth_positive(log((result.il6_peak + EPSILON) / (patient.il6_target + EPSILON)))
    tumor = tumor_gap^2 / patient.tumor_scale
    il6 = il6_gap^2 / patient.il6_scale
    cycle_totals = [doses[1] + doses[2] + doses[3], doses[4], doses[5], doses[6]]
    dose = mean(((cycle_totals .- 20.0) ./ 20.0) .^ 2)
    components = (
        total=tumor + il6 + dose,
        tumor,
        il6,
        dose,
        tumor_ratio=result.tumor_ratio,
        il6_peak=result.il6_peak,
    )
    hasproperty(result, :il6_auc) || return components
    return (
        ;
        components...,
        day84_tumor_change_pct=result.day84_tumor_change_pct,
        il6_auc=result.il6_auc,
        tumor_auc=result.tumor_auc,
        total_dose_mg=result.total_dose_mg,
        drug_exposure=result.drug_exposure,
    )
end

function loss_components(patient::Patient, doses; sensealg=nothing, posthoc=false)
    result = endpoints(patient.parameters, doses; sensealg=sensealg, posthoc=posthoc)
    return loss_components_from_endpoints(patient, doses, result)
end

function optimize_patient(patient::Patient; preset="paper")
    rng = MersenneTwister(SEED + patient.id + 101)
    initial = UPPER .* rand(rng, 6)
    history = NamedTuple[]
    evaluation = Ref(0)
    function primal_loss(doses)
        terms = loss_components(patient, doses)
        evaluation[] += 1
        push!(history, (
            vpop_id=patient.id,
            evaluation=evaluation[],
            total_loss=terms.total,
            tumor_loss=terms.tumor,
            il6_loss=terms.il6,
            dose_loss=terms.dose,
            C1D1_mg=doses[1],
            C1D8_mg=doses[2],
            C1D15_mg=doses[3],
            C2D1_mg=doses[4],
            C3D1_mg=doses[5],
            C4D1_mg=doses[6],
        ))
        return terms.total
    end
    # SensitivityADPassThrough lets ForwardDiff differentiate the discretized
    # solve, including the fixed-time jump updates in utils/MosunModel.jl.
    dto_loss = doses -> loss_components(patient, doses; sensealg=DTO).total
    gradient! = (storage, doses) -> ForwardDiff.gradient!(storage, dto_loss, doses)

    result = Optim.optimize(
        primal_loss,
        gradient!,
        LOWER,
        UPPER,
        initial,
        Fminbox(BFGS()),
        Optim.Options(
            iterations=preset == "quick" ? 3 : 50,
            f_calls_limit=preset == "quick" ? 30 : 1000,
            time_limit=preset == "quick" ? 300.0 : 7200.0,
        ),
    )
    optimum = clamp.(Optim.minimizer(result), LOWER, UPPER)
    initial_terms = loss_components(patient, initial)
    final_terms = loss_components(patient, optimum; posthoc=true)
    reference_terms = loss_components(patient, HOSSEINI_DOSES; posthoc=true)
    summary = (
        vpop_id=patient.id,
        converged=Optim.converged(result),
        initial_loss=initial_terms.total,
        final_loss=final_terms.total,
        tumor_loss=final_terms.tumor,
        il6_loss=final_terms.il6,
        dose_loss=final_terms.dose,
        reference_tumor_ratio=reference_terms.tumor_ratio,
        optimized_tumor_ratio=final_terms.tumor_ratio,
        reference_il6_peak=reference_terms.il6_peak,
        optimized_il6_peak=final_terms.il6_peak,
        reference_il6_auc=reference_terms.il6_auc,
        optimized_il6_auc=final_terms.il6_auc,
        reference_day84_tumor_change_pct=reference_terms.day84_tumor_change_pct,
        optimized_day84_tumor_change_pct=final_terms.day84_tumor_change_pct,
        reference_tumor_auc=reference_terms.tumor_auc,
        optimized_tumor_auc=final_terms.tumor_auc,
        reference_total_dose_mg=reference_terms.total_dose_mg,
        optimized_total_dose_mg=final_terms.total_dose_mg,
        reference_drug_exposure=reference_terms.drug_exposure,
        optimized_drug_exposure=final_terms.drug_exposure,
        C1D1_mg=optimum[1],
        C1D8_mg=optimum[2],
        C1D15_mg=optimum[3],
        C2D1_mg=optimum[4],
        C3D1_mg=optimum[5],
        C4D1_mg=optimum[6],
    )
    return summary, history
end

function parse_cli(args)
    options = Dict{String,String}(
        "preset" => "quick",
        "patients" => "auto",
        "vpop" => DEFAULT_VPOP,
        "parameter-ranges" => DEFAULT_PARAMETER_RANGES,
        "variants" => DEFAULT_VARIANTS,
        "output" => joinpath(@__DIR__, "output"),
    )
    index = 1
    while index <= length(args)
        flag = args[index]
        flag in ("--preset", "--patients", "--vpop", "--parameter-ranges", "--variants", "--output") ||
            throw(ArgumentError("Unknown option: $flag"))
        index == length(args) && throw(ArgumentError("Missing value after $flag"))
        options[flag[3:end]] = args[index + 1]
        index += 2
    end
    options["preset"] in ("quick", "paper") ||
        throw(ArgumentError("--preset must be quick or paper"))
    return options
end

function patient_selection(value, preset)
    value == "all" && return :all
    value == "auto" && return preset == "paper" ? :all : 1
    count = tryparse(Int, value)
    count === nothing && throw(ArgumentError("--patients must be all or a positive integer"))
    count > 0 || throw(ArgumentError("--patients must be positive"))
    return count
end

function main(args=ARGS)
    options = parse_cli(args)
    preset = options["preset"]
    selection = patient_selection(options["patients"], preset)
    patients = load_patients(
        abspath(options["vpop"]),
        abspath(options["parameter-ranges"]),
        abspath(options["variants"]),
        selection,
    )
    bundles = [optimize_patient(patient; preset=preset) for patient in patients]
    results = [bundle[1] for bundle in bundles]
    histories = reduce(vcat, [bundle[2] for bundle in bundles]; init=NamedTuple[])
    output_dir = abspath(options["output"])
    mkpath(output_dir)
    output_file = joinpath(output_dir, "Mosun_optimization.csv")
    CSV.write(output_file, DataFrame(results))
    CSV.write(joinpath(output_dir, "Mosun_optimization_history.csv"), DataFrame(histories))
    println("Wrote $(length(results)) individualized optimizations to $output_file")
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
