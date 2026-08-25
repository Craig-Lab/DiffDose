#!/usr/bin/env julia
"""
Generate the mosunetuzumab virtual-population campaign inputs.

Stages combine eFAST sampling/screening and candidate trajectory resimulation.
All generated banks are written beneath the requested output directory.

The original SimBiology project is distributed with Hosseini et al. (2020):
https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-020-00145-7/MediaObjects/41540_2020_145_MOESM3_ESM.zip
See `MosunModel.jl` for scientific references and a minimal simulation example.
"""

include(joinpath(@__DIR__, "MosunModel.jl"))
using .MosunModel
using DifferentialEquations
using Sundials
const MMC = MosunModel
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_RANGES = joinpath(REPO_ROOT, "data", "MosunParameterRanges.csv")
const DEFAULT_VARIANTS = joinpath(REPO_ROOT, "data", "MosunVariantOverrides.tsv")
const ACTIVE_VARIANTS_FILE = Ref(DEFAULT_VARIANTS)

function make_solver_alg(solver_name::AbstractString)
    solver = lowercase(String(solver_name))
    solver == "cvode_bdf" && return CVODE_BDF()
    solver == "qndf" && return QNDF(autodiff=false)
    solver == "rodas4p" && return Rodas4P(autodiff=false)
    solver == "kencarp4" && return KenCarp4(autodiff=false)
    solver == "vcabm" && return VCABM()
    solver == "tsit5" && return Tsit5()
    error("Unsupported solver=$solver")
end

function load_variant_overrides(path=ACTIVE_VARIANTS_FILE[])
    variants = Dict{Int,Vector{Pair{String,Float64}}}()
    for row in eachrow(CSV.read(path, DataFrame; delim='\t'))
        index = hasproperty(row, :variant_idx) ? Int(row.variant_idx) : Int(row.idx)
        String(row.action) == "parameter" || continue
        String(row.name) == "Value" || continue
        value = tryparse(Float64, strip(string(row.value)))
        value === nothing && continue
        push!(get!(variants, index, Pair{String,Float64}[]), String(row.class) => value)
    end
    return variants
end

function apply_variant_overrides!(parameters, variant_ids, overrides)
    for variant_id in variant_ids
        for (name, value) in get(overrides, variant_id, Pair{String,Float64}[])
            MMC.has_parameter(name) && MMC.set_param!(parameters, name, value)
        end
    end
    return parameters
end

# Susilo/Fig. 5 IL6 dummy-null eFAST campaign.
# Additive runner for scalar simulation banks and dummy-threshold sensitivity screens.

using CSV
using DataFrames
using Dates
using FFTW
using JSON3
using LinearAlgebra
using Random
using Statistics
using Base.Threads
using SciMLBase


const DLBCL_VARIANT_IDS = [5, 9, 14, 20, 24, 25, 27, 28]
const SWITCH_PARAMS = Set(["tumor_on", "tissue2on", "tissue3on"])
const DERIVED_TUMOR_PARAMS = Set(["KBptumor", "KTrptumor"])
const TUMOR_LATENT_PARAMS = Set(["tumor_burden_factor", "BT_ratio_tumor_init"])
const DUMMY_PARAMS = ["dummy_null_1", "dummy_null_2", "dummy_null_3"]
const EPS = 1e-12

Base.@kwdef struct Fig5RegimenSpec
    name::String
    label::String
    dose_mg::Vector{Float64}
end

function fig5_regimens()
    Fig5RegimenSpec[
        Fig5RegimenSpec(
            name = "no_dose",
            label = "No dose",
            dose_mg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        Fig5RegimenSpec(
            name = "c1d1_20_only_then_20_q3w",
            label = "20 mg on C1D1, then 20 mg q3w",
            dose_mg = [20.0, 0.0, 0.0, 20.0, 20.0, 20.0],
        ),
        Fig5RegimenSpec(
            name = "c1d1_d8_d15_6p7_then_20_q3w",
            label = "6.7/6.7/6.7 mg, then 20 mg q3w",
            dose_mg = [6.7, 6.7, 6.7, 20.0, 20.0, 20.0],
        ),
        Fig5RegimenSpec(
            name = "c1d1_d8_d15_1p6_10_10_then_20_q3w",
            label = "1.6/10/10 mg, then 20 mg q3w",
            dose_mg = [1.6, 10.0, 10.0, 20.0, 20.0, 20.0],
        ),
        Fig5RegimenSpec(
            name = "c1d1_d8_1p6_20_then_20_q3w",
            label = "1.6/20 mg, then 20 mg q3w",
            dose_mg = [1.6, 20.0, 0.0, 20.0, 20.0, 20.0],
        ),
    ]
end

env_string(name::String, default::String) = isempty(strip(get(ENV, name, ""))) ? default : strip(get(ENV, name, default))
env_int(name::String, default::Int) = isempty(strip(get(ENV, name, ""))) ? default : parse(Int, strip(get(ENV, name, "")))
env_float(name::String, default::Float64) = isempty(strip(get(ENV, name, ""))) ? default : parse(Float64, strip(get(ENV, name, "")))

function env_seed_list(name::String, default::Vector{Int})
    value = strip(get(ENV, name, ""))
    isempty(value) && return default
    [parse(Int, strip(x)) for x in split(value, ",") if !isempty(strip(x))]
end

function safe_field_float(x, names::Vector{String}; default = NaN)
    for name in names
        sym = Symbol(name)
        if hasproperty(x, sym)
            v = getproperty(x, sym)
            if v !== missing && !(v isa Nothing)
                try
                    return Float64(v)
                catch
                end
            end
        end
    end
    default
end

function classify_family(name::AbstractString)
    n = lowercase(String(name))
    if startswith(n, "dummy_null")
        "dummy_null"
    elseif occursin("il6", n) || n in ["kil6prod", "thalfil6"]
        "IL6"
    elseif occursin("tumor", n) || occursin("btumor", n) || n in ["kbtumorprolif", "kbptumor", "ktrptumor", "tumor_burden_factor", "bt_ratio_tumor_init"]
        "tumor_burden_growth_infiltration"
    elseif occursin("ta", n) || occursin("cd69", n) || occursin("tcell", n) || occursin("trpb", n) || occursin("trp", n)
        "t_cell_activation_infiltration"
    elseif occursin("kill", n) || occursin("bk", n) || occursin("btcell", n)
        "b_cell_killing"
    elseif occursin("pk", n) || occursin("cl", n) || occursin("vc", n) || occursin("vp", n) || occursin("tdbc", n)
        "PK_exposure"
    else
        "other_model"
    end
end

function build_shipped_dlbcl_base_params()
    p = deepcopy(MMC.default_params())
    overrides = load_variant_overrides()
    apply_variant_overrides!(p, DLBCL_VARIANT_IDS, overrides)
    MMC.has_parameter("VPid") && MMC.set_param!(p, "VPid", 1.0)
    MMC.has_parameter("PKflag") && MMC.set_param!(p, "PKflag", 1.0)
    MMC.has_parameter("fvalidation") && MMC.set_param!(p, "fvalidation", 0.0)
    p
end

function load_dlbcl_bounds(path::AbstractString)
    df = CSV.read(path, DataFrame)
    rows = NamedTuple[]
    for row in eachrow(df)
        name = String(row.parameter)
        (name in SWITCH_PARAMS || name in DERIVED_TUMOR_PARAMS) && continue
        lo = safe_field_float(row, ["dlbcl_min_value", "lower_bound", "min"])
        hi = safe_field_float(row, ["dlbcl_max_value", "upper_bound", "max"])
        base = safe_field_float(row, ["base_value", "final_value", "preferred_xlsx_value"])
        (!isfinite(lo) || !isfinite(hi) || !(hi > lo)) && continue
        MMC.has_parameter(name) || continue
        scale = (lo > 0.0 && hi > 0.0 && hi / max(lo, EPS) >= 3.0) ? "log10" : "linear"
        push!(rows, (
            parameter = name,
            lower_bound = lo,
            upper_bound = hi,
            base_value = base,
            sampling_scale = scale,
            source = "dlbcl_range_efast",
            family = classify_family(name),
            applied_to_model = true,
            note = "Previous DLBCL-range eFAST parameter.",
        ))
    end
    rows
end

function upsert_param!(rows::Vector{NamedTuple}, row::NamedTuple)
    idx = findfirst(r -> r.parameter == row.parameter, rows)
    idx === nothing ? push!(rows, row) : (rows[idx] = row)
    rows
end

function parameter_universe(bounds_path::AbstractString; param_limit::Int = 0)
    rows = collect(load_dlbcl_bounds(bounds_path))
    if param_limit > 0 && param_limit < length(rows)
        rows = rows[1:param_limit]
    end

    forced = [
        (
            parameter = "tumor_burden_factor",
            lower_bound = 0.5,
            upper_bound = 1.5,
            base_value = 1.0,
            sampling_scale = "linear",
            source = "forced_susilo_latent",
            family = "tumor_burden_growth_infiltration",
            applied_to_model = false,
            note = "Latent variable; derives KBptumor and KTrptumor from baseline tumor burden.",
        ),
        (
            parameter = "BT_ratio_tumor_init",
            lower_bound = 4.0,
            upper_bound = 80.0,
            base_value = 20.0,
            sampling_scale = "log10",
            source = "forced_susilo_latent",
            family = "tumor_burden_growth_infiltration",
            applied_to_model = false,
            note = "Latent tumor B:T infiltration ratio; derives KTrptumor.",
        ),
        (
            parameter = "kBtumorprolif",
            lower_bound = 0.0,
            upper_bound = 0.15,
            base_value = 0.0,
            sampling_scale = "linear",
            source = "forced_susilo_response",
            family = "tumor_burden_growth_infiltration",
            applied_to_model = MMC.has_parameter("kBtumorprolif"),
            note = "Forced Susilo tumor proliferation response variable.",
        ),
        (
            parameter = "kIL6prod",
            lower_bound = 1.0,
            upper_bound = 14.0,
            base_value = 1.0,
            sampling_scale = "log10",
            source = "curated_sensitivity_only",
            family = "IL6",
            applied_to_model = MMC.has_parameter("kIL6prod"),
            note = "Curated IL6 production range; old DLBCL table had no variable bound.",
        ),
        (
            parameter = "thalfIL6",
            lower_bound = 5.0,
            upper_bound = 120.0,
            base_value = 20.0,
            sampling_scale = "log10",
            source = "curated_sensitivity_only",
            family = "IL6",
            applied_to_model = MMC.has_parameter("thalfIL6"),
            note = "Curated IL6 half-life/clearance sensitivity range in minutes.",
        ),
        (
            parameter = "IL6_tiss_contribution",
            lower_bound = 0.0004,
            upper_bound = 0.02,
            base_value = 0.002,
            sampling_scale = "log10",
            source = "forced_susilo_il6",
            family = "IL6",
            applied_to_model = MMC.has_parameter("IL6_tiss_contribution"),
            note = "Forced IL6 tissue contribution parameter.",
        ),
    ]
    for row in forced
        upsert_param!(rows, row)
    end
    for name in DUMMY_PARAMS
        push!(rows, (
            parameter = name,
            lower_bound = 0.0,
            upper_bound = 1.0,
            base_value = 0.0,
            sampling_scale = "linear",
            source = "dummy_null",
            family = "dummy_null",
            applied_to_model = false,
            note = "Inert null parameter sampled for dummy-threshold significance; never applied to the model.",
        ))
    end
    seen = Set{String}()
    unique_rows = NamedTuple[]
    for row in rows
        row.parameter in seen && continue
        push!(seen, row.parameter)
        push!(unique_rows, row)
    end
    DataFrame(unique_rows)
end

function salib_fast_sample(n::Int, d::Int, m::Int, seed::Int)
    n > 2m || throw(ArgumentError("FAST sample size N=$(n) must be greater than 2M=$(2m)."))
    rng = MersenneTwister(seed)
    omega = floor(Int, (n - 1) / (2m))
    omega >= 1 || throw(ArgumentError("FAST omega must be positive; increase N."))
    omega2 = d > 1 ? [floor(Int, 1 + (omega - 1) * (i - 1) / max(d - 2, 1)) for i in 1:(d - 1)] : Int[]
    rows = Vector{Vector{Float64}}(undef, n * d)
    parameter_block = Vector{Int}(undef, n * d)
    sample_in_block = Vector{Int}(undef, n * d)
    idx = 1
    s = range(-pi, pi; length = n + 1)[1:n]
    for i in 1:d
        w = fill(omega, d)
        low = 1
        for j in 1:d
            if j != i
                w[j] = omega2[low]
                low += 1
            end
        end
        phase = 2pi * rand(rng)
        for k in 1:n
            x = similar(w, Float64)
            for j in 1:d
                g = 0.5 + asin(sin(w[j] * s[k] + phase)) / pi
                x[j] = clamp(g, EPS, 1.0 - EPS)
            end
            rows[idx] = x
            parameter_block[idx] = i
            sample_in_block[idx] = k
            idx += 1
        end
    end
    reduce(vcat, transpose.(rows)), parameter_block, sample_in_block, omega
end

function transform_unit_sample(u::AbstractMatrix, universe::DataFrame)
    x = Array{Float64}(undef, size(u))
    for j in axes(u, 2)
        lo = Float64(universe.lower_bound[j])
        hi = Float64(universe.upper_bound[j])
        scale = String(universe.sampling_scale[j])
        if scale == "log10"
            x[:, j] .= 10.0 .^ (log10(lo) .+ u[:, j] .* (log10(hi) - log10(lo)))
        else
            x[:, j] .= lo .+ u[:, j] .* (hi - lo)
        end
    end
    x
end

function fast_power_spectrum(values::AbstractVector{<:Real})
    y = collect(Float64, values)
    y .-= mean(y)
    abs.(fft(y)).^2 / length(y)^2
end

function salib_fast_analyze(values::AbstractVector{<:Real}, n::Int, d::Int, m::Int, omega::Int)
    length(values) == n * d || throw(ArgumentError("Expected $(n*d) values, got $(length(values))."))
    s1 = fill(NaN, d)
    st = fill(NaN, d)
    for i in 1:d
        y = collect(Float64, values[((i - 1) * n + 1):(i * n)])
        finite = isfinite.(y)
        count(finite) < max(8, div(n, 2)) && continue
        if !all(finite)
            med = median(y[finite])
            y[.!finite] .= med
        end
        total_var = var(y)
        !(isfinite(total_var) && total_var > 0) && continue
        spectrum = fast_power_spectrum(y)
        max_harmonic = min(m, floor(Int, (length(spectrum) - 1) / max(omega, 1)))
        max_harmonic < 1 && continue
        harmonics = [k * omega + 1 for k in 1:max_harmonic]
        d1 = 2.0 * sum(spectrum[harmonics])
        cutoff = max(1, floor(Int, omega / 2))
        dt = 2.0 * sum(spectrum[2:(cutoff + 1)])
        s1[i] = clamp(d1 / total_var, 0.0, Inf)
        st[i] = clamp(1.0 - dt / total_var, 0.0, Inf)
    end
    s1, st
end

nearest_value(t::Vector{Float64}, y::Vector{Float64}, target::Float64) = y[findmin(abs.(t .- target))[2]]

function max_window(t::Vector{Float64}, y::Vector{Float64}, t0::Float64, t1::Float64)
    vals = [y[i] for i in eachindex(t) if t0 - 1e-9 <= t[i] <= t1 + 1e-9 && isfinite(y[i])]
    isempty(vals) ? NaN : maximum(vals)
end

function trapz_window(t::Vector{Float64}, y::Vector{Float64}, t0::Float64, t1::Float64)
    idxs = [i for i in eachindex(t) if t0 - 1e-9 <= t[i] <= t1 + 1e-9 && isfinite(y[i])]
    length(idxs) < 2 && return NaN
    acc = 0.0
    for k in 1:(length(idxs) - 1)
        i = idxs[k]
        j = idxs[k + 1]
        acc += 0.5 * (y[i] + y[j]) * (t[j] - t[i])
    end
    acc
end

percent_change(day_value::Float64, base_value::Float64) = 100.0 * (day_value / max(base_value, EPS) - 1.0)

function metric_row(seed::Int, eval_idx::Int, parameter_block::Int, sample_in_block::Int,
    spec::Fig5RegimenSpec, status::String, t::Vector{Float64}, bt::Vector{Float64}, il6::Vector{Float64},
    taf::Vector{Float64}, drug::Vector{Float64})

    if isempty(bt)
        return (
            seed = seed,
            efast_eval_idx = eval_idx,
            parameter_block = parameter_block,
            sample_in_block = sample_in_block,
            regimen = spec.name,
            regimen_label = spec.label,
            status = status,
            first_peak_il6_0_7d = NaN,
            peak_il6_0_21d = NaN,
            peak_il6_0_42d = NaN,
            il6combo_auc_0_7d = NaN,
            il6combo_auc_0_21d = NaN,
            il6combo_auc_0_42d = NaN,
            tafraction_peak_0_42d = NaN,
            tafraction_auc_0_42d = NaN,
            day42_tumor_size_change_pct = NaN,
            day84_tumor_size_change_pct = NaN,
            responder_gt50_day42 = missing,
            responder_gt50_day84 = missing,
            btumor_auc_0_42d = NaN,
            btumor_auc_0_84d = NaN,
            drug_auc_0_42d = NaN,
            drug_auc_0_84d = NaN,
            best_spd_pct = NaN,
        )
    end
    bt0 = bt[1]
    bt42 = nearest_value(t, bt, 42.0)
    bt84 = nearest_value(t, bt, 84.0)
    d42 = percent_change(bt42, bt0)
    d84 = percent_change(bt84, bt0)
    best = minimum([percent_change(v, bt0) for v in bt if isfinite(v)])
    (
        seed = seed,
        efast_eval_idx = eval_idx,
        parameter_block = parameter_block,
        sample_in_block = sample_in_block,
        regimen = spec.name,
        regimen_label = spec.label,
        status = status,
        first_peak_il6_0_7d = max_window(t, il6, 0.0, 7.0),
        peak_il6_0_21d = max_window(t, il6, 0.0, 21.0),
        peak_il6_0_42d = max_window(t, il6, 0.0, 42.0),
        il6combo_auc_0_7d = trapz_window(t, il6, 0.0, 7.0),
        il6combo_auc_0_21d = trapz_window(t, il6, 0.0, 21.0),
        il6combo_auc_0_42d = trapz_window(t, il6, 0.0, 42.0),
        tafraction_peak_0_42d = max_window(t, taf, 0.0, 42.0),
        tafraction_auc_0_42d = trapz_window(t, taf, 0.0, 42.0),
        day42_tumor_size_change_pct = d42,
        day84_tumor_size_change_pct = d84,
        responder_gt50_day42 = isfinite(d42) ? Int(d42 <= -50.0) : missing,
        responder_gt50_day84 = isfinite(d84) ? Int(d84 <= -50.0) : missing,
        btumor_auc_0_42d = trapz_window(t, bt, 0.0, 42.0),
        btumor_auc_0_84d = trapz_window(t, bt, 0.0, 84.0),
        drug_auc_0_42d = trapz_window(t, drug, 0.0, 42.0),
        drug_auc_0_84d = trapz_window(t, drug, 0.0, 84.0),
        best_spd_pct = best,
    )
end

function baseline_float(p, name::String, default::Float64)
    sym = Symbol(name)
    hasproperty(p, sym) ? Float64(getproperty(p, sym)) : default
end

function apply_sample_to_params(base_params, universe::DataFrame, values::Vector{Float64})
    p = deepcopy(base_params)
    latent = Dict{String, Float64}()
    for j in 1:nrow(universe)
        name = String(universe.parameter[j])
        value = Float64(values[j])
        if name in DUMMY_PARAMS
            continue
        elseif name in TUMOR_LATENT_PARAMS
            latent[name] = value
            continue
        elseif Bool(universe.applied_to_model[j]) && MMC.has_parameter(name)
            MMC.set_param!(p, name, value)
        end
    end

    base_bpbo = baseline_float(base_params, "Bpbo_perml", 1.0)
    base_trpbo = baseline_float(base_params, "Trpbo_perml", 1.0)
    base_kbp = baseline_float(base_params, "KBptumor", 1.0)
    burden_factor = get(latent, "tumor_burden_factor", 1.0)
    bt_ratio = get(latent, "BT_ratio_tumor_init", 20.0)
    base_btumor_perml = base_kbp * max(base_bpbo, EPS)
    btumor_perml = base_btumor_perml * burden_factor
    MMC.has_parameter("KBptumor") && MMC.set_param!(p, "KBptumor", btumor_perml / max(base_bpbo, EPS))
    MMC.has_parameter("KTrptumor") && MMC.set_param!(p, "KTrptumor", btumor_perml / max(bt_ratio * base_trpbo, EPS))
    p
end

function regimen_from_spec(spec::Fig5RegimenSpec, bw_kg::Float64)
    events = MMC.MosunRegimenEvent{Float64}[]
    for (t, d) in zip([0.0, 7.0, 14.0, 21.0, 42.0, 63.0], spec.dose_mg)
        amount = d * 1000.0 / bw_kg
        iszero(amount) && continue
        push!(events, MMC.MosunRegimenEvent{Float64}(:TDBc_ugperkg, t, amount, 0.0))
    end
    MMC.MosunRegimen(events = events)
end

function simulate_regimen_metrics(seed::Int, eval_idx::Int, parameter_block::Int, sample_in_block::Int,
    p, spec::Fig5RegimenSpec, alg, save_times::Vector{Float64}, horizon_days::Float64,
    post_event_dt::Float64, bw_kg::Float64)

    regimen = regimen_from_spec(spec, bw_kg)
    try
        _, sol = MMC.solve_regimen(
            regimen,
            p,
            alg;
            tspan = (0.0, horizon_days),
            saveat = save_times,
            callback_mode = :callback,
            post_event_proposed_dt = post_event_dt,
            abstol = MMC.SOLVER_ABSTOL,
            reltol = MMC.SOLVER_RELTOL,
            save_everystep = false,
            maxiters = 1_000_000,
        )
        status = SciMLBase.successful_retcode(sol.retcode) ? "success" : String(Symbol(sol.retcode))
        cache = MMC.zero_observables_cache()
        t = collect(Float64, sol.t)
        bt = Vector{Float64}(undef, length(t))
        il6 = similar(bt)
        taf = similar(bt)
        drug = similar(bt)
        for i in eachindex(t)
            MMC.update_observables!(cache, sol.u[i], p, t[i])
            bt[i] = Float64(cache.Btumor_perml)
            il6[i] = Float64(cache.IL6combo)
            taf[i] = 100.0 * Float64(cache.Tafraction_pb)
            drug[i] = Float64(MMC.state_or_observable(sol.u[i], cache, :TDBc_ugperml))
        end
        metric_row(seed, eval_idx, parameter_block, sample_in_block, spec, status, t, bt, il6, taf, drug)
    catch err
        empty = Float64[]
        metric_row(seed, eval_idx, parameter_block, sample_in_block, spec, "error:$(typeof(err)):$(sprint(showerror, err))", empty, empty, empty, empty, empty)
    end
end

function build_paired_delta_metrics(metrics::DataFrame)
    no_dose = metrics[metrics.regimen .== "no_dose", :]
    treated = metrics[metrics.regimen .!= "no_dose", :]
    key = [:seed, :efast_eval_idx, :parameter_block, :sample_in_block]
    joined = innerjoin(treated, no_dose; on = key, makeunique = true, renamecols = "_treated" => "_nodose")
    rows = NamedTuple[]
    for row in eachrow(joined)
        push!(rows, (
            seed = row.seed,
            efast_eval_idx = row.efast_eval_idx,
            parameter_block = row.parameter_block,
            sample_in_block = row.sample_in_block,
            treated_regimen = row.regimen_treated,
            treated_regimen_label = row.regimen_label_treated,
            untreated_regimen = row.regimen_nodose,
            delta_day42_tumor_pct_nodose_minus_treated = row.day42_tumor_size_change_pct_nodose - row.day42_tumor_size_change_pct_treated,
            delta_day84_tumor_pct_nodose_minus_treated = row.day84_tumor_size_change_pct_nodose - row.day84_tumor_size_change_pct_treated,
            delta_log10_btumor_auc_0_42d_nodose_minus_treated = log10(max(row.btumor_auc_0_42d_nodose, EPS)) - log10(max(row.btumor_auc_0_42d_treated, EPS)),
            delta_log10_btumor_auc_0_84d_nodose_minus_treated = log10(max(row.btumor_auc_0_84d_nodose, EPS)) - log10(max(row.btumor_auc_0_84d_treated, EPS)),
            treated_first_peak_il6_0_7d = row.first_peak_il6_0_7d_treated,
            treated_peak_il6_0_21d = row.peak_il6_0_21d_treated,
            treated_peak_il6_0_42d = row.peak_il6_0_42d_treated,
            treated_il6combo_auc_0_7d = row.il6combo_auc_0_7d_treated,
            treated_il6combo_auc_0_21d = row.il6combo_auc_0_21d_treated,
            treated_il6combo_auc_0_42d = row.il6combo_auc_0_42d_treated,
            treated_tafraction_peak_0_42d = row.tafraction_peak_0_42d_treated,
            treated_tafraction_auc_0_42d = row.tafraction_auc_0_42d_treated,
        ))
    end
    DataFrame(rows)
end

const METRIC_ENDPOINTS = [
    "first_peak_il6_0_7d",
    "peak_il6_0_21d",
    "peak_il6_0_42d",
    "il6combo_auc_0_7d",
    "il6combo_auc_0_21d",
    "il6combo_auc_0_42d",
    "tafraction_peak_0_42d",
    "tafraction_auc_0_42d",
    "day42_tumor_size_change_pct",
    "day84_tumor_size_change_pct",
    "responder_gt50_day42",
    "responder_gt50_day84",
    "btumor_auc_0_42d",
    "btumor_auc_0_84d",
    "drug_auc_0_42d",
    "drug_auc_0_84d",
    "best_spd_pct",
]

const PAIRED_ENDPOINTS = [
    "delta_day42_tumor_pct_nodose_minus_treated",
    "delta_day84_tumor_pct_nodose_minus_treated",
    "delta_log10_btumor_auc_0_42d_nodose_minus_treated",
    "delta_log10_btumor_auc_0_84d_nodose_minus_treated",
    "treated_first_peak_il6_0_7d",
    "treated_peak_il6_0_21d",
    "treated_peak_il6_0_42d",
    "treated_il6combo_auc_0_7d",
    "treated_il6combo_auc_0_21d",
    "treated_il6combo_auc_0_42d",
    "treated_tafraction_peak_0_42d",
    "treated_tafraction_auc_0_42d",
]

function endpoint_family(endpoint::String)
    e = lowercase(endpoint)
    if startswith(e, "delta")
        "paired_delta"
    elseif occursin("il6", e)
        "IL6"
    elseif occursin("tafraction", e) || occursin("tcell", e)
        "T_cell_activation"
    elseif occursin("tumor", e) || occursin("btumor", e) || occursin("responder", e) || occursin("spd", e)
        "tumor_response"
    elseif occursin("drug", e)
        "PK_exposure"
    else
        "other"
    end
end

function endpoint_transform(metric::String)
    m = lowercase(metric)
    if occursin("auc", m) || occursin("peak", m) || occursin("first_peak", m)
        "log10_positive"
    else
        "identity"
    end
end

function transformed_endpoint_values(df::DataFrame, metric::String)
    v = Vector{Float64}(undef, nrow(df))
    col = df[!, Symbol(metric)]
    for (i, x) in enumerate(col)
        xf = try
            Float64(x)
        catch
            NaN
        end
        v[i] = endpoint_transform(metric) == "log10_positive" && isfinite(xf) ? log10(max(xf, EPS)) : xf
    end
    v
end

function rank_desc(values::Vector{Float64})
    order_idx = sortperm(values; rev = true, by = x -> isfinite(x) ? x : -Inf)
    ranks = fill(length(values), length(values))
    for (r, idx) in enumerate(order_idx)
        ranks[idx] = r
    end
    ranks
end

function analyze_endpoints(metrics::DataFrame, paired::DataFrame, universe::DataFrame, seeds::Vector{Int}, n::Int, d::Int, m::Int, omega::Int)
    rows = NamedTuple[]
    param_names = String.(universe.parameter)
    families = String.(universe.family)

    for seed in seeds
        seed_metrics = metrics[metrics.seed .== seed, :]
        seed_paired = paired[paired.seed .== seed, :]
        for regimen in unique(String.(seed_metrics.regimen))
            subset = seed_metrics[seed_metrics.regimen .== regimen, :]
            sort!(subset, [:parameter_block, :sample_in_block])
            for metric in METRIC_ENDPOINTS
                vals = transformed_endpoint_values(subset, metric)
                s1, st = salib_fast_analyze(vals, n, d, m, omega)
                ranks = rank_desc(st)
                endpoint = "$(regimen)__$(metric)"
                for j in 1:d
                    push!(rows, (
                        seed = seed,
                        endpoint = endpoint,
                        endpoint_family = endpoint_family(metric),
                        regimen = regimen,
                        metric = metric,
                        endpoint_source = "regimen_scalar",
                        transform = endpoint_transform(metric),
                        parameter = param_names[j],
                        parameter_family = families[j],
                        S1 = s1[j],
                        ST = st[j],
                        rank_ST = ranks[j],
                    ))
                end
            end
        end
        for regimen in unique(String.(seed_paired.treated_regimen))
            subset = seed_paired[seed_paired.treated_regimen .== regimen, :]
            sort!(subset, [:parameter_block, :sample_in_block])
            for metric in PAIRED_ENDPOINTS
                vals = transformed_endpoint_values(subset, metric)
                s1, st = salib_fast_analyze(vals, n, d, m, omega)
                ranks = rank_desc(st)
                endpoint = "delta__$(regimen)__$(metric)"
                fam = startswith(metric, "delta") ? "paired_delta" : endpoint_family(metric)
                for j in 1:d
                    push!(rows, (
                        seed = seed,
                        endpoint = endpoint,
                        endpoint_family = fam,
                        regimen = regimen,
                        metric = metric,
                        endpoint_source = "paired_delta",
                        transform = endpoint_transform(metric),
                        parameter = param_names[j],
                        parameter_family = families[j],
                        S1 = s1[j],
                        ST = st[j],
                        rank_ST = ranks[j],
                    ))
                end
            end
        end
    end
    DataFrame(rows)
end

function median_finite(v)
    vals = [Float64(x) for x in v if isfinite(Float64(x))]
    isempty(vals) ? NaN : median(vals)
end

function mean_finite(v)
    vals = [Float64(x) for x in v if isfinite(Float64(x))]
    isempty(vals) ? NaN : mean(vals)
end

function build_thresholds(indices::DataFrame)
    grouped = combine(groupby(indices, [:endpoint, :parameter, :parameter_family]),
        :ST => median_finite => :median_ST,
        :ST => mean_finite => :mean_ST)
    dummy = grouped[grouped.parameter_family .== "dummy_null", :]
    rows = NamedTuple[]
    for endpoint in unique(String.(grouped.endpoint))
        dsub = dummy[dummy.endpoint .== endpoint, :]
        if nrow(dsub) == 0
            push!(rows, (endpoint = endpoint, dummy_threshold_ST = NaN, dummy_parameter_argmax = "", n_dummy_params = 0))
        else
            idx = argmax(dsub.median_ST)
            push!(rows, (
                endpoint = endpoint,
                dummy_threshold_ST = Float64(dsub.median_ST[idx]),
                dummy_parameter_argmax = String(dsub.parameter[idx]),
                n_dummy_params = nrow(dsub),
            ))
        end
    end
    DataFrame(rows)
end

function build_significant_screen(indices::DataFrame, thresholds::DataFrame, seeds::Vector{Int})
    grouped = combine(groupby(indices, [:endpoint, :endpoint_family, :parameter, :parameter_family]),
        :ST => median_finite => :median_ST,
        :ST => mean_finite => :mean_ST,
        :S1 => median_finite => :median_S1)
    joined = leftjoin(grouped, thresholds; on = :endpoint)
    required = length(seeds) >= 3 ? 2 : 1
    counts = NamedTuple[]
    for row in eachrow(joined)
        sub = indices[(indices.endpoint .== row.endpoint) .& (indices.parameter .== row.parameter), :]
        thr = Float64(row.dummy_threshold_ST)
        count_above = sum([isfinite(Float64(x)) && isfinite(thr) && Float64(x) > thr for x in sub.ST])
        push!(counts, (
            endpoint = row.endpoint,
            parameter = row.parameter,
            seed_count_above_dummy = count_above,
        ))
    end
    joined = leftjoin(joined, DataFrame(counts); on = [:endpoint, :parameter])
    joined.significant = [
        (row.parameter_family != "dummy_null") &&
        isfinite(Float64(row.median_ST)) &&
        isfinite(Float64(row.dummy_threshold_ST)) &&
        Float64(row.median_ST) > Float64(row.dummy_threshold_ST) &&
        Int(row.seed_count_above_dummy) >= required
        for row in eachrow(joined)
    ]
    sort!(joined, [:endpoint, :median_ST], rev = [false, true])
    joined
end

function build_seed_stability(indices::DataFrame, thresholds::DataFrame)
    rows = NamedTuple[]
    for endpoint in unique(String.(indices.endpoint))
        sub = indices[(indices.endpoint .== endpoint) .& (indices.parameter_family .!= "dummy_null"), :]
        seed_topsets = Vector{Set{String}}()
        for seed in unique(sub.seed)
            ssub = sub[sub.seed .== seed, :]
            sort!(ssub, :ST, rev = true)
            push!(seed_topsets, Set(String.(ssub.parameter[1:min(10, nrow(ssub))])))
        end
        intersection_count = isempty(seed_topsets) ? 0 : length(reduce(intersect, seed_topsets))
        union_count = isempty(seed_topsets) ? 0 : length(reduce(union, seed_topsets))
        thr = thresholds[thresholds.endpoint .== endpoint, :]
        push!(rows, (
            endpoint = endpoint,
            n_seeds = length(seed_topsets),
            top10_intersection_count = intersection_count,
            top10_union_count = union_count,
            dummy_threshold_ST = nrow(thr) == 0 ? NaN : Float64(thr.dummy_threshold_ST[1]),
        ))
    end
    DataFrame(rows)
end

function write_meta(path::String, meta)
    open(path, "w") do io
        JSON3.pretty(io, meta)
        println(io)
    end
end

function run_efast(options)
    out_dir = joinpath(abspath(options["output"]), "efast")
    mkpath(out_dir)
    ACTIVE_VARIANTS_FILE[] = abspath(options["variants"])
    ranges_path = abspath(options["parameter-ranges"])
    n = options["preset"] == "quick" ? 16 : 256
    m = 4
    seeds = [parse(Int, options["seed"])]
    horizon_days = 84.0
    saveat_dt = options["preset"] == "quick" ? 0.5 : 0.125
    post_event_dt = 0.01
    bw_kg = 70.0
    solver = "rodas4p"
    regimen_limit = 0

    universe = CSV.read(ranges_path, DataFrame)
    d = nrow(universe)
    d > 0 || error("Parameter universe is empty.")
    CSV.write(joinpath(out_dir, "parameter_universe.csv"), universe)

    regimens = fig5_regimens()
    if regimen_limit > 0 && regimen_limit < length(regimens)
        regimens = regimens[1:regimen_limit]
    end
    any(r -> r.name == "no_dose", regimens) || error("no_dose regimen must be included for paired deltas.")

    all_sample_rows = DataFrame[]
    sample_values_by_seed = Dict{Int, Matrix{Float64}}()
    parameter_block_by_seed = Dict{Int, Vector{Int}}()
    sample_in_block_by_seed = Dict{Int, Vector{Int}}()
    omega_by_seed = Dict{Int, Int}()
    for seed in seeds
        u, parameter_block, sample_in_block, omega = salib_fast_sample(n, d, m, seed)
        sample_values = transform_unit_sample(u, universe)
        sample_values_by_seed[seed] = sample_values
        parameter_block_by_seed[seed] = parameter_block
        sample_in_block_by_seed[seed] = sample_in_block
        omega_by_seed[seed] = omega
        df = DataFrame(sample_values, Symbol.(universe.parameter))
        df.seed .= seed
        df.efast_eval_idx = collect(1:size(sample_values, 1))
        df.parameter_block = parameter_block
        df.sample_in_block = sample_in_block
        select!(df, [:seed, :efast_eval_idx, :parameter_block, :sample_in_block, Symbol.(universe.parameter)...])
        push!(all_sample_rows, df)
    end
    sample_matrix = vcat(all_sample_rows...; cols = :union)
    CSV.write(joinpath(out_dir, "efast_sample_matrix.csv"), sample_matrix)

    base_params = build_shipped_dlbcl_base_params()
    MMC.has_parameter("end_time") && MMC.set_param!(base_params, "end_time", horizon_days)
    alg = make_solver_alg(solver)
    save_times = collect(0.0:saveat_dt:horizon_days)

    jobs = NamedTuple[]
    for seed in seeds
        sample_values = sample_values_by_seed[seed]
        parameter_block = parameter_block_by_seed[seed]
        sample_in_block = sample_in_block_by_seed[seed]
        for eval_idx in 1:size(sample_values, 1)
            for spec in regimens
                push!(jobs, (
                    seed = seed,
                    eval_idx = eval_idx,
                    parameter_block = parameter_block[eval_idx],
                    sample_in_block = sample_in_block[eval_idx],
                    spec = spec,
                ))
            end
        end
    end

    println("Starting Susilo/Fig.5 eFAST scalar bank: D=$(d), N=$(n), seeds=$(seeds), jobs=$(length(jobs)), threads=$(nthreads())")
    results = Vector{Any}(undef, length(jobs))
    @threads for job_idx in eachindex(jobs)
        job = jobs[job_idx]
        sample_values = sample_values_by_seed[job.seed][job.eval_idx, :]
        p = apply_sample_to_params(base_params, universe, collect(sample_values))
        results[job_idx] = simulate_regimen_metrics(job.seed, job.eval_idx, job.parameter_block, job.sample_in_block, p, job.spec, alg, save_times, horizon_days, post_event_dt, bw_kg)
        if job_idx % max(1, div(length(jobs), 100)) == 0
            println("Completed $(job_idx) / $(length(jobs)) jobs at $(Dates.now())")
        end
    end

    metrics = DataFrame(collect(results))
    CSV.write(joinpath(out_dir, "simulation_metrics_long.csv"), metrics)
    paired = build_paired_delta_metrics(metrics)
    CSV.write(joinpath(out_dir, "paired_delta_metrics.csv"), paired)

    omega_values = unique(collect(Base.values(omega_by_seed)))
    length(omega_values) == 1 || error("Unexpected different omega values by seed: $(omega_values)")
    indices = analyze_endpoints(metrics, paired, universe, seeds, n, d, m, omega_values[1])
    CSV.write(joinpath(out_dir, "efast_indices_by_endpoint.csv"), indices)
    thresholds = build_thresholds(indices)
    CSV.write(joinpath(out_dir, "dummy_null_thresholds.csv"), thresholds)
    screen = build_significant_screen(indices, thresholds, seeds)
    CSV.write(joinpath(out_dir, "significant_parameter_screen.csv"), screen)
    stability = build_seed_stability(indices, thresholds)
    CSV.write(joinpath(out_dir, "seed_stability_summary.csv"), stability)

    meta = Dict(
        "created_at" => string(Dates.now()),
        "output_root" => out_dir,
        "parameter_ranges_csv" => ranges_path,
        "n" => n,
        "m" => m,
        "seeds" => seeds,
        "n_parameters" => d,
        "n_regimens" => length(regimens),
        "regimens" => [Dict("name" => r.name, "label" => r.label, "dose_mg" => r.dose_mg) for r in regimens],
        "dummy_parameters" => DUMMY_PARAMS,
        "forced_susilo_response_variables" => ["tumor_burden_factor", "BT_ratio_tumor_init", "kBtumorprolif", "kIL6prod", "thalfIL6", "IL6_tiss_contribution"],
        "simulation_settings" => Dict(
            "cohort_variant_ids" => DLBCL_VARIANT_IDS,
            "solver" => solver,
            "horizon_days" => horizon_days,
            "saveat_dt" => saveat_dt,
            "post_event_proposed_dt" => post_event_dt,
            "bw_kg" => bw_kg,
            "julia_threads" => nthreads(),
        ),
        "significance_rule" => Dict(
            "median_ST_across_seeds_above" => "max dummy-null median ST for endpoint",
            "seed_stability_required" => length(seeds) >= 3 ? "2/3 seeds above dummy threshold" : "1 seed above dummy threshold",
        ),
        "outputs" => Dict(
            "parameter_universe" => joinpath(out_dir, "parameter_universe.csv"),
            "efast_sample_matrix" => joinpath(out_dir, "efast_sample_matrix.csv"),
            "simulation_metrics_long" => joinpath(out_dir, "simulation_metrics_long.csv"),
            "paired_delta_metrics" => joinpath(out_dir, "paired_delta_metrics.csv"),
            "efast_indices_by_endpoint" => joinpath(out_dir, "efast_indices_by_endpoint.csv"),
            "dummy_null_thresholds" => joinpath(out_dir, "dummy_null_thresholds.csv"),
            "significant_parameter_screen" => joinpath(out_dir, "significant_parameter_screen.csv"),
            "seed_stability_summary" => joinpath(out_dir, "seed_stability_summary.csv"),
        ),
    )
    write_meta(joinpath(out_dir, "susilo_fig5_il6_dummy_efast_meta.json"), meta)
    println("Completed Susilo/Fig.5 eFAST campaign outputs in $(out_dir)")
end

using Base.Threads
using CSV
using DataFrames
using Dates
using JSON3
using SciMLBase

const VPOP_FIG5_REGIMENS = [r for r in fig5_regimens() if r.name != "no_dose"]

env_bool2(name::String, default::Bool) = begin
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return default
    value in ("1", "true", "yes", "y", "on")
end

function resim_shard_paths(out_dir::AbstractString, shard_id::Int)
    dir = joinpath(out_dir, "trajectory_resim_shards")
    stem = "shard_$(lpad(string(shard_id), 4, '0'))"
    (
        traj = joinpath(dir, "$(stem)_trajectories.csv"),
        waterfall = joinpath(dir, "$(stem)_waterfall.csv"),
        done = joinpath(dir, "$(stem).done.json"),
    )
end

function resim_shard_done(paths)
    isfile(paths.traj) && isfile(paths.waterfall) && isfile(paths.done)
end

function atomic_write_csv(path::AbstractString, df::DataFrame)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid(), ".", threadid())
    CSV.write(tmp, df)
    mv(tmp, path; force = true)
end

function atomic_write_json(path::AbstractString, payload)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid(), ".", threadid())
    open(tmp, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    mv(tmp, path; force = true)
end

function concat_csv_files(paths::Vector{String}, out_path::AbstractString)
    isempty(paths) && error("No shard files to concatenate for $(out_path)")
    mkpath(dirname(out_path))
    tmp = string(out_path, ".tmp")
    open(tmp, "w") do out
        first_file = true
        for path in paths
            open(path, "r") do input
                header = readline(input)
                if first_file
                    println(out, header)
                    first_file = false
                end
                for line in eachline(input)
                    println(out, line)
                end
            end
        end
    end
    mv(tmp, out_path; force = true)
end

function response_category_fig5(pct_change::Float64)
    if pct_change <= -99.9
        "cr"
    elseif pct_change <= -30.0
        "pr"
    elseif pct_change < 20.0
        "sd"
    else
        "pd"
    end
end

function candidate_params(base_params, universe::DataFrame, row)
    values = Vector{Float64}(undef, nrow(universe))
    for (i, pname) in enumerate(String.(universe.parameter))
        values[i] = Float64(row[Symbol(pname)])
    end
    p = apply_sample_to_params(base_params, universe, values)
    for pname in ("KBptumor", "KTrptumor")
        sym = Symbol(pname)
        if hasproperty(row, sym) && MMC.has_parameter(pname)
            value = Float64(getproperty(row, sym))
            isfinite(value) && value > 0.0 && MMC.set_param!(p, pname, value)
        end
    end
    p
end

function simulate_candidate_regimen(row, p, spec::Fig5RegimenSpec, alg, saveat::Vector{Float64}, horizon_days::Float64, profile_horizon_days::Float64, post_event_dt::Float64, bw_kg::Float64)
    regimen = regimen_from_spec(spec, bw_kg)
    try
        built, sol = MMC.solve_regimen(
            regimen,
            p,
            alg;
            tspan = (0.0, horizon_days),
            saveat = saveat,
            callback_mode = :callback,
            post_event_proposed_dt = post_event_dt,
            abstol = MMC.SOLVER_ABSTOL,
            reltol = MMC.SOLVER_RELTOL,
            save_everystep = false,
            maxiters = 1_000_000,
        )
        status = SciMLBase.successful_retcode(sol.retcode) ? "success" : String(Symbol(sol.retcode))
        cache = MMC.zero_observables_cache()
        traj_rows = NamedTuple[]
        for j in eachindex(sol.t)
            t = Float64(sol.t[j])
            t < profile_horizon_days || continue
            MMC.update_observables!(cache, sol.u[j], p, t)
            push!(traj_rows, (
                candidate_id = String(row.candidate_id),
                resim_candidate_rank = Int(row.resim_candidate_rank),
                seed = Int(row.seed),
                efast_eval_idx = Int(row.efast_eval_idx),
                parameter_block = Int(row.parameter_block),
                sample_in_block = Int(row.sample_in_block),
                regimen = spec.name,
                regimen_label = spec.label,
                status = status,
                time_day = t,
                il6combo = Float64(cache.IL6combo),
                tafraction_pb_pct = 100.0 * Float64(cache.Tafraction_pb),
                btumor_perml = Float64(cache.Btumor_perml),
            ))
        end
        MMC.update_observables!(cache, built.initial_u, p, 0.0)
        bt0 = Float64(cache.Btumor_perml)
        MMC.update_observables!(cache, sol.u[end], p, Float64(sol.t[end]))
        bt84 = Float64(cache.Btumor_perml)
        pct_change = 100.0 * (bt84 / max(bt0, 1e-12) - 1.0)
        waterfall = (
            candidate_id = String(row.candidate_id),
            resim_candidate_rank = Int(row.resim_candidate_rank),
            seed = Int(row.seed),
            efast_eval_idx = Int(row.efast_eval_idx),
            parameter_block = Int(row.parameter_block),
            sample_in_block = Int(row.sample_in_block),
            regimen = spec.name,
            regimen_label = spec.label,
            status = status,
            btumor_initial_perml = bt0,
            btumor_day84_perml = bt84,
            tumor_size_change_pct = pct_change,
            response_category = response_category_fig5(pct_change),
            responder_gt50 = pct_change <= -50.0,
        )
        traj_rows, waterfall
    catch err
        status = "error:$(typeof(err)):$(sprint(showerror, err))"
        traj_rows = NamedTuple[]
        waterfall = (
            candidate_id = String(row.candidate_id),
            resim_candidate_rank = Int(row.resim_candidate_rank),
            seed = Int(row.seed),
            efast_eval_idx = Int(row.efast_eval_idx),
            parameter_block = Int(row.parameter_block),
            sample_in_block = Int(row.sample_in_block),
            regimen = spec.name,
            regimen_label = spec.label,
            status = status,
            btumor_initial_perml = NaN,
            btumor_day84_perml = NaN,
            tumor_size_change_pct = NaN,
            response_category = "error",
            responder_gt50 = false,
        )
        traj_rows, waterfall
    end
end

function run_shards!(candidates::DataFrame, universe::DataFrame, out_dir::AbstractString, shard_size::Int, alg, saveat::Vector{Float64}, horizon_days::Float64, profile_horizon_days::Float64, post_event_dt::Float64, bw_kg::Float64)
    n = nrow(candidates)
    shard_count = cld(n, shard_size)
    base_params = build_shipped_dlbcl_base_params()
    MMC.has_parameter("end_time") && MMC.set_param!(base_params, "end_time", horizon_days)
    progress_lock = ReentrantLock()
    completed = Ref(0)
    skipped = Ref(0)

    @threads for shard_id in 1:shard_count
        paths = resim_shard_paths(out_dir, shard_id)
        if resim_shard_done(paths)
            lock(progress_lock)
            try
                skipped[] += 1
                println("Skipping completed trajectory shard $(skipped[] + completed[]) / $(shard_count): shard=$(shard_id)")
                flush(stdout)
            finally
                unlock(progress_lock)
            end
            continue
        end
        lo = (shard_id - 1) * shard_size + 1
        hi = min(shard_id * shard_size, n)
        traj_rows = NamedTuple[]
        waterfall_rows = NamedTuple[]
        for row in eachrow(candidates[lo:hi, :])
            p = candidate_params(base_params, universe, row)
            for spec in VPOP_FIG5_REGIMENS
                trs, wf = simulate_candidate_regimen(row, p, spec, alg, saveat, horizon_days, profile_horizon_days, post_event_dt, bw_kg)
                append!(traj_rows, trs)
                push!(waterfall_rows, wf)
            end
        end
        atomic_write_csv(paths.traj, DataFrame(traj_rows))
        atomic_write_csv(paths.waterfall, DataFrame(waterfall_rows))
        atomic_write_json(paths.done, Dict(
            "completed_at" => string(Dates.now()),
            "shard_id" => shard_id,
            "row_start" => lo,
            "row_end" => hi,
            "n_candidates" => hi - lo + 1,
            "n_trajectory_rows" => length(traj_rows),
            "n_waterfall_rows" => length(waterfall_rows),
        ))
        lock(progress_lock)
        try
            completed[] += 1
            println("Completed trajectory shard $(skipped[] + completed[]) / $(shard_count): shard=$(shard_id) candidates=$(hi - lo + 1) traj_rows=$(length(traj_rows)) waterfall_rows=$(length(waterfall_rows)) at $(Dates.now())")
            flush(stdout)
        finally
            unlock(progress_lock)
        end
    end
end

function write_resim_manifest(out_dir::AbstractString, shard_count::Int)
    rows = NamedTuple[]
    for shard_id in 1:shard_count
        paths = resim_shard_paths(out_dir, shard_id)
        push!(rows, (
            shard_id = shard_id,
            trajectories_path = paths.traj,
            waterfall_path = paths.waterfall,
            done_path = paths.done,
            trajectories_exists = isfile(paths.traj),
            waterfall_exists = isfile(paths.waterfall),
            done_exists = isfile(paths.done),
        ))
    end
    manifest = DataFrame(rows)
    CSV.write(joinpath(out_dir, "trajectory_resim_manifest.csv"), manifest)
    manifest
end

function resimulate_candidates(options)
    out_dir = abspath(options["output"])
    mkpath(out_dir)
    candidates_path = abspath(options["candidates"])
    isfile(candidates_path) || error("Missing candidates CSV: $(candidates_path)")
    candidates = CSV.read(candidates_path, DataFrame)
    sort!(candidates, :resim_candidate_rank)

    ACTIVE_VARIANTS_FILE[] = abspath(options["variants"])
    ranges_path = abspath(options["parameter-ranges"])
    universe = CSV.read(ranges_path, DataFrame)
    for pname in String.(universe.parameter)
        hasproperty(candidates, Symbol(pname)) || error("Candidate parameter CSV missing column $(pname)")
    end

    n_limit = options["preset"] == "quick" ? 20 : 0
    if n_limit > 0 && n_limit < nrow(candidates)
        candidates = candidates[1:n_limit, :]
    end
    shard_size = options["preset"] == "quick" ? 10 : 50
    horizon_days = 84.0
    profile_horizon_days = 42.0
    saveat_dt = options["preset"] == "quick" ? 0.5 : 0.125
    post_event_dt = 0.01
    bw_kg = 70.0
    solver = "rodas4p"
    aggregate = true
    alg = make_solver_alg(solver)
    saveat = collect(0.0:saveat_dt:horizon_days)
    if isempty(saveat) || saveat[end] < horizon_days
        push!(saveat, horizon_days)
    end
    shard_count = cld(nrow(candidates), shard_size)
    println("Starting eFAST-derived Hosseini Fig.5 candidate resimulation: candidates=$(nrow(candidates)), shards=$(shard_count), shard_size=$(shard_size), threads=$(nthreads()), out=$(out_dir)")
    flush(stdout)

    run_shards!(candidates, universe, out_dir, shard_size, alg, saveat, horizon_days, profile_horizon_days, post_event_dt, bw_kg)
    manifest = write_resim_manifest(out_dir, shard_count)
    all(manifest.done_exists) || error("Not all trajectory resimulation shards completed.")

    if aggregate
        traj_paths = String.(manifest.trajectories_path)
        wf_paths = String.(manifest.waterfall_path)
        concat_csv_files(traj_paths, joinpath(out_dir, "candidate_trajectory_summary.csv"))
        concat_csv_files(wf_paths, joinpath(out_dir, "candidate_waterfall.csv"))
    end
    meta = Dict(
        "created_by" => "utils/VPopGeneration.jl --stage resimulate",
        "created_at" => string(Dates.now()),
        "output_root" => out_dir,
        "candidate_parameters_csv" => candidates_path,
        "n_candidates" => nrow(candidates),
        "n_shards" => shard_count,
        "shard_size" => shard_size,
        "aggregate_outputs" => aggregate,
        "simulation_settings" => Dict(
            "solver" => solver,
            "horizon_days" => horizon_days,
            "profile_horizon_days" => profile_horizon_days,
            "saveat_dt" => saveat_dt,
            "post_event_proposed_dt" => post_event_dt,
            "bw_kg" => bw_kg,
            "julia_threads" => nthreads(),
        ),
        "outputs" => Dict(
            "trajectory_resim_manifest" => joinpath(out_dir, "trajectory_resim_manifest.csv"),
            "candidate_trajectory_summary" => joinpath(out_dir, "candidate_trajectory_summary.csv"),
            "candidate_waterfall" => joinpath(out_dir, "candidate_waterfall.csv"),
        ),
    )
    atomic_write_json(joinpath(out_dir, "candidate_resim_meta.json"), meta)
    println("Completed eFAST-derived Hosseini Fig.5 candidate resimulation in $(out_dir)")
    flush(stdout)
end

function parse_cli(args)
    options = Dict{String,String}(
        "stage" => "efast",
        "preset" => "quick",
        "output" => joinpath(REPO_ROOT, "output", "vpop"),
        "parameter-ranges" => DEFAULT_RANGES,
        "variants" => DEFAULT_VARIANTS,
        "candidates" => "auto",
        "seed" => "20260502",
    )
    index = 1
    while index <= length(args)
        flag = args[index]
        flag in ("--stage", "--preset", "--output", "--parameter-ranges", "--variants", "--candidates", "--seed") ||
            throw(ArgumentError("Unknown option: $flag"))
        index == length(args) && throw(ArgumentError("Missing value after $flag"))
        options[flag[3:end]] = args[index + 1]
        index += 2
    end
    options["stage"] in ("efast", "resimulate") ||
        throw(ArgumentError("--stage must be efast or resimulate"))
    options["preset"] in ("quick", "paper") ||
        throw(ArgumentError("--preset must be quick or paper"))
    if options["candidates"] == "auto"
        options["candidates"] = joinpath(abspath(options["output"]), "trajectory_resim_candidate_parameters.csv")
    end
    return options
end

function main(args=ARGS)
    options = parse_cli(args)
    if options["stage"] == "efast"
        run_efast(options)
    else
        resimulate_candidates(options)
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
