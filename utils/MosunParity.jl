#!/usr/bin/env julia

"""
Validate the translated 36-state mosunetuzumab model against an exported
MATLAB/SimBiology trajectory.

The reference trajectory was generated from `TDBr26_6_paper.sbproj` in the
official source-code archive accompanying Hosseini et al. (2020):
https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-020-00145-7/MediaObjects/41540_2020_145_MOESM3_ESM.zip

It represents patient 1 under 1, 2, 20, 20, 20, and 20 mg doses on days
0, 7, 14, 21, 42, and 63. The compact fixture checks the two outputs used by
the manuscript objective without redistributing the SimBiology project.

Run from the repository root:

    julia +1.12.5 --project=environments/mosun utils/MosunParity.jl
"""

using CSV
using DataFrames
using SciMLBase
using Test

include(joinpath(@__DIR__, "MosunModel.jl"))
using .MosunModel

const ROOT = normpath(joinpath(@__DIR__, ".."))
const REFERENCE_PATH = joinpath(ROOT, "data", "MosunParityReference.csv")
const PARAMETER_PATH = joinpath(ROOT, "data", "MosunParityParameters.csv")
const VARIANT_PATH = joinpath(ROOT, "data", "MosunVariantOverrides.tsv")
const ACTIVE_VARIANTS = Set([5, 9, 14, 20, 24, 25, 27, 28])
const DERIVED_CONCENTRATION_ROWS = Set([
    "Trtiss_perml",
    "Btiss_perml",
    "Trtiss2_perml",
    "Btiss2_perml",
    "Trtiss3_perml",
    "B1920tiss3_perml",
    "B19no20tiss3_perml",
    "B19tiss3_perml",
    "Trtumor_perml",
    "Btumor_perml",
])
const DOSE_DAYS = Float64[0, 7, 14, 21, 42, 63]
const DOSES_MG = Float64[1, 2, 20, 20, 20, 20]
const BODY_WEIGHT_KG = 70.0

function apply_active_variants!(parameters)
    variants = CSV.read(VARIANT_PATH, DataFrame; delim='\t')
    for row in eachrow(variants)
        Int(row.idx) in ACTIVE_VARIANTS || continue
        String(row.action) == "parameter" || continue
        String(row.name) == "Value" || continue
        value = tryparse(Float64, strip(string(row.value)))
        value === nothing && continue
        name = String(row.class)
        MosunModel.has_parameter(name) && MosunModel.set_param!(parameters, name, value)
    end
    return parameters
end

function parity_parameters()
    parameters = apply_active_variants!(MosunModel.default_params())
    for row in eachrow(CSV.read(PARAMETER_PATH, DataFrame))
        name = String(row.name)
        if MosunModel.has_parameter(name)
            MosunModel.set_param!(parameters, name, Float64(row.value))
        elseif !(name in DERIVED_CONCENTRATION_ROWS)
            error("Unknown parity parameter: $name")
        end
    end
    return parameters
end

function parity_regimen()
    events = [
        MosunModel.MosunRegimenEvent(
            target=:TDBc_ugperkg,
            time=day,
            amount=dose * 1000.0 / BODY_WEIGHT_KG,
            rate=0.0,
        ) for (day, dose) in zip(DOSE_DAYS, DOSES_MG)
    ]
    return MosunModel.MosunRegimen(events=events)
end

normalized_max_error(predicted, reference) =
    maximum(abs.(predicted .- reference)) / max(maximum(abs.(reference)), eps(Float64))

reference = CSV.read(REFERENCE_PATH, DataFrame)
parameters = parity_parameters()
_, solution = MosunModel.solve_regimen(
    parity_regimen(),
    parameters,
    MosunModel.make_solver_alg("cvode_bdf");
    tspan=(0.0, 84.0),
    saveat=Float64.(reference.time),
    callback_mode=:callback,
    post_event_proposed_dt=0.01,
    abstol=1e-8,
    reltol=1e-5,
    save_everystep=false,
    maxiters=1_000_000,
)

@testset "Mosunetuzumab SimBiology parity" begin
    @test solution.retcode == SciMLBase.ReturnCode.Success
    @test length(solution.t) == nrow(reference)
    @test maximum(abs.(Float64.(solution.t) .- Float64.(reference.time))) <= 1e-10

    cache = MosunModel.zero_observables_cache()
    predicted_il6 = [
        MosunModel.value_at(state, parameters, time, :IL6combo, cache)
        for (state, time) in zip(solution.u, solution.t)
    ]
    predicted_tumor = [
        MosunModel.value_at(state, parameters, time, :Btumor, cache)
        for (state, time) in zip(solution.u, solution.t)
    ]

    il6_error = normalized_max_error(predicted_il6, Float64.(reference.IL6combo))
    tumor_error = normalized_max_error(predicted_tumor, Float64.(reference.Btumor))

    @test il6_error <= 0.01
    @test tumor_error <= 0.002

    println("IL-6 normalized maximum error: ", il6_error)
    println("Tumor normalized maximum error: ", tumor_error)
end
