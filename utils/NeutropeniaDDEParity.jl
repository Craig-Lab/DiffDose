#!/usr/bin/env julia

"""
Validate the translated 12-state neutropenia DDE against MATLAB trajectories.

The fixtures were exported from the published Craig-model MATLAB workflow for
one CHOP-14 cycle under no G-CSF, daily 300 microgram G-CSF on days 1--13, and
one 300 microgram G-CSF dose on day 8. The declared-lag callback solver is used
here because this test concerns forward-model parity. The manuscript timing
gradients instead differentiate the explicitly smoothed, undeclared-lag solve.

Scientific sources:
  Craig et al. (2015), https://doi.org/10.1016/j.jtbi.2015.08.015
  Craig, Humphries & Mackey (2016), https://doi.org/10.1007/s11538-016-0179-8
  Craig (2017), https://doi.org/10.1002/psp4.12191

Run from the repository root:

    julia +1.11.6 --project=environments/dde utils/NeutropeniaDDEParity.jl
"""

using CSV
using DataFrames
using SciMLBase
using Test

include(joinpath(@__DIR__, "NeutropeniaDDEModel.jl"))
using .NeutropeniaDDEModel.Parameters
using .NeutropeniaDDEModel.Dosing
using .NeutropeniaDDEModel.SolverDeclared

const ROOT = normpath(joinpath(@__DIR__, ".."))
const ANC_SCALE = 8.19
const MAX_ANC_ERROR = 0.5

const CASES = (
    (
        name="no G-CSF",
        path=joinpath(ROOT, "data", "NeutropeniaParityNoGCSF.csv"),
        times=Float64[],
    ),
    (
        name="daily G-CSF",
        path=joinpath(ROOT, "data", "NeutropeniaParityDailyGCSF.csv"),
        times=collect(1.0:1.0:13.0),
    ),
    (
        name="day-8 G-CSF",
        path=joinpath(ROOT, "data", "NeutropeniaParityDay8GCSF.csv"),
        times=Float64[8.0],
    ),
)

function parity_parameters(has_gcsf)
    parameters = default_parameters()
    parameters.AdminChemo = 1
    parameters.AdminGCSF = has_gcsf ? 1 : 0
    parameters.DayAdminChemo = 0.0
    parameters.DeltaC = 1.0 / 24.0
    parameters.NumAdminsChemo = 1
    parameters.Period = 14.0
    parameters.DoseChemo = 4000.0
    parameters.TotalDose = parameters.DoseChemo * parameters.BSA
    parameters.GCSFPeriod = 1.0
    parameters.Dose = 300_000.0
    return recompute_derived!(parameters)
end

@testset "Neutropenia MATLAB parity" begin
    for case in CASES
        reference = CSV.read(case.path, DataFrame)
        parameters = parity_parameters(!isempty(case.times))
        solution = solve_scenario_declared(
            parameters,
            (0.0, 14.0);
            chemo_schedule=Float64[0.0],
            gcsf_schedule=case.times,
            smooth_chemo=false,
            smooth_gcsf=false,
            reltol=1e-6,
            abstol=1e-6,
            dtmax=5e-3,
            saveat=Float64.(reference.time),
            save_everystep=false,
            maxiters=400_000,
        )

        @test solution.retcode == SciMLBase.ReturnCode.Success
        @test length(solution.t) == nrow(reference)
        @test maximum(abs.(Float64.(solution.t) .- Float64.(reference.time))) <= 1e-10

        predicted_anc = ANC_SCALE .* [state[2] for state in solution.u]
        max_error = maximum(abs.(predicted_anc .- Float64.(reference.ANC)))
        @test max_error <= MAX_ANC_ERROR
        println(case.name, " maximum ANC error: ", max_error, " x 10^9 cells/L")
    end
end
