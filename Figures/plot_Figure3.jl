#!/usr/bin/env julia
"""Regenerate the computational panels underlying Main Figure 3."""

using CSV
using DataFrames
using Plots

function option(args, flag, default)
    index = findfirst(==(flag), args)
    index === nothing && return default
    index == length(args) && error("Missing value after $flag")
    return args[index + 1]
end

function main(args=ARGS)
    input_dir = abspath(option(args, "--input", "output"))
    output = abspath(option(args, "--output", "output/figures/Figure3_computational.svg"))
    convergence = CSV.read(joinpath(input_dir, "NeutropeniaDDE_convergence.csv"), DataFrame)
    anc = CSV.read(joinpath(input_dir, "NeutropeniaDDE_anc_traces.csv"), DataFrame)
    palette = Dict(
        "AD+ADAM" => "#a6611a",
        "AD+LBFGS" => "#df9b2b",
        "FD+ADAM" => "#244a87",
        "FD+LBFGS" => "#78a6d4",
        "Nelder-Mead" => "#bdbdbd",
        "Simulated Annealing" => "#636363",
    )
    p_time = plot(yscale=:log10, xlabel="Wall-clock time (s)", ylabel="ANC-deficit loss")
    p_solve = plot(yscale=:log10, xlabel="Counted forward solve", ylabel="")
    for method in unique(convergence.method)
        subset = convergence[convergence.method .== method, :]
        plot!(p_time, subset.wall_s, subset.best_loss; label=method, color=get(palette, method, :black))
        plot!(p_solve, subset.solves, subset.best_loss; label=false, color=get(palette, method, :black))
    end
    p_anc = plot(xlabel="Time (days)", ylabel="ANC (cells/L x 10^9)")
    for scenario in unique(anc.scenario)
        subset = anc[anc.scenario .== scenario, :]
        plot!(p_anc, subset.time, subset.anc; label=scenario)
    end
    hline!(p_anc, [2.0]; color=:gray, linestyle=:dot, label="neutropenia threshold")
    figure = plot(p_time, p_solve, p_anc; layout=(2, 2), size=(900, 650))
    mkpath(dirname(output))
    savefig(figure, output)
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
