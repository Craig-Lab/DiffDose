#!/usr/bin/env julia

using ForwardDiff
using Random

"""Draw standard Gumbel noise with finite endpoint protection."""
function gumbel_noise(rng::AbstractRNG, dims::Int...)
    u = clamp.(rand(rng, dims...), eps(Float64), 1.0 - eps(Float64))
    return -log.(-log.(u))
end

"""Map one logits column per control to relaxed values on `levels`."""
function relaxed_controls(logits, levels, noise; temperature=0.25)
    size(logits) == size(noise) || throw(DimensionMismatch("logits/noise mismatch"))
    size(logits, 1) == length(levels) || throw(DimensionMismatch("level mismatch"))
    scores = (logits .+ noise) ./ temperature
    scores = scores .- maximum(scores; dims=1)
    probabilities = exp.(scores)
    probabilities ./= sum(probabilities; dims=1)
    return vec(sum(reshape(levels, :, 1) .* probabilities; dims=1))
end

# Replace this toy loss with a differentiable mechanistic simulation and its
# clinically aligned objective. Hold the sampled noise fixed within a gradient
# evaluation so the optimized map is deterministic.
mechanistic_loss(controls) = sum(abs2, controls .- [1.0, 5.0, 10.0])

rng = MersenneTwister(42)
levels = [0.0, 1.0, 5.0, 10.0]
logits0 = zeros(length(levels), 3)
noise = gumbel_noise(rng, size(logits0)...)
loss(flat_logits) = mechanistic_loss(
    relaxed_controls(reshape(flat_logits, size(logits0)), levels, noise),
)

gradient = ForwardDiff.gradient(loss, vec(logits0))
relaxed = relaxed_controls(logits0, levels, noise)
quantized = [levels[argmin(abs.(levels .- value))] for value in relaxed]

println("relaxed controls = ", relaxed)
println("quantized controls = ", quantized)
println("gradient norm = ", sqrt(sum(abs2, gradient)))

# The same construction applies to quantized administration times by replacing
# `levels` with the admissible timing grid.
