# Phase E (stretch) — Continuous-time discourse process on S^{d-1}.
# Projected Euler-Maruyama for Stratonovich Brownian motion on the sphere
# (see src/sde.jl), driving the same log-linear word-emission model.
# Demonstrates that Lemma 2.1's partition-function concentration holds along
# a continuous-time trajectory using the trained embeddings.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using LatentRandomWalk.SDE: simulate, partition_function_along_path, simulate_corpus
using JLD2: jldsave, jldopen
using CairoMakie
using Statistics: mean, std
using LinearAlgebra: norm
using Printf: @sprintf

const VECTORS_PATH = joinpath(RESULTS_DIR, "vectors.jld2")
const SDE_OUT      = joinpath(RESULTS_DIR, "sde.jld2")

emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
d   = dim(emb)
@info "Loaded embeddings: d=$d, n=$(length(emb))"

@info "Simulating Brownian motion on S^{d-1}…"
snaps, ts = @phase "SDE simulate" simulate(d; tspan = (0.0, 50.0), dt = 0.05, seed = 0)

# Norm-preservation: the projected scheme is supposed to keep ‖c_t‖ = 1
# exactly, modulo floating-point. Print and assert the deviation.
norm_dev = maximum(abs.(map(norm, snaps) .- 1.0))
@info @sprintf("  %d snapshots;  max |‖c_t‖ − 1| = %.3e", length(snaps), norm_dev)
@assert norm_dev < 1e-10 "SDE snapshots drifted off the sphere by $norm_dev — projection broken?"

@info "Partition function along path…"
Zpath = @phase "Z along path" partition_function_along_path(emb, snaps)
Z̄        = mean(Zpath)
σ_over_Z = std(Zpath) / Z̄
@info @sprintf("  Z̄ = %.4e   σ_Z / Z̄ = %.4f", Z̄, σ_over_Z)

fig = Figure(size = (760, 420))
ax  = Axis(fig[1, 1],
           title = @sprintf("SDE  Z_c along a Brownian path on S^{d-1}   (σ_Z/Z̄ = %.4f,  max |‖c‖ − 1| = %.1e)",
                            σ_over_Z, norm_dev),
           xlabel = "time", ylabel = "Z_c / 𝔼[Z_c]")
lines!(ax, ts, Float64.(Zpath ./ Z̄); color = :steelblue, label = "Z_c(t) / 𝔼[Z_c]")
hlines!(ax, [0.9, 1.1]; color = :red, linestyle = :dash, label = "[0.9, 1.1]")
axislegend(ax; position = :rt)
save(joinpath(FIGURES_DIR, "E_partition_along_sde.pdf"), fig)

# Sanity: sample a short stream of words from the SDE and report unigram counts
@info "Sampling words from the SDE path (just the first 500 snapshots)…"
sampled = @phase "simulate_corpus" simulate_corpus(emb, snaps[1:min(500, end)])
@info "First 30 sampled words: $(join(emb.vocab.id2word[Int.(sampled[1:30])], ' '))"

jldsave(SDE_OUT; snapshots = snaps, times = ts, Z_path = Zpath, sampled = sampled,
                 norm_dev = norm_dev, σ_over_Z = σ_over_Z)
@info "Wrote $SDE_OUT"
