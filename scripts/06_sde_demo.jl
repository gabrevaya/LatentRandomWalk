# Phase E (stretch) — Continuous-time discourse process on S^{d-1}.
# Demonstrates that Brownian motion on the sphere, integrated by
# StochasticDiffEq.jl, reproduces the partition-function concentration
# (Lemma 2.1) along an SDE trajectory using the trained embeddings.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using LatentRandomWalk.SDE: simulate, partition_function_along_path, simulate_corpus
using JLD2: jldsave, jldopen
using CairoMakie
using Statistics: mean, std
using LinearAlgebra: norm

const VECTORS_PATH = joinpath(RESULTS_DIR, "vectors.jld2")
const SDE_OUT      = joinpath(RESULTS_DIR, "sde.jld2")

emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
d   = dim(emb)
@info "Loaded embeddings: d=$d, n=$(length(emb))"

@info "Simulating Brownian motion on S^{d-1}…"
snaps, ts = @phase "SDE simulate" simulate(d; tspan = (0.0, 50.0), dt = 0.05, seed = 0)
@info "$(length(snaps)) snapshots; norm range = $(extrema(norm.(snaps)))"

@info "Partition function along path…"
Zpath = @phase "Z along path" partition_function_along_path(emb, snaps)
@info "  Z̄ = $(mean(Zpath))   σ_Z / Z̄ = $(round(std(Zpath) / mean(Zpath); digits=4))"

fig = Figure(size = (760, 420))
ax  = Axis(fig[1, 1], title = "SDE  Z_c along a Brownian path on S^{d-1}",
           xlabel = "time", ylabel = "Z_c / 𝔼[Z_c]")
lines!(ax, ts, Float64.(Zpath ./ mean(Zpath)); color = :steelblue)
hlines!(ax, [0.9, 1.1]; color = :red, linestyle = :dash, label = "[0.9, 1.1]")
axislegend(ax; position = :rt)
save(joinpath(FIGURES_DIR, "E_partition_along_sde.pdf"), fig)

# Sanity: sample a short stream of words from the SDE and report unigram counts
@info "Sampling words from the SDE path (just the first 500 snapshots)…"
sampled = @phase "simulate_corpus" simulate_corpus(emb, snaps[1:min(500, end)])
@info "First 30 sampled words: $(join(emb.vocab.id2word[Int.(sampled[1:30])], ' '))"

jldsave(SDE_OUT; snapshots = snaps, times = ts, Z_path = Zpath, sampled = sampled)
@info "Wrote $SDE_OUT"
