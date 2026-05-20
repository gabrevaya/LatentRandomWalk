"""
    LatentRandomWalk.SDE

Continuous-time reformulation of the discourse random walk: Brownian motion
on `S^{d-1}` driving the same log-linear word-emission model. We integrate
the Stratonovich SDE

    dc_t = (I − c_t c_tᵀ) ∘ dW_t

which (unlike its Itô form) keeps `c_t` on the sphere exactly in the
continuous limit. In practice we use a projected Euler–Maruyama step —
tangent-space increment followed by renormalisation — which is the
standard simple geometric integrator for SDEs on the sphere and exactly
what the implementation plan calls for ("project back to the sphere after
each step"). For `d = 300` this is both fast and stable; the matrix-form ambient-space
Itô SDE has a stiff `(d−1)/2` drift that would force a much smaller step
from a generic Euler-Maruyama integrator.
"""
module SDE

using LinearAlgebra: norm, dot, mul!
using Random: AbstractRNG, MersenneTwister, randn!
using Statistics: mean
using ..LatentRandomWalk: Embeddings

"""
    simulate(d; tspan=(0.0, 100.0), dt=0.01, seed=0)
        -> snapshots::Vector{Vector{Float64}}, times::Vector{Float64}

Integrate Stratonovich Brownian motion on `S^{d-1}` via a projected Euler
step. Each saved point is on the unit sphere by construction.
"""
function simulate(d::Integer; tspan::Tuple{<:Real,<:Real} = (0.0, 100.0),
                  dt::Real = 0.01, seed::Integer = 0)
    rng = MersenneTwister(seed)
    t0, t1 = float(tspan[1]), float(tspan[2])
    dt = float(dt)
    n_steps = Int(round((t1 - t0) / dt))
    @assert n_steps > 0 "tspan / dt produces no steps"
    sqrt_dt = sqrt(dt)

    c = randn(rng, Float64, d); c ./= norm(c)
    ξ = Vector{Float64}(undef, d)

    snaps = Vector{Vector{Float64}}(undef, n_steps + 1)
    times = Vector{Float64}(undef, n_steps + 1)
    snaps[1] = copy(c);  times[1] = t0

    @inbounds for k in 1:n_steps
        randn!(rng, ξ)
        proj = dot(ξ, c)                 # project onto tangent space
        @simd for i in 1:d
            c[i] += sqrt_dt * (ξ[i] - proj * c[i])
        end
        c ./= norm(c)                    # renormalise (suppresses drift)
        snaps[k + 1] = copy(c)
        times[k + 1] = t0 + k * dt
    end
    return snaps, times
end

# ─── Emissions: sample words from the log-linear model ─────────────────────

# In-place sampling: given c and V (d × n), draw one word index from
# p(w | c) ∝ exp(⟨v_w, c⟩). `logits` is reused across calls.
function _sample_word!(logits::Vector{T}, c::AbstractVector{T}, V::AbstractMatrix{T},
                       rng::AbstractRNG) where T<:AbstractFloat
    mul!(logits, transpose(V), c)
    m = maximum(logits)
    Z = zero(T)
    @inbounds @simd for j in eachindex(logits)
        e = exp(logits[j] - m)
        logits[j] = e
        Z += e
    end
    u = rand(rng, T) * Z
    acc = zero(T)
    @inbounds for j in eachindex(logits)
        acc += logits[j]
        u <= acc && return j
    end
    return length(logits)
end

"""
    simulate_corpus(emb, snaps; rng=MersenneTwister(0)) -> Vector{Int32}

Given a trajectory `snaps` of discourse vectors on `S^{d-1}`, emit one word
per snapshot according to the log-linear model `p(w | c) ∝ exp(⟨v_w, c⟩)`.
Each `c` is rescaled to `‖c‖ = 4/μ_w` first (paper §5's regime).
"""
function simulate_corpus(emb::Embeddings{T}, snaps::Vector{<:AbstractVector};
                         rng::AbstractRNG = MersenneTwister(0)) where T<:AbstractFloat
    V = emb.V
    d, n = size(V)
    μ_w = mean(j -> norm(@view V[:, j]), 1:n)
    c_scale = T(4) / T(μ_w)

    out = Vector{Int32}(undef, length(snaps))
    c   = Vector{T}(undef, d)
    logits = Vector{T}(undef, n)
    @inbounds for t in eachindex(snaps)
        @views c .= snaps[t]
        c .*= c_scale
        out[t] = Int32(_sample_word!(logits, c, V, rng))
    end
    return out
end

# ─── Sanity check: Z_c along an SDE path ───────────────────────────────────

"""
    partition_function_along_path(emb, snaps)

Compute `Z_c = Σ_w exp(⟨v_w, c⟩)` along an SDE trajectory (snapshots
rescaled to `‖c‖ = 4/μ_w`). The model's self-normalisation prediction is
that this stays approximately constant — a striking direct test of
Lemma 2.1 with the *continuous-time* discourse process.
"""
function partition_function_along_path(emb::Embeddings{T},
                                       snaps::Vector{<:AbstractVector}) where T<:AbstractFloat
    V = emb.V
    d, n = size(V)
    Vt = transpose(V)
    μ_w = mean(j -> norm(@view V[:, j]), 1:n)
    c_scale = T(4) / T(μ_w)

    Z = Vector{T}(undef, length(snaps))
    c = Vector{T}(undef, d)
    buf = Vector{T}(undef, n)
    @inbounds for t in eachindex(snaps)
        @views c .= snaps[t]
        c .*= c_scale
        mul!(buf, Vt, c)
        Z[t] = sum(exp, buf)
    end
    return Z
end

end  # module SDE
