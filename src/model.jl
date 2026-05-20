using LinearAlgebra: norm, dot
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, nonzeros
using Random: Random, AbstractRNG, MersenneTwister, shuffle!
using Printf: @sprintf
using ProgressMeter: Progress, next!, finish!

# ───────────────────────────── public types ────────────────────────────────

"""
    Embeddings{T}

A trained set of word vectors. `V` is `d × n` (columns are word vectors), `C`
is the fitted bias scalar `≈ -2 log Z` from the SN objective, and `vocab` is
the associated vocabulary.
"""
struct Embeddings{T<:AbstractFloat}
    V::Matrix{T}
    C::T
    vocab::Vocabulary
end

dim(e::Embeddings) = size(e.V, 1)
Base.length(e::Embeddings) = size(e.V, 2)
Base.show(io::IO, e::Embeddings{T}) where T =
    print(io, "Embeddings{", T, "}(d=", dim(e), ", n=", length(e), ", C=", e.C, ")")

"""
    vector(emb, word) -> view into emb.V

Return the column of `emb.V` for `word`. Throws `KeyError` if absent.
"""
function vector(e::Embeddings, word::AbstractString)
    id = word_id(e.vocab, word)
    id == 0 && throw(KeyError(word))
    return @view e.V[:, id]
end

"""
    TrainingConfig(; d, lr, epochs, X_max, eps, shuffle, log_every, seed)

Hyperparameters for SN training. Defaults follow Arora et al. (2016):
`d=300`, `lr=0.05`, `X_max=100`, `epochs=100`. Numerical fields are
`Float32`; `train` converts them to `eltype(X)` as needed.
"""
Base.@kwdef struct TrainingConfig
    d::Int          = 300
    lr::Float32     = 0.05f0
    epochs::Int     = 100
    X_max::Float32  = 100f0
    eps::Float32    = 1f-8
    shuffle::Bool   = true
    log_every::Int  = 1
    seed::Int       = 0
end

# ─────────────────────────── pair extraction ───────────────────────────────

# Extract the strict upper-triangle (row < col) of a symmetric SparseMatrixCSC
# as plain dense vectors. CSC iteration via `nzrange` is allocation-free.
function _upper_triangle_pairs(X::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @assert size(X, 1) == size(X, 2) "X must be square"
    n = size(X, 2)
    rows = rowvals(X)
    nzv  = nonzeros(X)
    cap = nnz(X) ÷ 2 + 1
    I = Vector{Ti}(undef, cap); J = Vector{Ti}(undef, cap); V = Vector{Tv}(undef, cap)
    k = 0
    @inbounds for col in 1:n
        for idx in nzrange(X, col)
            row = rows[idx]
            row < col || continue
            k += 1
            if k > length(I)
                grown = 2 * length(I)
                resize!(I, grown); resize!(J, grown); resize!(V, grown)
            end
            I[k] = row; J[k] = col; V[k] = nzv[idx]
        end
    end
    resize!(I, k); resize!(J, k); resize!(V, k)
    return I, J, V
end

# ────────────────────────────── loss helper ────────────────────────────────

"""
    sn_loss(V, C, X; X_max=100)

Weighted SN loss (sum of `min(X, X_max) * (log X − ‖v_w + v_w'‖² − C)²` over
the strict upper triangle of `X`). Used in tests; the training loop computes
this inline.
"""
function sn_loss(V::AbstractMatrix{T}, C::T, X::SparseMatrixCSC{<:AbstractFloat,<:Integer};
                 X_max::Real = 100) where T<:AbstractFloat
    rows, cols, vals = _upper_triangle_pairs(X)
    d = size(V, 1)
    Xmax = T(X_max)
    L = zero(T)
    @inbounds for idx in eachindex(rows)
        w = Int(rows[idx]); w′ = Int(cols[idx])
        Xww = T(vals[idx])
        s = zero(T)
        @simd for k in 1:d
            vsum_k = V[k, w] + V[k, w′]
            s = muladd(vsum_k, vsum_k, s)
        end
        r = log(Xww) - s - C
        L += min(Xww, Xmax) * r * r
    end
    return L
end

# ─────────────────────────── training (hot path) ───────────────────────────

# A single epoch. Returns (loss_sum, weight_sum). Type-stable thanks to the
# function barrier — every numeric type is concrete inside this body.
function _epoch!(V::Matrix{T}, G_V::Matrix{T}, C::Ref{T}, G_C::Ref{T},
                 rowI::Vector{Ti}, colJ::Vector{Ti}, X_vals::Vector{T},
                 perm::Vector{Int},
                 lr::T, eps::T, Xmax::T, d::Int) where {T<:AbstractFloat, Ti<:Integer}
    loss_sum = zero(T)
    weight_sum = zero(T)
    @inbounds for idx in perm
        w  = Int(rowI[idx])
        w′ = Int(colJ[idx])
        Xww = X_vals[idx]
        X̄   = min(Xww, Xmax)
        logX = log(Xww)

        # 1) residual r = log X − ‖v_w + v_w'‖² − C
        s = zero(T)
        @simd for k in 1:d
            vsum_k = V[k, w] + V[k, w′]
            s = muladd(vsum_k, vsum_k, s)
        end
        r = logX - s - C[]

        loss_sum  += X̄ * r * r
        weight_sum += X̄

        # 2) AdaGrad updates for v_w and v_w' (gradient is identical for both)
        wr4 = T(-4) * X̄ * r
        @simd for k in 1:d
            vsum_k = V[k, w] + V[k, w′]
            g = wr4 * vsum_k
            g2 = g * g

            gw  = G_V[k, w] + g2;  G_V[k, w]  = gw
            V[k, w]  -= lr * g / (sqrt(gw)  + eps)

            gp  = G_V[k, w′] + g2; G_V[k, w′] = gp
            V[k, w′] -= lr * g / (sqrt(gp) + eps)
        end

        # 3) AdaGrad update for the bias scalar C
        gC = T(-2) * X̄ * r
        gC2 = G_C[] + gC * gC
        G_C[] = gC2
        C[]  -= lr * gC / (sqrt(gC2) + eps)
    end
    return loss_sum, weight_sum
end

"""
    train(X, vocab; kwargs...) -> (Embeddings, losses)

Fit SN word vectors to the symmetric co-occurrence matrix `X` by stochastic
gradient descent with per-parameter AdaGrad step sizes. Returns the
`Embeddings` and a `Vector{Float64}` of per-epoch weighted MSEs.

Keyword arguments populate a `TrainingConfig{eltype(X)}`. See its docstring.
"""
function train(X::SparseMatrixCSC{T, <:Integer}, vocab::Vocabulary; kwargs...) where T<:AbstractFloat
    return train(X, vocab, TrainingConfig(; kwargs...))
end

function train(X::SparseMatrixCSC{T, Ti}, vocab::Vocabulary, config::TrainingConfig) where {T<:AbstractFloat, Ti<:Integer}
    n = size(X, 1)
    @assert size(X, 2) == n "X must be square"
    @assert length(vocab) == n "X size and vocab length disagree"

    d   = config.d
    lr  = T(config.lr)
    eps = T(config.eps)
    Xmax = T(config.X_max)

    rng = MersenneTwister(config.seed)
    V   = (rand(rng, T, d, n) .- T(0.5)) ./ T(d)
    G_V = zeros(T, d, n)
    C   = Ref(log(Xmax))                # GloVe-style C₀ ≈ centre of residuals
    G_C = Ref(zero(T))

    rowI, colJ, X_vals = _upper_triangle_pairs(X)
    n_pairs = length(rowI)
    @info "SN training: n=$n  d=$d  pairs=$(n_pairs)  epochs=$(config.epochs)  lr=$(lr)"

    perm = collect(1:n_pairs)
    losses = Float64[]
    sizehint!(losses, config.epochs)

    prog = Progress(config.epochs; dt = 1.0, desc = "Training: ")
    for epoch in 1:config.epochs
        config.shuffle && shuffle!(rng, perm)
        loss_sum, weight_sum = _epoch!(V, G_V, C, G_C, rowI, colJ, X_vals, perm,
                                       lr, eps, Xmax, d)
        wmse = Float64(loss_sum / weight_sum)
        push!(losses, wmse)
        next!(prog; showvalues = [(Symbol("epoch"), epoch),
                                  (Symbol("weighted MSE"), @sprintf("%.6e", wmse))])
    end
    finish!(prog)

    return Embeddings(V, C[], vocab), losses
end
