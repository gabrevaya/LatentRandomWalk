using LinearAlgebra: norm, dot, svdvals, svd, mul!
using SparseArrays: SparseMatrixCSC
using Statistics: mean, std, cor
using Random: AbstractRNG, MersenneTwister, shuffle!

# ─── D.1  Partition-function concentration (Lemma 2.1) ─────────────────────

"""
    partition_function(emb; n_samples=1000, rng=MersenneTwister(0))

Sample `n_samples` directions uniformly on the unit sphere, rescale to
`‖c‖ = 4 / μ_w` (paper §5), and return the empirical distribution of
`Z_c = Σ_w exp(⟨v_w, c⟩)` together with its mean.

Each `Z_c` is one BLAS GEMV (`V' * c`) followed by an `exp`-`sum` reduction.
"""
function partition_function(emb::Embeddings{T};
                            n_samples::Integer = 1000,
                            rng::AbstractRNG  = MersenneTwister(0)) where T<:AbstractFloat
    V = emb.V
    d, n = size(V)

    μ_w = zero(T)
    @inbounds for j in 1:n
        μ_w += norm(@view V[:, j])
    end
    μ_w /= n
    c_scale = T(4) / μ_w

    Z   = Vector{T}(undef, n_samples)
    buf = Vector{T}(undef, n)
    Vt  = transpose(V)
    @inbounds for s in 1:n_samples
        c = randn(rng, T, d)
        c .*= c_scale / norm(c)
        mul!(buf, Vt, c)
        Z[s] = sum(exp, buf)
    end
    Z̄ = mean(Z)
    return (Z = Z, mean_Z = Z̄, c_norm = c_scale, normalised = Z ./ Z̄)
end

# ─── D.2  Squared norm vs log-frequency (Theorem 2.2 single-word case) ─────

"""
    norm_vs_frequency(emb)

Return `(sq_norms, log_freqs, correlation)` — the predicted linear relation
of Theorem 2.2.
"""
function norm_vs_frequency(emb::Embeddings)
    V = emb.V
    counts = emb.vocab.counts
    n = length(counts)
    total = sum(counts)

    sq_norms  = Vector{Float64}(undef, n)
    log_freqs = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        sq_norms[j]  = Float64(sum(abs2, @view V[:, j]))
        log_freqs[j] = log(counts[j] / total)
    end
    return (sq_norms = sq_norms, log_freqs = log_freqs,
            correlation = cor(sq_norms, log_freqs))
end

# ─── D.3  Random-matrix-like singular values (Theorem 4.1) ─────────────────

"""
    singular_value_isotropy(emb)

Return the singular values of `V` together with their RMS, the smallest
non-zero value and the ratio that the paper predicts to be ≈ 1/3.
"""
function singular_value_isotropy(emb::Embeddings)
    σ = svdvals(emb.V)
    rms = sqrt(mean(abs2, σ))
    nz_min = minimum(s for s in σ if s > 1e-10 * rms)
    return (singular_values = σ, rms = rms, min_nonzero = nz_min,
            ratio = nz_min / rms)
end

# ─── D.4  PMI ≈ ⟨v_w, v_w'⟩ / d (Corollary 2.3) ────────────────────────────

"""
    pmi_scatter(emb, X; n_samples=50_000, x_floor=10, rng=MersenneTwister(0))

Sample word pairs with `X[w, w'] ≥ x_floor` from the upper triangle of `X`,
return the empirical PMI (computed from windowed co-occurrence counts) and
the model's prediction `⟨v_w, v_w'⟩`. Corollary 2.3 (in SN units) predicts
a linear relation with slope 1 and intercept `γ = log(q(q−1)/2)` from the
window size — `γ = log 45 ≈ 3.81` for the default `q = 10`.

**SN vs model units.** Thm 2.2 in the paper writes `PMI ≈ ⟨v̂_w, v̂_w'⟩/d`
in *model units* where `‖v̂‖ = O(√d)`. The SN objective fits a rescaled
`v_SN = v̂/√(2d)` (so the `1/(2d)` factor of Thm 2.2's pair form is absorbed
into the squared-norm term), and substituting gives `PMI ≈ 2⟨v_SN, v_SN'⟩`.
In practice — because the single-word law of Thm 2.2 is not perfectly
satisfied by the SN solution — the factor of 2 is partly washed out and the
empirical slope sits closer to 1, matching the paper's heuristic eq. 1.1.
We follow eq. 1.1 here: the *predicted* relation is `⟨v_w, v_w'⟩ ≈ PMI(w, w')`.
"""
function pmi_scatter(emb::Embeddings{T}, X::SparseMatrixCSC{<:AbstractFloat, <:Integer};
                     n_samples::Integer = 50_000,
                     x_floor::Real = 10,
                     rng::AbstractRNG = MersenneTwister(0)) where T<:AbstractFloat
    row_sums = vec(sum(X; dims = 1))
    total = sum(row_sums)
    log_total = log(total)

    rowI, colJ, vals = _upper_triangle_pairs(X)
    keep = findall(>=(x_floor), vals)
    shuffle!(rng, keep)
    n_take = min(n_samples, length(keep))
    resize!(keep, n_take)

    pmi_emp  = Vector{Float64}(undef, n_take)
    pmi_pred = Vector{Float64}(undef, n_take)
    V = emb.V

    @inbounds for (i, idx) in enumerate(keep)
        w  = Int(rowI[idx])
        w′ = Int(colJ[idx])
        xij = Float64(vals[idx])
        pmi_emp[i]  = log(xij) + log_total - log(row_sums[w]) - log(row_sums[w′])
        pmi_pred[i] = dot(@view(V[:, w]), @view(V[:, w′]))
    end

    return (pmi_emp = pmi_emp, pmi_pred = pmi_pred,
            correlation = cor(pmi_emp, pmi_pred))
end

# ─── D.6  RELATIONS=LINES (paper §5.3, optional) ───────────────────────────

"""
    relations_lines(emb, analogies; lowercase=true)

For each relation in `analogies`, build `M = [v_a − v_b for (a, b) in pairs]`,
take its top-2 singular vectors, and report the mean and std of the (signed)
cosines of each column of `M` against each singular vector. The first
singular vector is sign-oriented so its mean cosine is non-negative.
"""
function relations_lines(emb::Embeddings{T}, analogies::Vector{Analogy};
                         lowercase::Bool = true,
                         min_pairs::Integer = 5) where T<:AbstractFloat
    V = emb.V
    d = size(V, 1)
    word2id = emb.vocab.word2id
    lk = lowercase ? (w -> get(word2id, Base.lowercase(w), Int32(0))) :
                     (w -> get(word2id, w, Int32(0)))

    pairs_by_rel = Dict{String, Vector{Tuple{Int32, Int32}}}()
    for q in analogies
        ai = lk(q.a); bi = lk(q.b)
        (ai == 0 || bi == 0) && continue
        push!(get!(Vector{Tuple{Int32, Int32}}, pairs_by_rel, q.relation), (ai, bi))
    end

    result = Dict{String, NamedTuple}()
    for (rel, pairs) in pairs_by_rel
        length(pairs) < min_pairs && continue
        M = Matrix{T}(undef, d, length(pairs))
        @inbounds for (i, (ai, bi)) in enumerate(pairs)
            @views M[:, i] .= V[:, ai] .- V[:, bi]
        end

        F = svd(M)
        u1 = @view F.U[:, 1]
        u2 = size(F.U, 2) >= 2 ? (@view F.U[:, 2]) : zeros(T, d)

        cos1 = Vector{Float64}(undef, length(pairs))
        cos2 = Vector{Float64}(undef, length(pairs))
        @inbounds for j in 1:length(pairs)
            v = @view M[:, j]
            nv = norm(v)
            cos1[j] = dot(v, u1) / nv
            cos2[j] = dot(v, u2) / nv
        end
        if mean(cos1) < 0
            cos1 .= .-cos1     # convention: orient u₁ so the average sign is +
        end

        result[rel] = (mean1 = mean(cos1), std1 = std(cos1),
                       mean2 = mean(cos2), std2 = std(cos2),
                       n_pairs = length(pairs))
    end
    return result
end
