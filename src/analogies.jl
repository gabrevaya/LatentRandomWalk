using LinearAlgebra: mul!, norm
using Printf: @printf

"""
    Analogy(relation, a, b, c, d)

A single analogy from the Google or MSR testbed: `a : b :: c : d`.
"""
struct Analogy
    relation::String
    a::String
    b::String
    c::String
    d::String
end

"""
    load_analogies(path) -> Vector{Analogy}

Read the Google / MSR analogy file format: lines beginning with `:` start a
new relation, all other non-blank lines are four whitespace-separated words.
"""
function load_analogies(path::AbstractString)
    out = Analogy[]
    rel = ""
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, ":")
            rel = strip(s[2:end])
        else
            parts = split(s)
            length(parts) == 4 || continue
            push!(out, Analogy(rel, parts[1], parts[2], parts[3], parts[4]))
        end
    end
    return out
end

"""
    AnalogyResult

Outcome of a testbed evaluation. Fields:
  * `overall`     — accuracy on eligible questions
  * `by_relation` — `relation => (correct, total)` for each relation
  * `n_eligible`  — questions where all four words are in the vocabulary
  * `n_total`     — questions in the input file (so we can report coverage)
"""
struct AnalogyResult
    overall::Float64
    by_relation::Dict{String, Tuple{Int, Int}}
    n_eligible::Int
    n_total::Int
end

function Base.show(io::IO, r::AnalogyResult)
    @printf(io, "AnalogyResult(overall=%.4f, eligible=%d/%d)",
            r.overall, r.n_eligible, r.n_total)
end

# ─────────────────────────── evaluation ────────────────────────────────────

"""
    evaluate_analogies(emb, analogies; batch_size=1024, lowercase=true) -> AnalogyResult

Solve `a : b :: c : ?` by `argmax_w cos(v_w, v_b − v_a + v_c)` with the search
restricted to words distinct from `a, b, c`. All cosine similarities are
computed by batched GEMM against `V / ‖V‖` (column-normalised), so the cost
is dominated by BLAS, not by Julia loops.
"""
function evaluate_analogies(emb::Embeddings{T}, analogies::Vector{Analogy};
                            batch_size::Integer = 1024,
                            lowercase::Bool = true) where T<:AbstractFloat
    V = emb.V
    d, n_words = size(V)
    word2id = emb.vocab.word2id

    # Column-normalised V (used for cosine via dot product)
    col_norms = vec(sqrt.(sum(abs2, V; dims = 1)))
    V_normed = V ./ reshape(col_norms, 1, :)
    V_normed_t = transpose(V_normed)  # n × d (lazy transpose)

    # Filter to questions whose four words are all in the vocabulary
    lookup = if lowercase
        w -> get(word2id, Base.lowercase(w), Int32(0))
    else
        w -> get(word2id, w, Int32(0))
    end

    ids = NTuple{4, Int32}[]
    rels = String[]
    for q in analogies
        ai = lookup(q.a); bi = lookup(q.b); ci = lookup(q.c); di = lookup(q.d)
        (ai == 0 || bi == 0 || ci == 0 || di == 0) && continue
        push!(ids, (ai, bi, ci, di))
        push!(rels, q.relation)
    end
    n_elig = length(ids)
    n_elig == 0 && return AnalogyResult(0.0, Dict{String,Tuple{Int,Int}}(), 0, length(analogies))

    Q      = Matrix{T}(undef, d, batch_size)
    scores = Matrix{T}(undef, n_words, batch_size)
    by_rel = Dict{String, Tuple{Int, Int}}()
    n_correct = 0

    for bstart in 1:batch_size:n_elig
        bend = min(bstart + batch_size - 1, n_elig)
        bsz  = bend - bstart + 1
        Qb   = @view Q[:, 1:bsz]
        Sb   = @view scores[:, 1:bsz]

        # query[j] = (v_b - v_a + v_c) / ‖·‖
        @inbounds for (j, qi) in enumerate(bstart:bend)
            ai, bi, ci, _ = ids[qi]
            @views Qb[:, j] .= V[:, bi] .- V[:, ai] .+ V[:, ci]
            qn = norm(@view Qb[:, j])
            @views Qb[:, j] ./= qn
        end

        # scores = V_normedᵀ * Qb  (n_words × bsz). One GEMM, no allocation.
        mul!(Sb, V_normed_t, Qb)

        @inbounds for (j, qi) in enumerate(bstart:bend)
            ai, bi, ci, di_true = ids[qi]
            best_id = 0
            best_score = T(-Inf)
            for k in 1:n_words
                (k == ai || k == bi || k == ci) && continue
                sk = Sb[k, j]
                if sk > best_score
                    best_score = sk
                    best_id = k
                end
            end
            ok = best_id == di_true
            n_correct += ok
            rel = rels[qi]
            (c, t) = get(by_rel, rel, (0, 0))
            by_rel[rel] = (c + ok, t + 1)
        end
    end

    return AnalogyResult(n_correct / n_elig, by_rel, n_elig, length(analogies))
end
