using OhMyThreads: tmapreduce, chunks
using SparseArrays: SparseMatrixCSC, sparse

# `Int` value type so the counter doesn't risk silent overflow on corpora
# larger than text8 (Int32 saturates at ~2.1e9). `_expand_symmetric` still
# materialises a `Float32` sparse matrix, so the public output is unchanged.
const PairDict = Dict{Tuple{Int32, Int32}, Int}

# Counts unordered pairs (i, j) with i < j whose token-stream positions lie in
# `range × (range + window)`. Each unordered pair is therefore counted exactly
# once across disjoint ranges.
function _count_in_chunk(tokens::Vector{Int32}, range, window::Integer, N::Integer)
    counts = PairDict()
    sizehint!(counts, max(1024, length(range) ÷ 2))
    @inbounds for i in range
        wi = tokens[i]
        jmax = min(i + window, N)
        for j in (i + 1):jmax
            wj = tokens[j]
            wi == wj && continue
            key = wi < wj ? (wi, wj) : (wj, wi)
            counts[key] = get(counts, key, 0) + 1
        end
    end
    return counts
end

# Convert an upper-triangle dict into a symmetric SparseMatrixCSC by emitting
# both (i, j) and (j, i) triples. `sparse(I, J, V, n, n, +)` does the assembly.
function _expand_symmetric(counts::PairDict, n::Integer)
    m = length(counts)
    I = Vector{Int32}(undef, 2m)
    J = Vector{Int32}(undef, 2m)
    V = Vector{Float32}(undef, 2m)
    k = 1
    @inbounds for ((i, j), c) in counts
        cf = Float32(c)
        I[k] = i; J[k] = j; V[k] = cf; k += 1
        I[k] = j; J[k] = i; V[k] = cf; k += 1
    end
    return sparse(I, J, V, Int(n), Int(n), +)
end

"""
    build_cooccurrence(tokens, n; window=10, ntasks=Threads.nthreads())
        -> SparseMatrixCSC{Float32, Int32}

Build the symmetric word-pair co-occurrence matrix `X` from a tokenised
corpus. `X[w, w']` is the number of times words `w` and `w'` appear within
`±window` of each other.

The pipeline partitions the token stream with `OhMyThreads.chunks` so each
chunk owns its *left* positions; right positions are allowed to spill into the
following chunk (read-only). Per-chunk dictionaries of unordered-pair counts
are merged with `mergewith!(+, ...)` via `tmapreduce`, and the result is
expanded to a symmetric `SparseMatrixCSC{Float32, Int32}`.
"""
function build_cooccurrence(tokens::Vector{Int32}, n::Integer;
                            window::Integer = 10,
                            ntasks::Integer = Threads.nthreads())
    N = length(tokens)
    @assert N > 0 "Empty corpus"
    @assert window >= 1 "window must be ≥ 1"

    merged = if ntasks <= 1
        _count_in_chunk(tokens, 1:N, window, N)
    else
        tmapreduce(c -> _count_in_chunk(tokens, c, window, N),
                   (a, b) -> mergewith!(+, a, b),
                   chunks(1:N; n = Int(ntasks)))
    end

    return _expand_symmetric(merged, n)
end
