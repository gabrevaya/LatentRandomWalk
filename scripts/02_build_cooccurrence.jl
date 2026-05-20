# Phase B — Build the sliding-window co-occurrence sparse matrix.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave, jldopen
using SparseArrays: nnz

const VOCAB_PATH   = joinpath(PROC_DIR, "vocab.jld2")
const TOKENS_PATH  = joinpath(PROC_DIR, "tokens.jld2")
const COOC_PATH    = joinpath(PROC_DIR, "cooccurrence.jld2")

const WINDOW = 10

vocab  = jldopen(f -> f["vocab"],  VOCAB_PATH,  "r")
tokens = jldopen(f -> f["tokens"], TOKENS_PATH, "r")
n = length(vocab)
@info "Loaded vocab ($n words) and tokens ($(length(tokens)) ids)"

@info "Building co-occurrence (window=$WINDOW, threads=$(Threads.nthreads()))…"
X = @phase "build_cooccurrence" build_cooccurrence(tokens, n; window = WINDOW)
@info "X: $(size(X, 1))×$(size(X, 2))  nnz = $(nnz(X))   symmetric = $(X == X')"

jldsave(COOC_PATH; X, window = WINDOW)
@info "Wrote $COOC_PATH"
