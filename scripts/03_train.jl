# Phase C — Fit SN word vectors with hand-coded AdaGrad.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave, jldopen
using SparseArrays: nnz

const VOCAB_PATH   = joinpath(PROC_DIR, "vocab.jld2")
const COOC_PATH    = joinpath(PROC_DIR, "cooccurrence.jld2")
const VECTORS_PATH = joinpath(RESULTS_DIR, "vectors.jld2")
const LOSSES_PATH  = joinpath(RESULTS_DIR, "losses.jld2")

# CLI knobs: defaults match the paper. Override with e.g.
#   DIM=100 EPOCHS=20 julia --project scripts/03_train.jl
const DIM    = parse(Int,     get(ENV, "DIM",    "300"))
const EPOCHS = parse(Int,     get(ENV, "EPOCHS", "100"))
const LR     = parse(Float32, get(ENV, "LR",     "0.05"))
const XMAX   = parse(Float32, get(ENV, "XMAX",   "100"))
const SEED   = parse(Int,     get(ENV, "SEED",   "0"))

vocab = jldopen(f -> f["vocab"], VOCAB_PATH, "r")
X     = jldopen(f -> f["X"],     COOC_PATH,  "r")
@info "Loaded vocab ($(length(vocab)) words)  and X (nnz = $(nnz(X)))"

config = TrainingConfig(; d = DIM, epochs = EPOCHS, lr = LR, X_max = XMAX, seed = SEED)
@info "Training: $config"

emb, losses = @phase "train" train(X, vocab, config)
@info "Final weighted MSE = $(losses[end])"

jldsave(VECTORS_PATH; embeddings = emb, config)
jldsave(LOSSES_PATH;  losses)
@info "Wrote $VECTORS_PATH and $LOSSES_PATH"
