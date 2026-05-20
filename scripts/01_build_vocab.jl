# Phase A — Build Vocabulary and token-ID stream from text8.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave

const TEXT8_PATH  = joinpath(RAW_DIR, "text8")
const VOCAB_PATH  = joinpath(PROC_DIR, "vocab.jld2")
const TOKENS_PATH = joinpath(PROC_DIR, "tokens.jld2")

const MIN_COUNT = 5
const MAX_VOCAB = 50_000

isfile(TEXT8_PATH) || error("text8 not found at $TEXT8_PATH. Run scripts/00_download_corpus.jl first.")

@info "Reading text8 ($(round(filesize(TEXT8_PATH) / 2^20; digits=1)) MiB)…"
text = @phase "read_text8" read_text8(TEXT8_PATH)

@info "Building vocabulary (min_count=$MIN_COUNT, max_size=$MAX_VOCAB)…"
vocab = @phase "build_vocabulary" build_vocabulary(text8_tokens(text);
                                                   min_count = MIN_COUNT,
                                                   max_size = MAX_VOCAB)
@info "Vocab size: $(length(vocab))   total kept count: $(sum(vocab.counts))"

@info "Tokenising corpus…"
tokens = @phase "tokenize" tokenize(text8_tokens(text), vocab)
@info "Token stream length: $(length(tokens))"

jldsave(VOCAB_PATH;  vocab)
jldsave(TOKENS_PATH; tokens)
@info "Wrote $VOCAB_PATH and $TOKENS_PATH"
