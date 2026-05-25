struct Vocabulary
    word2id::Dict{String, Int32}
    id2word::Vector{String}
    counts::Vector{Int}
end

Base.length(v::Vocabulary) = length(v.id2word)
Base.show(io::IO, v::Vocabulary) = print(io, "Vocabulary(", length(v), " words, total count ", sum(v.counts), ")")

word_id(v::Vocabulary, w::AbstractString) = get(v.word2id, w, Int32(0))

"""
    read_text8(path) -> String

Read the text8 corpus into a single String. text8 is one giant line of
lowercase ASCII tokens separated by single spaces; this is what makes the
single-`read` approach safe (no encoding surprises) and fast.
"""
read_text8(path::AbstractString) = read(path, String)

"""
    text8_tokens(text) -> iterator of SubString

Lazily split a text8 string on spaces. Tokens are `SubString`s that view into
the parent `text`, so iteration allocates nothing per token.
"""
text8_tokens(text::AbstractString) = eachsplit(text, ' '; keepempty = false)

"""
    build_vocabulary(tokens; min_count=5, max_size=typemax(Int)) -> Vocabulary

Single pass over `tokens` (any iterable of `AbstractString`). Keeps words whose
count is at least `min_count`, then sorts by frequency descending so that low
IDs correspond to common words. IDs are 1-based `Int32`.
"""
function build_vocabulary(tokens; min_count::Integer = 5, max_size::Integer = typemax(Int))
    raw = Dict{String, Int}()
    sizehint!(raw, 1 << 19)
    for tok in tokens
        s = String(tok)
        raw[s] = get(raw, s, 0) + 1
    end

    kept = [(w, c) for (w, c) in raw if c >= min_count]
    sort!(kept; by = last, rev = true)
    if length(kept) > max_size
        resize!(kept, max_size)
    end

    id2word = String[w for (w, _) in kept]
    counts  = Int[c for (_, c) in kept]
    word2id = Dict{String, Int32}()
    sizehint!(word2id, length(id2word))
    @inbounds for (i, w) in enumerate(id2word)
        word2id[w] = Int32(i)
    end

    return Vocabulary(word2id, id2word, counts)
end

"""
    tokenize(tokens, vocab) -> Vector{Int32}

Convert an iterable of string tokens to a dense `Vector{Int32}` of vocabulary
IDs. Tokens absent from `vocab` are dropped silently (text8 has none after
frequency filtering; the option matters for other corpora).
"""
function tokenize(tokens, vocab::Vocabulary)
    out = Int32[]
    word2id = vocab.word2id
    for tok in tokens
        id = get(word2id, tok, Int32(0))
        id != 0 && push!(out, id)
    end
    return out
end
