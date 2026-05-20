using LatentRandomWalk
using Test

@testset "corpus" begin
    corpus = ["the", "cat", "sat", "on", "the", "mat",
              "the", "dog", "ran", "the", "cat", "ran",
              "the", "dog", "sat"]

    vocab = build_vocabulary(corpus; min_count = 2)

    @testset "vocabulary structure" begin
        @test length(vocab) == 5                       # the, cat, sat, the, dog, ran — "mat" "on" drop out
        @test vocab.id2word[1] == "the"                # most-frequent word gets id 1
        @test sum(vocab.counts) == 5 + 2 + 2 + 2 + 2   # 13 kept tokens
        @test all(w -> vocab.word2id[w] isa Int32, vocab.id2word)
    end

    @testset "tokenisation drops OOV" begin
        toks = tokenize(corpus, vocab)
        @test eltype(toks) == Int32
        @test length(toks) == 13                       # "on" and "mat" are dropped
        @test toks[1] == vocab.word2id["the"]
    end

    @testset "min_count threshold respected" begin
        v2 = build_vocabulary(corpus; min_count = 3)
        @test all(c -> c >= 3, v2.counts)
        @test "ran" ∉ v2.id2word                       # appears twice; below threshold of 3
    end

    @testset "max_size truncates by frequency" begin
        v3 = build_vocabulary(corpus; min_count = 1, max_size = 2)
        @test length(v3) == 2
        @test v3.id2word[1] == "the"                   # unique maximum at count 5
        @test v3.counts[1] == 5
        @test v3.counts[2] == 2                        # whichever count-2 word
        @test v3.id2word[2] ∈ ("cat", "sat", "dog", "ran")
    end
end
