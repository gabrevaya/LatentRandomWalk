using LatentRandomWalk
using Test

@testset "analogies" begin
    @testset "load_analogies parses relations and quadruples" begin
        path, _ = mktemp()
        write(path, """
        : capital-common-countries
        Athens Greece Baghdad Iraq
        Athens Greece Bangkok Thailand

        : currency
        Algeria dinar Argentina peso
        """)

        as = load_analogies(path)
        @test length(as) == 3
        @test as[1].relation == "capital-common-countries"
        @test as[1].a == "Athens" && as[1].b == "Greece"
        @test as[3].relation == "currency"
        rm(path)
    end

    @testset "evaluate_analogies on a toy vocabulary" begin
        ids = ["a", "b", "c", "d", "e", "f"]
        word2id = Dict{String,Int32}(w => Int32(i) for (i, w) in enumerate(ids))
        vocab = Vocabulary(word2id, ids, ones(Int, length(ids)))

        # Choose columns so v_b − v_a + v_c = v_d, and the other candidates
        # (e, f) are orthogonal to v_d. a is excluded from the candidate set
        # anyway; we give it a tiny nonzero norm to avoid NaNs when columns
        # of V get normalised.
        V = Float32[
            1e-6  1.0  0.0  1.0  0.0  1.0;
            0.0   0.0  1.0  1.0  0.0 -1.0;
            0.0   0.0  0.0  0.0  1.0  0.0;
        ]
        emb = LatentRandomWalk.Embeddings(V, 0f0, vocab)

        analogies = [Analogy("rel", "a", "b", "c", "d")]
        result = evaluate_analogies(emb, analogies; lowercase = false)
        @test result.n_eligible == 1
        @test result.overall == 1.0
    end
end
