using LatentRandomWalk
using SparseArrays: nnz, SparseMatrixCSC, sparse
using Test

@testset "cooccurrence" begin
    @testset "hand-checked tiny example, window=2" begin
        # Tokens 1,2,3,4,5,1 with window=2 produces nine unique unordered pairs
        # (each with count 1):
        #   (1,2) (1,3)        from i=1
        #   (2,3) (2,4)        from i=2
        #   (3,4) (3,5)        from i=3
        #   (4,5) (4,1)        from i=4
        #         (5,1)        from i=5
        tokens = Int32[1, 2, 3, 4, 5, 1]
        X = build_cooccurrence(tokens, 5; window = 2, ntasks = 1)
        @test X == X'
        @test nnz(X) == 18                          # 9 unordered pairs × 2
        @test X[1, 1] == 0                          # diagonal empty
        @test X[1, 2] == 1
        @test X[1, 3] == 1
        @test X[1, 4] == 1
        @test X[1, 5] == 1
        @test X[2, 4] == 1
        @test X[3, 5] == 1
    end

    @testset "self-pairs are skipped" begin
        # [1,1,2,2] with window=2: the only non-self pair is (1,2),
        # counted from (i=1, j=3), (i=2, j=3), (i=2, j=4) → 3.
        tokens = Int32[1, 1, 2, 2]
        X = build_cooccurrence(tokens, 2; window = 2, ntasks = 1)
        @test X[1, 1] == 0
        @test X[2, 2] == 0
        @test X[1, 2] == 3
        @test X[1, 2] == X[2, 1]
    end

    @testset "threaded build matches single-threaded" begin
        # Random token stream
        n = 500
        vocab_n = 30
        tokens = Int32.(rand(1:vocab_n, n))
        X1 = build_cooccurrence(tokens, vocab_n; window = 5, ntasks = 1)
        X4 = build_cooccurrence(tokens, vocab_n; window = 5, ntasks = 4)
        @test X1 == X4
    end

    @testset "symmetry on a larger random stream" begin
        tokens = Int32.(rand(1:100, 5_000))
        X = build_cooccurrence(tokens, 100; window = 8)
        @test X == X'
        @test all(X[i, i] == 0 for i in axes(X, 1))
    end
end
