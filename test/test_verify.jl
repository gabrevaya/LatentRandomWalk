using LatentRandomWalk
using SparseArrays: sparse, SparseMatrixCSC
using LinearAlgebra: norm
using Random: MersenneTwister
using Test

# Reuse the toy fixtures from test_model.jl
isdefined(@__MODULE__, :_toy_vocab) || include("test_model.jl")

@testset "verify" begin
    n = 30
    X = _toy_X(n; seed = 7, density = 0.8, max_count = 30)
    vocab = _toy_vocab(n)
    emb, _ = train(X, vocab; d = 16, epochs = 80, log_every = 1000, seed = 0)

    @testset "partition_function returns positive concentrating samples" begin
        pf = partition_function(emb; n_samples = 200, rng = MersenneTwister(0))
        @test length(pf.Z) == 200
        @test all(>(0), pf.Z)
        @test pf.mean_Z > 0
        # Each sample should be within an order of magnitude of the mean — very loose.
        @test 0.1 < minimum(pf.normalised) < maximum(pf.normalised) < 10
    end

    @testset "norm_vs_frequency returns finite numbers" begin
        nf = norm_vs_frequency(emb)
        @test length(nf.sq_norms) == n
        @test all(isfinite, nf.sq_norms)
        @test all(isfinite, nf.log_freqs)
        @test isfinite(nf.correlation)
    end

    @testset "singular_value_isotropy returns sorted positive σ's" begin
        sv = singular_value_isotropy(emb)
        @test issorted(sv.singular_values; rev = true)
        @test sv.rms > 0
        @test sv.min_nonzero > 0
        @test 0 < sv.ratio <= 1.0
    end

    @testset "pmi_scatter returns matched-length vectors" begin
        pmi = pmi_scatter(emb, X; n_samples = 50, x_floor = 1, rng = MersenneTwister(0))
        @test length(pmi.pmi_emp) == length(pmi.pmi_pred)
        @test all(isfinite, pmi.pmi_emp)
        @test all(isfinite, pmi.pmi_pred)
    end
end
