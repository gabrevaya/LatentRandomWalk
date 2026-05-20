using LatentRandomWalk
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: dot
using Random: MersenneTwister
using Test
using Zygote

# Build a tiny symmetric co-occurrence matrix with no self-pairs.
function _toy_X(n::Int; seed::Int = 0, density::Float64 = 0.5, max_count::Int = 12)
    rng = MersenneTwister(seed)
    I = Int32[]; J = Int32[]; V = Float32[]
    for i in 1:n, j in (i+1):n
        if rand(rng) < density
            c = Float32(rand(rng, 1:max_count))
            push!(I, i); push!(J, j); push!(V, c)
            push!(I, j); push!(J, i); push!(V, c)
        end
    end
    return sparse(I, J, V, n, n, +)
end

# Tiny dummy vocabulary so `train` can build an `Embeddings` from it.
function _toy_vocab(n::Int)
    id2word = [string("w", i) for i in 1:n]
    word2id = Dict{String,Int32}(w => Int32(i) for (i, w) in enumerate(id2word))
    counts  = ones(Int, n)
    return Vocabulary(word2id, id2word, counts)
end

@testset "model" begin
    @testset "sn gradient matches Zygote autodiff" begin
        d, n = 4, 6
        rng = MersenneTwister(42)
        V = (rand(rng, Float32, d, n) .- 0.5f0) ./ Float32(d)
        C = log(100f0)
        X = _toy_X(n; seed = 1)
        Xmax = 100f0

        # Precompute the upper-triangle pairs outside the differentiable closure
        # so Zygote doesn't have to navigate the `resize!` in `_upper_triangle_pairs`.
        rowI, colJ, vals = LatentRandomWalk._upper_triangle_pairs(X)

        loss_fn = (V_, C_) -> begin
            L = zero(eltype(V_))
            for idx in eachindex(rowI)
                w, wp = Int(rowI[idx]), Int(colJ[idx])
                Xww = vals[idx]
                vsum = V_[:, w] .+ V_[:, wp]
                r = log(Xww) - sum(vsum .^ 2) - C_
                L += min(Xww, Xmax) * r * r
            end
            L
        end

        # The value of this hand loss must match `sn_loss` exactly (sanity)
        @test loss_fn(V, C) ≈ sn_loss(V, C, X; X_max = Xmax) rtol = 1e-5

        gV_zyg, gC_zyg = Zygote.gradient(loss_fn, V, C)

        # Hand-derived gradient: sum of −4·X̄·r·(v_w + v_w') over pairs touching w
        gV = zeros(Float32, d, n)
        gC = 0f0
        for idx in eachindex(rowI)
            w, wp = Int(rowI[idx]), Int(colJ[idx])
            Xval = vals[idx]
            X̄ = min(Xval, Xmax)
            vsum = V[:, w] .+ V[:, wp]
            r = log(Xval) - dot(vsum, vsum) - C
            coeff = -4f0 * X̄ * r
            gV[:, w]  .+= coeff .* vsum
            gV[:, wp] .+= coeff .* vsum
            gC += -2f0 * X̄ * r
        end

        @test gV ≈ gV_zyg  rtol = 1e-4
        @test gC ≈ gC_zyg  rtol = 1e-4
    end

    @testset "loss decreases on a small problem" begin
        n = 12
        X = _toy_X(n; seed = 2, density = 0.7, max_count = 25)
        vocab = _toy_vocab(n)
        _, losses = train(X, vocab; d = 16, epochs = 40, log_every = 1000, seed = 0)
        @test losses[end] < losses[1]
        @test losses[end] < 0.5 * losses[1]                # solid descent
        @test all(diff(losses[1:5]) .<= 1e-3)              # monotonic-ish early on
    end

    @testset "type-stable Float32 throughout" begin
        n = 6
        X = _toy_X(n; seed = 3)
        vocab = _toy_vocab(n)
        emb, _ = train(X, vocab; d = 4, epochs = 5, log_every = 1000)
        @test eltype(emb.V) === Float32
        @test typeof(emb.C) === Float32
    end
end
