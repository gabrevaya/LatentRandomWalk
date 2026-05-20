using LatentRandomWalk
using Test

@testset "LatentRandomWalk" begin
    include("test_corpus.jl")
    include("test_cooccurrence.jl")
    include("test_model.jl")
    include("test_verify.jl")
    include("test_analogies.jl")
end
