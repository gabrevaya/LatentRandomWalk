# End-to-end pipeline driver. Each phase script is skipped when its primary
# checkpoint file already exists. Override with FORCE=1.
include(joinpath(@__DIR__, "_setup.jl"))

const FORCE = get(ENV, "FORCE", "") == "1"

# Each row: (script path, output file produced by that script)
phases = [
    ("00_download_corpus.jl",   joinpath(RAW_DIR,     "text8")),
    ("01_build_vocab.jl",       joinpath(PROC_DIR,    "vocab.jld2")),
    ("02_build_cooccurrence.jl",joinpath(PROC_DIR,    "cooccurrence.jld2")),
    ("03_train.jl",             joinpath(RESULTS_DIR, "vectors.jld2")),
    ("04_verify.jl",            joinpath(RESULTS_DIR, "verification.jld2")),
    ("05_analogies.jl",         joinpath(RESULTS_DIR, "analogies.jld2")),
    ("06_sde_demo.jl",          joinpath(RESULTS_DIR, "sde.jld2")),
]

for (script, output) in phases
    if !FORCE && isfile(output)
        @info "✓ $script  (cached → $output)"
        continue
    end
    @info "▶ $script"
    include(joinpath(@__DIR__, script))
end
@info "Pipeline complete. Inspect figures/ and data/results/."
