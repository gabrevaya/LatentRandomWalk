# Phase D.5 — Google analogy testbed evaluation.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave, jldopen
using CairoMakie
using Printf: @sprintf

const VECTORS_PATH  = joinpath(RESULTS_DIR, "vectors.jld2")
const ANALOGY_PATH  = joinpath(RAW_DIR,     "questions-words.txt")
const ANALOGIES_OUT = joinpath(RESULTS_DIR, "analogies.jld2")

isfile(ANALOGY_PATH) || error("Analogy testbed not found at $ANALOGY_PATH. Run scripts/00_download_corpus.jl.")

emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
@info "Loaded embeddings: d=$(dim(emb)), n=$(length(emb))"

analogies = load_analogies(ANALOGY_PATH)
@info "Loaded $(length(analogies)) analogy questions across $(length(unique(getfield.(analogies, :relation)))) relations."

result = @phase "evaluate_analogies" evaluate_analogies(emb, analogies)
@info @sprintf("Overall: %.4f   eligible: %d/%d", result.overall, result.n_eligible, result.n_total)

@info "Per-relation accuracy:"
for (rel, (c, t)) in sort(collect(result.by_relation))
    @info @sprintf("  %-30s  %.4f   (%d / %d)", rel, c / t, c, t)
end

jldsave(ANALOGIES_OUT; result)
@info "Wrote $ANALOGIES_OUT"

# Plot per-relation accuracy
rels = sort(collect(keys(result.by_relation)))
accs = [result.by_relation[r][1] / result.by_relation[r][2] for r in rels]

fig = Figure(size = (760, 0.36 * length(rels) * 18 + 240))
ax  = Axis(fig[1, 1],
           title = @sprintf("Analogy accuracy by relation  (overall = %.3f, eligible = %d / %d)",
                            result.overall, result.n_eligible, result.n_total),
           xlabel = "accuracy", yticks = (1:length(rels), rels), xticks = 0:0.1:1.0)
barplot!(ax, 1:length(rels), accs; direction = :x, color = :seagreen)
vlines!(ax, [result.overall]; color = :red, linestyle = :dash, label = "overall")
axislegend(ax; position = :rt)
save(joinpath(FIGURES_DIR, "D5_analogy_accuracy.pdf"), fig)
@info "Wrote figures/D5_analogy_accuracy.pdf"
