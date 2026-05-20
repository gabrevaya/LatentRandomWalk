# Phase D — Empirical verification of Theorems 2.2, 4.1 + the PMI prediction.
# Produces JLD2 records and one Makie figure per diagnostic.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave, jldopen
using CairoMakie
using Statistics: mean
using Random: MersenneTwister

const VECTORS_PATH = joinpath(RESULTS_DIR, "vectors.jld2")
const COOC_PATH    = joinpath(PROC_DIR,    "cooccurrence.jld2")
const ANALOGY_PATH = joinpath(RAW_DIR,     "questions-words.txt")
const VERIFY_PATH  = joinpath(RESULTS_DIR, "verification.jld2")

emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
X   = jldopen(f -> f["X"],          COOC_PATH,    "r")
d   = dim(emb)
@info "Loaded embeddings: d=$d, n=$(length(emb))"

# ─── D.1 Partition-function concentration ──────────────────────────────────
@info "D.1  Partition function (Lemma 2.1)…"
pf = @phase "D.1 partition_function" partition_function(emb; rng = MersenneTwister(0))
@info "      mean Z = $(pf.mean_Z)"

fig1 = Figure(size = (640, 380))
ax1  = Axis(fig1[1, 1], title = "D.1  Partition function Z_c (norm = Z_c / 𝔼 Z_c)",
            xlabel = "Z_c / 𝔼[Z_c]", ylabel = "count")
hist!(ax1, Float64.(pf.normalised); bins = 60, strokecolor = :black, strokewidth = 0.5)
vlines!(ax1, [0.9, 1.1]; color = :red, linestyle = :dash)
save(joinpath(FIGURES_DIR, "D1_partition_function.pdf"), fig1)

# ─── D.2 Squared norm vs log frequency ─────────────────────────────────────
@info "D.2  ‖v_w‖² vs log p(w) (Thm 2.2 single-word case)…"
nf = @phase "D.2 norm_vs_frequency" norm_vs_frequency(emb)
@info "      Pearson r = $(round(nf.correlation; digits=3))"

fig2 = Figure(size = (640, 480))
ax2  = Axis(fig2[1, 1], title = "D.2  ‖v_w‖² vs log p(w)   (r=$(round(nf.correlation; digits=3)))",
            xlabel = "log p(w)", ylabel = "‖v_w‖²")
scatter!(ax2, nf.log_freqs, nf.sq_norms; markersize = 3, color = (:steelblue, 0.5))
save(joinpath(FIGURES_DIR, "D2_norm_vs_frequency.pdf"), fig2)

# ─── D.3 Singular-value isotropy ───────────────────────────────────────────
@info "D.3  Singular values (Thm 4.1)…"
sv = @phase "D.3 singular_value_isotropy" singular_value_isotropy(emb)
@info "      min σ / rms σ = $(round(sv.ratio; digits=3))   (predicted ~1/3)"

fig3 = Figure(size = (640, 380))
ax3  = Axis(fig3[1, 1], title = "D.3  Singular values of V   (min/rms = $(round(sv.ratio; digits=3)))",
            xlabel = "index", ylabel = "σ")
lines!(ax3, 1:length(sv.singular_values), Float64.(sv.singular_values))
hlines!(ax3, [Float64(sv.rms)]; color = :red, linestyle = :dash, label = "RMS")
axislegend(ax3)
save(joinpath(FIGURES_DIR, "D3_singular_values.pdf"), fig3)

# ─── D.4 PMI ≈ ⟨v_w, v_w'⟩ / d ─────────────────────────────────────────────
@info "D.4  PMI ≈ ⟨v_w, v_w'⟩ / d (Corollary 2.3)…"
pmi = @phase "D.4 pmi_scatter" pmi_scatter(emb, X; rng = MersenneTwister(1))
@info "      Pearson r = $(round(pmi.correlation; digits=3))"

fig4 = Figure(size = (640, 480))
ax4  = Axis(fig4[1, 1], title = "D.4  Empirical PMI vs predicted ⟨v, v'⟩/d   (r=$(round(pmi.correlation; digits=3)))",
            xlabel = "predicted  ⟨v_w, v_w'⟩ / d", ylabel = "empirical  PMI(w, w')")
scatter!(ax4, pmi.pmi_pred, pmi.pmi_emp; markersize = 3, color = (:darkorange, 0.3))
let (m, c) = let X̄ = mean(pmi.pmi_pred), Ȳ = mean(pmi.pmi_emp)
        slope = sum((pmi.pmi_pred .- X̄) .* (pmi.pmi_emp .- Ȳ)) /
                sum((pmi.pmi_pred .- X̄) .^ 2)
        slope, Ȳ - slope * X̄
    end
    xs = extrema(pmi.pmi_pred)
    lines!(ax4, collect(xs), m .* collect(xs) .+ c; color = :red, linewidth = 2,
           label = "slope=$(round(m; digits=2))  intercept=$(round(c; digits=2))")
    axislegend(ax4; position = :lt)
end
save(joinpath(FIGURES_DIR, "D4_pmi_scatter.pdf"), fig4)

# ─── D.6 (optional) RELATIONS=LINES ────────────────────────────────────────
rl_result = nothing
if isfile(ANALOGY_PATH)
    @info "D.6  RELATIONS=LINES…"
    analogies = load_analogies(ANALOGY_PATH)
    rl_result = @phase "D.6 relations_lines" relations_lines(emb, analogies)

    rels   = sort(collect(keys(rl_result)))
    mean1s = [rl_result[r].mean1 for r in rels]
    mean2s = [rl_result[r].mean2 for r in rels]

    fig6 = Figure(size = (800, 0.36 * length(rels) * 18 + 200))
    ax6  = Axis(fig6[1, 1], title = "D.6  Cosines of v_a − v_b with top-2 singular vectors per relation",
                xlabel = "cosine", yticks = (1:length(rels), rels))
    barplot!(ax6, 1:length(rels), mean1s; direction = :x, color = :steelblue, label = "u₁")
    barplot!(ax6, (1:length(rels)) .+ 0.35, mean2s; direction = :x, color = :darkorange, label = "u₂")
    axislegend(ax6; position = :rb)
    save(joinpath(FIGURES_DIR, "D6_relations_lines.pdf"), fig6)
else
    @warn "Skipping D.6: $ANALOGY_PATH not found. Run scripts/00_download_corpus.jl for the analogy file."
end

jldsave(VERIFY_PATH;
        partition_function = pf,
        norm_vs_frequency  = nf,
        singular_values    = sv,
        pmi_scatter        = pmi,
        relations_lines    = rl_result)
@info "Wrote $VERIFY_PATH"
@info "Figures in $FIGURES_DIR"
