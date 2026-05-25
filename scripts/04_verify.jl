# Phase D — Empirical verification of Theorems 2.2, 4.1 + the PMI prediction.
# Produces JLD2 records and one Makie figure per diagnostic.
include(joinpath(@__DIR__, "_setup.jl"))

using LatentRandomWalk
using JLD2: jldsave, jldopen
using CairoMakie
using Statistics: mean, std
using Random: MersenneTwister
using Printf: @sprintf

const VECTORS_PATH = joinpath(RESULTS_DIR, "vectors.jld2")
const COOC_PATH    = joinpath(PROC_DIR,    "cooccurrence.jld2")
const ANALOGY_PATH = joinpath(RAW_DIR,     "questions-words.txt")
const VERIFY_PATH  = joinpath(RESULTS_DIR, "verification.jld2")

# ─── helpers ───────────────────────────────────────────────────────────────

# Ordinary least-squares slope and intercept of y ≈ slope · x + intercept.
function _fit_line(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    x̄, ȳ = mean(x), mean(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    intercept = ȳ - slope * x̄
    return slope, intercept
end

# ─── load artifacts ────────────────────────────────────────────────────────

emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
X   = jldopen(f -> f["X"],          COOC_PATH,    "r")
d   = dim(emb)
@info "Loaded embeddings: d=$d, n=$(length(emb))"

# ─── D.1 Partition-function concentration ──────────────────────────────────

@info "D.1  Partition function (Lemma 2.1)…"
pf = @phase "D.1 partition_function" partition_function(emb; rng = MersenneTwister(0))
σ_over_mean  = std(Float64.(pf.normalised))   # already in units of mean Z
frac_in_band = count(x -> 0.9 ≤ x ≤ 1.1, pf.normalised) / length(pf.normalised)
@info @sprintf("      mean Z = %.4e   σ/𝔼 = %.3f   in [0.9, 1.1]: %.1f%%",
               pf.mean_Z, σ_over_mean, 100 * frac_in_band)

fig1 = Figure(size = (700, 400))
ax1  = Axis(fig1[1, 1],
            title = @sprintf("D.1  Partition-function concentration   σ/𝔼 = %.3f   in [0.9, 1.1]: %.1f%%",
                             σ_over_mean, 100 * frac_in_band),
            xlabel = "Z_c / 𝔼[Z_c]", ylabel = "count")
hist!(ax1, Float64.(pf.normalised); bins = 60, strokecolor = :black, strokewidth = 0.5)
vlines!(ax1, [0.9, 1.1]; color = :red, linestyle = :dash, label = "[0.9, 1.1]")
axislegend(ax1; position = :rt)
save(joinpath(FIGURES_DIR, "D1_partition_function.pdf"), fig1)

# ─── D.2 Squared norm vs log frequency ─────────────────────────────────────

@info "D.2  ‖v_w‖² vs log p(w) (Thm 2.2 single-word case)…"
nf = @phase "D.2 norm_vs_frequency" norm_vs_frequency(emb)
@info "      Pearson r = $(round(nf.correlation; digits=3))"

fig2 = Figure(size = (640, 480))
ax2  = Axis(fig2[1, 1],
            title = @sprintf("D.2  ‖v_w‖² vs log p(w)   (r = %.3f)", nf.correlation),
            xlabel = "log p(w)", ylabel = "‖v_w‖²")
scatter!(ax2, nf.log_freqs, nf.sq_norms; markersize = 3, color = (:steelblue, 0.5))
save(joinpath(FIGURES_DIR, "D2_norm_vs_frequency.pdf"), fig2)

# ─── D.3 Singular-value isotropy ───────────────────────────────────────────

@info "D.3  Singular values (Thm 4.1)…"
sv = @phase "D.3 singular_value_isotropy" singular_value_isotropy(emb)
@info "      min σ / rms σ = $(round(sv.ratio; digits=3))   (predicted ~1/3)"

fig3 = Figure(size = (700, 420))
ax3  = Axis(fig3[1, 1],
            title = @sprintf("D.3  Singular values of V   (min/RMS = %.3f,  predicted ≈ 1/3)", sv.ratio),
            xlabel = "index", ylabel = "σ")
lines!(ax3, 1:length(sv.singular_values), Float64.(sv.singular_values);
       color = :steelblue, label = "σ_k")
hlines!(ax3, [Float64(sv.rms)];     color = :red,     linestyle = :dash, label = "RMS")
hlines!(ax3, [Float64(sv.rms) / 3]; color = :seagreen, linestyle = :dot,  label = "RMS / 3  (predicted min)")
hlines!(ax3, [Float64(sv.min_nonzero)]; color = :black, linestyle = :dashdot, label = "observed min")
axislegend(ax3; position = :rt)
save(joinpath(FIGURES_DIR, "D3_singular_values.pdf"), fig3)

# ─── D.4 PMI ≈ ⟨v_w, v_w'⟩  (paper eq. 1.1) ────────────────────────────────

# Window size γ-correction from Corollary 2.3: PMI_q ≈ ⟨v, v'⟩ + γ
const WINDOW_FOR_GAMMA = 10                    # matches scripts/02_build_cooccurrence.jl
const γ = log(WINDOW_FOR_GAMMA * (WINDOW_FOR_GAMMA - 1) / 2)

@info "D.4  PMI ≈ ⟨v_w, v_w'⟩ + γ  (paper eq. 1.1 / Corollary 2.3, γ = log $(round(exp(γ); digits=1)) ≈ $(round(γ; digits=2)))"
pmi = @phase "D.4 pmi_scatter" pmi_scatter(emb, X; rng = MersenneTwister(1))
slope, intercept = _fit_line(pmi.pmi_pred, pmi.pmi_emp)
@info @sprintf("      Pearson r = %.3f   fitted slope = %.3f   intercept = %.3f   (predicted slope 1, intercept γ ≈ %.2f)",
               pmi.correlation, slope, intercept, γ)

fig4 = Figure(size = (700, 500))
ax4  = Axis(fig4[1, 1],
            title = @sprintf("D.4  Empirical PMI vs ⟨v_w, v_w'⟩   (r = %.3f,  fitted slope = %.2f,  intercept = %.2f)",
                             pmi.correlation, slope, intercept),
            xlabel = "predicted  ⟨v_w, v_w'⟩",
            ylabel = "empirical  PMI(w, w')")
scatter!(ax4, pmi.pmi_pred, pmi.pmi_emp; markersize = 4, color = (:darkorange, 0.5))
let (xmin, xmax) = extrema(pmi.pmi_pred)
    xs = collect(range(xmin, xmax; length = 50))
    lines!(ax4, xs, slope .* xs .+ intercept;
           color = :red, linewidth = 2,
           label = @sprintf("fit: y = %.2f·x + %.2f", slope, intercept))
    lines!(ax4, xs, xs .+ γ;
           color = :black, linestyle = :dash, linewidth = 2,
           label = @sprintf("predicted: y = x + γ  (γ = log %d ≈ %.2f)",
                            Int(round(exp(γ))), γ))
end
axislegend(ax4; position = :lt)
save(joinpath(FIGURES_DIR, "D4_pmi_scatter.pdf"), fig4)

# ─── D.6 (optional) RELATIONS=LINES ────────────────────────────────────────

rl_result = nothing
if isfile(ANALOGY_PATH)
    @info "D.6  RELATIONS=LINES…"
    analogies = load_analogies(ANALOGY_PATH)
    rl_result = @phase "D.6 relations_lines" relations_lines(emb, analogies)

    rels   = sort(collect(keys(rl_result)))
    n_rels = length(rels)
    mean1s = [rl_result[r].mean1 for r in rels]
    mean2s = [rl_result[r].mean2 for r in rels]

    # Grouped horizontal bars via `dodge` — replaces the old offset-by-0.35 hack.
    xs  = repeat(1:n_rels, 2)
    ys  = vcat(mean1s, mean2s)
    grp = vcat(fill(1, n_rels), fill(2, n_rels))
    cols = [g == 1 ? :steelblue : :darkorange for g in grp]

    fig6 = Figure(size = (820, max(360, 0.36 * n_rels * 18 + 220)))
    ax6  = Axis(fig6[1, 1],
                title = "D.6  Cosines of (v_a − v_b) with top-2 singular vectors per relation",
                xlabel = "mean cosine",
                yticks = (1:n_rels, rels))
    barplot!(ax6, xs, ys;
             direction = :x, dodge = grp, color = cols,
             gap = 0.15, dodge_gap = 0.05)

    # Manual two-entry legend (barplot dodge doesn't emit a Makie label).
    elem1 = PolyElement(color = :steelblue)
    elem2 = PolyElement(color = :darkorange)
    Legend(fig6[1, 2], [elem1, elem2], ["u₁ (top sing. vec.)", "u₂"];
           framevisible = false)
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
