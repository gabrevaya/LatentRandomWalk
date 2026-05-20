### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0000-0000-0000-000000000001
md"""
# Latent Random-Walk Word Embeddings — a walkthrough

Companion notebook to the journal-club talk on Arora, Li, Liang, Ma & Risteski
(TACL 2016).

The whole talk in one sentence: **a slow random walk of a "discourse vector"
on the unit sphere, emitting words through a log-linear model, predicts
PMI ≈ ⟨v_w, v_w'⟩ / d in closed form** — and it turns out to actually hold
on real data.

This notebook loads the artifacts produced by `scripts/run_all.jl` and
reproduces the five diagnostic plots live.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using LatentRandomWalk
    using JLD2: jldopen
    using CairoMakie
    using Statistics: mean, std
    using Printf: @sprintf
end

# ╔═╡ 11111111-0000-0000-0000-000000000003
md"""
## Load artifacts

If you haven't run the pipeline yet, do

```bash
julia --project --threads=auto scripts/run_all.jl
```

from the project root.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000004
begin
    const DATA_DIR    = joinpath(@__DIR__, "..", "data")
    const VECTORS_PATH = joinpath(DATA_DIR, "results", "vectors.jld2")
    const COOC_PATH    = joinpath(DATA_DIR, "processed", "cooccurrence.jld2")
    const VERIFY_PATH  = joinpath(DATA_DIR, "results", "verification.jld2")

    emb = jldopen(f -> f["embeddings"], VECTORS_PATH, "r")
    X   = jldopen(f -> f["X"],          COOC_PATH,    "r")
    ver = jldopen(f -> Dict(k => f[k] for k in keys(f)), VERIFY_PATH, "r")
    (d = dim(emb), n = length(emb), C = emb.C)
end

# ╔═╡ 11111111-0000-0000-0000-000000000005
md"""
## The model in one slide

For each step $t$ the discourse vector $c_t \in \mathbb{R}^d$ does a slow
random walk on the unit sphere, and the word at time $t$ is emitted from

$$\Pr[w \mid c_t] \propto \exp\langle v_w, c_t\rangle.$$

After integrating out the latent prior (Lemma 2.1, "self-normalisation"),
the joint co-occurrence probability has a closed form

$$\log p(w, w') \;\approx\; \frac{\lVert v_w + v_{w'}\rVert^2}{2d} - 2\log Z + \varepsilon$$

— Theorem 2.2. Subtracting the two single-word equations gives the **headline
prediction**

$$\boxed{\;\mathrm{PMI}(w, w') \;\approx\; \frac{\langle v_w, v_{w'}\rangle}{d}\;}$$

(eq. 1.1 / Corollary 2.3). The whole training algorithm is just a least-squares
fit to the pair form — no other tricks.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000006
md"""
## D.1  The partition function concentrates  (Lemma 2.1)

For 1000 random discourse directions $c$ of norm $4 / \mu_w$, the partition
function $Z_c = \sum_w \exp\langle v_w, c\rangle$ should cluster tightly
around its mean. This is the model's *self-normalisation* claim — and the
reason the SN objective doesn't need a partition-function term.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000007
let pf = ver["partition_function"]
    fig = Figure(size = (700, 360))
    ax  = Axis(fig[1, 1], xlabel = "Z_c / 𝔼[Z_c]", ylabel = "count")
    hist!(ax, Float64.(pf.normalised); bins = 60, strokecolor = :black, strokewidth = 0.4)
    vlines!(ax, [0.9, 1.1]; color = :red, linestyle = :dash, label = "[0.9, 1.1]")
    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 11111111-0000-0000-0000-000000000008
md"""
## D.2  Squared norm ↔ log frequency  (Theorem 2.2, single-word case)

Plugging a single word into Theorem 2.2 gives

$$\log p(w) \;\approx\; \frac{\lVert v_w\rVert^2}{2d} - \log Z.$$

The model therefore predicts a linear relationship between
$\lVert v_w\rVert^2$ and $\log p(w)$. The paper reports Pearson r ≈ 0.75.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000009
let nf = ver["norm_vs_frequency"]
    fig = Figure(size = (640, 480))
    ax  = Axis(fig[1, 1], title = @sprintf("Pearson r = %.3f", nf.correlation),
               xlabel = "log p(w)", ylabel = "‖v_w‖²")
    scatter!(ax, nf.log_freqs, nf.sq_norms; markersize = 3, color = (:steelblue, 0.5))
    fig
end

# ╔═╡ 11111111-0000-0000-0000-00000000000a
md"""
## D.3  Isotropy: singular values of $V$  (Theorem 4.1)

The "purification" argument that makes RELATIONS=LINES work needs the
$n \times d$ matrix of word vectors to behave like a random matrix —
specifically, smallest non-zero σ over RMS σ ≈ 1/3.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000b
let sv = ver["singular_values"]
    fig = Figure(size = (640, 380))
    ax  = Axis(fig[1, 1], title = @sprintf("min σ / RMS σ = %.3f", sv.ratio),
               xlabel = "index", ylabel = "σ")
    lines!(ax, 1:length(sv.singular_values), Float64.(sv.singular_values))
    hlines!(ax, [Float64(sv.rms)]; color = :red, linestyle = :dash, label = "RMS")
    axislegend(ax)
    fig
end

# ╔═╡ 11111111-0000-0000-0000-00000000000c
md"""
## D.4  PMI ≈ ⟨v_w, v_w'⟩ / d  (Corollary 2.3 — the headline)

This is the test of the model's headline equation. The SN training does
*not* fit PMI directly — it fits the *pair* form of Theorem 2.2 — so the
agreement here, combined with D.2, is genuine evidence that the model is
right about both the pair and single-word laws simultaneously.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000d
let pmi = ver["pmi_scatter"]
    X̄, Ȳ = mean(pmi.pmi_pred), mean(pmi.pmi_emp)
    m = sum((pmi.pmi_pred .- X̄) .* (pmi.pmi_emp .- Ȳ)) /
        sum((pmi.pmi_pred .- X̄) .^ 2)
    c = Ȳ - m * X̄
    fig = Figure(size = (640, 480))
    ax  = Axis(fig[1, 1],
               title = @sprintf("Pearson r = %.3f   slope = %.2f", pmi.correlation, m),
               xlabel = "⟨v_w, v_w'⟩ / d   (model)",
               ylabel = "PMI(w, w')   (empirical)")
    scatter!(ax, pmi.pmi_pred, pmi.pmi_emp; markersize = 3, color = (:darkorange, 0.3))
    xs = collect(extrema(pmi.pmi_pred))
    lines!(ax, xs, m .* xs .+ c; color = :red, linewidth = 2)
    fig
end

# ╔═╡ 11111111-0000-0000-0000-00000000000e
md"""
## D.5 / D.6  Analogy structure (optional, requires `questions-words.txt`)

The analogy testbed (D.5) and the RELATIONS=LINES diagnostic (D.6) are
produced by `scripts/05_analogies.jl` and `scripts/04_verify.jl` and live in
`data/results/analogies.jld2` and the `relations_lines` entry of
`verification.jld2`. They make `figures/D5_analogy_accuracy.pdf` and
`figures/D6_relations_lines.pdf`.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000f
md"""
## The continuous-time picture (stretch)

`LatentRandomWalk.SDE.simulate(d)` integrates the corresponding Itô SDE

$$dc_t = -\tfrac{d-1}{2}\, c_t\, dt + (I - c_t c_t^\top)\, dW_t$$

with `StochasticDiffEq.EM()`. `scripts/06_sde_demo.jl` produces a plot of
$Z_c$ along the trajectory, demonstrating that Lemma 2.1's concentration
holds *in continuous time too*.

This connects the paper to latent-SDE / state-space-model work and is the
original contribution of this prototype (small as it is).
"""

# ╔═╡ Cell order:
# ╟─11111111-0000-0000-0000-000000000001
# ╠═11111111-0000-0000-0000-000000000002
# ╟─11111111-0000-0000-0000-000000000003
# ╠═11111111-0000-0000-0000-000000000004
# ╟─11111111-0000-0000-0000-000000000005
# ╟─11111111-0000-0000-0000-000000000006
# ╠═11111111-0000-0000-0000-000000000007
# ╟─11111111-0000-0000-0000-000000000008
# ╠═11111111-0000-0000-0000-000000000009
# ╟─11111111-0000-0000-0000-00000000000a
# ╠═11111111-0000-0000-0000-00000000000b
# ╟─11111111-0000-0000-0000-00000000000c
# ╠═11111111-0000-0000-0000-00000000000d
# ╟─11111111-0000-0000-0000-00000000000e
# ╟─11111111-0000-0000-0000-00000000000f
