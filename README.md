# LatentRandomWalk

A Julia prototype of Arora, Li, Liang, Ma & Risteski,
[*"A Latent Variable Model Approach to PMI-based Word Embeddings"*](https://aclanthology.org/Q16-1028/)
(TACL 2016) — implemented as a journal-club presentation aid.

## What this package does

1. **Builds a vocabulary** from a tokenised corpus
   (defaults sized for [text8](http://mattmahoney.net/dc/text8.zip), ~17 M tokens).
2. **Builds the sliding-window co-occurrence matrix** as a symmetric
   `SparseMatrixCSC{Float32, Int32}` via `OhMyThreads`-parallel dict-merge.
3. **Trains word vectors** by stochastic gradient descent on the SN objective
   from §3 of the paper, with hand-coded AdaGrad fused into the inner loop.
4. **Empirically verifies the paper's predictions** via five diagnostics
   (D.1–D.6 below); each emits a JLD2 record and a Makie figure.
5. **Optionally** reformulates the discourse walk as a continuous-time
   Stratonovich SDE on $S^{d-1}$ and integrates it with a projected
   Euler–Maruyama scheme (tangent-space increment + renormalisation).

## Quick start

```bash
cd LatentRandomWalk
# put text8 and (optionally) questions-words.txt in data/raw/
julia --project --threads=auto scripts/run_all.jl
```

Each phase script is also runnable on its own and is skipped by `run_all.jl`
if its checkpoint already exists. `FORCE=1 julia --project scripts/run_all.jl`
re-runs everything.

Override training hyperparameters with environment variables:

```bash
DIM=100 EPOCHS=20 julia --project --threads=auto scripts/03_train.jl
```

## The verification protocol

Verifying the paper rather than the implementation is the whole point. The
five diagnostics are:

| ID  | Test                                                      | Paper reference                          | Predicted outcome                                                           |
|-----|-----------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------|
| D.1 | Partition function $Z_c$ concentrates                     | Lemma 2.1 / Fig. 1a                      | $Z_c$ clustered in $[0.9,\, 1.1] \cdot \mathbb{E}[Z_c]$                     |
| D.2 | $\Vert v_w \Vert^2$ linear in $\log p(w)$                 | Theorem 2.2                              | Pearson $r \approx 0.75$                                                    |
| D.3 | Singular values of $V$ are random-matrix-like             | Theorem 4.1                              | $\sigma_{\min} / \sigma_{\mathrm{RMS}} \approx 1/3$                         |
| D.4 | $\mathrm{PMI}(w, w') \approx \langle v_w, v_{w'} \rangle$ | Corollary 2.3 (paper's headline eq. 1.1) | slope $\approx 1$, intercept $\approx \gamma = \log(q(q-1)/2) \approx 3.81$ |
| D.5 | Google analogy testbed                                    | §5.2                                     | ~35–50 % on text8                                                           |
| D.6 | $v_a - v_b$ aligns with a single direction                | §5.3 / RELATIONS=LINES                   | $\overline{\cos(v_a - v_b,\, u_1)} \gg \overline{\cos(v_a - v_b,\, u_2)}$   |

All five run in well under a minute against trained `text8` vectors. They
constitute the central slides of the journal-club talk.

## Project layout

```
LatentRandomWalk/
├── Project.toml / Manifest.toml
├── src/
│   ├── LatentRandomWalk.jl       # module top, includes & exports
│   ├── corpus.jl                 # Vocabulary, tokenisation
│   ├── cooccurrence.jl           # sparse pair-count matrix
│   ├── model.jl                  # Embeddings, SN training, AdaGrad
│   ├── analogies.jl              # Google/MSR analogy evaluation
│   ├── verify.jl                 # D.1, D.2, D.3, D.4, D.6 diagnostics
│   └── sde.jl                    # LatentRandomWalk.SDE submodule
├── scripts/
│   ├── 00_download_corpus.jl
│   ├── 01_build_vocab.jl
│   ├── 02_build_cooccurrence.jl
│   ├── 03_train.jl
│   ├── 04_verify.jl
│   ├── 05_analogies.jl
│   ├── 06_sde_demo.jl
│   └── run_all.jl
├── test/                         # gradient check vs Zygote, etc.
├── notebooks/walkthrough.jl      # Pluto journal-club companion
├── data/                         # gitignored: raw/, processed/, results/
└── figures/                      # generated PDFs
```

## Implementation notes

### SN training (the hot path)

The SN loss is

```math
L(V, C) = \sum_{w, w'} \min(X_{w,w'},\, X_{\max}) \cdot \bigl(\log X_{w,w'} - \Vert v_w + v_{w'} \Vert^2 - C\bigr)^2
```

Per pair, with residual $r = \log X_{w,w'} - \Vert v_w + v_{w'} \Vert^2 - C$
and weight $\bar X = \min(X_{w,w'},\, X_{\max})$, the gradient update is

```math
\begin{aligned}
\nabla_{v_w} L &\mathrel{+}= -4\, \bar X\, r\, (v_w + v_{w'}) \quad (\text{same for } v_{w'}) \\
\partial_C L &\mathrel{+}= -2\, \bar X\, r .
\end{aligned}
```

We iterate over the upper triangle of the symmetric co-occurrence matrix
(`row < col`) and apply per-parameter AdaGrad inline. The inner loop is
`@inbounds @simd`, type-stable (`@code_warntype`-clean), allocation-free
after the upfront `randperm`. AdaGrad is *not* delegated to
`Optimisers.jl`: the optimizer step is fused with the sparse gradient
access pattern, which a generic library cannot exploit — see
[the discussion in `implementation-plan.md`](../implementation-plan.md).

**SN units vs model units.** Theorem 2.2 of the paper is written in vectors
$\hat v$ of typical norm $O(\sqrt{d})$ and predicts

```math
\log p(w, w') \approx \frac{\Vert \hat v_w + \hat v_{w'} \Vert^2}{2d} - 2 \log Z .
```

The SN objective drops the $1/(2d)$ factor and fits the rescaled
$v_{\mathrm{SN}} = \hat v / \sqrt{2d}$ — so what the SN solver returns has
norms of order $\sqrt{\log X}$ (compare D.2). The paper's headline equation
1.1, $\langle v_w, v_{w'} \rangle \approx \mathrm{PMI}(w, w')$, is the
*SN-units* form of Theorem 2.2's
$\langle \hat v_w, \hat v_{w'} \rangle / d \approx \mathrm{PMI}$: the
factor of $d$ is absorbed by the rescaling. That's why D.4's reference
line is $y = x + \gamma$ (slope 1, intercept
$\gamma = \log(q(q-1)/2) \approx 3.81$ for the window-size constant of
Corollary 2.3), not $y = x/d$.

### Performance choices

- **Float32 throughout.** Saves half the memory, halves cache pressure,
  precision is irrelevant for embeddings. (Note: `svdvals(V)` in D.3
  promotes to Float64 internally inside LAPACK — the trained vectors
  themselves remain Float32 everywhere else.)
- **Symmetric upper-triangle storage** during training (each unique pair
  visited once).
- **BLAS-vectorised verification.** D.1 is a series of GEMVs; D.5 is one
  GEMM per batch of analogy queries.
- **`OhMyThreads`** for the embarrassingly-parallel parts of the pipeline
  (Phase B co-occurrence build; D.6 SVDs).

### What the SDE submodule is for

The plan's stretch goal: take the discrete random walk's continuous-time
limit, which is Brownian motion on $S^{d-1}$. We integrate the Stratonovich
form

```math
dc_t = (I - c_t c_t^\top) \circ dW_t
```

with a *projected Euler step* — a tangent-space increment followed by
renormalisation back to the sphere. This is the standard geometric
integrator for SDEs on $S^{d-1}$ and is what the implementation plan calls
for ("project back to the sphere after each step"). The ambient-space Itô
form has an $(d-1)/2$ drift that's stiff at $d = 300$ and would force a
much smaller step from `StochasticDiffEq.EM()`; the projected scheme is
both simpler and exact-on-the-sphere by construction.

`partition_function_along_path` then shows that $Z_c$ stays approximately
constant along the trajectory — Lemma 2.1's prediction, now made about a
continuous-time process.

## Caveats from the implementer

- The paper is sometimes ambiguous about whether $X_{w,w'}$ is the raw
  count or a distance-weighted one. The SN derivation works with raw counts,
  and that's what we use.
- The bias scalar $C$ absorbs $-2 \log Z$; we don't try to recover $Z$
  separately.
- The Stratonovich SDE is integrated with a projected Euler step (tangent
  increment then renormalise), not a generic Euler–Maruyama solver. The
  ambient-space Itô form carries a stiff $(d-1)/2$ drift that would force
  a tiny timestep at $d = 300$; the projected scheme avoids it and keeps
  $\Vert c_t \Vert = 1$ to floating-point precision at every step.
- The Google testbed contains some questions whose answers aren't in the
  text8 vocabulary; we skip those and report coverage.

## Reproducing the figures

```bash
julia --project --threads=auto scripts/run_all.jl
ls figures/
```

All randomness is seeded from a single seed (`SEED=` env var; default 0).
