"""
    LatentRandomWalk

A faithful prototype of Arora, Li, Liang, Ma & Risteski,
*"A Latent Variable Model Approach to PMI-based Word Embeddings"* (TACL 2016).

Pipeline:
  1. `Vocabulary` from a tokenised corpus      (`build_vocabulary`)
  2. Sliding-window co-occurrence sparse mat.  (`build_cooccurrence`)
  3. SN-objective training with AdaGrad        (`train`)
  4. Empirical verification of the theorems    (`partition_function`,
                                                `norm_vs_frequency`,
                                                `singular_value_isotropy`,
                                                `pmi_scatter`,
                                                `relations_lines`)
  5. Analogy evaluation                        (`evaluate_analogies`)

Submodule `LatentRandomWalk.SDE` implements the continuous-time
reformulation as Brownian motion on `S^{d-1}` (stretch goal).
"""
module LatentRandomWalk

# Core public types
export Vocabulary, Embeddings, TrainingConfig, Analogy, AnalogyResult

# Vocabulary / corpus
export read_text8, text8_tokens, build_vocabulary, tokenize, word_id

# Co-occurrence
export build_cooccurrence

# Training
export train, sn_loss, vector, dim

# Verification diagnostics
export partition_function, norm_vs_frequency, singular_value_isotropy,
       pmi_scatter, relations_lines

# Analogy evaluation
export load_analogies, evaluate_analogies

include("corpus.jl")
include("cooccurrence.jl")
include("model.jl")
include("analogies.jl")
include("verify.jl")
include("sde.jl")

end # module LatentRandomWalk
