# Shared boilerplate: activate the project no matter how the script is
# invoked, and resolve repo paths from `$REPO/scripts/` upwards.
import Pkg
const REPO_DIR = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(REPO_DIR)

const DATA_DIR    = joinpath(REPO_DIR, "data")
const RAW_DIR     = joinpath(DATA_DIR, "raw")
const PROC_DIR    = joinpath(DATA_DIR, "processed")
const RESULTS_DIR = joinpath(DATA_DIR, "results")
const FIGURES_DIR = joinpath(REPO_DIR, "figures")

foreach(p -> mkpath(p), (PROC_DIR, RESULTS_DIR, FIGURES_DIR))

# Helper for timing a phase: prints duration and peak Δ-allocated bytes.
macro phase(label, expr)
    quote
        local _t0 = time_ns()
        local _b0 = Base.gc_live_bytes()
        local _val = $(esc(expr))
        local _dt = (time_ns() - _t0) / 1e9
        local _db = (Base.gc_live_bytes() - _b0) / 2^20
        @info "─── " * $(esc(label)) * " ──── " *
              "Δt = $(round(_dt; digits=2)) s   Δlive = $(round(_db; digits=1)) MiB"
        _val
    end
end
