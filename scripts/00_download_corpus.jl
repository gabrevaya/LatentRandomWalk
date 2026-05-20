# Download text8 (Matt Mahoney's 100 MB cleaned Wikipedia extract) and the
# Google analogy testbed. Skipped if the files are already present.
include(joinpath(@__DIR__, "_setup.jl"))

using Downloads: download
using CodecZlib: GzipDecompressorStream

const TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
const QWORDS_URL = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"

text8_path  = joinpath(RAW_DIR, "text8")
qwords_path = joinpath(RAW_DIR, "questions-words.txt")
mkpath(RAW_DIR)

if !isfile(text8_path)
    @info "Downloading text8…"
    zip_path = joinpath(RAW_DIR, "text8.zip")
    download(TEXT8_URL, zip_path)
    run(`unzip -o $zip_path -d $RAW_DIR`)
    rm(zip_path)
else
    @info "text8 already present, skipping download."
end

if !isfile(qwords_path)
    @info "Downloading Google analogy testbed…"
    download(QWORDS_URL, qwords_path)
else
    @info "questions-words.txt already present, skipping download."
end
