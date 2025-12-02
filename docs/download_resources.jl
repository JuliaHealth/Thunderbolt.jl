# Download some assets necessary for docs/testing not stored in the repo
import Downloads

const directory = joinpath(@__DIR__, "src", "tutorials")
mkpath(directory)

assets_url_base = "https://raw.githubusercontent.com/JuliaHealth/Thunderbolt.jl/gh-pages/assets/"
for (file, url) in [
        "spiral-wave.gif" => assets_url_base * "spiral-wave-lts-amr.gif",
        "contracting-left-ventricle.gif" => assets_url_base * "contracting-biv-simple.gif",
    ]
    afile = joinpath(directory, file)
    if !isfile(afile)
        Downloads.download(url, afile)
    end
end
