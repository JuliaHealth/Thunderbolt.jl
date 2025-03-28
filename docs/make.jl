using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

using Documenter, DocumenterCitations, Thunderbolt

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate tutorials and how-to guides
include("generate.jl")

bibtex_plugin = CitationBibliography(
    joinpath(@__DIR__, "src", "assets", "references.bib"),
    style=:numeric
)

# Build documentation.
@timeit dto "makedocs" makedocs(
    format = Documenter.HTML(
        assets = [
            "assets/custom.css",
            "assets/citations.css",
            # "assets/favicon.ico"
        ],
        # canonical = "https://localhost/",
        collapselevel = 1,
    ),
    sitename = "Thunderbolt.jl",
    doctest = false,
    warnonly = true,
    draft = liveserver,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Continuum Mechanics" => [
                "CM01: Simple Active Stress" => "tutorials/cm01_simple-active-stress.md",
                "CM02: Prestressing (WIP)" => "tutorials/cm02_prestress.md",
                "CM03: 0D Blood Circuit" => "tutorials/cm03_3d0d-coupling.md",
                "CM04: Pericadium (TODO)" => "tutorials/cm04_pericardium.md",
                "CM05: Four Chamber (TODO)" => "tutorials/cm05_fourchambers.md",
                "CM06: Heart Valves (TODO)" => "tutorials/cm06_heartvalves.md",
            ],
            "Electrophysiology" => [
                "EP01: Spiral Wave" => "tutorials/ep01_spiral-wave.md",
                "EP02: Purkinje Network (TODO)" => "tutorials/ep02_purkinje.md",
                "EP03: Defibrillation (TODO)" => "tutorials/ep03_bidomain.md",
                "EP04: Monodomain ECG" => "tutorials/ep04_geselowitz-ecg.md",
                "EP05: Eikonal Models (WIP)" => "tutorials/ep05_eikonal.md",
                "EP06: Pacemakers (TODO)" => "tutorials/ep06_pacemaker.md",
            ],
        ],
        "Topic Guides" => [
            "Overview" => "topics/index.md",
            "topics/operators.md",
            "topics/couplers.md",
            "topics/time-integration.md",
            "topics/nonlinear-solver.md",
        ],
        "How-to guides" => [
            "Overview" => "howto/index.md",
            "howto/benchmarking.md",
            "howto/custom-ep-cell-model.md",
            "howto/custom-stimulation-protocols.md",
            "howto/custom-energies.md",
            "howto/custom-sarcomere.md",
            "howto/custom-elements.md",
        ],
        "API Reference" => [
            "Overview" => "api-reference/index.md",
            "api-reference/models.md",
            "api-reference/mesh.md",
            "api-reference/functions.md",
            "api-reference/problems.md",
            "api-reference/discretization.md",
            "api-reference/operators.md",
            "api-reference/solver.md",
            "api-reference/utility.md",
        ],
        "Developer Documentation" => [
            "Overview" => "devdocs/index.md",
            "devdocs/element_interface.md",
            "devdocs/domain_management.md",
        ],
        "vroom.md",
        "references.md",
        ],
    plugins = [
        bibtex_plugin,
    ]
)

# make sure there are no *.vtu files left around from the build
@timeit dto "remove vtk files" cd(joinpath(@__DIR__, "build", "tutorials")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end

# Deploy built documentation
if !liveserver
    @timeit dto "deploydocs" deploydocs(
        repo = "github.com/JuliaHealth/Thunderbolt.jl.git",
        push_preview=true,
        versions = [
            "stable" => "v^",
            "dev" => "dev"
        ]
    )
end

print_timer(dto)
