using Quantica
using Documenter

DocMeta.setdocmeta!(Quantica, :DocTestSetup, :(using Quantica); recursive=true)

makedocs(;
    modules=[Quantica],
    authors="Pablo San-Jose",
    repo="https://github.com/pablosanjose/Quantica.jl/blob/{commit}{path}#L{line}",
    sitename="Quantica.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pablosanjose.github.io/Quantica.jl",
        assets=["assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
            "Welcome" => "tutorial/tutorial.md",
            "Glossary" => "tutorial/glossary.md",
            "Lattices" => "tutorial/lattices.md",
            "Models" => "tutorial/models.md",
            "Hamiltonians" => "tutorial/hamiltonians.md",
            "Bandstructures" => "tutorial/bandstructures.md",
            "GreenFunctions" => "tutorial/greenfunctions.md",
            "Observables" => "tutorial/observables.md"
            ],
        "Advanced" => [
            "Self-consistent mean fields" => "advanced/meanfield.md",
            "Wannier90 imports" => "advanced/wannier90.md"
            ],
        "Examples" => "examples.md",
        "API" => "api.md",
    ]
)

deploydocs(;
    repo="github.com/pablosanjose/Quantica.jl",
)
