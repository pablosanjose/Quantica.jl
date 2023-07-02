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
        "Tutorial" => Any[
            "tutorial/tutorial.md",
            "tutorial/glossary.md",
            "tutorial/lattices.md",
            "tutorial/models.md",
            "tutorial/hamiltonians.md",
            "tutorial/greenfunctions.md",
            "tutorial/observables.md"
            ],
        "Examples" => "examples.md",
        "API" => "api.md",
    ]
)

deploydocs(;
    repo="github.com/pablosanjose/Quantica.jl",
)
