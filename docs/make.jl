using Quantica
using Documenter

DocMeta.setdocmeta!(Quantica, :DocTestSetup, :(using Quantica); recursive=true)

makedocs(;
    modules=[Quantica],
    authors="Pablo San-Jose",
    repo="https://github.com/BacAmorim/Quantica.jl/blob/{commit}{path}#L{line}",
    sitename="Quantica.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/BacAmorim/Quantica.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => "examples.md",
        "Reference" => "reference.md",
    ],
    doctest = false,
)

deploydocs(;
    repo="https://github.com/BacAmorim/Quantica.jl",
)
