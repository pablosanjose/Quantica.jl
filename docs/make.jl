using Quantica
using Documenter

DocMeta.setdocmeta!(Quantica, :DocTestSetup, :(using Quantica); recursive=true)

makedocs(;
    modules=[Quantica],
    authors="Pablo San-Jose",
    # repo="https://github.com/pablosanjose/Quantica.jl/blob/{commit}{path}#L{line}",
    sitename="Quantica.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pablosanjose.github.io/Quantica.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => "examples.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/pablosanjose/Quantica.jl",
)
