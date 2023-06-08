![Quantica.jl logo](assets/banner.png)

[Quantica.jl](https://github.com/pablosanjose/Quantica.jl/) is a Julia package for building generic tight-binding models and computing spectral and transport properties.

```@contents
Pages = [
    "manual.md",
    "examples.md",
    "reference.md",
]
Depth = 1
```

## Installation

```julia
import Pkg; Pkg.add("Quantica")
```

Quantica.jl requires Julia v1.9 or later. Some of its functionality, notably plotting, will become available only after `using GLMakie`, or some other plotting package from the [Makie.jl](https://docs.makie.org/stable/) family. Install `GLMakie` with
```julia
import Pkg; Pkg.add("GLMakie")
```

## Asking questions, reporting bugs

If you encounter problems, please read the manual and examples, your question is probably answered there. You can also check the docstring of each Quantica command within the Julia REPL, by entering the command preceded by a `?`, e.g. `?hamiltonian`.

If you are still stuck, you may sometimes find me (`@pablosanjose`) at the [Julia Slack](https://julialang.slack.com) or [Julia Discourse](https://discourse.julialang.org).

If you believe you found a bug in Quantica.jl, please don't hesitate to file a [GitHub issue](https://github.com/pablosanjose/Quantica.jl/issues), preferably with detailed instructions to reproduce it. Pull requests are also welcome!