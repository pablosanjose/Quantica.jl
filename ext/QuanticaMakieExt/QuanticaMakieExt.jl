module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, LatticeSlice, AbstractHamiltonian, Hamiltonian,
      ParametricHamiltonian, Harmonic, Bravais, SVector, GreenFunction, GreenSolution,
      argerror, harmonics, sublats, siterange, site, norm,
      normalize, nsites, nzrange, rowvals, nonzeros, sanitize_SVector

import Quantica: plotlattice, plotlattice!, plotbands, plotbands!, qplot, qplot!, qplotdefaults

# Currying fallback
Quantica.qplot(; kw...) = x -> Quantica.qplot(x; kw...)
Quantica.qplot!(; kw...) = x -> Quantica.qplot!(x; kw...)

include("plotlattice.jl")
include("plotbands.jl")
include("tools.jl")
include("defaults.jl")
include("docstrings.jl")

end # module
