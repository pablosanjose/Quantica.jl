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

## PlotArgumentTypes

const PlotLatticeArgumentType{E} = Union{
    Lattice{<:Any,E},
    LatticeSlice{<:Any,E},
    AbstractHamiltonian{<:Any,E},
    GreenFunction{<:Any,E}}

const PlotBandsArgumentType{E} =
    Union{Quantica.Bandstructure{<:Any,E},
          Quantica.Subband{<:Any,E},
          AbstractVector{<:Quantica.Subband{<:Any,E}},
          Quantica.Mesh{<:Quantica.BandVertex{<:Any,E}},
          AbstractVector{<:Quantica.Mesh{<:Quantica.BandVertex{<:Any,E}}}}

const PlotArgumentType{E} = Union{PlotLatticeArgumentType{E},PlotBandsArgumentType{E}}

## Currying fallbacks
Quantica.qplot(; kw...) = x -> Quantica.qplot(x; kw...)
Quantica.qplot!(; kw...) = x -> Quantica.qplot!(x; kw...)

include("plotlattice.jl")
include("plotbands.jl")
include("tools.jl")
include("defaults.jl")
include("docstrings.jl")

end # module
