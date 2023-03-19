module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, LatticeSlice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      GreenFunction, GreenSolution, argerror, harmonics, sublats, siterange, site, norm,
      normalize, nsites, nzrange, rowvals, sanitize_SVector

import Quantica: plotlattice, plotlattice!, qplot, plottables

include("plotlattice.jl")

end # module