module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, LatticeSlice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      GreenFunction, GreenSolution, Bands,
      argerror, harmonics, sublats, siterange, site, norm,
      normalize, nsites, nzrange, rowvals, sanitize_SVector

import Quantica: plotlattice, plotlattice!, plotbands, plotbands!, qplot

include("plotlattice.jl")
include("plotbands.jl")
include("tools.jl")

end # module