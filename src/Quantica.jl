module Quantica

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) Quantica

using Requires

function __init__()
      @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("plot.jl")
end

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays,
      ProgressMeter, LinearMaps, Random

using ExprTools

using SparseArrays: getcolptr, AbstractSparseMatrix

export sublat, bravais, lattice, dims, sites, supercell, unitcell,
       hopping, onsite, @onsite!, @hopping!, parameters,
       hamiltonian, parametric, bloch, bloch!, optimize!, similarmatrix,
       flatten, wrap, transform!, combine,
       spectrum, bandstructure, marchingmesh, defaultmethod, bands, vertices,
       energies, states,
       momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM

export LatticePresets, RegionPresets, HamiltonianPresets

export LinearAlgebraPackage, ArpackPackage, KrylovKitPackage

export @SMatrix, @SVector, SMatrix, SVector

export ishermitian, I

const NameType = Symbol
const nametype = Symbol

const TOOMANYITERS = 10^8

include("iterators.jl")
include("presets.jl")
include("lattice.jl")
include("model.jl")
include("hamiltonian.jl")
include("parametric.jl")
include("mesh.jl")
include("diagonalizer.jl")
include("bandstructure.jl")
include("KPM.jl")
include("convert.jl")
include("tools.jl")

end