module Quantica

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) Quantica

using Requires

function __init__()
      @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("plot_makie.jl")
      @require VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a" include("plot_vegalite.jl")
end

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays,
      ProgressMeter, LinearMaps, Random, SuiteSparse

using ExprTools

using SparseArrays: getcolptr, AbstractSparseMatrix

using Statistics: mean

export sublat, bravais, lattice, dims, supercell, unitcell,
       hopping, onsite, @onsite!, @hopping!, parameters, siteselector, hopselector, nrange,
       sitepositions, siteindices, not,
       ket, randomkets,
       hamiltonian, parametric, bloch, bloch!, similarmatrix,
       flatten, unflatten, orbitalstructure, wrap, transform!, combine,
       spectrum, bandstructure, diagonalizer, cuboid, isometric, splitbands,
       bands, vertices,
       energies, states, degeneracy,
       momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
       greens, greensolver, Schur1D

export RegionPresets, RP, LatticePresets, LP, HamiltonianPresets, HP

export LinearAlgebraPackage, ArpackPackage, ArnoldiMethodPackage, KrylovKitPackage

export @SMatrix, @SVector, SMatrix, SVector, SA

export ishermitian, I

export SparseMatrixCSC

const NameType = Symbol
const nametype = Symbol

const TOOMANYITERS = 10^8

include("iterators.jl")
include("tools.jl")
include("presets.jl")
include("lattice.jl")
include("model.jl")
include("hamiltonian.jl")
include("parametric.jl")
include("mesh.jl")
include("diagonalizer.jl")
include("bandstructure.jl")
include("KPM.jl")
include("greens.jl")
include("convert.jl")

end