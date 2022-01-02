module Quantica

# Use README as the docstring of the module:
using Base.Threads: Iterators
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) Quantica

using StaticArrays
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ProgressMeter
using Random
using SuiteSparse
using FunctionWrappers: FunctionWrapper
using ExprTools
using SparseArrays: getcolptr, AbstractSparseMatrix, AbstractSparseMatrixCSC
using Statistics: mean
using Compat # for use of argmin/argmax in bandstructure.jl

export sublat, lattice, dims, supercell, bravais_matrix, siteindices, sitepositions,
       hopping, onsite, @onsite!, @hopping!, @block!, parameters, neighbors,
       ket, ketmodel, randomkets, basiskets,
       hamiltonian, parametric, bloch, bloch!, similarmatrix,
       flatten, unflatten, orbitalstructure, wrap, transform, translate, combine,
       spectrum, bandstructure, diagonalizer, mesh, isometric, splitbands,
       bands, vertices, minima, maxima, gapedge, gap, isinband,
       energies, states, degeneracy,
       momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
       greens, greensolver, Schur1D

export eigensolver

export LatticePresets, LP, RegionPresets, RP, Eigensolvers, ES #, HamiltonianPresets, HP
export @SMatrix, @SVector, SMatrix, SVector, SA
export ishermitian, I

include("types.jl")
include("apply.jl")
include("iterators.jl")
include("selector.jl")
include("lattice.jl")
include("model.jl")
include("builders.jl")
include("hamiltonian.jl")
include("supercell.jl")
include("transform.jl")
include("eigensolver.jl")
# include("ket.jl")
# include("parametric.jl")
# include("slice.jl")
# include("mesh.jl")
# include("diagonalizer.jl")
include("bandstructure.jl")
# include("KPM.jl")
# include("effective.jl")
# include("greens.jl")
include("sanitize.jl")
include("show.jl")
include("tools.jl")
include("convert.jl")

include("presets.jl")

include("precompile.jl")

end