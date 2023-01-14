module Quantica

const REPOISSUES = "https://github.com/pablosanjose/Quantica.jl/issues"

using Base.Threads: Iterators

using StaticArrays
using NearestNeighbors
using SparseArrays
using SparseArrays: getcolptr, AbstractSparseMatrix, AbstractSparseMatrixCSC
using LinearAlgebra
using ProgressMeter
using Random
using SuiteSparse
using FunctionWrappers: FunctionWrapper
using ExprTools
using IntervalTrees
using FrankenTuples
using Statistics: mean

export sublat, lattice, supercell, bravais_matrix,
       hopping, onsite, @onsite, @hopping, @onsite!, @hopping!, neighbors,
       hamiltonian, parametric, call!,
       flat, unflat, wrap, transform, transform!, translate, translate!,
       spectrum, energies, states, bands, mesh, subbands, slice,
       greenfunction, attach

# export sublat, lattice, dims, supercell, bravais_matrix, siteindices, sitepositions,
#        hopping, onsite, @onsite!, @hopping!, @block!, parameters, neighbors,
#        ket, ketmodel, randomkets, basiskets,
#        hamiltonian, parametric, bloch, bloch!, similarmatrix,
#        flatten, unflatten, orbitalstructure, wrap, transform, translate, combine,
#        spectrum, band, mesh, isometric, subbands,
#        vertices, minima, maxima, gapedge, gap, isinband,
#        energies, states, degeneracy,
#        momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
#        greens, greensolver, Schur1D

export LatticePresets, LP, RegionPresets, RP  #, HamiltonianPresets, HP
export EigenSolvers, ES, GreenSolvers, GS
export @SMatrix, @SVector, SMatrix, SVector, SA
export ishermitian, I
export ftuple

# Preamble
include("iterators.jl")
include("builders.jl")
include("tools.jl")
include("sanitize.jl")
include("specialmatrices.jl")

# Core
include("types.jl")
include("selector.jl")
include("lattice.jl")
include("model.jl")
include("hamiltonian.jl")
include("greenfunction.jl")
include("selfenergy.jl")
include("supercell.jl")
include("transform.jl")
include("mesh.jl")
include("spectrum.jl")
# include("coupler.jl")
include("apply.jl")
include("show.jl")
include("convert.jl")
# include("ket.jl")
# include("KPM.jl")
# include("effective.jl")
# include("greens.jl")

# Solvers
include("solvers/eigensolvers.jl")
include("solvers/greensolvers.jl")

# Presets
include("presets/lattices.jl")
include("presets/regions.jl")
# include("presets/hamiltonians.jl")


include("precompile.jl")

end