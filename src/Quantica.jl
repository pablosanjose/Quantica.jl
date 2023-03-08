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

using Infiltrator # debugging

export sublat, lattice, supercell, bravais_matrix,
       hopping, onsite, @onsite, @hopping, @onsite!, @hopping!, neighbors,
       hamiltonian, call!,
       flat, unflat, wrap, transform, transform!, translate, translate!,
       spectrum, energies, states, bands, mesh, subbands, slice,
       greenfunction, attach, contact, cellsites,
       plotlattice, plotlattice!, qplot

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
export ishermitian, tr, I
export ftuple

# Types
include("types.jl")

# Preamble
include("iterators.jl")
include("builders.jl")
include("tools.jl")

# API
include("specialmatrices.jl")
include("selector.jl")
include("lattice.jl")
include("slice.jl")
include("model.jl")
include("hamiltonian.jl")
include("supercell.jl")
include("transform.jl")
include("mesh.jl")
include("spectrum.jl")
include("greenfunction.jl")
# Plumbing
include("apply.jl")
include("show.jl")
include("convert.jl")
include("sanitize.jl")


# Solvers
include("solvers/eigensolvers.jl")
include("solvers/greensolvers.jl")

# Presets
include("presets/lattices.jl")
include("presets/regions.jl")
# include("presets/hamiltonians.jl")

# include("precompile.jl")


# Extension stubs for QuanticaMakieExt
function plotlattice end
function plotlattice! end
function qplot end

end