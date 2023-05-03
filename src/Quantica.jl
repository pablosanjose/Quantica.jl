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
using QuadGK

using Infiltrator # debugging

export sublat, bravais_matrix, lattice, sites, supercell,
       hopping, onsite, @onsite, @hopping, @onsite!, @hopping!, neighbors,
       siteselector, hopselector,
       hamiltonian,
       flat, unflat, wrap, transform, translate, combine,
       spectrum, energies, states, bands, subbands, slice,
       greenfunction, selfenergy, attach, contact, cellsites,
       plotlattice, plotlattice!, plotbands, plotbands!, qplot, qplot!,
       conductance, josephson, ldos, current

export LatticePresets, LP, RegionPresets, RP, HamiltonianPresets, HP
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
include("docstrings.jl")

# API
include("specialmatrices.jl")
include("selectors.jl")
include("lattice.jl")
include("slices.jl")
include("models.jl")
include("hamiltonian.jl")
include("supercell.jl")
include("transform.jl")
include("mesh.jl")
include("spectrum.jl")
include("greenfunction.jl")
include("observables.jl")
# Plumbing
include("apply.jl")
include("show.jl")
include("convert.jl")
include("sanitizers.jl")


# Solvers
include("solvers/eigensolvers.jl")
include("solvers/greensolvers.jl")

# Presets
include("presets/regions.jl")
include("presets/lattices.jl")
include("presets/hamiltonians.jl")

# include("precompile.jl")

# Extension stubs for QuanticaMakieExt
function plotlattice end
function plotlattice! end
function plotbands end
function plotbands! end
function qplot end
function qplot! end

end