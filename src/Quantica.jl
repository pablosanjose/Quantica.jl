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
<<<<<<< HEAD
using QuadGK

using Infiltrator # debugging

export sublat, bravais_matrix, lattice, sites, supercell,
       hopping, onsite, @onsite, @hopping, @onsite!, @hopping!, plusadjoint, neighbors,
       siteselector, hopselector,
       hamiltonian,
       unflat, wrap, transform, translate, combine,
       spectrum, energies, states, bands, subdiv,
       greenfunction, selfenergy, attach, contact, cellsites,
       plotlattice, plotlattice!, plotbands, plotbands!, qplot, qplot!,
       conductance, josephson, ldos, current, transmission

export LatticePresets, LP, RegionPresets, RP, HamiltonianPresets, HP
export EigenSolvers, ES, GreenSolvers, GS
=======

using Compat # for use of argmin/argmax in bandstructure.jl

export sublat, bravais, lattice, dims, supercell, unitcell,
       hopping, onsite, @onsite!, @hopping!, @block!, parameters, siteselector, hopselector, nrange,
       sitepositions, siteindices, not,
       ket, ketmodel, randomkets, basiskets,
       hamiltonian, parametric, bloch, bloch!, similarmatrix,
       flatten, unflatten, orbitalstructure, wrap, transform!, combine,
       spectrum, bandstructure, diagonalizer, cuboid, isometric, splitbands,
       bands, vertices, minima, maxima, gapedge, gap, isinband,
       energies, states, degeneracy,
       momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
       greens, greensolver, Schur1D, proj_DACP, DACPdiagonaliser

export RegionPresets, RP, LatticePresets, LP, HamiltonianPresets, HP

export LinearAlgebraPackage, ArpackPackage, ArnoldiMethodPackage, KrylovKitPackage

>>>>>>> master
export @SMatrix, @SVector, SMatrix, SVector, SA
export ishermitian, tr, I, norm, dot, diag, det
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
<<<<<<< HEAD
include("bands.jl")
include("greenfunction.jl")
include("observables.jl")
# Plumbing
include("apply.jl")
include("show.jl")
=======
include("diagonalizer.jl")
include("bandstructure.jl")
include("KPM.jl")
include("DACP.jl")
include("effective.jl")
include("greens.jl")
>>>>>>> master
include("convert.jl")
include("sanitizers.jl")


# Solvers
include("solvers/eigen.jl")
include("solvers/green.jl")
include("solvers/selfenergy.jl")

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