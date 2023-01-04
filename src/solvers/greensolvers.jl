############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica,
      AbstractGreenSolver, AppliedGreenSolver, DecoupledGreenSolver,
      AbstractSelfEnergySolver, SelfEnergySolver, ExtendedSelfEnergySolver,
      LatticeSlice, Hamiltonian, ParametricHamiltonian, AbstractHamiltonian, SublatBlockStructure,
      HybridSparseMatrixCSC, lattice, zerocell, SVector, sanitize_SVector, siteselector,
      foreach_cell, foreach_site, store_diagonal_ptrs, cell
import Quantica: call!, apply, SelfEnergy

export default_green_solver

const AbstractHamiltonian0D{T,E,B} = AbstractHamiltonian{T,E,0,B}
const AbstractHamiltonian1D{T,E,B} = AbstractHamiltonian{T,E,1,B}

include("greensolvers/diagonal.jl")
# include("greensolvers/sparselu.jl")
# include("greensolvers/schur.jl")
# include("greensolvers/bands.jl")

default_green_solver(::AbstractHamiltonian0D) = SparseLU()
default_green_solver(::AbstractHamiltonian1D) = Schur()
default_green_solver(::AbstractHamiltonian) = Bands()

end # module

const GS = GreenSolvers

