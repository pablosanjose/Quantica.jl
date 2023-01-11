############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!, UniformScaling
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica,
      AbstractGreenSolver, AppliedGreenSolver, ParametricOnsiteTerm, ParametricHoppingTerm,
      AbstractSelfEnergySolver, RegularSelfEnergySolver, ExtendedSelfEnergySolver,
      LatticeSlice, Hamiltonian, ParametricHamiltonian, AbstractHamiltonian, SublatBlockStructure,
      HybridSparseBlochMatrix, lattice, zerocell, SVector, sanitize_SVector, siteselector,
      foreach_cell, foreach_site, store_diagonal_ptrs, cell
import Quantica: call!, apply, SelfEnergy

include("greensolvers/nosolver.jl")
# include("greensolvers/sparselu.jl")
# include("greensolvers/schur.jl")
# include("greensolvers/bands.jl")

end # module

const GS = GreenSolvers

