############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using Quantica: AbstractGreenSolver

struct SparseLU <:AbstractGreenSolver end

struct NoSolver <:AbstractGreenSolver end

end # module

const GS = GreenSolvers

include("greensolvers/nosolver.jl")
include("greensolvers/sparselu.jl")
# include("greensolvers/schur.jl")
# include("greensolvers/bands.jl")

