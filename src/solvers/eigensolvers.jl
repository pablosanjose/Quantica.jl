# extension stubs for EigenSolvers
get_eigen(s::AbstractEigenSolver, _...) =
    argerror("The eigensolver backend for EigenSolvers.$(nameof(typeof(s))) is not loaded. Did you first do `using $(ES.solverbackends(s))`?")

# fallback without workspace
get_eigen(solver::AbstractEigenSolver, mat, workspace::Missing) = get_eigen(solver, mat)

############################################################################################
# EigenSolvers module
#   An AbstractEigenSolver is defined by a set of kwargs for the eigensolver and a set of
#   methods AbstractMatrix -> Eigen associated to that AbstractEigenSolver
#region

module EigenSolvers

using FunctionWrappers: FunctionWrapper
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using FastLapackInterface: HermitianEigenWs, EigenWs
using Quantica: Eigen, I, lu, ldiv!
using Quantica: Quantica, AbstractEigenSolver, SVector, SMatrix,
                sanitize_eigen, call!_output, argerror, ishermitian, isdiag
import Quantica: get_eigen

#endregion

############################################################################################
# AbstractEigensolvers
#region

# Extensions should add methods to get_eigen for their specific solver. It should return a
# tuple (ε, Ψ) where ε is the eigenvalues and Ψ the eigenvectors.
(s::AbstractEigenSolver)(mat, workspace = missing) = sanitize_eigen(Quantica.get_eigen(s, mat, workspace)...)

## Fallbacks

# unless otherwise specified, the backend package matches the solver name
solverbackends(s::AbstractEigenSolver) = string(nameof(typeof(s)))

# an alias of h's call! output makes apply call! conversion a no-op, see apply.jl
# IMPORTANT: any solver that supports workspaces is allowed to overwrite the input matrix mat
# so it must implement an input_matrix method that returns a copy of call!_output(h)
input_matrix(::AbstractEigenSolver, h) = call!_output(h)

# workspace is used to store workspaces for solvers that support it, e.g. LinearAlgebra
eigen_workspace(::AbstractEigenSolver, mat, use_workspace::Val{false}) = missing
eigen_workspace(s::AbstractEigenSolver, mat, use_workspace::Val{true}) = eigen_workspace(s, mat)
eigen_workspace(::AbstractEigenSolver, mat) = missing

is_thread_safe(::AbstractEigenSolver) = true

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigenSolver
    kwargs::K
end

LinearAlgebra(; kw...) = LinearAlgebra(NamedTuple(kw))

# no extension for LinearAlgebra, it is a strong dependency
function Quantica.get_eigen(solver::LinearAlgebra, mat::AbstractMatrix{<:Number})
    ϵ, ψ = Quantica.LinearAlgebra.eigen(mat; solver.kwargs...)  # destructure: splatting an Eigen allocates
    return ϵ, ψ
end

# workspaces are only used without eigen kwargs (e.g. sortby), which they don't support
# mat is overwritten by this get_eigen method, but it is a copy of call!_output(h), so its safe
# the ishermitian call is negligible compared to eigen, and LinearAlgebra actually does that check internally
# the workspace tuple type is required to avoid ambiguity with the missing-workspace method
function Quantica.get_eigen(solver::LinearAlgebra, mat::AbstractMatrix{<:Number},
                   (hermitian_ws, general_ws)::Tuple{HermitianEigenWs,EigenWs})
    if !isempty(solver.kwargs) || isdiag(mat)   # isdiag: LinearAlgebra's no-LAPACK shortcut
        Quantica.get_eigen(solver, mat)
    elseif ishermitian(mat)
        copy.(Quantica.eigen_ws!(hermitian_ws, mat))
    else
        copy.(Quantica.eigen_ws!(general_ws, mat))
    end
end

# LinearAlgebra.eigen doesn't like sparse Matrices as input, must convert
input_matrix(::LinearAlgebra, h) = Matrix(call!_output(h))

# We use FastLapackInterface to create a workspace
eigen_workspace(::LinearAlgebra, mat) = (HermitianEigenWs(mat; vecs = true), EigenWs(mat))


#### Arpack #####

struct Arpack{K<:NamedTuple} <: AbstractEigenSolver
    kwargs::K
end

Arpack(; kw...) = Arpack(NamedTuple(kw))

# See https://github.com/JuliaLinearAlgebra/Arpack.jl/issues/86
is_thread_safe(::Arpack) = false

#### KrylovKit #####

struct KrylovKit{P<:Tuple,K<:NamedTuple} <: AbstractEigenSolver
    params::P
    kwargs::K
end

KrylovKit(params...; kw...) = KrylovKit(params, NamedTuple(kw))

#### ArnoldiMethod #####

struct ArnoldiMethod{K<:NamedTuple} <: AbstractEigenSolver
    kwargs::K
end

ArnoldiMethod(; kw...) = ArnoldiMethod(NamedTuple(kw))

#### ShiftInvert ####

struct ShiftInvert{T,E<:AbstractEigenSolver} <: AbstractEigenSolver
    eigensolver::E
    origin::T
end

solverbackends(s::ShiftInvert) = "LinearMaps"

#endregion

end # module

const ES = EigenSolvers

#endregion
