############################################################################################
# EigenSolvers module
#   An AbstractEigenSolver is defined by a set of kwargs for the eigensolver and a set of
#   methods AbstractMatrix -> Eigen associated to that AbstractEigenSolver
#region

module EigenSolvers

using FunctionWrappers: FunctionWrapper
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Eigen, I, lu, ldiv!
using Quantica: Quantica, AbstractEigenSolver, ensureloaded, SVector, SMatrix,
                sanitize_eigen, call!_output

#endregion

############################################################################################
# AbstractEigensolvers
#region

# Extensions should add methods to get_eigen for their specific solver. It should return a
# tuple (ε, Ψ) where ε is the eigenvalues and Ψ the eigenvectors.
(s::AbstractEigenSolver)(mat) = sanitize_eigen(get_eigen(s, mat))

## Fallbacks

get_eigen(s::AbstractEigenSolver, _) =
    argerror("The eigensolver backend for $(typeof(s)) is not loaded (did you do `using $(solverbackends(s))`?)")

# unless otherwise specified, the backend package matches the solver name
solverbackends(s::AbstractEigenSolver) = string(nameof(typeof(s)))

# an alias of h's call! output makes apply call! conversion a no-op, see apply.jl
input_matrix(::AbstractEigenSolver, h) = call!_output(h)

is_thread_safe(::AbstractEigenSolver) = true

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigenSolver
    kwargs::K
end

LinearAlgebra(; kw...) = LinearAlgebra(NamedTuple(kw))

# no extension for LinearAlgebra, it is a strong dependency
get_eigen(solver::LinearAlgebra, mat::AbstractMatrix{<:Number}) =
    Quantica.LinearAlgebra.eigen(mat; solver.kwargs...)

# LinearAlgebra.eigen doesn't like sparse Matrices as input, must convert
input_matrix(::LinearAlgebra, h) = Matrix(call!_output(h))

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
