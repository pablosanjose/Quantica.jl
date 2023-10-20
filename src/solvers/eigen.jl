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

## Fallbacks

(s::AbstractEigenSolver)(mat) =
    throw(ArgumentError("The eigensolver backend $(typeof(s)) is not defined to work on $(typeof(mat))"))

# an alias of h's call! output makes apply call! conversion a no-op, see apply.jl
input_matrix(::AbstractEigenSolver, h) = call!_output(h)

is_thread_safe(::AbstractEigenSolver) = true

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigenSolver
    kwargs::K
end

function LinearAlgebra(; kw...)
    return LinearAlgebra(kw)
end

function (solver::LinearAlgebra)(mat::AbstractMatrix{<:Number})
    ε, Ψ = Quantica.LinearAlgebra.eigen(mat; solver.kwargs...)
    return sanitize_eigen(ε, Ψ)
end

# LinearAlgebra.eigen doesn't like sparse Matrices as input, must convert
input_matrix(::LinearAlgebra, h) = Matrix(call!_output(h))

#### Arpack #####

struct Arpack{K} <: AbstractEigenSolver
    kwargs::K
end

function Arpack(; kw...)
    ensureloaded(:Arpack)
    return Arpack(kw)
end

function (solver::Arpack)(mat::AbstractMatrix{<:Number})
    ε, Ψ, _ = Quantica.Arpack.eigs(mat; solver.kwargs...)
    return sanitize_eigen(ε, Ψ)
end

# See https://github.com/JuliaLinearAlgebra/Arpack.jl/issues/86
is_thread_safe(::Arpack) = false

#### KrylovKit #####

struct KrylovKit{P,K} <: AbstractEigenSolver
    params::P
    kwargs::K
end

function KrylovKit(params...; kw...)
    ensureloaded(:KrylovKit)
    return KrylovKit(params, kw)
end

function (solver::KrylovKit)(mat)
    ε, Ψ, _ = Quantica.KrylovKit.eigsolve(mat, solver.params...; solver.kwargs...)
    return sanitize_eigen(ε, Ψ)
end

#### ArnoldiMethod #####

struct ArnoldiMethod{K} <: AbstractEigenSolver
    kwargs::K
end

function ArnoldiMethod(; kw...)
    ensureloaded(:ArnoldiMethod)
    return ArnoldiMethod(kw)
end

function (solver::ArnoldiMethod)(mat)
    pschur, _ = Quantica.ArnoldiMethod.partialschur(mat; solver.kwargs...)
    ε, Ψ = Quantica.ArnoldiMethod.partialeigen(pschur)
    return sanitize_eigen(ε, Ψ)
end

#### ShiftInvert ####

struct ShiftInvert{T,E<:AbstractEigenSolver} <: AbstractEigenSolver
    eigensolver::E
    origin::T
    function ShiftInvert{T,E}(eigensolver, origin) where {T,E<:AbstractEigenSolver}
        ensureloaded(:LinearMaps)
        return new(eigensolver, origin)
    end
end

ShiftInvert(eigensolver::E, origin::T) where {E,T} = ShiftInvert{T,E}(eigensolver, origin)

function (solver::ShiftInvert)(mat::AbstractSparseMatrix{T}) where {T<:Number}
    mat´ = mat - I * solver.origin
    F = lu(mat´)
    lmap = Quantica.LinearMaps.LinearMap{T}((x, y) -> ldiv!(x, F, y), size(mat)...;
        ismutating = true, ishermitian = false)
    eigen = solver.eigensolver(lmap)
    @. eigen.values = 1 / (eigen.values) + solver.origin
    return eigen
end

#endregion

end # module

const ES = EigenSolvers

#endregion
