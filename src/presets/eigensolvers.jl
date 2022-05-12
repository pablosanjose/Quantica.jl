############################################################################################
# Eigensolvers module
#   An AbstractEigensolver is defined by a set of kwargs for the eigensolver and a set of
#   methods AbstractMatrix -> Eigen associated to that AbstractEigensolver
#region

module EigensolverPresets

using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Eigen, I, lu, ldiv!
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, AbstractEigensolver, ensureloaded, SVector, SMatrix,
                sanitize_eigen, call!_output

#endregion

############################################################################################
# AbstractEigensolvers
#region

## Fallbacks

(s::AbstractEigensolver)(mat) =
    throw(ArgumentError("The eigensolver backend $(typeof(s)) is not defined to work on $(typeof(mat))"))

# an alias of h's call! output makes apply call! conversion a no-op, see apply.jl
input_matrix(::AbstractEigensolver, h) = call!_output(h)

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigensolver
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

struct Arpack{K} <: AbstractEigensolver
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

#### KrylovKit #####

struct KrylovKit{P,K} <: AbstractEigensolver
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

struct ArnoldiMethod{K} <: AbstractEigensolver
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

#### ShiftInvertSparse ####

struct ShiftInvertSparse{T,E<:AbstractEigensolver} <: AbstractEigensolver
    origin::T
    eigensolver::E
end

function ShiftInvertSparse(e::AbstractEigensolver, origin)
    ensureloaded(:LinearMaps)
    return ShiftInvertSparse(origin, e)
end

function (solver::ShiftInvertSparse)(mat::AbstractSparseMatrix{T}) where {T<:Number}
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

const EP = EigensolverPresets

#endregion