############################################################################################
# Eigensolvers module
#   Strategy: apply an AbstractEigensolver to a Hamiltonian (or Bloch) to produce an
#   AppliedEigensolver, which is essentially a FunctionWrapper from an SVector to a
#   Spectrum === Eigen. An AbstractEigensolver is defined by a set of kwargs for the
#   eigensolver and a set of methods AbstractMatrix -> Eigen associated to that
#   AbstractEigensolver
#region

module EigensolverPresets

using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Eigen, I, lu, ldiv!
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, Bloch, Spectrum, AbstractEigensolver, AbstractHamiltonian, call!,
      ensureloaded, spectrumtype, SVector, SMatrix
import Quantica: bloch

#endregion

############################################################################################
# AbstractEigensolvers
#region

## Fallbacks

Quantica.bloch(h::AbstractHamiltonian, ::AbstractEigensolver) = bloch(h)

(b::AbstractEigensolver)(m) =
    throw(ArgumentError("The eigensolver backend $(typeof(b)) is not defined to work on $(typeof(m))"))

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigensolver
    kwargs::K
end

function LinearAlgebra(; kw...)
    return LinearAlgebra(kw)
end

function (solver::LinearAlgebra)(mat::AbstractMatrix{<:Number})
    ε, Ψ = Quantica.LinearAlgebra.eigen(mat; solver.kwargs...)
    return Spectrum(ε, Ψ)
end

Quantica.bloch(h::AbstractHamiltonian, ::LinearAlgebra) = bloch(flatten(h), Matrix)

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
    return Spectrum(ε, Ψ)
end

Quantica.bloch(h::AbstractHamiltonian, ::Arpack) = bloch(flatten(h))

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
    return Spectrum(ε, Ψ)
end

# KrylovKit gives strange error with SMatrix eltypes
Quantica.bloch(h::AbstractHamiltonian, ::KrylovKit) = bloch(flatten(h))

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
    return Spectrum(ε, Ψ)
end

# ArnoldiMethod complains of missing eps method with SMatrix eltypes
Quantica.bloch(h::AbstractHamiltonian, ::ArnoldiMethod) = bloch(flatten(h))

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
    spectrum = solver.eigensolver(lmap)
    @. spectrum.values = 1 / (spectrum.values) + solver.origin
    return spectrum
end

Quantica.bloch(h::AbstractHamiltonian, ::ShiftInvertSparse) = bloch(flatten(h))

#endregion

end # module

const EP = EigensolverPresets

#endregion