############################################################################################
# AppliedEigensolver and Spectrum
#region

(s::AppliedEigensolver{<:Any,L})(φs::Vararg{<:Any,L}) where {L} = s.solver(SVector(φs))
(s::AppliedEigensolver{<:Any,L})(φs::SVector{L}) where {L} = s.solver(φs)
(s::AppliedEigensolver{<:Any,L})(φs...) where {L} =
    throw(ArgumentError("AppliedEigensolver call requires $L parameters/Bloch phases"))

Spectrum(evals, evecs) = Eigen(sorteigs!(evals, evecs)...)
Spectrum(evals::AbstractVector, evecs::AbstractVector{<:AbstractVector}) =
    Spectrum(evals, hcat(evecs...))
Spectrum(evals::AbstractVector{<:Real}, evecs::AbstractMatrix) =
    Spectrum(complex.(evals), evecs)

function sorteigs!(ϵ::AbstractVector, ψ::AbstractMatrix)
    p = Vector{Int}(undef, length(ϵ))
    p´ = similar(p)
    sortperm!(p, ϵ, by = real, alg = Base.DEFAULT_UNSTABLE)
    Base.permute!!(ϵ, copy!(p´, p))
    Base.permutecols!!(ψ, copy!(p´, p))
    return ϵ, ψ
end

#endregion

############################################################################################
# Dynamic package loader
#   This is in global Quantica scope to avoid name collisions between package and
#   Eigensolvers.AbstractEigensolver. We `import` instead of `using` to avoid collisions
#   between several backends
#region

function ensureloaded(package::Symbol)
    if !isdefined(Quantica, package)
        @warn("Required package $package not loaded. Loading...")
        eval(:(import $package))
    end
    return nothing
end

#endregion

############################################################################################
# Eigensolvers module
#   Strategy: apply an AbstractEigensolver to a Hamiltonian (or Bloch) to produce an
#   AppliedEigensolver, which is essentially a FunctionWrapper from an SVector to a 
#   Spectrum === Eigen. An AbstractEigensolver is defined by a set of kwargs for the
#   eigensolver and a set of methods AbstractMatrix -> Eigen associated to that
#   AbstractEigensolver
#region

module Eigensolvers

using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Eigen, I, lu, ldiv!
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, Bloch, Spectrum, AbstractEigensolver, AbstractHamiltonian, call!,
      ensureloaded, flatten, spectrumtype, SVector, SMatrix
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

function (backend::LinearAlgebra)(mat::AbstractMatrix{<:Number})
    ε, Ψ = Quantica.LinearAlgebra.eigen(mat; backend.kwargs...)
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

function (backend::Arpack)(mat::AbstractMatrix{<:Number})
    ε, Ψ, _ = Quantica.Arpack.eigs(mat; backend.kwargs...)
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

function (backend::KrylovKit)(mat)
    ε, Ψ, _ = Quantica.KrylovKit.eigsolve(mat, backend.params...; backend.kwargs...)
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

function (backend::ArnoldiMethod)(mat)
    pschur, _ = Quantica.ArnoldiMethod.partialschur(mat; backend.kwargs...)
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

function (backend::ShiftInvertSparse)(mat::AbstractSparseMatrix{T}) where {T<:Number}
    mat´ = mat - I*backend.origin
    F = lu(mat´)
    lmap = Quantica.LinearMaps.LinearMap{T}((x, y) -> ldiv!(x, F, y), size(mat)...;
        ismutating = true, ishermitian = false)
    spectrum = backend.eigensolver(lmap)
    @. spectrum.values = 1 / (spectrum.values) + backend.origin
    return spectrum
end

Quantica.bloch(h::AbstractHamiltonian, ::ShiftInvertSparse) = bloch(flatten(h))

#endregion

end # module

const ES = Eigensolvers

#endregion