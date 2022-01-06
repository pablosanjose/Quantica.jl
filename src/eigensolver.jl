############################################################################################
# Eigensolver and Spectrum
#region

(s::Eigensolver{<:Any,L})(φs::Vararg{<:Any,L}) where {L} = s.solver(SVector(φs))
(s::Eigensolver{<:Any,L})(φs::SVector{L}) where {L} = s.solver(φs)
(s::Eigensolver{<:Any,L})(φs...) where {L} =
    throw(ArgumentError("Eigensolver call requires $L parameters/Bloch phases"))

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
#   Eigensolvers.EigensolverBackend. We `import` instead of `using` to avoid collisions
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
#   Strategy: combine a EigensolverBackend with a Hamiltonian (or Bloch) to produce an
#   Eigensolver, is essentially a FunctionWrapper from an SVector to a Spectrum === Eigen
#   An EigensolverBackend is defined by a set of kwargs for the eigensolver and a set of
#   methods AbstractMatrix -> Eigen associated to that EigensolverBackend
#region

module Eigensolvers

using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Eigen, I, lu, ldiv!
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, Bloch, Spectrum, ensureloaded, AbstractHamiltonian, call!,
    flatten, spectrumtype, SVector, SMatrix
import Quantica: bloch, Eigensolver

#endregion

############################################################################################
# Types
#region

abstract type EigensolverBackend end

function Quantica.Eigensolver{T,L}(backend::EigensolverBackend, bloch::Bloch, mapping = missing) where {T,L}
    S = spectrumtype(bloch)
    solver = mappedsolver(backend, bloch, mapping)
    return Eigensolver(FunctionWrapper{S,Tuple{SVector{L,T}}}(solver))
end

Quantica.Eigensolver{T,L}(backend::EigensolverBackend, h::AbstractHamiltonian, mapping = missing) where {T,L} =
    Eigensolver{T,L}(backend, bloch(h, backend), mapping)

mappedsolver(backend::EigensolverBackend, bloch, ::Missing) =
    φs -> backend(call!(bloch, φs))
mappedsolver(backend::EigensolverBackend, bloch, mapping) =
    φs -> backend(call!(bloch, mapping(Tuple(φs)...)))

#endregion

############################################################################################
# EigensolverBackend's
#region

## Fallbacks

Quantica.bloch(h::AbstractHamiltonian, ::EigensolverBackend) = bloch(h)

(b::EigensolverBackend)(m) =
    throw(ArgumentError("The eigensolver backend $(typeof(b)) is not defined to work on $(typeof(m))"))

#### LinearAlgebra #####

struct LinearAlgebra{K} <: EigensolverBackend
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

struct Arpack{K} <: EigensolverBackend
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

struct KrylovKit{P,K} <: EigensolverBackend
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

#### ArnoldiMethod #####

struct ArnoldiMethod{K} <: EigensolverBackend
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

#### ShiftInvertSparse ####

struct ShiftInvertSparse{T,E<:EigensolverBackend} <: EigensolverBackend
    origin::T
    eigensolver::E
end

function ShiftInvertSparse(e::EigensolverBackend, origin)
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

############################################################################################
# show
#region

function Base.show(io::IO, s::EigensolverBackend)
    i = get(io, :indent, "")
    print(io, i, summary(s))
end

Base.summary(s::EigensolverBackend) =
    "EigensolverBackend ($(Base.nameof(typeof(s))))"

function Base.show(io::IO, s::Eigensolver)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "  ")
    print(io, i, summary(s), "\n")
end

Base.summary(::Eigensolver{T,L}) where {T,L} =
    "Eigensolver{$T,$L}: Eigensolver over an $L-dimensional parameter manifold of type $T"

#endregion

end # module

const ES = Eigensolvers

#endregion