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
using Quantica: Quantica, Bloch, ensureloaded, AbstractHamiltonian, bloch, call!, orbtype,
    blocktype, OrbitalStructure, orbitalstructure, SVector, SMatrix

export eigensolver

#endregion

############################################################################################
# EigensolverBackend's
#region

abstract type EigensolverBackend end

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

#endregion

############################################################################################
# Eigensolver
#region

const Spectrum{E<:Complex,S} = Eigen{S,E,Matrix{S},Vector{E}}

struct Eigensolver{T,L,S<:Spectrum}
    solver::FunctionWrapper{S,Tuple{SVector{L,T}}}
end

(s::Eigensolver{<:Any,L})(φs::Vararg{<:Any,L}) where {L} = s.solver(SVector(φs))
(s::Eigensolver{<:Any,L})(φs::SVector{L}) where {L} = s.solver(φs)
(s::Eigensolver{<:Any,L})(φs...) where {L} =
    throw(ArgumentError("Eigensolver call requires $L parameters/Bloch phases"))

# default Eigensolver uses the LinearAlgebra backend
Eigensolver{T,L}(bloch::Bloch, mapping = missing) where {T,L} =
    Eigensolver{T,L}(LinearAlgebra(), bloch, mapping)

function Eigensolver{T,L}(backend::EigensolverBackend, bloch::Bloch, mapping = missing) where {T,L}
    E = complex(eltype(blocktype(bloch)))
    S = orbtype(bloch)
    solver = mappedsolver(backend, bloch, mapping)
    return Eigensolver(FunctionWrapper{Spectrum{E,S},Tuple{SVector{L,T}}}(solver))
end

mappedsolver(backend::EigensolverBackend, bloch, ::Missing) =
    φs -> backend(call!(bloch, φs))
mappedsolver(backend::EigensolverBackend, bloch, mapping) =
    φs -> backend(call!(bloch, mapping(Tuple(φs)...)))

Spectrum(args...) = Eigen(args...)
Spectrum(evals::AbstractVector, evecs::AbstractVector{<:AbstractVector}) =
    Spectrum(evals, hcat(evecs...))
Spectrum(evals::AbstractVector{<:Real}, evecs::AbstractMatrix) =
    Spectrum(complex.(evals), evecs)

#endregion

end # module

const ES = Eigensolvers

#endregion