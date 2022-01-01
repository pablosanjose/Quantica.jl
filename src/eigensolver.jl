############################################################################################
# Dynamic package loader
#region

# This is in global Quantica scope to avoid name collisions between package and
# Eigensolvers.EigensolverBackend. We `import` instead of `using` to avoid collisions
# between several backends
function ensureloaded(package::Symbol)
    if !isdefined(Quantica, package)
        @warn("Required package $package not loaded. Loading...")
        eval(:(import $package))
    end
    return nothing
end

# eigensolver(x...; kw...) = Eigensolvers.eigensolver(x...; kw...)

#endregion

############################################################################################
# Eigensolvers module
# Strategy: combine a EigensolverBackend with a Hamiltonian (or Bloch) to produce an
# Eigensolver, is essentially a FunctionWrapper from an SVector to a Spectrum === Eigen
# An EigensolverBackend is defined by a set of kwargs for the eigensolver and a set of
# methods AbstractMatrix -> Eigen associated to that EigensolverBackend
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

(b::EigensolverBackend)(m) = throw(ArgumentError("The eigensolver backend $(typeof(b)) is not defined to work on $(typeof(m))"))

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

############################################################################################
# ShiftInvert
#region

# function shiftinvert(matrix::SparseMatrixCSC{O}, origin)
#     cols, rows = size(matrix)
#     matrix´ = diagshift!(matrix, origin)
#     F = factorize(matrix´)
#     lmap = LinearMap{Tv}((x, y) -> ldiv!(x, F, y), cols, rows,
#                          ismutating = true, ishermitian = false)
#     return lmap
# end

# function diagshift!(matrix::SparseMatrixCSC, origin)
#     vals = nonzeros(matrix)
#     rowval = rowvals(matrix)
#     for col in 1:size(matrix, 2)
#         found_diagonal = false
#         for ptr in nzrange(matrix, col)
#             if col == rowval[ptr]
#                 found_diagonal = true
#                 vals[ptr] -= origin * I  # To respect non-scalar eltypes
#                 break
#             end
#         end
#         found_diagonal || throw(error("Sparse work matrix must include the diagonal."))
#     end
#     return matrix
# end

# function invertshift(ϵ::Vector{T}, origin) where {T}
#     ϵ´ = similar(ϵ, real(T))
#     ϵ´ .= real(inv.(ϵ) .+ (origin))  # Caution: we assume a real spectrum
#     return ϵ´
# end

#endregion


end # module

const ES = Eigensolvers

#endregion