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

eigensolver(x...; kw...) = Eigensolvers.eigensolver(x...; kw...)

#endregion

############################################################################################
# Eigensolvers module
#region

module Eigensolvers

using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Eigen
using SparseArrays: SparseMatrixCSC
using Quantica: Quantica, Bloch, ensureloaded, AbstractHamiltonian, bloch, call!, orbtype,
    blocktype, OrbitalStructure, orbitalstructure, SVector, SMatrix

export eigensolver

#endregion

############################################################################################
# Spectrum (alias of Eigen)
#region

const Spectrum{E,S} = Eigen{S,E,Matrix{S},Vector{E}}
Spectrum(x...) = Eigen(x...)

#endregion

############################################################################################
# EigensolverBackend's
#region

abstract type EigensolverBackend end

#### Arpack #####

struct Arpack{F} <: EigensolverBackend
    spectrum::F
end

function Arpack(; sigma = 0.0, nev = 6, kw...)
    ensureloaded(:Arpack)
    function spectrum(mat::AbstractMatrix{<:Number})
        ε, Ψ, _ = Quantica.Arpack.eigs(mat; sigma, nev, kw...)
        return Spectrum(ε, Ψ)
    end
    function spectrum(::AbstractMatrix{<:SMatrix})
        throw(ArgumentError("Arpack only admits scalar eltypes. Try flattening your Hamiltonian."))
    end
    return Arpack(spectrum)
end

eigensolver(s::Arpack, ::OrbitalStructure{T}) where {T<:Number} =
    Eigensolver{Spectrum{complex(T),T},SparseMatrixCSC{T,Int}}(s.εΨ)

eigensolver(::Arpack, ::OrbitalStructure) =
    throw(ArgumentError("Arpack only admits scalar eltypes. Try flattening your Hamiltonian."))

#### KrylovKit #####

struct KrylovKit{F} <: EigensolverBackend
    spectrum::F
end

function KrylovKit(; howmany = 6, which = :LM, kw...)
    ensureloaded(:KrylovKit)
    if which isa Number # use shift-invert
        return KrylovKitShiftInvert(which; howmany, kw...)
    else
        function spectrum(mat::AbstractMatrix{<:Number})
            ε, Ψ, _ = Quantica.Arpack.eigs(mat; sigma, nev, kw...)
            return Spectrum(ε, Ψ)
        end
        return KrylovKit(spectrum)
    end
end


#endregion

############################################################################################
# Eigensolver
#region

# The idea is that bandstructure and spectrum take (AbstractHamiltonian, ::EigensolverBackend)
# and transform that into (::Bloch, ::EigensolverBackend), and then into (::Eigensolver)
# which is the complex function that has a simple type but contains the Hamiltonian/Bloch.
# Eigensolver only FunctionWraps the relevant solver method once it knows what eltype the
# eigenstates will have (eltype(bloch.output))

struct Eigensolver{T,L,S<:Spectrum}
    solver::FunctionWrapper{S,Tuple{SVector{L,T}}}
end

function Eigensolver{T,L}(s::EigensolverBackend, b::Bloch, mapping = missing) where {T,L}
    E = complex(eltype(blocktype(b)))
    S = orbtype(b)
    solver = mappedsolver(s, b, mapping)
    return Eigensolver(FunctionWrapper{Spectrum{E,S},Tuple{SVector{L,T}}}(solver))
end
mappedsolver(s, b, ::Missing) = φs -> s.spectrum(call!(b, φs))
mappedsolver(s, b, mapping::Function) = φs -> s.spectrum(call!(b, mapping(Tuple(φs)...)))

(s::Eigensolver)(m::AbstractMatrix) = s.solver(m)
(s::Eigensolver{<:Any,<:Number})(h::AbstractHamiltonian, φs...; flatten = true, kw...) =
    flatten ? s(bloch(flatten(h), φs; kw...)) : s(bloch(h, φs; kw...))

(s::Eigensolver)(b::Bloch, φs...; kw...) = s.solver(call!(b, φs; kw...))

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