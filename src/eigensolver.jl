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

using Quantica: Quantica, ensureloaded, AbstractHamiltonian, bloch, call!

export eigensolver

struct Spectrum{E,S}
    energies::Vector{E}
    states::Matrix{S}
end

struct Eigensolver{S<:Spectrum,M<:AbstractMatrix}
    solver::FunctionWrapper{S,Tuple{M}}
end

(s::Eigensolver)(m::AbstractMatrix) = s.solver(m)
(s::Eigensolver)(h::AbstractHamiltonian, φs...; kw...) = s.solver(bloch(h, φs...; kw...))
(s::Eigensolver)(b::Bloch, φs...; kw...) = s.solver(call!(b, φs; kw...))


############################################################################################
# Arpack Backend
#region

abstract type EigensolverBackend end

struct Arpack{F} <: EigensolverBackend
    εΨ::F
end

function Arpack(; sigma = 0.0, nev = 6, kw...)
    ensureloaded(:Arpack)
    function εΨ(mat::AbstractMatrix{<:Number})
        ε, Ψ, _ = Quantica.Arpack.eigs(mat; sigma, nev, kw...)
        return ε, Ψ
    end
    return Arpack(εΨ)
end

function eigensolver(s::Arpack, h::AbstractHamiltonian)
    O = orbtype(h)
    return Eigensolver{O,SparseMatrixCSC{O}}(s.εΨ)
end

#endregion

end # module

const ES = Eigensolvers

#endregion