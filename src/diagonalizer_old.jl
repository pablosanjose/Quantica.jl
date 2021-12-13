#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizeMethod end

struct HamiltonianBlochFunctor{H<:Union{Hamiltonian,ParametricHamiltonian},A<:AbstractMatrix,M}
    h::H
    matrix::A
    mapping::M
end

(f::HamiltonianBlochFunctor)(vertex) = bloch!(f.matrix, f.h, map_phiparams(f.mapping, vertex))

struct Diagonalizer{M<:AbstractDiagonalizeMethod,F,O<:Union{OrbitalStructure,Missing}}
    method::M
    matrixf::F         # functor or function matrixf(φs) that produces matrices to be diagonalized
    orbstruct::O       # store structure of original Hamiltonian if available (to allow unflattening eigenstates)
    perm::Vector{Int}  # reusable permutation vector
    perm´::Vector{Int} # reusable permutation vector
end

struct NoUnflatten end

"""
    diagonalizer(h::Union{Hamiltonian,ParametricHamiltonian}; method = LinearAlgebraPackage(), mapping = missing)

Build a `d::Diagonalizer` object that, when called as `d(φs)` , uses the specified
diagonalization `method` to produce the sorted eigenpairs `(εs, ψs)` of `h` at Bloch
phases/parameters given by `mapping`. See `bandstructure` for further details.

A 0D Hamiltonian `h` also supports `d = diagonalizer(h)`. In this case `d` can be called
with no arguments and gives the same information as `spectrum`, `d() ≈ Tuple(spectrum(h))`.

# Examples
```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(1));

julia> d = diagonalizer(h)
Diagonalizer with method : LinearAlgebraPackage{NamedTuple{(), Tuple{}}}

julia> d((0, 0)) |> first
2-element Vector{Float64}:
 -3.0
  3.0

julia> h = wrap(h); d = diagonalizer(h);

julia> d() .≈ Tuple(spectrum(h))
(true, true)
```

# See also
    `bandstructure`, `spectrum`
"""
function diagonalizer(h::Union{Hamiltonian,ParametricHamiltonian}; method = LinearAlgebraPackage(), mapping = missing)
    matrix = similarmatrix(h, method_matrixtype(method, h))
    matrixf = HamiltonianBlochFunctor(h, matrix, mapping)
    perm = Vector{Int}(undef, size(matrix, 2))
    orbstruct = parent(h).orbstruct
    return Diagonalizer(method, matrixf, orbstruct, perm, copy(perm))
end

function diagonalizer(matrixf::Function, dimh; method = LinearAlgebraPackage())
    perm = Vector{Int}(undef, dimh)
    return Diagonalizer(method, matrixf, missing, perm, copy(perm))
end

@inline function (d::Diagonalizer)(φs, ::NoUnflatten)
    ϵ, ψ = diagonalize(d.matrixf(φs), d.method)
    issorted(ϵ, by = real) || sorteigs!(ϵ, ψ, d)
    fixphase!(ψ)
    return ϵ, ψ
end

@inline (d::Diagonalizer)(n::NoUnflatten) = d((), n)

function (d::Diagonalizer)(φs)
    ϵ, ψ = d(φs, NoUnflatten())
    ψ´ = unflatten_orbitals_or_reinterpret(ψ, d.orbstruct)
    return ϵ, ψ´
end

@inline (d::Diagonalizer)() = d(())

function sorteigs!(ϵ::AbstractVector, ψ::AbstractMatrix, d)
    p, p´ = d.perm, d.perm´
    resize!(p, length(ϵ))
    resize!(p´, length(ϵ))
    sortperm!(p, ϵ, by = real, alg = Base.DEFAULT_UNSTABLE)
    Base.permute!!(ϵ, copy!(p´, p))
    Base.permutecols!!(ψ, copy!(p´, p))
    return ϵ, ψ
end

fixphase!(ψ::AbstractArray{<:Real}) = ψ
function fixphase!(ψ::AbstractArray{<:Complex})
    for col in eachcol(ψ)
        col ./= cis(angle(mean(col)))
    end
    return ψ
end

function Base.show(io::IO, d::Diagonalizer)
    print(io, "Diagonalizer with method : $(summary(d.method))")
end

## Diagonalize methods ##

checkloaded(package::Symbol) = isdefined(Main, package) ||
    throw(ArgumentError("Package $package not loaded, need to be `using $package`."))

## LinearAlgebra ##
struct LinearAlgebraPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

LinearAlgebraPackage(; kw...) = LinearAlgebraPackage(values(kw))

function diagonalize(matrix, method::LinearAlgebraPackage)
    ϵ, ψ = eigen!(matrix; (method.kw)...)
    return ϵ, ψ
end

## Arpack ##
struct ArpackPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

ArpackPackage(; kw...) = (checkloaded(:Arpack); ArpackPackage(values(kw)))

function diagonalize(matrix, method::ArpackPackage)
    ϵ, ψ = Main.Arpack.eigs(matrix; (method.kw)...)
    return ϵ, ψ
end

## ArnoldiMethod ##
struct ArnoldiMethodPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

ArnoldiMethodPackage(; kw...) = (checkloaded(:ArnoldiMethod); ArnoldiMethodPackage(values(kw)))

function diagonalize(matrix, method::ArnoldiMethodPackage)
    p = first(Main.ArnoldiMethod.partialschur(matrix; (method.kw)...))
    ϵ, ψ = p.eigenvalues, p.Q
    return ϵ, ψ
end

## IterativeSolvers ##

struct KrylovKitPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

KrylovKitPackage(; kw...) = (checkloaded(:KrylovKit); KrylovKitPackage(values(kw)))

function diagonalize(matrix::AbstractMatrix{M}, method::KrylovKitPackage) where {M}
    ishermitian(matrix) || throw(ArgumentError("Only Hermitian matrices supported with KrylovKitPackage for the moment"))
    origin = get(method.kw, :origin, 0.0)
    howmany = get(method.kw, :howmany, 1)
    kw´ = Base.structdiff(method.kw, NamedTuple{(:origin,:howmany)}) # Remove origin option
    lmap = shiftandinvert(matrix, origin)
    T = eltypevec(matrix)
    n = size(matrix, 2)
    x0 = rand(T, n)
    ϵ, ψ, _ = Main.KrylovKit.eigsolve(x -> lmap * x, x0, howmany, :LM; kw´...)

    ϵ´ = invertandshift(ϵ, origin)
    resize!(ϵ´, howmany)

    dimh = size(matrix, 2)
    ψ´ = Matrix{T}(undef, dimh, howmany)
    for i in 1:howmany
        copyslice!(ψ´, CartesianIndices((1:dimh, i:i)), ψ[i], CartesianIndices((1:dimh,)))
    end

    return ϵ´, ψ´
end

### matrix/vector types

similarmatrix(h, method::AbstractDiagonalizeMethod) = similarmatrix(h, method_matrixtype(method, h))

method_matrixtype(::LinearAlgebraPackage, h) = Matrix{blockeltype(h)}
method_matrixtype(::AbstractDiagonalizeMethod, h) = flatten