#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizeMethod end

struct HamiltonianBlochFunctor{H<:Union{Hamiltonian,ParametricHamiltonian},A,M}
    h::H
    matrix::A
    mapping::M
end

(f::HamiltonianBlochFunctor)(vertex) = bloch!(f.matrix, f.h, map_phiparams(f.mapping, vertex))

struct Diagonalizer{M<:AbstractDiagonalizeMethod,T<:Real,F}
    method::M
    minoverlap::T
    perm::Vector{Int} # reusable permutation vector
    matrixf::F        # functor or function matrixf(φs) that produces matrices to be diagonalized
end

diagonalizer(matrixf, matrix, method, minoverlap = 0) =
    Diagonalizer(method, float(minoverlap), Vector{Int}(undef, size(matrix, 2)), matrixf)

@inline function (d::Diagonalizer)(φs)
    ϵ, ψ = diagonalize(d.matrixf(φs), d.method)
    issorted(ϵ, by = real) || sorteigs!(d.perm, ϵ, ψ)
    return ϵ, ψ
end

function sorteigs!(perm, ϵ::AbstractVector, ψ::AbstractMatrix)
    resize!(perm, length(ϵ))
    p = sortperm!(perm, ϵ, by = real, alg = Base.DEFAULT_UNSTABLE)
    permute!(ϵ, p)
    Base.permutecols!!(ψ, p)
    return ϵ, ψ
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
    ϵ, ψ = Main.ArnoldiMethod.partialschur(matrix; (method.kw)...)
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