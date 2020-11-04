#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizeMethod end

struct Diagonalizer{M<:AbstractDiagonalizeMethod,T<:Real}
    method::M
    minoverlap::T
end

diagonalizer(method, minoverlap) = Diagonalizer(method, float(minoverlap))

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

# # Type of states in bandstructure. Should be Matrix view, because eigvecs are dense and we need to support degeneracies
# method_matviewtype(::AbstractDiagonalizeMethod, ::AbstractMatrix{T}) where {T} = subarray_matrix_type(orbitaltype(T))

# subarray_matrix_type(::Type{M}) where {M} = typeof(view(Matrix{eltype(M)}(undef, 2, 2), :, 1:0))