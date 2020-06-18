#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizeMethod end

struct DiagonalizeHelper{S<:AbstractDiagonalizeMethod,C}
    method::S
    codiag::C
    minprojection::Float64
end

DiagonalizeHelper(method, codiag; minprojection = 0.5) =
    DiagonalizeHelper(method, codiag, minprojection)

## Diagonalize methods ##

defaultmethod(h::Union{Hamiltonian,ParametricHamiltonian,AbstractMatrix}) = LinearAlgebraPackage()

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

similarmatrix(h, ::LinearAlgebraPackage) = similarmatrix(h, Matrix{blockeltype(h)})

## Arpack ##
struct ArpackPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

ArpackPackage(; kw...) = (checkloaded(:Arpack); ArpackPackage(values(kw)))

function diagonalize(matrix, method::ArpackPackage)
    ϵ, ψ = Main.Arpack.eigs(matrix; (method.kw)...)
    return ϵ, ψ
end

similarmatrix(h, ::ArpackPackage) = similarmatrix(h, SparseMatrixCSC{blockeltype(h)})

## ArnoldiMethod ##
struct ArnoldiMethodPackage{K<:NamedTuple} <: AbstractDiagonalizeMethod
    kw::K
end

ArnoldiMethodPackage(; kw...) = (checkloaded(:ArnoldiMethod); ArnoldiMethodPackage(values(kw)))

function diagonalize(matrix, method::ArnoldiMethodPackage)
    ϵ, ψ = Main.ArnoldiMethod.partialschur(matrix; (method.kw)...)
    return ϵ, ψ
end

similarmatrix(h, ::ArnoldiMethodPackage) = similarmatrix(h, SparseMatrixCSC{blockeltype(h)})

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

similarmatrix(h, ::KrylovKitPackage) = similarmatrix(h, SparseMatrixCSC{blockeltype(h)})

#######################################################################
# shift and invert methods
#######################################################################

function shiftandinvert(matrix::AbstractMatrix{Tv}, origin) where {Tv}
    cols, rows = size(matrix)
    # Shift away from real axis to avoid pivot point error in factorize
    matrix´ = diagshift!(parent(matrix), origin + im)
    F = factorize(matrix´)
    lmap = LinearMap{Tv}((x, y) -> ldiv!(x, F, y), cols, rows,
                         ismutating = true, ishermitian = false)
    return lmap
end

function diagshift!(matrix::AbstractMatrix, origin)
    matrix´ = parent(matrix)
    vals = nonzeros(matrix´)
    rowval = rowvals(matrix´)
    for col in 1:size(matrix, 2)
        found_diagonal = false
        for ptr in nzrange(matrix´, col)
            if col == rowval[ptr]
                found_diagonal = true
                vals[ptr] -= origin * I  # To respect non-scalar eltypes
                break
            end
        end
        found_diagonal || throw(error("Sparse work matrix must include the diagonal. Possible bug in `similarmatrix`."))
    end
    return matrix
end

function invertandshift(ϵ::Vector{T}, origin) where {T}
    ϵ´ = similar(ϵ, real(T))
    ϵ´ .= real(inv.(ϵ) .+ (origin + im))  # Caution: we assume a real spectrum
    return ϵ´
end

#######################################################################
# resolve_degeneracies
#######################################################################
# Tries to make states continuous at crossings. Here, ϵ needs to be sorted
function resolve_degeneracies!(ϵ, ψ, ϕs, codiag)
    issorted(ϵ, by = real) || sorteigs!(codiag.perm, ϵ, ψ)
    hasapproxruns(ϵ, codiag.degtol) || return ϵ, ψ
    ranges, ranges´ = codiag.rangesA, codiag.rangesB
    resize!(ranges, 0)
    pushapproxruns!(ranges, ϵ, 0, codiag.degtol) # 0 is an offset
    for n in codiag.matrixindices
        v = codiag.cmatrixf(ϕs, n)
        resize!(ranges´, 0)
        for (i, r) in enumerate(ranges)
            subspace = view(ψ, :, r)
            vsubspace = subspace' * v * subspace
            veigen = eigen!(Hermitian(vsubspace))
            if hasapproxruns(veigen.values, codiag.degtol)
                roffset = minimum(r) - 1 # Range offset within the ϵ vector
                pushapproxruns!(ranges´, veigen.values, roffset, codiag.degtol)
            end
            subspace .= subspace * veigen.vectors
        end
        ranges, ranges´ = ranges´, ranges
        isempty(ranges) && break
    end
    return ψ
end

# Could perhaps be better/faster using a generalized CoSort
function sorteigs!(perm, ϵ::Vector, ψ::Matrix)
    resize!(perm, length(ϵ))
    p = sortperm!(perm, ϵ, by = real)
    # permute!(ϵ, p)
    sort!(ϵ, by = real)
    Base.permutecols!!(ψ, p)
    return ϵ, ψ
end

#######################################################################
# Codiagonalizer
#######################################################################

## Codiagonalizer
## Uses velocity operators along different directions. If not enough, use finite differences
## along mesh directions
struct Codiagonalizer{T,F<:Function}
    cmatrixf::F
    matrixindices::UnitRange{Int}
    degtol::T
    rangesA::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    rangesB::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    perm::Vector{Int}               # Prealloc for sortperm!
end

function codiagonalizer(h::Union{Hamiltonian,ParametricHamiltonian}, matrix, mesh, lift; kw...)
    veldirs = velocitydirections(parent(h); kw...)
    veldirs_with_params = padparams.(veldirs, Ref(h))
    nv = length(veldirs)
    matrixindices = 1:(nv + nv + 1)
    degtol = sqrt(eps(real(blockeltype(h))))
    delta = meshdelta(mesh, lift)
    delta = iszero(delta) ? degtol : delta
    aom = anyoldmatrix(matrix)
    cmatrixf(meshϕs, n) =
        if n <= nv
            bloch!(matrix, h, applylift(lift, meshϕs), dn -> im * veldirs[n]' * dn)
        elseif n - nv <= nv # resort to finite differences
            bloch!(matrix, h, applylift(lift, meshϕs) + delta * veldirs_with_params[n - nv])
        else # use a fixed arbitrary matrix
            aom
        end
    return Codiagonalizer(cmatrixf, matrixindices, degtol, UnitRange{Int}[], UnitRange{Int}[], Int[])
end

function codiagonalizer(matrixf::Function, matrix::AbstractMatrix{T}, mesh; kw...) where {T}
    meshdirs = meshdirections(mesh; kw...)
    nm = length(meshdirs)
    matrixindices = 1:(nm + 1)
    degtol = sqrt(eps(real(eltype(T))))
    delta = meshdelta(mesh)
    delta = iszero(delta) ? degtol : delta
    aom = anyoldmatrix(matrix)
    cmatrixf(meshϕs, n) =
        if n <= nm # finite differences
            matrixf(meshϕs + delta * meshdirs[n])
        else # use a fixed arbitrary matrix
            aom
        end
    return Codiagonalizer(cmatrixf, matrixindices, degtol, UnitRange{Int}[], UnitRange{Int}[], Int[])
end

velocitydirections(::Hamiltonian{LA,L}; kw...) where {LA,L} = _directions(Val(L); kw...)

meshdirections(::Mesh{L}; kw...) where {L} = _directions(Val(L); kw...)

padparams(v, ::Hamiltonian) = v
padparams(v::SVector{L,T}, ::ParametricHamiltonian{P}) where {L,T,P} = vcat(zero(SVector{P,T}), v)

function _directions(::Val{L}; direlements = 0:1, onlypositive = true) where {L}
    directions = vec(SVector{L,Int}.(Iterators.product(ntuple(_ -> direlements, Val(L))...)))
    onlypositive && filter!(ispositive, directions)
    unique!(normalize, directions)
    sort!(directions, by = norm, rev = false)
    return directions
end

meshdelta(mesh::Mesh{<:Any,T}, lift = missing) where {T} =
    T(0.1) * norm(applylift(lift, first(minmax_edge(mesh))))

# function anyoldmatrix(matrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
#     n = size(matrix, 1)
#     ri = one(Ti):Ti(n)
#     rv = Tv.(im .* (1:n))
#     s = sparse(ri, ri, rv, n, n)
#     return s
# end

function anyoldmatrix(matrix::SparseMatrixCSC, rng = MersenneTwister(1))
    s = copy(matrix)
    rand!(rng, nonzeros(s))
    return s
end

# anyoldmatrix(m::M) where {T,M<:DenseArray{T}} = M(Diagonal(T.(im .* (1:size(m,1)))))
anyoldmatrix(m::DenseArray, rng = MersenneTwister(1)) = rand!(rng, copy(m))