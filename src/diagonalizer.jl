#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizeMethod end

struct Diagonalizer{S<:AbstractDiagonalizeMethod,C}
    method::S
    codiag::C
    minoverlap::Float64
end

diagonalizer(method, codiag, minoverlap) = Diagonalizer(method, codiag, Float64(minoverlap))

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

### matrix types

similarmatrix(h, method::AbstractDiagonalizeMethod) = similarmatrix(h, method_matrixtype(method, h))

method_matrixtype(::LinearAlgebraPackage, h) = Matrix{blockeltype(h)}
method_matrixtype(::AbstractDiagonalizeMethod, h) = flatten

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
        v = codiag.comatrix(ϕs, n)
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
    comatrix::F
    matrixindices::UnitRange{Int}
    degtol::T
    rangesA::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    rangesB::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    perm::Vector{Int}               # Prealloc for sortperm!
end

# lift = missing is assumed when h is a Function that generates matrices, instead of a Hamiltonian or ParametricHamiltonian
function codiagonalizer(h, matrix::AbstractMatrix{T}, mesh, lift) where {T}
    dirs = codiag_directions(h, mesh)
    degtol = sqrt(eps(real(eltype(T))))
    delta = meshdelta(mesh)
    delta = iszero(delta) ? degtol : delta
    comatrix, matrixindices = codiag_function(h, matrix, lift, dirs, delta)
    return Codiagonalizer(comatrix, matrixindices, degtol, UnitRange{Int}[], UnitRange{Int}[], Int[])
end

function codiag_function(h::Union{Hamiltonian,ParametricHamiltonian}, matrix, lift, dirs, delta)
    hdual = dual_if_parametric(h)
    matrixdual = dualarray(matrix)
    anyold = anyoldmatrix(matrix)
    ndirs = length(dirs)
    matrixindices = 1:(ndirs + ndirs + 1)
    comatrix(meshϕs, n) =
        if n <= ndirs # automatic differentiation using dual numbers
            ϕs´ = dualϕs(applylift(lift, meshϕs), dirs[n])
            dualpart.(bloch!(matrixdual, hdual, ϕs´))
        elseif n - ndirs <= ndirs # resort to finite differences
            ϕs´ = deltaϕs(applylift(lift, meshϕs), delta * dirs[n - ndirs])
            bloch!(matrix, h, ϕs´)
        else # use a fixed arbitrary matrix
            anyold
        end
    return comatrix, matrixindices
end

dual_if_parametric(ph::ParametricHamiltonian) = Dual(ph)
dual_if_parametric(h::Hamiltonian) = h

# In the Function case we cannot know what directions to scan (arguments of matrixf). Also,
# we cannot be sure that dual numbers propagate. We thus restrict to finite differences in the mesh
function codiag_function(matrixf::Function, matrix, lift, meshdirs, delta)
    anyold = anyoldmatrix(matrix)
    ndirs = length(meshdirs)
    matrixindices = 1:(ndirs + 1)
    comatrix(meshϕs, n) =
        if n <= ndirs # finite differences
            matrixf(applylift(lift, meshϕs + delta * meshdirs[n]))
        else # use a fixed arbitrary matrix
            anyold
        end
    return comatrix, matrixindices
end

codiag_directions(h::Union{Hamiltonian,ParametricHamiltonian}, mesh) = codiag_directions(_valdim(h))
codiag_directions(::Function, mesh) = codiag_directions(_valdim(mesh))

@inline _valdim(::Hamiltonian{<:Any,L}) where {L} = Val(L)
@inline _valdim(ph::ParametricHamiltonian{N}) where {N} = Val(N+latdim(ph.h))
@inline _valdim(::Mesh{N}) where {N} = Val(N)

function codiag_directions(::Val{L}, direlements = 0:1, onlypositive = true) where {L}
    directions = vec(SVector{L,Int}.(Iterators.product(ntuple(_ -> direlements, Val(L))...)))
    onlypositive && filter!(ispositive, directions)
    unique!(normalize, directions)
    sort!(directions, by = norm, rev = false)
    return directions
end

dualϕs(liftedϕs, dir) = maybe_dual.(liftedϕs, dir)

maybe_dual(liftedϕ::Number, ε) = Dual(liftedϕ, ε)
maybe_dual(liftedϕ, ε) = liftedϕ

deltaϕs(liftedϕs, dir) = liftedϕs + dir

meshdelta(mesh::Mesh{<:Any,T}) where {T} = T(0.1) * norm(first(minmax_edge(mesh)))

function anyoldmatrix(matrix::SparseMatrixCSC, rng = MersenneTwister(1))
    s = copy(matrix)
    rand!(rng, nonzeros(s))
    return s
end

anyoldmatrix(m::DenseArray, rng = MersenneTwister(1)) = rand!(rng, copy(m))