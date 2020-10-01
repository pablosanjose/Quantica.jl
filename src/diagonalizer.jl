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

diagonalizer(h, matrix, mesh, method, minoverlap, mapping) =
    Diagonalizer(method, codiagonalizer(h, matrix, mesh, mapping), Float64(minoverlap))

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

# mapping = missing is assumed when h is a Function that generates matrices, instead of a Hamiltonian or ParametricHamiltonian
function codiagonalizer(h, matrix::AbstractMatrix{T}, mesh, mapping) where {T}
    sample_phiparams = map_phiparams(mapping, first(vertices(mesh)))
    dirs = codiag_directions(sample_phiparams)
    degtol = sqrt(eps(real(eltype(T))))
    delta = meshdelta(mesh)
    delta = iszero(delta) ? degtol : delta
    comatrix, matrixindices = codiag_function(h, matrix, mapping, dirs, delta)
    return Codiagonalizer(comatrix, matrixindices, degtol, UnitRange{Int}[], UnitRange{Int}[], Int[])
end

function codiag_function(h::Union{Hamiltonian,ParametricHamiltonian}, matrix, mapping, dirs, delta)
    hdual = dual_if_parametric(h)
    matrixdual = dualarray(matrix)
    anyold = anyoldmatrix(matrix)
    ndirs = length(dirs)
    matrixindices = 1:(ndirs + ndirs + 1)
    comatrix(meshϕs, n) =
        if n <= ndirs # automatic differentiation using dual numbers
            ϕsparams = dual_phisparams(map_phiparams(mapping, meshϕs), dirs[n])
            dualpart.(bloch!(matrixdual, hdual, ϕsparams))
        elseif n - ndirs <= ndirs # resort to finite differences
            ϕsparams = delta_phisparams(map_phiparams(mapping, meshϕs), delta * dirs[n - ndirs])
            bloch!(matrix, h, ϕsparams)
        else # use a fixed arbitrary matrix
            anyold
        end
    return comatrix, matrixindices
end

dual_if_parametric(ph::ParametricHamiltonian) = Dual(ph)
dual_if_parametric(h::Hamiltonian) = h

# In the Function case we cannot know what directions to scan (arguments of matrixf). Also,
# we cannot be sure that dual numbers propagate. We thus restrict to finite differences in the mesh
# Note that mapping is already wrapped into matrixf in the calling bandstructure(::Function,...)
function codiag_function(matrixf::Function, matrix, mapping::Missing, meshdirs, delta)
    anyold = anyoldmatrix(matrix)
    ndirs = length(meshdirs)
    matrixindices = 1:(ndirs + 1)
    comatrix(meshϕs, n) =
        if n <= ndirs # finite differences
            matrixf(meshϕs + delta * meshdirs[n])
        else # use a fixed arbitrary matrix
            anyold
        end
    return comatrix, matrixindices
end

val_length(::SVector{N}, nt::NamedTuple) where {N} = Val(N+length(nt))

codiag_directions(phiparams) = codiag_directions(val_length(phiparams...), phiparams)

function codiag_directions(::Val{L}, phiparams, direlements = 0:1) where {L}
    directions = vec(SVector{L,Int}.(Iterators.product(ntuple(_ -> direlements, Val(L))...)))
    mask_dirs!(directions, phiparams)
    filter!(ispositive, directions)
    unique!(directions)
    sort!(directions, by = norm)
    return directions
end

# Zeros out any direction that cannot modify a param because it is not a number
function mask_dirs!(dirs::Vector{S}, pp) where {L,S<:SVector{L}}
    valparams = values(last(pp))
    valids = valparams .!= maybe_sum.(valparams, 1)
    n = length(first(pp))
    mask = SVector(ntuple(i -> i <= n || valids[i - n] , Val(L)))
    map!(dir -> mask .* dir, dirs, dirs)
    return dirs
end

dual_phisparams(ϕs::SVector{N}, dir) where {N} = Dual.(ϕs, frontsvec(dir, Val(N)))
dual_phisparams(params::NamedTuple, dir) = NamedTuple{keys(params)}(maybe_dual.(values(params), tailtuple(dir, Val(length(params)))))
dual_phisparams((ϕs, params)::Tuple, dir) = (dual_phisparams(ϕs, dir), dual_phisparams(params, dir))

maybe_dual(param::Number, ε) = Dual(param, ε)
maybe_dual(param, ε) = param

delta_phisparams(ϕs::SVector{N}, dir) where {N} = ϕs + frontsvec(dir, Val(N))
delta_phisparams(params::NamedTuple, dir) = NamedTuple{keys(params)}(maybe_sum.(values(params), tailtuple(dir, Val(length(params)))))
delta_phisparams((ϕs, params)::Tuple, dir) = (delta_phisparams(ϕs, dir), delta_phisparams(params, dir))

maybe_sum(param::Number, ε) = param + ε
maybe_sum(param, ε) = param

meshdelta(mesh::Mesh{<:Any,T}) where {T} = T(0.1) * norm(first(minmax_edge(mesh)))

function anyoldmatrix(matrix::SparseMatrixCSC, rng = MersenneTwister(1))
    s = copy(matrix)
    rand!(rng, nonzeros(s))
    return s
end

anyoldmatrix(m::DenseArray, rng = MersenneTwister(1)) = rand!(rng, copy(m))