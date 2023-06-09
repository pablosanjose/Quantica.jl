############################################################################################
# sublat
#region

sublat(sites::Union{Number,Tuple,SVector,AbstractVector{<:Number}}...; name = :A) =
    Sublat([float.(promote(sanitize_SVector.(sites)...))...], Symbol(name))

function sublat(sites::AbstractVector; name = :A)
    T = foldl((x, y) -> promote_type(x, eltype(sanitize_SVector(y))), sites; init = Bool)
    return Sublat(sanitize_SVector.(float(T), sites), Symbol(name))
end

#endregion

############################################################################################
# lattice
#region

lattice(s::Sublat, ss::Sublat...; kw...) = _lattice(promote(s, ss...)...; kw...)

lattice(ss::AbstractVector{<:Sublat}; kw...) = _lattice(ss; kw...)

# Start with an empty list of nranges, to be filled as they are requested
Lattice(b::Bravais{T}, u::Unitcell{T}) where {T} = Lattice(b, u, Tuple{Int,T}[])

_lattice(ss::Sublat{T,E}...;
    bravais = (),
    dim = Val(E),
    type::Type{T´} = T,
    names = sublatname.(ss)) where {T,E,T´} =
    _lattice(ss, bravais, dim, type, names)

_lattice(ss::AbstractVector{S};
    bravais = (),
    dim = Val(E),
    type::Type{T´} = T,
    names = sublatname.(ss)) where {T,E,T´,S<:Sublat{T,E}} =
    _lattice(ss, bravais, dim, type, names)

_lattice(ss, bravais, dim, type, names) =
    Lattice(Bravais(type, dim, bravais), unitcell(ss, names, postype(dim, type)))

function lattice(lat::Lattice{T,E};
                 bravais = bravais_matrix(lat),
                 dim = Val(E),
                 type::Type{T´} = T,
                 names = sublatnames(lat)) where {T,E,T´}
    u = unitcell(unitcell(lat), names, postype(dim, type))
    b = Bravais(type, dim, bravais)
    return Lattice(b, u)
end

postype(dim, type) = SVector{dim,type}
postype(::Val{E}, type) where {E} = SVector{E,type}

function unitcell(sublats, names, postype::Type{S}) where {S<:SVector}
    sites´ = S[]
    offsets´ = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sites(sublats[s])
            push!(sites´, sanitize_SVector(S, site))
        end
        push!(offsets´, length(sites´))
    end
<<<<<<< HEAD
    return Unitcell(sites´, names, offsets´)
=======
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

Base.:*(factor::Number, b::Bravais) = Bravais(factor * b.matrix)
Base.:*(b::Bravais, factor::Number) = Bravais(b.matrix * factor)

#######################################################################
# Lattice
#######################################################################
# Need mutable to be able to transform! in place (change positions *and* bravais)
mutable struct Lattice{E,L,T<:AbstractFloat,B<:Bravais{E,L,T},U<:Unitcell{E,T}} <: AbstractLattice{E,L,T}
    bravais::B
    unitcell::U
end

displaynames(l::AbstractLattice) = display_as_tuple(l.unitcell.names, ":")

function Base.show(io::IO, lat::Lattice)
    i = get(io, :indent, "")
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(bravais(lat); digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell")
end

Base.summary(::Lattice{E,L,T}) where {E,L,T} =
    "Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space"

# External API #

"""
    lattice(sublats::Sublat...; bravais = (), dim::Val{E}, type::T, names = missing)

Create a `Lattice{E,L,T}` with Bravais vectors `bravais` and sublattices `sublats`
converted to a common  `E`-dimensional embedding space and type `T`. To override the
embedding  dimension `E`, use keyword `dim = Val(E)`. Similarly, override type `T` with
`type = T`.

The keyword `bravais` indicates one or more Bravais vectors in the form of tuples or other
iterables. It can also be an `AbstractMatrix` of dimension `E×L`. The default `bravais = ()`
corresponds to a bounded lattice with no Bravais vectors.

A keyword `names` can be used to rename `sublats`. Given names can be replaced to ensure
that all sublattice names are unique.

    lattice(lat::AbstractLattice; bravais = missing, dim = missing, type = missing, names = missing)

Create a new lattice by applying any non-missing `kw` to `lat`. For performance, allocations
will be avoided if possible (depends on `kw`), so the result can share memory of `lat`. To
avoid that, do `lattice(copy(lat); kw...)`.

See also `LatticePresets` for built-in lattices.

# Examples

```jldoctest
julia> lattice(sublat((0, 0)), sublat((0, Float32(1))); bravais = (1, 0), dim = Val(3))
Lattice{3,1,Float32} : 1D lattice in 3D space
  Bravais vectors : ((1.0f0, 0.0f0, 0.0f0),)
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.honeycomb(names = (:C, :D))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.cubic(bravais = ((1, 0), (0, 2)))
Lattice{3,2,Float64} : 2D lattice in 3D space
  Bravais vectors : ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (1) --> 1 total per unit cell
```

# See also
    `LatticePresets`, `bravais`, `sublat`, `supercell`, `intracell`
"""
function lattice(s::Sublat, ss::Sublat...; bravais = (), kw...)
    u = Unitcell(s, ss...; kw...)
    b = Bravais(bravais, u)
    return lattice(u, b)
end

function lattice(unitcell::U, bravais::B) where {E2,L2,E,T,B<:Bravais{E2,L2},U<:Unitcell{E,T}}
    L = min(E,L2) # L should not exceed E
    bravais´ = convert(Bravais{E,L,T}, bravais)
    return Lattice(bravais´, unitcell)
end

function lattice(lat::Lattice; bravais = bravais(lat), kw...)
    u = Unitcell(lat.unitcell; kw...)
    b = Bravais(bravais, u)
    return lattice(u, b)
end
lattice(; kw...) = lat -> lattice(lat; kw...)

zerocell(::Lattice{<:Any,L}) where {L} = zero(SVector{L,Int})

#######################################################################
# Supercell
#######################################################################
struct Supercell{L,L´,M<:Union{Missing,OffsetArray{Bool}},S<:SMatrix{L,L´}} # L´ is supercell dim
    matrix::S
    sites::UnitRange{Int}
    cells::CartesianIndices{L,NTuple{L,UnitRange{Int}}}
    mask::M
>>>>>>> master
end

function unitcell(u::Unitcell{T´,E´}, names, postype::Type{S}) where {T´,E´,S<:SVector}
    sites´ = sanitize_SVector.(S, sites(u))
    offsets´ = offsets(u)
    Unitcell(sites´, names, offsets´)
end

# with simple rename, don't copy sites
unitcell(u::Unitcell{T,E}, names, postype::Type{S})  where {T,E,S<:SVector{E,T}} =
    Unitcell(sites(u), names, offsets(u))

#endregion

# ############################################################################################
# # foreach_site(l::Lattice)
# #region

# function foreach_site(f, lat::Lattice, sublatsrc = missing)
#     itr = sublatsrc === missing ? siterage(lat) : siterange(lat, sublatsrc)
#     for i in itr
#         f(site(lat, i))
#     end
#     return nothing
# end

############################################################################################
# combine lattices - combine sublats if equal name
#region

function combine(lats::Lattice{<:Any,E,L}...) where {E,L}
    isapprox_modulo_shuffle(bravais_matrix.(lats)...) ||
        throw(ArgumentError("To combine lattices they must all share the same Bravais matrix. They read $(bravais_matrix.(lats))"))
    bravais´ = bravais(first(lats))
    unitcell´ = combine(unitcell.(lats)...)
    return Lattice(bravais´, unitcell´)
end

combine(lats::Lattice{<:Any}...) =
    argerror("Tried to combine lattices with different dimension or embedding dimension")

function combine(ucells::Unitcell...)
    names´ = vcat(sublatnames.(ucells)...)
    sites´ = vcat(sites.(ucells)...)
    offsets´ = combined_offsets(offsets.(ucells)...)
    return Unitcell(sites´, names´, offsets´)
end

isapprox_modulo_shuffle() = true

function isapprox_modulo_shuffle(s::AbstractMatrix, ss::AbstractMatrix...)
    for s´ in ss, c´ in eachcol(s´)
        any(c -> c ≈ c´ || c ≈ -c´, eachcol(s)) || return false
    end
    return true
end

combined_offsets(offsets...) = lengths_to_offsets(Iterators.flatten(diff.(offsets)))

#endregion

############################################################################################
# neighbors
#   TODO: may be simplified/optimized
#region

function nrange(n, lat)
    for (n´, r) in nranges(lat)
        n == n´ && return r
    end
    r = compute_nrange(n, lat)
    push!(nranges(lat), (n, r))
    return r
end


function compute_nrange(n, lat::Lattice{T}) where {T}
    latsites = sites(lat)
    dns = BoxIterator(zerocell(lat))
    br = bravais_matrix(lat)
    # 128 is a heuristic cutoff for kdtree vs brute-force search
    if length(latsites) <= 128
        dists = fill(T(Inf), n)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for (i, ri) in enumerate(latsites), (j, rj) in enumerate(latsites)
                j <= i && iszero(dn) && continue
                r = ri - rj + br * dn
                update_dists!(dists, r'r)
            end
            isfinite(last(dists)) || acceptcell!(dns, dn)
        end
        dist = sqrt(last(dists))
    else
        tree = KDTree(latsites)
        dist = T(Inf)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for r0 in latsites
                r = r0 + br * dn
                dist = min(dist, compute_nrange(n, tree, r, nsites(lat)))
            end
            isfinite(dist) || acceptcell!(dns, dn)
        end
    end
    return dist
end

function update_dists!(dists, dist)
    len = length(dists)
    for (n, d) in enumerate(dists)
        isapprox(dist, d) && break
        if dist < d
            dists[n+1:len] .= dists[n:len-1]
            dists[n] = dist
            break
        end
    end
    return dists
end

function compute_nrange(n, tree, r::AbstractVector, nmax)
    for m in n:nmax
        _, dists = knn(tree, r, 1 + m, true)
        popfirst!(dists)
        unique_sorted_approx!(dists)
        length(dists) == n && return maximum(dists)
    end
    return convert(eltype(r), Inf)
end

function unique_sorted_approx!(v::AbstractVector)
    i = 1
    xprev = first(v)
    for j in 2:length(v)
        if v[j] ≈ xprev
            xprev = v[j]
        else
            i += 1
            xprev = v[i] = v[j]
        end
    end
    resize!(v, i)
    return v
end

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

#endregion
