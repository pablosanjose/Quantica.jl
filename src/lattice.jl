abstract type AbstractLattice{E,L,T} end

#######################################################################
# Sublat (sublattice)
#######################################################################
struct Sublat{E,T,V<:AbstractVector{SVector{E,T}}}
    sites::V
    name::NameType
end

Base.empty(s::Sublat) = Sublat(empty(s.sites), s.name)

Base.copy(s::Sublat) = Sublat(copy(s.sites), s.name)

Base.show(io::IO, s::Sublat{E,T}) where {E,T} = print(io,
"Sublat{$E,$T} : sublattice of $T-typed sites in $(E)D space
  Sites    : $(length(s.sites))
  Name     : $(displayname(s))")

displayname(s::Sublat) = s.name == nametype(:_) ? "pending" : string(":", s.name)
# displayorbitals(s::Sublat) = string("(", join(string.(":", s.orbitals), ", "), ")")
nsites(s::Sublat) = length(s.sites)

# External API #

"""
    sublat(sites...; name::$(NameType))
    sublat(sites::Vector{<:SVector}; name::$(NameType))

Create a `Sublat{E,T,D}` that adds a sublattice, of name `name`, with sites at positions
`sites` in `E` dimensional space. Sites can be entered as tuples or `SVectors`.

# Examples

```jldoctest
julia> sublat((0.0, 0), (1, 1), (1, -1), name = :A)
Sublat{2,Float64} : sublattice of Float64-typed sites in 2D space
  Sites    : 3
  Name     : :A
```
"""
sublat(sites::AbstractVector{<:SVector}; name = :_, kw...) =
    Sublat(sites, nametype(name))
sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = sublat(toSVectors(vs...); kw...)

toSVectors(vs...) = [promote(toSVector.(vs)...)...]

dims(::NTuple{N,Sublat{E,T}}) where {N,E,T} = E

sublatnames(ss::NTuple{N,Sublat{E,T}}) where {N,E,T} = (s -> s.name).(ss)

#######################################################################
# Unitcell
#######################################################################
struct Unitcell{E,T,N}
    sites::Vector{SVector{E,T}}
    names::NTuple{N,NameType}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end                             # so that diff(offset) == sublatlengths

Unitcell(sublats::Sublat...; kw...) = Unitcell(promote(sublats...); kw...)

Unitcell(s; dim = Val(dims(s)), type = float(coordtype(s)), names = sublatnames(s)) =
    _unitcell(s, dim, type, names)

# Dynamic dispatch
_unitcell(sublats, dim::Integer, type, names) = _unitcell(sublats, Val(dim), type, names)

function _unitcell(sublats::NTuple{N,Sublat}, dim::Val{E}, type::Type{T}, names) where {N,E,T}
    sites = SVector{E,T}[]
    offsets = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sublats[s].sites
            push!(sites, padtotype(site, SVector{E,T}))
        end
        push!(offsets, length(sites))
    end
    names´ = uniquenames(sanitize_names(names, Val(N)))
    return Unitcell(sites, names´, offsets)
end

_unitcell(u::Unitcell{E,T,N}, dim::Val{E}, type::Type{T}, names) where {E,T,N} =
    Unitcell(u.sites, uniquenames(sanitize_names(names, Val(N))), u.offsets)

_unitcell(u::Unitcell{E,T,N}, dim::Val{E2}, type::Type{T2}, names) where {E,T,E2,T2,N} =
    Unitcell(padtotype.(u.sites, SVector{E2,T2}), uniquenames(sanitize_names(names, Val(N))), u.offsets)

sanitize_names(name::Union{NameType,Int}, ::Val{N}) where {N} = ntuple(_ -> NameType(name), Val(N))
sanitize_names(names::AbstractVector, ::Val{N}) where {N} = ntuple(i -> NameType(names[i]), Val(N))
sanitize_names(names::NTuple{N,Union{NameType,Int}}, ::Val{N}) where {N} = NameType.(names)

function uniquenames(names::NTuple{N,NameType}) where {N}
    namesvec = [names...]
    allnames = NameType[:_]
    for i in eachindex(names)
        namesvec[i] in allnames && (namesvec[i] = uniquename(allnames, namesvec[i], i))
        push!(allnames, namesvec[i])
    end
    names´ = ntuple(i -> namesvec[i], Val(N))
    return names´
end

function uniquename(allnames, name, i)
    newname = nametype(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

sitepositions(u::Unitcell) = u.sites
sitepositions(u::Unitcell, s::Int) = view(u.sites, siterange(u, s))

siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

enumeratesites(u::Unitcell, sublat) = ((i, sitepositions(u)[i]) for i in siterange(u, sublat))

nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]

offsets(u::Unitcell) = u.offsets

function sublat(u::Unitcell, siteidx)
    l = length(u.offsets)
    for s in 2:l
        @inbounds u.offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

sublatlengths(u::Unitcell) = diff(u.offsets)

nsublats(u::Unitcell) = length(u.names)

sublats(u::Unitcell) = 1:nsublats(u)

sublatname(u::Unitcell, s) = u.names[s]

sublatnames(u::Unitcell) = u.names

transform!(f::Function, u::Unitcell) = (u.sites .= f.(u.sites); u)

dims(::Unitcell{E}) where {E} = E

Base.copy(u::Unitcell) = Unitcell(copy(u.sites), u.names, copy(u.offsets))

Base.isequal(u1::Unitcell, u2::Unitcell) =
    isequal(u1.sites, u2.sites) && isequal(u1.names, u2.names) && isequal(u1.offsets, u2.offsets)

#######################################################################
# Bravais
#######################################################################
struct Bravais{E,L,T,EL}
    matrix::SMatrix{E,L,T,EL}
end

Bravais(vs::Tuple{}, ucell::Unitcell{E,T}) where {E,T} =  Bravais(SMatrix{E,0,T,0}())
Bravais(vs, ucell) = Bravais(toSMatrix(vs))

# External API #

"""
    bravais(lat::Lattice)
    bravais(h::Hamiltonian)

Obtain the Bravais matrix of lattice `lat` or Hamiltonian `h`

# Examples

```jldoctest
julia> bravais((1.0, 2), (3, 4))
Bravais{2,2,Float64} : set of 2 Bravais vectors in 2D space.
  Vectors     : ((1.0, 2.0), (3.0, 4.0))
  Matrix      : [1.0 3.0; 2.0 4.0]
```

# See also:
    `lattice`
"""
bravais(lat::AbstractLattice) = lat.bravais.matrix

transform(f::F, b::Bravais{E,0}) where {E,F<:Function} = b

function transform(f::F, b::Bravais{E,L,T}) where {E,L,T,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

Base.:*(factor::Number, b::Bravais) = Bravais(factor * b.matrix)
Base.:*(b::Bravais, factor::Number) = Bravais(b.matrix * factor)

#######################################################################
# Lattice
#######################################################################
struct Lattice{E,L,T<:AbstractFloat,B<:Bravais{E,L,T},U<:Unitcell{E,T}} <: AbstractLattice{E,L,T}
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

# See also:
    `LatticePresets`, `bravais`, `sublat`, `supercell`, `intracell`
"""
function lattice(ss::Sublat...; bravais = (), kw...)
    u = Unitcell(ss...; kw...)
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

#######################################################################
# Supercell
#######################################################################
struct Supercell{L,L´,M<:Union{Missing,OffsetArray{Bool}},S<:SMatrix{L,L´}} # L´ is supercell dim
    matrix::S
    sites::UnitRange{Int}
    cells::CartesianIndices{L,NTuple{L,UnitRange{Int}}}
    mask::M
end

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell{L,L´,<:OffsetArray}) where {L,L´} = sum(s.mask)
nsites(s::Supercell{L,L´,Missing}) where {L,L´} = length(s.sites) * length(s.cells)

Base.CartesianIndices(s::Supercell) = s.cells

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    i = get(io, :indent, "")
    print(io, i,
"Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice
$i  Supervectors  : $(displayvectors(s.matrix))
$i  Supersites    : $(nsites(s))")
end

ismasked(s::Supercell{L,L´,<:OffsetArray})  where {L,L´} = true
ismasked(s::Supercell{L,L´,Missing})  where {L,L´} = false

isinmask(mask::Missing, inds::Vararg{Any,N}) where {N} = true
isinmask(mask::OffsetArray, inds::Vararg{Any,N}) where {N} = checkbounds(Bool, mask, inds...) && mask[inds...]

Base.copy(s::Supercell{<:Any,<:Any,Missing}) =
    Supercell(s.matrix, s.sites, s.cells, s.mask)

Base.copy(s::Supercell{<:Any,<:Any,<:AbstractArray}) =
    Supercell(s.matrix, s.sites, s.cells, copy(s.mask))

Base.isequal(s1::Supercell, s2::Supercell) =
    isequal(s1.matrix, s2.matrix) && isequal(s1.sites, s2.sites) &&
    isequal(s1.cells, s2.cells) && isequal(s1.mask, s2.mask)

## Boolean masking
(Base.:&)(s1::Supercell, s2::Supercell) = boolean_mask_supercell(Base.:&, s1, s2)
(Base.:|)(s1::Supercell, s2::Supercell) = boolean_mask_supercell(Base.:|, s1, s2)
(Base.xor)(s1::Supercell, s2::Supercell) = boolean_mask_supercell(Base.xor, s1, s2)

function boolean_mask_supercell(f, s1::Supercell, s2::Supercell)
    check_compatible_supercell(s1, s2)
    cells = boolean_mask_bbox(f, s1.cells, s2.cells)
    indranges = (s1.sites, cells.indices...)
    mask  = boolean_mask(f, s1.mask, s2.mask, indranges)
    mask´ = all(mask) ? missing : mask
    return Supercell(s1.matrix, s1.sites, cells, mask´)
end

function check_compatible_supercell(s1, s2)
    compatible = isequal(s1.matrix, s2.matrix) && isequal(s1.sites, s2.sites)
    compatible || throw(ArgumentError("Supercells are incompatible"))
    return nothing
end

boolean_mask_bbox(f, bbox1::CartesianIndices, bbox2::CartesianIndices) =
    CartesianIndices(boolean_mask_bbox.(f, bbox1.indices, bbox2.indices))
boolean_mask_bbox(::typeof(Base.:&),  r1::AbstractRange, r2::AbstractRange) =
    max(first(r1), first(r2)):min(last(r1), last(r2))
boolean_mask_bbox(::typeof(Base.:|),  r1::AbstractRange, r2::AbstractRange) =
    min(first(r1), first(r2)):max(last(r1), last(r2))
boolean_mask_bbox(::typeof(Base.xor), r1::AbstractRange, r2::AbstractRange) =
    min(first(r1), first(r2)):max(last(r1), last(r2))

function boolean_mask(f, mask1, mask2, indranges)
    mask = OffsetArray{Bool}(undef, indranges)
    for I in CartesianIndices(mask)
        mask[I] = f(isinmask(mask1, I), isinmask(mask2, I))
    end
    return mask
end

#######################################################################
# Superlattice
#######################################################################
struct Superlattice{E,L,T<:AbstractFloat,L´,S<:Supercell{L,L´},B<:Bravais{E,L,T},U<:Unitcell{E,T}} <: AbstractLattice{E,L,T}
    bravais::B
    unitcell::U
    supercell::S
end

function Base.show(io::IO, lat::Superlattice)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(bravais(lat); digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell\n")
    print(ioindent, lat.supercell)
end

Base.summary(::Superlattice{E,L,T,L´}) where {E,L,T,L´} =
    "Superlattice{$E,$L,$T,$L´} : $(L)D lattice in $(E)D space, filling a $(L´)D supercell"

# apply f to trues in mask. Arguments are s = sublat, oldi = old site, dn, newi = new site
function foreach_supersite(f::F, lat::Superlattice) where {F<:Function}
    newi = 0
    @inbounds for s in 1:nsublats(lat), oldi in siterange(lat, s)
        for dn in CartesianIndices(lat.supercell)
            if isinmask(lat.supercell.mask, oldi, Tuple(dn)...)
                newi += 1
                f(s, oldi, toSVector(Int, Tuple(dn)), newi)
            end
        end
    end
    return nothing
end

## Boolean masking
(Base.:&)(s1::Superlattice, s2::Superlattice) = boolean_mask_superlattice(Base.:&, s1, s2)
(Base.:|)(s1::Superlattice, s2::Superlattice) = boolean_mask_superlattice(Base.:|, s1, s2)
(Base.xor)(s1::Superlattice, s2::Superlattice) = boolean_mask_superlattice(Base.xor, s1, s2)

function boolean_mask_superlattice(f, s1::Superlattice, s2::Superlattice)
    check_compatible_superlattice(s1, s2)
    return Superlattice(s1.bravais, s1.unitcell, f(s1.supercell, s2.supercell))
end

function check_compatible_superlattice(s1, s2)
    compatible = isequal(s1.unitcell, s2.unitcell) &&
                 is_bravais_compatible(s1.bravais, s2.bravais)
    compatible || throw(ArgumentError("Superlattices are incompatible for boolean masking"))
    return nothing
end

#######################################################################
# AbstractLattice interface
#######################################################################

Base.copy(lat::Lattice) = Lattice(lat.bravais, copy(lat.unitcell))
Base.copy(lat::Superlattice) = Superlattice(lat.bravais, copy(lat.unitcell), copy(lat.supercell))

Base.isequal(l1::Lattice, l2::Lattice) =
    isequal(l1.unitcell, l2.unitcell) && isequal(l1.bravais, l2.bravais)

Base.isequal(l1::Superlattice, l2::Superlattice) =
    isequal(l1.unitcell, l2.unitcell) && isequal(l1.bravais, l2.bravais) &&
    isequal(l1.supercell, l2.supercell)

coordtype(::AbstractLattice{E,L,T}) where {E,L,T} = T
coordtype(::NTuple{N,Sublat{E,T}}) where {N,E,T} = T
coordtype(::Unitcell{E,T}) where {E,T} = T

positiontype(::AbstractLattice{E,L,T}) where {E,L,T} = SVector{E,T}
dntype(::AbstractLattice{E,L}) where {E,L} = SVector{L,Int}

sublat(lat::AbstractLattice, siteidx) = sublat(lat.unitcell, siteidx)
sublats(lat::AbstractLattice) = sublats(lat.unitcell)

siterange(lat::AbstractLattice, sublat) = siterange(lat.unitcell, sublat)

allsitepositions(lat::AbstractLattice) = sitepositions(lat.unitcell)

siteposition(i, lat::AbstractLattice) = allsitepositions(lat)[i]
siteposition(i, dn::SVector, lat::AbstractLattice) = siteposition(i, lat) + bravais(lat) * dn

sitesublats(lat::AbstractLattice) = sitesublats(lat.unitcell)

offsets(lat::AbstractLattice) = offsets(lat.unitcell)

sublatlengths(lat::AbstractLattice) = sublatlengths(lat.unitcell)

enumeratesites(lat::AbstractLattice, sublat) = enumeratesites(lat.unitcell, sublat)

sublatname(lat::AbstractLattice, s = sublats(lat)) = sublatname(lat.unitcell, s)

nsites(lat::AbstractLattice) = nsites(lat.unitcell)
nsites(lat::AbstractLattice, sublat) = nsites(lat.unitcell, sublat)

nsublats(lat::AbstractLattice) = nsublats(lat.unitcell)

issuperlattice(lat::Lattice) = false
issuperlattice(lat::Superlattice) = true

ismasked(lat::Lattice) = false
ismasked(lat::Superlattice) = ismasked(lat.supercell)

maskranges(lat::Superlattice) = (1:nsites(lat), lat.supercell.cells.indices...)
maskranges(lat::Lattice) = (1:nsites(lat),)

"""
    transform!(f::Function, lat::Lattice)

Transform the site positions of `lat` by applying `f` to them in place.
"""
function transform!(f::Function, lat::Lattice)
    transform!(f, lat.unitcell)
    bravais´ = transform(f, lat.bravais)
    return Lattice(bravais´, lat.unitcell)
end

"""
    combine(lats::Lattice...)

If all `lats` have compatible Bravais vectors, combine them into a single lattice.
Sublattice names are renamed to be unique if necessary.
"""
function combine(lats::Lattice...)
    is_bravais_compatible(lats...) || throw(ArgumentError("Lattices must share all Bravais vectors, $(bravais.(lats))"))
    bravais´ = first(lats).bravais
    unitcell´ = combine((l -> l.unitcell).(lats)...)
    return Lattice(bravais´, unitcell´)
end

function combine(us::Vararg{Unitcell,N}) where {N}
    sites = vcat(ntuple(i -> us[i].sites, Val(N))...)
    names = uniquenames(tuplejoin(ntuple(i -> us[i].names, Val(N))...))
    offsets = combined_offsets(us...)
    return Unitcell(sites, names, offsets)
end

is_bravais_compatible() = true
is_bravais_compatible(lat::Lattice, lats::Lattice...) = all(l -> isequal(lat.bravais, l.bravais), lats)
is_bravais_compatible(b::Bravais, bs::Bravais...) = all(bi -> isequal(b, bi), bs)

function Base.isequal(b1::Bravais{E,L}, b2::Bravais{E,L}) where {E,L}
    vs1 = ntuple(i -> b1.matrix[:, i], Val(L))
    vs2 = ntuple(i -> b2.matrix[:, i], Val(L))
    for v2 in vs2
        found = false
        for v1 in vs1
            (isapprox(v1, v2) || isapprox(v1, -v2)) && (found = true; break)
        end
        !found && return false
    end
    return true
end

# should be cumsum([diff.(offset)...]) but non-allocating
function combined_offsets(us::Unitcell...)
    nsubs = sum(nsublats, us)
    offsets = Vector{Int}(undef, nsubs + 1)
    offsets[1] = 0
    idx = 2
    for u in us
        for idx´ in 2:length(u.offsets)
            offsets[idx] = offsets[idx-1] + u.offsets[idx´] - u.offsets[idx´-1]
            idx += 1
        end
    end
    return offsets
end

#######################################################################
# supercell
#######################################################################
"""
    supercell(lat::AbstractLattice{E,L}, v::NTuple{L,Integer}...; seed = missing, kw...)
    supercell(lat::AbstractLattice{E,L}, sc::SMatrix{L,L´,Int}; seed = missing, kw...)

Generates a `Superlattice` from an `L`-dimensional lattice `lat` with Bravais vectors
`br´= br * sc`, where `sc::SMatrix{L,L´,Int}` is the integer supercell matrix with the `L´`
vectors `v`s as columns. If no `v` are given, the superlattice will be bounded.

Only sites selected by `siteselector(; kw...)` will be included in the supercell (see
`siteselector` for details on the available keywords `kw`). The search for included sites
will start from point `seed::Union{Tuple,SVector}`, or the origin if `seed = missing`. If no
keyword `region` is given in `kw`, a Bravais unit cell perpendicular to the `v` axes will be
selected for the `L-L´` non-periodic directions.

    supercell(lattice::AbstractLattice{E,L}, factor::Integer; kw...)

Calls `supercell` with a uniformly scaled `sc = SMatrix{L,L}(factor * I)`

    supercell(lattice::AbstractLattice, factors::Integer...; kw...)

Calls `supercell` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    lat |> supercell(v...; kw...)

Curried syntax, equivalent to `supercell(lat, v...; kw...)

    supercell(h::Hamiltonian, v...; kw...)

Promotes the `Lattice` of `h` to a `Superlattice` without changing the Hamiltonian itself,
which always refers to the unitcell of the lattice.

# Examples

```jldoctest
julia> supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Superlattice{2,2,Float64,0} : 2D lattice in 2D space, filling a 0D supercell
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell
  Supercell{2,0} for 0D superlattice of the base 2D lattice
    Supervectors  : ()
    Supersites    : 652966

julia> supercell(LatticePresets.triangular(), (1,1), (1, -1))
Superlattice{2,2,Float64,2} : 2D lattice in 2D space, filling a 2D supercell
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (1) --> 1 total per unit cell
  Supercell{2,2} for 2D superlattice of the base 2D lattice
    Supervectors  : ((1, 1), (1, -1))
    Supersites    : 2

julia> LatticePresets.square() |> supercell(3)
Superlattice{2,2,Float64,2} : 2D lattice in 2D space, filling a 2D supercell
  Bravais vectors : ((1.0, 0.0), (0.0, 1.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (1) --> 1 total per unit cell
  Supercell{2,2} for 2D superlattice of the base 2D lattice
    Supervectors  : ((3, 0), (0, 3))
    Supersites    : 9
```

# See also:
    `unitcell`, `siteselector`
"""
supercell(v...; kw...) = lat -> supercell(lat, v...; kw...)

function supercell(lat::Lattice{E,L}, v...; seed = missing, kw...) where {E,L}
    scmatrix = sanitize_supercell(Val(L), v...)
    pararegion, perpregion = supercell_regions(lat, scmatrix)
    perpselector = siteselector(; region = perpregion, kw...)  # user kw can override region
    return _superlat(lat, scmatrix, pararegion, perpselector, seed)
end

# Fast-path methods
supercell(lat::Lattice{E,L}, factors::Vararg{Integer,L}) where {E,L} =
    _superlat_fastpath(lat, factors)
supercell(lat::Lattice{E,L}, factor::Integer) where {E,L} =
    _superlat_fastpath(lat, filltuple(factor, Val(L)))
supercell(lat::Lattice) = _superlat_fastpath(lat, ())

sanitize_supercell(::Val{L}) where {L} = SMatrix{L,0,Int}()
sanitize_supercell(::Val{L}, ::Tuple{}) where {L} = SMatrix{L,0,Int}()
sanitize_supercell(::Val{L}, v::NTuple{L,Int}...) where {L} = toSMatrix(Int, v)
sanitize_supercell(::Val{L}, s::SMatrix{L}) where {L} = toSMatrix(Int, s)
sanitize_supercell(::Val{L}, v::Integer) where {L} = SMatrix{L,L,Int}(v*I)
sanitize_supercell(::Val{L}, ss::Integer...) where {L} = SMatrix{L,L,Int}(Diagonal(SVector(ss)))
sanitize_supercell(::Val{L}, v) where {L} =
    throw(ArgumentError("Improper supercell specification $v for an $L lattice dimensions, see `supercell`"))

function supercell_regions(lat::Lattice{E,L,T}, sc::SMatrix{L,L´}) where {E,L,T,L´}
    br = bravais(lat)
    extsc = extended_supercell(br, sc)
    projector = pinverse(br * extsc) # E need not be equal to L, hence pseudoinverse
    siteshift = ntuple(Val(L)) do i
        minimum((projector * r)[i] for r in allsitepositions(lat)) - sqrt(eps(T))
    end
    pararegion(r) = all(i -> 0 <= (projector * r)[i] - siteshift[i] < 1, 1:L´)
    perpregion(r) = all(i -> 0 <= (projector * r)[i] - siteshift[i] < 1, (1+L´):L)
    return pararegion, perpregion
end

# supplements supercell with most orthogonal bravais axes
function extended_supercell(bravais, supercell::SMatrix{L,L´}) where {L,L´}
    L == L´ && return supercell
    bravais_new = bravais * supercell
    # νnorm are the L projections of old bravais on new bravais axis subspace
    ν = bravais' * bravais_new  # L×L´
    νnorm = SVector(ntuple(row -> norm(ν[row,:]), Val(L)))
    νorder = sortperm(νnorm)
    ext_axes = hcat(ntuple(i -> unitvector(SVector{L,Int}, νorder[i]), Val(L-L´))...)
    ext_supercell = hcat(supercell, ext_axes)
    return ext_supercell
end

function _superlat(lat, scmatrix, pararegion, selector_perp, seed)
    br = bravais(lat)
    rsel = resolve(selector_perp, lat)
    cells = _cell_iter(lat, pararegion, rsel, seed)
    ns = nsites(lat)
    mask = OffsetArray(falses(ns, size(cells)...), 1:ns, cells.indices...)
    @inbounds for dn in cells
        dntup = Tuple(dn)
        dnvec = toSVector(dntup)
        for i in siteindices(rsel, dnvec)
            r = siteposition(i, dnvec, lat)
            # site i is already in perpregion through rsel. Is it also in pararegion?
            mask[i, dntup...] = pararegion(r)
        end
    end
    supercell = Supercell(scmatrix, 1:ns, cells, all(mask) ? missing : mask)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

function _cell_iter(lat::Lattice{E,L}, pararegion, rsel_perp, seed) where {E,L}
    seed´ = seed === missing ? zero(SVector{L,Int}) : seedcell(SVector{E}(seed), bravais(lat))
    iter = BoxIterator(seed´)
    foundfirst = false
    counter = 0
    br = bravais(lat)
    for dnvec in iter
        found = false
        counter += 1; counter == TOOMANYITERS &&
            throw(ArgumentError("`region` seems unbounded (after $TOOMANYITERS iterations)"))
        foundfirst || acceptcell!(iter, dnvec)
        for i in siteindices(rsel_perp, dnvec)
            r = siteposition(i, dnvec, lat)
            # site i is already in perpregion through rsel. Is it also in pararegion?
            found_in_cell = pararegion(r)
            if found_in_cell
                acceptcell!(iter, dnvec)
                foundfirst = true
                break
            end
        end
    end
    c = CartesianIndices(iter)
    return c
end

seedcell(seed::NTuple{N,Any}, brmat) where {N} = seedcell(SVector{N}(seed), brmat)
seedcell(seed::SVector{E}, brmat::SMatrix{E}) where {E} = round.(Int, brmat \ seed)

function _superlat_fastpath(lat::Lattice{E,L}, factors) where {E,L}
    scmatrix = sanitize_supercell(Val(L), factors...)
    sites = 1:nsites(lat)
    cells = _cells_fastpath(Val(L), factors)
    mask = missing
    supercell = Supercell(scmatrix, sites, cells, mask)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

_cells_fastpath(::Val, factors) = CartesianIndices((i -> 0 : i - 1).(factors))
_cells_fastpath(::Val{L}, factors::Tuple{}) where {L} = CartesianIndices(filltuple(0:0, Val(L)))

#######################################################################
# unitcell
#######################################################################
"""
    unitcell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; seed = missing, kw...)
    unitcell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; seed = missing, kw...)

Generates a `Lattice` from an `L`-dimensional lattice `lat` and a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
unitcell matrix, with the `L´` vectors `v`s as columns. If no `v` are given, the new lattice
will be bounded.

Only sites selected by `siteselector(; kw...)` will be included in the supercell (see
`siteselector` for details on the available keywords `kw`). The search for included sites
will start from point `seed::Union{Tuple,SVector}`, or the origin if `seed = missing`. If no
keyword `region` is given in `kw`, a Bravais unit cell perpendicular to the `v` axes will be
selected for the `L-L´` non-periodic directions.

    unitcell(lattice::Lattice{E,L}, factor::Integer; kw...)

Calls `unitcell` with a uniformly scaled `uc = SMatrix{L,L}(factor * I)`

    unitcell(lattice::Lattice{E,L}, factors::Integer...; kw...)

Calls `unitcell` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    unitcell(slat::Superlattice)

Convert Superlattice `slat` into a lattice with its unit cell matching `slat`'s supercell.

    unitcell(h::Hamiltonian, v...; modifiers = (), kw...)

Transforms the `Lattice` of `h` to have a larger unitcell, while expanding the Hamiltonian
accordingly. The modifiers (a tuple of `ElementModifier`s, either `@onsite!` or `@hopping!`
with no free parameters) will be applied to onsite and hoppings as the hamiltonian is
expanded. See `@onsite!` and `@hopping!` for details

Note: for performance reasons, in sparse hamiltonians only the stored onsites and hoppings
will be transformed by `ElementModifier`s, so you might want to add zero onsites or hoppings
when building `h` to have a modifier applied to them later.

    lat_or_h |> unitcell(v...; kw...)

Curried syntax, equivalent to `unitcell(lat_or_h, v...; kw...)

# Examples

```jldoctest
julia> unitcell(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Lattice{2,0,Float64} : 0D lattice in 2D space
  Bravais vectors : ()
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (326483, 326483) --> 652966 total per unit cell

julia> unitcell(LatticePresets.triangular(), (1,1), (1, -1))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.0, 1.732051), (1.0, 0.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (2) --> 2 total per unit cell

julia> LatticePresets.square() |> unitcell(3)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (9) --> 9 total per unit cell

julia> supercell(LatticePresets.square(), 3) |> unitcell
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (9) --> 9 total per unit cell
```

# See also:
    `supercell`, `siteselector`
"""
unitcell(v::Union{SMatrix,Tuple,SVector,Integer}...; kw...) = lat -> unitcell(lat, v...; kw...)
unitcell(lat::Lattice, args...; kw...) = unitcell(supercell(lat, args...; kw...))

function unitcell(lat::Superlattice)
    newoffsets = supercell_offsets(lat)
    newsites = supercell_sites(lat)
    unitcell = Unitcell(newsites, lat.unitcell.names, newoffsets)
    br = Bravais(bravais(lat) * lat.supercell.matrix)
    return Lattice(br, unitcell)
end

function supercell_offsets(lat::Superlattice)
    sitecounts = zeros(Int, nsublats(lat) + 1)
    foreach_supersite((s, oldi, dn, newi) -> sitecounts[s + 1] += 1, lat)
    newoffsets = cumsum!(sitecounts, sitecounts)
    return newoffsets
end

function supercell_sites(lat::Superlattice)
    oldsites = allsitepositions(lat)
    newsites = similar(oldsites, nsites(lat.supercell))
    br = bravais(lat)
    foreach_supersite((s, oldi, dn, newi) -> newsites[newi] = br * dn + oldsites[oldi], lat)
    return newsites
end
