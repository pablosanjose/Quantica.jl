abstract type AbstractLattice{E,L,T} end

#######################################################################
# Sublat (sublattice)
#######################################################################
struct Sublat{E,T}
    sites::Vector{SVector{E,T}}
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
sublat(sites::Vector{<:SVector}; name = :_, kw...) =
    Sublat(sites, nametype(name))
sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = sublat(toSVectors(vs...); kw...)

toSVectors(vs...) = [promote(toSVector.(vs)...)...]

#######################################################################
# Bravais
#######################################################################
struct Bravais{E,L,T,EL}
    matrix::SMatrix{E,L,T,EL}
    semibounded::SVector{L,Bool}
end

Bravais{E,T}() where {E,T} = Bravais(SMatrix{E,0,T,0}(), SVector{0,Bool}())

displayvectors(br::Bravais) = displayvectors(br.matrix)

Base.show(io::IO, b::Bravais{E,L,T}) where {E,L,T} = print(io,
"Bravais{$E,$L,$T} : set of $L Bravais vectors in $(E)D space.
  Vectors     : $(displayvectors(b))
  Matrix      : $(b.matrix),
  Semibounded : $(issemibounded(b) ? display_as_tuple(findall(b.semibounded)) : "none")")

# External API #

"""
    bravais(vecs...; semibounded = false)
    bravais(matrix; semibounded = false)

Create a `Bravais{E,L}` that adds `L` Bravais vectors `vecs` in `E` dimensional space,
alternatively given as the columns of matrix `mat`. For higher instantiation efficiency
enter `vecs` as `Tuple`s or `SVector`s and `mat` as `SMatrix`.

To create semibounded lattices along some or all Bravais vectors, use `semibounded`. A
`semibounded = true` makes all axes semibounded. A `semibounded = (axes::Int...)` indicates
the indices of axes to be made semibounded. A tuple of Booleans, one per Bravais axes,
`semibounded = issemi::NTuple{L,Bool}` is also allowed. Note that semibounded lattices
always extend toward positive multiples of Bravais vectors. To invert the direction, invert
the vectors.

We can scale a `b::Bravais` simply by multiplying it with a factor `a`, like `a * b`.

    bravais(lat::Lattice)
    bravais(h::Hamiltonian)

Obtain the Bravais matrix of lattice `lat` or Hamiltonian `h`

# Examples

```jldoctest
julia> bravais((1.0, 2), (3, 4))
Bravais{2,2,Float64} : set of 2 Bravais vectors in 2D space.
  Vectors     : ((1.0, 2.0), (3.0, 4.0))
  Matrix      : [1.0 3.0; 2.0 4.0],
  Semibounded : none
```

# See also:
    `semibounded`
"""
bravais(vs::Union{Tuple,AbstractVector,AbstractMatrix}...; semibounded = false, kw...) =
    (s = toSMatrix(vs...); Bravais(s, sanitize_semibounded(semibounded, s)))

bravais(lat::AbstractLattice) = lat.bravais.matrix

sanitize_semibounded(sb::Bool, ::SMatrix{E,L}) where {E,L} =
    SVector{L,Bool}(filltuple(sb, Val(L))) # need to specify type becaus of L=0 case
sanitize_semibounded(sb::Int, s::SMatrix{E,L}) where {E,L} =
    sanitize_semibounded((sb,), s)
sanitize_semibounded(sb::NTuple{L,Bool}, ::SMatrix{E,L}) where {E,L} = SVector{L,Bool}(sb)
sanitize_semibounded(sb, ::SMatrix{E,L}) where {E,L} =
    SVector{L,Bool}(ntuple(i -> i in sb, Val(L)))

transform(f::F, b::Bravais{E,0}) where {E,F<:Function} = b

function transform(f::F, b::Bravais{E,L,T}) where {E,L,T,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix, b.semibounded)
end

Base.:*(factor::Number, b::Bravais) = Bravais(factor * b.matrix, b.semibounded)
Base.:*(b::Bravais, factor::Number) = Bravais(b.matrix * factor, b.semibounded)

issemibounded(b::Bravais) = !iszero(b.semibounded)

#######################################################################
# Unitcell
#######################################################################
struct Unitcell{E,T,N}
    sites::Vector{SVector{E,T}}
    names::NTuple{N,NameType}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end                             # so that diff(offset) == sublatsites


Unitcell(sublats::Sublat...; kw...) = Unitcell(promote(sublats...); kw...)
Unitcell(sublats::NTuple{N,Sublat{E,T}};
    dim = Val(E), type = float(T), names = [s.name for s in sublats], kw...) where {N,E,T} =
    _Unitcell(sublats, dim, type, names)

# Dynamic dispatch
_Unitcell(sublats, dim::Integer, type, names) = _Unitcell(sublats, Val(dim), type, names)

function _Unitcell(sublats::NTuple{N,Sublat}, dim::Val{E}, type::Type{T}, names) where {N,E,T}
    sites = SVector{E,T}[]
    offsets = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sublats[s].sites
            push!(sites, padright(site, Val(E)))
        end
        push!(offsets, length(sites))
    end
    names´ = uniquenames(sanitize_names(names, Val(N)))
    return Unitcell(sites, names´, offsets)
end

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

sites(u::Unitcell) = u.sites
sites(u::Unitcell, s::Int) = view(u.sites, siterange(u, s))

siteindex(u::Unitcell, sublat, idx) = idx + u.offsets[sublat]

siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

enumeratesites(u::Unitcell, sublat) = ((i, sites(u)[i]) for i in siterange(u, sublat))

nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatsites(u)[sublat]

offsets(u::Unitcell) = u.offsets

function sublat(u::Unitcell, siteidx)
    l = length(u.offsets)
    for s in 2:l
        @inbounds u.offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

# function boundingbox(u::Unitcell{E,T}) where {E,T}
#     min´ = max´ = first(u.sites)
#     for r in u.sites
#         min´ = min.(min´, r)
#         max´ = max.(max´, r)
#     end
#     return min´, max´
# end

sublatsites(u::Unitcell) = diff(u.offsets)

nsublats(u::Unitcell) = length(u.names)

sublats(u::Unitcell) = 1:nsublats(u)

sublatname(u::Unitcell, s) = u.names[s]

transform!(f::Function, u::Unitcell) = (u.sites .= f.(u.sites); u)

Base.copy(u::Unitcell) = Unitcell(copy(u.sites), u.names, copy(u.offsets))

Base.isequal(u1::Unitcell, u2::Unitcell) =
    isequal(u1.sites, u2.sites) && isequal(u1.names, u2.names) && isequal(u1.offsets, u2.offsets)

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
    hassemi = issemibounded(lat.bravais)
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))", hassemi ? "
$i    Semibounded   : $(display_as_tuple(findall(lat.bravais.semibounded)))" : "", "
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell")
end

Base.summary(::Lattice{E,L,T}) where {E,L,T} =
    "Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space"

# External API #

"""
    lattice([bravais::Bravais,] sublats::Sublat...; dim::Val{E}, type::T, names)

Create a `Lattice{E,L,T}` with Bravais matrix `bravais` and sublattices `sublats`
converted to a common  `E`-dimensional embedding space and type `T`. To override the
embedding  dimension `E`, use keyword `dim = Val(E)`. Similarly, override type `T` with
`type = T`.

The keywords `names` can be used to rename `sublats`. Given names can be replaced to ensure
that all sublattice names are unique.

See also `LatticePresets` for built-in lattices.

# Examples

```jldoctest
julia> lattice(bravais((1, 0)), sublat((0, 0)), sublat((0, Float32(1))); dim = Val(3))
Lattice{3,1,Float32} : 1D lattice in 3D space
  Bravais vectors : ((1.0f0, 0.0f0, 0.0f0),)
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.honeycomb(semibounded = 1, names = (:C, :D))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
    Semibounded   : (1)
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell
```

# See also:
    `LatticePresets`, `bravais`, `sublat`, `supercell`, `intracell`
"""
lattice(s::Sublat, ss::Sublat...; kw...) = _lattice(Unitcell(s, ss...; kw...))
_lattice(u::Unitcell{E,T}) where {E,T} = Lattice(Bravais{E,T}(), u)
lattice(br::Bravais, s::Sublat, ss::Sublat...; kw...) = lattice(br, Unitcell(s, ss...; kw...))

function lattice(bravais::Bravais{E2,L2}, unitcell::Unitcell{E,T}) where {E2,L2,E,T}
    L = min(E,L2) # L should not exceed E
    Lattice(convert(Bravais{E,L,T}, bravais), unitcell)
end

issemibounded(lat::Lattice) = issemibounded(lat.bravais)

"""
    dims(lat::Lattice{E,L}) -> (E, L)

Return a tuple `(E, L)` of the embedding and lattice dimensions of `lat`
"""
dims(lat::Lattice{E,L}) where {E,L} = E, L

#######################################################################
# Supercell
#######################################################################
struct Supercell{L,L´,M<:Union{Missing,OffsetArray{Bool}},S<:SMatrix{L,L´}} # L´ is supercell dim
    matrix::S
    sites::UnitRange{Int}
    cells::CartesianIndices{L,NTuple{L,UnitRange{Int}}}
    mask::M
    semibounded::SVector{L´,Bool}
end

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell{L,L´,<:OffsetArray}) where {L,L´} = sum(s.mask)
nsites(s::Supercell{L,L´,Missing}) where {L,L´} = length(s.sites) * length(s.cells)

Base.CartesianIndices(s::Supercell) = s.cells

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    hassemi = issemibounded(s)
    i = get(io, :indent, "")
    print(io, i,
"Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice
$i  Supervectors  : $(displayvectors(s.matrix))
$i  Supersites    : $(nsites(s))", hassemi ? "
$i  Semibounded   : $(display_as_tuple(findall(s.semibounded)))" : "")
end

ismasked(s::Supercell{L,L´,<:OffsetArray})  where {L,L´} = true
ismasked(s::Supercell{L,L´,Missing})  where {L,L´} = false

isinmask(mask::Missing, inds::Vararg{Any,N}) where {N} = true
isinmask(mask::OffsetArray, inds::Vararg{Any,N}) where {N} = checkbounds(Bool, mask, inds...) && mask[inds...]

issemibounded(sc::Supercell) = !iszero(sc.semibounded)

Base.copy(s::Supercell{<:Any,<:Any,Missing}) =
    Supercell(s.matrix, s.sites, s.cells, s.mask, s.semibounded)

Base.copy(s::Supercell{<:Any,<:Any,<:AbstractArray}) =
    Supercell(s.matrix, s.sites, s.cells, copy(s.mask), s.semibounded)

Base.isequal(s1::Supercell, s2::Supercell) =
    isequal(s1.matrix, s2.matrix) && isequal(s1.sites, s2.sites) &&
    isequal(s1.cells, s2.cells) && isequal(s1.mask, s2.mask) &&
    isequal(s1.semibounded, s2.semibounded)

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
    return Supercell(s1.matrix, s1.sites, cells, mask´, s1.semibounded)
end

function check_compatible_supercell(s1, s2)
    compatible = isequal(s1.matrix, s2.matrix) && isequal(s1.sites, s2.sites) &&
                 isequal(s1.semibounded, s2.semibounded)
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
    hassemi = issemibounded(lat.bravais)
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))", hassemi ? "
$i    Semibounded   : $(display_as_tuple(findall(lat.bravais.semibounded)))" : "", "
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell\n")
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

issemibounded(lat::Superlattice) = issemibounded(lat.supercell)

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

numbertype(::AbstractLattice{E,L,T}) where {E,L,T} = T
positiontype(::AbstractLattice{E,L,T}) where {E,L,T} = SVector{E,T}
dntype(::AbstractLattice{E,L}) where {E,L} = SVector{L,Int}

sublat(lat::AbstractLattice, siteidx) = sublat(lat.unitcell, siteidx)
sublats(lat::AbstractLattice) = sublats(lat.unitcell)

siterange(lat::AbstractLattice, sublat) = siterange(lat.unitcell, sublat)

siteindex(lat::AbstractLattice, sublat, idx) = siteindex(lat.unitcell, sublat, idx)

offsets(lat::AbstractLattice) = offsets(lat.unitcell)

sublatsites(lat::AbstractLattice) = sublatsites(lat.unitcell)

enumeratesites(lat::AbstractLattice, sublat) = enumeratesites(lat.unitcell, sublat)

sublatname(lat::AbstractLattice, s = sublats(lat)) = sublatname(lat.unitcell, s)

# function boundingbox(lat::AbstractLattice{E,L}, dns = (zero(SVector{L,Int}),)) where {E,L}
#     minn, maxn = min0, max0 = boundingbox(lat.unitcell)
#     for dn in dns
#         minn = min.(minn, min0 + bravais(lat) * dn)
#         maxn = max.(maxn, max0 + bravais(lat) * dn)
#     end
#     return minn, maxn
# end

nsites(lat::AbstractLattice) = nsites(lat.unitcell)
nsites(lat::AbstractLattice, sublat) = nsites(lat.unitcell, sublat)

nsublats(lat::AbstractLattice) = nsublats(lat.unitcell)

issuperlattice(lat::Lattice) = false
issuperlattice(lat::Superlattice) = true

ismasked(lat::Lattice) = false
ismasked(lat::Superlattice) = ismasked(lat.supercell)

maskranges(lat::Superlattice) = (1:nsites(lat), lat.supercell.cells.indices...)
maskranges(lat::Lattice) = (1:nsites(lat),)

# External API #

"""
    sites(lat[, sublat::Int])

Extract the positions of all sites in a lattice, or in a specific sublattice
"""
sites(lat::AbstractLattice) = sites(lat.unitcell)
sites(lat::AbstractLattice, s) = sites(lat.unitcell, s)

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

function combined_offsets(us::Unitcell...)
    nsubs = sum(nsublats, us)
    offsets = Vector{Int}(undef, nsubs + 1)
    lastoffset = 0
    idx = 1
    for u in us
        for idx´ in 1:(length(u.offsets) - 1)
            offsets[idx] = lastoffset + u.offsets[idx´]
            idx += 1
        end
        lastoffset = u.offsets[end]
    end
    offsets[end] = lastoffset + last(us).offsets[end]
    return offsets
end

#######################################################################
# supercell
#######################################################################
"""
    supercell(lat::AbstractLattice{E,L}, v::NTuple{L,Integer}...; region = missing, seed = missing)
    supercell(lat::AbstractLattice{E,L}, sc::SMatrix{L,L´,Int}; region = missing, seed = missing)

Generates a `Superlattice` from an `L`-dimensional lattice `lat` with Bravais vectors
`br´= br * sc`, where `sc::SMatrix{L,L´,Int}` is the integer supercell matrix with the `L´`
vectors `v`s as columns. If no `v` are given, the superlattice will be bounded. If `lat` has
semibounded axes, these cannot be mixed with any other axes (they can only be removed, kept
intact or scaled by a factor, see below).

Only sites at position `r` such that `region(r) == true` will be included in the supercell.
The search for included sites will start from point `seed::Union{Tuple,SVector}`, or the
origin if missing. If `region` is missing, a Bravais unit cell perpendicular to the `v` axes
will be selected for the `L-L´` non-periodic directions.

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
    `unitcell`
"""
supercell(v::Union{SMatrix,Tuple,SVector,Integer}...; kw...) = lat -> supercell(lat, v...; kw...)

supercell(lat::AbstractLattice{E,L}, v::Tuple{} = (); kw...) where {E,L} =
    supercell(lat, SMatrix{L,0,Int}(); kw...)
supercell(lat::AbstractLattice{E,L}, factors::Vararg{<:Integer,L}; kw...) where {E,L} =
    _supercell(lat, factors...)
supercell(lat::AbstractLattice{E,L}, factors::Vararg{<:Integer,L´}; kw...) where {E,L,L´} =
    throw(ArgumentError("Provide either a single scaling factor or one for each of the $L lattice dimensions"))
supercell(lat::AbstractLattice{E,L}, factor::Integer; kw...) where {E,L} =
    _supercell(lat, ntuple(_ -> factor, Val(L))...)
supercell(lat::AbstractLattice, vecs::NTuple{L,Integer}...; region = missing, seed = missing) where {L} =
    _supercell(lat, toSMatrix(Int, vecs...), region, seed)
supercell(lat::AbstractLattice, s::SMatrix; region = missing, seed = missing) =
    _supercell(lat, s, region, seed)

function _supercell(lat::AbstractLattice{E,L}, factors::Vararg{Integer,L}) where {E,L}
    scmatrix = SMatrix{L,L,Int}(Diagonal(SVector(factors)))
    sites = 1:nsites(lat)
    cells = CartesianIndices((i -> 0 : i - 1).(factors))
    mask = missing
    semibounded = supercell_semibounded(lat, scmatrix)
    supercell = Supercell(scmatrix, sites, cells, mask, semibounded)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

function _supercell(lat::AbstractLattice{E,L}, scmatrix::SMatrix{L,L´,Int}, region, seed) where {E,L,L´}
    semibounded = supercell_semibounded(lat, scmatrix)
    brmatrix = lat.bravais.matrix
    regionfunc = region === missing ? ribbonfunc(brmatrix, scmatrix) : region
    in_supercell_func = is_perp_dir(scmatrix)
    cells = supercell_cells(lat, regionfunc, in_supercell_func, seed)
    ns = nsites(lat)
    mask = OffsetArray(BitArray(undef, ns, size(cells)...), 1:ns, cells.indices...)
    @inbounds for dn in cells
        dntup = Tuple(dn)
        dnvec = toSVector(Int, dntup)
        in_supercell = in_supercell_func(dnvec)
        in_supercell || (mask[:, dntup...] .= false; continue)
        r0 = brmatrix * dnvec
        for (i, site) in enumerate(lat.unitcell.sites)
            r = site + r0
            mask[i, dntup...] = in_supercell && regionfunc(r)
        end
    end
    supercell = Supercell(scmatrix, 1:ns, cells, all(mask) ? missing : mask, semibounded)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

# This is true whenever old ndist is perpendicular to new lattice
is_perp_dir(supercell) = let invs = pinvmultiple(supercell); dn -> iszero(new_dn(dn, invs)); end

new_dn(oldndist, (pinvs, n)) = fld.(pinvs * oldndist, n)
new_dn(oldndist, ::Tuple{<:SMatrix{0,0},Int}) = SVector{0,Int}()

function ribbonfunc(bravais::SMatrix{E,L,T}, supercell::SMatrix{L,L´}) where {E,L,T,L´}
    L <= L´ && return r -> true
    # real-space supercell axes + all space
    s = hcat(bravais * supercell, SMatrix{E,E,T}(I))
    q = qr(s).Q
    # last vecs in Q are orthogonal to axes
    os = ntuple(i -> SVector(q[:,i+L´]), Val(E-L´))
    # range of projected bravais, including zero
    ranges = (o -> extrema(vcat(SVector(zero(T)), bravais' * o)) .- sqrt(eps(T))).(os)
    # projector * r gives the projection of r on orthogonal vectors
    projector = hcat(os...)'
    # ribbon defined by r's with projection within ranges for all orthogonal vectors
    regionfunc(r) = all(first.(ranges) .<= Tuple(projector * r) .< last.(ranges))
    return regionfunc
end

function supercell_cells(lat::Lattice{E,L}, regionfunc, in_supercell_func, seed) where {E,L}
    bravais = lat.bravais.matrix
    seed´ = seed === missing ? zero(SVector{L,Int}) : seedcell(SVector{E}(seed), bravais)
    iter = BoxIterator(seed´)
    foundfirst = false
    counter = 0
    for dn in iter
        found = false
        counter += 1; counter == TOOMANYITERS &&
            throw(ArgumentError("`region` seems unbounded (after $TOOMANYITERS iterations)"))
        in_supercell = in_supercell_func(toSVector(Int, dn))
        r0 = bravais * toSVector(Int, dn)
        for site in lat.unitcell.sites
            r = r0 + site
            found = in_supercell && regionfunc(r)
            if found || !foundfirst
                acceptcell!(iter, dn)
                foundfirst = found
                break
            end
        end
    end
    c = CartesianIndices(iter)
    return c
end

seedcell(seed::NTuple{N,Any}, brmat) where {N} = seedcell(SVector{N}(seed), brmat)
seedcell(seed::SVector{E}, brmat::SMatrix{E}) where {E} = round.(Int, brmat \ seed)

supercell_semibounded(lat::Lattice, s::SMatrix) = supercell_semibounded(lat.bravais, s)
function supercell_semibounded(b::Bravais, s::SMatrix{L,L´,Int}) where {L,L´}
    lsb = b.semibounded
    err = ArgumentError("Cannot mix semibounded axes in a supercell")
    ssb = ntuple(Val(L´)) do i
        vec_semi = s[lsb, i]
        vec_other = s[(!).(lsb), i]
        if iszero(vec_other)
            count(!iszero, vec_semi) != 1 && throw(err)
            return true
        else
            iszero(vec_semi) || throw(err)
            return false
        end
    end
    return SVector{L´,Bool}(ssb)
end

#######################################################################
# unitcell
#######################################################################
"""
    unitcell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; region = missing, seed = missing)
    unitcell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; region = missing, seed = missing)

Generates a `Lattice` from an `L`-dimensional lattice `lat` and a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
unitcell matrix, with the `L´` vectors `v`s as columns. If no `v` are given, the new lattice
will be bounded.

Only sites at position `r` such that `region(r) == true` will be included in the new
unitcell. The search for included sites will start from point `seed::Union{Tuple,SVector}`,
or the origin if missing. If `region` is missing, a Bravais unitcell perpendicular to the
`v` axes will be selected for the `L-L´` non-periodic directions.

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
when building `h` to have a modifier applied to them later. Note also that additional
onsites and hoppings may be stored when calling `optimize!` or `bloch`/`bloch!` on `h` for
the first time.

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
    `supercell`
"""
unitcell(v::Union{SMatrix,Tuple,SVector,Integer}...; kw...) = lat -> unitcell(lat, v...; kw...)
unitcell(lat::Lattice, args...; kw...) = unitcell(supercell(lat, args...; kw...))

function unitcell(lat::Superlattice)
    newoffsets = supercell_offsets(lat)
    newsites = supercell_sites(lat)
    unitcell = Unitcell(newsites, lat.unitcell.names, newoffsets)
    bravais = Bravais(lat.bravais.matrix * lat.supercell.matrix, lat.supercell.semibounded)
    return Lattice(bravais, unitcell)
end

function supercell_offsets(lat::Superlattice)
    sitecounts = zeros(Int, nsublats(lat) + 1)
    foreach_supersite((s, oldi, dn, newi) -> sitecounts[s + 1] += 1, lat)
    newoffsets = cumsum!(sitecounts, sitecounts)
    return newoffsets
end

function supercell_sites(lat::Superlattice)
    newsites = similar(lat.unitcell.sites, nsites(lat.supercell))
    oldsites = lat.unitcell.sites
    bravais = lat.bravais.matrix
    foreach_supersite((s, oldi, dn, newi) -> newsites[newi] = bravais * dn + oldsites[oldi], lat)
    return newsites
end
