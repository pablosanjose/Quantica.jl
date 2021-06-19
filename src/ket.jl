#######################################################################
# KetModel
#######################################################################
struct KetModel{O<:Val,M<:TightbindingModel,R<:Union{Missing,Real}}
    model::M
    normalization::R
    maporbitals::O
end
"""
    ketmodel(a; region = missing, sublats = missing, normalization = 1, maporbitals = false)

Create an `KetModel` of amplitude `a` on the specified `region` and `sublats`. For
single-column kets, the amplitude `a` can be a `Number`, an `AbstractVector`, or for a
position-dependent amplitude a function of the form `r -> ...` returning either. For
multi-column kets, make `a` an `AbstractMatrix{<:Number}` or a function returning one, which
will be sliced into each ket column as appropriate. An error will be thrown if the slicing
is impossible, due e.g. to a mismatch of `size(a, 1)` and the number of orbitals in an
applicable sublattice.

# Keyword arguments

If keyword `normalization` is not `missing`, each column of the ket will be rescaled to have
norm `normalization` when the `KetModel` is applied to a specific Hamiltonian. If a ket
column `iszero`, however, it will not be normalized.

If keyword `maporbitals == true` and amplitude `a` is a scalar or a scalar function, `a`
will be applied to each orbital independently. This is particularly useful in multiorbital
systems with random amplitudes, e.g. `a = r -> randn()`. If `a` is not a scalar and
`maporbitals == true`, an error will be thrown.

Keywords `region` and `sublats` are the same as for `siteselector`. Only sites at position
`r` in sublattice with name `s::NameType` will be selected if `region(r) && s in sublats` is
true. Any missing `region` or `sublat` will not be used to constraint the selection.

The keyword `sublats` allows the following formats:

    sublats = :A           # Onsite on sublat :A only
    sublats = (:A,)        # Same as above
    sublats = (:A, :B)     # Onsite on sublat :A and :B

# Ket algebra

`KetModel`s created with `ket` can added or substracted together or be multiplied by scalars
to build more elaborate `KetModel`s, e.g. `ket(1) - 3 * ket(2, region = r -> norm(r) < 10)`

# Examples

```jldoctest
julia> k = ketmodel(1, sublats=:A) - ketmodel(1, sublats=:B)
KetModel{2}: model with 2 terms
  Normalization : 1
  Map orbitals  : Val{false}()
  OnsiteTerm{Int64}:
    Sublattices      : A
    Coefficient      : 1
  OnsiteTerm{Int64}:
    Sublattices      : B
    Coefficient      : -1
```
# See also
    `ket`, `onsite`, `orbitalstructure`

"""
ketmodel(f; normalization = 1, maporbitals::Bool = false, kw...) =
    KetModel(onsite(f; kw...), normalization, Val(maporbitals))

maporbitals(m::KetModel{Val{true}}) = true
maporbitals(m::KetModel{Val{false}}) = false

function Base.show(io::IO, ::MIME"text/plain", k::KetModel{<:Any,M}) where {N,M<:TightbindingModel{N}}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "$i  ")
    print(io, "$(i)KetModel{$N}: model with $N terms
$i  Normalization : $(k.normalization)
$i  Map orbitals  : $(k.maporbitals)")
    foreach(t -> print(ioindent, "\n", t), k.model.terms)
end

Base.:*(x::Number, k::KetModel) = KetModel(k.model * x, k.normalization, k.maporbitals)
Base.:*(k::KetModel, x::Number) = KetModel(x * k.model, k.normalization, k.maporbitals)
Base.:-(k::KetModel) = KetModel(-k.model, k.normalization, k.maporbitals)
Base.:-(k1::KetModel, k2::KetModel) = KetModel(k1.model - k2.model, _checknorm(k1.normalization, k2.normalization), _andVal(k1.maporbitals, k2.maporbitals))
Base.:+(k1::KetModel, k2::KetModel) = KetModel(k1.model + k2.model, _checknorm(k1.normalization, k2.normalization), _andVal(k1.maporbitals, k2.maporbitals))

_andVal(::Val{A},::Val{B}) where {A,B} = Val(A && B)

function _checknorm(n1, n2)
    n1 ≈ n2 || @warn "Combining `KetModel`s with different normalizations, choosing $n1"
    return n1
end

resolve(k::KetModel, lat::AbstractLattice) = KetModel(resolve(k.model, lat), k.normalization, k.maporbitals)

#######################################################################
# Ket
#######################################################################
# The eltype T of any ket must be equal to orbitaltype(orbstruct), i.e. Number or SVector
struct Ket{T,M<:AbstractMatrix{T},O<:OrbitalStructure{T}} <: AbstractMatrix{T}
    amplitudes::M
    orbstruct::O
    function Ket{T,M,O}(amplitudes, orbstruct) where {T,M<:AbstractMatrix{T},O<:OrbitalStructure{T}}
        check_compatible_ket(amplitudes, orbstruct)
        new(amplitudes, orbstruct)
    end
end

Ket(amplitudes::M, orbstruct::O) where {T,M<:AbstractMatrix{T},O<:OrbitalStructure{T}} =
    Ket{T,M,O}(amplitudes, orbstruct)

Ket(amplitudes::M, orbstruct::O) where {T,M<:AbstractMatrix,O<:OrbitalStructure{T}} =
    Ket{T,M,O}(T.(amplitudes), orbstruct)

check_compatible_ket(a::AbstractMatrix{T}, o::OrbitalStructure) where {T} =
    T == orbitaltype(o) && size(a, 1) == dimh(o) ||
        throw(ArgumentError("Ket is incompatible with OrbitalStructure"))

check_compatible_ket(ket::Ket, h::Hamiltonian) =
    orbitalstructure(ket) == orbitalstructure(h) ||
        throw(ArgumentError("Ket is incompatible with Hamiltonian"))

function Base.show(io::IO, ::MIME"text/plain", k::Ket{T}) where {T}
    ioindent = IOContext(io, :indent => "  ")
    print(io, "Ket{$T}: ket with a $(size(k.amplitudes, 1)) × $(size(k.amplitudes, 2)) amplitude matrix\n")
  show(ioindent, k.orbstruct)
end

orbitalstructure(k::Ket) = k.orbstruct

# AbstractArray interface
Base.getindex(k::Ket, args...) = getindex(k.amplitudes, args...)
Base.setindex!(k::Ket, args...) = setindex!(k.amplitudes, args...)
Base.firstindex(k::Ket) = firstindex(k.amplitudes)
Base.lastindex(k::Ket) = lastindex(k.amplitudes)
Base.length(k::Ket) = length(k.amplitudes)
Base.size(k::Ket, args...) = size(k.amplitudes)
Base.IndexStyle(k::Ket) = IndexStyle(k.amplitudes)
Base.similar(k::Ket) = Ket(similar(k.amplitudes), k.orbstruct)
Base.parent(k::Ket) = k.amplitudes

"""
    ket(m::AbstractArray, o::OrbitalStructure)
    ket(m::AbstractArray, h::Hamiltonian)

Construct a `Ket` `|k⟩` with amplitudes `⟨i|k⟩ = m[i]`, which can be scalars or `SVector`s
depending on the number of orbitals on site `i`. If `m` is an `AbstractMatrix` instead of an
`AbstractVector`, the `Ket` represents a multi-column ket (i.e. a collection of kets `|kⱼ⟩`,
one per column), such that `⟨i|kⱼ⟩ = m[i,j]`. The orbitals per sublattice are encoded in `o
= orbitalstructure(h)`.

    ket(km::KetModel, h::Hamiltonian)

Construct a `Ket` by applying model `km` to Hamiltonian `h` (see also `ketmodel` for
details).

    ket(h::Hamiltonian)

Construct a zero ket, equivalent to `ket(0, h; maporbitals = true, normalization = missing)`

# See also
    `ketmodel`, `ket!`, `onsite`
"""
ket(m::AbstractVector, o::OrbitalStructure) = Ket(hcat(m), o)
ket(m::AbstractMatrix, o::OrbitalStructure) = Ket(m, o)
ket(m::AbstractArray, h::Hamiltonian) = ket(m, orbitalstructure(h))
ket(h::Hamiltonian) = ket(ketmodel(0; maporbitals = true, normalization = missing), h)

#######################################################################
# flatten and unflatten
#######################################################################

flatten(k::Ket) = ket(flatten(parent(k), orbitalstructure(k)), flatten(orbitalstructure(k)))

unflatten(k::Ket, o::OrbitalStructure) = Ket(unflatten_orbitals(parent(k), o), o)

#######################################################################
# ket(::KetModel, ::Hamiltonian)
#######################################################################
function ket(kmodel::KetModel, h::Hamiltonian)
    ncols = guess_ket_columns(kmodel, h)
    T = orbitaltype(h)
    kmat = Matrix{T}(undef, size(h, 1), ncols)
    k = ket(kmat, h)
    return ket!(k, kmodel, h)
end

function guess_ket_columns(km, h)
    term = first(km.model.terms)
    r = first(allsitepositions(h.lattice))
    t = term(r, r)
    return _guess_ket_columns(t)
end

_guess_ket_columns(::Number) = 1
_guess_ket_columns(::AbstractVector) = 1
_guess_ket_columns(t::AbstractMatrix) = size(t, 2)

# Model application, possibly flattening if target requires it (like bloch!)
# The type instability in ket! (due to orbs in multi-orbital h's) is harmless
function ket!(k::Ket, kmodel, h)
    kmat = parent(k)
    T = eltype(kmat)
    fill!(kmat, zero(T))
    kmodel´ = resolve(kmodel, h.lattice)    # resolve sublat names into sublat indices
    for (sublat, orbs) in enumerate(orbitals(h))
        ket_applyterms_sublat!(kmat, sublat, orbs, h, kmodel´)
    end
    kmodel.normalization === missing || normalize_columns!(kmat, kmodel.normalization)
    return ket(kmat, k.orbstruct)
end

# function barrier for orbs type-stability
function ket_applyterms_sublat!(kmat::AbstractArray{T}, sublat, orbs, h, kmodel) where {T}
    allpos = allsitepositions(h.lattice)
    orbstruct = orbitalstructure(h)
    for i in siterange(h.lattice, sublat), term in kmodel.model.terms
        if i in term.selector
            r = allpos[i]
            orbsmat = ket_orbs_matrix(kmodel.maporbitals, term, r, orbs)
            copy_rows!(kmat, i, orbstruct, orbsmat, orbs)
        end
    end
    return kmat
end

# should return only orbs rows, so that to_orbtype pads with zeros correctly
ket_orbs_matrix(::Val{true}, term, r, orbs) = rows_to_matrix((orb -> ensure_singlerow(term(r, r))).(orbs))
ket_orbs_matrix(::Val{false}, term, r, orbs) = ensure_orbs_matrix(term(r, r), orbs)

ensure_singlerow(row::Number) = row
ensure_singlerow(row) = size(row, 1) == 1 ? row :
    throw(ArgumentError("Expected a scalar or a single-row matrix in ket model with `maporbitals = true`, got $(size(row, 1)) rows"))

rows_to_matrix(rows::NTuple{<:Any,Number}) = hcat(vcat(rows...))
rows_to_matrix(rows::NTuple{<:Any,AbstractVector}) = hcat(vcat(rows...))
rows_to_matrix(rows) = vcat(rows...)

ensure_orbs_matrix(x::Number, ::NTuple{N}) where {N} = N == 1 ? SMatrix{1,1}(x) :
    throw(ArgumentError("Expected an array with $N rows in ket model with `maporbitals = false`, got a scalar"))
ensure_orbs_matrix(v::AbstractVector, ::NTuple{N}) where {N} = length(v) == N ? hcat(v) :
    throw(ArgumentError("Expected an array with $N rows in ket model with `maporbitals = false`, got a vector of length $(length(v))"))
ensure_orbs_matrix(mat::AbstractMatrix, ::NTuple{N}) where {N} = size(mat, 1) == N ? mat :
    throw(ArgumentError("Expected an array with $N rows in ket model with `maporbitals = false`, got $(size(mat, 1)) rows"))

# copy SVectors
function copy_rows!(kmat::AbstractMatrix{T}, i, orbstruct, orbsmat, orbs) where {T<:SVector}
    kmat[i, :] .+= to_orbtype.(eachcol(orbsmat), T, Ref(orbs))
    return kmat
end

# copy Scalars
function copy_rows!(kmat::AbstractMatrix{T}, i, orbstruct, orbsmat, orbs) where {T<:Number}
    row = flatoffset_site(i, orbstruct) + 1
    dr = size(orbsmat, 1)
    kmat[row:row+dr-1, :] .+= T.(orbsmat)
    return kmat
end

to_orbtype(t::Number, ::Type{S}, t1::NTuple{1}) where {S<:SVector} = padtotype(t, S)
to_orbtype(t::AbstractVector, ::Type{S}, t1::NTuple{N}) where {N,S} = padtotype(SVector{N}(t), S)
