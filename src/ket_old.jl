#######################################################################
# KetModel
#######################################################################
struct KetModel{O<:Val,M<:TightbindingModel,R<:Union{Missing,Real}}
    model::M
    normalization::R
    maporbitals::O       # Val(false) or Val(true) to aid in type-stability
    singlesitekets::Bool
end
"""
    ketmodel(a; region = missing, sublats = missing, normalization = 1, maporbitals = false, singlesitekets = false)

Create an `KetModel` of amplitude `a` on any site in the specified `region` and `sublats`.
For single-column kets, the amplitude `a` can be a `Number`, an `AbstractVector`, or for a
position-dependent amplitude a function of the form `r -> ...` returning either. For
multi-column kets, make `a` an `AbstractMatrix{<:Number}` or a function returning one, which
will be sliced into each ket column as appropriate. An error will be thrown if the slicing
is impossible, due e.g. to a mismatch of `size(a, 1)` and the number of orbitals in an
applicable sublattice.

# Keyword arguments

If keyword `normalization` is not `missing` or `false`, each column of the ket will be
rescaled to have norm `normalization` when the `KetModel` is applied to a specific
Hamiltonian. If a ket column `iszero`, however, it will not be normalized.

If keyword `maporbitals = true` and amplitude `a` is a scalar or a scalar function, `a`
will be applied to each orbital independently. This is particularly useful in multiorbital
systems with random amplitudes, e.g. `a = r -> randn()`. If `a` is not a scalar and
`maporbitals == true`, an error will be thrown.

If keyword `singlesitekets = true`, then the model represents a multicolumn ket, where each
column (or block of columns for `a::AbstractMatrix`) has amplitude `a` on a single site of
those selected by `region` and `sublats` (as opposed to having the same amplitude `a` on all
said sites if `singlesitekets = false`). This is useful e.g. to build a basis for the selected
sites.

Keywords `region` and `sublats` are the same as for `siteselector`. Only sites at position
`r` in sublattice with name `s::NameType` will be selected if `region(r) && s in sublats` is
true. Any missing `region` or `sublat` will not be used to constraint the selection.

The keyword `sublats` allows the following formats:

    sublats = :A           # Onsite on sublat :A only
    sublats = (:A,)        # Same as above
    sublats = (:A, :B)     # Onsite on sublat :A and :B

# Ket algebra

`KetModel`s created with `ket` can added or substracted together or be multiplied by scalars
to build more elaborate `KetModel`s, e.g. `ket(1) - 3 * ket(2, region = r -> norm(r) < 10)`.
Only models with the same `maporbitals` can be combined. When combining two models with
different `singlesitekets`, the result has `singlesitekets = true`.

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
ketmodel(f; normalization = 1, maporbitals::Bool = false, singlesitekets::Bool = false, kw...) =
    KetModel(onsite(f; kw...), sanitize_normalization(normalization), Val(maporbitals), singlesitekets)

maporbitals(m::KetModel{Val{true}}) = true
maporbitals(m::KetModel{Val{false}}) = false

sanitize_normalization(::Missing) = missing
sanitize_normalization(b::Bool) = b ? 1 : missing
sanitize_normalization(b) = b

function Base.show(io::IO, ::MIME"text/plain", k::KetModel{<:Any,M}) where {N,M<:TightbindingModel{N}}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "$i  ")
    print(io, "$(i)KetModel{$N}: model with $N terms
$i  Normalization    : $(k.normalization)
$i  Map orbitals     : $(k.maporbitals)
$i  Single-site kets : $(k.singlesitekets)")
    foreach(t -> print(ioindent, "\n", t), k.model.terms)
end

Base.:*(x::Number, k::KetModel) =
    KetModel(k.model * x, k.normalization, k.maporbitals, k.singlesitekets)
Base.:*(k::KetModel, x::Number) =
    KetModel(x * k.model, k.normalization, k.maporbitals, k.singlesitekets)
Base.:-(k::KetModel) =
    KetModel(-k.model, k.normalization, k.maporbitals, k.singlesitekets)
Base.:-(k1::KetModel, k2::KetModel) = k1 + (-k2)

function Base.:+(k1::KetModel, k2::KetModel)
    newnormalization = sanitize_norm(k1.normalization, k2.normalization)
    if k1.maporbitals == k2.maporbitals
        return KetModel(k1.model + k2.model, newnormalization,
            k1.maporbitals, k1.singlesitekets || k2.singlesitekets)
    else
        throw(ArgumentError("Cannot combine ket models with different `maporbitals`"))
    end
end

sanitize_norm(n1::Number, n2::Number) = n1 ≈ n2 ? n1 : _normwarn()
sanitize_norm(::Missing, ::Missing) = missing
sanitize_norm(n1, n2) = _normwarn()

function _normwarn()
    @warn "Combining `KetModel`s with different normalizations, choosing `normalization = missing`"
    return missing
end

resolve(k::KetModel, lat::AbstractLattice) =
    KetModel(resolve(k.model, lat), k.normalization, k.maporbitals, k.singlesitekets)

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
    kmodel´ = resolve(kmodel, h.lattice)
    ncols = guess_ket_columns(kmodel´, h)
    T = orbitaltype(h)
    kmat = Matrix{T}(undef, size(h, 1), ncols)
    k = ket(kmat, h)
    return ket!(k, kmodel´, h)
end

function guess_ket_columns(km, h)
    cols = 0
    r = first(allsitepositions(h.lattice))
    if km.singlesitekets
        cols = 0
        for term in km.model.terms
            t = term(r,r)
            for (sublat, orbs) in enumerate(orbitals(h)), i in siterange(h.lattice, sublat)
                if i in term.selector
                    cols += _term_columns(t, orbs)
                end
            end
        end
    else
        # This guess assumes that all model terms have the same size(amplitude, 2)
        term = first(km.model.terms)
        cols = _term_columns(term(r, r))
    end
    return cols
end

_term_columns(::Number, orbs...) = 1
_term_columns(::AbstractVector, orbs...) = 1
_term_columns(t::AbstractMatrix, orbs...) = size(t, 2)
_term_columns(::UniformScaling, orbs) = length(orbs)
_term_columns(x, orbs...) = throw(ArgumentError("Ket model amplitude should be of type `T` or a function `a(r)::T`, where `T is a `Number`, an `AbstractVector`, an `AbstractMatrix` or, in the case of `singlesitekets = true`, a `Uniformscaling`"))

# Model application, possibly flattening if target requires it (like bloch!)
# The type instability in ket! (due to orbs in multi-orbital h's) is harmless
function ket!(k::Ket, kmodel, h)
    kmat = parent(k)
    T = eltype(kmat)
    fill!(kmat, zero(T))
    kmodel´ = resolve(kmodel, h.lattice)    # resolve sublat names into sublat indices
    coloffset = Ref(0)                      # column counter for singlesitekets = true
    for (sublat, orbs) in enumerate(orbitals(h))
        ket_applyterms_sublat!(kmat, sublat, orbs, h, kmodel´, coloffset)
    end
    @assert !kmodel.singlesitekets || coloffset[] == size(kmat, 2) "Internal error, bug in ket!"
    kmodel.normalization === missing || normalize_columns!(kmat, kmodel.normalization)
    return ket(kmat, k.orbstruct)
end

# function barrier for orbs type-stability
function ket_applyterms_sublat!(kmat::AbstractArray{T}, sublat, orbs, h, kmodel, coloffset) where {T}
    allpos = allsitepositions(h.lattice)
    orbstruct = orbitalstructure(h)
    for i in siterange(h.lattice, sublat), term in kmodel.model.terms
        if i in term.selector
            r = allpos[i]
            orbsmat = ket_orbs_matrix(kmodel.maporbitals, term, r, orbs)
            di = size(orbsmat, 2)
            if kmodel.singlesitekets
                kmatview = view(kmat, :, coloffset[]+1:coloffset[]+di)
                coloffset[] += di
            else
                di == size(kmat, 2) ||
                    throw(ArgumentError("All ket model terms should have amplitudes with the same number of columns"))
                kmatview = view(kmat, :, 1:di)  # for type stability in branch
            end
            copy_rows!(kmatview, i, orbstruct, orbsmat, orbs)
        end
    end
    return kmat
end

# should return only orbs matrix, so that to_orbtype pads with zeros correctly
# must evaluate term once per orbital, in case it is a random function
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
ensure_orbs_matrix(mat::UniformScaling, ::NTuple{N}) where {N} = SMatrix{N,N}(mat)

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