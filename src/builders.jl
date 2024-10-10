############################################################################################
# IJV sparse matrix builders
#region

struct IJV{B}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{B}
end

function IJV{B}(nnzguess = missing) where {B}
    i, j, v = Int[], Int[], B[]
    if nnzguess isa Integer
        sizehint!(i, nnzguess)
        sizehint!(j, nnzguess)
        sizehint!(v, nnzguess)
    end
    return IJV(i, j, v)
end

Base.push!(ijv::IJV, (i, j, v)) =
    (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))

Base.append!(ijv::IJV, (is, js, vs)) =
    (append!(ijv.i, is); append!(ijv.j, js); append!(ijv.v, vs))

Base.isempty(s::IJV) = length(s) == 0

Base.length(s::IJV) = length(s.v)

# cannot combine these two due to ambiguity with sparse(I, J, v::Number)
SparseArrays.sparse(c::IJV, m::Integer, n::Integer) = sparse(c.i, c.j, c.v, m, n)
SparseArrays.sparse(c::IJV) = sparse(c.i, c.j, c.v)

#endregion

############################################################################################
# CSC sparse matrix builder
#region

mutable struct CSC{B}   # must be mutable to update counters
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{B}
    colcounter::Int
    rowvalcounter::Int
    cosorter::CoSort{Int,B}
end

function CSC{B}(cols = missing, nnzguess = missing) where {B}
    colptr = [1]
    rowval = Int[]
    nzval = B[]
    if cols isa Integer
        sizehint!(colptr, cols + 1)
    end
    if nnzguess isa Integer
        sizehint!(nzval, nnzguess)
        sizehint!(rowval, nnzguess)
    end
    colcounter = 1
    rowvalcounter = 0
    cosorter = CoSort(rowval, nzval)
    return CSC(colptr, rowval, nzval, colcounter, rowvalcounter, cosorter)
end

function pushtocolumn!(s::CSC, row::Int, x, skipdupcheck::Bool = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
    end
    return s
end

function appendtocolumn!(s::CSC, firstrow::Int, vals, skipdupcheck::Bool = true)
    len = length(vals)
    if skipdupcheck || !any(i->isintail(firstrow + i - 1, s.rowval, s.colptr[s.colcounter]), 1:len)
        append!(s.rowval, firstrow:firstrow+len-1)
        append!(s.nzval, vals)
        s.rowvalcounter += len
    end
    return s
end

function isintail(element, container, start::Int)
    for i in start:length(container)
        container[i] == element && return true
    end
    return false
end

function sync_columns!(s::CSC, col)
    missing_cols = col - s.colcounter
    for _ in 1:missing_cols
        finalizecolumn!(s)
    end
    return nothing
end

function finalizecolumn!(s::CSC, sortcol::Bool = true)
    if sortcol
        s.cosorter.offset = s.colptr[s.colcounter] - 1
        sort!(s.cosorter)
        isgrowing(s.cosorter) || internalerror("finalizecolumn!: repeated rows")
    end
    s.colcounter += 1
    push!(s.colptr, s.rowvalcounter + 1)
    return nothing
end

function completecolptr!(colptr, cols, lastrowptr)
    colcounter = length(colptr)
    if colcounter < cols + 1
        resize!(colptr, cols + 1)
        for col in (colcounter + 1):(cols + 1)
            colptr[col] = lastrowptr + 1
        end
    end
    return colptr
end

function SparseArrays.sparse(s::CSC, m::Integer, n::Integer)
    completecolptr!(s.colptr, n, s.rowvalcounter)
    rows, cols = isempty(s.rowval) ? 0 : maximum(s.rowval), length(s.colptr) - 1
    rows <= m && cols == n ||
        internalerror("sparse: matrix size $((rows, cols)) is inconsistent with lattice size $((m, n))")
    return SparseMatrixCSC(m, n, s.colptr, s.rowval, s.nzval)
end

Base.isempty(s::CSC) = length(s) == 0

Base.length(s::CSC) = length(s.nzval)

#endregion

############################################################################################
# IJVBuilder and CSCBuilder <: AbstractHamiltonianBuilder
#region

abstract type AbstractHarmonicBuilder{L,B} end
abstract type AbstractHamiltonianBuilder{T,E,L,B} end

struct IJVHarmonic{L,B} <: AbstractHarmonicBuilder{L,B}
    dn::SVector{L,Int}
    collector::IJV{B}
end

struct CSCHarmonic{L,B} <: AbstractHarmonicBuilder{L,B}
    dn::SVector{L,Int}
    collector::CSC{B}
end

struct IJVBuilder{T,E,L,B,M<:Union{Missing,Vector{Any}}} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{IJVHarmonic{L,B}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
    modifiers::M
end

struct CSCBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{CSCHarmonic{L,B}}
end

const IJVBuilderWithModifiers = IJVBuilder{<:Any,<:Any,<:Any,<:Any,Vector{Any}}

## Constructors ##

function CSCBuilder(lat::Lattice{<:Any,<:Any,L}, blockstruct::OrbitalBlockStructure{B}) where {L,B}
    harmonics = CSCHarmonic{L,B}[]
    return CSCBuilder(lat, blockstruct, harmonics)
end

IJVBuilder(lat::Lattice{T}, orbitals, modifiers = missing) where {T} =
    IJVBuilder(lat, OrbitalBlockStructure(T, orbitals, sublatlengths(lat)), modifiers)

function IJVBuilder(lat::Lattice{T,E,L}, blockstruct::OrbitalBlockStructure{B}, modifiers = missing) where {E,T,L,B}
    harmonics = IJVHarmonic{L,B}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, blockstruct, harmonics, kdtrees, modifiers)
end

# with no modifiers
function IJVBuilder(lat::Lattice{T}, hams::Hamiltonian...) where {T}
    bs = blockstructure(lat, hams...)
    builder = IJVBuilder(lat, bs)
    push_ijvharmonics!(builder, hams...)
    return builder
end

# with some modifiers
function IJVBuilder(lat::Lattice{T}, hams::AbstractHamiltonian...) where {T}
    bs = blockstructure(lat, hams...)
    builder = IJVBuilderWithModifiers(lat, bs)
    push_ijvharmonics!(builder, hams...)
    mss = modifiers.(hams)
    mss´ = applyrange.(mss, hams)
    bis = blockindices(hams)
    unapplied_block_modifiers = ((ms, bi) -> intrablock.(parent.(ms), Ref(bi))).(mss´, bis)
    push!(builder, tupleflatten(unapplied_block_modifiers...)...)
    return builder
end

(::Type{IJVBuilderWithModifiers})(lat, orbitals) = IJVBuilder(lat, orbitals, Any[])
# add modifiers to existing builder
(::Type{IJVBuilderWithModifiers})(b::IJVBuilder) =
    IJVBuilder(b.lat, b.blockstruct, b.harmonics, b.kdtrees, Any[])
# only if it doesn't already have modifiers!
(::Type{IJVBuilderWithModifiers})(b::IJVBuilderWithModifiers) = b

push_ijvharmonics!(builder, ::OrbitalBlockStructure) = builder
push_ijvharmonics!(builder, hars::Vector{<:IJVHarmonic}) = copy!(builder.harmonics, hars)
push_ijvharmonics!(builder) = builder

function push_ijvharmonics!(builder::IJVBuilder, hs::AbstractHamiltonian...)
    offset = 0
    B = blocktype(builder)
    for h in hs
        for har in harmonics(h)
            ijv = builder[dcell(har)]
            hmat = unflat(matrix(har))
            I,J,V = findnz(hmat)
            V´ = maybe_mask_blocks(B, V)
            append!(ijv, (I .+ offset, J .+ offset, V´))
        end
        offset += nsites(lattice(h))
    end
    return builder
end

maybe_mask_blocks(::Type{B}, V::Vector{B}) where {B} = V
maybe_mask_blocks(::Type{B}, V::Vector) where {B} = mask_block.(B, V)

empty_harmonic(b::CSCBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    CSCHarmonic{L,B}(dn, CSC{B}(nsites(b.lat)))

empty_harmonic(::IJVBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    IJVHarmonic{L,B}(dn, IJV{B}())

builder(; kw...) = lat -> builder(lat; kw...)

builder(lat::Lattice; orbitals = Val(1)) = IJVBuilderWithModifiers(lat, orbitals)

## API ##

collector(har::AbstractHarmonicBuilder) = har.collector  # for IJVHarmonic and CSCHarmonic

dcell(har::AbstractHarmonicBuilder) = har.dn

kdtrees(b::IJVBuilder) = b.kdtrees

modifiers(b::IJVBuilderWithModifiers) = b.modifiers

finalizecolumn!(b::CSCBuilder, x...) =
    foreach(har -> finalizecolumn!(collector(har), x...), b.harmonics)

nsites(b::AbstractHamiltonianBuilder) = nsites(lattice(b))

ncells(b::AbstractHamiltonianBuilder) = length(harmonics(b))

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(s::CSCHarmonic) = isempty(collector(s))

lattice(b::AbstractHamiltonianBuilder) = b.lat

blockstructure(b::AbstractHamiltonianBuilder) = b.blockstruct

blocktype(::AbstractHamiltonianBuilder{<:Any,<:Any,<:Any,B}) where {B} = B

harmonics(b::AbstractHamiltonianBuilder) = b.harmonics

Base.length(b::AbstractHarmonicBuilder) = length(collector(b))

Base.push!(b::IJVBuilderWithModifiers, ms::AnyModifier...) = push!(b.modifiers, ms...)

Base.pop!(b::IJVBuilderWithModifiers) = pop!(b.modifiers)

Base.empty!(b::IJVBuilderWithModifiers) = (empty!(b.harmonics); empty!(b.modifiers); b)

Base.empty!(b::IJVBuilder) = (empty!(b.harmonics); b)

function Base.getindex(b::AbstractHamiltonianBuilder{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    hars = b.harmonics
    for har in hars
        dcell(har) == dn && return collector(har)
    end
    har = empty_harmonic(b, dn)
    push!(hars, har)
    return collector(har)
end

function SparseArrays.sparse(builder::AbstractHamiltonianBuilder{T,<:Any,L,B}) where {T,L,B}
    HT = Harmonic{T,L,B}
    b = blockstructure(builder)
    n = nsites(builder)
    hars = HT[sparse(b, har, n, n) for har in harmonics(builder) if !isempty(har)]
    return hars
end

function SparseArrays.sparse(b::OrbitalBlockStructure{B}, har::AbstractHarmonicBuilder{L,B}, m::Integer, n::Integer) where {L,B}
    s = sparse(har, m, n)
    return Harmonic(dcell(har), HybridSparseMatrix(b, s))
end

# cannot combine these two due to ambiguity with sparse(I, J, v::Number)
SparseArrays.sparse(har::AbstractHarmonicBuilder, n::Integer, m::Integer) =
    sparse(collector(har), m, n)
SparseArrays.sparse(har::AbstractHarmonicBuilder) = sparse(collector(har))

#endregion
