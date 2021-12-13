############################################################################################
# Abstract Hamiltonian builders
#region

abstract type AbstractSparseBuilder{T,E,L,B} end

lattice(b::AbstractSparseBuilder) = b.lat

orbitalstructure(b::AbstractSparseBuilder) = b.orbstruct

function Base.getindex(b::AbstractSparseBuilder{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    hars = b.harmonics
    for har in hars
        har.dn == dn && return collector(har)
    end
    har = empty_harmonic(b, dn)
    push!(hars, har)
    return collector(har)
end

function harmonics(builder::AbstractSparseBuilder{<:Any,<:Any,L,B}) where {L,B}
    HT = Harmonic{L,SparseMatrixCSC{B,Int}}
    n = nsites(lattice(builder))
    hars = HT[HT(har.dn, sparse(collector(har), n)) for har in builder.harmonics if !isempty(har)]
    return hars
end

collector(har) = har.collector  # for IJVHarmonic and CSCHarmonic

#endregion

############################################################################################
# IJV Hamiltonian builders
#region

struct IJV{B}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{B}
end

struct IJVHarmonic{L,B}
    dn::SVector{L,Int}
    collector::IJV{B}
end

struct IJVBuilder{T,E,L,B} <: AbstractSparseBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    orbstruct::OrbitalStructure{B}
    harmonics::Vector{IJVHarmonic{L,B}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
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

empty_harmonic(::IJVBuilder{<:Any,<:Any,L,B}, dn) where {L,B} = IJVHarmonic{L,B}(dn, IJV{B}())

function IJVBuilder(lat::Lattice{T,E,L}, orbstruct::OrbitalStructure{B}) where {E,L,T,B}
    harmonics = IJVHarmonic{L,B}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, orbstruct, harmonics, kdtrees)
end

Base.push!(ijv::IJV, (i, j, v)::Tuple) = (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(h::IJV) = length(h.i) == 0

SparseArrays.sparse(c::IJV, n) = sparse(c.i, c.j, c.v, n, n)

kdtrees(b::IJVBuilder) = b.kdtrees

#endregion

############################################################################################
# CSC Hamiltonian builder
#region

mutable struct CSC{B}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{B}
    colcounter::Int
    rowvalcounter::Int
    cosorter::CoSort{Int,B}
end

mutable struct CSCHarmonic{L,B}
    dn::SVector{L,Int}
    collector::CSC{B}
end

struct CSCBuilder{T,E,L,B} <: AbstractSparseBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    orbstruct::OrbitalStructure{B}
    harmonics::Vector{CSCHarmonic{L,B}}
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

empty_harmonic(b::CSCBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    CSCHarmonic{L,B}(dn, CSC{B}(nsites(b.lat)))

CSCBuilder(lat, orbstruct) =
    CSCBuilder(lat, orbstruct, CSCHarmonic{latdim(lat),blocktype(orbstruct)}[])

function pushtocolumn!(s::CSC, row::Int, x, skipdupcheck::Bool = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
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
        isgrowing(s.cosorter) || throw(error("Internal error: repeated rows"))
    end
    s.colcounter += 1
    push!(s.colptr, s.rowvalcounter + 1)
    return nothing
end

finalizecolumn!(b::CSCBuilder, x...) =
    foreach(har -> finalizecolumn!(collector(har), x...), b.harmonics)

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

function SparseArrays.sparse(s::CSC, dim)
    completecolptr!(s.colptr, dim, s.rowvalcounter)
    rows, cols = isempty(s.rowval) ? 0 : maximum(s.rowval), length(s.colptr) - 1
    rows <= dim && cols == dim || throw(error("Internal error: matrix size $((rows, cols)) is inconsistent with lattice size $dim"))
    return SparseMatrixCSC(dim, dim, s.colptr, s.rowval, s.nzval)
end

Base.isempty(s::CSCHarmonic) = isempty(collector(s))
Base.isempty(s::CSC) = length(s.nzval) == 0

#endregion