############################################################################################
# Abstract Hamiltonian builders
#region

abstract type AbstractSparseBuilder{T,E,L,O} end

function Base.getindex(b::AbstractSparseBuilder{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for c in b.collectors
        c.dn == dn && return c
    end
    C = eltype(b.collectors)
    collector = C(dn)
    push!(b.collectors, collector)
    return collector
end

orbitalstructure(b::AbstractSparseBuilder) = b.orbstruct
lattice(b::AbstractSparseBuilder) = b.lat
collectors(b::AbstractSparseBuilder) = b.collectors

function harmonics(builder::AbstractSparseBuilder{<:Any,<:Any,L,O}) where {L,O}
    HT = HamiltonianHarmonic{L,O}
    n = nsites(lattice(builder))
    hars = HT[HT(c.dn, sparse(c, n)) for c in builder.collectors if !isempty(c)]
    return hars
end

#endregion

############################################################################################
# IJV Hamiltonian builders
#region

struct IJV{L,O}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{O}
end

struct IJVBuilder{T,E,L,O} <: AbstractSparseBuilder{T,E,L,O}
    lat::Lattice{T,E,L}
    orbstruct::OrbitalStructure{O}
    collectors::Vector{IJV{L,O}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,O}(dn = zero(SVector{L,Int})) where {L,O} = IJV(dn, Int[], Int[], O[])

function IJVBuilder(lat::Lattice{T,E,L}, orbstruct::OrbitalStructure{O}) where {E,L,T,O}
    collectors = IJV{L,O}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, orbstruct, collectors, kdtrees)
end

Base.push!(ijv::IJV, (i, j, v)::Tuple) = (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))
Base.isempty(h::IJV) = length(h.i) == 0

SparseArrays.sparse(c::IJV, n) = sparse(c.i, c.j, c.v, n, n)

kdtrees(b::IJVBuilder) = b.kdtrees

#endregion

############################################################################################
# CSC Hamiltonian builder
#region

mutable struct CSC{L,O}
    dn::SVector{L,Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{O}
    colcounter::Int
    rowvalcounter::Int
    cosorter::CoSort{Int,O}
end

struct CSCBuilder{T,E,L,O} <: AbstractSparseBuilder{T,E,L,O}
    lat::Lattice{T,E,L}
    orbstruct::OrbitalStructure{O}
    collectors::Vector{CSC{L,O}}
end

function CSC{L,O}(dn::SVector = zero(SVector{L,Int})) where {L,O}
    colptr = [1]
    rowval = Int[]
    nzval = O[]
    colcounter = 1
    rowvalcounter = 0
    cosorter = CoSort(rowval, nzval)
    return CSC(dn, colptr, rowval, nzval, colcounter, rowvalcounter, cosorter)
end

CSCBuilder(lat, orbstruct) =
    CSCBuilder(lat, orbstruct, CSC{latdim(lat),blocktype(orbstruct)}[])

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

function finalizecolumn!(s::CSC, ncols::Int)
    for _ in 1:ncols
        finalizecolumn!(s)
    end
    return nothing
end

function SparseArrays.sparse(s::CSC, dim)
    completecolptr!(s.colptr, dim, s.rowvalcounter)
    rows, cols = isempty(s.rowval) ? 0 : maximum(s.rowval), length(s.colptr) - 1
    rows <= dim && cols == dim || throw(error("Internal error: matrix size $((rows, cols)) is inconsistent with lattice size $dim"))
    return SparseMatrixCSC(dim, dim, s.colptr, s.rowval, s.nzval)
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

Base.isempty(s::CSC) = length(s.nzval) == 0

#endregion