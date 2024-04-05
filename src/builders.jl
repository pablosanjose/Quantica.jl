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

Base.isempty(h::IJV) = length(h.i) == 0

SparseArrays.sparse(c::IJV, m::Integer, n::Integer) = sparse(c.i, c.j, c.v, m, n)

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

Base.isempty(s::CSC) = length(s.nzval) == 0

#endregion

############################################################################################
# Harmonic and Hamiltonian builders
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

struct IJVBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{IJVHarmonic{L,B}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

struct CSCBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{CSCHarmonic{L,B}}
end

## Constructors ##

function CSCBuilder(lat::Lattice{<:Any,<:Any,L}, blockstruct::OrbitalBlockStructure{B}) where {L,B}
    harmonics = CSCHarmonic{L,B}[]
    return CSCBuilder(lat, blockstruct, harmonics)
end

function IJVBuilder(lat::Lattice{T,E,L}, blockstruct::OrbitalBlockStructure{B}) where {E,T,L,B}
    harmonics = IJVHarmonic{L,B}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, blockstruct, harmonics, kdtrees)
end

function IJVBuilder(lat::Lattice{T}, hams::AbstractHamiltonian...) where {T}
    orbs = vcat(norbitals.(hams)...)
    builder = IJVBuilder(lat, orbs)
    push_ijvharmonics!(builder, hams...)
    return builder
end

IJVBuilder(lat::Lattice{T}, orbitals) where {T} =
    IJVBuilder(lat, OrbitalBlockStructure(T, orbitals, sublatlengths(lat)))

push_ijvharmonics!(builder, ::OrbitalBlockStructure) = builder
push_ijvharmonics!(builder) = builder

function push_ijvharmonics!(builder::IJVBuilder, hs::AbstractHamiltonian...)
    offset = 0
    for h in hs
        for har in harmonics(h)
            ijv = builder[dcell(har)]
            hmat = unflat(matrix(har))
            I,J,V = findnz(hmat)
            append!(ijv, (I .+ offset, J .+ offset, V))
        end
        offset += nsites(lattice(h))
    end
    return builder
end

empty_harmonic(b::CSCBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    CSCHarmonic{L,B}(dn, CSC{B}(nsites(b.lat)))

empty_harmonic(::IJVBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    IJVHarmonic{L,B}(dn, IJV{B}())

## API ##

collector(har::AbstractHarmonicBuilder) = har.collector  # for IJVHarmonic and CSCHarmonic

dcell(har::AbstractHarmonicBuilder) = har.dn

kdtrees(b::IJVBuilder) = b.kdtrees

finalizecolumn!(b::CSCBuilder, x...) =
    foreach(har -> finalizecolumn!(collector(har), x...), b.harmonics)

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(s::CSCHarmonic) = isempty(collector(s))

lattice(b::AbstractHamiltonianBuilder) = b.lat

blockstructure(b::AbstractHamiltonianBuilder) = b.blockstruct

blocktype(::AbstractHamiltonianBuilder{<:Any,<:Any,<:Any,B}) where {B} = B

harmonics(b::AbstractHamiltonianBuilder) = b.harmonics

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
    n = nsites(lattice(builder))
    hars = HT[sparse(b, har, n, n) for har in harmonics(builder) if !isempty(har)]
    return hars
end

function SparseArrays.sparse(b::OrbitalBlockStructure{B}, har::AbstractHarmonicBuilder{L,B}, m::Integer, n::Integer) where {L,B}
    s = sparse(collector(har), m, n)
    return Harmonic(dcell(har), HybridSparseMatrix(b, s))
end

#endregion

############################################################################################
# ParametricHamiltonianBuilder - see src/hamiltonian.jl
#   created with b = builder(lat::Lattice; orbitals = Val(1))
#   implements: add!(b, AbstractModel), push!(b, ::Modifier...), hamiltonian(b), empty!(b)
#   also add!(b, val, ::CellSites...) - for internal use only, not block size checks.
#region

struct ParametricHamiltonianBuilder{T,E,L,B}
    ijvbuilder::IJVBuilder{T,E,L,B}
    modifiers::Vector{Any}    # modifier list need to be mutable and abritrary
end

#region ## Constructors ##

ParametricHamiltonianBuilder(ijvbuilder) = ParametricHamiltonianBuilder(ijvbuilder, Any[])

#endregion

#region ## API ##

builder(; kw...) = lat -> builder(lat; kw...)

builder(lat::Lattice; orbitals = Val(1)) =
    ParametricHamiltonianBuilder(IJVBuilder(lat, orbitals))

modifiers(b::ParametricHamiltonianBuilder) = b.modifiers

blockstructure(b::ParametricHamiltonianBuilder) = blockstructure(b.ijvbuilder)

lattice(b::ParametricHamiltonianBuilder) = lattice(b.ijvbuilder)

kdtrees(b::ParametricHamiltonianBuilder) = kdtrees(b.ijvbuilder)

blocktype(b::ParametricHamiltonianBuilder) = blocktype(b.ijvbuilder)

Base.parent(b::ParametricHamiltonianBuilder) = b.ijvbuilder

SparseArrays.sparse(b::ParametricHamiltonianBuilder) = sparse(b.ijvbuilder)

Base.push!(b::ParametricHamiltonianBuilder, ms::Modifier...) = push!(b.modifiers, ms...)

Base.pop!(b::ParametricHamiltonianBuilder) = pop!(b.modifiers)

Base.empty!(b::ParametricHamiltonianBuilder) = (empty!(b.ijvbuilder); empty!(b.modifiers); b)

Base.getindex(b::ParametricHamiltonianBuilder, dn::SVector) = b.ijvbuilder[dn]

#endregion
#endregion
