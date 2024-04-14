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
    orbs = vcat(norbitals.(hams)...)
    builder = IJVBuilder(lat, orbs)
    push_ijvharmonics!(builder, hams...)
    return builder
end

# with some modifiers
function IJVBuilder(lat::Lattice{T}, hams::AbstractHamiltonian...) where {T}
    orbs = vcat(norbitals.(hams)...)
    builder = IJVBuilderWithModifiers(lat, orbs)
    push_ijvharmonics!(builder, hams...)
    unapplied_modifiers = tupleflatten(parent.(modifiers.(hams))...)
    push!(builder, unapplied_modifiers...)
    return builder
end

(::Type{IJVBuilderWithModifiers})(lat, orbitals) = IJVBuilder(lat, orbitals, Any[])

push_ijvharmonics!(builder, ::OrbitalBlockStructure) = builder
push_ijvharmonics!(builder, hars::Vector{<:IJVHarmonic}) = copy!(builder.harmonics, hars)
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

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(s::CSCHarmonic) = isempty(collector(s))

lattice(b::AbstractHamiltonianBuilder) = b.lat

blockstructure(b::AbstractHamiltonianBuilder) = b.blockstruct

blocktype(::AbstractHamiltonianBuilder{<:Any,<:Any,<:Any,B}) where {B} = B

harmonics(b::AbstractHamiltonianBuilder) = b.harmonics

Base.push!(b::IJVBuilderWithModifiers, ms::Modifier...) = push!(b.modifiers, ms...)

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

SparseArrays.sparse(har::AbstractHarmonicBuilder, m::Integer, n::Integer) =
    sparse(collector(har), m, n)

#endregion

############################################################################################
# WannierBuilder
#region

struct WannierBuilder{T,L} <: AbstractHamiltonianBuilder{T,L,L,Complex{T}}
    hbuilder::IJVBuilder{T,L,L,Complex{T},Missing}
    rharmonics::Vector{IJVHarmonic{L,SVector{L,Complex{T}}}}
end

struct Wannier90Data{T,L}
    brvecs::NTuple{L,SVector{L,T}}
    norbs::Int
    ncells::Int
    h::Vector{IJVHarmonic{L,Complex{T}}}             # must be dn-sorted, with (0,0,0) first
    r::Vector{IJVHarmonic{L,SVector{L,Complex{T}}}}  # must be dn-sorted, with (0,0,0) first
end

#region ## Constructors ##

wannier90(file::AbstractString; kw...) = WannierBuilder(file; kw...) # exported

function WannierBuilder(file::AbstractString; kw...)
    data = load_wannier90(file; kw...)
    lat = lattice(data)
    hbuilder = IJVBuilder(lat, Val(1)) # one orbital per site
    push_ijvharmonics!(hbuilder, data.h)
    rharmonics = data.r
    return WannierBuilder(hbuilder, rharmonics)
end

#endregion

#region ## API ##

hamiltonian(b::WannierBuilder) = hamiltonian(b.hbuilder)

positions(b::WannierBuilder) = BarebonesOperator(b.rharmonics)

#endregion

#region ## Wannier90Data ##

load_wannier90(filename; type = Float64, dim = 3, kw...) =
    load_wannier90(filename, postype(dim, type); kw...)

function load_wannier90(filename, ::Type{SVector{L,T}}; htol = 1e-8, rtol = 1e-8) where {L,T}
    data = open(filename, "r") do f
        # skip header
        readline(f)
        # read Bravais vectors
        brvecs3 = ntuple(_ -> readline_types(f, SVector{L,T}), Val(3))
        brvecs = L < 3 ? ntuple(i -> brvecs3[i], Val(L)) : brvecs3
        # read number of orbitals
        norbs = readline_types(f, Int)
        # read number of cells
        ncells = readline_types(f, Int)
        # skip symmetry degeneracies
        while !eof(f)
            isempty(readline(f)) && break
        end
        # read Hamiltonian
        h = load_harmonics(f, Val(L), Complex{T}, norbs, ncells; atol = htol)
        # read positions
        r = load_harmonics(f, Val(L), SVector{L,Complex{T}}, norbs, ncells; atol = rtol)
        return Wannier90Data(brvecs, norbs, ncells, h, r)
    end
    return data
end

function load_harmonics(f, ::Val{L}, ::Type{B}, norbs, ncells; atol) where {L,B}
    ncell = 0
    hars = IJVHarmonic{L,B}[]
    ijv = IJV{B}(norbs^2)
    while !eof(f)
        ncell += 1
        dn = readline_types(f, SVector{L,Int})
        for j in 1:norbs, i in 1:norbs
            i´, j´, v = readline_types(f, Int, Int, B)
            i´ == i && j´ == j ||
                argerror("load_wannier90: unexpected entry in file at element $((dn, i, j))")
            push_if_nonzero!(ijv, (i, j, v), atol)
        end
        if !isempty(ijv)
            push!(hars, IJVHarmonic(dn, ijv))
            ijv = IJV{B}(norbs^2)    # allocate new IJV
        end
        isempty(readline(f)) ||
            argerror("load_wannier90: unexpected line after harmonic $dn")
        ncell == ncells && break
    end
    sort!(hars, by = ijv -> abs.(dcell(ijv)))
    iszero(dcell(first(hars))) ||
        pushfirst!(hars, IJVHarmonic(zero(SVector{L,Int}), IJV{B}()))
    return hars
end

# readline_types(f::IO, t) = only(readline_types(split(readline(f)), t))
# readline_types(f::IO, t1, t2, ts...) =
#     readline_types(split(readline(f)), t1, t2, ts...)
# readline_types(v::Vector, ts...) = _readline_types((), v, ts...)

# _readline_types(rs::Tuple, tokens, t, ts...) =
#     _readline_types((rs..., poptype!(tokens, t)), tokens, ts...)
# _readline_types(rs::Tuple, _) = rs

# poptype!(tokens, ::Type{T}) where {T<:Real} = parse(T, popfirst!(tokens))

# function poptype!(tokens, ::Type{C}) where {T,C<:Complex{T}}
#     r = parse(T, popfirst!(tokens))
#     i = parse(T, popfirst!(tokens))
#     return C(r,i)
# end

# poptype!(tokens, ::Type{S}) where {N,T,S<:SVector{N,T}} =
#     S(ntuple(_ -> poptype!(tokens, T), Val(N)))

readline_types(f::IO, t1) = only(line_types(readline(f), t1))
readline_types(f::IO, t1, t2, ts...) = line_types(readline(f), t1, t2, ts...)

function line_types(s::String, types...)
    tokens = split(s)
    realtypes = tupleflatten(realtype.(types)...)
    reals = parse_reals(tokens, realtypes...)
    vals = reals_to_types(reals, types...)
    return vals
end

realtype(t::Type{<:Real}) = t
realtype(::Type{Complex{T}}) where {T} = (T, T)
realtype(::Type{SVector{N,T}}) where {N,T<:Real} = ntuple(Returns(T), Val(N))
realtype(::Type{SVector{N,Complex{T}}}) where {N,T<:Real} =
    (t -> (t..., t...))(realtype(SVector{N,T}))

parse_reals(tokens, realtypes::Vararg{<:Type,N}) where {N} =
    ntuple(i -> parse(realtypes[i], tokens[i]), Val(N))

function reals_to_types(reals, types::Vararg{<:Type,N}) where {N}
    offset = Ref(0)
    return ntuple(i -> real_to_type!(offset, reals, types[i]), Val(N))
end

real_to_type!(offset, reals, ::Type{T}) where {T<:Real} =
    (o = offset[]; offset[] += 1; T(reals[o+1]))
real_to_type!(offset, reals, ::Type{C}) where {C<:Complex} =
    (o = offset[]; offset[] += 2; C(reals[o+1], reals[o+2]))
real_to_type!(offset, reals, ::Type{S}) where {N,T,S<:SVector{N,T}} =
    S(ntuple(_ -> real_to_type!(offset, reals, T), Val(N)))

# reals_to_types(offset::Int, reals, ::Type{T}, ts...) where {T<:Real} =
#     (T(reals[offset+1]), reals_to_types(offset+1, reals, ts...)...)
# reals_to_types(offset, reals, ::Type{C}, ts...) where {C<:Complex} =
#     (C(reals[offset+1], reals[offset+2]), reals_to_types(offset+2, reals, ts...)...)
# reals_to_types(offset, reals, ::Type{S}, ts...) where {N,T,S<:SVector{N,T}} =
#     (ntuple(i->C(reals[offset+1], reals[offset+2]), reals_to_types(offset+2, reals, ts...)...)
# reals_to_types(offset, reals) = ()


# reals_to_types(, ::Type{<:Real}, ts...) = (r, reals_to_types(x, ts...))
# reals_to_types((r, x...), ::Type{<:Real}, ts...) = (r, reals_to_types(x, ts...))

# realtype(t::Type{<:Real}) = t
# realtype(::Type{Complex{T}}) where {T} = (T, T)
# realtype(::Type{SVector{N,T}}) where {N,T<:Real} = ntuple(Returns(T), Val(N))
# realtype(::Type{SVector{N,Complex{T}}}) where {N,T<:Real} =
#     (t -> (t..., t...))(realtype(SVector{N,T}))

push_if_nonzero!(ijv::IJV{<:Number}, (i, j, v), htol) =
    abs(v) > htol && push!(ijv, (i, j, v))
push_if_nonzero!(ijv::IJV{<:SVector}, (i, j, v), rtol) =
    (i == j || any(>(rtol), abs.(v))) && push!(ijv, (i, j, v))

function lattice(data::Wannier90Data{T,L}) where {T,L}
    bravais = hcat(data.brvecs...)
    rs = SVector{L,T}[]
    ijv = collector(first(data.r))
    for (i, j, r) in zip(ijv.i, ijv.j, ijv.v)
        i == j && push!(rs, real(r))
    end
    return lattice(sublat(rs); bravais)
end

#endregion

#endregion
