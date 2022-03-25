############################################################################################
# hamiltonian
#region

hamiltonian(m::TightbindingModel = TightbindingModel(); kw...) = lat -> hamiltonian(lat, m; kw...)

# Base.@constprop :aggressive needed for type-stable non-Val orbitals
Base.@constprop :aggressive function hamiltonian(lat::Lattice, m = TightbindingModel(); orbitals = Val(1), type = numbertype(lat))
    orbstruct = OrbitalStructure(lat, orbitals, type)
    builder = IJVBuilder(lat, orbstruct)
    apmod = apply(m, (lat, orbstruct))
    # using foreach here foils precompilation of applyterm! for some reason
    applyterm!.(Ref(builder), terms(apmod))
    hars = harmonics(builder)
    return Hamiltonian(lat, orbstruct, hars)
end

function applyterm!(builder, term::AppliedOnsiteTerm)
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    sel = selector(term)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_site(sel, dn0) do s, i, r
        n = norbs[s]
        v = term(r, n)
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder, term::AppliedHoppingTerm, (irng, jrng) = (:, :))
    trees = kdtrees(builder)
    sel = selector(term)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_cell(sel) do dn, cell_iter
        ijv = builder[dn]
        foreach_hop!(sel, cell_iter, trees, dn) do (si, sj), (i, j), (r, dr)
            isinblock(i, irng) && isinblock(j, jrng) || return
            ni = norbs[si]
            nj = norbs[sj]
            v = term(r, dr, (ni, nj))
            push!(ijv, (i, j, v))
        end
    end
    return nothing
end

isinblock(i, ::Colon) = true
isinblock(i, irng) = i in irng

#endregion

############################################################################################
# parametric
#region

parametric(modifiers::Modifier...) = h -> parametric(h, modifiers...)

function parametric(hparent::Hamiltonian)
    modifiers = ()
    allptrs = [Int[] for _ in harmonics(hparent)]
    allparams = Symbol[]
    h = copy_harmonics(hparent)
    return ParametricHamiltonian(hparent, h, modifiers, allptrs, allparams)
end

parametric(f::FlatHamiltonian, ms::AbstractModifier...) =
    flatten(parametric(parent(f), ms...))
parametric(h::Hamiltonian, m::AbstractModifier, ms::AbstractModifier...) =
    _parametric!(parametric(h), m, ms...)
parametric(p::ParametricHamiltonian, ms::AbstractModifier...) =
    _parametric!(copy(p), ms...)

# This should not be exported, because it doesn't modify p in place (because of modifiers)
function _parametric!(p::ParametricHamiltonian, ms::Modifier...)
    ams = apply.(ms, Ref(parent(p)))
    return _parametric!(p, ams...)
end

function _parametric!(p::ParametricHamiltonian, ms::AppliedModifier...)
    hparent = parent(p)
    h = hamiltonian(p)
    allmodifiers = (modifiers(p)..., ms...)
    allptrs = pointers(p)
    allparams = parameters(p)
    merge_pointers!(allptrs, ms...)
    merge_parameters!(allparams, ms...)
    return ParametricHamiltonian(hparent, h, allmodifiers, allptrs, allparams)
end

merge_pointers!(p, m, ms...) = merge_pointers!(_merge_pointers!(p, m), ms...)

function merge_pointers!(p)
    for pn in p
        unique!(sort!(pn))
    end
    return p
end

function _merge_pointers!(p, m::AppliedOnsiteModifier)
    p0 = first(p)
    for (ptr, _) in pointers(m)
        push!(p0, ptr)
    end
    return p
end

function _merge_pointers!(p, m::AppliedHoppingModifier)
    for (pn, pm) in zip(p, pointers(m)), (ptr, _) in pm
        push!(pn, ptr)
    end
    return p
end

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameters(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(ph::ParametricHamiltonian)(; kw...) = copy_harmonics(call!(ph; kw...))
(f::FlatHamiltonian)(; kw...) = flatten(parent(f)(; kw...))

function call!(ph::ParametricHamiltonian; kw...)
    h = hamiltonian(ph)
    reset_pointers!(ph)
    applymodifiers!(h, modifiers(ph)...; kw...)
    return h
end

function reset_pointers!(ph::ParametricHamiltonian)
    h = hamiltonian(ph)
    hparent = parent(ph)
    for (har, har´, ptrs) in zip(harmonics(h), harmonics(hparent), pointers(ph))
        nz = nonzeros(matrix(har))
        nz´ = nonzeros(matrix(har´))
        for ptr in ptrs
            nz[ptr] = nz´[ptr]
        end
    end
    return ph
end

applymodifiers!(h, m, m´, ms...; kw...) = applymodifiers!(applymodifiers!(h, m; kw...), m´, ms...; kw...)

applymodifiers!(h, m::Modifier; kw...) = applymodifiers!(h, apply(m, h); kw...)

function applymodifiers!(h, m::AppliedOnsiteModifier; kw...)
    nz = nonzeros(matrix(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        nz[ptr] = m(nz[ptr], r, norbs; kw...)
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier; kw...)
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(matrix(har))
        for (ptr, r, dr, norbs) in p
            nz[ptr] = m(nz[ptr], r, dr, norbs; kw...)
        end
    end
    return h
end

#endregion

############################################################################################
# flatten and unflatten
#region

flatten(h::FlatHamiltonian) = h
flatten(h::AbstractHamiltonian{<:Any,<:Any,<:Any,<:Number}) = h
flatten(h::AbstractHamiltonian{<:Any,<:Any,<:Any,<:SMatrix}) =
    FlatHamiltonian(h, flatten(orbitalstructure(h)))

flatten(os::OrbitalStructure{<:Number}) = os

function flatten(os::OrbitalStructure{<:SMatrix})
    blocktype´ = eltype(blocktype(os))
    norbitals´ = [1 for _ in norbitals(os)]
    flatoffsets´ = flatoffsets(offsets(os), norbitals(os))
    return OrbitalStructure(blocktype´, norbitals´, flatoffsets´)
end

# sublat offsets after flattening (without padding zeros)
function flatoffsets(offsets0, norbs)
    nsites = diff(offsets0)
    nsites´ = norbs .* nsites
    offsets´ = cumsum!(nsites´, nsites´)
    prepend!(offsets´, 0)
    return offsets´
end

function flatten(lat::Lattice, os)
    norbs = norbitals(os)
    sites´ = similar(sites(lat), 0)
    names´ = sublatnames(lat)
    offsets´ = [0]
    for s in sublats(lat)
        norb = norbs[s]
        for r in sites(lat, s), _ in 1:norb
            push!(sites´, r)
        end
        push!(offsets´, length(sites´))
    end
    lat = Lattice(bravais(lat), Unitcell(sites´, names´, offsets´))
    return lat
end

function hamiltonian(f::FlatHamiltonian{<:Any,<:Any,L,B}) where {L,B}
    os = orbitalstructure(parent(f))
    flatos = orbitalstructure(f)
    lat = flatten(lattice(f), os)
    HT = Harmonic{L,SparseMatrixCSC{B,Int}}
    hars = HT[HT(dcell(har), flatten(matrix(har), os, flatos)) for har in harmonics(f)]  # see tools.jl
    return Hamiltonian(lat, flatos, hars)
end

unflatten(h::FlatHamiltonian) = parent(h)
unflatten(h::AbstractHamiltonian) = h

 #endregion

############################################################################################
# bloch
#region

function bloch(h::Union{Hamiltonian,ParametricHamiltonian}, ::Type{M} = SparseMatrixCSC) where {M<:AbstractSparseMatrixCSC}
    output = convert(M, merge_sparse(harmonics(h)))
    return Bloch(h, output)
end

function bloch(h::Union{Hamiltonian,ParametricHamiltonian}, ::Type{M}) where {M<:AbstractMatrix}
    output = convert(M, matrix(first(harmonics(h))))
    return Bloch(h, output)
end

function bloch(f::FlatHamiltonian, ::Type{M} = SparseMatrixCSC) where {M<:AbstractSparseMatrixCSC}
    os = orbitalstructure(parent(f))
    flatos = orbitalstructure(f)
    output = convert(M, merge_flatten_sparse(harmonics(f), os, flatos))
    return Bloch(f, output)
end

function bloch(f::FlatHamiltonian, ::Type{M}) where {M<:AbstractMatrix}
    flatos = orbitalstructure(f)
    B = blocktype(flatos)
    n = nsites(flatos)
    output = convert(M, zeros(B, n, n))
    return Bloch(f, output)
end

bloch(φs::Number...; kw...) = h -> bloch(h, φs; kw...)
bloch(φs::Tuple; kw...) = h -> bloch(h, φs; kw...)
bloch(h::AbstractHamiltonian, φs::Tuple; kw...) = bloch(h)(φs...; kw...)

# see tools.jl
merge_sparse(hars::Vector{<:Harmonic}) = merge_sparse(matrix(har) for har in hars)

merge_flatten_sparse(hars::Vector{<:Harmonic}, os::OrbitalStructure{<:SMatrix}, flatos::OrbitalStructure{<:Number}) =
    merge_flatten_sparse((matrix(har) for har in hars), os, flatos)

#endregion

############################################################################################
# AbstractBloch call API
#region

(b::AbstractBloch)(φs...; kw...) = copy(call!(b, φs...; kw...))

call!(b::AbstractBloch{L}, φs::Vararg{Number,L}; kw...) where {L} = call!(b, sanitize_SVector(φs); kw...)
call!(b::AbstractBloch{L}, φs::NTuple{L,Number}; kw...) where {L} = call!(b, sanitize_SVector(φs); kw...)
# support for (φs, (; kw...))
call!(b::AbstractBloch, φskw::Tuple{<:Any,NamedTuple}) = call!(b, first(φskw); last(φskw)...)
# support for (φs..., (; kw...))
call!(b::AbstractBloch, φskw::Tuple) = call!(b, Base.front(φskw); last(φskw)...)
call!(b::AbstractBloch, φs...; kw...) =
    throw(ArgumentError("Wrong call! argument syntax. Possible mismatch between input Bloch phases $(length(φs)) and lattice dimention $(latdim(b))."))

call!(b::Bloch, φs::SVector; kw...) =
    maybe_flatten_bloch!(matrix(b), hamiltonian(b), φs; kw...)
call!(b::Velocity, φs::SVector; kw...) =
    maybe_flatten_bloch!(matrix(b), hamiltonian(b), φs, axis(b); kw...)

maybe_flatten_bloch!(output, h::FlatHamiltonian, φs, axis...; kw...) =
    maybe_flatten_bloch!(output, parent(h), φs, axis...; kw...)
maybe_flatten_bloch!(output, h::ParametricHamiltonian, φs, axis...; kw...) =
    maybe_flatten_bloch!(output, call!(h; kw...), φs, axis...)

# Adds harmonics, assuming sparse output with the same structure of merged harmonics.
# If axis !== missing, compute velocity[axis]
function maybe_flatten_bloch!(output, h::Hamiltonian{<:Any,<:Any,L}, φs::SVector{L}, axis = missing) where {L}
    hars = harmonics(h)
    os = orbitalstructure(h)
    flatos = flatten(os)
    fill!(output, zero(eltype(output)))
    isvelocity = axis !== missing
    for har in hars
        iszero(dcell(har)) && isvelocity && continue
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        isvelocity && (e⁻ⁱᵠᵈⁿ *= - im * dcell(har)[axis])
        maybe_flatten_mul!(output, (os, flatos), matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return output
end

#endregion

############################################################################################
# indexing into AbstractHamiltonian
#region

Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} = h[zero(SVector{L,Int})]
Base.getindex(h::AbstractHamiltonian, is::Int...) = h[][is...]
Base.getindex(h::AbstractHamiltonian, dn::SVector, i, is...) = h[dn][i, is...]
Base.getindex(h::AbstractHamiltonian, dn::Tuple, is...) = getindex(h, SVector(dn), is...)

function Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for har in harmonics(h)
        dn == dcell(har) && return matrix(har)
    end
    throw(BoundsError(harmonics(h), dn))
end

Base.isassigned(h::AbstractHamiltonian, dn::Tuple) = isassigned(h, SVector(dn))

function Base.isassigned(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for har in harmonics(h)
        dn == dcell(har) && return true
    end
    return false
end

#endregion

############################################################################################
# push! and deleteat! of a Harmonic into AbstractHamiltonian
#region

Base.deleteat!(h::AbstractHamiltonian, dn::Tuple) = deleteat!(h, SVector(dn))

function Base.deleteat!(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    iszero(dn) && throw(ArgumentError("Cannot delete base harmonic"))
    hars = harmonics(h)
    for (i, har) in enumerate(hars)
        dn == dcell(har) && return deleteat!(hars, i)
    end
    return hars
end

Base.push!(h::AbstractHamiltonian, dn::Tuple) = push!(h, SVector(dn))

function Base.push!(h::AbstractHamiltonian{<:Any,<:Any,L,B}, dn::SVector{L,Int}) where {L,B}
    if !isassigned(h, dn)
        har = Harmonic(dn, spzeros(B, size(h)))
        push!(harmonics(h), har)
    end
    return h
end

#endregion

############################################################################################
# coordination
#region

function nhoppings(h::AbstractHamiltonian)
    count = 0
    for har in harmonics(h)
        count += iszero(dcell(har)) ? (_nnz(matrix(har)) - _nnzdiag(matrix(har))) : _nnz(matrix(har))
    end
    return count
end

function nhoppings(h::AbstractHamiltonian, col)
    n = 0
    for har in harmonics(h)
        mat = matrix(har)
        dn = dcell(har)
        rows = view(rowvals(mat), nzrange(mat, col))
        n += length(rows)
        if iszero(dn) && col in rows
            n -= 1
        end
    end
    return n
end

function nonsites(h::AbstractHamiltonian)
    count = 0
    for har in harmonics(h)
        iszero(dcell(har)) && (count += _nnzdiag(matrix(har)))
    end
    return count
end

coordination(h::AbstractHamiltonian) = round(nhoppings(h) / nsites(lattice(h)), digits = 5)

_nnz(s) = count(!iszero, nonzeros(s)) # Exclude stored zeros

function _nnzdiag(s)
    count = 0
    rowptrs = rowvals(s)
    nz = nonzeros(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += !iszero(nz[ptr]); break)
        end
    end
    return count
end

#endregion