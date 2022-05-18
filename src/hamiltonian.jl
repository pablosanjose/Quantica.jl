############################################################################################
# hamiltonian
#region

hamiltonian(m::TightbindingModel = TightbindingModel(); kw...) = lat -> hamiltonian(lat, m; kw...)

# Base.@constprop :aggressive needed for type-stable non-Val orbitals
Base.@constprop :aggressive function hamiltonian(lat::Lattice{T}, m = TightbindingModel(); orbitals = Val(1)) where {T}
    blockstruct = BlockStructure(T, orbitals, sublatlengths(lat))
    builder = IJVBuilder(lat, blockstruct)
    apmod = apply(m, (lat, blockstruct))
    # using foreach here foils precompilation of applyterm! for some reason
    applyterm!.(Ref(builder), terms(apmod))
    hars = sparse(builder)
    return Hamiltonian(lat, blockstruct, hars)
end

function applyterm!(builder, term::AppliedOnsiteTerm)
    sel = selector(term)
    isempty(cells(sel)) || argerror("Cannot constrain cells in an onsite term, cell periodicity is assumed.")
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_site(sel, dn0) do s, i, r
        n = bsizes[s]
        v = term(r, n)
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder, term::AppliedHoppingTerm, (irng, jrng) = (:, :))
    trees = kdtrees(builder)
    sel = selector(term)
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_cell(sel) do dn
        ijv = builder[dn]
        found = foreach_hop(sel, trees, dn) do (si, sj), (i, j), (r, dr)
            isinblock(i, irng) && isinblock(j, jrng) || return nothing
            ni = bsizes[si]
            nj = bsizes[sj]
            v = term(r, dr, (ni, nj))
            push!(ijv, (i, j, v))
        end
        return found
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
    allparams = Symbol[]
    allptrs = [Int[] for _ in harmonics(hparent)]
    # We must decouple hparent from the result, which will modify h in various ways
    h = copy_callsafe(hparent)
    return ParametricHamiltonian(hparent, h, modifiers, allptrs, allparams)
end

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
    allparams = parameters(p)
    merge_parameters!(allparams, ms...)
    allptrs = pointers(p)
    merge_pointers!(allptrs, ms...)
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
# copy_callsafe - minimal copy without side effects and race conditions between call!'s
#region

copy_bloch(h::Hamiltonian) = Hamiltonian(
    lattice(h), blockstructure(h), harmonics(h), copy(bloch(h)))

copy_harmonics(h::Hamiltonian) = Hamiltonian(
    lattice(h), blockstructure(h), copy.(harmonics(h)), bloch(h))

copy_callsafe(h::Hamiltonian) = copy_bloch(h)

copy_callsafe(p::ParametricHamiltonian) = ParametricHamiltonian(
    p.hparent, copy_harmonics(copy_bloch(p.h)), p.modifiers, p.allptrs, p.allparams)

#endregion

############################################################################################
# Hamiltonian call API
#region

(h::Hamiltonian)(phi...) = call!(copy_callsafe(h), phi...)

call!(h::Hamiltonian, phi) = bloch_flat!(h, sanitize_SVector(phi))
call!(h::Hamiltonian, phi...) = bloch_flat!(h, sanitize_SVector(phi))

# returns a flat sparse matrix
function bloch_flat!(h::Hamiltonian{T}, φs::SVector, axis = missing) where {T}
    hbloch = bloch(h)
    needs_initialization(hbloch) && initialize_bloch!(hbloch, harmonics(h))
    fbloch = flat(hbloch)
    fill!(fbloch, zero(Complex{T}))  # This preserves sparsity structure
    addblochs!(fbloch, h, φs, axis)
    return fbloch
end

function addblochs!(dst::SparseMatrixCSC, h::Hamiltonian, φs, axis)
    checkbloch(h, φs)
    hars = harmonics(h)
    isvelocity = axis isa Integer
    for har in hars
        iszero(dcell(har)) && isvelocity && continue
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        isvelocity && (e⁻ⁱᵠᵈⁿ *= - im * dcell(har)[axis])
        merged_mul!(dst, matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return dst
end

is_bloch_initialized(h) = !needs_full_update(bloch(h))

function initialize_bloch!(bloch, hars)
    fbloch = flat_unsafe(bloch)
    fbloch´ = merge_sparse(flat.(matrix.(hars)))
    copy!(fbloch, fbloch´)
    needs_no_sync!(bloch)
    return bloch
end

@noinline checkbloch(::AbstractHamiltonian{<:Any,<:Any,L}, ::SVector{L´}) where {L,L´} =
    L == L´ || throw(ArgumentError("Need $L Bloch phases, got $(L´)"))

call!_output(h::Hamiltonian) = flat(bloch(h))

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(p::ParametricHamiltonian)(; kw...) = call!(copy_callsafe(p); kw...)
(p::ParametricHamiltonian)(x, xs...; kw...) = call!(call!(copy_callsafe(p); kw...), x, xs...)

call!(p::ParametricHamiltonian, x, xs...; kw...) = call!(call!(p; kw...), x, xs...)
call!(p::ParametricHamiltonian, ft::FrankenTuple) = call!(p, Tuple(ft); NamedTuple(ft)...)

function call!(ph::ParametricHamiltonian; kw...)
    reset_to_parent!(ph)
    h = hamiltonian(ph)
    applymodifiers!(h, modifiers(ph)...; kw...)
    return h
end

function reset_to_parent!(ph)
    h = hamiltonian(ph)
    hparent = parent(ph)
    nnzfraction = 0.3  # threshold to revert to full copyto!
    for (har, har´, ptrs) in zip(harmonics(h), harmonics(hparent), pointers(ph))
        m, m´ = matrix(har), matrix(har´)
        nz = nonzeros(needs_initialization(m) ? unflat(m) : unflat_unsafe(m))
        nz´ = nonzeros(unflat(m´))
        if length(ptrs) < length(nz) * nnzfraction
            @simd for ptr in ptrs
                nz[ptr] = nz´[ptr]
            end
        else
            copyto!(nz, nz´)
        end
        needs_flat_sync!(m)
    end
    return ph
end

applymodifiers!(h, m, m´, ms...; kw...) = applymodifiers!(applymodifiers!(h, m; kw...), m´, ms...; kw...)

applymodifiers!(h, m::Modifier; kw...) = applymodifiers!(h, apply(m, h); kw...)

function applymodifiers!(h, m::AppliedOnsiteModifier; kw...)
    nz = nonzeros(unflat(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        nz[ptr] = m(nz[ptr], r, norbs; kw...)
    end
    return h
end

function applymodifiers!(h, m::AppliedOnsiteModifier{B}; kw...) where {B<:SMatrixView}
    nz = nonzeros(unflat(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        val = view(nz[ptr], 1:norbs, 1:norbs)
        nz[ptr] = m(val, r, norbs; kw...) # this allocates, currently unavoidable
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier; kw...)
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        for (ptr, r, dr, orborb) in p
            nz[ptr] = m(nz[ptr], r, dr, orborb; kw...)
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier{B}; kw...) where {B<:SMatrixView}
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        for (ptr, r, dr, (norbs, norbs´)) in p
            val = view(nz[ptr], 1:norbs, 1:norbs´)
            nz[ptr] = m(val, r, dr, (norbs, norbs´); kw...)  # this allocates, unavoidable
        end
    end
    return h
end

call!_output(p::ParametricHamiltonian) = flat(bloch(p))

#endregion

############################################################################################
# indexing into AbstractHamiltonian
#region

Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} = h[zero(SVector{L,Int})]
Base.getindex(h::AbstractHamiltonian, dn::Union{Integer,Tuple}) = getindex(h, SVector(dn))

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
# coordination
#region

function nhoppings(h::AbstractHamiltonian)
    count = 0
    for har in harmonics(h)
        count += iszero(dcell(har)) ? (nnz(matrix(har)) - nnzdiag(matrix(har))) : nnz(matrix(har))
    end
    return count
end

nonsites(h::AbstractHamiltonian) = nnzdiag(h[])

coordination(h::AbstractHamiltonian) = iszero(nhoppings(h)) ? 0.0 : round(nhoppings(h) / nsites(lattice(h)), digits = 5)

#endregion