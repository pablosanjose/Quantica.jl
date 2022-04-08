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
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    sel = selector(term)
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
    foreach_cell(sel) do dn, cell_iter
        ijv = builder[dn]
        foreach_hop!(sel, cell_iter, trees, dn) do (si, sj), (i, j), (r, dr)
            isinblock(i, irng) && isinblock(j, jrng) || return
            ni = bsizes[si]
            nj = bsizes[sj]
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
    allparams = Symbol[]
    # allptrs = [Int[] for _ in harmonics(hparent)]
    h = copy_only_harmonics(hparent)
    # return ParametricHamiltonian(hparent, h, modifiers, allptrs, allparams)
    return ParametricHamiltonian(hparent, h, modifiers, allparams)
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
    # allptrs = pointers(p)
    # merge_pointers!(allptrs, ms...)
    # return ParametricHamiltonian(hparent, h, allmodifiers, allptrs, allparams)
    return ParametricHamiltonian(hparent, h, allmodifiers, allparams)
end

# merge_pointers!(p, m, ms...) = merge_pointers!(_merge_pointers!(p, m), ms...)

# function merge_pointers!(p)
#     for pn in p
#         unique!(sort!(pn))
#     end
#     return p
# end

# function _merge_pointers!(p, m::AppliedOnsiteModifier)
#     p0 = first(p)
#     for (ptr, _) in pointers(m)
#         push!(p0, ptr)
#     end
#     return p
# end

# function _merge_pointers!(p, m::AppliedHoppingModifier)
#     for (pn, pm) in zip(p, pointers(m)), (ptr, _) in pm
#         push!(pn, ptr)
#     end
#     return p
# end

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameters(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

#endregion

############################################################################################
# Hamiltonian call API
#region

(h::Hamiltonian)(phi...) = copy(call!(h, phi...))

call!(h::Hamiltonian, phi) = call!(h, sanitize_SVector(phi))
call!(h::Hamiltonian, phi...) = call!(h, sanitize_SVector(phi))

# returns a HybridSparseMatrixCSC
function call!(h::Hamiltonian{T}, φs::SVector, axis = missing) where {T}
    checkbloch(h, φs)
    hbloch = bloch(h)
    hars = harmonics(h)
    needs_initialization!(hbloch) || initialize_bloch!(bloch´, hars)
    fbloch = flat(hbloch)
    fill!(fbloch, zero(Complex{T}))  # This preserves sparsity structure
    isvelocity = axis !== missing
    for har in hars
        iszero(dcell(har)) && isvelocity && continue
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        isvelocity && (e⁻ⁱᵠᵈⁿ *= - im * dcell(har)[axis])
        merged_mul!(fbloch, flat(matrix(har)), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return hbloch
end

is_bloch_initialized(h) = !needs_full_update(bloch(h))

function initialize_bloch!(bloch, hars)
    bloch´ = merge_sparse(flat.(matrix.(hars)))
    copy!(bloch, bloch´)
    return bloch
end

@noinline checkbloch(::AbstractHamiltonian{<:Any,<:Any,L}, ::SVector{L´}) where {L,L´} =
    L == L´ || throw(ArgumentError("Need $L Bloch phases, got $(L´)"))

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(ph::ParametricHamiltonian)(; kw...) = copy_only_harmonics(call!(ph; kw...))

function call!(ph::ParametricHamiltonian; kw...)
    h = hamiltonian(ph)
    reset_to_parent!(ph)
    applymodifiers!(h, modifiers(ph)...; kw...)
    return h
end

function reset_to_parent!(ph::ParametricHamiltonian)
    h = hamiltonian(ph)
    hparent = parent(ph)
    for (har, har´) in zip(harmonics(h), harmonics(hparent))
        m, m´ = matrix(har), matrix(har´)
        nz = nonzeros(unflat(m))
        nz´ = nonzeros(unflat(m´))
        copyto!(nz, nz´)
        needs_flat_sync!(m)
    end
    return ph
end

# function reset_pointers!(ph::ParametricHamiltonian)
#     h = hamiltonian(ph)
#     hparent = parent(ph)
#     for (har, har´, ptrs) in zip(harmonics(h), harmonics(hparent), pointers(ph))
#         nz = nonzeros(matrix(har))
#         nz´ = nonzeros(matrix(har´))
#         for ptr in ptrs
#             nz[ptr] = nz´[ptr]
#         end
#     end
#     return ph
# end

applymodifiers!(h, m, m´, ms...; kw...) = applymodifiers!(applymodifiers!(h, m; kw...), m´, ms...; kw...)

applymodifiers!(h, m::Modifier; kw...) = applymodifiers!(h, apply(m, h); kw...)

function applymodifiers!(h, m::AppliedOnsiteModifier; kw...)
    nz = nonzeros(unflat(matrix(first(harmonics(h)))))
    for (ptr, r, norbs) in pointers(m)
        nz[ptr] = m(nz[ptr], r, norbs; kw...)
    end
    return h
end

function applymodifiers!(h, m::AppliedOnsiteModifier{B}; kw...) where {B<:SMatrixView}
    nz = nonzeros(unflat(matrix(first(harmonics(h)))))
    for (ptr, r, norbs) in pointers(m)
        val = view(parent(nz[ptr]), 1:norbs, 1:norbs)
        nz[ptr] = m(val, r, norbs; kw...)
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier; kw...)
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(matrix(har)))
        for (ptr, r, dr, orborb) in p
            nz[ptr] = m(nz[ptr], r, dr, orborb; kw...)
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier{B}; kw...) where {B<:SMatrixView}
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(matrix(har)))
        for (ptr, r, dr, (norbs, norbs´)) in p
            val = view(parent(nz[ptr]), 1:norbs, 1:norbs´)
            nz[ptr] = m(val, r, dr, (norbs, norbs´); kw...)
        end
    end
    return h
end

#endregion

############################################################################################
# indexing into AbstractHamiltonian
#region

Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} = h[zero(SVector{L,Int})]
Base.getindex(h::AbstractHamiltonian, dn::Tuple) = getindex(h, SVector(dn))

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

coordination(h::AbstractHamiltonian) = round(nhoppings(h) / nsites(lattice(h)), digits = 5)

#endregion