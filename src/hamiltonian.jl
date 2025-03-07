############################################################################################
# add!(::IJVBuilder, ...)
#   add matrix elements to a builder before assembly into a (Parametric)Hamiltonian
#region

add!(m::TightbindingModel) = b -> add!(b, m)

# direct site indexing
function add!(b::IJVBuilder, val, c::CellSites, d::CellSites)
    c´, d´ = sanitize_cellindices(c, b), sanitize_cellindices(d, b)
    ijv = b[cell(d´)-cell(c´)]
    B = blocktype(b)
    val´ = mask_block(B, val)   # Warning: we don't check matrix size here, just conversion to B
    add!(ijv, val´, siteindices(c´), siteindices(d´))
    return b
end

function add!(b::IJVBuilder, val, c::CellSites)
    c´ = sanitize_cellindices(c, b)
    ijv = b[zero(cell(c´))]
    B = blocktype(b)
    val´ = mask_block(B, val)   # Warning: we don't check matrix size here, just conversion to B
    add!(ijv, val´, siteindices(c´))
    return b
end

add!(ijv::IJV, v, i::Integer, j::Integer=i) = push!(ijv, (i, j, v))

function add!(ijv::IJV, v, is, js)
    foreach(Iterators.product(is, js)) do (i, j)
        push!(ijv, (i, j, v))
    end
    return ijv
end

function add!(ijv::IJV, v, is)
    foreach(is) do i
        push!(ijv, (i, i, v))
    end
    return ijv
end


# block may be a tuple of row-col index ranges, to restrict application of model
function add!(b::IJVBuilder, model::TightbindingModel, block=missing)
    lat = lattice(b)
    bs = blockstructure(b)
    amodel = apply(model, (lat, bs))
    addterm!.(Ref(b), Ref(block), terms(amodel))
    return b
end

function add!(b::IJVBuilderWithModifiers, model::ParametricModel, block=missing)
    m0 = basemodel(model)
    ms = filterblock.(modifier.(terms(model)), Ref(block))
    add!(b, m0, block)
    push!(b, ms...)
    return b
end

# ensure modifiers (not only m0) remain restricted to their block
filterblock(m, ::Missing) = m
filterblock(m, b::UnitRange) = Intrablock(m, b)
filterblock(m, b::Tuple) = Interblock(m, b)

function addterm!(builder, block, term::AppliedOnsiteTerm)
    sel = selector(term)
    isempty(cells(sel)) || argerror("Cannot constrain cells in an onsite term, cell periodicity is assumed.")
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_site(sel, dn0) do s, i, r
        isinblock(i, block) || return nothing
        n = bsizes[s]
        # conventional terms are never non-spatial, only modifiers can be
        vr = term(r, n)
        push!(ijv, (i, i, vr))
    end
    return nothing
end

function addterm!(builder, block, term::AppliedHoppingTerm)
    trees = kdtrees(builder)
    sel = selector(term)
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_cell(sel) do dn
        ijv = builder[dn]
        found = foreach_hop(sel, trees, dn) do (si, sj), (i, j), (r, dr)
            isinblock(i, j, block) || return nothing
            ni = bsizes[si]
            nj = bsizes[sj]
            # conventional terms are never non-spatial, only modifiers can be
            vr = term(r, dr, (ni, nj))
            push!(ijv, (i, j, vr))
        end
        return found
    end
    return nothing
end

# If block is not Missing, restrict to ranges
# Interblock
isinblock(i, j, ::Missing) = true

function isinblock(i, j, rngs::Tuple)
    for (jr, rng´) in enumerate(rngs), (ir, rng) in enumerate(rngs)
        jr == ir && continue
        isinblock(i, rng) && isinblock(j, rng´) && return true
    end
    return false
end

# Intrablock
isinblock(i, j, rng) = isinblock(i, rng) && isinblock(j, rng)
isinblock(i, ::Missing) = true
isinblock(i, rng) = i in rng

#endregion

############################################################################################
# hamiltonian
#region

(model::AbstractModel)(lat::Lattice) = hamiltonian(lat, model)
(modifier::Modifier)(h::AbstractHamiltonian) = hamiltonian(h, modifier)

hamiltonian(args...; kw...) = lat -> hamiltonian(lat, args...; kw...)

hamiltonian(h::AbstractHamiltonian, m::AbstractModifier, ms::AbstractModifier...) =
    parametric(h, m, ms...)

hamiltonian(lat::Lattice, m0::TightbindingModel, m::Modifier, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat, m0; kw...), m, ms...)

hamiltonian(lat::Lattice, m0::ParametricModel, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat, basemodel(m0); kw...), modifier.(terms(m0))..., ms...)

hamiltonian(lat::Lattice, m::Modifier, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat; kw...), m, ms...)

hamiltonian(lat::Lattice, m::Interblock{<:TightbindingModel}; kw...) =
    hamiltonian(lat, parent(m), block(m); kw...)

hamiltonian(lat::Lattice, m::Interblock{<:ParametricModel}; kw...) = parametric(
    hamiltonian(lat, basemodel(parent(m)), block(m); kw...),
    modifier.(terms(parent(m)))...)

hamiltonian(lat::Lattice, m::TightbindingModel=TightbindingModel(), block=missing; orbitals=Val(1)) =
    hamiltonian!(IJVBuilder(lat, orbitals), m, block)

function hamiltonian!(b::IJVBuilder, m::TightbindingModel, block=missing)
    add!(b, m, block)
    return hamiltonian(b)
end

hamiltonian(b::IJVBuilder) = Hamiltonian(lattice(b), blockstructure(b), sparse(b))

hamiltonian(b::IJVBuilderWithModifiers) =
    maybe_parametric(Hamiltonian(lattice(b), blockstructure(b), sparse(b)), modifiers(b)...)

maybe_parametric(h) = h
maybe_parametric(h, m, ms...) = parametric(h, m, ms...)

#endregion

############################################################################################
# parametric
#region

function parametric(hparent::Hamiltonian)
    modifiers = ()
    allparams = Symbol[]
    allptrs = [Int[] for _ in harmonics(hparent)]
    # We must decouple hparent from the result, which will modify h in various ways
    h = minimal_callsafe_copy(hparent)
    return ParametricHamiltonian(hparent, h, modifiers, allptrs, allparams)
end

# Any means perhaps wrapped in Intrablock or Interblock
parametric(h::Hamiltonian, m::AnyAbstractModifier, ms::AnyAbstractModifier...) =
    parametric!(parametric(h), m, ms...)
parametric(p::ParametricHamiltonian, ms::AnyAbstractModifier...) =
    parametric!(copy(p), ms...)

parametric!(p::ParametricHamiltonian) = p

# This should not be exported, because it doesn't modify p in place (because of modifiers)
function parametric!(p::ParametricHamiltonian, ms::AnyModifier...)
    ams = apply.(ms, Ref(parent(p)))
    return parametric!(p, ams...)
end

function parametric!(p::ParametricHamiltonian, ms::AppliedModifier...)
    hparent = parent(p)
    h = hamiltonian(p)
    allmodifiers = (modifiers(p)..., ms...)
    # restores aliasing of any serializer to h
    relinked_modifiers = maybe_relink_serializer.(allmodifiers, Ref(h))
    allparams = parameter_names(p)
    merge_parameters!(allparams, ms...)
    allptrs = pointers(p)
    merge_pointers!(allptrs, ms...)
    return ParametricHamiltonian(hparent, h, relinked_modifiers, allptrs, allparams)
end

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameter_names(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

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

function _merge_pointers!(p, sm::AppliedSerializer)
    for (pn, ps) in zip(p, pointers(sm)), p in ps
        push!(pn, Base.front(p)...)
    end
    return p
end

#endregion

############################################################################################
# Hamiltonian call API
#   call!(::AbstractHamiltonian; params...) returns a Hamiltonian with params applied
#   call!(::AbstractHamiltonian, ϕs; params...) returns a HybridSparseMatrix with Bloch phases
#     ϕs and params applied
#   h(...; ...) is a copy decoupled from future call!'s
#region

(h::Hamiltonian)(phi...; params...) = copy(call!(h, phi...; params...))

call!(h::Hamiltonian; params...) = flat_sync!(h)  # mimic partial call!(p::ParametricHamiltonian; params...)
call!(h::Hamiltonian, φ1::Number, φ2::Number, φs::Number...; params...) =
    argerror("To obtain the (flat) Bloch matrix of `h` use `h(ϕs)`, where `ϕs` is a collection of `L=$(latdim(lattice(h)))` Bloch phases")
call!(h::Hamiltonian{T}, φs; params...) where {T} = flat_bloch!(h, sanitize_SVector(T, φs))
call!(h::Hamiltonian{<:Any,<:Any,0}, ::Tuple{}; params...) = h[()]
call!(h::Hamiltonian, ft::FrankenTuple) = call!(h, Tuple(ft))

# shortcut (see call!_output further below)
flat_bloch!(h::Hamiltonian{<:Any,<:Any,0}, ::SVector{0}, axis=missing) = h[()]

# returns a flat sparse matrix
function flat_bloch!(h::Hamiltonian{T}, φs::SVector, axis=missing) where {T}
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
        isvelocity && (e⁻ⁱᵠᵈⁿ *= -im * dcell(har)[axis])
        merged_mul!(dst, matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return dst
end

function initialize_bloch!(bloch, hars)
    fbloch = flat_unsafe(bloch)
    fbloch´ = merge_sparse(flat.(matrix.(hars)))
    copy!(fbloch, fbloch´)
    ubloch = unflat_unsafe(bloch)
    ubloch === fbloch || copy!(ubloch, unflat(blockstructure(bloch), fbloch))
    needs_no_sync!(bloch)
    update_nnz!(bloch)
    return bloch
end

@noinline checkbloch(::AbstractHamiltonian{<:Any,<:Any,L}, ::SVector{L´}) where {L,L´} =
    L == L´ || throw(ArgumentError("Need $L Bloch phases, got $(L´)"))

# ouput of a call!(h, ϕs)
call!_output(h::Hamiltonian) = flat_unsafe(bloch(h))
call!_output(h::Hamiltonian{<:Any,<:Any,0}) = flat_unsafe(h[hybrid()])

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(p::ParametricHamiltonian)(; kw...) = copy(call!(p; kw...))
(p::ParametricHamiltonian)(phis; kw...) = copy(call!(call!(p; kw...), phis))

call!(p::ParametricHamiltonian, phi; kw...) = call!(call!(p; kw...), phi)
call!(p::ParametricHamiltonian, ft::FrankenTuple) = call!(p, Tuple(ft); NamedTuple(ft)...)

function call!(ph::ParametricHamiltonian; kw...)
    reset_to_parent!(ph)
    h = hamiltonian(ph)
    applymodifiers!(h, modifiers(ph)...; kw...)
    flat_sync!(h)  # modifiers are applied to unflat, need to be synced to flat
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
            for ptr in ptrs
                nz[ptr] = nz´[ptr]
            end
        else
            copyto!(nz, nz´)
        end
        needs_flat_sync!(m)
    end
    return ph
end

applymodifiers!(h; kw...) = h

applymodifiers!(h, m, m´, ms...; kw...) = applymodifiers!(applymodifiers!(h, m; kw...), m´, ms...; kw...)

applymodifiers!(h, m::Modifier; kw...) = applymodifiers!(h, apply(m, h); kw...)

function applymodifiers!(h, m::AppliedOnsiteModifier; kw...)
    nz = nonzeros(unflat(first(harmonics(h))))
    if is_spatial(m)    # Branch outside loop
        @simd for p in pointers(m)
            (ptr, r, s, norbs) = p
            @inbounds nz[ptr] = m(nz[ptr], r, norbs; kw...)   # @inbounds too risky?
        end
    else
        @simd for p in pointers(m)
            (ptr, r, s, norbs) = p
            @inbounds nz[ptr] = m(nz[ptr], s, norbs; kw...)
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedOnsiteModifier{B}; kw...) where {B<:SMatrixView}
    nz = nonzeros(unflat(first(harmonics(h))))
    if is_spatial(m)    # Branch outside loop
        @simd for p in pointers(m)
            (ptr, r, s, norbs) = p
            val = view(nz[ptr], 1:norbs, 1:norbs)  # this might be suboptimal - do we need view?
            @inbounds nz[ptr] = m(val, r, norbs; kw...)   # @inbounds too risky?
        end
    else
        @simd for p in pointers(m)
            (ptr, r, s, norbs) = p
            val = view(nz[ptr], 1:norbs, 1:norbs)
            @inbounds nz[ptr] = m(val, s, norbs; kw...)
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier; kw...)
    for (har, ptrs) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        if is_spatial(m)    # Branch outside loop
            @simd for p in ptrs
                (ptr, r, dr, si, sj, orborb) = p
                @inbounds nz[ptr] = m(nz[ptr], r, dr, orborb; kw...)
            end
        else
            @simd for p in ptrs
                (ptr, r, dr, si, sj, orborb) = p
                @inbounds nz[ptr] = m(nz[ptr], si, sj, orborb; kw...)
            end
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier{B}; kw...) where {B<:SMatrixView}
    for (har, ptrs) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        if is_spatial(m)    # Branch outside loop
            @simd for p in ptrs
                (ptr, r, dr, si, sj, (oi, oj)) = p
                val = view(nz[ptr], 1:oi, 1:oj)  # this might be suboptimal - do we need view?
                @inbounds nz[ptr] = m(val, r, dr, (oi, oj); kw...)
            end
        else
            @simd for p in ptrs
                (ptr, r, dr, si, sj, (oi, oj)) = p
                val = view(nz[ptr], 1:oi, 1:oj)
                @inbounds nz[ptr] = m(val, si, sj, (oi, oj); kw...)
            end
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedSerializer; kw...)
    pname = only(parameter_names(m))
    nkw = NamedTuple(kw)
    # this should override hamiltonian(s), which should be aliased with h
    haskey(nkw, pname) && deserialize!(m, nkw[pname])
    return h
end

# ouput of a *full* call!(p, ϕs; kw...)
call!_output(p::ParametricHamiltonian) = call!_output(hamiltonian(p))

#endregion

############################################################################################
# indexing into AbstractHamiltonian - see also slices.jl
#region

# Extraction of Harmonics

Base.getindex(h::AbstractHamiltonian, dn::Union{Tuple,Integer,SVector,AbstractVector}) =
    flat(h[hybrid(dn)])

Base.getindex(h::AbstractHamiltonian, dn::UnflatInds) = unflat(h[hybrid(parent(dn))])

Base.getindex(h::AbstractHamiltonian, dn::HybridInds{<:Union{Integer,Tuple}}) =
    h[hybrid(SVector(parent(dn)))]

Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}, ::HybridInds{Tuple{}}) where {L} =
    h[hybrid(zero(SVector{L,Int}))]

Base.getindex(h::ParametricHamiltonian{<:Any,<:Any,L}, dn::HybridInds{SVector{L,Int}}) where {L} =
    getindex(hamiltonian(h), dn)

Base.@propagate_inbounds function Base.getindex(h::Hamiltonian{<:Any,<:Any,L}, dn::HybridInds{SVector{L,Int}}) where {L}
    for har in harmonics(h)
        parent(dn) == dcell(har) && return matrix(har)
    end
    @boundscheck(boundserror(harmonics(h), parent(dn)))
    # this is unreachable, but avoids allocations by having non-Union return type
    return matrix(first(harmonics(h)))
end

Base.isassigned(h::AbstractHamiltonian, dn::Tuple) = isassigned(h, SVector(dn))

function Base.isassigned(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for har in harmonics(h)
        dn == dcell(har) && return true
    end
    return false
end

# SiteSelector indexing - replicates GreenSolution indexing - see GreenFunctions.jl

Base.getindex(h::ParametricHamiltonian; kw...) = getindex(call!(h); kw...)
Base.getindex(h::Hamiltonian; kw...) = h[siteselector(; kw...)]

# conversion down to CellOrbitals. See sites_to_orbs in slices.jl
Base.getindex(h::ParametricHamiltonian, i, j) = getindex(call!(h), i, j)
Base.getindex(h::Hamiltonian, i, j) = getindex(h, sites_to_orbs(i, h), sites_to_orbs(j, h))
# we need AbstractHamiltonian here to avoid ambiguities with dn above
Base.getindex(h::AbstractHamiltonian, i) = (i´ = sites_to_orbs(i, h);
getindex(h, i´, i´))

Base.getindex(h::AbstractHamiltonian, ::SparseIndices...) =
    argerror("Sparse indexing not yet supported for AbstractHamiltonian")

# wrapped matrix for end user consumption
Base.getindex(h::Hamiltonian, i::OrbitalSliceGrouped, j::OrbitalSliceGrouped) =
    OrbitalSliceMatrix(
        mortar((h[si, sj] for si in cellsdict(i), sj in cellsdict(j))),
        (i, j))

Base.getindex(h::Hamiltonian, i::AnyOrbitalSlice, j::AnyOrbitalSlice) =
    mortar((h[si, sj] for si in cellsdict(i), sj in cellsdict(j)))

Base.getindex(h::Hamiltonian, i::AnyOrbitalSlice, j::AnyCellOrbitals) =
    mortar((h[si, sj] for si in cellsdict(i), sj in (j,)))

Base.getindex(h::Hamiltonian, i::AnyCellOrbitals, j::AnyOrbitalSlice) =
    mortar((h[si, sj] for si in (i,), sj in cellsdict(j)))

function Base.getindex(h::Hamiltonian{T}, i::AnyCellOrbitals, j::AnyCellOrbitals) where {T}
    dn = cell(i) - cell(j)
    oi, oj = orbindices(i), orbindices(j)
    mat = isassigned(h, dn) ? h[dn][oi, oj] : spzeros(Complex{T}, length(oi), length(oj))
    return mat
end

Base.view(h::ParametricHamiltonian, i::CellSites, j::CellSites=i) = view(call!(h), i, j)

function Base.view(h::Hamiltonian, i::CellSites, j::CellSites=i)
    oi, oj = sites_to_orbs_nogroups(i, h), sites_to_orbs_nogroups(j, h)
    dn = cell(oi) - cell(oj)
    return view(h[dn], orbindices(oi), orbindices(oj))
end

#endregion

############################################################################################
# coordination
#region

function nhoppings(h::AbstractHamiltonian)
    count = 0
    for har in harmonics(h)
        umat = unflat(matrix(har))
        count += iszero(dcell(har)) ? (nnz(umat) - nnzdiag(umat)) : nnz(umat)
    end
    return count
end

nonsites(h::Hamiltonian) = nnzdiag(h[unflat()])
# avoid call!, since params can have no default
nonsites(h::ParametricHamiltonian) = nonsites(hamiltonian(h))

coordination(h::AbstractHamiltonian) = iszero(nhoppings(h)) ? 0.0 : round(nhoppings(h) / nsites(lattice(h)), digits=5)

#endregion

############################################################################################
# unitcell_hamiltonian
#    Builds the intra-unitcell 0D Hamiltonian. If parent is p::ParametricHamiltonian, the
#    obtained uh::Hamiltonian is aliased with p, so call!(p,...) also updates uh
#region

function unitcell_hamiltonian(h::Hamiltonian)
    lat = lattice(lattice(h); bravais=())
    bs = blockstructure(h)
    hars = [Harmonic(SVector{0,Int}(), matrix(first(harmonics(h))))]
    return Hamiltonian(lat, bs, hars)
end

unitcell_hamiltonian(ph::ParametricHamiltonian) = unitcell_hamiltonian(hamiltonian(ph))

#endregion

############################################################################################
# combine
#   type-stable with Hamiltonians, but not with ParametricHamiltonians, as the field
#   builder.modifiers isa Vector{Any} in that case.
#region

function combine(hams::AbstractHamiltonian...; coupling::AbstractModel=TightbindingModel())
    check_unique_names(coupling, hams...)
    lat = combine(lattice.(hams)...)
    builder = IJVBuilder(lat, hams...)
    interblockmodel = interblock(coupling, hams...)
    builder´ = maybe_add_modifiers(builder, coupling)
    model´, block´ = parent(interblockmodel), block(interblockmodel)
    add!(builder´, model´, block´)
    return hamiltonian(builder´)
end

# No need to have unique names if nothing is parametric
check_unique_names(::TightbindingModel, ::Hamiltonian...) = nothing

function check_unique_names(::AbstractModel, hs::AbstractHamiltonian...)
    names = tupleflatten(sublatnames.(lattice.(hs))...)
    allunique(names) || argerror("Cannot combine ParametricHamiltonians with non-unique sublattice names, since modifiers could be tied to the original names. Assign unique names on construction.")
    return nothing
end

function check_unique_names(::AbstractModel, hs::Hamiltonian...)
    names = tupleflatten(sublatnames.(lattice.(hs))...)
    allunique(names) || argerror("Cannot combine Hamiltonians with non-unique sublattice names using a ParametricModel, since modifiers could be tied to the original names. Assign unique names on construction.")
    return nothing
end

maybe_add_modifiers(b, ::ParametricModel) = IJVBuilderWithModifiers(b)
maybe_add_modifiers(b, ::TightbindingModel) = b

#endregion

############################################################################################
# stitch(::Hamiltonian, phases::Tuple)
# stitch(::Hamiltonian, wrapaxes::SVector)
#region

stitch(phases) = h -> stitch(h, phases)

#region ## stitch(::Hamiltonian, phases)

function stitch(h::AbstractHamiltonian, phases)
    wp, wa, ua = split_axes(h, phases)
    isempty(wa) && return minimal_callsafe_copy(h)
    return _torus(h, wp, wa, ua)
end

# wa, ua = tuples of indices of wrapped/unwrapped axes
# wp = phases along wrapped axes
function _torus(h::Hamiltonian, wp, wa, ua)
    lat = lattice(h)
    b´ = bravais_matrix(lat)[:, SVector(ua)]
    lat´ = lattice(lat; bravais=b´)
    bs´ = blockstructure(h)
    bloch´ = copy_matrices(bloch(h))
    hars´ = stitch_harmonics(harmonics(h), wp, wa, ua)
    return Hamiltonian(lat´, bs´, hars´, bloch´)
end

function _torus(p::ParametricHamiltonian, wp, wa, ua)
    h = parent(p)
    h´ = _torus(h, wp, wa, ua)
    ams = modifiers(p)
    L = latdim(h)
    S = SMatrix{L,L,Int}(I)[SVector(ua), :] # dnnew = S * dnold
    ptrmap = pointer_map(h, h´, S)    # [[(ptr´ of har´[S*dn]) for ptr in har] for har in h]
    harmap = harmonics_map(h, h´, S)  # [(index of har´[S*dn]) for har in h]
    ams´ = stitch_modifier.(ams, Ref(ptrmap), Ref(harmap))
    p´ = hamiltonian(h´, ams´...)
    return p´
end

# indices for wrapped and unwrapped axes, and wrapped phases
function split_axes(::AbstractHamiltonian{<:Any,<:Any,L}, phases::Tuple) where {L}
    length(phases) == L || argerror("Expected $L `stitch` phases, got $(length(phases))")
    return _split_axes((), (), (), 1, phases...)
end

_split_axes(wp, wa, ua, n, ::Colon, xs...) = _split_axes(wp, wa, (ua..., n), n + 1, xs...)
_split_axes(wp, wa, ua, n, x, xs...) = _split_axes((wp..., x), (wa..., n), ua, n + 1, xs...)
_split_axes(wp, wa, ua, n) = wp, wa, ua

function split_axes(::AbstractHamiltonian{<:Any,<:Any,L}, wrapaxes::SVector) where {L}
    allunique(wrapaxes) && issorted(wrapaxes) && all(i -> 1<=i<=L, wrapaxes) ||
        argerror("Wrap axes should be a sorted SVector of unique axis indices between 1 and $L")
    wa = Tuple(wrapaxes)
    wp = Tuple(zero(wrapaxes))
    ua = inds_complement(Val(L), wa)
    return wp, wa, ua
end

function stitch_harmonics(hars, phases_w, wa::NTuple{W}, ua::NTuple{U}) where {W,U}
    dcells_u = SVector{U,Int}[dcell(har)[SVector(ua)] for har in hars]
    dcells_w = SVector{W,Int}[dcell(har)[SVector(wa)] for har in hars]
    unique_dcells_u = unique!(sort(dcells_u, by=norm))
    groups = [findall(==(dcell), dcells_u) for dcell in unique_dcells_u]
    hars´ = [summed_harmonic(inds, hars, phases_w, dcells_u, dcells_w) for inds in groups]
    return hars´
end

function summed_harmonic(inds, hars::Vector{<:Harmonic{<:Any,<:Any,B}}, phases_w, dcells_u, dcells_w) where {B}
    I, J, V = Int[], Int[], B[]
    for i in inds
        I´, J´, V´ = findnz(unflat(matrix(hars[i])))
        dn_w = dcells_w[i]
        e⁻ⁱᵠᵈⁿ = cis(-dot(phases_w, dn_w))
        V´ .*= e⁻ⁱᵠᵈⁿ
        append!(I, I´)
        append!(J, J´)
        append!(V, V´)
    end
    dn_u = dcells_u[first(inds)]
    bs = blockstructure(matrix(hars[first(inds)]))
    n = unflatsize(bs)
    mat = sparse(I, J, V, n, n)
    return Harmonic(dn_u, HybridSparseMatrix(bs, mat))
end

pointer_map(h, h´, S) =
    [pointer_map(har, first(harmonic_index(h´, S * dcell(har)))) for har in harmonics(h)]

function pointer_map(har, har´)
    ptrs´ = Int[]
    mat, mat´ = unflat(matrix(har)), unflat(matrix(har´))
    rows, rows´ = rowvals(mat), rowvals(mat´)
    for col in axes(mat, 2), ptr in nzrange(mat, col)
        row = rows[ptr]
        for ptr´ in nzrange(mat´, col)
            if row == rows´[ptr´]
                push!(ptrs´, ptr´)
                break
            end
        end
    end
    return ptrs´
end

harmonics_map(h, h´, S) = [last(harmonic_index(h´, S * dcell(har))) for har in harmonics(h)]

function stitch_modifier(m::AppliedOnsiteModifier, ptrmap, _)
    ptrs´ = first(ptrmap)
    p´ = [(ptrs´[ptr], r, s, orbs) for (ptr, r, s, orbs) in pointers(m)]
    return AppliedOnsiteModifier(m, p´)
end

function stitch_modifier(m::AppliedHoppingModifier, ptrmap, harmap)
    ps = pointers(m)
    ps´ = [similar(first(ps), 0) for _ in 1:maximum(harmap)]
    for (i, p) in enumerate(ps), (ptr, r, dr, si, sj, orborbs) in p
        i´ = harmap[i]
        ptrs´ = ptrmap[i]
        push!(ps´[i´], (ptrs´[ptr], r, dr, si, sj, orborbs))
    end
    sort!.(ps´)
    check_ptr_duplicates(ps´, ptrmap, harmap)
    return AppliedHoppingModifier(m, ps´)
end

# pss = [ps...], and ps = [(ptr, ...), ...] are modified hoppings on a given wrapped harmonic
# We firts check if no pointer ptr appears twice in each harmonic
# if it does it means that two hoppings in two original harmonics affected by the modifier
# become the same hopping in the wrapped Hamiltonian, which can be problematic
# Then we check if a modified and a non-modified original hoppings become the same after wrapping
# first append all the ptr´ pointers coming from each unwrapped harmonics into the same wrapped harmonic
# now find all ptr´ that are duplicate (i.e. that have multiple unwrapped harmonic sources)
# now, for each of these wrapped harmonics, find the modified ps = [(ptr´,...),...]
# if any ptr´ in ps appears in the duplicate list of that harmonic, it means that the
# corresponding wrapped hopping comes from summing a modified hopping and an unmodified hopping
function check_ptr_duplicates(pss, ptrmap, harmap)
    message0 = "The modifier will be applied to the sum, which may lead to unexpected results e.g. with non-linear or position-dependent modifiers. A possible fix is to use an enlarged supercell."
    message1 = "Two modified hoppings have been wrapped into one. " * message0
    message2 = "A modified and a non-modified hopping have been wrapped into one. " * message0
    # First check if two modified hoppings have been wrapped into one
    for ps in pss
        has_duplicates_ptrs = !allunique(first(p) for p in ps)
        if has_duplicates_ptrs
            @warn message1
            return nothing # warn only once
        end
    end
    # Now check if a modified and a non-modified hoppings have been wrapped into one
    merged_ptrmaps = Int[]
    for (indhar´, ps) in enumerate(pss)
        empty!(merged_ptrmaps)
        for (har_ptrmap, indhar) in zip(ptrmap, harmap)
            if indhar == indhar´
                append!(merged_ptrmaps, har_ptrmap)
            end
        end
        repeated_ptrs = onlyrepeated!(merged_ptrmaps)
        isempty(repeated_ptrs) && continue
        for (p, _...) in ps  # p are not repeated
            if p in repeated_ptrs
                @warn message2
                return nothing
            end
        end
    end
    return nothing
end

#endregion
#endregion


############################################################################################
# @stitch(h, phases, ϕname)
#region

macro stitch(h, phases_or_axes, name)
    quote
        wp, wa, ua = split_axes($(esc(h)), $(esc(phases_or_axes)))
        was = SVector(wa)
        mod = @hopping!((t, i, j; $name = $missing) -->
            t * blochphase((cell(i) - cell(j))[was], $(esc(name))); dcells = !iszero)
        h´ = $(esc(h)) |> mod
        _torus(h´, wp, wa, ua)
    end
end

blochphase(dn, ::Missing) = 1
blochphase(dn, ϕ) = blochphase(dn, sanitize_SVector(ϕ))
blochphase(dn, ϕ::AbstractVector) = cis(-dot(dn, ϕ))

#endregion
