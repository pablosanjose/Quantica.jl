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

# This should not be exported, because it doesn't actually modify p in place (because of modifiers)
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
call!(h::Hamiltonian{T}, φs, axis = missing; params...) where {T} = flat_bloch!(h, sanitize_SVector(T, φs), axis)
call!(h::Hamiltonian{<:Any,<:Any,0}, ::Tuple{}; params...) = h[()]
call!(h::Hamiltonian, ft::FrankenTuple, axis = missing) = call!(h, Tuple(ft), axis)

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
        αe⁻ⁱᵠᵈⁿ = blochfactor(dcell(har), φs, axis)
        merged_mul!(dst, matrix(har), αe⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
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

# rerturns e⁻ⁱᵠᵈⁿ or 1 if φ is missing
blochfactor(dn, ::Missing) = 1
blochfactor(dn, φ) = blochfactor(dn, sanitize_SVector(φ))
blochfactor(dn, φ::AbstractVector) = cis(-dot(dn, φ))
# returns e⁻ⁱᵠᵈⁿ or a ϕ-velocity derivative if an axis::Integer is given
blochfactor(dn, φ, axis::Missing) = blochfactor(dn, φ)
blochfactor(dn, φ, axis::Integer) = -im * dn[axis]*blochfactor(dn, φ)

# ouput of a call!(h, ϕs)
call!_output(h::Hamiltonian) = flat_unsafe(bloch(h))
call!_output(h::Hamiltonian{<:Any,<:Any,0}) = flat_unsafe(h[hybrid()])

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(p::ParametricHamiltonian)(; kw...) = copy(call!(p; kw...))
(p::ParametricHamiltonian)(phis, axis = missing; kw...) = copy(call!(call!(p; kw...), phis, axis))

call!(p::ParametricHamiltonian, phi, axis = missing; kw...) = call!(call!(p; kw...), phi, axis)
call!(p::ParametricHamiltonian, ft::FrankenTuple, axis = missing) =
    call!(p, sanitize_SVector(Tuple(ft)...), axis; NamedTuple(ft)...)

function call!(ph::ParametricHamiltonian; kw...)
    reset_to_parent!(ph)
    h = hamiltonian(ph)
    foreach(modifiers(ph)) do m
        applymodifiers!(h, m; kw...)
    end
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
