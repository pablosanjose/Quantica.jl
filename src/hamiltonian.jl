############################################################################################
# add!(::IJVBuilder, ...)
#   add matrix elements to a builder before assembly into a (Parametric)Hamiltonian
#region

add!(m::TightbindingModel) = b -> add!(b, m)

# direct site indexing
function add!(b::IJVBuilder, val, c::CellSites, d::CellSites)
    c´, d´ = sanitize_cellindices(c, b), sanitize_cellindices(d, b)
    ijv = b[cell(d´) - cell(c´)]
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

add!(ijv::IJV, v, i::Integer, j::Integer = i) = push!(ijv, (i, j, v))

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
function add!(b::IJVBuilder, model::TightbindingModel, block = missing)
    lat = lattice(b)
    bs = blockstructure(b)
    amodel = apply(model, (lat, bs))
    addterm!.(Ref(b), Ref(block), terms(amodel))
    return b
end

function add!(b::IJVBuilderWithModifiers, model::ParametricModel, block = missing)
    m0 = basemodel(model)
    ms = modifier.(terms(model))
    add!(b, m0, block)
    push!(b, ms...)
    return b
end

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

function hamiltonian(lat::Lattice{T}, m::TightbindingModel = TightbindingModel(), block = missing; orbitals = Val(1)) where {T}
    b = IJVBuilder(lat, orbitals)
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

# This should not be exported, because it doesn't modify p in place (because of modifiers)
function parametric!(p::ParametricHamiltonian, ms::AnyModifier...)
    ams = apply.(ms, Ref(parent(p)))
    return parametric!(p, ams...)
end

function parametric!(p::ParametricHamiltonian, ms::AppliedModifier...)
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
flat_bloch!(h::Hamiltonian{<:Any,<:Any,0}, ::SVector{0}, axis = missing) = h[()]

# returns a flat sparse matrix
function flat_bloch!(h::Hamiltonian{T}, φs::SVector, axis = missing) where {T}
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
    # flat_sync!(h)  # modifiers are applied to unflat, need to be synced to flat
    # return h
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

function Base.getindex(h::Hamiltonian{<:Any,<:Any,L}, dn::HybridInds{SVector{L,Int}}) where {L}
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
Base.getindex(h::AbstractHamiltonian, i) = (i´ = sites_to_orbs(i, h); getindex(h, i´, i´))

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

Base.view(h::ParametricHamiltonian, i::CellSites, j::CellSites = i) = view(call!(h), i, j)

function Base.view(h::Hamiltonian, i::CellSites, j::CellSites = i)
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

coordination(h::AbstractHamiltonian) = iszero(nhoppings(h)) ? 0.0 : round(nhoppings(h) / nsites(lattice(h)), digits = 5)

#endregion

############################################################################################
# unitcell_hamiltonian
#    Builds the intra-unitcell 0D Hamiltonian. If parent is p::ParametricHamiltonian, the
#    obtained uh::Hamiltonian is aliased with p, so call!(p,...) also updates uh
#region

function unitcell_hamiltonian(h::Hamiltonian)
    lat = lattice(lattice(h); bravais = ())
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

function combine(hams::AbstractHamiltonian...; coupling::AbstractModel = TightbindingModel())
    check_unique_names(coupling, hams...)
    lat = combine(lattice.(hams)...)
    builder = IJVBuilder(lat, hams...)
    interblockmodel = interblock(coupling, hams...)
    model´, blocks´ = parent(interblockmodel), block(interblockmodel)
    add!(builder, model´, blocks´)
    return hamiltonian(builder)
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

#endregion

############################################################################################
# torus(::Hamiltonian, phases)
#region

torus(phases) = h -> torus(h, phases)

function torus(h::Hamiltonian{<:Any,<:Any,L}, phases) where {L}
    check_torus_phases(phases, L)
    wa, ua = split_axes(phases)  # indices for wrapped and unwrapped axes
    iszero(length(wa)) && return minimal_callsafe_copy(h)
    lat = lattice(h)
    b´ = bravais_matrix(lat)[:, SVector(ua)]
    lat´ = lattice(lat; bravais = b´)
    bs´ = blockstructure(h)
    bloch´ = copy_matrices(bloch(h))
    hars´ = stitch_harmonics(harmonics(h), phases, wa, ua)
    return Hamiltonian(lat´, bs´, hars´, bloch´)
end

check_torus_phases(phases, L) = length(phases) == L ||
    argerror("Expected $L `torus` phases, got $(length(phases))")

split_axes(phases) = split_axes((), (), 1, phases...)
split_axes(wa, ua, n, x::Colon, xs...) = split_axes(wa, (ua..., n), n+1, xs...)
split_axes(wa, ua, n, x, xs...) = split_axes((wa..., n), ua, n+1, xs...)
split_axes(wa, ua, n) = wa, ua

function stitch_harmonics(hars, phases, wa::NTuple{W}, ua::NTuple{U}) where {W,U}
    phases_w = SVector(phases)[SVector(wa)]
    dcells_u = SVector{U,Int}[dcell(har)[SVector(ua)] for har in hars]
    dcells_w = SVector{W,Int}[dcell(har)[SVector(wa)] for har in hars]
    unique_dcells_u = unique!(sort(dcells_u, by = norm))
    groups = [findall(==(dcell), dcells_u) for dcell in unique_dcells_u]
    hars´ = [summed_harmonic(inds, hars, phases_w, dcells_u, dcells_w) for inds in groups]
    return hars´
end

function summed_harmonic(inds, hars::Vector{<:Harmonic{<:Any,<:Any,B}}, phases_w, dcells_u, dcells_w) where {B}
    I,J,V = Int[], Int[], B[]
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

#endregion

############################################################################################
# torus(::ParametricHamiltonian, phases)
#region

function torus(p::ParametricHamiltonian, phases)
    wa, ua = split_axes(phases)  # indices for wrapped and unwrapped axes
    iszero(length(wa)) && return minimal_callsafe_copy(p)
    h = parent(p)
    h´ = torus(h, phases)
    ams = modifiers(p)
    L = latdim(lattice(h))
    S = SMatrix{L,L,Int}(I)[SVector(ua), :] # dnnew = S * dnold
    ptrmap = pointer_map(h, h´, S)    # [[(ptr´ of har´[S*dn]) for ptr in har] for har in h]
    harmap = harmonics_map(h, h´, S)  # [(index of har´[S*dn]) for har in h]
    ams´ = stitch_modifier.(ams, Ref(ptrmap), Ref(harmap))
    p´ = hamiltonian(h´, ams´...)
    return p´
end

pointer_map(h, h´, S) =
    [pointer_map(har, first(harmonic_index(h´, S*dcell(har)))) for har in harmonics(h)]

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

harmonics_map(h, h´, S) = [last(harmonic_index(h´, S*dcell(har))) for har in harmonics(h)]

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
    check_ptr_duplicates(first(ps´))
    return AppliedHoppingModifier(m, ps´)
end

function check_ptr_duplicates(h0ptrs)
    has_duplicates_ptrs = !allunique(first(p) for p in h0ptrs)
    has_duplicates_ptrs &&
        @warn "The wrapped ParametricHamiltonian has a modifier on hoppings that are the sum of intra- and intercell hoppings. The modifier will be applied to the sum, which may lead to unexpected results for position-dependent modifiers."
    return nothing
end

#endregion
