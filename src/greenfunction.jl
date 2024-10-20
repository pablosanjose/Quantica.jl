############################################################################################
# greenfunction
#region

greenfunction(s::AbstractGreenSolver) = oh -> greenfunction(oh, s)

greenfunction() = h -> greenfunction(h)

greenfunction(h::AbstractHamiltonian, args...) = greenfunction(OpenHamiltonian(h), args...)

function greenfunction(oh::OpenHamiltonian, s::AbstractGreenSolver = default_green_solver(hamiltonian(oh)))
    cs = Contacts(oh)
    h = hamiltonian(oh)
    as = apply(s, h, cs)
    return GreenFunction(h, as, cs)
end

default_green_solver(::AbstractHamiltonian0D) = GS.SparseLU()
default_green_solver(::AbstractHamiltonian1D) = GS.Schur()
default_green_solver(::AbstractHamiltonian) = GS.Bands()

#endregion

############################################################################################
# Contacts call! API
#region

function call!(c::Contacts, ω; params...)
    Σblocks = selfenergyblocks(c)
    call!.(c.selfenergies, Ref(ω); params...) # updates matrices in Σblocks
    return Σblocks
end

call!_output(c::Contacts) = selfenergyblocks(c)

#endregion

############################################################################################
# GreenFuntion call! API
#region

(g::GreenFunction)(ω; params...) = minimal_callsafe_copy(call!(g, ω; params...))
(g::GreenSlice)(ω; params...) = copy(call!(g, ω; params...))

call!(g::G, ω; params...) where {T,G<:Union{GreenFunction{T},GreenSlice{T}}} =
    call!(g, real_or_complex_convert(T, ω); params...)

function call!(g::GreenFunction{T}, ω::T; params...) where {T}
    ω´ = retarded_omega(ω, solver(g))
    return call!(g, ω´; params...)
end

function call!(g::GreenFunction{T}, ω::Complex{T}; params...) where {T}
    h = parent(g)   # not hamiltonian(h). We want the ParametricHamiltonian if it exists.
    contacts´ = contacts(g)
    call!(h; params...)
    Σblocks = call!(contacts´, ω; params...)
    corbs = contactorbitals(contacts´)
    slicer = solver(g)(ω, Σblocks, corbs)
    return GreenSolution(g, slicer, Σblocks, corbs)
end

call!(g::GreenSlice{T}, ω::T; params...) where {T} =
    call!(g, retarded_omega(ω, solver(parent(g))); params...)

call!(g::GreenSlice{T}, ω::Complex{T}; params...) where {T} =
    getindex!(call!_output(g), call!(greenfunction(g), ω; params...), orbinds_or_contactinds(g)...)

real_or_complex_convert(::Type{T}, ω::Real) where {T<:Real} = convert(T, ω)
real_or_complex_convert(::Type{T}, ω::Complex) where {T<:Real} = convert(Complex{T}, ω)

retarded_omega(ω::T, s::AppliedGreenSolver) where {T<:Real} =
    ω + im * sqrt(eps(float(T))) * needs_omega_shift(s)

# fallback, may be overridden
needs_omega_shift(s::AppliedGreenSolver) = true

#endregion

############################################################################################
# GreenSolution indexing
#   We convert any index down to cellorbs to pass to slicer, except contacts (Int, Colon)
#   If we index with CellIndices, we bypass mortaring and return a bare matrix
#region

## GreenFunction -> GreenSlice

Base.getindex(g::GreenFunction, i, j = i) = GreenSlice(g, i, j)
Base.getindex(g::GreenFunction; kw...) = g[siteselector(; kw...)]
Base.getindex(g::GreenFunction, kw::NamedTuple) = g[siteselector(; kw...)]

## GreenSolution -> OrbitalSliceMatrix or AbstractMatrix

# general entry point, conversion down to CellOrbitals. See sites_to_orbs in slices.jl
Base.getindex(g::GreenSolution, i, j; kw...) = getindex(g, sites_to_orbs(i, g), sites_to_orbs(j, g); kw...)
Base.getindex(g::GreenSolution, i; kw...) = (i´ = sites_to_orbs(i, g); getindex(g, i´, i´; kw...))

# g[::Integer, ::Integer] and g[:, :] - intra and inter contacts
Base.view(g::GreenSolution, i::CT, j::CT = i) where {CT<:Union{Integer,Colon}} =
    view(slicer(g), i, j)

Base.getindex(g::GreenSolution; kw...) = g[getindex(lattice(g); kw...)]

# args could be anything: CellSites, CellOrbs, Colon, Integer, DiagIndices...
function Base.getindex(g::GreenSolution, args...)
    gs = parent(g)[args...]  # get GreenSlice
    output = call!_output(gs)
    getindex!(output, g, orbinds_or_contactinds(gs)...)
    return output
end

Base.getindex(g::GreenSolution, i::AnyCellOrbitals, j::AnyCellOrbitals) =
    slicer(g)[sanitize_cellorbs(i), sanitize_cellorbs(j)]

# must ensure that orbindices is not a scalar, to consistently obtain a Matrix
sanitize_cellorbs(c::CellOrbitals) = c
sanitize_cellorbs(c::CellOrbital) = CellOrbitals(cell(c), orbindex(c):orbindex(c))
sanitize_cellorbs(c::CellOrbitalsGrouped) = CellOrbitals(cell(c), orbindices(c))

## getindex! - preallocated output for getindex

# fastpath for intra and inter-contact
getindex!(output, g::GreenSolution, i::Union{Integer,Colon}, j::Union{Integer,Colon}) =
    (copy!(parent(output), view(g, i, j)); output)

# indexing over single cells
getindex!(output, g::GreenSolution, i::AnyCellOrbitals, j::AnyCellOrbitals) =
    copy!(output, g[i, j])

function getindex!(output::OrbitalSliceMatrix, g::GreenSolution, i::AnyCellOrbitals, j::AnyCellOrbitals)
    oi, oj = orbaxes(output)
    rows, cols = orbrange(oi, cell(i)), orbrange(oj, cell(j))
    v = view(parent(output), rows, cols)
    getindex!(v, g, i, j)
    return output
end

# indexing over several cells
getindex!(output, g::GreenSolution, ci::AnyOrbitalSlice, cj::AnyOrbitalSlice) =
    getindex_cells!(output, g, cellsdict(ci), cellsdict(cj))
getindex!(output, g::GreenSolution, ci::AnyOrbitalSlice, cj::AnyCellOrbitals) =
    getindex_cells!(output, g, cellsdict(ci), (cj,))
getindex!(output, g::GreenSolution, ci::AnyCellOrbitals, cj::AnyOrbitalSlice) =
    getindex_cells!(output, g, (ci,), cellsdict(cj))

function getindex_cells!(output, g::GreenSolution, cis, cjs)
    for ci in cis, cj in cjs
        getindex!(output, g, ci, cj)
    end
    return output
end

## GreenSlicer -> Matrix, dispatches to each solver's implementation

# fallback conversion to CellOrbitals
Base.getindex(s::GreenSlicer, i::AnyCellOrbitals, j::AnyCellOrbitals = i) =
    getindex(s, sanitize_cellorbs(i), sanitize_cellorbs(j))

# fallback, for GreenSlicers that forgot to implement getindex
Base.getindex(s::GreenSlicer, ::CellOrbitals, ::CellOrbitals) =
    argerror("getindex of $(nameof(typeof(s))) not implemented")

# fallback, for GreenSlicers that don't implement view
Base.view(g::GreenSlicer, args...) =
    argerror("GreenSlicer of type $(nameof(typeof(g))) doesn't implement view for these arguments")

#endregion

############################################################################################
# GreenSolution diagonal indexing
#   returns a Diagonal matrix
#   one element per site (if kernel is not missing) or one per orbital (if kernel::Missing)
#   We don't support view with diagonal indexing, since Diagonal makes a copy
#region

function Base.getindex(g::GreenSolution{T}, di::D, dj::D = di; post = identity) where {T,I<:Union{Colon,Integer},D<:DiagIndices{I}}
    i, ker = parent(di), kernel(di)
    os = sites_to_orbs(i, g)
    d = post.(Complex{T}[])
    append_diagonal!(d, view(g, i, i), contactindranges(ker, i, g), ker; post)
    m = Diagonal(d)
    i´ = maybe_scalarize(os, ker)
    return OrbitalSliceMatrix(m, (i´, i´))
end

function Base.getindex(g::GreenSolution{T}, di::D, dj::D = di; post = identity) where {T,I<:OrbitalSliceGrouped,D<:DiagIndices{I}}
    os, ker = parent(di), kernel(di)
    d = post.(Complex{T}[])
    for o in cellsdict(os)
        append_diagonal!(d, g[o,o], orbindranges(ker, o), ker; post)
    end
    m = Diagonal(d)
    i´ = maybe_scalarize(os, ker)
    return OrbitalSliceMatrix(m, (i´, i´))
end

# no wrapping in OrbitalSliceMatrix for consistency with non-diagonal case
function Base.getindex(g::GreenSolution{T}, di::D, dj::D = di; post = identity) where {T,I<:CellSites,D<:DiagIndices{I}}
    i, ker = parent(di), kernel(di)
    o = sites_to_orbs(i, g)
    d = post.(Complex{T}[])
    append_diagonal!(d, g[o, o], orbindranges(ker, o), ker; post)
    m = Diagonal(d)
    return m
end

# If no kernel is provided, we return the whole diagonal
contactindranges(::Missing, o::Colon, gω) = 1:norbitals(contactorbitals(gω))
contactindranges(::Missing, i::Integer, gω) = 1:norbitals(contactorbitals(gω), i)
contactindranges(kernel, ::Colon, gω) = orbranges(contactorbitals(gω))
contactindranges(kernel, i::Integer, gω) = orbranges(contactorbitals(gω), i)

orbindranges(::Missing, o::AnyCellOrbitals) = eachindex(orbindices(o))
orbindranges(kernel, o::AnyCellOrbitals) = orbranges(o)

## core driver
#   append to d the elements Tr(kernel*x[i,i]) for each site encoded in i, or each orbital
#   if kernel is missing. g is the GreenFunction, used to convert i to CellOrbitals
function append_diagonal!(d, blockmat::AbstractMatrix, blockrngs, kernel; post = identity)
    for rng in blockrngs  # rng::Int
        val = post(apply_kernel(kernel, blockmat, rng))
        push!(d, val)
    end
    return d
end

apply_kernel(kernel, gblock, rows, cols = rows) = apply_kernel(kernel, view_or_scalar(gblock, rows, cols))
apply_kernel(kernel::Missing, v) = v
apply_kernel(kernel, v) = trace_prod(kernel, v)

view_or_scalar(gblock, rows::UnitRange, cols::UnitRange) = view(gblock, rows, cols)
view_or_scalar(gblock, orb::Integer, orb´::Integer) = gblock[orb, orb´]

#endregion

############################################################################################
# GreenSolution sparse indexing
#   returns an OrbitalSliceMatrix
#   one element per site (if kernel is not missing) or one per orbital (if kernel::Missing)
#region

function Base.getindex(g, di::D, dj::D = di; post = identity) where {D<:AppliedPairIndices}
    h, ker = parent(di), kernel(di)
    ldst, lsrc = ham_to_latslices(h)
    populate_sparse!(h, g, ker; post)
    return h[ldst, lsrc]
end

## core driver, can be overloaded by solvers
#   overwrite nonzero h elements with Tr(kernel*x[i,j]) for each site pair i,j
function populate_sparse!(h::Hamiltonian, g, ker; post = identity)
    bs = blockstructure(h)
    for har in harmonics(h)
        populate_sparse!(har, g, bs, ker; post)
    end
    return h
end

function populate_sparse!(har::Harmonic, g::GreenSolution, bs, ker; post)
    dn = dcell(har)
    gcell = g[sites(dn, :), sites(zero(dn), :)]
    return populate_sparse!(har, gcell, bs, ker; post)
end

function populate_sparse!(har::Harmonic, mat::AbstractMatrix, bs, kernel; post = identity)
    B = blocktype(har)
    hmat = unflat(har)
    rows = rowvals(hmat)
    nzs = nonzeros(hmat)
    for col in axes(hmat, 2), ptr in nzrange(hmat, col)
        row = rows[ptr]
        orows, ocols = flatrange(bs, row), flatrange(bs, col)
        val = post(apply_kernel(kernel, mat, orows, ocols))
        nzs[ptr] = mask_block(B, val)
    end
    return har
end

function ham_to_latslices(h::Hamiltonian)
    lat = lattice(h)
    srcinds = sourceinds(h[unflat()])
    cssrc = CellSites(zerocell(lat), srcinds)
    lsrc = LatticeSlice(lat, [cssrc])
    csdst = [CellSites(dcell(har), destinds(unflat(har))) for har in harmonics(h)]
    ldst = LatticeSlice(lat, csdst)
    return ldst, lsrc
end

destinds(mat) = unique!(sort(rowvals(mat)))
sourceinds(mat) = [col for col in axes(mat, 2) if !isempty(nzrange(mat, col))]

#endregion

############################################################################################
# FixedParamGreenSolver
#   support for g(; params...) --> GreenFunction (not a wrapper, completely independent)
#   FixedParamGreenSolver doesn't need to implement the AppliedGreenSolver API, since it
#   forwards to its parent
#region

struct FixedParamGreenSolver{P,G<:GreenFunction} <: AppliedGreenSolver
    gparent::G
    params::P
end

Base.parent(s::FixedParamGreenSolver) = s.gparent

parameters(s::FixedParamGreenSolver) = s.params

function (g::GreenFunction)(; params...)
    h´ = minimal_callsafe_copy(parent(g))
    c´ = minimal_callsafe_copy(contacts(g))
    s´ = minimal_callsafe_copy(solver(g), h´, c´)
    gparent = GreenFunction(h´, s´, c´)
    return GreenFunction(h´, FixedParamGreenSolver(gparent, params), c´)
end

(g::GreenSlice)(; params...) = GreenSlice(parent(g)(; params...), greenindices(g)...)

# params are ignored, solver.params are used instead. T required to disambiguate.
function call!(g::GreenFunction{T,<:Any,<:Any,<:FixedParamGreenSolver}, ω::Complex{T}; params...) where {T}
    s = solver(g)
    return call!(s.gparent, ω; s.params...)
end

function minimal_callsafe_copy(s::FixedParamGreenSolver, parentham, parentcontacts)
    solver´ = minimal_callsafe_copy(solver(s.gparent), parentham, parentcontacts)
    gparent = GreenFunction(parentham, solver´, parentcontacts)
    s´ = FixedParamGreenSolver(gparent, s.params)
    return s´
end

default_hamiltonian(g::GreenFunction{<:Any,<:Any,<:Any,<:FixedParamGreenSolver}) =
    default_hamiltonian(parent(g); parameters(solver(g))...)

#endregion

############################################################################################
# selfenergy(::GreenSolution, contactinds::Int...; onlyΓ = false)
#   if no contactinds are provided, all are returned. Otherwise, a view of Σ over contacts.
# selfenergy!(Σ::AbstractMatrix, ::GreenSolution, cinds...)
#   add blocks from cinds selfenergies to Σ (or all), and return a view over them
# similar_contactΣ(g)
#   Matrix to hold MatrixBlocks of any self-energy
# similar_contactΣ(g, cind)
#   Matrix that can hold a view of one self-energy
#region

selfenergy(g::GreenSolution, cinds::Int...; kw...) =
    selfenergy!(similar_contactΣ(g), g, cinds...; kw...)

# we support also the case for one contact, but it is only used elsewhere
function similar_contactΣ(g::Union{GreenFunction{T},GreenSolution{T}}, cind...) where {T}
    n = norbitals(contactorbitals(g), cind...)
    Σ = zeros(Complex{T}, n, n)
    return Σ
end

function maybe_selfenergy_view(Σ, g, cind, cinds...)
    inds = copy(contactinds(g, cind))
    foreach(i -> append!(inds, contactinds(g, i)), cinds)
    isempty(cinds) || unique!(sort!(inds))
    Σv = view(Σ, inds, inds)
    return Σv
end

maybe_selfenergy_view(Σ, g) = Σ

function selfenergy!(Σ::AbstractMatrix{T}, g::GreenSolution, cinds...; onlyΓ = false) where {T}
    fill!(Σ, zero(T))
    addselfenergy!(Σ, g, cinds...)
    onlyΓ && extractΓ!(Σ)   # faster to do this on Σ than on Σv
    Σv = maybe_selfenergy_view(Σ, g, cinds...)
    return Σv
end

addselfenergy!(Σ, g::GreenSolution, cind::Int, cinds...) =
    addselfenergy!(addselfenergy!(Σ, selfenergies(g)[cind]), g, cinds...)
addselfenergy!(Σ, ::GreenSolution) = Σ

# RegularSelfEnergy case
function addselfenergy!(Σ, b::MatrixBlock)
    v = view(Σ, blockrows(b), blockcols(b))
    v .+= blockmat(b)
    return Σ
end

# ExtendedSelfEnergy case
function addselfenergy!(Σ, (V´, g⁻¹, V)::NTuple{<:Any,MatrixBlock})
    v = view(Σ, blockrows(V´), blockcols(V))
    Vd = denseblockmat(V)
    copy!(Vd, blockmat(V))
    ldiv!(lu(blockmat(g⁻¹)), Vd)
    mul!(v, blockmat(V´), Vd, 1, 1)
    return Σ
end

function extractΓ!(Σ)
    Σ .-= Σ'
    Σ .*= im
    return Σ
end

#endregion

############################################################################################
# selfenergyblocks
#    Build MatrixBlocks from contacts, including extended inds for ExtendedSelfEnergySolvers
#region

function selfenergyblocks(contacts::Contacts)
    Σs = selfenergies(contacts)
    solvers = solver.(Σs)
    extoffset = norbitals(contactorbitals(contacts))
    cinds = contactinds(contacts)
    Σblocks = selfenergyblocks(extoffset, cinds, 1, (), solvers...)
    return Σblocks
end

# extoffset: current offset where extended indices start
# contactinds: orbital indices for all selfenergies in contacts
# ci: auxiliary index for current selfenergy being processed
# blocks: tuple accumulating all MatrixBlocks from all selfenergies
# solvers: selfenergy solvers that will update the MatrixBlocks
selfenergyblocks(extoffset, contactinds, ci, blocks) = blocks

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::RegularSelfEnergySolver, ss...)
    c = contactinds[ci]
    Σblock = MatrixBlock(call!_output(s), c, c)
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., Σblock), ss...)
end

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::ExtendedSelfEnergySolver, ss...)
    Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = shiftedmatblocks(call!_output(s), contactinds[ci], extoffset)
    extoffset += size(gₑₑ⁻¹, 1)
    # there is no minus sign here!
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., (Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)), ss...)
end

function shiftedmatblocks((Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)::NTuple{3,AbstractArray}, cinds, shift)
    extsize = size(gₑₑ⁻¹, 1)
    Vᵣₑ´ = MatrixBlock(Vᵣₑ, cinds, shift+1:shift+extsize)
    # adds denseblock for selfenergy ldiv
    Vₑᵣ´ = MatrixBlock(Vₑᵣ, shift+1:shift+extsize, cinds, Matrix(Vₑᵣ))
    gₑₑ⁻¹´ = MatrixBlock(gₑₑ⁻¹, shift+1:shift+extsize, shift+1:shift+extsize)
    return Vᵣₑ´, gₑₑ⁻¹´, Vₑᵣ´
end

#endregion

############################################################################################
# ContactOrbitals constructors
#   Build a ContactOrbitals from a Hamiltonian and a set of latslices
#region

ContactOrbitals(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} =
    ContactOrbitals{L}()

ContactOrbitals(h::AbstractHamiltonian, os, oss...) =
    ContactOrbitals(blockstructure(h), os, oss...)

# probably unused, since oss comes from SelfEnergy, so it is already OrbitalSliceGrouped
ContactOrbitals(bs::OrbitalBlockStructure, oss...) =
    ContactOrbitals(bs, sites_to_orbs.(oss, Ref(bs))...)

function ContactOrbitals(bs::OrbitalBlockStructure, oss::OrbitalSliceGrouped...)
    codicts = cellsdict.(oss)
    codictall = combine(codicts...)
    contactinds = Vector{Int}[]
    corb_to_ind = dictionary(corb => ind for (ind, corb) in enumerate(cellorbs(codictall)))
    for codict in codicts
        i = [corb_to_ind[co] for co in cellorbs(codict)]
        push!(contactinds, i)
    end
    offsets = lengths_to_offsets(length, codictall)
    corbs = ContactOrbitals(codictall, [codicts...], contactinds, offsets)
    return corbs
end

#endregion

############################################################################################
# TMatrixSlicer <: GreenSlicer
#    Given a slicer that works without any contacts, implement slicing with contacts through
#    a T-Matrix equation g(i, j) = g0(i, j) + g0(i, k)T(k,k')g0(k', j), and T = (1-Σ*g0)⁻¹*Σ
#region

struct TMatrixSlicer{C,L,V<:AbstractArray{C},S} <: GreenSlicer{C}
    g0slicer::S
    tmatrix::V
    gcontacts::V
    contactorbs::ContactOrbitals{L}
end

struct NothingSlicer{C} <: GreenSlicer{C}
end

#region ## Constructors ##

# if there are no Σblocks, return g0slicer. Otherwise build TMatrixSlicer.
maybe_TMatrixSlicer(g0slicer::GreenSlicer, Σblocks::Tuple{}, contactorbs) = g0slicer
maybe_TMatrixSlicer(g0slicer, Σblocks, contactorbs) =
    TMatrixSlicer(g0slicer, Σblocks, contactorbs)

# Uses getindex(g0slicer) to construct g0contacts
function TMatrixSlicer(g0slicer::GreenSlicer{C}, Σblocks, contactorbs) where {C}
    if isempty(Σblocks)
        tmatrix = gcontacts = view(zeros(C, 0, 0), 1:0, 1:0)
    else
        nreg = norbitals(contactorbs)
        g0contacts = zeros(C, nreg, nreg)
        off = offsets(contactorbs)
        for (j, sj) in enumerate(cellsdict(contactorbs)), (i, si) in enumerate(cellsdict(contactorbs))
            irng = off[i]+1:off[i+1]
            jrng = off[j]+1:off[j+1]
            g0view = view(g0contacts, irng, jrng)
            copy!(g0view, g0slicer[si, sj])
        end
        Σblocks´ = tupleflatten(Σblocks...)
        tmatrix, gcontacts = t_g_matrices(g0contacts, contactorbs, Σblocks´...)
    end
    return TMatrixSlicer(g0slicer, tmatrix, gcontacts, contactorbs)
end

# Takes a precomputed g0contacts (for dummy g0slicer that doesn't implement indexing)
function TMatrixSlicer(g0contacts::AbstractMatrix{C}, Σblocks, contactorbs) where {C}
    tmatrix, gcontacts = t_g_matrices(g0contacts, contactorbs, Σblocks...)
    g0slicer = NothingSlicer{C}()
    return TMatrixSlicer(g0slicer, tmatrix, gcontacts, contactorbs)
end

# empty Σblocks
function t_g_matrices(g0contacts::AbstractMatrix{C}, contactorbs) where {C}
    tmatrix = gcontacts = view(zeros(C, 0, 0), 1:0, 1:0)
    return tmatrix, gcontacts
end

# Check whether Σblocks are all spzeros, and if not, compute G and T
function t_g_matrices(g0contacts::AbstractMatrix{C}, contactorbs, Σblocks::MatrixBlock...) where {C}
    if isspzeros(Σblocks)
        gcontacts = g0contacts
        tmatrix = zero(g0contacts)
    else
        tmatrix, gcontacts = t_g_matrices!(copy(g0contacts), contactorbs, Σblocks...)
    end
    return tmatrix, gcontacts
end

# rewrites g0contacts
function t_g_matrices!(g0contacts::AbstractMatrix{C}, contactorbs, Σblocks::MatrixBlock...) where {C}
    nreg = norbitals(contactorbs)                            # number of regular orbitals
    n = max(nreg, maxrows(Σblocks), maxcols(Σblocks))        # includes extended orbitals
    Σmatext = Matrix{C}(undef, n, n)
    Σbm = BlockMatrix(Σmatext, Σblocks)
    update!(Σbm)                                             # updates Σmat with Σblocks
    Σmatᵣᵣ = view(Σmatext, 1:nreg, 1:nreg)
    Σmatₑᵣ = view(Σmatext, nreg+1:n, 1:nreg)
    Σmatᵣₑ = view(Σmatext, 1:nreg, nreg+1:n)
    Σmatₑₑ = view(Σmatext, nreg+1:n, nreg+1:n)
    Σmat = copy(Σmatᵣᵣ)
    Σmat´ = ldiv!(lu!(Σmatₑₑ), Σmatₑᵣ)
    mul!(Σmat, Σmatᵣₑ, Σmat´, 1, 1)                  # Σmat = Σmatᵣᵣ + ΣmatᵣₑΣmatₑₑ⁻¹ Σmatₑᵣ
    den = Matrix{C}(I, nreg, nreg)
    mul!(den, Σmat, g0contacts, -1, 1)                       # den = 1-Σ*g0
    luden = lu!(den)
    tmatrix = ldiv!(luden, Σmat)                             # tmatrix = (1 - Σ*g0)⁻¹Σ
    gcontacts = rdiv!(g0contacts, luden)                     # gcontacts = g0 * (1 - Σ*g0)⁻¹
    return tmatrix, gcontacts
end

#endregion

#region ## API ##

Base.view(s::TMatrixSlicer, i::Integer, j::Integer) =
    view(s.gcontacts, contactinds(s.contactorbs, i), contactinds(s.contactorbs, j))

Base.view(s::TMatrixSlicer, ::Colon, ::Colon) = view(s.gcontacts, :, :)

function Base.getindex(s::TMatrixSlicer, i::CellOrbitals, j::CellOrbitals)
    g0 = s.g0slicer
    g0ij = ensure_mutable_matrix(g0[i, j])
    tkk´ = s.tmatrix
    isempty(tkk´) && return g0ij
    k = s.contactorbs
    g0ik = mortar((g0[si, sk] for si in (i,), sk in cellsdict(k)))
    g0k´j = mortar((g0[sk´, sj] for sk´ in cellsdict(k), sj in (j,)))
    gij = mul!(g0ij, g0ik, tkk´ * g0k´j, 1, 1)  # = g0ij + g0ik * tkk´ * g0k´j
    return gij
end

ensure_mutable_matrix(m::SMatrix) = Matrix(m)
ensure_mutable_matrix(m::AbstractMatrix) = m

minimal_callsafe_copy(s::TMatrixSlicer, parentham, parentcontacts) = TMatrixSlicer(
    minimal_callsafe_copy(s.g0slicer, parentham, parentcontacts),
    s.tmatrix, s.gcontacts, s.contactorbs)

Base.view(::NothingSlicer, i::Union{Integer,Colon}...) =
    internalerror("view(::NothingSlicer): unreachable reached")

Base.getindex(::NothingSlicer, i::CellOrbitals...) = argerror("Slicer does not support generic indexing")

minimal_callsafe_copy(s::NothingSlicer, parentham, parentcontacts) = s

#endregion
#endregion

############################################################################################
# GreenSolutionCache
#   Cache that memoizes columns of GreenSolution[ci,cj] on columns of single CellSite{L}
#   It does not support more general indices, but upon creation, the cache includes the data
#   already computed for the intra-contacts Green function (with noncontact sites as undefs)
#region

struct GreenSolutionCache{T,L,G<:GreenSolution{T,<:Any,L}}
    gω::G
    cache::Dict{Tuple{SVector{L,Int},SVector{L,Int},Int},Matrix{Complex{T}}}
end

function GreenSolutionCache(gω::GreenSolution{T,<:Any,L}) where {T,L}
    cache = Dict{Tuple{SVector{L,Int},SVector{L,Int},Int},Matrix{Complex{T}}}()
    # if contacts exists, we preallocate columns for each of their sites
    if ncontacts(gω) > 0
        gmat = gω[:]
        g = parent(gω)
        h = hamiltonian(g)
        bs = blockstructure(h)
        co = contactorbitals(g)
        corngs = collect(orbranges(co))
        cls = latslice(g, :)
        nrows = flatsize(h)
        j = 0
        for colsc in cellsdict(cls)
            nj = cell(colsc)
            for j´ in siteindices(colsc)
                j += 1
                jrng = corngs[j]
                i = 0
                for rowsc in cellsdict(cls)
                    ni = cell(rowsc)
                    undefs = Matrix{Complex{T}}(undef, nrows, length(jrng))
                    for i´ in siteindices(rowsc)
                        i += 1
                        irng = corngs[i]
                        irng´ = flatrange(bs, i´)
                        copy!(view(undefs, irng´, :), view(gmat, irng, jrng))
                    end
                    push!(cache, (ni, nj, j´) => undefs)
                end
            end
        end
    end
    return GreenSolutionCache(gω, cache)
end

function Base.getindex(c::GreenSolutionCache{<:Any,L}, ci::CellSite, cj::CellSite) where {L}
    ci´, cj´ = sanitize_cellindices(ci, Val(L)), sanitize_cellindices(cj, Val(L))
    ni, i = cell(ci´), siteindex(ci´)
    nj, j = cell(cj´), siteindex(cj´)
    if haskey(c.cache, (ni, nj, j))
        gs = c.cache[(ni, nj, j)]
    else
        gs = c.gω[sites(ni, :), cj]
        push!(c.cache, (ni, nj, j) => gs)
    end
    h = hamiltonian(c.gω)
    rows = flatrange(h, i)
    return view(gs, rows, :)
end

#endregion
