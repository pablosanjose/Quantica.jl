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
# default_green_solver(::AbstractHamiltonian) = GS.Bands()

#endregion

############################################################################################
# GreenFuntion call! API
#region

## TODO: test copy(g) for aliasing problems
(g::GreenFunction)(; params...) = minimal_callsafe_copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = minimal_callsafe_copy(call!(g, ω; params...))

function call!(g::GreenFunction; params...)
    h´ = call!(hamiltonian(g); params...)
    solver´ = call!(solver(g); params...)
    contacts´ = call!(contacts(g); params...)
    return GreenFunction(h´, solver´, contacts´)
end

function call!(g::GreenFunction, ω; params...)
    h = parent(g)
    contacts´ = contacts(g)
    call!(h; params...)
    Σblocks = call!(contacts´, ω; params...)
    cbs = blockstructure(contacts´)
    slicer = solver(g)(ω, Σblocks, cbs)
    return GreenSolution(h, slicer, Σblocks, cbs)
end

#endregion

############################################################################################
# GreenSolution indexing
#region

Base.view(g::GreenSolution, i::ContactIndex, j::ContactIndex = i) = view(slicer(g), i, j)
Base.view(g::GreenSolution, i::Colon, j::Colon = i) = view(slicer(g), i, j)
Base.getindex(g::GreenSolution, i::ContactIndex, j::ContactIndex = i) = copy(view(g, i, j))
Base.getindex(g::GreenSolution, ::Colon, ::Colon = :) = copy(view(g, :, :))

Base.getindex(g::GreenSolution; kw...) = g[(; kw...)]

function Base.getindex(g::GreenSolution, i)
    ai = ind_to_slice(i, g)
    return getindex(g, ai, ai)
end

Base.getindex(g::GreenSolution, i, j) = getindex(g, ind_to_slice(i, g), ind_to_slice(j, g))

Base.getindex(g::GreenSolution, i::OrbitalSlice, j::OrbitalSlice) =
    mortar([g[si, sj] for si in subcells(i), sj in subcells(j)])

# fallback for cases where i and j are not *both* ContactIndex -> convert to OrbitalSlice
function ind_to_slice(c::ContactIndex, g)
    contactbs = blockstructure(g)
    cinds = contactinds(contactbs, Int(c))
    os = orbslice(contactbs)[cinds]
    return os
end

ind_to_slice(c::CellSites, g) = orbslice(c, hamiltonian(g))
ind_to_slice(l::LatticeSlice, g) = orbslice(l, hamiltonian(g))
ind_to_slice(kw::NamedTuple, g) = ind_to_slice(getindex(lattice(g); kw...), g)
ind_to_slice(cell::Union{SVector,Tuple}, g::GreenSolution{<:Any,<:Any,L}) where {L} =
    ind_to_slice(cellsites(sanitize_SVector(SVector{L,Int}, cell), :), g)
ind_to_slice(c::CellSites{<:Any,Colon}, g) =
    ind_to_slice(cellsites(cell(c), siterange(lattice(g))), g)
ind_to_slice(c::CellSites{<:Any,Symbol}, g) =
    ind_to_slice(cellsites(cell(c), siterange(lattice(g), siteindices(c))), g)
ind_to_slice(c::CellOrbitals, g) = c

Base.getindex(g::GreenSolution, i::CellOrbitals, j::CellOrbitals) = slicer(g)[i, j]

# fallback
Base.getindex(s::GreenSlicer, ::CellOrbitals, ::CellOrbitals) =
    internalerror("getindex of $(nameof(typeof(s))): not implemented")

#endregion

############################################################################################
# TMatrixSlicer <: GreenSlicer
#    Given a slicer that works without any contacts, implement slicing with contacts through
#    a T-Matrix equation g(i, j) = g0(i, j) + g0(i, k)T(k,k')g0(k', j), and T = (1-Σ*g0)⁻¹*Σ
#region

struct TMatrixSlicer{C,L,V<:SubArray{C},S} <: GreenSlicer{C}
    g0slicer::S
    tmatrix::V
    gcontacts::V
    blockstruct::ContactBlockStructure{L}
end

#region ## Constructors ##

function TMatrixSlicer(g0slicer::GreenSlicer{C}, Σblocks::NTuple{<:Any,MatrixBlock{C}}, blockstruct) where {C}
    if isempty(Σblocks)
        zeromat = view(zeros(C, 0, 0), 1:0, 1:0)
        return TMatrixSlicer(g0slicer, zeromat, zeromat, blockstruct)
    else
        os = orbslice(blockstruct)
        nos = norbs(os)
        n = max(nos, maxrows(Σblocks), maxcols(Σblocks))        # includes extended sites
        Σmat = Matrix{C}(undef, n, n)
        Σbm = BlockMatrix(Σmat, Σblocks)
        update!(Σbm)                                            # updates Σmat with Σblocks
        g0mat = zeros(C, n, n)
        off = offsets(os)
        for (j, sj) in enumerate(subcells(os)), (i, si) in enumerate(subcells(os))
            irng = off[i]+1:off[i+1]
            jrng = off[j]+1:off[j+1]
            g0view = view(g0mat, irng, jrng)
            copy!(g0view, g0slicer[si, sj])
        end
        den = Matrix{C}(I, n, n)
        mul!(den, Σmat, g0mat, -1, 1)                           # den = 1-Σ*g0
        luden = lu!(den)
        tmatrix = view(ldiv!(luden, Σmat), 1:nos, 1:nos)        # tmatrix = (1 - Σ*g0)⁻¹Σ
        gcontacts = view(rdiv!(g0mat, luden), 1:nos, 1:nos)     # gcontacts = g0*(1 - Σ*g0)⁻¹
        return TMatrixSlicer(g0slicer, tmatrix, gcontacts, blockstruct)
    end
end

#endregion

#region ## API ##

Base.view(s::TMatrixSlicer, i::ContactIndex, j::ContactIndex) =
    view(s.gcontacts, contactinds(s.blockstruct, Int(i)), contactinds(s.blockstruct, Int(j)))

Base.view(s::TMatrixSlicer, ::Colon, ::Colon) = s.gcontacts

function Base.getindex(s::TMatrixSlicer, i::CellOrbitals, j::CellOrbitals)
    g0 = s.g0slicer
    g0ij = g0[i, j]
    tkk´ = s.tmatrix
    isempty(tkk´) && return g0ij
    k = orbslice(s.blockstruct)
    g0ik = mortar([g0[si, sk] for si in (i,), sk in subcells(k)])
    g0k´j = mortar([g0[sk´, sj] for sk´ in subcells(k), sj in (j,)])
    gij = g0ij + g0ik * tkk´ * g0k´j
    return gij
end

#endregion

#endregion

############################################################################################
# conductance
#region

conductance(g::GreenFunctionSlice, ω; params...) = conductance(call!(g, ω; params...))

#endregion