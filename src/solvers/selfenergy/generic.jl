############################################################################################
# SelfEnergy(h, gs::GreenSlice, model::AbstractModel; sites...)
#   A RegularSelfEnergy -> Σ(ω) = V' gs(ω) V, where V = model coupling from sites to slice
#region

struct SelfEnergyGenericSolver{C,G<:GreenSlice,S,S´,H<:AbstractHamiltonian} <: RegularSelfEnergySolver
    hcoupling::H
    V´::S´                              # parent <- bath block view of call!_output(hcoupling)
    gslice::G                           # bath GreenSlice, allows gslice(ω)
    V::S                                # bath <- parent block view of call!_output(hcoupling)
    V´g::Matrix{C}                      # prealloc matrix
    Σ::Matrix{C}                        # prealloc matrix
end

#region ## Constructors ##

# nparent is the number of sites in gs, to know how to split hcoupling into (parent, bath) blocks
function SelfEnergyGenericSolver(gslice::GreenSlice, hcoupling::AbstractHamiltonian, nparent::Integer)
    lastflatparent = last(flatrange(hcoupling, nparent))
    parentinds, bathinds = 1:lastflatparent, lastflatparent+1:flatsize(hcoupling)
    hmatrix = call!_output(hcoupling)
    V = SparseMatrixView(view(hmatrix, bathinds, parentinds))
    V´ = SparseMatrixView(view(hmatrix, parentinds, bathinds))
    return SelfEnergyGenericSolver(gslice, hcoupling, V´, V)
end

function SelfEnergyGenericSolver(gslice::GreenSlice{T}, hcoupling::AbstractHamiltonian, V´::SparseMatrixView, V::SparseMatrixView) where {T}
    V´g = Matrix{Complex{T}}(undef, size(V´, 1), size(V, 1))
    Σ = Matrix{Complex{T}}(undef, size(V´, 1), size(V, 2))
    return SelfEnergyGenericSolver(hcoupling, V´, gslice, V, V´g, Σ)
end

#endregion

#region ## API ##

function SelfEnergy(hparent::AbstractHamiltonian, gslice::GreenSlice, model::AbstractModel; sites...)
    slicerows(gslice) === slicecols(gslice) ||
        argerror("To attach a Greenfunction with `attach(h, g[cols, rows], coupling; ...)`, we must have `cols == rows`")

    lsbath = latslice(parent(gslice), slicerows(gslice))
    lat0bath = lattice0D(lsbath)
    lsparent = getindex(lattice(hparent); sites...)
    lat0parent = lattice0D(lsparent)
    lat0 = combine(lat0parent, lat0bath)
    nparent, ntotal = nsites(lat0parent), nsites(lat0)
    # apply model to lat0 to get hcoupling
    interblockmodel = interblock(model, 1:nparent, nparent+1:ntotal)
    hcoupling = hamiltonian(lat0, interblockmodel;
        orbitals = vcat(norbitals(hparent), norbitals(gslice)))
    solver´ = SelfEnergyGenericSolver(gslice, hcoupling, nparent)
    plottables = (hcoupling,)
    return SelfEnergy(solver´, lsparent, plottables)
end

function call!(s::SelfEnergyGenericSolver, ω; params...)
    gω = call!(s.gslice, ω; params...)
    call!(s.hcoupling; params...)
    V = matrix(update!(s.V))
    V´ = matrix(update!(s.V´))
    mul!(s.V´g, V´, gω)
    Σω = mul!(s.Σ, s.V´g, V)
    return Σω
end

call!_output(s::SelfEnergyGenericSolver) = s.Σ

minimal_callsafe_copy(s::SelfEnergyGenericSolver) =
    SelfEnergyGenericSolver(
        minimal_callsafe_copy(s.hcoupling),
        minimal_callsafe_copy(s.V´),
        minimal_callsafe_copy(s.gslice),
        minimal_callsafe_copy(s.V),
        copy(s.V´g), copy(s.Σ))

#endregion



#endregion