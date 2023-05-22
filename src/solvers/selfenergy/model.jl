############################################################################################
# SelfEnergy(h, model::AbstractModel; sites...)
#   A SelfEnergy solver that implements an AbstractModel on a selection of sites
#region

struct SelfEnergyModelSolver{T,E,P<:ParametricHamiltonian{T,E,0}} <: RegularSelfEnergySolver
    ph::P                      # pham over lat0D lattice == parent latslice
                               # pham has an extra parameter :ω_internal for the frequency
    parentinds::Vector{Int}    # stores the orb index in parent latslice for each ph orbital
end

#region ## Constructor ##

function SelfEnergyModelSolver(h::AbstractHamiltonian, model::AbstractModel, latslice::LatticeSlice)
    modelω = model_ω_to_param(model)  # see models.jl - transforms ω into a ω_internal param
    siteinds = Int[]
    # this converts latslice to a 0D Lattice lat0
    # and fills siteinds::Vector{Int} with the latslice index for each lat0 site
    lat0 = lattice0D(latslice, siteinds)
    # this is a 0D ParametricHamiltonian to build the Σ(ω) as a view over flat(ph(; ...))
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h))
    # this build siteoffsets for all h orbitals over latslice
    cbs = contact_blockstructure(h, latslice)
    # translation from lat0 to latslice orbital indices
    # i.e. orbital index on latslice for each orbital in lat0 (this is just a reordering!)
    parentinds = contact_sites_to_orbitals(siteinds, cbs)
    return SelfEnergyModelSolver(ph, parentinds)
end

#endregion

#region ## API ##

function SelfEnergy(h::AbstractHamiltonian, model::AbstractModel; kw...)
    sel = siteselector(; kw...)
    latslice = lattice(h)[sel]
    solver = SelfEnergyModelSolver(h, model, latslice)
    plottables = (solver.ph,)
    return SelfEnergy(solver, latslice, plottables)
end

function call!(s::SelfEnergyModelSolver, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.parentinds
    return view(m, rows, cols)
end

call!_output(s::SelfEnergyModelSolver) =
    view(call!_output(s.ph), s.parentinds, s.parentinds)

#endregion

#endregion top