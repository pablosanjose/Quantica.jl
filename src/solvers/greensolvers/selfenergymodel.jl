############################################################################################
# SelfEnergyModelSolver <: RegularSelfEnergySolver <: AbstractSelfEnergySolver
#   A SelfEnergy solver that implements an AbstractModel on a selection of sites
#region

struct SelfEnergyModelSolver{T,E,P<:ParametricHamiltonian{T,E,0}} <: RegularSelfEnergySolver
    ph::P                      # has an extra parameter :ω_internal for the frequency
    parentinds::Vector{Int}   # stores the orb index in parent latslice for each ph orbital
end

#region ## Constructor ##

function SelfEnergyModelSolver(h::AbstractHamiltonian, model::ParametricModel, latslice::LatticeSlice)
    modelω = model_ω_to_param(model)  # see model.jl - transforms ω into a ω_internal param
    sliceinds = Int[]
    # this fills sliceinds::Vector{Int} with the latslice index for each lat0 site
    lat0 = lattice0D(latslice, sliceinds)
    # this is a 0D ParametricHamiltonian to build the Σ(ω) as a view over flat(ph(; ...))
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h))
    # this build siteoffsets for all h orbitals over latslice
    bs = contact_blockstructure(h, latslice)
    # translation from lat0 to latslice orbital indices
    # i.e. orbital index on latslice for each orbital in lat0
    parentinds = contact_sites_to_orbitals(sliceinds, bs)
    return SelfEnergyModelSolver(ph, parentinds)
end

#endregion

#region ## SelfEnergy (attach) API ##

function SelfEnergy(h::AbstractHamiltonian, model::ParametricModel; kw...)
    sel = siteselector(; kw...)
    latslice = lattice(h)[sel]
    solver = SelfEnergyModelSolver(h, model, latslice)
    return SelfEnergy(solver, latslice)
end

#endregion

#region ## API ##

function call!(s::SelfEnergyModelSolver, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.parentinds
    return view(m, rows, cols)
end

call!_output(s::SelfEnergyModelSolver) =
    view(call!_output(s.ph), s.parentinds, s.parentinds)

#endregion

#endregion top