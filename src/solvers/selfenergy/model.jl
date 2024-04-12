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

function SelfEnergyModelSolver(h::AbstractHamiltonian, model::AbstractModel, orbslice::OrbitalSliceGrouped)
    modelω = model_ω_to_param(model)  # see models.jl - transforms ω into a ω_internal param
    siteinds = Int[]
    # this converts orbslice to a 0D Lattice lat0
    # and fills siteinds::Vector{Int} with the site index for each lat0 site (i.e. for sites ordered by sublattices)
    lat0 = lattice0D(orbslice, siteinds)
    # this is a 0D ParametricHamiltonian to build the Σ(ω) as a view over flat(ph(; ...))
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h)) # WARNING: type-unstable orbs
    # translation from orbitals in lat0 to orbslice indices
    # i.e. orbital index on orbslice for each orbital in lat0 (this is just a reordering!)
    parentinds = reordered_site_orbitals(siteinds, orbslice)
    return SelfEnergyModelSolver(ph, parentinds)
end

#endregion

#region ## API ##

function SelfEnergy(h::AbstractHamiltonian, model::AbstractModel; kw...)
    orbslice = contact_orbslice(h; kw...)
    solver = SelfEnergyModelSolver(h, model, orbslice)
    return SelfEnergy(solver, orbslice)
end

function call!(s::SelfEnergyModelSolver, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.parentinds
    return view(m, rows, cols)
end

call!_output(s::SelfEnergyModelSolver) =
    view(call!_output(s.ph), s.parentinds, s.parentinds)

minimal_callsafe_copy(s::SelfEnergyModelSolver) =
    SelfEnergyModelSolver(minimal_callsafe_copy(s.ph), s.parentinds)

#endregion

#endregion top
