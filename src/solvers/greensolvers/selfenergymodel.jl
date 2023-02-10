############################################################################################
# SelfEnergyModel <: RegularSelfEnergySolver <: AbstractSelfEnergySolver
#   A SelfEnergy solver that implements an AbstractModel on a selection of sites
#region

struct SelfEnergyModel{T,E,P<:ParametricHamiltonian{T,E,0}} <: RegularSelfEnergySolver
    flatorbinds::Vector{Int}   # stores the orb index in parent latslice for each ph orbital
    ph::P                      # has an extra parameter :ω_internal for the frequency
end

#region ## Constructor ##

function SelfEnergyModel(h::AbstractHamiltonian, model::ParametricModel, latslice::LatticeSlice)
    modelω = model_ω_to_param(model)  # see model.jl - transforms ω into a ω_internal param
    sliceinds = Int[]
    # this fills sliceinds::Vector{Int} with the latslice index for each lat0 site
    lat0 = lattice(latslice, sliceinds)
    # this is a 0D ParametricHamiltonian to build the Σ(ω) as a view over flat(ph(; ...))
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h))
    # this build siteoffsets for all h orbitals over latslice
    bs = contact_blockstructure(h, latslice)
    # translation from lat0 to latslice orbital indices
    # i.e. orbital index on latslice for each orbital in lat0
    flatorbinds = flat_orbital_indices(sliceinds, bs)
    return SelfEnergyModel(flatorbinds, ph)
end

function flat_orbital_indices(sliceinds, bs::ContactBlockStructure)
    finds = Int[]
    for iunflat in sliceinds
        append!(finds, siterange(bs, iunflat))
    end
    return finds
end

#endregion

#region ## API ##

function call!(s::SelfEnergyModel, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.flatorbinds
    return view(m, rows, cols)
end

call!_output(s::SelfEnergyModel) =
    view(call!_output(s.ph), s.flatorbinds, s.flatorbinds)

#endregion

#endregion top