############################################################################################
# SelfEnergyModel <: RegularSelfEnergySolver <: AbstractSelfEnergySolver
#region

struct SelfEnergyModel{T,E,L,P<:ParametricHamiltonian{T,E,0}} <: RegularSelfEnergySolver
    latslice::LatticeSlice{T,E,L}
    flatorbinds::Vector{Int}      # stores the latslice orbital index for each orbital in ph
    ph::P                         # has an extra parameter :ω_internal for the frequency
end

#region ## API ##

SelfEnergy(h::AbstractHamiltonian, model::ParametricModel; kw...) =
    SelfEnergy(h, model, siteselector(; kw...))

function SelfEnergy(h::AbstractHamiltonian{T}, model::ParametricModel, sel::SiteSelector) where {T}
    modelω = model_ω_to_param(model)  # see model.jl - transforms ω into a ω_internal param
    latslice = lattice(h)[sel]
    sliceinds = Int[]
    # this fills sliceinds with the latslice index for each lat0 site
    lat0 = lattice(latslice, sliceinds)
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h))
    bs = MultiBlockStructure(latslice, h)
    # orbital index on latslice for each orbital in lat0
    flatorbinds = flatinds(sliceinds, bs)
    solver = SelfEnergyModel(latslice, flatorbinds, ph)
    return SelfEnergy(solver, latslice)
end

function flatinds(sliceinds, bs::MultiBlockStructure)
    finds = Int[]
    for iunflat in sliceinds
        append!(finds, siterange(bs, iunflat))
    end
    return finds
end

function call!(s::SelfEnergyModel{T}, ω; params...) where {T}
    h = s.ph(; ω_internal = ω, params...)
    m = call!(h, ())
    v = view(m, s.flatorbinds, s.flatorbinds)
    m´ = HybridMatrix(v, s.latslice, s.ph)
    return m´
end

#endregion
#endregion

