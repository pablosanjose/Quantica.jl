############################################################################################
# contact_block_structure constructors
#region

contact_block_structure(h::AbstractHamiltonian, lss...) =
    contact_block_structure(blockstructure(h), lss...)

function contact_block_structure(bs::OrbitalBlockStructure, lss...)
    lsall = merge(lss...)
    subcelloffsets = [0]
    siteoffsets = [0]
    orbscells = [Subcell(cell(sc), Int[]) for sc in subcells(lsall)]
    offsetall = offsetcell = 0
    for (oc, sc) in zip(orbscells, subcells(lsall))
        offsetcell = 0
        for i in siteindices(sc)
            bsize = blocksize(bs, i)
            append!(siteindices(oc), offsetcell+1:offsetcell+bsize)
            offsetall += bsize
            offsetcell += bsize
            push!(siteoffsets, offsetall)
        end
        push!(subcelloffsets, offsetall)
    end
    contactinds = [contact_indices(lsall, siteoffsets, ls) for ls in lss]
    osall = OrbitalSlice(orbscells)
    return ContactBlockStructure(osall, contactinds, siteoffsets, subcelloffsets)
end

function contact_indices(lsall::LatticeSlice, siteoffsets, ls::LatticeSlice)
    contactinds = Int[]
    for scell´ in subcells(ls)
        so = findsubcell(cell(scell´), lsall)
        so === nothing && continue
        # here offset is the number of sites in lsall before scell
        (scell, offset) = so
        for i´ in siteindices(scell´), (n, i) in enumerate(siteindices(scell))
            n´ = offset + n
            i == i´ && append!(contactinds, siteoffsets[n´]+1:siteoffsets[n´+1])
        end
    end
    return contactinds
end

#endregion

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

function SelfEnergy(h::AbstractHamiltonian, model::ParametricModel, sel::SiteSelector)
    modelω = model_ω_to_param(model)  # see model.jl - transforms ω into a ω_internal param
    latslice = lattice(h)[sel]
    sliceinds = Int[]
    # this fills sliceinds::Vector{Int} with the latslice index for each lat0 site
    lat0 = lattice(latslice, sliceinds)
    # this is a 0D ParametricHamiltonian to build the Σ(ω) as a view over flat(ph(; ...))
    ph = hamiltonian(lat0, modelω; orbitals = norbitals(h))
    # this build siteoffsets for all h orbitals over latslice
    bs = contact_block_structure(h, latslice)
    # translation from lat0 to latslice orbital indices
    # i.e. orbital index on latslice for each orbital in lat0
    flatorbinds´ = flatorbinds(sliceinds, bs)
    solver = SelfEnergyModel(latslice, flatorbinds´, ph)
    return SelfEnergy(solver, latslice)
end

function flatorbinds(sliceinds, bs::ContactBlockStructure)
    finds = Int[]
    for iunflat in sliceinds
        append!(finds, siterange(bs, iunflat))
    end
    return finds
end

function call!(s::SelfEnergyModel, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.flatorbinds
    return view(m, rows, cols)
end

call!_output(s::SelfEnergyModel) =
    view(call!_output(s.ph), s.flatorbinds, s.flatorbinds)

#endregion
#endregion

