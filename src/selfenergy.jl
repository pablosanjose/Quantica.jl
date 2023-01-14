############################################################################################
# blockshift for ExtendedSelfEnergySolver output
#    ExtendedSelfEnergySolver returns Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ, which are MatrixBlock's
#    All but Σᵣᵣ are defined over extended orbitals, adjacent to those of Σᵣᵣ
#    blockshift shifts the orbital indices by a fixed quantity
#region

function blockshift!((Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)::Tuple{MatrixBlock}, shift)
    Vᵣₑ´ = blockshift!(Vᵣₑ, (0, shift))
    Vₑᵣ´ = blockshift!(Vₑᵣ, (0, shift))
    gₑₑ⁻¹ = blockshift!(gₑₑ⁻¹, (shift, shift))
    return Σᵣᵣ, Vᵣₑ´, gₑₑ⁻¹´, Vₑᵣ´
end

function blockshift!(b::MatrixBlock, (rowshift, colshift))
    mat = blockmat(m)
    coeff = coefficient(m)
    rows = blockshift!(blockrows(b), rowshift)
    cols = blockshift!(blockcols(b), colshift)
    return MatrixBlock(mat, rows, cols, coeff)
end

blockshift!(r::AbstractUnitRange, shift) = r .+ shift
blockshift!(r::AbstractVector, shift) = (r .+= shift)

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

function call!(s::SelfEnergyModel, ω; params...)
    m = call!(s.ph, (); ω_internal = ω, params...)
    rows = cols = s.flatorbinds
    return MatrixBlock(m, rows, cols)
end

call!_output(s::SelfEnergyModel) =
    MatrixBlock(call!_output(s.ph), s.flatorbinds, s.flatorbinds)

#endregion
#endregion

