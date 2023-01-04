############################################################################################
# SelfEnergyModel
#region

using Quantica: KDTree, Euclidean, HybridMatrix, ParametricModel, AppliedParametricModel,
      AppliedParametricOnsiteTerm, AppliedParametricHoppingTerm, SiteSelector, HopSelector,
      terms, selector, foreach_site, foreach_hop, sites

struct SelfEnergyModel{T,E,L,A<:AppliedParametricModel} <: AbstractSelfEnergySolver
    model::A
    latslice::LatticeSlice{T,E,L}
    kdtree::KDTree{SVector{E,T},Euclidean,T}  # needed to efficiently apply hopping terms
    mat::HybridMatrix{Complex{T},L}  # preallocation
end

#region ## API ##

SelfEnergy(h::AbstractHamiltonian, m::ParametricModel; kw...) =
    SelfEnergy(h, siteselector(;kw...), m)

function SelfEnergy(h::AbstractHamiltonian, sel::SiteSelector, model::ParametricModel)
    lat = lattice(h)
    amodel = apply(model, lat)
    asel = apply(sel, lat)
    latslice = lat[asel]
    mat = HybridMatrix(latslice, h)
    kdtree = KDTree(collect(sites(latslice)))
    solver = SelfEnergyModel(amodel, latslice, kdtree, mat)
    return SelfEnergy(solver, latslice)
end

function call!(s::SelfEnergyModel{T}, ω; params...) where {T}
    fill!(s.mat, zero(Complex{T}))
    foreach(terms(s.model)) do term
        apply_term!(s, term, ω; params...)
    end
    return s.mat
end

function apply_term!(s::SelfEnergyModel, o::AppliedParametricOnsiteTerm, ω; params...)
    foreach_site(selector(o), s.latslice) do i, r, n, islice
        s.mat[islice, islice] = o(ω, r; params...)
    end
    return s
end

function apply_term!(s, t::AppliedParametricHoppingTerm, ω; params...)
    foreach_hop(selector(t), s.latslice, s.kdtree) do is, (r, dr), ns, (islice, jslice)
        s.mat[islice, jslice] = t(ω, r, dr; params...)
    end
    return s
end

#endregion
#endregion

