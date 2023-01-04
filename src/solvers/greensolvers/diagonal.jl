############################################################################################
# SelfEnergyModel
#region

using Quantica: HybridMatrix, ParametricModel, AppliedParametricModel,
      AppliedParametricOnsiteTerm, AppliedParametricHoppingTerm, SiteSelector, HopSelector,
      terms, selector, foreach_site, foreach_hop

struct SelfEnergyModel{T,E,L,A<:AppliedParametricModel} <: AbstractSelfEnergySolver
    model::A
    latslice::LatticeSlice{T,E,L}
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
    solver = SelfEnergyModel(amodel, latslice, mat)
    return SelfEnergy(solver, latslice)
end

function call!(s::SelfEnergyModel{T}, ω; params...) where {T}
    fill!(s.mat, zero(Complex{T}))
    foreach(terms(s.model)) do term
        apply_term!(s.mat, s.latslice, term, ω; params...)
    end
    return s.mat
end

function apply_term!(mat, latslice, o::AppliedParametricOnsiteTerm, ω; params...)
    foreach_site(selector(o), latslice) do i, r, n, islice
        mat[islice, islice] = o(ω, r; params...)
    end
    return mat
end

function apply_term!(mat, latslice, t::AppliedParametricHoppingTerm, ω; params...)
    foreach_hop(selector(t), latslice) do is, (r, dr), ns, (islice, jslice)
        mat[islice, jslice] = t(ω, r, dr; params...)
    end
    return mat
end

#endregion
#endregion

