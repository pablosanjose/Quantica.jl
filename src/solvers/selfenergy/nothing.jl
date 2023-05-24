############################################################################################
# Selfenergy(h, nothing; siteselection...)
#    Empty self energy at selectors
#region

struct SelfEnergyEmptySolver{C} <: RegularSelfEnergySolver
    emptymat::SparseMatrixCSC{C,Int}
end

function SelfEnergy(h::AbstractHamiltonian{T}, ::Nothing; kw...) where {T}
    sel = siteselector(; kw...)
    latslice = lattice(h)[sel]
    plottables = ()
    norbs = flatsize(h, latslice)
    emptyΣ = spzeros(Complex{T}, norbs, norbs)
    solver = SelfEnergyEmptySolver(emptyΣ)
    return SelfEnergy(solver, latslice, plottables)
end

call!(s::SelfEnergyEmptySolver, ω; params...) = s.emptymat

call!_output(s::SelfEnergyEmptySolver) = s.emptymat

minimal_callsafe_copy(s::SelfEnergyEmptySolver) = s

#endregion