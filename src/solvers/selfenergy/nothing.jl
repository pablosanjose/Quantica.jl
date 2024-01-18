############################################################################################
# Selfenergy(h, nothing; siteselection...)
#    Empty self energy at selectors
#region

struct SelfEnergyEmptySolver{C} <: RegularSelfEnergySolver
    emptymat::SparseMatrixCSC{C,Int}
end

function SelfEnergy(h::AbstractHamiltonian{T}, ::Nothing; kw...) where {T}
    sel = siteselector(; kw...)
    orbslice = sites_to_orbs(lattice(h)[sel], h)
    plottables = ()
    norbs = norbitals(orbslice)
    emptyΣ = spzeros(Complex{T}, norbs, norbs)
    solver = SelfEnergyEmptySolver(emptyΣ)
    return SelfEnergy(solver, orbslice, plottables)
end

call!(s::SelfEnergyEmptySolver, ω; params...) = s.emptymat

call!_output(s::SelfEnergyEmptySolver) = s.emptymat

minimal_callsafe_copy(s::SelfEnergyEmptySolver) = s

#endregion
