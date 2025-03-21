############################################################################################
# Selfenergy(h, nothing; siteselection...)
#    Empty self energy at selectors
#region

struct SelfEnergyNothingSolver{C} <: EmptySelfEnergySolver      # <: RegularSelfEnergySolver
    emptymat::SparseMatrixCSC{C,Int}
end

function SelfEnergy(h::AbstractHamiltonian{T}, ::Nothing; kw...) where {T}
    orbslice = contact_orbslice(h; kw...)
    norbs = norbitals(orbslice)
    emptyΣ = spzeros(Complex{T}, norbs, norbs)
    solver = SelfEnergyNothingSolver(emptyΣ)
    return SelfEnergy(solver, orbslice)
end

call!(s::SelfEnergyNothingSolver, ω; params...) = s.emptymat

call!_output(s::SelfEnergyNothingSolver) = s.emptymat

minimal_callsafe_copy(s::SelfEnergyNothingSolver) = s

needs_omega_shift(::SelfEnergyNothingSolver) = false

#endregion
