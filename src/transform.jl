############################################################################################
# Currying
#region

transform(f::Function) = x -> transform(x, f)

transform!(f::Function) = x -> transform!(x, f)

translate(δr) = x -> translate(x, δr)

translate!(δr) = x -> translate!(x, δr)

#endregion

############################################################################################
# Lattice transform/translate
#region

function transform!(l::Lattice, f::Function, keepranges = false)
    return keepranges ?
        Lattice(transform!(bravais(l), f), transform!(unitcell(l), f), nranges(l)) :
        Lattice(transform!(bravais(l), f), transform!(unitcell(l), f))
end

function transform!(b::Bravais{<:Any,E}, f::Function) where {E}
    m = matrix(b)
    for j in axes(m, 2)
        v = SVector(ntuple(i -> m[i, j], Val(E)))
        m[:, j] .= f(v) - f(zero(v))
    end
    return b
end

transform!(u::Unitcell, f::Function) = (map!(f, sites(u), sites(u)); u)
transform(l::Lattice, f::Function) = transform!(copy(l), f)

# translate! does not change neighbor ranges, keep whichever have already been computed
translate!(lat::Lattice{T,E}, δr::SVector{E,T}) where {T,E} = transform!(lat, r -> r + δr, true)
translate!(lat::Lattice{T,E}, δr) where {T,E} = translate!(lat, sanitize_SVector(SVector{E,T}, δr))
translate(l::Lattice, δr) = translate!(copy(l), δr)

#endregion

############################################################################################
# reverse - flip all Bravais vectors of a lattice, and all dn in hamiltonian harmonics
#   As a general rule, reverse does not change the Hamiltonian, only the meaning of the
#   Bloch phase ϕ -> -ϕ, so that H(k) -> H(k), but H(ϕ) -> H(-ϕ)
#   reverse_bravais!(ph::ParametricHamiltonian) is dangerous - it flips the harmonics of parent(ph)!
#   We don't export it or document it to avoid user surprises.
#region

Base.reverse(lat::Lattice) = reverse_bravais!(copy(lat))

Base.reverse(h::AbstractHamiltonian) = reverse_bravais!(copy(h))

# unexported
reverse_bravais!(lat::Lattice) = (matrix(bravais(lat)) .*= -1; lat)

function reverse_bravais!(h::Hamiltonian)
    reverse_bravais!(lattice(h))
    hars = harmonics(h)
    for (i, har) in enumerate(hars)
        hars[i] = Harmonic(-dcell(har), matrix(har))
    end
    return h
end

function reverse_bravais!(ph::ParametricHamiltonian)
    reverse_bravais!(parent(ph))
    reverse_bravais!(hamiltonian(ph))
    reverse_bravais!.(modifiers(ph))
    return ph
end

# by default, modifiers do not care about reverse
reverse_bravais!(m::AbstractModifier) = m

# AppliedHoppingModifiers contain CellSite's that contain nonzero dcell that must be flipped
function reverse_bravais!(m::AppliedHoppingModifier)
    ptrs = pointers(m)
    for pcell in ptrs, (i, p) in enumerate(pcell)
        (ptr, r, dr, si, sj, norbs) = p
        pcell[i] = (ptr, r, dr, reverse(si), reverse(sj), norbs)
    end
    return m
end

# The StitchModifier is special, in that it contains a reference to the dn of stitched
# harmonics that are a sum over subsets of parent harmonics. If the dcell of the former are
# flipped, we must flip the dcell reference to them as well.
reverse_bravais!(m::StitchModifier) = flip_dcells!(m)

#endregion

############################################################################################
# Hamiltonian transform/translate
#region

transform(h::AbstractHamiltonian, f::Function) = transform!(copy_lattice(h), f)
transform!(h::AbstractHamiltonian, f::Function) = (transform!(lattice(h), f); h)

translate(h::AbstractHamiltonian, δr) = translate!(copy_lattice(h), δr)
translate!(h::AbstractHamiltonian, δr) = (translate!(lattice(h), δr); h)

#endregion
