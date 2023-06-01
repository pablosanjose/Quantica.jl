############################################################################################
# HamiltonianPresets
#region

module HamiltonianPresets

using Quantica, LinearAlgebra

function graphene(; a0 = 0.246, range = a0/sqrt(3), t0 = 2.7, β = 3, dim = 2, type = Float64, names = (:A, :B), kw...)
    lat = LatticePresets.honeycomb(; a0, dim, type, names)
    h = hamiltonian(lat,
        hopping((r, dr) -> t0 * exp(-β*(sqrt(3) * norm(dr)/a0 - 1)) * I,range = range); kw...)
    return h
end

function twisted_bilayer_graphene(;
    twistindex = 1, twistindices = (twistindex, 1), a0 = 0.246,
    interlayerdistance = 1.36a0, rangeintralayer = a0/sqrt(3), rangeinterlayer = 4a0/sqrt(3),
    hopintra = 2.70 * I, hopinter = 0.48, modelintra = hopping(hopintra, range = rangeintralayer),
    type = Float64, names = (:Ab, :Bb, :At, :Bt),
    kw...)

    (m, r) = twistindices
    θ = acos((3m^2 + 3m*r +r^2/2)/(3m^2 + 3m*r + r^2))
    sAbot = sublat((0.0, -0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Ab)
    sBbot = sublat((0.0,  0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Bb)
    sAtop = sublat((0.0, -0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :At)
    sBtop = sublat((0.0,  0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :Bt)
    brbot = a0 * SA[ cos(pi/3) sin(pi/3) 0; -cos(pi/3) sin(pi/3) 0]'
    brtop = a0 * SA[ cos(pi/3) sin(pi/3) 0; -cos(pi/3) sin(pi/3) 0]'
    # Supercell matrices sc.
    # The one here is a [1 0; -1 1] rotation of the one in Phys. Rev. B 86, 155449 (2012)
    if gcd(r, 3) == 1
        scbot = SA[m -(m+r); (m+r) 2m+r] * SA[1 0; -1 1]
        sctop = SA[m+r -m; m 2m+r] * SA[1 0; -1 1]
    else
        scbot = SA[m+r÷3 -r÷3; r÷3 m+2r÷3] * SA[1 0; -1 1]
        sctop = SA[m+2r÷3 r÷3; -r÷3 m+r÷3] * SA[1 0; -1 1]
    end

    latbot = lattice(sAbot, sBbot; bravais = brbot, dim = Val(3), type, names = (names[1], names[2]))
    lattop = lattice(sAtop, sBtop; bravais = brtop, dim = Val(3), type, names = (names[3], names[4]))
    htop = hamiltonian(lattop, modelintra; kw...) |> supercell(sctop)
    hbot = hamiltonian(latbot, modelintra; kw...) |> supercell(scbot)
    let R = SA[cos(θ/2) -sin(θ/2) 0; sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(htop, r -> R * r)
    end
    let R = SA[cos(θ/2) sin(θ/2) 0; -sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(hbot, r -> R * r)
    end
    modelinter = hopping((r,dr) -> (
        I * hopintra * exp(-3*(norm(dr)/a0 - 1))  *  dot(dr, SVector(1,1,0))^2/sum(abs2, dr) -
        I * hopinter * exp(-3*(norm(dr)/a0 - interlayerdistance/a0)) * dr[3]^2/sum(abs2, dr)),
        range = rangeinterlayer)
    return combine(hbot, htop; coupling = modelinter)
end

end # module

const HP = HamiltonianPresets

#endregion