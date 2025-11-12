############################################################################################
# HamiltonianPresets
#region

module HamiltonianPresets

using Quantica, LinearAlgebra

function graphene(; a0 = 0.246, range = neighbors(1), t0 = 2.7, β = 3, dim = 2, type = Float64, names = (:A, :B), kw...)
    lat = LatticePresets.honeycomb(; a0, dim, type, names)
    h = hamiltonian(lat,
        hopping((r, dr) -> t0 * exp(-β*(sqrt(3) * norm(dr)/a0 - 1)) * I, range = range); kw...)
    return h
end

function twisted_bilayer_graphene(;
    twistindex = 1, twistindices = (twistindex, 1), a0 = 0.246,
    interlayerdistance = 1.36a0, rangeintralayer = neighbors(1), rangeinterlayer = 4a0/sqrt(3),
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
        Quantica.transform!(htop, r -> R * r)
    end
    let R = SA[cos(θ/2) sin(θ/2) 0; -sin(θ/2) cos(θ/2) 0; 0 0 1]
        Quantica.transform!(hbot, r -> R * r)
    end
    modelinter = hopping((r,dr) -> (
        I * hopintra * exp(-3*(norm(dr)/a0 - 1))  *  dot(dr, SVector(1,1,0))^2/sum(abs2, dr) -
        I * hopinter * exp(-3*(norm(dr)/a0 - interlayerdistance/a0)) * dr[3]^2/sum(abs2, dr)),
        range = rangeinterlayer)
    return combine(hbot, htop; coupling = modelinter)
end

# from arXiv:2502.17555v1, but γ2 seems wrong
function rhombohedral_graphene(; N = 4, spinoperator = I, a0 = 0.246, interlayerdistance = a0, γ0 = 3.16, γ1 = 0.445, γ2 = -0.0182, γ3 = -0.319, γ4 = -0.079, γ5 = 0, δ = -0.000066, δAB = 0, kw...)
    aCC = a0 / sqrt(3)
    az = interlayerdistance
    Δz = az*(N-1)
    d0 = aCC
    d1 = az
    d2 = 2 * az
    d34 = hypot(aCC, az)
    d5 = hypot(aCC, 2az)
    op = spinoperator
    subs3 = (:A => :B, :B => :A)
    subs4 = (:A => :A, :B => :B)
    subs5 = subs3
    inplane(_, dr) = iszero(dr[3])
    vertical(_, dr) = norm(dr[SA[1,2]]) < eps(Float64)

    lat0 = LP.honeycomb(; a0, dim = 3)
    Asites = [only(sites(lat0, :A)) + SA[0, aCC, az]*n for n in 0: N-1]
    Bsites = [only(sites(lat0, :B)) + SA[0, aCC, az]*n for n in 0: N-1]
    lat = lattice(sublat(Asites, name = :A), sublat(Bsites, name = :B),
        bravais = bravais_matrix(lat0))
    Quantica.translate!(lat, -mean(sites(lat)))

    model =
        @onsite((r; U = 0) -> U * (r[3]/Δz) * op) +
        onsite(r-> (δAB + δ*ifelse(r[3]≈-Δz/2,1,0)) * op, sublats = :A) +
        onsite(r-> (-δAB + δ*ifelse(r[3]≈Δz/2,1,0)) * op, sublats = :B) +
        hopping(γ0*op, range = (d0, d0), region = inplane) +
        hopping(γ1*op, range = (d1, d1), region = vertical) +
        hopping(0.5*γ2*op, range = (d2, d2), region = vertical) +
        hopping(γ3*op, range = (d34, d34), sublats = subs3, region = !inplane) +
        hopping(γ4*op, range = (d34, d34), sublats = subs4, region = !inplane) +
        hopping(γ5*op, range = (d5, d5), sublats = subs5, region = !inplane)
    h = lat |> hamiltonian(model; kw...)

    return h
end

end # module

const HP = HamiltonianPresets

#endregion
