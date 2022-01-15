############################################################################################
# LatticePresets
#region

module LatticePresets

using Quantica

linear(; a0 = 1, kw...) =
    lattice(sublat((0.,)); bravais = a0 .* (1,), kw...)

square(; a0 = 1, kw...) =
    lattice(sublat((0., 0.)); bravais = a0 * SA[1. 0.; 0. 1.]', kw...)

triangular(; a0 = 1, kw...) =
    lattice(sublat((0., 0.)); bravais = a0 * SA[cos(pi/3) sin(pi/3); -cos(pi/3) sin(pi/3)]', kw...)

honeycomb(; a0 = 1, kw...) =
    lattice(sublat((0.0, -0.5*a0/sqrt(3.0)), name = :A),
            sublat((0.0,  0.5*a0/sqrt(3.0)), name = :B);
            bravais = a0 * SA[cos(pi/3) sin(pi/3); -cos(pi/3) sin(pi/3)]', kw...)

cubic(; a0 = 1, kw...) =
    lattice(sublat((0., 0., 0.)); bravais = a0 * SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]', kw...)

fcc(; a0 = 1, kw...) =
    lattice(sublat((0., 0., 0.)); bravais = (a0/sqrt(2.)) * SA[-1. -1. 0.; 1. -1. 0.; 0. 1. -1.]', kw...)

bcc(; a0 = 1, kw...) =
    lattice(sublat((0., 0., 0.)); bravais = a0 * SA[1. 0. 0.; 0. 1. 0.; 0.5 0.5 0.5]', kw...)

hcp(; a0 = 1, kw...) =
    lattice(sublat((0., 0., 0.), (a0*0.5, a0*0.5/sqrt(3.), a0*0.5)); bravais = a0 * SA[1. 0. 0.; -0.5 0.5*sqrt(3.) 0.; 0. 0. 1.]', kw...)

end # module

const LP = LatticePresets

#endregion