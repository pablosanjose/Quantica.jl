using Quantica, ArnoldiMethod, LinearAlgebra, NumericalIntegration, FFTW


h = LP.honeycomb() |>
    hamiltonian(hopping(1) + onsite(0., sublats = :A) - onsite(0., sublats = :B)) |>
    unitcell(region = RegionPresets.circle(20))

a = 0.1
eigs_dacp = DACPdiagonaliser(h, a,  maxdeg = 2, store_basis = false, d = 20)
eigs_dacp = DACPdiagonaliser(h, a,  maxdeg = 2, store_basis = true, d = 20)
eigs_dacp = DACPdiagonaliser(h, a,  maxdeg = 1, store_basis = true)
eigs_dacp = DACPdiagonaliser(h, a,  maxdeg = 1, store_basis = false)