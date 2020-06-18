using ArnoldiMethod, Random, FFTW

@testset "basic KPM" begin
    h = LatticePresets.honeycomb() |>
        hamiltonian(hopping(SA[0 1; 1 0], range = 1/sqrt(3)), orbitals = Val(2)) |>
        unitcell(region = RegionPresets.circle(10))
    brange = bandrangeKPM(h)
    m1 = momentaKPM(h, order = 30)
    m2 = momentaKPM(h, bandrange = brange, order = 30)
    @test abs(m1.mulist[1]) ≈ abs(m2.mulist[1])
    dos = dosKPM(h, order = 10, bandrange = (-3,3))
    @test all(ρ -> 0 < ρ < 10, last(dos))
end
