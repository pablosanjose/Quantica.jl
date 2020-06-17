using ArnoldiMethod, Random, FFTW

@testset "basic KPM" begin
    h = LatticePresets.honeycomb() |>
        hamiltonian(hopping(-1, range = 1/sqrt(3))) |>
        unitcell(region = RegionPresets.circle(10))
    dos = dosKPM(h, order = 10, bandrange = (-3,3))
    @test all(ρ -> 0 < ρ < 10, last(dos))
end