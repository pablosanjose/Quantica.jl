using ArnoldiMethod, Random, FFTW

@testset "basic KPM" begin
    h = LatticePresets.honeycomb() |>
        hamiltonian(hopping(1.0I, range = 1/sqrt(3)), orbitals = (Val(1), Val(2))) |>
        unitcell(region = RegionPresets.circle(10))
    brange = bandrangeKPM(h)
    m1 = momentaKPM(h, order = 30, ket = randomkets(1, maporbitals = true))
    m2 = momentaKPM(h, bandrange = brange, order = 30, ket = randomkets(1, maporbitals = true))
    @test abs(m1.mulist[1]) ≈ abs(m2.mulist[1])
    dos1 = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(1, maporbitals = true))
    dos2 = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(1, r -> randn(), maporbitals = true))
    @test all(>(0), last(dos1))
    @test all(>(0), last(dos2))
    dos = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(2, r -> randn() * SA[1 0; 0 -1], sublats = :B))
    @test all(>(0), last(dos))
end
