using LinearAlgebra: tr

@testset "basic greens function" begin
    g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> unitcell((2,0), (0,1)) |>
        greens(bandstructure(resolution = 17))
    @test g(0.2) ≈ transpose(g(0.2))
    @test imag(tr(g(1))) < 0
    @test Quantica.chop(imag(tr(g(5)))) == 0
end

@testset "greens functions spectra" begin
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    g = greens(h, SingleShot1D())
    @test_throws ArgumentError g(0)
    dos = [-imag(tr(g(w + 1.0e-6im))) for w in range(-3.2, 3.2, length = 1001)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    h = LP.honeycomb() |> hamiltonian(hopping(1, range = 2)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    g = greens(h, SingleShot1D())
    dos = [-imag(tr(g(w))) for w in range(-6, 6, length = 1001)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(3,3)) |> wrap(2)
    g = greens(h, SingleShot1D())
    dos = [-imag(tr(g(w))) for w in range(-3.2, 3.2, length = 1001)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((2,-2),(3,3)) |> unitcell(1, 1, indices = not(1)) |> wrap(2)
    g = greens(h, SingleShot1D())
    @test_broken imag(tr(g(0.2))) < 0
    g = greens(h, SingleShot1D(direct = false))
    @test imag(tr(g(0.2))) < 0
    dos = [-imag(tr(g(w + 1.0e-6im))) for w in range(-3.2, 3.2, length = 1001)]
    @test all(x -> Quantica.chop(x) >= 0, dos)
end

@testset "greens functions spatial" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(3,3)) |> unitcell((1,0))
    g = greens(h, SingleShot1D(), boundaries = (0,))
    g0 = g(0.01, missing)
    ldos = [-imag(tr(g0(n=>n))) for n in 1:200]
    @test all(x -> Quantica.chop(x) >= 0, ldos)
end

