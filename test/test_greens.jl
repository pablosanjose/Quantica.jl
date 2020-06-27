using LinearAlgebra: tr

@testset "basic green's function" begin
    g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> unitcell((2,0), (0,1)) |>
        greens(bandstructure(resolution = 17))
    @test g(0.2) ≈ transpose(g(0.2))
    @test imag(tr(g(1))) < 0
    @test Quantica.chop(imag(tr(g(5)))) == 0
end