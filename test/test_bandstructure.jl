@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A => :B))
    b = bandstructure(h, npoints = 13)
    @test length(bands(b)) == 1

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, sublats = :A => :B))
    b = bandstructure(h, npoints = 13)
    @test length(bands(b)) == 2

    h = LatticePresets.square() |> hamiltonian(hopping(1)) |> unitcell(2)
    b = bandstructure(h, npoints = 13)
    @test length(bands(b)) == 4
end

@testset "functional bandstructures" begin
    const hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A => :B))
    const matrix = similarmatrix(hc, LinearAlgebraPackage())
    hf(x) = bloch!(matrix, hc, (x, -x))
    b = bandstructure(hf, npoints = 13)
    @test length(bands(b)) == 2

    hc2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) 
    const hp = parametric(hc2, hopping!((t; s) -> s*t))
    const matrix = similarmatrix(hc2, LinearAlgebraPackage())
    hf(s, x) = bloch!(matrix, hp(s = s), (x, x))
    b = bandstructure(hf, npoints = (13,13))
    @test length(bands(b)) == 2
end