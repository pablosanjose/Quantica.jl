@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, range = 1/√3))
    b = bandstructure(h, resolution = 13)
    @test length(bands(b)) == 1

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = bandstructure(h, resolution = 13)
    @test length(bands(b)) == 2

    h = LatticePresets.square() |> hamiltonian(hopping(1)) |> unitcell(2)
    b = bandstructure(h, resolution = 13)
    @test length(bands(b)) == 4
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B, plusadjoint = true))
    matrix = similarmatrix(hc, LinearAlgebraPackage())
    hf((x,)) = bloch!(matrix, hc, (x, -x))
    mesh = marchingmesh(range(0, 1, length = 13))
    b = bandstructure(hf, mesh)
    @test length(bands(b)) == 2

    hc2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    hp2 = parametric(hc2, @hopping!((t; s) -> s*t))
    matrix2 = similarmatrix(hc2, LinearAlgebraPackage())
    hf2((s, x)) = bloch!(matrix2, hp2(s = s), (x, x))
    mesh2 = marchingmesh(range(0, 1, length = 13), range(0, 1, length = 13))
    b = bandstructure(hf2, mesh2)
    @test length(bands(b)) == 2
end

@testset "bandstructures cuts" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, range = 1/√3))
    mesh1D = marchingmesh(range(0, 2π, length = 13))
    b = bandstructure(h, mesh1D, cut = φ -> (φ, 0))
    @test length(bands(b)) == 2
    b = bandstructure(h, mesh1D, cut = φ -> (φ, -φ))
    @test length(bands(b)) == 2
end


@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
         parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k)-> t - k*I))
    mesh2D = marchingmesh(range(0, 1, length = 13), range(0, 2π, length = 13))
    mesh1D = marchingmesh(range(0, 2π, length = 13))
    b = bandstructure(ph, mesh2D)
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, cut = (k, φ) -> (k + φ/2π, φ))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, cut = (k, φ) -> (k + φ/2π, φ))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh1D, cut = k -> (k, 2π*k))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh1D, cut = φ -> (φ, -φ))
    @test length(bands(b)) == 4
end