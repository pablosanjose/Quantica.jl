@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bandstructure(h, points = 13)
    @test length(bands(b)) == 1

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = bandstructure(h, points = (13, 23))
    @test length(bands(b)) == 2

    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    b = bandstructure(h, points = (5, 9, 5))
    @test length(bands(b)) == 8

    b = bandstructure(h, :Γ, :X; points = 4)
    @test length(bands(b)) == 8

    b = bandstructure(h, :Γ, :X, (0, π), :Z, :Γ; points = 4)
    @test length(bands(b)) == 8

    b = bandstructure(h, :Γ, :X, (0, π), :Z, :Γ; points = (4,5,6,7))
    @test length(bands(b)) == 8
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B, plusadjoint = true))
    matrix = similarmatrix(hc, LinearAlgebraPackage())
    hf((x,)) = bloch!(matrix, hc, (x, -x))
    m = mesh((0, 1))
    b = bandstructure(hf, m)
    @test length(bands(b)) == 2

    hc2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    hp2 = parametric(hc2, @hopping!((t; s) -> s*t))
    matrix2 = similarmatrix(hc2, LinearAlgebraPackage())
    hf2((s, x)) = bloch!(matrix2, hp2(s = s), (x, x))
    m2 = mesh((0, 1), (0, 1))
    b = bandstructure(hf2, m2)
    @test length(bands(b)) == 2
end

@testset "bandstructures lifts & transforms" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2) + hopping(-1, range = 1/√3))
    mesh1D = mesh((0, 2π))
    b = bandstructure(h, mesh1D, mapping = φ -> (φ, -φ), transform = inv)
    b´ = transform!(inv, bandstructure(h, mesh1D, mapping = φ -> (φ, -φ)))
    @test length(bands(b)) == length(bands(b´)) == 2
    @test vertices(bands(b)[1]) == vertices(bands(b´)[1])
    h´ = unitcell(h)
    s1 = spectrum(h´, transform = inv)
    s2 = transform!(inv, spectrum(h´))
    @test energies(s1) == energies(s2)
    # no automatic mapping from 2D to 3D
    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    @test_throws DimensionMismatch bandstructure(h, mesh((0, 2pi), (0, 2pi)))
end

@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
         parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I + p'p))
    mesh2D = mesh((0, 1), (0, 2π), points = 25)
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> (x, (;k = k)))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> ((x,), (;k = k)))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> (SA[x], (;k = k)))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> (1, (;k = k, p = SA[1, φ])))
    @test length(bands(b)) == 4

    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
        unitcell |> parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I + p'p))
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> (;k = k, p = SA[1, φ]))
    @test length(bands(b)) == 4
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> ((;k = k, p = SA[1, φ]),))
    @test length(bands(b)) == 4
    @test_throws UndefKeywordError bandstructure(ph, mesh2D, mapping = (k, φ) -> ((;p = SA[1, φ]),))
end