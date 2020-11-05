using Quantica: nbands, nvertices, nedges, nsimplices
using Arpack

@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bandstructure(h, subticks = 13, showprogress = false)
    @test nbands(b)  == 1

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(2)
    b = bandstructure(h, cuboid(0.99 .* (-π, π), 0.99 .* (-π, π), subticks = 13), showprogress = false)
    @test nbands(b)  == 3

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = bandstructure(h, subticks = (13, 15), showprogress = false)
    @test nbands(b)  == 2

    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    b = bandstructure(h, subticks = (5, 7, 5), showprogress = false)
    @test nbands(b)  == 1

    b = bandstructure(h, :Γ, :X; subticks = 4, showprogress = false)
    @test nbands(b)  == 1

    b = bandstructure(h, :Γ, :X, (0, π), :Z, :Γ; subticks = 4, showprogress = false)
    @test nbands(b)  == 1
    @test nvertices(b) == 73

    b = bandstructure(h, :Γ, :X, (0, π), :Z, :Γ; subticks = (4,5,6,7), showprogress = false)
    @test nbands(b) == 1
    @test nvertices(b) == 113

    # complex spectra
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(im) + hopping(-1)) |> unitcell(2)
    b = bandstructure(h, cuboid((-π, π), (-π, π), subticks = 7), showprogress = false)
    @test nbands(b)  == 1

    # spectrum sorting
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(10)
    b = bandstructure(h, :K, :K´, subticks = 7, method = ArpackPackage(sigma = 0.1im, nev = 12), showprogress = false)
    @test nbands(b) == 1
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B, plusadjoint = true))
    matrix = similarmatrix(hc, LinearAlgebraPackage())
    hf((x,)) = bloch!(matrix, hc, (x, -x))
    m = cuboid((0, 1))
    b = bandstructure(hf, m, showprogress = false)
    @test nbands(b)  == 2

    hc2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    hp2 = parametric(hc2, @hopping!((t; s) -> s*t))
    matrix2 = similarmatrix(hc2, LinearAlgebraPackage())
    hf2((s, x)) = bloch!(matrix2, hp2(s = s), (x, x))
    m2 = cuboid((0, 1), (0, 1))
    b = bandstructure(hf2, m2, showprogress = false)
    @test nbands(b)  == 1
end

@testset "bandstructures lifts & transforms" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2) + hopping(-1, range = 1/√3))
    mesh1D = cuboid((0, 2π))
    b = bandstructure(h, mesh1D, mapping = φ -> (φ, -φ), transform = inv, showprogress = false)
    b´ = transform!(inv, bandstructure(h, mesh1D, mapping = φ -> (φ, -φ), showprogress = false))
    @test nbands(b)  == nbands(b´) == 1
    @test vertices(bands(b)[1]) == vertices(bands(b´)[1])
    h´ = unitcell(h)
    s1 = spectrum(h´, transform = inv)
    s2 = transform!(inv, spectrum(h´))
    @test energies(s1) == energies(s2)
    # no automatic mapping from 2D to 3D
    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    @test_throws DimensionMismatch bandstructure(h, cuboid((0, 2pi), (0, 2pi)), showprogress = false)
end

@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
         parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I + p'p))
    mesh2D = cuboid((0, 1), (0, 2π), subticks = 15)
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> (x, (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> ((x,), (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bandstructure(ph, mesh2D, mapping = (x, k) -> (SA[x], (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> (1, (;k = k, p = SA[1, φ])), showprogress = false)
    @test nbands(b)  == 1

    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
        unitcell |> parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I + p'p))
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> (;k = k, p = SA[1, φ]), showprogress = false)
    @test nbands(b)  == 1
    b = bandstructure(ph, mesh2D, mapping = (k, φ) -> ((;k = k, p = SA[1, φ]),), showprogress = false)
    @test nbands(b)  == 1
    @test_throws UndefKeywordError bandstructure(ph, mesh2D, mapping = (k, φ) -> ((;p = SA[1, φ]),), showprogress = false)
end

@testset "unflatten" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> unitcell(2) |> unitcell
    sp = states(spectrum(h))[:,1]
    sp´ = Quantica.unflatten_or_reinterpret(sp, h)
    l = size(h, 1)
    @test length(sp) == 1.5 * l
    @test length(sp´) == l
    @test all(x -> iszero(last(x)), sp´[l+1:end])
    @test sp´ isa Vector
    @test sp´ !== sp

    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> unitcell(2) |> unitcell
    sp = states(spectrum(h))[:,1]
    sp´ = Quantica.unflatten_or_reinterpret(sp, h)
    l = size(h, 1)
    @test length(sp) == 2 * l
    @test length(sp´) == l
    @test sp´ isa Base.ReinterpretArray

    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> unitcell(2) |> unitcell
    sp = states(spectrum(h))[:,1]
    sp´ = Quantica.unflatten_or_reinterpret(sp, h)
    @test sp === sp
end