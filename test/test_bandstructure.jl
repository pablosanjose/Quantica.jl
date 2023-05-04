using Quantica: nbands, nvertices, nedges, nsimplices, Subspace
using Arpack

@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bands(h, subticks = 13, showprogress = false)
    @test nbands(b)  == 1

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(2)
    b = bands(h, cuboid(0.99 .* (-π, π), 0.99 .* (-π, π), subticks = 13), showprogress = false)
    @test nbands(b)  == 3

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = bands(h, subticks = (13, 15), showprogress = false)
    @test nbands(b)  == 2

    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    b = bands(h, subticks = (5, 7, 5), showprogress = false)
    @test nbands(b)  == 1

    b = bands(h, :Γ, :X; subticks = 4, showprogress = false)
    @test nbands(b)  == 1

    b = bands(h, :Γ, :X, (0, π), :Z, :Γ; subticks = 4, showprogress = false)
    @test nbands(b)  == 1
    @test nvertices(b) == 73

    b = bands(h, :Γ, :X, (0, π), :Z, :Γ; subticks = (4,5,6,7), showprogress = false)
    @test nbands(b) == 1
    @test nvertices(b) == 113

    # complex spectra
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(im) + hopping(-1)) |> unitcell(2)
    b = bands(h, cuboid((-π, π), (-π, π), subticks = 7), showprogress = false)
    @test nbands(b)  == 1

    # spectrum sorting
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(10)
    b = bands(h, :K, :K´, subticks = 7, method = ArpackPackage(sigma = 0.1im, nev = 12), showprogress = false)
    @test nbands(b) == 1

    # number of simplices contant across BZ (computing their degeneracy with the projs at each vertex j)
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(3)
    nev = size(h, 1)
    b = bands(h, splitbands = false)
    @test nbands(b) == 1
    band = bands(b, 1)
    @test all(j -> all(rng -> sum(i -> degeneracy(band.sbases[i][j]), rng) == nev, band.sptrs), 1:3)
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B, plusadjoint = true))
    matrix = similarmatrix(hc, LinearAlgebraPackage())
    hf((x,)) = bloch!(matrix, hc, (x, -x))
    m = cuboid((0, 1))
    b = bands(hf, m, showprogress = false)
    @test nbands(b) == 2
    @test nsimplices(b)  == 24
    @test b[(1,), around = 0] isa Subspace

    hc2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    hp2 = parametric(hc2, @hopping!((t; s) -> s*t))
    matrix2 = similarmatrix(hc2, LinearAlgebraPackage())
    hf2((s, x)) = bloch!(matrix2, hp2(s = s), (x, x))
    m2 = cuboid((0, 1), (0, 1))
    b = bands(hf2, m2, showprogress = false)
    @test nbands(b) == 1
    @test nsimplices(b)  == 576
    @test degeneracy.(b[(0,0), around = (0, 2)]) == [1, 1]
end

@testset "bandstructures lifts & transforms" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2) + hopping(-1, range = 1/√3))
    mesh1D = cuboid((0, 2π))
    b = bands(h, mesh1D, mapping = φ -> (φ, -φ), transform = inv, showprogress = false)
    b´ = transform!(inv, bands(h, mesh1D, mapping = φ -> (φ, -φ), showprogress = false))
    @test nbands(b)  == nbands(b´) == 1
    @test vertices(bands(b)[1]) == vertices(bands(b´)[1])
    h´ = unitcell(h)
    s1 = spectrum(h´, transform = inv)
    s2 = transform!(inv, spectrum(h´))
    @test s1.energies == s2.energies
    # no automatic mapping from 2D to 3D
    h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(2)
    @test_throws DimensionMismatch bands(h, cuboid((0, 2pi), (0, 2pi)), showprogress = false)
end

@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
         parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    mesh2D = cuboid((0, 1), (0, 2π), subticks = 15)
    b = bands(ph, mesh2D, mapping = (x, k) -> (x, (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bands(ph, mesh2D, mapping = (x, k) -> ((x,), (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bands(ph, mesh2D, mapping = (x, k) -> (SA[x], (;k = k)), showprogress = false)
    @test nbands(b)  == 4
    b = bands(ph, mesh2D, mapping = (k, φ) -> (1, (;k = k, p = SA[1, φ])), showprogress = false)
    @test nbands(b)  == 1

    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> unitcell(2) |>
        unitcell |> parametric(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    b = bands(ph, mesh2D, mapping = (k, φ) -> (;k = k, p = SA[1, φ]), showprogress = false)
    @test nbands(b)  == 1
    b = bands(ph, mesh2D, mapping = (k, φ) -> ((;k = k, p = SA[1, φ]),), showprogress = false)
    @test nbands(b)  == 1
    @test_throws UndefKeywordError bands(ph, mesh2D, mapping = (k, φ) -> ((;p = SA[1, φ]),), showprogress = false)
end

@testset "unflatten" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> unitcell(2) |> unitcell
    sp = spectrum(h).states[:,1]
    sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
    l = size(h, 1)
    @test length(sp) == 1.5 * l
    @test length(sp´) == l
    @test all(x -> iszero(last(x)), sp´[l+1:end])
    @test sp´ isa Vector
    @test sp´ !== sp

    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> unitcell(2)
    b = bands(h)
    psi = b[(0,0), around = 0]
    psi´ = flatten(psi)
    psi´´ = unflatten(psi´, orbitalstructure(psi))
    @test eltype(psi.basis) == eltype(psi´´.basis) == SVector{2, ComplexF64}
    @test eltype(psi´.basis) == ComplexF64
    @test psi.basis == psi´´.basis

    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> unitcell(2) |> unitcell
    sp = spectrum(h).states[:,1]
    sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
    l = size(h, 1)
    @test length(sp) == 2 * l
    @test length(sp´) == l
    @test sp´ isa Base.ReinterpretArray

    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> unitcell(2) |> unitcell
    sp = spectrum(h).states[:,1]
    sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
    @test sp === sp
end

@testset "spectrum indexing" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell(2) |> wrap
    s = spectrum(h)
    ϵv, ψm = s
    @test ϵv == first(s) == s.energies
    @test ψm == last(s) == s.states
    @test s[1] isa Subspace
    @test degeneracy(s[2]) == 3
    @test s[1:3] isa Vector{<:Subspace}
    @test s[[1,3]] isa Vector{<:Subspace}
    ϵ, ψs = s[1]
    @test ϵ isa Number
    @test ψs isa SubArray{<:Complex, 2}
end

@testset "bands indexing" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell(2)
    bs = bands(h, subticks = 13)
    @test sum(degeneracy, bs[(1,2)]) == size(h,1)
    @test sum(degeneracy, bs[(0.2,0.3)]) == size(h,1)
    @test size(bs[(1,2), around = 0].basis, 1) == size(h, 2)
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1f0*I), orbitals = (Val(1),Val(2))) |> unitcell(2)
    bs = bands(h, subticks = 13)
    @test sum(degeneracy, bs[(1,2)]) == size(h,1) * 1.5
    @test sum(degeneracy, bs[(0.2,0.3)]) == size(h,1) * 1.5
    @test bs[(1,2), around = 0] |> last |> eltype == SVector{2, ComplexF64}
    @test size(bs[(1,2), around = 0].basis, 1) == size(h, 2)
end

@testset "subspace flatten" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(2))) |> unitcell(2) |> unitcell
    sub = spectrum(h)[1]
    @test size(sub.basis) == (8,1) && size(flatten(sub).basis) == (12,1)
end

@testset "diagonalizer" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1))
    d = diagonalizer(h)
    @test first(d((0, 0))) ≈ [-3,3]
    h = wrap(h)
    d = diagonalizer(h)
    @test first(d()) ≈ [-3,3]
    @test first(d(())) ≈ [-3,3]
    @test all(d() .≈ Tuple(spectrum(h)))
end

@testset "bands extrema" begin
    h = LP.honeycomb() |>
        hamiltonian(hopping(I) + onsite(0.1, sublats = :A) - onsite(0.1, sublats = :B)) |>
        unitcell(3) |>
        Quantica.wrap(1)
    b = bands(h, subticks = 20)
    @test !isapprox(gap(b, 0; refinesteps = 0), 0.2)
    @test isapprox(gap(b, 0; refinesteps = 1), 0.2)
    @test isapprox(gap(b, 0.3; refinesteps = 1), 0)
    @test isapprox(gap(b, 4; refinesteps = 1), Inf)
    @test all(gapedge(b, 0, +; refinesteps = 1) .≈ (0, 0.1))
    @test all(gapedge(b, 0, -; refinesteps = 1) .≈ (0, -0.1))

    b = bands(h, subticks = 20; method = ArpackPackage(nev = 6, sigma = 0.1*im))
    @test !isapprox(gap(b, 0; refinesteps = 0), 0.2)
    @test isapprox(gap(b, 0; refinesteps = 1), 0.2)
    @test isapprox(gap(b, 0.3; refinesteps = 1), 0)
    @test isapprox(gap(b, 4; refinesteps = 1), Inf)
    @test all(gapedge(b, 0, +; refinesteps = 1) .≈ (0, 0.1))
    @test all(gapedge(b, 0, -; refinesteps = 1) .≈ (0, -0.1))

    h = LP.honeycomb() |> hamiltonian(hopping(I)) |> Quantica.wrap(1)
    b = bands(h, subticks = 20)
    length.(minima(b, refinesteps = 0)) == [2, 0]
    length.(maxima(b, refinesteps = 0)) == [0, 2]
    length.(minima(b, refinesteps = 1)) == [1, 0]
    length.(maxima(b, refinesteps = 1)) == [0, 1]
end