using Quantica: nsubbands, nvertices, nedges, nsimplices

@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi, pi, 13), showprogress = false)
    @test nsubbands(b)  == 1

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> supercell(2)
    b = bands(h, 0.99 * subdiv(-pi, pi, 13), 0.99 * subdiv(-pi, pi, 13), showprogress = false)
    @test nsubbands(b)  == 3

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi, pi, 15), showprogress = false)
    @test nsubbands(b)  == 2

    h = LatticePresets.cubic() |> hamiltonian(hopping((r,dr)-> im*dr'SA[1,1.5,2])) |> supercell(2)
    b = bands(h, subdiv(-pi, pi, 5), subdiv(-pi, pi, 7), subdiv(-pi, pi, 5), showprogress = false)
    @test nsubbands(b)  == 1

    b = bands(h, subdiv(0, 1, 4), mapping = (:Γ, :X), showprogress = false)
    @test nsubbands(b)  == 1

    b = bands(h, subdiv(0, 4, 13), mapping = (:Γ, :X, (0, π), :Z, :Γ), showprogress = false)
    @test nsubbands(b)  == 1
    @test nvertices(b) == 32

    b = bands(h, subdiv(1:5, (4,5,6,7)), mapping = [1,2,3,5,4] => (:Γ, :X, (0, π), :Z, :Γ), showprogress = false)
    @test nsubbands(b) == 1
    @test nvertices(b) == 42

    b = bands(h, subdiv((1,3,4), 5), mapping = (1,4,3) => (:Γ, :X, :Z), showprogress = false)
    @test nsubbands(b) == 1
    @test nvertices(b) == 23

    # complex spectra
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(im) + hopping(-1)) |> supercell(2)
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi,pi, 13), showprogress = false)
    @test nsubbands(b)  == 1

    # spectrum sorting
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> supercell(10)
    b = bands(h, subdiv(0, 1, 7); solver = ES.Arpack(sigma = 0.1im, nev = 12), mapping = (:K, :K´), showprogress = false)
    @test nsubbands(b) == 1

    # defect insertion and patching
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bands(h, subdiv(0, 2pi, 8), subdiv(0, 2pi, 8); showprogress = false, defects = ((2pi/3, 4pi/3), (4pi/3, 2pi/3)), patches = 10)
    @test nvertices(b) == 140
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B) |> plusadjoint) |> supercell(3)
    hf((x,)) = Matrix(Quantica.call!(hc, (x, -x)))
    m = subdiv(0, 1, 4)
    b = bands(hf, m, showprogress = false, mapping = x -> 2π * x)
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 36

    hp2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1), @hopping!((t; s) -> s*t))
    hf2((s, x)) = Matrix(Quantica.call!(hp2, (x, x); s))
    m = subdiv(0, pi, 13)
    b = bands(hf2, m, m, showprogress = false)
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 288 * 2
    b = bands(hf2, m, m, showprogress = false, transform = ω -> ω^2)    # degeneracies doubled
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 288
end

@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> supercell(2) |>
         hamiltonian(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    mesh2D = subdiv(0, 1, 15), subdiv(0, 2π, 15)
    b = bands(ph, mesh2D..., mapping = (x, k) -> ftuple(x; k = k), showprogress = false)
    @test nsubbands(b)  == 4
    b = bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(1; k = k, p = SA[1, φ]), showprogress = false)
    @test nsubbands(b)  == 1

    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> supercell(2) |>
        supercell |> hamiltonian(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    b = bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(; k = k, p = SA[1, φ]), showprogress = false)
    @test nsubbands(b)  == 1
    # multithreading loop throws a CompositeException
    @test_throws CompositeException bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(; p = SA[1, φ]), showprogress = false)
end

@testset "spectrum" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> supercell(2) |> supercell
    sp = spectrum(h)
    e1, s1 = sp
    e2, s2 = first(sp), last(sp)
    e3, s3 = energies(sp), states(sp)
    @test e1 === e2 === e3
    @test s1 === s2 === s3
    @test Tuple(sp) === (e1, s1)
    for sp´ in (sp[[3,5]], sp[1:2], sp[[2,3], around = 0])
        @test length(sp´[1]) == 2
        @test size(sp´[2]) == (Quantica.flatsize(h), 2)
    end
    @test sp[around = 0] isa Tuple
    @test sp[2, around = 0] isa Tuple
    @test sp[[2,3,4], around = 0] == sp[2:4, around = 0]
    @test sp[[2,3,4], around = 0] !== sp[2:4, around = 0]
    @test sp[[2,4,3], around = -Inf][2] == hcat(sp[2][2], sp[4][2], sp[3][2])
end

@testset "bandstructures/spectrum slices" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, range = 1/√3)) |> supercell(4)
    b = bands(h, subdiv(0, 2pi, 13), subdiv(0, 2pi, 15), showprogress = false)
    @test b[(:, pi)] isa Vector{Quantica.Subband{Float64,2}}
    @test length(b[(pi,)]) == length(b[(pi, :)]) == length(b[(pi, :, :)]) == 1
    @test b[1] isa Subband
    m = Quantica.slice(b, (:,1))
    @test only(m) isa Quantica.Mesh

    s = spectrum(b, (pi, pi))
    s´ = spectrum(wrap(h, (pi, pi)))
    s´´ = spectrum(h, (pi, pi))
    ϵs, ψs = ES.LinearAlgebra()(Matrix(h((pi, pi))))
    @test s isa Quantica.Spectrum
    @test Tuple(s) == Tuple(s´) == Tuple(s´´) == (ϵs, ψs)
    @test s[1:10, around = 0] == s´[1:10, around = 0] == s´´[1:10, around = 0]
end

# @testset "unflatten" begin
#     h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> supercell(2) |> supercell
#     sp = spectrum(h).states[:,1]
#     sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
#     l = size(h, 1)
#     @test length(sp) == 1.5 * l
#     @test length(sp´) == l
#     @test all(x -> iszero(last(x)), sp´[l+1:end])
#     @test sp´ isa Vector
#     @test sp´ !== sp

#     h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> supercell(2)
#     b = bands(h)
#     psi = b[(0,0), around = 0]
#     psi´ = flatten(psi)
#     psi´´ = unflatten(psi´, orbitalstructure(psi))
#     @test eltype(psi.basis) == eltype(psi´´.basis) == SVector{2, ComplexF64}
#     @test eltype(psi´.basis) == ComplexF64
#     @test psi.basis == psi´´.basis

#     h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> supercell(2) |> supercell
#     sp = spectrum(h).states[:,1]
#     sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
#     l = size(h, 1)
#     @test length(sp) == 2 * l
#     @test length(sp´) == l
#     @test sp´ isa Base.ReinterpretArray

#     h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = Val(2)) |> supercell(2) |> supercell
#     sp = spectrum(h).states[:,1]
#     sp´ = Quantica.unflatten_orbitals_or_reinterpret(sp, orbitalstructure(h))
#     @test sp === sp
# end

# @testset "spectrum indexing" begin
#     h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> supercell(2) |> wrap
#     s = spectrum(h)
#     ϵv, ψm = s
#     @test ϵv == first(s) == s.energies
#     @test ψm == last(s) == s.states
#     @test s[1] isa Subspace
#     @test degeneracy(s[2]) == 3
#     @test s[1:3] isa Vector{<:Subspace}
#     @test s[[1,3]] isa Vector{<:Subspace}
#     ϵ, ψs = s[1]
#     @test ϵ isa Number
#     @test ψs isa SubArray{<:Complex, 2}
# end

# @testset "bands indexing" begin
#     h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> supercell(2)
#     bs = bands(h, subticks = 13)
#     @test sum(degeneracy, bs[(1,2)]) == size(h,1)
#     @test sum(degeneracy, bs[(0.2,0.3)]) == size(h,1)
#     @test size(bs[(1,2), around = 0].basis, 1) == size(h, 2)
#     h = LatticePresets.honeycomb() |> hamiltonian(hopping(1f0*I), orbitals = (Val(1),Val(2))) |> supercell(2)
#     bs = bands(h, subticks = 13)
#     @test sum(degeneracy, bs[(1,2)]) == size(h,1) * 1.5
#     @test sum(degeneracy, bs[(0.2,0.3)]) == size(h,1) * 1.5
#     @test bs[(1,2), around = 0] |> last |> eltype == SVector{2, ComplexF64}
#     @test size(bs[(1,2), around = 0].basis, 1) == size(h, 2)
# end

# @testset "subspace flatten" begin
#     h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(2))) |> supercell(2) |> supercell
#     sub = spectrum(h)[1]
#     @test size(sub.basis) == (8,1) && size(flatten(sub).basis) == (12,1)
# end

# @testset "diagonalizer" begin
#     h = LatticePresets.honeycomb() |> hamiltonian(hopping(1))
#     d = diagonalizer(h)
#     @test first(d((0, 0))) ≈ [-3,3]
#     h = wrap(h)
#     d = diagonalizer(h)
#     @test first(d()) ≈ [-3,3]
#     @test first(d(())) ≈ [-3,3]
#     @test all(d() .≈ Tuple(spectrum(h)))
# end

# @testset "bands extrema" begin
#     h = LP.honeycomb() |>
#         hamiltonian(hopping(I) + onsite(0.1, sublats = :A) - onsite(0.1, sublats = :B)) |>
#         supercell(3) |>
#         Quantica.wrap(1)
#     b = bands(h, subticks = 20)
#     @test !isapprox(gap(b, 0; refinesteps = 0), 0.2)
#     @test isapprox(gap(b, 0; refinesteps = 1), 0.2)
#     @test isapprox(gap(b, 0.3; refinesteps = 1), 0)
#     @test isapprox(gap(b, 4; refinesteps = 1), Inf)
#     @test all(gapedge(b, 0, +; refinesteps = 1) .≈ (0, 0.1))
#     @test all(gapedge(b, 0, -; refinesteps = 1) .≈ (0, -0.1))

#     b = bands(h, subticks = 20; method = ArpackPackage(nev = 6, sigma = 0.1*im))
#     @test !isapprox(gap(b, 0; refinesteps = 0), 0.2)
#     @test isapprox(gap(b, 0; refinesteps = 1), 0.2)
#     @test isapprox(gap(b, 0.3; refinesteps = 1), 0)
#     @test isapprox(gap(b, 4; refinesteps = 1), Inf)
#     @test all(gapedge(b, 0, +; refinesteps = 1) .≈ (0, 0.1))
#     @test all(gapedge(b, 0, -; refinesteps = 1) .≈ (0, -0.1))

#     h = LP.honeycomb() |> hamiltonian(hopping(I)) |> Quantica.wrap(1)
#     b = bands(h, subticks = 20)
#     length.(minima(b, refinesteps = 0)) == [2, 0]
#     length.(maxima(b, refinesteps = 0)) == [0, 2]
#     length.(minima(b, refinesteps = 1)) == [1, 0]
#     length.(maxima(b, refinesteps = 1)) == [0, 1]
# end