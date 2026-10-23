using LinearAlgebra

@testset "densitymatrix" begin
    g1 = LP.honeycomb() |> hamiltonian(hopping(0.1I, sublats = (:A,:B) .=> (:A,:B), range = 1) - plusadjoint(@hopping((; t=1) -> t*SA[0.4 1], sublats = :B => :A)), orbitals = (1,2)) |>
        supercell((1,-1), region = r -> -2<=r[2]<=2) |> attach(nothing, region = RP.circle(1), sublats = :B) |>
        attach(nothing, sublats = :A, cells = 0, region = r->r[2]<0) |> greenfunction(GS.Schur())
    g2 = LP.square() |> hopping(1) - onsite(1) |> supercell(3) |> supercell |> attach(nothing, region = RP.circle(2)) |> greenfunction(GS.Spectrum())
    g3 = LP.triangular() |> hamiltonian(hopping(-I) + onsite(1.8I), orbitals = 2) |> supercell(10) |> supercell |>
        attach(nothing, region = RP.circle(2)) |> attach(nothing, region = RP.circle(2, SA[1,2])) |> greenfunction(GS.KPM(bandrange=(-4,5)))
    # g3 excluded since KPM doesn't support finite temperatures or generic indexing yet
    # all inds × path combinations, with g cycled through (see testutils.jl)
    indss = (diagonal(1), diagonal(:), sitepairs(range = 1))
    paths = (5, Paths.radial(1,π/6), Paths.sawtooth(5))
    greens = (g1, g2)
    for (inds, path, g) in cyclic_product((indss, paths), (greens,))
        gs = g[inds]
        gs´ = Quantica.minimal_callsafe_copy(gs)  # exercise minimal_callsafe_copy for different solvers
        ρ0 = densitymatrix(gs, path)
        ρ = densitymatrix(gs´)
        @test isapprox(ρ0(), ρ(); atol = 1e-6)
        @test isapprox(ρ0(0.2), ρ(0.2); atol = 1e-6)
        @test isapprox(ρ0(0.2, 0.3), ρ(0.2, 0.3); atol = 1e-6)
    end
    # g2 excluded since it is single-orbital
    for g in (g1, g3)
        ρ0 = densitymatrix(g[diagonal(1, kernel = SA[0 1; 1 0])], 5)
        ρ = densitymatrix(g[diagonal(1, kernel = SA[0 1; 1 0])])
        @test isapprox(ρ0(), ρ(); atol = 1e-8)
        @test isapprox(ρ0(0.2), ρ(0.2); atol = 1e-8)
    end
    # 2D Schur solver: path integration is too slow for this solver, not tested here
    h = LP.square() |> supercell(1,3) |> hamiltonian(hopping(I), orbitals = 2) |> attach(nothing, region = iszero)
    g = h |> greenfunction(GS.Schur())
    for inds in (diagonal(1), diagonal(:), sitepairs(range = 1, includeonsite = true))
        ρ = densitymatrix(g[inds])
        @test Diagonal(ρ(0, 0)) ≈ 0.5*I     # half filling kBT == 0
        @test Diagonal(ρ(0, 1)) ≈ 0.5*I     # half filling kBT > 0
    end
    g´ = h |> greenfunction(GS.Bands(range(0, 2π, 100), range(0, 2π, 100)))
    # precision of Bands solver is low, but the results of the two solvers are within 2%
    @test maximum(abs, g[](0.2) - g´[](0.2)) < 0.005

    g = HP.graphene(orbitals = 2) |> supercell((1,-1)) |> greenfunction
    ρ0 = densitymatrix(g[sitepairs(range = 1)], 6)
    mat = ρ0(0.2, 0.3)
    @test Quantica.nnz(parent(mat)) < length(parent(mat))  # no broken sparsity
end
