using Quantica: GreenFunction, GreenSlice, GreenSolution, zerocell, CellOrbitals, ncontacts,
    solver

using ArnoldiMethod  # for KPM bandrange

function testgreen(h, s; kw...)
    ω = 0.2
    g = greenfunction(h, s)
    @test g isa GreenFunction
    gω = g(ω; kw...)
    @test gω isa GreenSolution
    @test g(0) isa GreenSolution # promote Int to AbstractFloat to add eps
    L = Quantica.latdim(lattice(h))
    z = zero(SVector{L,Int})
    o = Quantica.unitvector(1, SVector{L,Int})
    conts = ntuple(identity, ncontacts(h))
    locs = (sites(z, :), sites(z, 2), sites(1:2), sites(o, (1,2)), (; cells = z),
        CellOrbitals(o, 1), CellOrbitals(z, :), CellOrbitals(z, SA[1,2]), conts...)
    for loc in locs, loc´ in locs
        gs = g[loc, loc´]
        @test gs isa GreenSlice
        gsω = gs(ω; kw...)
        gωs = gω[loc, loc´]
        @test gsω == gωs
        loc === loc´ && @test all(x->imag(x)<=0, diag(gωs))
    end
    return nothing
end

@testset "basic greenfunctions" begin
    h0 = LP.honeycomb() |> hamiltonian(hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell(region = RP.circle(10))
    s0 = GS.SparseLU()
    s0´ = GS.Spectrum()
    h1 = LP.square() |> hamiltonian(@onsite((; o = 1) -> o*I) + hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> abs(r[2]) < 2)
    s1 = GS.Schur()
    s1´ = GS.Schur(boundary = -1)
    for (h, s) in zip((h0, h0, h1, h1), (s0, s0´, s1, s1´))
        testgreen(h, s; o = 2)
    end
    # This ensures that flat_sync! is called with multiorbitals when call!-ing ph upon calling g
    g = LP.square() |> hamiltonian(@onsite(()->I), orbitals = 2) |> supercell |> greenfunction
    @test g[](0.0 + 0im) ≈ SA[-1 0; 0 -1]
    g = LP.linear() |> hopping(0) + @onsite((;ω=0) -> ω)|> greenfunction;
    @test only(g[sites(SA[1],1)](1.0; ω = 0)) ≈ 1.0   atol = 0.000001
    @test only(g[sites(SA[1],1)](1.0; ω = -1)) ≈ 0.5  atol = 0.000001
    gs = g[sites(SA[1],1)]
    @test gs(0.2) !== gs(0.3)
    @test Quantica.call!(gs, 0.2) === Quantica.call!(gs, 0.2)
    gs = g[cells = SA[2]]
    @test Quantica.call!(gs, 0.2) === Quantica.call!(gs, 0.2)
end

@testset "greenfunction with contacts" begin
    g = LP.linear() |> hamiltonian(hopping(I), orbitals = 2) |> attach(@onsite(ω->im*I), cells = 1) |> attach(@onsite(ω->im*I), cells = 4) |> greenfunction
    @test size(g[(; cells = 2), (; cells = 3)](0.2)) == (2,2)

    h0 = LP.square() |> hamiltonian(hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell(region = RP.circle(10))
    s0 = GS.SparseLU()
    s0´ = GS.Spectrum()
    h1 = LP.square() |> hamiltonian(@onsite((; o = 1) -> o*I) + hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> abs(r[2]) < 2)
    s1 = GS.Schur()
    s1´ = GS.Schur(boundary = -1)
    sites´ = (; region = r -> abs(r[2]) < 2 && r[1] == 0)
    # non-hermitian Σ model
    mod = @onsite((ω, r; o = 1) -> (o - im*ω)*I) + @onsite((ω, s; o = 1, b = 2) --> o*b*pos(s)[1]*I) +
          plusadjoint(@onsite((ω; p=1)-> p*I) + @hopping((ω, r, dr; t = 1) -> im*dr[1]*t*I; range = 1))
    g0, g0´, g1´ = greenfunction(h0, s0), greenfunction(h0, s0´), greenfunction(h1, s1´)
    for (h, s) in zip((h0, h0, h1, h1), (s0, s0´, s1, s1´))
        oh = h |> attach(nothing; sites´...)
        testgreen(oh, s)
        oh = h |> attach(mod; sites´...)
        testgreen(oh, s)
        for glead in (g0, g0´, g1´)
            oh = h |> attach(glead[sites´], hopping(I; range = (1,2)); sites´...)
            testgreen(oh, s)
            L´ = Quantica.latdim(lattice(parent(glead)))
            iszero(L´) && continue
            oh = h |> attach(glead; sites´...)
            testgreen(oh, s)
            oh = h |> attach(glead, hopping(I; range = (1,2)); sites´...)
            testgreen(oh, s)
        end
    end

    # decay_lengths
    glead = LP.honeycomb() |> hopping(1) + @onsite((; w=0) -> w) |> supercell((1, -1)) |> greenfunction(GS.Schur(boundary = 0));
    @test only(Quantica.decay_lengths(glead, 0.2)) < 0.4

    # SparseLU and Spectrum are equivalent
    @test g0[(; region = RP.circle(2)), (; region = RP.circle(3))](0.2) ≈
          g0´[(; region = RP.circle(2)), (; region = RP.circle(3))](0.2)

    # contacts that don't include all sublattices
    h = lattice(sublat(0, name = :L), sublat(1, name = :R)) |> hamiltonian
    @test h |> attach(onsite(ω->1), sublats = :L) |> greenfunction isa GreenFunction

    # multiple contacts
    h0 = LP.square(type = Float32) |> hopping(1)
    hc = h0 |> supercell(region = RP.rectangle((2, 2)))
    glead = h0 |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> attach(nothing; cells = SA[1]) |> greenfunction(GS.Schur(boundary = 0));
    g = hc |> attach(glead, region = r -> r[1] == 1) |> attach(glead, region = r -> r[1] == -1, reverse = true)  |> attach(onsite(ω->1), region = r -> r == SA[0,0]) |> greenfunction
    @test Quantica.ncontacts(g) == 3

    # 1D leads with contacts
    glead = LP.honeycomb() |> hopping(1) |> supercell((1,-1), region = r -> 0<=r[2]<=5) |> attach(nothing, cells = SA[5]) |> greenfunction(GS.Schur(boundary = 0));
    g = LP.honeycomb() |> hopping(1) |> supercell(region = r -> -6<=r[1]<=6 && 0<=r[2]<=5) |> attach(glead, region = r -> r[1] > 5.1) |> greenfunction
    @test g isa GreenFunction

    # attach(gslice, model; transform)
    Rot = r -> SA[0 -1; 1 0] * r
    glead = LP.square(a0 = 1, dim = 2) |> onsite(4) - hopping(1) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    central = LP.honeycomb() |> onsite(4) - hopping(1) |> supercell(region = RP.rectangle((3,3))) |> transform(Rot)
    g = central |> attach(glead[cells = 1], -hopping(1), region = r -> r[1] > 1.3 && -1.1 <= r[2] <= 1.1, transform = r -> r + SA[1.2,0]) |> greenfunction
    @test g isa GreenFunction
    @test_throws ArgumentError central |> attach(glead[cells = 1], -hopping(1), region = r -> r[1] > 2.3 && -1.1 <= r[2] <= 1.1, transform = r -> r + SA[1.2,0])

    # cheap views
    g = LP.linear() |> hopping(1) |> attach(@onsite((ω; p = 1) -> p), cells = 1) |> attach(@onsite((ω; p = 1) -> p), cells = 3) |> greenfunction
    gs = g(0.2)
    @test view(gs, 1, 2) isa SubArray
    @test (@allocations view(gs, 1, 2)) <= 2  # should be 1, but in some platforms/versions it could be 2
end

@testset "GreenFunction partial evaluation" begin
    g = LP.linear() |> hamiltonian(@hopping((; q = 1) -> q*I), orbitals = 2) |> attach(@onsite((ω; p = 0) ->p*SA[0 1; 1 0]), cells = 1) |> greenfunction
    g´ = g(p = 2)
    h´, h = hamiltonian(g´), hamiltonian(g)
    @test h´ !== h
    @test h´ == h(p = 2)
    @test g(0.2, p = 2)[] == g´(0.2)[]
    @test g(0.2, p = 1)[] != g´(0.2, p = 1)[]       # p = 1 is ignored by g´, fixed to p = 2
    @test solver(g) !== solver(parent(solver(g´)))  # no aliasing
    gs = g[]
    gs´ = gs(p = 2)
    @test gs´ isa GreenSlice
    @test gs´(0.3) == gs(0.3, p = 2)
    @test gs´(0.3, p = 1) != gs(0.3, p = 1)
end

@testset "GreenSolvers applicability" begin
    h = HP.graphene()
    @test_throws ArgumentError greenfunction(h, GS.KPM())
    @test_throws ArgumentError greenfunction(h, GS.SparseLU())
    @test_throws ArgumentError greenfunction(h, GS.Spectrum())
    h = supercell(h)
    @test_throws ArgumentError greenfunction(h, GS.Schur())
    @test_throws ArgumentError greenfunction(h, GS.Bands())
    h = LP.honeycomb() |> @onsite((; o = 1) -> o*I) |> supercell
    @test_throws ArgumentError greenfunction(h, GS.Spectrum())
    @test greenfunction(h(), GS.Spectrum()) isa GreenFunction
end

@testset "greenfunction KPM" begin
    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell(region = RP.circle(20)) |>
        attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 300, bandrange = (-3.1, 3.1)))
    ρs = ldos(g[1], kernel = missing)
    for ω in -3:0.1:3
        @test all(>=(0), ρs(ω))
    end

    ω = -0.1
    gωs = g[1](ω)
    ρflat = -imag.(diag(gωs))/pi
    @test all(>(0), ρflat)
    ρs = ldos(g[1], kernel = missing)
    ρ = ρs(ω)
    @test sum(ρ) ≈ sum(ρflat)
    @test (length(ρflat), length(ρ)) == (9, 9)

    ω = -0.1
    gωs = g[1](ω)
    ρflat = -imag.(diag(gωs))/pi
    @test all(>(0), ρflat)
    ρs = ldos(g[1], kernel = I)
    ρ = ρs(ω)
    @test ρ isa OrbitalSliceVector
    @test sum(ρ) ≈ sum(ρflat)
    @test (length(ρflat), length(ρ)) == (9, 6)

    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell(region = RP.circle(20)) |>
        attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 500))
    gωs = g[1](ω)
    ρflat´ = -imag.(diag(gωs))/pi
    ρ´ = ldos(g[1], kernel = I)(ω)
    @test all(<(0.01), abs.(ρ .- ρ´))
    @test all(<(0.01), abs.(ρflat .- ρflat´))
    @test_throws ArgumentError g[sites((), 3:4)](ω)
    @test g[:](ω) == g[1](ω) == g(ω)[1] == g(ω)[:]

    g´ = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell(region = RP.circle(20)) |>
        attach(g[1], hopping((r, dr) -> I, range = 1), region = RP.circle(1)) |> attach(nothing, region = RP.circle(1, (2,2))) |>
        greenfunction(GS.KPM(order = 500))
    ρs = ldos(g´[1])
    for ω in -3:0.1:3
        @test all(>=(0), ρs(ω))
    end

    h = HP.graphene(a0 = 1) |> supercell(region = RP.circle(20))
    hmat = h[()]
    g = h |> attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 500, kernel = hmat))
    ρs = ldos(g[1])
    for ω in subdiv(-3, 3, 20)
        @test all(x -> sign(x) == sign(ω), ρs(ω))
    end
end

@testset "greenfunction bands" begin
    h = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1))
    ga = h |> attach(nothing, cells = 1) |> greenfunction()
    gb = h |> attach(nothing, cells = 1) |> greenfunction(GS.Bands(boundary = 2 => -3))
    for g in (ga, gb)
        @test Quantica.solver(g) isa Quantica.AppliedBandsGreenSolver
        @test bands(g) isa Quantica.Subband
        g1 = g[1](0.2)
        @test size(g1) == (3,3)
        @test abs(g1[2,2] - 1/0.2) < 0.02
        @test all(x -> Quantica.chopsmall(imag(x)) ≈ 0, g1[2,:])
        @test all(x -> Quantica.chopsmall(imag(x)) ≈ 0, g1[:,2])
        g2 = g[sites((0,0),:), sites((1,1),2)](0.2)
        @test size(g2) == (3,1)
    end

    ha = LP.honeycomb() |> hopping(1) |> supercell((1,-1), region = r -> abs(r[2])<2)
    hb = LP.honeycomb() |> hopping(1, range = 1) |> supercell((1,-1), region = r -> abs(r[2])<2)
    hc = LP.honeycomb() |> hamiltonian(hopping(I, range = 1), orbitals = (2,1)) |> supercell((3,-3), region = r -> abs(r[2])<2)
    hd = LP.square() |> hopping(1) |> supercell((1,0), region = r -> abs(r[2])<2)
    for h in (ha, hb, hc, hd)
        gc = h |> attach(nothing, cells = 3) |> greenfunction(GS.Bands(subdiv(-π, π, 89)))
        gd = h |> attach(nothing, cells = 3) |> greenfunction(GS.Schur())
        @test maximum(abs.(gc[cells = 1](0.5) - gd[cells = 1](0.5))) < 0.08
        @test all(>=(0), ldos(gc[1])(0.2))
        @test all(>=(0), ldos(gc[region = RP.circle(2)])(0.2))

        gc = h |> attach(nothing, cells = 3) |> greenfunction(GS.Bands(subdiv(-π, π, 89), boundary = 1 => 0))
        gd = h |> attach(nothing, cells = 3) |> greenfunction(GS.Schur(boundary = 0))
        @test maximum(abs.(gc[cells = 1](0.5) - gd[cells = 1](0.5))) < 0.08
        @test all(>=(0), ldos(gc[1])(0.2))
        @test all(>=(0), ldos(gc[region = RP.circle(2)])(0.2))
    end

    # Issue #252
    g = LP.honeycomb() |> hopping(1, range = 3) |> supercell((1,-1), (1,1)) |>
        attach(nothing, region = RP.circle(1, SA[2,3])) |> attach(nothing, region = RP.circle(1, SA[3,-3])) |>
        greenfunction(GS.Bands(subdiv(-π, π, 13), subdiv(-π, π, 13), boundary = 2=>-3))
    @test g isa GreenFunction
    @test iszero(g[cells = SA[1,-3]](0.2))

    # Issue #257
    h = LP.square() |> hopping(-1)
    ϕs = subdiv(0, 0.6π, 2)
    b = bands(h, ϕs, ϕs, showprogress = false)
    g = h |> attach(@onsite(ω->-im), cells = SA[20,0]) |> greenfunction(GS.Bands(ϕs, ϕs))
    @test g(-1)[sites(SA[1,-1], 1), sites(1)] isa AbstractMatrix

    # exercise band digagonal fastpath
    g = h |> greenfunction(GS.Bands(ϕs, ϕs))
    @test g(-0.2)[diagonal(sites(SA[1,2], :))] isa AbstractMatrix
end

@testset "greenfunction 32bit" begin
    # GS.Bands
    # skip for older versions due to https://github.com/JuliaLang/julia/issues/53054
    # which makes solve not deterministic when adding sufficiently many band simplices
    if VERSION >= v"1.11.0-DEV.632"
        h = HP.graphene(type = Float32)
        s = GS.Bands()
        testgreen(h, s)
    end
    h = HP.graphene(type = Float32, a0 = 1) |> supercell(10) |> supercell
    s = GS.SparseLU()
    testgreen(h, s)

    # GS.Schur
    h = HP.graphene(type = Float32, a0 = 1) |> supercell((1,-1), region = r -> 0<=r[2]<=3)
    s = GS.Schur()
    testgreen(h, s)
    s = GS.Schur(boundary = -1)
    testgreen(h, s)

    # GS.KPM
    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1), type = Float32) |>
        supercell(region = RP.circle(20)) |>
        attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 300, bandrange = (-3.1, 3.1)))
    ρs = ldos(g[1], kernel = missing)
    for ω in -3:0.1:3
        @test all(>=(0), ρs(ω))
    end
end

@testset "greenfunction sparse slicing" begin
    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell((2,-2), region = r -> 0<=r[2]<=5) |>
        attach(nothing, cells = 2, region = r -> 0<=r[2]<=2) |> attach(nothing, cells = 3) |>
        greenfunction(GS.Schur(boundary = -2))
    ω = 0.6
    @test g[diagonal(2)](ω) isa OrbitalSliceMatrix
    @test g[diagonal(2)](ω) == g(ω)[diagonal(2)]
    @test g[diagonal(:)](ω) == g(ω)[diagonal(:)]
    @test size(g[diagonal(1)](ω)) == (12,12)
    @test size(g[diagonal(1, kernel = I)](ω)) == (8,8)
    @test size(g[diagonal(:)](ω),1) == size(g[diagonal(1)](ω),1) + size(g[diagonal(2)](ω),1) == 48
    @test size(g[diagonal(:, kernel = I)](ω),1) == size(g[diagonal(1, kernel = I)](ω),1) + size(g[diagonal(2, kernel = I)](ω),1) == 32
    @test size(g[diagonal(cells = 2:3)](ω)) == (72, 72)
    @test size(g[diagonal(cells = 2:3, kernel = I)](ω)) == (48, 48)
    @test ldos(g[1], kernel = I)(ω) ≈ -imag.(diag(g[diagonal(1; kernel = I)](ω))) ./ π
    gs = g[diagonal(cells = 2:3)]
    @test Quantica.call!(gs, ω) === Quantica.call!(gs, 2ω)
    inds = (1, :, (; cells = 1, sublats = :B), sites(2, 1:3), sites(0, 2), sites(3, :))
    for i in inds
        @test g(0.2)[diagonal(i)] isa Union{OrbitalSliceMatrix, AbstractMatrix}
    end
    gg = g(ω)[sitepairs(range = 1/sqrt(3))]
    @test gg isa OrbitalSliceMatrix{ComplexF64,Quantica.SparseMatrixCSC{ComplexF64, Int64}}
    @test size(gg) == Quantica.flatsize(g) .* (3, 1)
    @test_throws DimensionMismatch g(ω)[sitepairs(range = 1, sublats = :B => :B, kernel = SA[1 0; 0 -1])]
    gg = g(ω)[sitepairs(range = 1, sublats = :A => :A, kernel = I)]
    @test size(gg) == size(g, 1) .* (3, 1)

    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell((1, -1)) |>
           greenfunction(GS.Schur(boundary = -2))
    ω = 0.6
    gg = g(ω)[sitepairs(range = 1)]
    gg´ = g(ω)[orbaxes(gg)...]
    @test gg .* gg´ ≈ gg .* gg
end

@testset "densitymatrix sparse slicing" begin
    g1 = LP.honeycomb() |> hamiltonian(hopping(0.1I, sublats = (:A,:B) .=> (:A,:B), range = 1) - plusadjoint(@hopping((; t=1) -> t*SA[0.4 1], sublats = :B => :A)), orbitals = (1,2)) |>
        supercell((1,-1), region = r -> -2<=r[2]<=2) |> attach(nothing, region = RP.circle(1), sublats = :B) |>
        attach(nothing, sublats = :A, cells = 0, region = r->r[2]<0) |> greenfunction(GS.Schur())
    g2 = LP.square() |> hopping(1) - onsite(1) |> supercell(3) |> supercell |> attach(nothing, region = RP.circle(2)) |> greenfunction(GS.Spectrum())
    g3 = LP.triangular() |> hamiltonian(hopping(-I) + onsite(1.8I), orbitals = 2) |> supercell(10) |> supercell |>
        attach(nothing, region = RP.circle(2)) |> attach(nothing, region = RP.circle(2, SA[1,2])) |> greenfunction(GS.KPM(bandrange=(-4,5)))
    # g3 excluded since KPM doesn't support finite temperatures or generic indexing yet
    for g in (g1, g2), inds in (diagonal(1), diagonal(:), sitepairs(range = 1)), path in (5, Paths.radial(1,π/6), Paths.sawtooth(5))
        ρ0 = densitymatrix(g[inds], path)
        ρ = densitymatrix(g[inds])
        @test isapprox(ρ0(), ρ(); atol = 1e-7)
        @test isapprox(ρ0(0.2), ρ(0.2); atol = 1e-7)
        @test isapprox(ρ0(0.2, 0.3), ρ(0.2, 0.3))
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

@testset "OrbitalSliceArray slicing" begin
    g = LP.linear() |> hopping(1) + onsite(1) |> supercell(4) |> greenfunction
    gmat = g[cells = SA[2]](0.2)
    @test gmat isa Quantica.OrbitalSliceMatrix
    @test size(gmat) == (4, 4)
    gmat´ = gmat[cells = SA[1]]
    @test gmat´ isa Quantica.OrbitalSliceMatrix
    @test isempty(gmat´)
    gmat = g[(; cells = SA[2]), sites(SA[1], 1:2)](0.2)
    @test gmat isa Matrix
    @test size(gmat) == (4, 2)
    gmat = g[(; cells = SA[1]), (; region = r -> 3<=r[1]<=5)](0.2)
    @test gmat isa Quantica.OrbitalSliceMatrix
    @test size(gmat) == (4, 3)
    gmat´ = gmat[(; cells = SA[1]), (; cells = SA[0])]
    @test gmat´ isa Quantica.OrbitalSliceMatrix
    @test size(gmat´) == (4, 1)
    gmat´ = gmat[(; cells = SA[1])]
    @test gmat´ isa Quantica.OrbitalSliceMatrix
    @test size(gmat´) == (4, 2)
    @test_throws Quantica.Dictionaries.IndexError gmat[sites(SA[1],:)]  # `:` means all sites in cell
    gmat´ = gmat[sites(SA[1], 1:2)]
    @test gmat´ isa Matrix
    @test size(gmat´) == (2, 2)
    c = sites(SA[1], 1)
    view(gmat, c)
    @test (@allocations view(gmat, c)) <= 2
    i, j = orbaxes(gmat)
    @test g(0.2)[i, j] isa Quantica.OrbitalSliceMatrix
    @test gmat[c] isa Matrix
end

function testcond(g0; nambu = false)
    G1 = conductance(g0[1]; nambu)
    G2 = conductance(g0[2]; nambu)
    G12 = conductance(g0[1,2]; nambu)
    T12 = transmission(g0[1,2])
    @test_throws ArgumentError transmission(g0[1])
    @test_throws ArgumentError transmission(g0[1, 1])
    for ω in -3:0.1:3
        ωc = ω + im*1e-10
        @test ifelse(nambu, 6.000001, 3.00000001) >= G1(ωc) >= 0
        @test G1(ωc) ≈ G1(-ωc) ≈ G2(ωc) ≈ G2(-ωc)
        @test T12(ωc) ≈ T12(-ωc)
        nambu || @test G1(ωc) ≈ T12(ωc) atol = 0.000001
        @test G12(ωc) ≈ G12(-ωc) atol = 0.000001
        nambu || @test G1(ωc) ≈ -G12(ωc) atol = 0.000001
    end
end

function testjosephson(g0)
    J1 = josephson(g0[1], 1; phases = subdiv(0, pi, 10))
    J2 = josephson(g0[2], 1; phases = subdiv(0, pi, 10))
    J3 = josephson(g0[1], (-5, 0); phases = subdiv(0, pi, 10))
    J4 = josephson(g0[1], Paths.sawtooth(range(-5, 0, 3)); phases = subdiv(0, pi, 10))
    p5(µ, T) = ifelse(iszero(T),(-5,-2+3im,0),(-5,-2+3im,0,2+3im,5)) .+ sqrt(eps(Float64))*im
    J5 = josephson(g0[1], Paths.polygon(p5); phases = subdiv(0, pi, 10))
    j1 = J1()
    @test all(>=(0), Quantica.chopsmall.(j1))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J2()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J3()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J4()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J5()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(J1(0.1), J5(0.1)))

    j = Quantica.integrand(J1)
    @test Quantica.call!(j, 0.2) isa Vector
    # static length of integration points (required for QuadGK.quadgk type-stability)
    @test typeof.(Quantica.points.((J1, J2, J3, J4, J5))) ==
        (Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}, Tuple{ComplexF64, ComplexF64, ComplexF64})
    # integration path logic
    J = josephson(g0[1], Paths.sawtooth(-4, -1))
    @test J() <= eps(Float64)
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3    # tuple sawtooth
    @test p2 isa Vector{ComplexF64} && length(p2) == 3
    J = josephson(g0[1], Paths.sawtooth(-4, 0); phases = subdiv(0, pi, 10))
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3    # tuple sawtooth
    @test p2 isa Vector{ComplexF64} && length(p2) == 3
    @test all(p2 .≈ p1)
    J = josephson(g0[1], Paths.sawtooth(-4, 1); phases = subdiv(0, pi, 10))
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3 && maximum(real(p1)) == 0
    @test p2 isa Vector{ComplexF64} && length(p2) == 5 && maximum(real(p2)) == 1
end

@testset "greenfunction observables" begin
    # single-orbital vs two-orbital
    g1 = LP.square() |> supercell((1,0), region = r->-2<r[2]<2) |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 1) |> greenfunction(GS.Schur(boundary = 0));
    g2 = LP.square() |> supercell((1,0), region = r->-2<r[2]<2) |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 2) |> greenfunction(GS.Schur(boundary = 0));
    J1 = current(g1[cells = SA[1]])
    J2 = current(g2[cells = SA[1]])
    @test size(J1(0.2)) == size(J2(0.2)) == (3, 3)
    @test 2*J1(0.2; B = 0.1) ≈ J2(0.2; B = 0.1)

    ρ = densitymatrix(g1[cells = SA[1]], Paths.sawtooth(5))
    @test length(Quantica.points(ρ)) == 3
    @test Quantica.points(ρ) isa Vector{ComplexF64}
    @test all(≈(0.5), diag(ρ(0, 0; B=0.3))) # half filling
    ρ = densitymatrix(g1[cells = SA[1]], 7)
    @test all(<(0.96), real(diag(ρ(4, 1; B=0.1)))) # thermal depletion
    @test diag(ρ(0)) ≈ SA[0.5, 0.5, 0.5]
    @test diag(ρ(4)) ≈ SA[1, 1, 1]
    h = LP.honeycomb() |> hopping(1) |> supercell(region = RP.circle(10))
    reg = (; region = RP.circle(0.5))
    gLU = h |> greenfunction(GS.SparseLU());
    gSpectrum = h |> greenfunction(GS.Spectrum());
    gKPM = h |> attach(nothing; reg...) |> greenfunction(GS.KPM(order = 100000, bandrange = (-3,3)));
    ρ1, ρ2, ρ3 = densitymatrix(gLU[reg], (-3,3)), densitymatrix(gSpectrum[reg]), densitymatrix(gKPM[1])
    @test ρ1() ≈ ρ2() atol = 0.00001
    @test ρ2() ≈ ρ3() atol = 0.00001
    gLU´ = h |> attach(nothing; reg...) |> greenfunction(GS.SparseLU());
    ρ1´ = densitymatrix(gLU´[1], Paths.sawtooth(-3, 3))
    @test ρ1() ≈ ρ1´()
    gSpectrum´ = h |> attach(nothing; reg...) |> greenfunction(GS.Spectrum());
    ρ2´ = densitymatrix(gSpectrum´[1])
    @test ρ2() ≈ ρ2´()

    # parameter-dependent paths
    g = LP.linear() |> supercell(3) |> hamiltonian(@onsite((; Δ = 0.1) -> SA[0 Δ; Δ 0]) + hopping(SA[1 0; 0 -1]), orbitals = 2) |> greenfunction
    ρ = densitymatrix(g[sites(1:2), sites(1:2)], Paths.polygon((µ,kBT; Δ) -> (-5, -Δ+im, 0, Δ+im, 5)))
    @test tr(ρ(; Δ = 0.2)) ≈ 2
    @test all(Quantica.points(ρ; Δ = 0.3) .== (-5, -0.3 + 1.0im, 0, 0.3 + 1.0im, 5))

    ρ = densitymatrix(g2[(; cells = SA[0]), (; cells = SA[1])], 5)
    ρ0 = ρ(0, 0; B=0.3)
    @test ρ0 isa OrbitalSliceMatrix
    @test iszero(ρ0)        # rows are on boundary
    @test ρ0[sites(1), sites(SA[1], 1)] isa Matrix
    @test size(view(ρ0, sites(1), sites(SA[1], 1))) == (2, 2)

    glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((0,1), region = r -> -1 <= r[1] <= 1) |> attach(nothing; cells = SA[10]) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[2] ≈ 5 && -1 <= r[1] <= 1
    g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = RP.square(10)) |> attach(glead, reverse = true; region = contact2) |> attach(glead; transform = r->SA[0 1; 1 0] * r, region = contact1) |> greenfunction;
    testcond(g0)

    glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[1] ≈ -5 && -1 <= r[2] <= 1
    g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = RP.square(10)) |> attach(glead, reverse = true; region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    testcond(g0)

    glead = LP.square() |> hamiltonian(hopping(I) + onsite(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[1] ≈ -5 && -1 <= r[2] <= 1
    g0 = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(region = RP.square(10)) |> attach(glead, reverse = true; region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    testcond(g0; nambu = true)
    testjosephson(g0)

    # test omegamap gets passed to integrand
    glead = LP.square() |> hamiltonian(@onsite((;ω=0) -> SA[im*ω 1; 1 im*ω]) + hopping(I), orbitals = 2) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    g0 = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(region = RP.square(10)) |> attach(glead, reverse = true; region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    J1 = josephson(g0[1], 4; phases = subdiv(0, pi, 10), omegamap = ω ->(; ω))
    J2 = josephson(g0[1], 4; phases = subdiv(0, pi, 10))
    @test Quantica.integrand(J1)(-2) != Quantica.integrand(J2)(-2)

    # test fermi at zero temperature
    g = LP.linear() |> hopping(1) |> supercell(3) |> supercell |> greenfunction(GS.Spectrum())
    @test ρ = diag(densitymatrix(g[])()) ≈ [0.5, 0.5, 0.5]

    # test DensityMatrixSchurSolver
    g = LP.honeycomb() |> hamiltonian(hopping(I) + @onsite((; w=0) -> SA[w 1; 1 -w], sublats = :A), orbitals = (2,1)) |> supercell((1,-1), region = r->-2<r[2]<2) |> greenfunction(GS.Schur());
    ρ0 = densitymatrix(g[cells = (SA[2],SA[4]), sublats = :A], Paths.sawtooth(-4, 4))
    ρ = densitymatrix(g[cells = (SA[2],SA[4]), sublats = :A])
    ρ0sol = ρ(0.2, 0.3)
    ρsol = ρ(0.2, 0.3)
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ0sol = ρ()
    ρsol = ρ()
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ = densitymatrix(g[sites(:)])
    ρ0 = densitymatrix(g[sites(:)])
    ρ0sol = ρ(0.2, 0.3)
    ρsol = ρ(0.2, 0.3)
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ0sol = ρ()
    ρsol = ρ()
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)

    # off-diagonal rho
    is = (; region = r -> 0 <= r[1] <= 4)
    js = (; region = r -> 2 <= r[1] <= 6)
    ρ0 = densitymatrix(g[is, js])
    ρ1 = densitymatrix(g[is, js], 4)
    ρ2 = densitymatrix(g[is, js], Paths.sawtooth(4))
    @test ρ0() ≈ ρ1() ≈ ρ2()
    @test ρ0(0.1, 0.2) ≈ ρ1(0.1, 0.2) ≈ ρ2(0.1, 0.2)

    is = (; region = r -> 0 <= r[1] <= 4)
    js = sites(SA[1], 1:2)
    ρ0 = densitymatrix(g[is, js])
    ρ1 = densitymatrix(g[is, js], 4)
    ρ2 = densitymatrix(g[is, js], Paths.sawtooth(4))
    @test ρ0() ≈ ρ1() ≈ ρ2()
    @test ρ0(0.1, 0.2) ≈ ρ1(0.1, 0.2) ≈ ρ2(0.1, 0.2)

    # parametric path and system
    g = LP.linear() |> supercell |> @onsite((; o = 0.5) -> o) |> greenfunction
    ρ = densitymatrix(g[], Paths.polygon((µ, T; o = 0.5) -> o > µ ? (-1, µ, o + im, 1) : (-1, o + im, µ, 1)))
    @test real(only(ρ(0, 0; o = -0.5))) ≈ 1
    @test real(only(ρ(0, 0; o = 0.5))) < 1e-7
    @test real(only(ρ(0, 0; o = 1.5))) < 1e-7
    @test real(only(ρ(0, 0; o = -1.5))) < 1e-7
    d = Quantica.integrand(ρ)
    @test_throws MethodError d(2; o = 0.8)
end

@testset "greenfunction aliasing" begin
    # Issue #267
    g = LP.linear() |> hamiltonian(@hopping((; q = 1) -> q*I), orbitals = 2) |> greenfunction
    g´ = Quantica.minimal_callsafe_copy(g)
    @test g(0.2; q = 2)[cells = 1] == g´(0.2; q = 2)[cells = 1]
    @test g´.solver.fsolver.h0 === g´.parent.h.harmonics[1].h
    # The Schur slicer has uninitialized fields whose eventual value depends on the parent hamiltonian
    # at call time, not creation time. This can produce bugs if Quantica.call!(g, ω; params...) is used.
    # The exported g(ω; params...) syntax decouples the slicer's parent Hamiltonian, so it is safe.
    g = LP.linear() |> hamiltonian(@onsite((; q = 1) -> q) + @hopping((; q = 1) -> q)) |> greenfunction
    m = g(0.4, q = 0.1)[cells = 0]
    gs = Quantica.call!(g, 0.4, q = 0.1)
    @test gs[cells = 0] == m
    gs = Quantica.call!(g, 0.4, q = 0.1)
    parent(g)(q = 2)
    @test_broken gs[cells = 0] == m
    # Ensure that g.solver.invgreen alias of parent contacts is not broken by minimal_callsafe_copy
    glead = LP.linear() |> @onsite((; o = 1) -> o) - hopping(1) |> greenfunction(GS.Schur(boundary = 0));
    g = LP.linear() |> -hopping(1) |> supercell |> attach(glead) |> greenfunction;
    g´ = Quantica.minimal_callsafe_copy(g);
    @test g´(0, o = 0)[] == g(0, o = 0)[]
    @test g´(0, o = 1)[] == g(0, o = 1)[]
    # two leads from the same g
    glead = LP.honeycomb() |> onsite(4) - hopping(1) |> supercell(4,2) |> supercell((0,-1))|> greenfunction(GS.Schur(boundary = 0));
    g = LP.honeycomb() |> onsite(4) - hopping(1) |> supercell(4,5) |> supercell |>
       attach(glead, region = r -> SA[-√3/2,1/2]' * r > 3.5, reverse = true) |>
       attach(glead, region = r -> SA[-√3/2,1/2]' * r < 0, reverse = false) |> greenfunction;
    @test g.contacts.selfenergies[2].solver.hlead[(0,)] === g.contacts.selfenergies[1].solver.hlead[(0,)]
    @test g.contacts.selfenergies[2].solver.hlead[(1,)] === g.contacts.selfenergies[1].solver.hlead[(-1,)]
    @test g.contacts.selfenergies[2].solver.hlead[(-1,)] === g.contacts.selfenergies[1].solver.hlead[(1,)]

    # ensure full dealiasing of lattices in attach
    model = hopping(SA[1 0; 0 -1]) + @onsite((; µ = 0) -> SA[-µ 0; 0 µ])
    h = LP.linear() |> hamiltonian(model, orbitals = 2)
    glead = h |> greenfunction(GS.Schur(boundary = 0))
    g = h |> attach(glead, cells = 1) |> greenfunction(GS.Schur(boundary = 0));
    @test sites(lattice(h)) == [SA[0.0]]
end

@testset "meanfield" begin
    oh = LP.honeycomb() |> hamiltonian(hopping(SA[1 im; -im -1]) - onsite(SA[1 0; 0 -1], sublats = :A), orbitals = 2) |> supercell((1,-1))
    g = oh |> greenfunction
    Q = SA[0 1; im 0]  # nonhermitian charge not allowed
    @test_throws ArgumentError meanfield(g; selector = (; range = 1), potential = 2, fock = 1.5, charge = Q)
    Q = SA[0 -im; im 0]
    m = meanfield(g; selector = (; range = 1), potential = 2, fock = 1.5, charge = Q)
    Φ = m(0.2, 0.3)
    ρ12 = m.rho(0.2, 0.3)[sites(1), sites(2)]
    @test Φ[sites(1), sites(2)] ≈ -1.5 * Q * ρ12 * Q

    # spinless nambu
    oh = LP.linear() |> hamiltonian(hopping((r, dr) -> SA[1 sign(dr[1]); -sign(dr[1]) -1]) - onsite(SA[1 0; 0 -1]), orbitals = 2)
    g = oh |> greenfunction
    Q = SA[1 0; 0 -1]
    m = meanfield(g; selector = (; range = 1), nambu = true, hartree = r -> 1/(1+norm(r)), fock = 1.5, charge = Q)
    @test_throws ArgumentError m(0.2, 0.3)  # µ cannot be nonzero
    Φ = m(0, 0.3)
    ρ11 = m.rho(0, 0.3)[sites(1), sites(1)] - SA[0 0; 0 1]
    fock = -1.5 * Q * ρ11 * Q
    hartree = 0.5*Q*(tr(Q*ρ11)*(1+1/2+1/2))
    @test Φ[sites(1), sites(1)] ≈ hartree + fock

    # spinful nambu - unrotated
    σzτz = SA[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    σyτy = SA[0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0]
    oh = LP.linear() |> hamiltonian(hopping(σzτz) - onsite(0.1 * σyτy), orbitals = 4)
    g = oh |> greenfunction
    Q = σzτz
    m = meanfield(g; selector = (; range = 1), nambu = true, namburotation = false, hartree = r -> 1/(1+norm(r)), fock = 1.5, charge = Q)
    Φ = m(0, 0.3)
    ρ11 = m.rho(0, 0.3)[sites(1), sites(1)] - SA[0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    fock = -1.5 * Q * ρ11 * Q
    hartree = 0.5*Q*(tr(Q*ρ11)*(1+1/2+1/2))
    @test Φ[sites(1), sites(1)] ≈ hartree + fock

    # spinful nambu - rotated
    σzτz = SA[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    σ0τx = SA[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
    oh = LP.linear() |> hamiltonian(hopping(σzτz) - onsite(0.1 * σ0τx), orbitals = 4)
    g = oh |> greenfunction
    Q = σzτz
    m = meanfield(g; selector = (; range = 1), nambu = true, namburotation = true, hartree = r -> 1/(1+norm(r)), fock = 1.5, charge = Q)
    Φ = m(0, 0.3)
    ρ11 = m.rho(0, 0.3)[sites(1), sites(1)] - SA[0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    fock = -1.5 * Q * ρ11 * Q
    hartree = 0.5*Q*(tr(Q*ρ11)*(1+1/2+1/2))
    @test Φ[sites(1), sites(1)] ≈ hartree + fock

    g = LP.linear() |> hopping(1) |> greenfunction
    pot(r) = 1/norm(r)
    Φ = meanfield(g; selector = (; range = 10), potential = pot, onsite = 3)()
    # onsite fock cancels hartree in the spinless case
    @test only(view(Φ, sites(1))) ≈ 0.5 * (0*3 + 2*sum(pot, 1:10))
    # unless we disable fock
    Φ = meanfield(g; selector = (; range = 10), potential = pot, onsite = 3, fock = nothing)()
    @test only(view(Φ, sites(1))) ≈ 0.5 * (3 + 2*sum(pot, 1:10))

    g = oh |> greenfunction
    @test_throws ArgumentError meanfield(g, nambu = true, charge = I)
    g = oh |> greenfunction(GS.Schur(boundary = 0))
    @test_throws ArgumentError meanfield(g)
    g = LP.honeycomb() |> hamiltonian(hopping(I), orbitals = (2,1)) |> greenfunction
    @test_throws ArgumentError meanfield(g)
end

using Distributed; addprocs(1) # for Serialization below
@everywhere using Quantica

@testset begin "OrbitalSliceArray serialization"
    g = LP.linear() |> supercell(4) |> supercell |> hamiltonian(hopping(I), orbitals = 2) |> greenfunction
    m = g(0.2)[cells = 0]
    v = serialize(m)
    @test v === parent(m)
    @test deserialize(m, v) === m
    m = g(0.2)[diagonal(cells = 0)]
    v = serialize(m)
    @test v === parent(m).diag
    @test deserialize(m, v) === m
    m = g(0.2)[sitepairs(range = 1)]
    v = serialize(m)
    @test v === Quantica.nonzeros(parent(m))
    @test deserialize(m, v) === m

    m = g(0.2)[(; cells = 0), (; cells = 0, region = r -> r[1] < 2)]
    m´ = g(0.2)[cells = 0]
    v = serialize(m)
    @test_throws ArgumentError deserialize(m´, v)

    # issue #327
    g = LP.linear() |> hopping(1) |> attach(nothing, cells = 0) |> greenfunction
    @test remotecall_fetch(g[1], 2, 0) isa OrbitalSliceMatrix
end
