using Quantica: GreenFunction, GreenSlice, GreenSolution, zerocell, CellOrbitals, ncontacts

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
    locs = (cellsites(z, :), cellsites(z, 2), cellsites(o, 1:2), cellsites(o, (1,2)),
            CellOrbitals(o, 1), CellOrbitals(z, 1:2), CellOrbitals(z, SA[2,1]),
            conts...)
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
    @test only(g[cellsites(SA[1],1)](1.0; ω = 0)) ≈ 1.0   atol = 0.000001
    @test only(g[cellsites(SA[1],1)](1.0; ω = -1)) ≈ 0.5  atol = 0.000001
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
end

@testset "GreenSolvers applicability" begin
    h = HP.graphene()
    @test_throws ArgumentError greenfunction(h, GS.Schur())
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
    ρ´ = ldos(g[1])(ω)
    @test all(<(0.01), abs.(ρ .- ρ´))
    @test all(<(0.01), abs.(ρflat .- ρflat´))
    @test_throws ArgumentError g[cellsites((), 3:4)](ω)
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
        @test all(x -> Quantica.chop(imag(x)) ≈ 0, g1[2,:])
        @test all(x -> Quantica.chop(imag(x)) ≈ 0, g1[:,2])
        g2 = g[cellsites((0,0),:), cellsites((1,1),2)](0.2)
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
    @test g(-1)[cellsites(SA[1,-1], 1), cellsites(SA[0,0],1)] isa AbstractMatrix
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

@testset "diagonal slicing" begin
    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell((2,-2), region = r -> 0<=r[2]<=5) |>
        attach(nothing, cells = 2, region = r -> 0<=r[2]<=2) |> attach(nothing, cells = 3) |>
        greenfunction(GS.Schur(boundary = -2))
    ω = 0.6
    @test g[diagonal(2)](ω) isa OrbitalSliceVector
    @test g[diagonal(2)](ω) == g(ω)[diagonal(2)]
    @test size(g[diagonal(1)](ω)) == (12,)
    @test size(g[diagonal(1, kernel = I)](ω)) == (8,)
    @test length(g[diagonal(:)](ω)) == length(g[diagonal(1)](ω)) + length(g[diagonal(2)](ω)) == 48
    @test length(g[diagonal(:, kernel = I)](ω)) == length(g[diagonal(1, kernel = I)](ω)) + length(g[diagonal(2, kernel = I)](ω)) == 32
    @test length(g[diagonal(cells = 2:3)](ω)) == 72
    @test length(g[diagonal(cells = 2:3, kernel = I)](ω)) == 48
    @test ldos(g[1])(ω) ≈ -imag.(g[diagonal(1; kernel = I)](ω)) ./ π
end

@testset "OrbitalSliceArray slicing" begin
    g = LP.linear() |> hopping(1) + onsite(1) |> supercell(4) |> greenfunction
    gmat = g[cells = SA[2]](0.2)
    @test gmat isa Quantica.OrbitalSliceMatrix
    @test size(gmat) == (4, 4)
    gmat´ = gmat[cells = SA[1]]
    @test gmat´ isa Quantica.OrbitalSliceMatrix
    @test isempty(gmat´)
    gmat = g[(; cells = SA[2]), cellsites(SA[1], 1:2)](0.2)
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
    @test_throws Quantica.Dictionaries.IndexError gmat[cellsites(SA[1],:)]  # `:` means all sites in cell
    gmat´ = gmat[cellsites(SA[1], 1:2)]
    @test gmat´ isa Matrix
    @test size(gmat´) == (2, 2)
    c = cellsites(SA[1], 1)
    view(gmat, c)
    @test (@allocations view(gmat, c)) <= 2
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
    J1 = josephson(g0[1], 4; phases = subdiv(0, pi, 10))
    J2 = josephson(g0[2], 4; phases = subdiv(0, pi, 10))
    @test all(>=(0), Quantica.chop.(J1()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 0.000001)).(J1(), J2()))
end

@testset "greenfunction observables" begin
    # single-orbital vs two-orbital
    g1 = LP.square() |> supercell((1,0), region = r->-2<r[2]<2) |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 1) |> greenfunction(GS.Schur(boundary = 0));
    g2 = LP.square() |> supercell((1,0), region = r->-2<r[2]<2) |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 2) |> greenfunction(GS.Schur(boundary = 0));
    J1 = current(g1[cells = SA[1]])
    J2 = current(g2[cells = SA[1]])
    @test size(J1(0.2)) == size(J2(0.2)) == (3, 3)
    @test 2*J1(0.2; B = 0.1) ≈ J2(0.2; B = 0.1)

    ρ = densitymatrix(g1[cells = SA[1]], 5)
    @test all(≈(0.5), diag(ρ(0, 0; B=0.3))) # half filling
    ρ = densitymatrix(g1[cells = SA[1]], 7)
    @test all(<(0.96), real(diag(ρ(4, 1; B=0.1)))) # thermal depletion
    h = LP.honeycomb() |> hopping(1) |> supercell(region = RP.circle(10))
    reg = (; region = RP.circle(0.5))
    gLU = h |> greenfunction(GS.SparseLU());
    gSpectrum = h |> greenfunction(GS.Spectrum());
    gKPM = h |> attach(nothing; reg...) |> greenfunction(GS.KPM(order = 100000, bandrange = (-3,3)));
    ρ1, ρ2, ρ3 = densitymatrix(gLU[reg], (-3,3)), densitymatrix(gSpectrum[reg]), densitymatrix(gKPM[1])
    @test ρ1() ≈ ρ2() atol = 0.00001
    @test ρ2() ≈ ρ3() atol = 0.00001
    gLU´ = h |> attach(nothing; reg...) |> greenfunction(GS.SparseLU());
    ρ1´ = densitymatrix(gLU´[1], (-3, 3))
    @test ρ1() ≈ ρ1´()
    gSpectrum´ = h |> attach(nothing; reg...) |> greenfunction(GS.Spectrum());
    ρ2´ = densitymatrix(gSpectrum´[1])
    @test ρ2() ≈ ρ2´()

    ρ = densitymatrix(g2[(; cells = SA[0]), (; cells = SA[1])], 5)
    ρ0 = ρ(0, 0; B=0.3)
    @test ρ0 isa OrbitalSliceMatrix
    @test iszero(ρ0)        # rows are on boundary
    @test ρ0[cellsites(SA[0], 1), cellsites(SA[1], 1)] isa Matrix
    @test size(view(ρ0, cellsites(SA[0], 1), cellsites(SA[1], 1))) == (2, 2)

    # Diagonal slicing not yet supported
    @test_broken densitymatrix(g1[diagonal(cells = SA[1])], 5)
    @test_broken densitymatrix(gSpectrum[diagonal(cells = SA[])])
    @test_broken densitymatrix(gKPM[diagonal(1)])

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
    g0 = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(region = RP.square(10)) |> attach(glead, reverse = true; region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    testcond(g0; nambu = true)
    testjosephson(g0)

    # test fermi at zero temperature
    g = LP.linear() |> hopping(1) |> supercell(3) |> supercell |> greenfunction(GS.Spectrum())
    @test ρ = diag(densitymatrix(g[])()) ≈ [0.5, 0.5, 0.5]
end

@testset "mean-field models" begin
    h1 = LP.honeycomb() |> supercell(2) |> supercell |> hamiltonian(onsite(0I) + hopping(I), orbitals = 1)
    h2 = LP.honeycomb() |> supercell(2) |> supercell |> hamiltonian(onsite(0I) + hopping(I), orbitals = 2)
    h3 = LP.honeycomb() |> supercell(2) |> supercell |> hamiltonian(onsite(0I) + hopping(I), orbitals = (1,2))
    for h0 in (h1, h2, h3)
        ρ0 = densitymatrix(greenfunction(h0, GS.Spectrum())[cells = SA[]])();
        h = h0 |> @onsite!((o, s; ρ = ρ0, t) --> o + t*ρ[s])
        @test diag(h(t = 2)[()]) ≈ 2*diag(ρ0) atol = 0.0000001
        h = h0 |> @hopping!((t, si, sj; ρ = ρ0, α = 2) --> α*ρ[si, sj])
        @test h() isa Quantica.Hamiltonian
        diff = (h()[()] - 2ρ0) .* h()[()]
        @test iszero(diff)
    end
end
