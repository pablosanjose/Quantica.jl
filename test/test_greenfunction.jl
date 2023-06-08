using Quantica: GreenFunction, GreenSlice, GreenSolution, zerocell, cellorbs, cellorb

function testgreen(h, s; kw...)
    ω = 0.2
    g = greenfunction(h, s)
    @test g isa GreenFunction
    gω = g(ω; kw...)
    @test gω isa GreenSolution
    L = Quantica.latdim(lattice(h))
    z = zero(SVector{L,Int})
    o = Quantica.unitvector(1, SVector{L,Int})
    locs = (cellsites(z, :), cellsites(z, 2:3), cellsites(z, 2), cellsites(o, :), cellorb(o, 1), cellorbs(z, 2:3))
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

@testset "bare greenfunctions" begin
    h0 = LP.honeycomb() |> hamiltonian(hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell(region = RP.circle(10))
    s0 = GS.SparseLU()
    h1 = LP.square() |> hamiltonian(@onsite((; o = 1) -> o*I) + hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> abs(r[2]) < 2)
    s1 = GS.Schur()
    s1´ = GS.Schur(boundary = -1)
    for (h, s) in zip((h0, h1, h1), (s0, s1, s1´))
        testgreen(h, s; o = 2)
    end
end

@testset "greenfunction with contacts" begin
    g = LP.linear() |> hamiltonian(hopping(I), orbitals = 2) |> attach(@onsite(ω->im*I), cells = 1) |> attach(@onsite(ω->im*I), cells = 4) |> greenfunction
    @test size(g[(; cells = 2), (; cells = 3)](0.2)) == (2,2)

    h0 = LP.square() |> hamiltonian(hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell(region = RP.circle(10))
    s0 = GS.SparseLU()
    h1 = LP.square() |> hamiltonian(@onsite((; o = 1) -> o*I) + hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> abs(r[2]) < 2)
    s1 = GS.Schur()
    s1´ = GS.Schur(boundary = -1)
    sites´ = (; region = r -> abs(r[2]) < 2 && r[1] == 0)
    # non-hermitian Σ model
    mod = @onsite((ω, r; o = 1) -> (o - im*ω)*I) +
          plusadjoint(@onsite((ω; p=1)-> p*I) +  @hopping((ω, r, dr; t = 1) -> im*dr[1]*t*I; range = 1))
    g0, g1´ = greenfunction(h0, s0), greenfunction(h1, s1´)
    for (h, s) in zip((h0, h1, h1), (s0, s1, s1´))
        oh = h |> attach(nothing; sites´...)
        testgreen(oh, s)
        oh = h |> attach(mod; sites´...)
        testgreen(oh, s)
        for glead in (g0, g1´)
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
end

@testset "greenfunction KPM" begin
    g = HP.graphene(a0 = 1, t0 = 1, orbitals = (2,1)) |> supercell(region = RP.circle(20)) |>
        attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 300, bandrange = (-3.1, 3.1)))
    ρs = ldos(g[1])
    for ω in -3:0.1:3
        @test all(>=(0), ρs(ω))
    end

    ω = -0.1
    gωs = g[1](ω)
    ρflat = -imag.(diag(gωs))/pi
    @test all(>(0), ρflat)
    ρ = ρs(ω)
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
    g1 = LP.square() |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 1) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));
    g2 = LP.square() |> hamiltonian(@hopping((r, dr; B = 0.1) -> I * cis(B * dr' * SA[r[2],-r[1]])), orbitals = 2) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));
    J1 = current(g1[cells = SA[1]])
    J2 = current(g2[cells = SA[1]])
    @test size(J1(0.2)) == size(J2(0.2)) == (3, 3)
    @test 2*J1(0.2; B = 0.1) ≈ J2(0.2; B = 0.1)

    glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((0,1), region = r -> -1 <= r[1] <= 1) |> greenfunction(GS.Schur(boundary = 0));
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
end

