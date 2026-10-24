using Quantica: GreenFunction
using LinearAlgebra

@testset "contacts" begin
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

    # check reverse gives same as non-reverse
    model = -hopping(SA[1 0; 0 -1]) + @hopping((r, dr) ->  SA[0 dr[1]; -dr[1] 0])
    h = LP.square() |> hamiltonian(model, orbitals = 2) |> stitch(SA[2])
    glead = h |> greenfunction(GS.Schur(boundary = 0))
    gR = h |> attach(glead, cells = SA[0], reverse = true) |> greenfunction(GS.Schur(boundary = -1))
    gL = h |> attach(glead, cells = SA[0]) |> greenfunction(GS.Schur(boundary = 1))
    @test gR[](0.01) ≈ gL[](0.01)
end
