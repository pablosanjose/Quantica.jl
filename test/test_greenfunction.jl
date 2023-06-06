using Quantica: GreenFunction, GreenSlice, GreenSolution, zerocell

function testgreen(h, s; kw...)
    ω = 0.2
    g = greenfunction(h, s)
    @test g isa GreenFunction
    gω = g(ω; kw...)
    @test gω isa GreenSolution
    for loc in (cellsites(zerocell(h), :), cellsites(zerocell(h), 2:3), cellsites(zerocell(h), 2))
        gs = g[loc]
        @test gs isa GreenSlice
        gsω = gs(ω; kw...)
        gωs = gω[loc]
        @test gsω == gωs
        @test all(x->imag(x)<0, diag(gωs))
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
    h0 = LP.square() |> hamiltonian(hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell(region = RP.circle(10))
    s0 = GS.SparseLU()
    h1 = LP.square() |> hamiltonian(@onsite((; o = 1) -> o*I) + hopping(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> abs(r[2]) < 2)
    s1 = GS.Schur()
    s1´ = GS.Schur(boundary = -1)
    sites´ = (; region = r -> abs(r[2]) < 2 && r[1] == 0)
    mod = @onsite((ω, r; o = 1) -> (o + ω)*I)
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