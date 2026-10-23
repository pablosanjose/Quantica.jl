using Quantica: GreenFunction, GreenSlice, GreenSolution, OrbitalSliceMatrix, Diagonal

using ArnoldiMethod  # for KPM bandrange
using LinearAlgebra

@testset "densitymatrix sparse slicing" begin
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

function testcond(@nospecialize(g0); nambu = false)
    G1 = conductance(g0[1]; nambu)
    G2 = conductance(g0[2]; nambu)
    G12 = conductance(g0[1,2]; nambu)
    T12 = transmission(g0[1,2])
    @test_throws ArgumentError transmission(g0[1])
    @test_throws ArgumentError transmission(g0[1, 1])
    for ω in -3:0.1:3
        @test ifelse(nambu, 6.000001, 3.00000001) >= G1(ω) >= 0
        @test G1(ω) ≈ G1(-ω) ≈ G2(ω) ≈ G2(-ω)
        @test T12(ω) ≈ T12(-ω)
        nambu || @test G1(ω) ≈ T12(ω) atol = 1e-8
        @test G12(ω) ≈ G12(-ω) atol = 1e-8
        nambu || @test G1(ω) ≈ -G12(ω) atol = 1e-8
    end
end

# @nospecialize: whole-body inference over all the Josephson types takes minutes to compile otherwise
function testjosephson(@nospecialize(g0))
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
    J = josephson(g0[1], Paths.sawtooth(-4, -1); callback = ((x,y) -> @test x isa ComplexF64 && y isa ComplexF64))
    @test J() <= 10*eps(Float64)
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

@testset "observables" begin
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

    # Quantized conductance
    glead = LP.square() |> hopping(1) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));
    g0 = LP.square() |> hopping(1) |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> attach(glead, reverse = true) |> attach(glead) |> greenfunction;
    T = transmission(g0[2, 1])
    @test T(0) ≈ T(0.2) ≈ 3
    @test T(1) ≈ 2
    # band-edge effects
    @test T(1.99999999999) ≈ 2
    @test T(2.00000000001) ≈ 1

    # Schur velocity classifier stability
    nnn(dr, ϕ) = ifelse(iseven(round(Int, atan(dr[2], dr[1])/(pi/3))), cis(ϕ), cis(-ϕ))
    haldane_hop(sub, ϕ) = @hopping((r, dr; α = Returns(0)) -> α(r)/(6√3)*nnn(dr, ϕ), sublats = sub=>sub, range = (0.9,1.1))
    haldane = haldane_hop(:A, π/2) + haldane_hop(:B, -π/2)
    h = LP.honeycomb() |> supercell((2,-1), (-1,2)) |> hopping(-3.1) + haldane |> supercell((1,1), (1,0)) |> stitch((0, :))
    gR = h(α = Returns(0.02)) |> greenfunction(GS.Schur(boundary = 0))
    gL = h(α = Returns(-0.02)) |> greenfunction(GS.Schur(boundary = 0))
    gC = supercell(h(α = r -> 0.02 * sign(r[1])); region = r -> abs(r[1]) < 0.51) |>  attach(gR; region = r->r[1] >= 0) |> attach(gL; region = r->r[1] <= 0, reverse = true) |> greenfunction
    G = conductance(gC[1])
    @test 0 < G(0.019) < 2

    glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((0,1), region = r -> -1 <= r[1] <= 1) |> attach(nothing; cells = SA[10]) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[2] ≈ 5 && -1 <= r[1] <= 1
    g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = RP.square(10)) |> attach(glead; reverse = true, region = contact2) |> attach(glead; transform = r->SA[0 1; 1 0] * r, region = contact1) |> greenfunction;
    testcond(g0)

    glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[1] ≈ -5 && -1 <= r[2] <= 1
    g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = RP.square(10)) |> attach(glead; reverse = true, region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    testcond(g0)

    glead = LP.square() |> hamiltonian(hopping(I) + onsite(SA[0 1; 1 0]), orbitals = 2) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    contact1 = r -> r[1] ≈ 5 && -1 <= r[2] <= 1
    contact2 = r -> r[1] ≈ -5 && -1 <= r[2] <= 1
    g0 = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(region = RP.square(10)) |> attach(glead; reverse = true, region = contact2) |> attach(glead; region = contact1) |> greenfunction;
    testcond(g0; nambu = true)
    testjosephson(g0)

    # test omegamap gets passed to integrand
    glead = LP.square() |> hamiltonian(@onsite((;ω=0) -> SA[im*ω 1; 1 im*ω]) + hopping(I), orbitals = 2) |> supercell((1,0), region = r -> -1 <= r[2] <= 1) |> greenfunction(GS.Schur(boundary = 0));
    g0 = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(region = RP.square(10)) |> attach(glead; reverse = true, region = contact2) |> attach(glead; region = contact1) |> greenfunction;
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
    ρ0sol = ρ0(0.2, 0.3)
    ρsol = ρ(0.2, 0.3)
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ0sol = ρ0()
    ρsol = ρ()
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ = densitymatrix(g[sites(:)])
    ρ0 = densitymatrix(g[sites(:)], Paths.sawtooth(-4, 4))
    ρ0sol = ρ0(0.2, 0.3)
    ρsol = ρ(0.2, 0.3)
    @test maximum(abs, ρ0sol - ρsol) < 1e-7
    @test typeof(ρ0sol) == typeof(ρsol)
    ρ0sol = ρ0()
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

    # test DensityMatrixSchurSolver in 2D
    ref = Ref(0.0)
    callback(x,z) = (ref[] = sum(abs, z))
    callback(x,y,z) = (ref[] = sum(abs, z))
    g = LP.square() |> hopping(1) |> supercell(3,1) |> greenfunction(GS.Schur(; axis = 2, atol = 1e-2, callback))
    ρ = densitymatrix(g[])
    @test iszero(ref[])
    # LDOS singularity at ω = 0 has zero real part of g. Needs broadening for integral to converge
    @test all(x->abs(real(x)) < 1e-4, diag(g(0 + 1e-8im)[]))
    @test !iszero(ref[])
    ref = Ref(0.0)
    @test diag(ρ()) ≈ [0.5,0.5,0.5]
    @test !iszero(ref[])

    # parametric path and system
    g = LP.linear() |> supercell |> @onsite((; o = 0.5) -> o) |> greenfunction
    ρ = densitymatrix(g[], Paths.polygon((µ, T; o = 0.5) -> o > µ ? (-1, µ, o + im, 1) : (-1, o + im, µ, 1)))
    @test real(only(ρ(0, 0; o = -0.5))) ≈ 1
    @test real(only(ρ(0, 0; o = 0.5))) < 1e-7
    @test real(only(ρ(0, 0; o = 1.5))) < 1e-7
    @test real(only(ρ(0, 0; o = -1.5))) < 1e-7
    d = Quantica.integrand(ρ)
    @test_throws MethodError d(2; o = 0.8)

    # test that contacts are ignored when building a GreenSlice identity
    g = LP.linear() |> supercell(2) |> hopping(1) |> attach(@onsite(ω->error())) |> greenfunction
    @test_throws ErrorException g(0)[]
    @test g[sites(1), sites(1:2)](2I) == SA[2 0]  # doesn't throw
end

@testset "meanfield" begin
    oh = LP.honeycomb() |> hamiltonian(hopping(SA[1 im; -im -1]) - onsite(SA[1 0; 0 -1], sublats = :A), orbitals = 2) |> supercell((1,-1))
    g = oh |> greenfunction
    Q = SA[0 1; im 0]  # nonhermitian charge not allowed
    @test_throws ArgumentError meanfield(g; selector = (; range = 1), potential = 2, fock = 1.5, charge = Q)
    Q = SA[0 -im; im 0]
    m = meanfield(g; selector = (; range = 1), potential = 2, fock = 1.5, hartree = 0, charge = Q, nambu = false)
    Φ = m(0.2, 0.3)
    ρ = m.rho(0.2, 0.3)
    ρ12 = ρ[sites(1), sites(2)]
    ρ11 = ρ[sites(1), sites(1)]
    @test !iszero(Φ[sites(1), sites(2)])
    @test Φ[sites(1), sites(2)] ≈ - 1.5 * Q * ρ12 * Q
    @test Φ[sites(1), sites(1)] ≈ - 1.5 * Q * ρ11 * Q

    # model potential
    g = LP.linear() |> hamiltonian(hopping(SA[1 im; -im -1]) - onsite(SA[1 0; 0 -1]), orbitals = 2) |> greenfunction
    Q = SA[0 -im; im 0]
    m = meanfield(g; hartree = hopping((r,dr)->norm(dr), range = 1) + onsite(2), fock = hopping(2, range = 1)+onsite(2), charge = Q, nambu = false)
    Φ = m(0.2, 0.3)
    ρ = m.rho(0.2, 0.3)
    ρ11 = ρ[sites(1), sites(1)]
    @test !iszero(ρ11)
    @test Φ[sites(1), sites(1)] ≈ Q*(tr(Q*ρ11)*(2 + 2)) - 2 * Q * ρ11 * Q

    # spinless nambu
    oh = LP.linear() |> hamiltonian(hopping((r, dr) -> SA[1 sign(dr[1]); -sign(dr[1]) -1]) - onsite(SA[1 0.1; 0.1 -1]), orbitals = 2)
    g = oh |> greenfunction
    Q = SA[1 0; 0 -1]
    m = meanfield(g; selector = (; range = 1), nambu = true, hartree = r -> 1/(1+norm(r)), fock = 1.5, charge = Q)
    @test !iszero(m()[sites(1)][1,2])       # decoder preserves pairing
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

@testset "SpectralSum" begin
    g = LP.linear() |> @hopping((;t = 1) -> t) |> supercell(10) |> supercell |> greenfunction(GS.Spectrum())
    gs = getindex(g, siteselector(region = r -> 2<=r[1]<=5))
    ss = Quantica.SpectralSum(gs)
    @test ss(0.1, 0; t = 0) ≈ I
    @test ss(identity; t = 1) do tmp, fn, φn
        mul!(tmp, φn, φn', 1, 1)
    end ≈ I
    ss = Quantica.SpectralSum(gs; inplace = false)
    @test ss(identity; t = 1) do fn, φn
        φn * φn'
    end ≈ I
    @test ss(<(0)) do fn, φn
        (φn * φn') .* fn
    end |> diag ≈ SA[1,1,1,1]/2
end


@testset "Scattering" begin
    model = hopping(1, range = √2)
    glead = LP.square() |> model |> supercell((4,0), region = r->-3<r[2]<3-0.5*r[1]) |> greenfunction(GS.Schur(boundary = 0))
    glead´ = LP.square() |> model |> supercell((4,0), region = r->-2<r[2]<2-0.5*r[1]) |> greenfunction(GS.Schur(boundary = 0))
    g = LP.square() |> model |> supercell(region = r->abs(r[2])<4 && 0<=r[1]<=3) |>
        attach(glead´, model; reverse = true) |>
        attach(@onsite((ω; Γ = 0) -> -im*Γ*I), region = r -> r[1] == 1) |>
        attach(glead, model) |> greenfunction
    s = scattering(g)
    S = Matrix(s(0.2))
    @test S'S ≈ I
    # with loss
    S = Matrix(s(0.2; Γ = 0.1))
    @test real(tr(S'S)) < 0.9*size(S,1)
    # with gain
    S = Matrix(s(0.2; Γ = -0.1))
    @test real(tr(S'S)) > 1.1*size(S,1)

    # Methods without a model for coupling
    glead = LP.square() |> hopping(1) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0))
    g = LP.square() |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> hopping(1) + @onsite((r; U = 0) -> U * (r[2] ≈ 1)) |>
        attach(glead, reverse = true) |> attach(glead) |> greenfunction
    s = scattering(g)
    sω = s(0.2; U = 2)
    @test Quantica.islead(s) == Quantica.islead(sω) == (true, true)
    @test Quantica.numleads(s) == Quantica.numleads(sω) == 2
    S = Matrix(sω)
    @test S'S ≈ I
    @test sω[2, 1] == Quantica.matrix(sω)[2, 1]
    @test tr(sω[2, 1]'sω[2, 1]) ≈ transmission(g[2, 1])(0.2; U = 2)
end
