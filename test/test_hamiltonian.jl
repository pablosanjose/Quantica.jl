using Quantica: Hamiltonian, ParametricHamiltonian, sites, nsites, nonsites, nhoppings, coordination, flat, hybrid, transform!

@testset "basic hamiltonians" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular, LatticePresets.honeycomb,
               LatticePresets.cubic, LatticePresets.fcc, LatticePresets.bcc, LatticePresets.hcp)
    types = (Float16, Float32, Float64)
    ts = (1, 2.0I, @SMatrix[1 2; 3 4], 1.0f0*I)
    orbs = (1, Val(1), Val(2), 2)
    for preset in presets, type in types, lat in (preset(; type), supercell(preset(; type)))
        E, L = Quantica.embdim(lat), Quantica.latdim(lat)
        dn0 = ntuple(_ -> 1, Val(L))
        for (t, o) in zip(ts, orbs)
            @test hamiltonian(lat, onsite(t) + hopping(t; range = 1), orbitals = o) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) - hopping(t; dcells = dn0), orbitals = o) isa Hamiltonian
        end
    end
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3))
    @test h[SA[0,0]] === h[()] === flat(h.harmonics[1].h)
    # Inf range
    h = LatticePresets.square() |> supercell(region = RegionPresets.rectangle((5,6))) |>
        hamiltonian(hopping(1, range = Inf))
    @test Quantica.nhoppings(h) == 1190
    h = LatticePresets.square() |> hamiltonian(hopping(1, dcells = (10,0), range = Inf)) |> transform(r -> SA[0 1; 2 0] * r)
    @test Quantica.nhoppings(h) == 1
    @test isassigned(h, (10,0))
    @test bravais_matrix(h) == SA[0 1; 2 0]
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(1.0, sublats = :A) + hopping(I, range = 2/√3), orbitals = (Val(1), Val(2)))
    @test Quantica.nonsites(h) == 1
    @test Quantica.nhoppings(h) == 24
    @test ishermitian(h)
    h = LatticePresets.square() |> supercell(3) |> hamiltonian(hopping(1, range = 3) |> plusadjoint)
    @test Quantica.nhoppings(h) == 252
    @test ishermitian(h)
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 1)))
    @test Quantica.nhoppings(h) == 12
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 2), sublats = :A => :B))
    @test Quantica.nhoppings(h) == 9
    @test !ishermitian(h)
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 2), sublats = :A => :B) |> plusadjoint)
    @test Quantica.nhoppings(h) == 18
    @test ishermitian(h)
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 2/√3)))
    @test Quantica.nhoppings(h) == 18
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (2, 1)))
    @test Quantica.Quantica.nhoppings(h) == 0
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (30, 30)))
    @test Quantica.Quantica.nhoppings(h) == 12
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (10, 10.1)))
    @test Quantica.Quantica.nhoppings(h) == 48
end

@testset "hamiltonian orbitals" begin
    lat = LP.honeycomb()
    hop = hopping(SA[1 0], sublats = :A => :B)
    h = hamiltonian(lat, plusadjoint(hopping(hop)) + onsite(1, sublats = 2), orbitals = (2, Val(1)))
    h´ = hamiltonian(lat, hop + hop' + onsite(1, sublats = 2), orbitals = (2, Val(1)))
    @test h isa Hamiltonian
    @test h == h´
    @test_throws ArgumentError hamiltonian(lat, hopping(SA[1 0]), orbitals = (2, Val(1)))
    h = hamiltonian(lat, hopping(I) + onsite(SA[1 0; 0 1], sublats = :A), orbitals = (2, Val(1)))
    @test h isa Hamiltonian
end

@testset "hamiltonian sublats" begin
    lat = LP.honeycomb()
    h = hamiltonian(lat, hopping(1, sublats = (:A,:B) .=> (:A, :B)))
    @test iszero(h((1,2)))
    h = hamiltonian(lat, hopping(1, sublats = :A => :B))
    @test h((0,0)) == SA[0 0; 3 0]
    h = hamiltonian(lat, hopping(1, sublats = :A => :B) |> plusadjoint)
    @test h((0,0)) == SA[0 3; 3 0]
    h = hamiltonian(lat, hopping(1, sublats = (:A,:B) .=> (:B, :A), range = neighbors(2)))
    @test h((0,0)) == SA[0 3; 3 0]
    h = hamiltonian(lat, hopping(1, sublats = (:A,:B) => (:A, :B), range = neighbors(2)))
    @test h((0,0)) == SA[6 3; 3 6]
    h = hamiltonian(lat, hopping(1, sublats = (:A => :B, :A => :A), range = neighbors(2)))
    @test h((0,0)) == SA[6 0; 3 0]
    h = hamiltonian(lat, hopping(1, sublats = (:A => :B, (:A, :B) .=> (:A, :B)), range = neighbors(2)))
    @test h((0,0)) == SA[6 0; 3 6]
end

@testset "hamiltonian presets" begin
    h = HP.graphene(; a0 = 1, dim = 3, range = neighbors(2), t0 = 2.7, β = 3, orbitals = 2, names = (:a, :b), type = Float32)
    @test h isa Hamiltonian{Float32,3,2}
    h = HP.twisted_bilayer_graphene(; twistindex = 0, rangeinterlayer = 2, interlayerdistance = 1, a0 = 1, hopintra = 2.7 * I, orbitals = 2, names = (:A, :B, :A´, :B´))
    @test h isa Hamiltonian{Float64,3,2}
end

@testset "siteselectors/hopselectors" begin
    lat = LatticePresets.linear()
    @test supercell(lat, region = RP.segment(10)) isa Quantica.Lattice
    lat = LatticePresets.bcc()
    for r in (RP.sphere(3), RP.cube(3), RP.spheroid((3,4,5), (3,3,3)), RP.cuboid((3,4,5), (2,3,4)))
        @test supercell(lat, region = r) isa Quantica.Lattice
    end
    lat = LatticePresets.honeycomb()
    for r in (RP.circle(10), missing), s in (:A, 1, (:A, :B), [1, :B], missing), c in (SA[0,1], (0,1), [0,1]), cs in (c, (c, 2 .* c), [c, 2 .* c], missing)
        @test supercell(lat, region = r, sublats = s, cells = cs) isa Quantica.Lattice
    end
    r1 = RP.ellipse((10,15))
    r2 = (r, dr) -> norm(r) + norm(dr) < 2
    for region in (r1, missing),
        sublats in (:A, 1, (:A, :B), missing),
        c in (SA[0,1], (0,1)), dcells in (c, (c, 2 .* c), missing),
        range in (1, neighbors(2), (1, 2.0), (1, neighbors(3))),
        sublats´ in (:A, 1, (:A, :B)),
        regionhop in (r2, missing)

        sublatshop = ifelse(sublats === missing, missing, sublats´ .=> sublats)
        @test Quantica.apply(siteselector(; region, sublats), lat) isa Quantica.AppliedSiteSelector
        @test Quantica.apply(hopselector(; range, dcells, region = regionhop, sublats = sublatshop), lat) isa Quantica.AppliedHopSelector
        sublatshop = ifelse(sublats === missing, missing, sublats´ => sublats)
        @test Quantica.apply(hopselector(; range, dcells, region = regionhop, sublats = sublatshop), lat) isa Quantica.AppliedHopSelector
    end
end

@testset "models" begin
    mo = (onsite(1), onsite(r-> r[1]), @onsite((; o) -> o), @onsite((r; o) -> r[1]*o))
    mh = (hopping(1), hopping((r, dr)-> im*dr[1]), @hopping((; t) -> t), @hopping((r, dr; t) -> r[1]*t))
    for o in mo, h in mh
        @test length(Quantica.allterms(-o - 2*h)) == 2
        @test length(Quantica.allterms(onsite(2o + h))) == 1
        @test length(Quantica.allterms(hopping(o + h))) == 1
    end
end

@testset "hamiltonian supercell" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> supercell((1,-1), region = r -> abs(r[2])<2)
    @test nhoppings(h) == 22
    h = LatticePresets.square() |> hamiltonian(hopping(1)) |> supercell(3) |> supercell((1,0))
    @test nsites(h) == 9
    h = LatticePresets.square() |> hamiltonian(hopping(1, range = √2)) |> supercell(5) |> supercell((1,0))
    @test nhoppings(h) == 170
    @test nsites(h) == 25
    # dim-preserving supercell reshaping should always preserve coordination
    lat = lattice(sublat((0.0, -0.1), (0,0), (0,1/√3), (0.0,1.6/√3)), bravais = SA[cos(pi/3) sin(pi/3); -cos(pi/3) sin(pi/3)]')
    h = lat |> hamiltonian(hopping(1, range = 1/√3))
    c = coordination(h)
    iter = CartesianIndices((-3:3, -3:3))
    for Ic´ in iter, Ic in iter
        sc = SMatrix{2,2,Int}(Tuple(Ic)..., Tuple(Ic´)...)
        iszero(det(sc)) && continue
        h´ = supercell(h, sc)
        h´´ = supercell(h´, 2)
        @test coordination(h´´) ≈ coordination(h´) ≈ c
        h´ = supercell(h´, Tuple(Ic))
        h´´ = supercell(h´, 2)
        @test coordination(h´´) ≈ coordination(h´)
    end
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> supercell(2) |> supercell(mincoordination = 2)
    @test nsites(h) == 6
    h = LP.cubic() |> hamiltonian(hopping(1)) |> supercell(4) |> supercell(mincoordination = 4)
    @test nsites(h) == 0
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> supercell(region = RP.circle(5) & !RP.circle(2)) |> supercell(mincoordination = 2)
    @test nsites(h) == 144
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> supercell(10, region = !RP.circle(2, (0,8)))
    h´ = h |> supercell(1, mincoordination = 2)
    @test nsites(h´) == nsites(h) - 1
    h = LP.square() |> hamiltonian(hopping(0)) |> supercell(4, mincoordination = 2)
    @test nsites(h) == 0
    # check dn=0 invariant
    h = LP.linear() |> hamiltonian(hopping(1)) |> supercell((1,))
    @test length(h.harmonics) == 3 && iszero(first(h.harmonics).dn)
    # seeds
    p1 = SA[100,0]
    p2 = SA[0,20]
    lat = LatticePresets.honeycomb()
    model = hopping(1, range = 1/√3) + onsite(2)
    h1 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p1)<3, seed = p1)
    h2 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p2)<3, seed = p2)
    h = combine(h1, h2)
    h3 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p1)<3 || norm(r-p2)<3, seed = p2)
    @test Quantica.nsites(h) == 130
    @test Quantica.nsites(h3) == 64
end

@testset "hamiltonian wrap" begin
    for o in (1, (2,3))
        h = LatticePresets.honeycomb() |> hamiltonian(hopping((r, dr) -> 1/norm(dr) * I, range = 10), orbitals = o)
        @test_throws ArgumentError wrap(h, (1,2,3))
        @test_throws ArgumentError wrap(h, 1)
        wh = wrap(h, (1,2))
        @test wh(()) ≈ h(SA[1,2])
        wh = wrap(h, (1,:))
        @test wh(SA[2]) ≈ h(SA[1,2])
        h = LatticePresets.honeycomb() |> hamiltonian(hopping((r, dr) -> 1/norm(dr) * I, range = 10)) |> supercell(3)
        wh = wrap(h, (1,2))
        @test wh(()) ≈ h(SA[1,2])
        wh = wrap(h, (1,:))
        @test wh(SA[2]) ≈ h(SA[1,2])
        wh = wrap(h, (:,:))
        @test wh == h
        @test wh !== h
    end
end

@testset "hamiltonian HybridSparseMatrix" begin
    for o in (1, 2, (2, 2), (2,3))
        h = HP.graphene(orbitals = o) |> supercell(2)
        @test h[()] === h[(0,0)]
        s = h[hybrid((1,0))]
        @test !Quantica.needs_unflat_sync(s)
        if o == 1
            @test eltype(h[hybrid()]) === ComplexF64
            @test !Quantica.needs_flat_sync(s)
            @test Quantica.isaliased(s)
            @test h[()] === h[unflat()]
        elseif o == (2,2) || o == 2
            @test eltype(h[unflat()]) <: Quantica.SMatrix
            @test Quantica.needs_flat_sync(s)
            @test !Quantica.isaliased(s)
            h[()]
            @test Quantica.needs_flat_sync(s)
        else
            @test eltype(h[unflat()]) <: Quantica.SMatrixView
            @test Quantica.needs_flat_sync(s)
            @test !Quantica.isaliased(s)
            h[()]
            @test Quantica.needs_flat_sync(s)
        end
        @test_throws BoundsError h[(1,1)]
    end
end

@testset "hamiltonian call" begin
    for o in (1, (2,3))
        h = HP.graphene(orbitals = o) |> supercell(2)
        @test h() == h
        @test h() !== h
        @test h(; param = 1) == h
        @test_throws ArgumentError h(())
        @test_throws ArgumentError h(1,2)
        @test h((1,2)) == h(SA[1,2])
        @test Quantica.call!(h, (1,2)) === Quantica.call!(h, SA[2,3]) === Quantica.call!_output(h)
        h = supercell(h)
        @test h(()) == h(SA[]) == h[()]
        @test Quantica.call!(h, ()) === Quantica.call!(h, SA[]) === h[()] === Quantica.call!_output(h)
    end
    h = hamiltonian(LP.linear(), hopping(1))
    @test h(π) == h((π,)) == h([π]) == [-2;;]
end

@testset "parametric hamiltonian" begin
    lat = LP.honeycomb() |> supercell(2)
    h = hamiltonian(lat, @onsite((; o) -> o))
    h0 = h((0,0); o = 2)
    @test h0 == 2I
    @test Quantica.nnz(h0) == 8
    @test_throws UndefKeywordError h((0,0))
    h = hamiltonian(lat, @onsite((r; o = 0) -> o*r[1]))
    @test h((0,0); o = 2) ≈ Quantica.Diagonal([0, -1, 1, 0, 0, -1, 1, 0])
    @test h(; o = 1)[()] !== h(; o = 2)[()]
    @test Quantica.call!(h; o = 1)[()] === Quantica.call!(h; o = 2)[()]
    h´ = hamiltonian(h, @onsite!((o´, r; o = 0) -> o´ - o*r[1]))
    @test iszero(h´((0,0), o = 3))
    @test Quantica.nnz(h´((0,0), o = 3)) == 8
    h = hamiltonian(lat, onsite(1) + @hopping((r, dr; t) -> t * cis(dr[2])),
        @hopping!((t´, r, dr; takeabs = false) -> ifelse(takeabs, abs(t´), t´)))
    @test ishermitian(h(t = 1))
    h0 = h(t = 1)((0,0))
    @test !all(x -> x isa Real, h0)
    h0 = h(t = 1, takeabs = true)((0,0))
    @test all(==(1), Quantica.nonzeros(h0))
    h = LatticePresets.linear() |> hamiltonian(hopping(1), @onsite!((o; k) -> o + k*I))
    # No onsites, no need to specify k
    @test h() isa Hamiltonian
    # Issue #35
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> supercell(10)
        h´ = hamiltonian(h, @onsite!((o, r; b) -> o+b*I), @hopping!((t, r, dr; a = 2) -> t+r[1]*I),
                       @onsite!((o, r; b) -> o-b*I), @hopping!((t, r, dr; a = 2) -> t-r[1]*I))
        @test isapprox(h´(a=1, b=2)((1, 2)), h((1, 2)))
    end
    # Issue #37
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> supercell(10)
        h´ = hamiltonian(h, @onsite!(o -> o*cis(1)))
        @test h´()[()][1,1] ≈ h[()][1,1]*cis(1)
    end
    # Issue #54. Parametric Haldane model, unbounded modifiers
    sK(dr::SVector) = sK(atan(dr[2],dr[1]))
    sK(ϕ) = 2*mod(round(Int, 6*ϕ/(2π)), 2) - 1
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1),
        @hopping!((t, r, dr; λ, k = SA[0,0]) ->  λ*im*sK(dr+k); sublats = :A=>:A), # These should have range = Inf
        @hopping!((t, r, dr; λ, k = SA[0,0]) -> -λ*im*sK(dr+k); sublats = :B=>:B)) # These should have range = Inf
    @test h((π/2, -π/2), λ=1) ≈ [4 1; 1 -4]
    # Non-numeric parameters
    @test h((π/2, -π/2); λ = 1, k = SA[1,0]) ≈ [-4 1; 1 4]
    # # Issue 61, type stability
    # h = LatticePresets.honeycomb() |> hamiltonian(onsite(0))
    # @inferred hamiltonian(h, @onsite!((o;μ) -> o- μ))
    # @inferred hamiltonian(h, @onsite!(o->2o), @hopping!((t)->2t), @onsite!((o, r)->o+r[1]))
    # @inferred hamiltonian(h, @onsite!((o, r)->o*r[1]), @hopping!((t; p)->p*t), @onsite!((o; μ)->o-μ))

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(2I), orbitals = (Val(2), Val(1)), @hopping!((t; α, β = 0) -> α * t .+ β))
    b = h((0, 0); α = 2)
    @test b == [0 0 12; 0 0 0; 12 0 0]
end

@testset "hamiltonian nrange" begin
    lat = LatticePresets.honeycomb(a0 = 2)
    @test Quantica.nrange(1, lat) ≈ 2/√3
    @test Quantica.nrange(2, lat) ≈ 2
    @test Quantica.nrange(3, lat) ≈ 4/√3
    @test hamiltonian(lat, hopping(1)) |> nhoppings == 6
    @test hamiltonian(lat, hopping(1, range = neighbors(2))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = neighbors(3))) |> nhoppings == 24
    @test hamiltonian(lat, hopping(1, range = (neighbors(2), neighbors(3)))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = (neighbors(20), neighbors(20)))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = (neighbors(20), neighbors(21)))) |> nhoppings == 30
end

@testset "transform! hamiltonians" begin
    h = LP.honeycomb(dim = 3) |> hamiltonian(hopping(1))
    h1 = copy(h)
    h2 = transform!(h1, r -> SA[1 2 0; 2 3 0; 0 0 1] * r + SA[0,0,1])
    h3 = h1 |> transform!(r -> SA[1 2 0; 2 3 0; 0 0 1] * r + SA[0,0,1])
    @test h1 === h2 === h3
    @test all(r->r[3] == 2.0, sites(lattice(h3)))
    @test h((1,2)) == h3((1,2))
end

@testset "combine hamiltonians" begin
    h1 = LP.square(dim = Val(3)) |> hamiltonian(hopping(1))
    h2 = transform!(copy(h1), r -> r + SA[0,0,1])
    h = combine(h1, h2; coupling = hopping((r,dr) -> exp(-norm(dr)), range = √2))
    @test Quantica.coordination(h) == 9
    h0 = LP.honeycomb(dim = Val(3), names = (:A, :B)) |> hamiltonian(hopping(1))
    ht = LP.honeycomb(dim = Val(3), names = (:At, :Bt)) |> hamiltonian(hopping(1)) |> transform!(r -> r + SA[0,1/√3,1])
    hb = LP.honeycomb(dim = Val(3), names = (:Ab, :Bb)) |> hamiltonian(hopping(1)) |> transform!(r -> r + SA[0,1/√3,-1])
    h = combine(hb, h0, ht; coupling = hopping((r,dr) -> exp(-norm(dr)), range = 2,
        sublats = ((:A,:B) => (:At, :Bt), (:A,:B) => (:Ab, :Bb))) |>  plusadjoint)
    @test iszero(h((0,0))[1:2, 5:6])
    h = combine(hb, h0, ht; coupling = hopping((r,dr) -> exp(-norm(dr)), range = 2))
    @test !iszero(h((0,0))[1:2, 5:6])
end
