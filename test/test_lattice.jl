using Quantica: Sublat, Lattice, transform!, translate!, nsites

@testset "Internal API" begin
    s = sublat((0,0,0))
    @test Quantica.embdim(s) == 3
    @test Quantica.numbertype(s) <: AbstractFloat

    lat = LP.honeycomb()
    @test length(Quantica.bravais_vectors(lat)) == 2
    @test_throws BoundsError sites(lat, :C)
    @test length(sites(lat, 1)) == 1
    @test length(lat) == 2

    nt = (; region = r -> r[1] == 1, sublats = :C)
    s = siteselector(; nt...)
    @test NamedTuple(s) === (; nt..., cells = missing)
    h = hopselector(; nt...)
    @test NamedTuple(h).sublats == :C

    h = HP.graphene(orbitals = 2)
    @test Quantica.flatrange(h, :B) === 3:4
end

@testset "lattice sublats" begin
    sitelist = [(3,3), (3,3.), [3,3.], SA[3, 3], SA[3, 3f0], SA[3f0, 3.0]]
    for site2 in sitelist, site1 in sitelist
        T = float(promote_type(typeof.(site1)..., typeof.(site2)...))
        @test sublat(site1, site2) isa Sublat{T,2}
        @test sublat([site1, site2]) isa Sublat{T,2}
        @test lattice(sublat(site1), sublat(site2)) isa Lattice
    end
    @test sublat((3,)) isa Sublat{Float64,1}
    @test sublat(()) isa Sublat{Float64,0}
    @test sublat(SVector{3,Float64}[]) isa Sublat{Float64,3}
end

@testset "lattice construction" begin
    s = sublat((1, 2))
    s´ = sublat([0,0f0])
    for t in (Float32, Float64), e in 1:4, l = 1:4
        br = SMatrix{e,l,Float64}(I)
        if l > e
            @test_throws DimensionMismatch lattice(s; bravais = br, type = t, dim = Val(e))
            @test_throws DimensionMismatch lattice(s; bravais = br, type = t, dim = e)
        else
            @test lattice(s; bravais = br, type = t, dim = Val(e)) isa Lattice{t,e,l}
            @test lattice([s, s´]; bravais = Matrix(br), type = t, dim = e) isa Lattice{t,e,l}
        end
        if l > 2
            @test_throws DimensionMismatch lattice(s; bravais = br, type = t)
        else
            @test lattice(s; bravais = br, type = t) isa Lattice{t,2,l}
        end
    end
    lat = lattice(sublat((0,0,0)); names = :A)      # scalar `names`
    @test lat isa Lattice{Float64,3,0}
    lat = lattice(sublat((0,0,0f0)), sublat((1,1,1f0)); bravais = SMatrix{3,3}(I))
    lat2 = lattice(lat, bravais = ())
    @test lat2 isa Lattice{Float32,3,0}
    @test sites(lat) === sites(lat2)                # site aliasing
    lat2 = lattice(lat, bravais = (), names = (:A,:B))
    @test lat2 isa Lattice{Float32,3,0}
    @test sites(lat) === sites(lat2)                # site aliasing
    @test_throws ArgumentError lattice(lat, names = (:A,:B,:C)) # too many `names`
    lat2 = lattice(lat, type = Float64)
    @test lat2 isa Lattice{Float64,3,3}
    @test sites(lat) !== sites(lat2)                # no site aliasing
    lat2 = lattice(lat, dim = Val(2), bravais = SA[1 2; 3 4])
    @test lat2 isa Lattice{Float32,2,2}             # dimension cropping
    @test bravais_matrix(lat2) == SA[1 2; 3 4]
    # dim/type promotion
    @test lattice(sublat(Float16(0), name = :A), sublat(SA[1,2f0], name = :B)) isa Lattice{Float32,2}
end

@testset "lattice presets" begin
    a0s = (1, 2)
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc, LatticePresets.hcp)
    for a0 in a0s, t in (Float32, Float64), preset in presets
        @test preset(; a0 = a0, type = t) isa Lattice{t}
    end
    @test LatticePresets.cubic(bravais = (1,0)) isa Lattice{Float64,3,1}
    @test LatticePresets.cubic(bravais = ((1,0), (0,1)), dim = Val(2)) isa Lattice{Float64,2,2}
end

# @testset "siteindices/sitepositions" begin
#     lat = LatticePresets.honeycomb() |> unitcell(region = RegionPresets.circle(10))
#     @test sum(sitepositions(lat, sublats = :A)) ≈ -sum(sitepositions(lat, sublats = :B))
#     @test length(collect(siteindices(lat, sublats = :A))) == nsites(lat) ÷ 2

#     lat = LatticePresets.honeycomb() |> unitcell(2)
#     @test collect(siteindices(lat)) == 1:8
#     @test collect(siteindices(lat; indices = 10)) == Int[]
#     @test collect(siteindices(lat; indices = 5:10)) == 5:8
#     @test collect(siteindices(lat; indices = 5:10)) == 5:8
#     @test collect(siteindices(lat; indices = 5:10)) == 5:8
#     @test collect(siteindices(lat; indices = (1, 5:10))) == [1, 5 ,6, 7, 8]
#     @test collect(siteindices(lat; indices = (1, 10))) == [1]
# end

@testset "lattice combine" begin
    lat0 = transform!(LatticePresets.honeycomb(), r -> SA[r[2], -r[1]]) |> supercell((1,1), (-1,1))
    br = bravais_matrix(lat0)
    cell_1 = lat0 |>
        supercell(region = r -> -1.01/√3 <= r[1] <= 4/√3 && 0 <= r[2] <= 3.5)
    cell_2 = translate(transform(cell_1, r -> r + br * SA[2.2, -1]), SA[1,2])
    cell_p = lattice(sublat(br * SA[1.6,0.73], br * SA[1.6,1.27]))
    cells = combine(cell_1, cell_2, cell_p)
    @test length.(sites.(Ref(cells), 1:5)) == [14, 14, 14, 14, 2]
    @test_throws ArgumentError combine(LatticePresets.honeycomb(), LatticePresets.square())
    @test_throws ArgumentError combine(LatticePresets.honeycomb(), LatticePresets.linear())
    lat1 = transform(LatticePresets.honeycomb(type = Float32), r -> SA[r[2], -r[1]]) |> supercell((-1,1), (1,1))
    lat2 = combine(lat0, lat1)
    @test lat2 isa typeof(lat0)
    @test allunique(Quantica.sublatnames(lat2))
    lat1 = transform(LatticePresets.honeycomb(type = Float32), r -> SA[r[2], -r[1]]) |> supercell((-3,3), (1,1))
    @test_throws ArgumentError combine(lat0, lat1)
end

@testset "lattice nrange" begin
    lat = LP.honeycomb(a0 = 1)
    @test Quantica.nrange(1, lat) ≈ 1/√3
    @test Quantica.nrange(2, lat) ≈ 1
    @test Quantica.nrange(3, lat) ≈ 2/√3
    lat = LP.cubic(a0 = 1) |> supercell(10)
    @test Quantica.nrange(1, lat) ≈ 1
    @test Quantica.nrange(2, lat) ≈ √2
    @test Quantica.nrange(3, lat) ≈ √3
end

@testset "lattice supercell" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc, LatticePresets.hcp)
    for preset in presets
        lat = preset()
        E, L = Quantica.embdim(lat), Quantica.latdim(lat)
        for l in 1:L
            # some ramdon but deterministic svecs
            svecs = ntuple(i -> ntuple(j -> i*round(Int, cos(2j)) + j*round(Int, sin(2i)) , Val(E)), L-l)
            @test supercell(lat, svecs...) isa Lattice{Float64,E,L-l}
            @test supercell(lat, l) isa Lattice{Float64,E,L}
        end
    end
    @test supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(10, (10,0))) isa Lattice{Float64,2,0}
    @test supercell(LatticePresets.honeycomb(), (2,1), region = RegionPresets.circle(10)) isa Lattice{Float64,2,1}
    @test supercell(LatticePresets.bcc(), (2,1,0), region = RegionPresets.circle(10)) isa Lattice{Float64,3,1}
    @test supercell(LatticePresets.cubic(), (2,1,0), region = RegionPresets.sphere(10, (10,2,1))) isa Lattice{Float64,3,1}
end

@testset "lattice boolean regions" begin
    lat = supercell(LP.square(), region = xor(RP.square(10), RP.square(20)), seed = SA[20,0])
    @test length(sites(lat)) == 320
    lat = supercell(LP.honeycomb(), region = xor(RP.circle(20), RP.square(10)))
    lat´ = supercell(LP.honeycomb(), region = RP.circle(20) & !RP.square(10))
    @test sites(lat) == sites(lat´)
    lat = supercell(LP.honeycomb(), region = RP.circle(5, (5,0)) | RP.circle(5, (15,0)) | RP.circle(5, (25,0)))
    lat´ = supercell(LP.honeycomb(), region = RP.circle(5, (5,0)))
    @test length(sites(lat)) == 3 * length(sites(lat´))
end

@testset "lattice slices" begin
    lat = LP.honeycomb() |> supercell(2)
    ls1 = lat[sublats = :B, region = RP.ellipse((10, 20), (0, 1/√3))]
    ls2 = lat[sublats = :A, region = RP.ellipse((10, 20))]
    ls3 = lat[region = RP.ellipse((10, 20))]
    @test length(ls1) == length(ls2)
    @test ls2[2] isa Tuple{SVector{2, Int}, Int}
    ls = ls1[sublats = :B]
    @test isempty(ls)
    ls = ls3[sublats = :B, region = RP.ellipse((1, 2))]
    @test !isempty(ls)
    ls´ = ls3[(; sublats = :B, region = RP.ellipse((1, 2)))]
    @test nsites(ls) == nsites(ls´)
    ls = lat[cells = n -> 5 < norm(n) < 10]
    @test !isempty(Quantica.cells(ls)) && all(n -> 5 < norm(n) < 10, Quantica.cells(ls))
    ls = lat[region = r -> 5 < norm(r) < 10]
    @test !isempty(Quantica.cells(ls)) && all(r -> 5 < norm(r) < 10, Quantica.sites(ls))
    ls = lat[sites(SA[1,0], 1:3)]
    @test ls isa Quantica.SiteSlice
    @test nsites(ls) == 3
    ls = lat[sites(SA[1,0], 2)]
    @test ls isa Quantica.SiteSlice
    @test nsites(ls) == 1
    # test the difference between a null selector and an unspecified one
    ls = lat[cells = SVector{2,Int}[]]
    @test isempty(ls)
    ls = lat[cells = missing]
    @test !isempty(ls)
    ls = lat[region = RP.circle(4)]
    ls´ = ls[cells = n -> !isreal(n[1])]
    @test isempty(ls´)
end
