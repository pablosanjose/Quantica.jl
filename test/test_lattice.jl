using Quantica: Sublat
using Random

@testset "sublat input" begin
    sitelist = [(3,3), (3,3.), [3,3.], SA[3, 3], SA[3, 3f0], SA[3f0, 3.]]
    for site2 in sitelist, site1 in sitelist
        T = float(promote_type(typeof.(site1)..., typeof.(site2)...))
        @test sublat(site1, site2) isa Sublat{T,2}
        @test sublat([site1, site2]) isa Sublat{T,2}
    end
    @test sublat((3,)) isa Sublat{Float64,1}
    @test sublat(()) isa Sublat{Float64,0}
    @test_throws ArgumentError sublat(SVector{3,Float64}[])
end

# @testset "lattice" begin
#     s = sublat((1, 2))
#     for t in (Float32, Float64), e in 1:4, l = 1:4
#         br = SMatrix{l,l,Float64}(I)
#         @test lattice(s; bravais = br, type = t, dim = Val(e)) isa Lattice{e,min(l,e),t}
#         @test lattice(s; bravais = br, type = t, dim = e) isa Lattice{e,min(l,e),t}
#     end
#     lat = lattice(sublat((0,0,0)), sublat((1,1,1f0)); bravais = SMatrix{3,3}(I))
#     lat2 = lattice(lat, bravais = ())
#     @test lat2 isa Lattice{3,0}
#     @test allsitepositions(lat) === allsitepositions(lat2)
#     lat2 = lattice(lat, bravais = (), names = :A)
#     @test lat2 isa Lattice{3,0}
#     @test allsitepositions(lat) === allsitepositions(lat2)
#     lat2 = lattice(lat, dim = Val(2))
#     @test lat2 isa Lattice{2,2} # must be L <= E
#     @test allsitepositions(lat) !== allsitepositions(lat2)
#     lat2 = lattice(lat, type = Float64)
#     @test lat2 isa Lattice{3,3}
#     @test allsitepositions(lat) !== allsitepositions(lat2)
#     lat2 = lattice(lat, dim = Val(2), bravais = SA[1 2; 3 4])
#     @test bravais(lat2) == SA[1 2; 3 4]
# end

# @testset "lattice presets" begin
#     a0s = (1, 2)
#     presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
#                LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
#                LatticePresets.bcc, LatticePresets.hcp)
#     for a0 in a0s, t in (Float32, Float64), e in 1:4, preset in presets
#         @test preset(; a0 = a0, type = t, dim = e) isa Lattice{e,<:Any,t}
#     end
#     @test LatticePresets.cubic(bravais = (1,0)) isa Lattice{3,1}
#     @test LatticePresets.cubic(bravais = ((1,0), (0,1)), dim = Val(2)) isa Lattice{2,2}
# end

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

# @testset "lattice combine" begin
#     lat0 = transform!(r -> SA[r[2], -r[1]], LatticePresets.honeycomb()) |> unitcell((1,1), (-1,1))
#     br = bravais(lat0)
#     cell_1 = lat0 |>
#         unitcell(region = r -> -1.01/√3 <= r[1] <= 4/√3 && 0 <= r[2] <= 3.5)
#     cell_2 = transform!(r -> r + br * SA[2.2, -1], copy(cell_1))
#     cell_p = lattice(sublat(br * SA[1.6,0.73], br * SA[1.6,1.27]))
#     cells = combine(cell_1, cell_2, cell_p)
#     @test Quantica.nsites.(Ref(cells), 1:5) == [14, 14, 14, 14, 2]
# end

# @testset "lattice unitcell" begin
#     presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
#                LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
#                LatticePresets.bcc, LatticePresets.hcp)
#     for preset in presets
#         lat = preset()
#         E, L = dims(lat)
#         for l in 1:L
#             # some ramdon but deterministic svecs
#             svecs = ntuple(i -> ntuple(j -> i*round(Int, cos(2j)) + j*round(Int, sin(2i)) , Val(E)), L-l)
#             @test unitcell(lat, svecs...) isa Lattice{E,L-l}
#             @test unitcell(lat, l) isa Lattice{E,L}
#         end
#     end
#     @test unitcell(LatticePresets.honeycomb(), region = RegionPresets.circle(10, (10,0))) isa Lattice{2,0}
#     @test unitcell(LatticePresets.honeycomb(), (2,1), region = RegionPresets.circle(10)) isa Lattice{2,1}
#     @test unitcell(LatticePresets.bcc(), (2,1,0), region = RegionPresets.circle(10)) isa Lattice{3,1}
#     @test unitcell(LatticePresets.cubic(), (2,1,0), region = RegionPresets.sphere(10, (10,2,1))) isa Lattice{3,1}
# end

# @testset "lattice supercell" begin
#     presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
#                LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
#                LatticePresets.bcc, LatticePresets.hcp)
#     for preset in presets
#         lat = preset()
#         E, L = dims(lat)
#         for l in 1:L
#             # some ramdon but deterministic svecs
#             svecs = ntuple(i -> ntuple(j -> i*round(Int, cos(2j)) + j*round(Int, sin(2i)) , Val(E)), L-l)
#             @test supercell(lat, svecs...) isa Superlattice{E,<:Any,<:Any,L-l}
#             @test supercell(lat, l) isa Superlattice{E,<:Any,<:Any,L}
#         end
#     end
#     @test supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(10, (0,2))) isa Superlattice{2,2}
#     @test supercell(LatticePresets.honeycomb(), (2,1), region = RegionPresets.circle(10)) isa Superlattice{2,2}
#     @test supercell(LatticePresets.bcc(), (2,1,0), region = RegionPresets.circle(10, (1,0))) isa Superlattice{3,3}
#     @test supercell(LatticePresets.cubic(), (2,1,0), region = RegionPresets.sphere(10)) isa Superlattice{3,3}
# end

# @testset "boolean regions" begin
#     lat = unitcell(LP.square(), region = xor(RP.square(10), RP.square(20)))
#     @test nsites(lat) == 320
#     lat = unitcell(LP.honeycomb(), region = xor(RP.circle(20), RP.square(10)))
#     lat´ = unitcell(LP.honeycomb(), region = RP.circle(20) & !RP.square(10))
#     @test allsitepositions(lat) == allsitepositions(lat´)
#     lat = unitcell(LP.honeycomb(), region = RP.circle(5, (5,0)) | RP.circle(5, (15,0)) | RP.circle(5, (25,0)))
#     lat´ = unitcell(LP.honeycomb(), region = RP.circle(5, (5,0)))
#     @test nsites(lat) == 3 * nsites(lat´)
# end