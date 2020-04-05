using Quantica: nsites, Sublat, Bravais, Lattice, Superlattice
using Random

@testset "bravais" begin
    @test bravais() isa Bravais{0,0,Float64,0}
    @test bravais((1, 2), (3, 3)) isa Bravais{2,2,Int,4}
    @test bravais(@SMatrix[1.0 2; 3 3]) isa Bravais{2,2,Float64,4}
    @test bravais((1,0), semibounded = false) isa Bravais{2,1,Int,2}
end

@testset "sublat" begin
    sitelist = [(3,3), (3,3.), [3,3.], @SVector[3, 3], @SVector[3, 3f0], @SVector[3f0, 3.]]
    for site2 in sitelist, site1 in sitelist
        @test sublat(site1, site2) isa
            Sublat{2,promote_type(typeof.(site1)..., typeof.(site2)...)}
    end
    @test sublat((3,)) isa Sublat{1,Int}
    @test sublat(()) isa Sublat{0,Float64}
end

@testset "lattice" begin
    s = sublat((1, 2))
    for t in (Float32, Float64), e in 1:4, l = 1:4
        b = bravais(ntuple(_ -> (1,), l)...)
        @test lattice(b, s, type = t, dim = Val(e)) isa Lattice{e,min(l,e),t}
        @test lattice(b, s, type = t, dim = e) isa Lattice{e,min(l,e),t}
    end
end

@testset "lattice presets" begin
    a0s = (1, 2)
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    for a0 in a0s, s in (true, false), t in (Float32, Float64), e in 1:4, preset in presets
        @test preset(; a0 = a0, semibounded = s, type = t, dim = e) isa Lattice{e,<:Any,t}
    end
end

@testset "unitcell" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    Random.seed!(1234)
    for preset in presets
        lat = preset()
        E, L = dims(lat)
        for l in 1:L
            svecs = ntuple(i -> ntuple(j -> rand(1:5) , Val(E)), L-l)
            @test unitcell(preset(), svecs...) isa Lattice{E,L-l}
            @test unitcell(preset(), l) isa Lattice{E,L}
        end
    end
    @test unitcell(LatticePresets.honeycomb(), region = RegionPresets.circle(10)) isa Lattice{2,0}
    @test unitcell(LatticePresets.honeycomb(), (2,1), region = RegionPresets.circle(10)) isa Lattice{2,1}
    @test unitcell(LatticePresets.bcc(), (2,1,0), region = RegionPresets.circle(10)) isa Lattice{3,1}
    @test unitcell(LatticePresets.cubic(), (2,1,0), region = RegionPresets.sphere(10)) isa Lattice{3,1}
end

@testset "supercell" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    Random.seed!(1234)
    for preset in presets
        lat = preset()
        E, L = dims(lat)
        for l in 1:L
            svecs = ntuple(i -> ntuple(j -> rand(1:5) , Val(E)), L-l)
            @test supercell(preset(), svecs...) isa Superlattice{E,<:Any,<:Any,L-l}
            @test supercell(preset(), l) isa Superlattice{E,<:Any,<:Any,L}
        end
    end
    @test supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(10)) isa Superlattice{2,2}
    @test supercell(LatticePresets.honeycomb(), (2,1), region = RegionPresets.circle(10)) isa Superlattice{2,2}
    @test supercell(LatticePresets.bcc(), (2,1,0), region = RegionPresets.circle(10)) isa Superlattice{3,3}
    @test supercell(LatticePresets.cubic(), (2,1,0), region = RegionPresets.sphere(10)) isa Superlattice{3,3}
end