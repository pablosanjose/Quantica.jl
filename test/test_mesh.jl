using Quantica: nvertices, nedges, nsimplices

@testset "meshes" begin
    mesh = buildmesh(marchingmesh((0,1), (0,2), points = (10, 20)))
    @test nvertices(mesh) == 200 && nedges(mesh) == 541 && nsimplices(mesh) == 342
    @test_throws ArgumentError marchingmesh((SA[1,2], SA[2,3]))
end