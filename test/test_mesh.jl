using Quantica: nvertices, nedges, nsimplices

@testset "meshes" begin
    m = cuboid((0,1), (0,2), subticks = (10, 20))
    @test nvertices(m) == 200
    @test_throws MethodError cuboid((SA[1,2], SA[2,3]))
end