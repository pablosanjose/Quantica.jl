using Quantica: nvertices, nedges, nsimplices

@testset "meshes" begin
    m = mesh((0,1), (0,2), points = (10, 20))
    @test nvertices(m) == 200 && nedges(m) == 541 && nsimplices(m) == 342
    @test_throws MethodError mesh((SA[1,2], SA[2,3]))
end