using Quantica: nvertices, nedges, nsimplices

@test begin
    mesh = buildmesh(marchingmesh((0,1), (0,2), resolution = (10, 20)))
    nvertices(mesh) == 200 && nedges(mesh) == 541 && nsimplices(mesh) == 342
end