using Quantica: nvertices, nedges, nsimplices

@test begin
    mesh = marchingmesh(0:0.1:1, 1:0.1:2)
    nvertices(mesh) == 121 && nedges(mesh) == 320 && nsimplices(mesh) == 200
end