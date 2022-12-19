############################################################################################
# GreenFunction and GreenBlock indexing
#region

Base.getindex(g::GreenFunction; kw...) = g[siteselector(; kw...)]

function Base.getindex(g::GreenFunction, s::SiteSelector)
    latblock = lattice(g)[s]
    asolver = apply(solver(g), hamiltonian(g), latblock, boundaries(g))
    return GreenBlock(asolver, (latblock,), g)
end

Base.getindex(g::GreenBlock; kw...) = g[siteselector(; kw...)]

Base.getindex(g::GreenBlock{<:GreenFunction}, s::SiteSelector) = parent(g)[s]

function Base.getindex(g::GreenBlock, s::SiteSelector)
    s.cells === missing || argerror("Cannot select cells when indexing this GreenBlock")
    indexlist = Int[]
    # this call populates indexlist with the selected latblock indices
    latblocks´ = getindex.(latblocks(g), Ref(s), Ref(indexlist))
    solver = GS.BlockView(indexlist, g)
    g´ = GreenBlock(solver, latblocks´, g)
    return g´
end

#region