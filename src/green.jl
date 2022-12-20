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

############################################################################################
# call API
#region

(g::AbstractGreen)(; params...) = call!(copy_callsafe(g); params...)
(g::AbstractGreen)(ω; params...) = call!(copy_callsafe(g), ω; params...)

copy_callsafe(g::Green) = Green(copy_callsafe(g.h), g.boundaries, g.solver)
copy_callsafe(g::GreenLead) = GreenLead(copy_callsafe(g.solver), g.latblock, g.parent)
copy_callsafe(g::GreenBlock) = GreenBlock(copy_callsafe(g.solver), g.latblock, g.parent)
copy_callsafe(g::GreenBlockInverse) =
    GreenBlockInverse(copy_callsafe(g.solver), g.latblock, g.parent)

call!(g::Green; params...) =
    Green(call!(hamiltonian(g); params...), boundaries(g), solver(g))

call!(g::GreenBlock; params...) =
    GreenBlock(call!(solver(g); params...), latblock(g), parent(g))

call!(g::GreenBlockInverse; params...) =
    GreenBlockInverse(call!(solver(g); params...), latblock(g), parent(g))

call!(l::GreenLead; params...) =
    GreenLead(call!(solver(l); params...), latblock(l), parent(l))

#region