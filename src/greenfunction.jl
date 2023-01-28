############################################################################################
# greenfunction
#region

greenfunction(s::AbstractGreenSolver) = oh -> greenfunction(oh, s)

greenfunction() = h -> greenfunction(h)

function greenfunction(oh::OpenHamiltonian, s::AbstractGreenSolver = default_green_solver(hamiltonian(oh)))
    cs = Contacts(oh)
    h = hamiltonian(oh)
    return GreenFunction(h, apply(s, oh, cs), cs)
end

default_green_solver(::AbstractHamiltonian0D) = GS.SparseLU()
# default_green_solver(::AbstractHamiltonian1D) = GS.Schur()
# default_green_solver(::AbstractHamiltonian) = GS.Bands()
default_green_solver(::AbstractHamiltonian) = GS.NoSolver()

#endregion

############################################################################################
# GreenFunction call API
#region

## TODO: test copy(g) for aliasing problems
(g::GreenFunction)(; params...) = minimal_callsafe_copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = minimal_callsafe_copy(call!(g, ω; params...))

function call!(g::GreenFunction; params...)
    h´ = call!(hamiltonian(g); params...)
    solver´ = call!(solver(g); params...)
    contacts´ = call!(contacts(g); params...)
    return GreenFunction(h´, solver´, contacts´)
end

function call!(g::GreenFunction, ω; params...)
    h = parent(g)
    cs = contacts(g)
    call!(h, (); params...)               # call!_output is wrapped in solver(g) - update it
    Σs = call!(cs, ω; params...)                                        # same for Σs blocks
    cbs = blockstructure(cs)
    slicer = solver(g)(ω, Σs, cbs)
    return GreenSolution(h, slicer, Σs, cbs)
end

#endregion

############################################################################################
# GreenSolution indexing
#region

# green_inds(c::ContactIndex, _) = c
# green_inds(s::LatticeSlice, g) = orbslice(s, hamiltonian(g))
# green_inds(s::CellSites, g) = orbslice(s, hamiltonian(g))
# green_inds(kw::NamedTuple, g) = orbslice(getindex(lattice(g); kw...), hamiltonian(g))
# green_inds(cell::NTuple{<:Any,Int}, g) = orbslice(cellsites(cell, :), hamiltonian(g))

# Base.getindex(g::GreenSolution, i, j) = getindex(g, green_inds(i, g), green_inds(j, g))

# Base.getindex(g::GreenSolution; kw...) = g[getindex(lattice(g); kw...)]

# function Base.getindex(g::GreenSolution, i)
#     ai = green_inds(i, g)
#     return getindex(g, ai, ai)
# end

# Base.getindex(g::GreenSolution, i::LatticeSlice, j::ContactIndex) =
#     mortar([g[si, j] for si in subcells(i), _ in 1:1])
# Base.getindex(g::GreenSolution, i::ContactIndex, j::LatticeSlice) =
#     mortar([g[i, sj] for _ in 1:1, sj in subcells(j)])
# Base.getindex(g::GreenSolution, i::LatticeSlice, j::LatticeSlice) =
#     mortar([g[si, sj] for si in subcells(i), sj in subcells(j)])
# Base.getindex(g::GreenSolution, i::ContactIndex, j::ContactIndex) = copy(view(g, i, j))

Base.view(g::GreenSolution, i::ContactIndex, j::ContactIndex = i) = view(slicer(g), i, j)
Base.view(g::GreenSolution, i::Colon, j::Colon = i) = view(slicer(g), i, j)
Base.getindex(g::GreenSolution, i::ContactIndex, j::ContactIndex = i) = copy(view(g, i, j))
Base.getindex(g::GreenSolution, ::Colon, ::Colon = :) = copy(view(g, :, :))

Base.getindex(g::GreenSolution; kw...) = g[(; kw...)]

function Base.getindex(g::GreenSolution, i)
    ai = ind_to_slice(i, g)
    return getindex(g, ai, ai)
end

Base.getindex(g::GreenSolution, i, j) = getindex(g, ind_to_slice(i, g), ind_to_slice(j, g))

Base.getindex(g::GreenSolution, i::OrbitalSlice, j::OrbitalSlice) =
    mortar([g[si, sj] for si in subcells(i), sj in subcells(j)])

function ind_to_slice(c::ContactIndex, g)
    contactbs = blockstructure(g)
    cinds = contactinds(contactbs, Int(c))
    os = orbslice(contactbs)[cinds]
    return os
end

ind_to_slice(c::CellSites, g) = orbslice(c, hamiltonian(g))
ind_to_slice(l::LatticeSlice, g) = orbslice(l, hamiltonian(g))
ind_to_slice(kw::NamedTuple, g) = ind_to_slice(getindex(lattice(g); kw...), g)

Base.getindex(g::GreenSolution, i::CellOrbitals, j::CellOrbitals) = slicer(g)[i, j]

Base.getindex(s::GreenSlicer, ::CellOrbitals, ::CellOrbitals) =
    internalerror("getindex of $(nameof(typeof(s))): not implemented")

#endregion

############################################################################################
# GreenFunction indexing
#region

# Base.getindex(g::GreenFunction, c, c´ = c) =
#     GreenFunctionSlice(g, sanitize_Green_index(c, lattice(g)), sanitize_Green_index(c´, lattice(g)))

# # function Base.getindex(g::GreenFunction; kw...)
# #     ls = lattice(g)[kw...]
# #     return GreenFunctionSlice(g, ls, ls)
# # end

# # overrides the base method above
# Base.getindex(g::GreenFunction) = GreenFunctionSlice(g, :, :)

#endregion

############################################################################################
# GreenSlicer call API
#region

# function (s::GreenSlicer)(i::Integer, j::Integer)

# end

#endregion

############################################################################################
# conductance
#region

conductance(g::GreenFunctionSlice, ω; params...) = conductance(call!(g, ω; params...))

#endregion