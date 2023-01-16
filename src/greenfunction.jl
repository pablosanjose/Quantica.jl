############################################################################################
# greenfunction
#region

greenfunction(s::AbstractGreenSolver) = oh -> greenfunction(oh, s)

greenfunction() = h -> greenfunction(h)

greenfunction(oh::OpenHamiltonian, s::AbstractGreenSolver = default_green_solver(hamiltonian(oh))) =
    GreenFunction(oh, apply(s, oh))

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

call!(g::GreenFunction, ω; params...) =
    call!(solver(g), hamiltonian(g), contacts(g), ω; params...)

call!(g::GreenFunctionSlice, ω; params...) =
    call!(parent(g), ω; params...)[slicerows(g), slicecols(g)]

#endregion

############################################################################################
# GreenFixed indexing
#region

Base.view(g::GreenFixed, i, j = i) =
    view(greencontacts(g), greenfix_inds(i, g), greenfix_inds(j, g))

green_inds(c::ContactIndex, g::GreenFixed) = contactinds(g, contact(c))
green_inds(x, g::GreenFixed) = green_inds(x, greencontacts(g))

Base.getindex(g::GreenFixed) =  copy(view(g))
Base.getindex(g::GreenFixed, i, j) = copy(view(g, i, j))

function Base.getindex(g::GreenFixed, lsrow::LatticeSlice, lscol::LatticeSlice = lsrow)
    lattice(g) == parent(lsrow) == parent(lscol) ||
        argerror("Can only index a GreenFixed over its own Lattice")
    ls = merge(lrow, lcol)

end

#g.solver(sanitize_Green_index(c, lattice(g)), sanitize_Green_index(c´, lattice(g)))

#endregion

############################################################################################
# GreenFunction indexing
#region

Base.getindex(g::GreenFunction, c, c´ = c) =
    GreenFunctionSlice(g, sanitize_Green_index(c, lattice(g)), sanitize_Green_index(c´, lattice(g)))

# function Base.getindex(g::GreenFunction; kw...)
#     ls = lattice(g)[kw...]
#     return GreenFunctionSlice(g, ls, ls)
# end

# overrides the base method above
Base.getindex(g::GreenFunction) = GreenFunctionSlice(g, :, :)

#endregion

############################################################################################
# GreenFixedSolver call API
#region

function (s::GreenFixedSolver)(i::Integer, j::Integer)

end

#endregion

############################################################################################
# conductance
#region

conductance(g::GreenFunctionSlice, ω; params...) = conductance(call!(g, ω; params...))

#endregion