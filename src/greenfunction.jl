############################################################################################
# GreenMatrix indexing
#region

Base.view(m::GreenMatrix, i, j) = view(parent(m), green_inds(i, m), green_inds(j, m))

green_inds(i::Integer, m::GreenMatrix) = siterange(m, i)
green_inds(c::SVector{<:Any,<:Integer}, m::GreenMatrix) = subcellrange(m, c)
green_inds(c::NTuple{<:Any,Integer}, m::GreenMatrix) = subcellrange(m, c)
green_inds(::Colon, m::GreenMatrix) = Colon()
green_inds(xs, m::GreenMatrix) = Interators.flatten(green_inds(x, m) for x in xs)

Base.size(m::GreenMatrix) = (unflatsize(m), unflatsize(m))

function Base.size(m::GreenMatrix, i::Integer)
    s = if i<1
        @boundscheck(boundserror(m, i))
    elseif i<=2
        unflatsize(m)
    else
        1
    end
    return s
end

Base.getindex(m::GreenMatrix, i...) = copy(view(m, i...))

Base.setindex!(m::GreenMatrix, val, i...) = (view(m, i...) .= val)

function Base.setindex!(m::GreenMatrix, val::UniformScaling, i...)
    v = view(m, i...)
    λ = val.λ
    for c in CartesianIndices(v)
        (i, j) = Tuple(c)
        @inbounds v[c] = λ * (i == j)
    end
    return v
end

#endregion

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
(g::GreenFunction)(; params...) = copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = copy(call!(g, ω; params...))

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