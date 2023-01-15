############################################################################################
# HybridMatrix indexing
#region

Base.view(m::HybridMatrix) = m
Base.view(m::HybridMatrix, i, j) =
    HybridMatrix(view(parent(m), hybrid_inds(m, i), hybrid_inds(m, j)), blockstruct(m))

hybrid_inds(i::Integer, m) = siterange(m, i)
hybrid_inds(i::ContactIndex, m) = siterange(m, i)
hybrid_inds(c::SVector{<:Any,<:Integer}, m) = subcellrange(m, c)
hybrid_inds(c::NTuple{<:Any,<:Integer}, m) = subcellrange(m, c)

Base.size(m::HybridMatrix) = (unflatsize(m), unflatsize(m))

function Base.size(m::HybridMatrix, i::Integer)
    s = if i<1
        @boundscheck(boundserror(m, i))
    elseif i<=2
        unflatsize(m)
    else
        1
    end
    return s
end

Base.getindex(m::HybridMatrix, i...) = copy(view(m, i...))

Base.setindex!(m::HybridMatrix, val, i...) = (view(m, i...) .= val)

function Base.setindex!(m::HybridMatrix, val::UniformScaling, i...)
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
# call API
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
# GreenMatrix indexing
#region

Base.view(g::GreenMatrix) = view(greencontacts(g))
Base.view(g::GreenMatrix, ::Missing, ::Missing) = view(greencontacts(g))
Base.view(g::GreenMatrix, i, j) = view(greencontacts(g), i, j)

Base.getindex(g::GreenMatrix) =  copy(view(g))
Base.getindex(g::GreenMatrix, i, j) = copy(view(g, i, j))

function Base.getindex(g::GreenMatrix, lsrow::LatticeSlice, lscol::LatticeSlice = lsrow)
    lattice(g) == parent(lsrow) == parent(lscol) ||
        argerror("Can only index a GreenMatrix over its own Lattice")
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
Base.getindex(g::GreenFunction) = GreenFunctionSlice(g, missing, missing)

#endregion

############################################################################################
# GreenMatrixSolver call API
#region

function (s::GreenMatrixSolver)(i::Integer, j::Integer)

end

#endregion

############################################################################################
# conductance
#region

conductance(g::GreenFunctionSlice, ω; params...) = conductance(call!(g, ω; params...))

#endregion