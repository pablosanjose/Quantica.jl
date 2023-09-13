
# apply: takes generic user input (model/selector/modifier/etc), and specializes it
# to a given object (Lattice/Hamiltonian/etc), performing some preprocessing task on the
# input that allows to use it on that object (it gets transformed into an AppliedInput)
# Example: a HopSelector input with a `range = neighbors(1)` gets applied onto a lattice
# by computing the actual range for nearest neighbors in that lattice.

############################################################################################
# apply selector
#region

function apply(s::SiteSelector, lat::Lattice{T,E,L}) where {T,E,L}
    region = r -> applied_region(r, s.region)
    intsublats = recursive_apply(name -> sublatindex_or_zero(lat, name), s.sublats)
    sublats = recursive_push!(Int[], intsublats)
    cells = recursive_push!(SVector{L,Int}[], sanitize_cells(s.cells, Val(L)))
    unique!(sort!(sublats))
    unique!(sort!(cells))
    return AppliedSiteSelector{T,E,L}(lat, region, sublats, cells)
end

function apply(s::HopSelector, lat::Lattice{T,E,L}) where {T,E,L}
    rmin, rmax = sanitize_minmaxrange(s.range, lat)
    L > 0 && s.dcells === missing && rmax === missing &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    sign = ifelse(s.adjoint, -1, 1)
    region = (r, dr) -> applied_region((r, sign*dr), s.region)
    intsublats = recursive_apply(names -> sublatindex_or_zero(lat, names), s.sublats)
    sublats = recursive_push!(Pair{Int,Int}[], intsublats)
    dcells = recursive_push!(SVector{L,Int}[], sanitize_cells(s.dcells, Val(L)))
    unique!(sublats)
    unique!(dcells)
    if s.adjoint
        sublats .= reverse.(sublats)
        dcells .*= -1
    end
    return AppliedHopSelector{T,E,L}(lat, region, sublats, dcells, (rmin, rmax))
end

sublatindex_or_zero(lat, ::Missing) = missing
sublatindex_or_zero(lat, i::Integer) = ifelse(1 <= i <= nsublats(lat), Int(i), 0)

function sublatindex_or_zero(lat, name::Symbol)
    i = sublatindex(lat, name)
    return ifelse(i === nothing, 0, Int(i))
end

sanitize_minmaxrange(r, lat) = sanitize_minmaxrange((zero(numbertype(lat)), r), lat)
sanitize_minmaxrange((rmin, rmax)::Tuple{Any,Any}, lat) =
    padrange(applyrange(rmin, lat), -1), padrange(applyrange(rmax, lat), 1)

sanitize_cells(cell::Number, ::Val{1}) = (sanitize_SVector(SVector{1,Int}, cell),)
sanitize_cells(cell::Union{NTuple{L,<:Integer},SVector{L,<:Number}}, ::Val{L´}) where {L,L´} =
    argerror("Dimension $L of `cells` does not match lattice dimension $(L´)")
sanitize_cells(cell::Union{NTuple{L,<:Integer},SVector{L,<:Number}}, ::Val{L}) where {L} =
    (sanitize_SVector(SVector{L,Int}, cell),)
sanitize_cells(::Missing, ::Val{L}) where {L} = missing
sanitize_cells(f::Function, ::Val{L}) where {L} = f
sanitize_cells(cells, ::Val{L}) where {L} = sanitize_SVector.(SVector{L,Int}, cells)

applyrange(r::Neighbors, lat) = nrange(Int(r), lat)
applyrange(r::Real, lat) = r

padrange(r::Real, m) = isfinite(r) ? float(r) + m * sqrt(eps(float(r))) : float(r)

applied_region(r, ::Missing) = true
applied_region((r, dr)::Tuple{SVector,SVector}, region::Function) = ifelse(region(r, dr), true, false)
applied_region(r::SVector, region::Function) = ifelse(region(r), true, false)

recursive_apply(f, t::Tuple) = recursive_apply.(f, t)
recursive_apply(f, t::AbstractVector) = recursive_apply.(f, t)
recursive_apply(f, (a,b)::Pair) = recursive_apply(f, a) => recursive_apply(f, b)
recursive_apply(f, x) = f(x)

recursive_push!(v::Vector, ::Missing) = v
recursive_push!(v::Vector{T}, x::T) where {T} = push!(v, x)
recursive_push!(v::Vector{S}, x::NTuple{<:Any,Integer}) where {S<:SVector} = push!(v, S(x))
recursive_push!(v::Vector{S}, x::Number) where {S<:SVector{1}} = push!(v, S(x))
recursive_push!(v::Vector{Pair{T,T}}, x::T) where {T} = push!(v, x => x)
recursive_push!(v::Vector{Pair{T,T}}, (x, y)::Tuple{T,T}) where {T} = push!(v, x => y)
recursive_push!(v::Vector{Pair{T,T}}, x::Pair{T,T}) where {T} = push!(v, x)

function recursive_push!(v::Vector, xs)
    foreach(x -> recursive_push!(v, x), xs)
    return v
end

function recursive_push!(v::Vector{Pair{T,T}}, (xs, ys)::Pair) where {T}
    for (x, y) in Iterators.product(xs, ys)
        push!(v, x => y)
    end
    return v
end

# for cells = function
function recursive_push!(v::Vector{SVector{L,Int}}, fcell::Function) where {L}
    iter = BoxIterator(zero(SVector{L,Int}))
    for cell in iter
        fcell(cell) || continue
        acceptcell!(iter, cell)
        push!(v, cell)
    end
    return v
end

#endregion

############################################################################################
# apply model terms
#region

function apply(o::OnsiteTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalBlockStructure{B}}) where {T,E,L,B}
    f = (r, orbs) -> mask_block(B, o(r), (orbs, orbs))
    asel = apply(selector(o), lat)
    return AppliedOnsiteTerm{T,E,L,B}(f, asel)         # f gets wrapped in a FunctionWrapper
end

function apply(t::HoppingTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalBlockStructure{B}}) where {T,E,L,B}
    f = (r, dr, orbs) -> mask_block(B, t(r, dr), orbs)
    asel = apply(selector(t), lat)
    return AppliedHoppingTerm{T,E,L,B}(f, asel)        # f gets wrapped in a FunctionWrapper
end

apply(m::TightbindingModel, latos) = TightbindingModel(apply.(terms(m), Ref(latos)))

apply(t::ParametricOnsiteTerm, lat::Lattice) =
    ParametricOnsiteTerm(functor(t), apply(selector(t), lat), coefficient(t))

apply(t::ParametricHoppingTerm, lat::Lattice) =
    ParametricHoppingTerm(functor(t), apply(selector(t), lat), coefficient(t))

apply(m::ParametricModel, lat) = ParametricModel(apply.(terms(m), Ref(lat)))

#endregion

############################################################################################
# apply parametric modifiers
#   shifts allows to transform lattice(h) sites into the sites of some originating lattice
#   unit cell: shifts = [bravais * dn] for each site in lattice(h)
#   shifts is useful for supercell, where we want to keep the r, dr of original lat
#region

function apply(m::OnsiteModifier, h::Hamiltonian, shifts = missing)
    f = parametric_function(m)
    sel = selector(m)
    asel = apply(sel, lattice(h))
    ptrs = pointers(h, asel, shifts)
    B = blocktype(h)
    return AppliedOnsiteModifier(sel, B, f, ptrs)
end

function apply(m::HoppingModifier, h::Hamiltonian, shifts = missing)
    f = parametric_function(m)
    sel = selector(m)
    asel = apply(sel, lattice(h))
    ptrs = pointers(h, asel, shifts)
    B = blocktype(h)
    return AppliedHoppingModifier(sel, B, f, ptrs)
end

function pointers(h::Hamiltonian{T,E}, s::AppliedSiteSelector{T,E}, shifts) where {T,E}
    isempty(cells(s)) || argerror("Cannot constrain cells in an onsite modifier, cell periodicity is assumed.")
    ptr_r = Tuple{Int,SVector{E,T},Int}[]
    lat = lattice(h)
    har0 = first(harmonics(h))
    umat = unflat(har0)
    rows = rowvals(umat)
    norbs = norbitals(h)
    for scol in sublats(lat), col in siterange(lat, scol), p in nzrange(umat, col)
        row = rows[p]
        col == row || continue
        r = site(lat, col)
        r = apply_shift(shifts, r, col)
        if (scol, r) in s
            n = norbs[scol]
            push!(ptr_r, (p, r, n))
        end
    end
    return ptr_r
end

function pointers(h::Hamiltonian{T,E}, s::AppliedHopSelector{T,E}, shifts) where {T,E}
    hars = harmonics(h)
    ptr_r_dr = [Tuple{Int,SVector{E,T},SVector{E,T},Tuple{Int,Int}}[] for _ in hars]
    lat = lattice(h)
    dn0 = zerocell(lat)
    norbs = norbitals(h)
    for (har, ptr_r_dr) in zip(hars, ptr_r_dr)
        mh = unflat(har)
        rows = rowvals(mh)
        for scol in sublats(lat), col in siterange(lat, scol), p in nzrange(mh, col)
            row = rows[p]
            srow = sitesublat(lat, row)
            dn = dcell(har)
            rcol = site(lat, col, dn0)
            rrow = site(lat, row, dn)
            r, dr = rdr(rcol => rrow)
            r = apply_shift(shifts, r, col)
            if (scol => srow, (r, dr), dn) in s
                ncol = norbs[scol]
                nrow = norbs[srow]
                push!(ptr_r_dr, (p, r, dr, (nrow, ncol)))
            end
        end
    end
    return ptr_r_dr
end

apply_shift(::Missing, r, _) = r
apply_shift(shifts, r, i) = r - shifts[i]

#endregion

############################################################################################
# apply AbstractEigenSolver
#region

function apply(solver::AbstractEigenSolver, h::AbstractHamiltonian, ::Type{S}, mapping, transform) where {T<:Real,S<:SVector{<:Any,T}}
    h´ = minimal_callsafe_copy(h)
    # Some solvers (e.g. ES.LinearAlgebra) only accept certain matrix types
    # so this mat´ could be an alias of the call! output, or an unaliased conversion
    mat´ = ES.input_matrix(solver, h)
    function sfunc(φs)
        φs´ = apply_map(mapping, φs)    # this can be a FrankenTuple
        mat = call!(h´, φs´)
        mat´ === mat || copy!(mat´, mat)
        # the solver always receives the matrix type declared by ES.input_matrix
        eigen = solver(mat´)
        apply_transform!(eigen, transform)
        return eigen
    end
    asolver = AppliedEigenSolver(FunctionWrapper{EigenComplex{T},Tuple{S}}(sfunc))
    return asolver
end

function apply(solver::AbstractEigenSolver, hf::Function, ::Type{S}, mapping, transform) where {T<:Real,S<:SVector{<:Any,T}}
    function sfunc(φs)
        φs´ = apply_map(mapping, φs)    # can be a FrankenTuple, should be accepted by hf
        mat = hf(φs´)
        eigen = solver(mat)
        apply_transform!(eigen, transform)
        return eigen
    end
    asolver = AppliedEigenSolver(FunctionWrapper{EigenComplex{T},Tuple{S}}(sfunc))
    return asolver
end

apply(solver::AbstractEigenSolver, h, S, mapping, transform) =
    argerror("Encountered an unexpected type as argument to an eigensolver. Are your mesh vertices real?")

apply_transform!(eigen, ::Missing) = eigen

function apply_transform!(eigen, transform)
    ϵs = first(eigen)
    map!(transform, ϵs, ϵs)
    return eigen
end

apply_map(::Missing, φs) = φs
apply_map(mapping, φs) = mapping(Tuple(φs)...)

function apply_map(mapping, h::AbstractHamiltonian{T}, ::Type{S}) where {T,S<:SVector}
    function sfunc(φs)
        h´ = minimal_callsafe_copy(h)
        φs´ = apply_map(mapping, φs)    # can be a FrankenTuple
        mat = call!(h´, φs´)
        return mat
    end
    return FunctionWrapper{SparseMatrixCSC{Complex{T},Int},Tuple{S}}(sfunc)
end

function apply_map(mapping, hf::Function, ::Type{S}) where {T,S<:SVector{T}}
    function sfunc(φs)
        φs´ = apply_map(mapping, φs)    # can be a FrankenTuple, should be accepted by hf
        mat = hf(φs´)
        return mat
    end
    return FunctionWrapper{SparseMatrixCSC{Complex{T},Int},Tuple{S}}(sfunc)
end


#endregion

#endregion
