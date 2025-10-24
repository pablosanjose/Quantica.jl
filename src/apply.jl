
# apply: takes generic user input (model/selector/modifier/etc), and specializes it
# to a given object (Lattice/Hamiltonian/etc), performing some preprocessing task on the
# input that allows to use it on that object (it gets transformed into an AppliedInput)
# Example: a HopSelector input with a `range = neighbors(1)` gets applied onto a lattice
# by computing the actual range for nearest neighbors in that lattice.

############################################################################################
# apply selector
#region

apply(s::Union{SiteSelector,HopSelector}, l::LatticeSlice) = apply(s, parent(l), cells(l))

function apply(s::SiteSelector, lat::Lattice{T,E,L}, cells...) where {T,E,L}
    region = applied_region(s.region)
    intsublats = recursive_apply(name -> sublatindex_or_zero(lat, name), s.sublats)
    sublats = recursive_push!(Int[], intsublats)
    cells, cellsf = applied_cells!(SVector{L,Int}[], sanitize_cells(s.cells, Val(L)), cells...)
    unique!(sort!(sublats))
    # we don't sort cells, in case we have received them as an explicit list
    unique!(cells)
    # isnull: to distinguish in a type-stable way between s.cells::Union{Missing,Function}
    # and no-selected-cells; and the same for sublats
    isnull = isnull_sel(s.cells, cells) || isnull_sel(s.sublats, sublats)
    return AppliedSiteSelector{T,E,L}(lat, region, sublats, cells, cellsf, isnull)
end

function apply(s::HopSelector, lat::Lattice{T,E,L}, cells...) where {T,E,L}
    rmin, rmax = sanitize_minmaxrange(s.range, lat)
    L > 0 && s.dcells === missing && rmax === missing &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    flipdr = ifelse(s.adjoint, -1, 1)
    region = applied_region(flipdr, s.region)
    intsublats = recursive_apply(names -> sublatindex_or_zero(lat, names), s.sublats)
    sublats = recursive_push!(Pair{Int,Int}[], intsublats)
    dcells, dcellsf = applied_cells!(SVector{L,Int}[], sanitize_cells(s.dcells, Val(L)), cells...)
    unique!(sublats)
    unique!(dcells)
    if s.adjoint
        sublats .= reverse.(sublats)
        dcells .*= -1
    end
    includeonsite = s.includeonsite
    # isnull: see above
    isnull = isnull_sel(s.dcells, dcells) || isnull_sel(s.sublats, sublats)
    return AppliedHopSelector{T,E,L}(lat, region, sublats, dcells, dcellsf, (rmin, rmax), includeonsite, isnull)
end

sublatindex_or_zero(lat, ::Missing) = missing
sublatindex_or_zero(lat, i::Integer) = ifelse(1 <= i <= nsublats(lat), Int(i), 0)

function sublatindex_or_zero(lat, name::Symbol)
    i = sublatindex_or_nothing(lat, name)
    return sublatindex_or_zero(i)
end

sublatindex_or_zero(i::Number) = Int(i)
sublatindex_or_zero(_) = 0

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

padrange(r::Real, m) = isfinite(r) ? float(r) + m * sqrt(eps(float(r))) : float(r)

applied_region(::Missing) = Returns(true)
applied_region(region::Function) = r -> ifelse(region(r), true, false)
applied_region(flipdr, ::Missing) = Returns(true)
applied_region(flipdr, region::Function) = (r, dr) -> ifelse(region(r, flipdr*dr), true, false)

# returns filled v::Vector, cellsf::Function (latter is Returns(true) by default)
# if cells is a function, we return an empty v, and keep cellsf = cells for later evaluation
applied_cells!(v, cells) = (recursive_push!(v, cells), Returns(true))
applied_cells!(v, fcells::Function) = (v, fcells)
applied_cells!(v, _, cells) = (copyto!(v, cells),  Returns(true))

function applied_cells!(v, fcells::Function, cells)
    for cell in cells
        fcells(cell) && push!(v, cell)
    end
    return (v,  Returns(true))
end

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

applyrange(ss::Tuple, h::AbstractHamiltonian) = applyrange.(ss, Ref(h))
applyrange(s::Modifier, h::AbstractHamiltonian) = s
applyrange(s::AppliedHoppingModifier, h::ParametricHamiltonian) =
    AppliedHoppingModifier(s, applyrange(selector(s), lattice(h)))
applyrange(s::HopSelector, lat::Lattice) = hopselector(s; range = sanitize_minmaxrange(hoprange(s), lat))

applyrange(r::Neighbors, lat::Lattice) = nrange(Int(r), lat)
applyrange(r::Real, lat::Lattice) = r

isnull_sel(::Union{Missing,Function}, _) = false
isnull_sel(_, list) = isempty(list)

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
    ParametricOnsiteTerm(functor(t), apply(selector(t), lat), coefficient(t), is_spatial(t))

apply(t::ParametricHoppingTerm, lat::Lattice) =
    ParametricHoppingTerm(functor(t), apply(selector(t), lat), coefficient(t), is_spatial(t))

apply(m::ParametricModel, lat) = ParametricModel(apply.(terms(m), Ref(lat)))

#endregion

############################################################################################
# apply parametric modifiers
#   shifts allows to transform lattice(h) sites into the sites of some originating lattice
#   unit cell: shifts = [bravais * dn] for each site in lattice(h)
#   shifts is useful for supercell, where we want to keep the r, dr of original lat
#region

apply(m::BlockModifier, h::Hamiltonian, shifts = missing) =
    apply(parent(m), h, shifts, block(m))

function apply(m::OnsiteModifier, h::Hamiltonian, shifts = missing, block = missing)
    f = parametric_function(m)
    sel = selector(m)
    asel = apply(sel, lattice(h), ()) # no selected cells if cells::Function
    ptrs = modifier_pointers(h, asel, shifts, block)
    B = blocktype(h)
    spatial = is_spatial(m)
    return AppliedOnsiteModifier(sel, B, f, ptrs, spatial)
end

function apply(m::HoppingModifier, h::Hamiltonian, shifts = missing, block = missing)
    f = parametric_function(m)
    sel = selector(m)
    dcells = dcell.(harmonics(h))
    asel = apply(sel, lattice(h), dcells) # we pass available dcells, in case sel is unbounded
    ptrs = modifier_pointers(h, asel, shifts, block)
    B = blocktype(h)
    spatial = is_spatial(m)
    return AppliedHoppingModifier(sel, B, f, ptrs, spatial)
end

function modifier_pointers(h::Hamiltonian{T,E,L,B}, s::AppliedSiteSelector{T,E,L}, shifts = missing, block = missing) where {T,E,L,B}
    isempty(cells(s)) || iszero(only(cells(s))) || argerror("Cannot constrain to nonzero cells in an onsite modifier, cell periodicity is assumed.")
    ptrs = Tuple{Int,SVector{E,T},CellSitePos{T,E,L,B},Int}[]
    har0 = first(harmonics(h))
    return push_pointers!(ptrs, h, har0, s, shifts, block)
end

function modifier_pointers(h::Hamiltonian{T,E,L,B}, s::AppliedHopSelector{T,E,L}, shifts = missing, block = missing) where {T,E,L,B}
    hars = harmonics(h)
    harptrs = [Tuple{Int,SVector{E,T},SVector{E,T},CellSitePos{T,E,L,B},CellSitePos{T,E,L,B},Tuple{Int,Int}}[] for _ in hars]
    for (har, ptrs) in zip(hars, harptrs)
        push_pointers!(ptrs, h, har, s, shifts, block)
    end
    return harptrs
end

function push_pointers!(ptrs, h, har0, s::AppliedSiteSelector, shifts = missing, block = missing)
    isnull(s) && return ptrs
    dn0 = dcell(har0)
    iszero(dn0) || return ptrs
    B = blocktype(h)
    lat = lattice(h)
    umat = unflat(har0)
    rows = rowvals(umat)
    norbs = norbitals(h, :)
    for scol in sublats(lat), col in siterange(lat, scol)
        isinblock(col, block) || continue
        for p in nzrange(umat, col)
            row = rows[p]
            col == row || continue
            r = site(lat, col)
            r = apply_shift(shifts, r, col)
            if (scol, r) in s
                n = norbs[scol]
                sp = CellSitePos(dn0, col, r, B)
                tup = (p, r, sp, n)
                push_pointer!(ptrs, tup, h, har0, (row, col))
            end
        end
    end
    return ptrs
end

function push_pointers!(ptrs, h, har, s::AppliedHopSelector, shifts = missing, block = missing)
    isnull(s) && return ptrs
    B = blocktype(h)
    lat = lattice(h)
    dn0 = zerocell(lat)
    dn = dcell(har)
    norbs = norbitals(h, :)
    umat = unflat(har)
    rows = rowvals(umat)
    for scol in sublats(lat), col in siterange(lat, scol), p in nzrange(umat, col)
        row = rows[p]
        isinblock(row, col, block) || continue
        srow = sitesublat(lat, row)
        rcol = site(lat, col, dn0)
        rrow = site(lat, row, dn)
        r, dr = rdr(rcol => rrow)
        r = apply_shift(shifts, r, col)
        if (col => row, scol => srow, (r, dr), dn) in s
            ncol = norbs[scol]
            nrow = norbs[srow]
            sprow, spcol = CellSitePos(dn, row, rrow, B), CellSitePos(dn0, col, rcol, B)
            tup = (p, r, dr, sprow, spcol, (nrow, ncol))
            # push_pointer! may be extended for various ptrs types
            push_pointer!(ptrs, tup, h, har, (row, col))
        end
    end
    return ptrs
end

push_pointer!(ptrs::Vector{T}, tup::T, _...) where {T<:Tuple} = push!(ptrs, tup)

apply_shift(::Missing, r, _) = r
apply_shift(shifts, r, i) = r - shifts[i]

#endregion

############################################################################################
# apply Serializers
#   construct pointer Dictionaries of the form
#   [dn => [(ptr, ptr´, serialrng)...]...] or [dn => [(ptr, serialrng)...]...]
#   depending on whether enc/dec is a tuple of functions (onsite/hopping) or a function
#   ptr, ptr´ are pointers to nonzeros of h[unflat(dn)] and h[unflat(-dn)] respectively
#   serialrng is the index range inside the serialized vector corresponding to the pointer
#region

# we support shifts, for supercell, but not block, for BlockModifiers
function apply(s::Serializer, h::AbstractHamiltonian, shifts = missing)
    ptrs = serializer_pointers(h, encoder(s), selectors(s), shifts)
    len = update_serial_ranges!(ptrs, h, s)
    return AppliedSerializer(s, h, ptrs, len)
end

## serializer_pointers

serializer_pointers(h::ParametricHamiltonian, args...) =
    serializer_pointers(hamiltonian(h), args...)

function serializer_pointers(h::Hamiltonian{<:Any,<:Any,L}, encoder, selectors, shifts) where {L}
    E = serializer_pointer_type(encoder)
    skip_reverse = encoder isa Tuple
    d = Dictionary{SVector{L,Int},Vector{E}}()
    for har in harmonics(h)
        dn = dcell(har)
        if skip_reverse && haskey(d, -dn)
            ptrs = E[]
        else
            ptrs = push_and_merge_pointers!(E[], h, har, shifts, selectors...)
        end
        insert!(d, dn, ptrs)
    end
    return d
end

serializer_pointer_type(::Function) = Tuple{Int,UnitRange{Int}}
serializer_pointer_type(::Tuple{Function,Function}) = Tuple{Int,Int,UnitRange{Int}}

function push_and_merge_pointers!(ptrs, h, har, shifts, sel::Selector, selectors...)
    asel = apply(sel, lattice(h))
    push_pointers!(ptrs, h, har, asel, shifts)
    return push_and_merge_pointers!(ptrs, h, har, shifts, selectors...)
end

push_and_merge_pointers!(ptrs, h, har, shifts) = unique!(sort!(ptrs))

# gets called by push_pointers!, see Modifier section above
function push_pointer!(ptrs::Vector{Tuple{Int,Int,UnitRange{Int}}}, (p, _...), h, har, (row, col))
    dn = dcell(har)
    if row == col && iszero(dn)
        p´ = p
    elseif row > col && iszero(dn)
        return ptrs     # we don't want duplicates
    else
        mat´ = h[unflat(-dn)]
        p´ = sparse_pointer(mat´, (col, row)) # adjoint element
    end
     # we leave the serial range empty (initialized later), as it depends on the encoder
    push!(ptrs, (p, p´, 1:0))
    return ptrs
end

push_pointer!(ptrs::Vector{Tuple{Int,UnitRange{Int}}}, (p, _...), _...) = push!(ptrs, (p, 1:0))

## update_serial_ranges!

function update_serial_ranges!(ptrs, h, s::Serializer)
    enc = encoder(s)
    offset = 0
    for (har, ps) in zip(harmonics(h), ptrs), (i, p) in enumerate(ps)
        len = length(serialize_core(h, har, p, enc))
        ps[i] = (Base.front(p)..., offset+1:offset+len)
        offset += len
    end
    return offset   # we return the total length of the serialized vector
end

#endregion

############################################################################################
# apply AbstractEigenSolver
#region

function apply(solver::AbstractEigenSolver, h::AbstractHamiltonian, ::Type{S}, mapping, transform) where {T<:Real,S<:SVector{<:Any,T}}
    # Some solvers (e.g. ES.LinearAlgebra) only accept certain matrix types
    # so this mat´ could be an alias of the call! output, or an unaliased conversion
    mat´ = ES.input_matrix(solver, h)
    function sfunc(φs, dryrun = false)
        # this branch seems to solve #235 and out-of-order extension loading bugs
        # The errors stem from FunctionWrappers (world-age incompatible?). TODO: Why??
        if dryrun
            eigen = EigenComplex{T}(Complex{T}[], Complex{T}[;;])
        else
            φs´ = apply_map(mapping, φs)      # this can be a FrankenTuple
            mat = call!(h, φs´)
            mat´ === mat || copy!(mat´, mat)
            # mat´ could be dense, while mat is sparse, so if not egal, we copy
            # the solver always receives the type of matrix mat´ declared by ES.input_matrix
            eigen = solver(mat´)
            apply_transform!(eigen, transform)
        end
        return eigen
    end
    sfunc(zero(S), true)  # dryrun to avoid world age bugs
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

function apply_map(mapping, hf::Function, ::Type{S}) where {T,S<:SVector{<:Any,T}}
    function sfunc(φs)
        φs´ = apply_map(mapping, φs)    # can be a FrankenTuple, should be accepted by hf
        mat = hf(φs´)
        return mat
    end
    return FunctionWrapper{SparseMatrixCSC{Complex{T},Int},Tuple{S}}(sfunc)
end

#endregion

############################################################################################
# apply CellSites
#region

apply(c::AnyCellSite, ::Lattice{<:Any,<:Any,L}) where {L} = c

apply(c::CellSites{L,Vector{Int}}, ::Lattice{<:Any,<:Any,L}) where {L} = c

apply(c::CellSites{L,Colon}, l::Lattice{<:Any,<:Any,L}) where {L} =
    CellSites(cell(c), collect(siterange(l)))

apply(c::CellSites{L}, l::Lattice{<:Any,<:Any,L}) where {L} =
    CellSites(cell(c), [i for i in siteindices(c) if i in siterange(l)])

apply(::CellSites{L}, l::Lattice{<:Any,<:Any,L´}) where {L,L´} =
    argerror("Cell sites must have $(L´)-dimensional cell indices")

#endregion
