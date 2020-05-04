#######################################################################
# ParametricHamiltonian
#######################################################################
struct ParametricHamiltonian{P,N,M<:NTuple{N,ElementModifier},D<:NTuple{N,Any},H<:Hamiltonian}
    baseh::H
    h::H
    modifiers::M                   # N modifiers
    ptrdata::D                     # P is an NTuple{N,Vector{Vector{ptrdata}}}, one per harmonic
    allptrs::Vector{Vector{Int}}   # ptrdata may be a nzval ptr, a (ptr,r) or a (ptr, r, dr)
    parameters::NTuple{P,NameType} # allptrs are modified ptrs in each harmonic
end

function Base.show(io::IO, ::MIME"text/plain", pham::ParametricHamiltonian{N}) where {N}
    i = get(io, :indent, "")
    print(io, i, "Parametric")
    show(io, pham.h)
    print(io, i, "\n", "$i  Parameters       : $(parameters(pham))")
end

"""
    parametric(h::Hamiltonian, modifiers::ElementModifier...)

Builds a `ParametricHamiltonian` that can be used to efficiently apply `modifiers` to `h`.
`modifiers` can be any number of `@onsite!(args -> body; kw...)` and `@hopping!(args -> body;
kw...)` transformations, each with a set of parameters `ps` given as keyword arguments of
functions `f = (...; ps...) -> body`. The resulting `ph::ParamtricHamiltonian` can be used
to produced the modified Hamiltonian simply by calling it with those same parameters as
keyword arguments.

Note 1: for sparse `h`, `parametric` only modifies existing onsites and hoppings in `h`,
so be sure to add zero onsites and/or hoppings to `h` if they are originally not present but
you need to apply modifiers to them.

Note 2: `optimize!(h)` is called prior to building the parametric Hamiltonian. This can lead
to extra zero onsites and hoppings being stored in sparse `h`s.

    h |> parametric(modifiers::ElementModifier...)

Function form of `parametric`, equivalent to `parametric(h, modifiers...)`.

# Examples

```jldoctest
julia> ph = LatticePresets.honeycomb() |> hamiltonian(onsite(0) + hopping(1, range = 1/√3)) |>
       unitcell(10) |> parametric(@onsite!((o; μ) -> o - μ))
ParametricHamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 200 × 200
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 600
  Coordination     : 3.0
  Parameters       : (:μ,)

julia> ph(μ = 2)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 200 × 200
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 200
  Hoppings         : 600
  Coordination     : 3.0
```
# See also
    `@onsite!`, `@hopping!`
"""
function parametric(h::Hamiltonian, ts::ElementModifier...)
    ts´ = resolve.(ts, Ref(h.lattice))
    optimize!(h)  # to avoid ptrs getting out of sync if optimize! later
    allptrs = [Int[] for _ in h.harmonics]
    ptrdata = parametric_ptrdata!.(Ref(allptrs), Ref(h), ts´)
    foreach(sort!, allptrs)
    foreach(unique!, allptrs)
    params = parameters(ts...)
    return ParametricHamiltonian(h, copy(h), ts´, ptrdata, allptrs, params)
end

parametric(ts::ElementModifier...) = h -> parametric(h, ts...)

function parametric_ptrdata!(allptrs, h::Hamiltonian{LA,L,M,<:AbstractSparseMatrix}, t::ElementModifier) where {LA,L,M}
    harmonic_ptrdata = empty_ptrdata(h, t)
    lat = h.lattice
    selector = t.selector
    for (har, ptrdata, allptrs_har) in zip(h.harmonics, harmonic_ptrdata, allptrs)
        matrix = har.h
        dn = har.dn
        rows = rowvals(matrix)
        for col in 1:size(matrix, 2), ptr in nzrange(matrix, col)
            row = rows[ptr]
            selected = selector(lat, (row, col), (dn, zero(dn)))
            if selected
                push!(ptrdata, ptrdatum(t, lat, ptr, (row, col), dn))
                push!(allptrs_har, ptr)
            end
        end
    end
    return harmonic_ptrdata
end

# Uniform case, one vector of nzval ptr per harmonic
empty_ptrdata(h, t::UniformModifier)  = [Int[] for _ in h.harmonics]

# Non-uniform case, one vector of (ptr, r, dr) per harmonic
function empty_ptrdata(h, t::OnsiteModifier)
    S = positiontype(h.lattice)
    return [Tuple{Int,S}[] for _ in h.harmonics]
end

function empty_ptrdata(h, t::HoppingModifier)
    S = positiontype(h.lattice)
    return [Tuple{Int,S,S}[] for _ in h.harmonics]
end

# Uniform case
ptrdatum(t::UniformModifier, lat, ptr, (row, col), dn) = ptr

# Non-uniform case
ptrdatum(t::OnsiteModifier, lat, ptr, (row, col), dn) = (ptr, sites(lat)[col])

function ptrdatum(t::HoppingModifier, lat, ptr, (row, col), dn)
    r, dr = _rdr(sites(lat)[col], sites(lat)[row] + bravais(lat) * dn)
    return (ptr, r, dr)
end

function (ph::ParametricHamiltonian)(; kw...)
    checkconsistency(ph, false) # only weak check for performance
    reset_harmonic!.(ph.h.harmonics, ph.baseh.harmonics, ph.allptrs)
    applymodifier_ptrdata!.(Ref(ph.h), ph.modifiers, ph.ptrdata, Ref(values(kw)))
    return ph.h
end

function reset_harmonic!(har, basehar, ptrs)
    nz = nonzeros(har.h)
    onz = nonzeros(basehar.h)
    @simd for ptr in ptrs
        nz[ptr] = onz[ptr]
    end
    return har
end

function applymodifier_ptrdata!(h, modifier, ptrdata, kw)
    for (har, hardata) in zip(h.harmonics, ptrdata)
        nz = nonzeros(har.h)
        @simd for data in hardata  # @simd is valid because ptrs are not repeated
            ptr = first(data)
            args = modifier_args(nz, data)
            val = modifier(args...; kw...)
            nz[ptr] = val
        end
    end
    return h
end

# A negative ptr corresponds to a forced-hermitian element
modifier_args(nz, ptr::Int) = (nz[ptr],)
modifier_args(nz, (ptr, r)::Tuple{Int,SVector}) = (nz[ptr], r)
modifier_args(nz, (ptr, r, dr)::Tuple{Int,SVector,SVector}) = (nz[ptr], r, dr)

function checkconsistency(ph::ParametricHamiltonian, fullcheck = true)
    isconsistent = true
    length(ph.baseh.harmonics) == length(ph.h.harmonics) || (isconsitent = false)
    if fullcheck && isconsistent
        for (basehar, har) in zip(ph.baseh.harmonics, ph.h.harmonics)
            length(nonzeros(har.h)) == length(nonzeros(basehar.h)) || (isconsistent = false; break)
            rowvals(har.h) == rowvals(basehar.h) || (isconsistent = false; break)
            getcolptr(har.h) == getcolptr(basehar.h) || (isconsistent = false; break)
        end
    end
    isconsistent ||
        throw(error("ParametricHamiltonian is not internally consistent, it may have been modified after creation"))
    return nothing
end

# ParametricHamiltonian's are already optimized upon creation
optimize!(ph::ParametricHamiltonian) = ph

"""
    parameters(ph::ParametricHamiltonian)

Return the names of the parameter that `ph` depends on
"""
parameters(ph::ParametricHamiltonian) = ph.parameters

# This is not inferred, but it is only needed when calling parametric
parameters(ms::ElementModifier...) = mergetuples(parameters.(ms)...)

matrixtype(ph::ParametricHamiltonian) = matrixtype(parent(ph))

blockeltype(ph::ParametricHamiltonian) = blockeltype(parent(ph))

bravais(ph::ParametricHamiltonian) = bravais(ph.hamiltonian.lattice)

Base.parent(ph::ParametricHamiltonian) = ph.h

Base.copy(ph::ParametricHamiltonian) =
    ParametricHamiltonian(copy(ph.baseh), copy(ph.h), ph.modifiers, copy(h.ptrdata), copy(h.allptrs))

Base.size(ph::ParametricHamiltonian, n...) = size(ph.h, n...)

Base.eltype(ph::ParametricHamiltonian) = eltype(ph.h)

#######################################################################
# bloch! for parametric
#######################################################################

bloch(ph::ParametricHamiltonian, args...) = bloch!(similarmatrix(ph), ph, args...)

bloch!(matrix, ph::ParametricHamiltonian, pϕs = (), axis = 0) =
    bloch!(matrix, h_phases(ph, toSVector(pϕs))..., axis)

@inline function h_phases(ph::ParametricHamiltonian, pϕs)
    pnames = parameters(ph)
    ps, ϕs = extract_parameters_phases(pnames, pϕs)
    h = ph(; ps...)
    return (h, ϕs)
end

extract_parameters_phases(pnames::NTuple{N,NameType}, ϕs::SVector{M}) where {N,M} =
    (NamedTuple{pnames}(ntuple(i->ϕs[i], Val(N))), ntuple(i->ϕs[i+N], Val(M-N)))