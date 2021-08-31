############################################################################################
# OrbitalStructure constructor
#region

# norbs is a collection of number of orbitals, one per sublattice (or a single one for all)
# surprisingly, O type instability has no performance or allocations penalty
function OrbitalStructure(lat::Lattice, norbs, ::Type{T} = numbertype(lat)) where {T}
    O = blocktype(T, norbs)
    return OrbitalStructure{O}(lat, norbs)
end

function OrbitalStructure{O}(lat::Lattice, norbs) where {O}
    norbs´ = sanitize_Vector_of_Type(Int, nsublats(lat), norbs)
    offsets´ = offsets(lat)
    return OrbitalStructure{O}(O, norbs´, offsets´)
end

blocktype(T::Type, norbs) = blocktype(T, val_maximum(norbs))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

val_maximum(n::Int) = Val(n)
val_maximum(ns::Tuple) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

# Equality does not need equal T
Base.:(==)(o1::OrbitalStructure, o2::OrbitalStructure) =
    o1.norbs == o2.norbs && o1.offsets == o2.offsets

#endregion

############################################################################################
# Hamiltonian constructors
#region

hamiltonian(m::TightbindingModel = TightbindingModel(); kw...) = lat -> hamiltonian(lat, m; kw...)

function hamiltonian(lat::Lattice, m = TightbindingModel(); orbitals = Val(1), type = numbertype(lat))
    orbstruct = OrbitalStructure(lat, orbitals, type)
    builder = IJVBuilder(lat, orbstruct)
    apmod = apply(m, (lat, orbstruct))
    # using foreach here foils precompilation of applyterm! for some reason
    applyterm!.(Ref(builder), terms(apmod))
    hars = harmonics(builder)
    return Hamiltonian(lat, orbstruct, hars)
end

function applyterm!(builder, term::AppliedOnsiteTerm)
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    sel = selector(term)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_site(sel, dn0) do s, i, r
        n = norbs[s]
        v = term(r, n)
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder, term::AppliedHoppingTerm)
    trees = kdtrees(builder)
    sel = selector(term)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_cell(sel) do dn, cell_iter
        ijv = builder[dn]
        foreach_hop!(sel, cell_iter, trees, dn) do (si, sj), (i, j), (r, dr)
            ni = norbs[si]
            nj = norbs[sj]
            v = term(r, dr, (ni, nj))
            push!(ijv, (i, j, v))
        end
    end
    return nothing
end

#endregion

############################################################################################
# ParametricHamiltonian constructors
#region

parametric(modifiers::Modifier...) = h -> parametric(h, modifiers...)

function parametric(hparent::Hamiltonian, modifiers::Modifier...)
    modifiers´ = apply.(modifiers, Ref(hparent))  # PartiallyAppliedModifiers
    allptrs = merge_pointers(hparent, modifiers´...)
    allparams = merge_parameters(modifiers´...)
    h = copy_harmonics(hparent)
    return ParametricHamiltonian(hparent, h, modifiers´, allptrs, allparams)
end

merge_pointers(h, m...) = merge_pointers!([Int[] for _ in harmonics(h)], m...)

function merge_pointers!(p, m::PartiallyAppliedOnsiteModifier, ms...)
    p0 = first(p)
    for (ptr, _) in pointers(m)
        push!(p0, ptr)
    end
    return merge_pointers!(p, ms...)
end

function merge_pointers!(p, m::PartiallyAppliedHoppingModifier, ms...)
    for (pn, pm) in zip(p, pointers(m)), (ptr, _, _) in pm
        push!(pn, ptr)
    end
    return merge_pointers!(p, ms...)
end

function merge_pointers!(p)
    for pn in p
        unique!(sort!(pn))
    end
    return p
end

merge_parameters(m...) = _merge_parameters(Symbol[], m...)
_merge_parameters(p, m, ms...) = _merge_parameters(append!(p, parameters(m)), ms...)
_merge_parameters(p) = unique!(sort!(p))

# Call API #

function (ph::ParametricHamiltonian)(; kw...)
    hparent = parent(ph)
    h = hamiltonian(ph)
    reset_pointers!(ph)
    modifiers´ = apply.(modifiers(ph), Ref(hparent); kw...)
    applymodifiers!(h, modifiers´...)
    return h
end

function reset_pointers!(ph::ParametricHamiltonian)
    h = hamiltonian(ph)
    hparent = parent(ph)
    for (har, har´, ptrs) in zip(harmonics(h), harmonics(hparent), pointers(ph))
        nz = nonzeros(matrix(har))
        nz´ = nonzeros(matrix(har´))
        for ptr in ptrs
            nz[ptr] = nz´[ptr]
        end
    end
    return ph
end

function applymodifiers!(h, m::AppliedOnsiteModifier, ms...)
    nz = nonzeros(matrix(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        nz[ptr] = m(nz[ptr], r, norbs)
    end
    return applymodifiers!(h, ms...)
end

function applymodifiers!(h, m::AppliedHoppingModifier, ms...)
    nz = nonzeros(h)
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(matrix(har))
        for (ptr, r, dr, norbs) in p
            nz[ptr] = m(nz[ptr], r, dr, norbs)
        end
    end
    return applymodifiers!(h, ms...)
end

applymodifiers!(h) = h

#endregion

############################################################################################
# Flat constructors (flatten)
#region

flatten(h::AbstractHamiltonian{<:Any,<:Any,<:Any,<:Number}) = h
flatten(h::Union{Hamiltonian,ParametricHamiltonian}) = FlatHamiltonian(h, flatten(orbitalstructure(h)))
flatten(h::FlatHamiltonian) = h

flatten(os::OrbitalStructure{<:Number}) = os

function flatten(os::OrbitalStructure{<:SMatrix})
    blocktype´ = eltype(blocktype(os))
    norbitals´ = [1 for _ in norbitals(os)]
    flatoffsets´ = flatoffsets(offsets(os), norbitals(os))
    return OrbitalStructure(blocktype´, norbitals´, flatoffsets´)
end

# sublat offsets after flattening (without padding zeros)
function flatoffsets(offsets0, norbs)
    nsites = diff(offsets0)
    nsites´ = norbs .* nsites
    offsets´ = cumsum!(nsites´, nsites´)
    prepend!(offsets´, 0)
    return offsets´
end

function flatten(lat::Lattice, os)
    norbs = norbitals(os)
    sites´ = similar(sites(lat), 0)
    names´ = sublatnames(lat)
    offsets´ = [0]
    for s in sublats(lat)
        norb = norbs[s]
        for r in sites(lat, s), _ in 1:norb
            push!(sites´, r)
        end
        push!(offsets´, length(sites´))
    end
    lat = Lattice(bravais(lat), Unitcell(sites´, names´, offsets´))
    return lat
end

function hamiltonian(f::FlatHamiltonian{<:Any,<:Any,L,O}) where {L,O}
    os = orbitalstructure(parent(f))
    flatos = orbitalstructure(f)
    lat = flatten(lattice(f), os)
    HT = HamiltonianHarmonic{L,O}
    hars = HT[HT(dcell(har), flatten(matrix(har), os, flatos)) for har in harmonics(f)]  # see tools.jl
    return Hamiltonian(lat, flatos, hars)
end

 #endregion