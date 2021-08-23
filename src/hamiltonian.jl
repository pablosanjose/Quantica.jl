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
    flatoffsets´ = flatoffsets(offsets´, norbs´)
    return OrbitalStructure{O}(O, norbs´, offsets´, flatoffsets´)
end

blocktype(T::Type, norbs) = blocktype(T, val_maximum(norbs))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

val_maximum(n::Int) = Val(n)
val_maximum(ns::Tuple) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

# sublat offsets after flattening (without padding zeros)
function flatoffsets(offsets0, norbs)
    nsites = diff(offsets0)
    nsites´ = norbs .* nsites
    offsets´ = cumsum!(nsites´, nsites´)
    prepend!(offsets´, 0)
    return offsets´
 end

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
    applyterm!.(Ref(builder), terms(m))
    hars = harmonics(builder)
    return Hamiltonian(lat, orbstruct, hars)
end

function applyterm!(builder, term::OnsiteTerm)
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    latsel = appliedon(selector(term), lat)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_site(latsel, dn0) do s, i, r
        n = norbs[s]
        v = sanitize_block(blocktype(os), term(r, r), (n, n))
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L}, term::HoppingTerm) where {L}
    lat = lattice(builder)
    trees = kdtrees(builder)
    latsel = appliedon(selector(term), lat)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_cell(latsel) do dn, iter_dn
        ijv = builder[dn]
        foreach_hop!(iter_dn, latsel, trees, dn) do (si, sj), (i, j), (r, dr)
            ni = norbs[si]
            nj = norbs[sj]
            v = sanitize_block(blocktype(os), term(r, dr), (ni, nj))
            push!(ijv, (i, j, v))
        end
    end
    return nothing
end

#endregion
