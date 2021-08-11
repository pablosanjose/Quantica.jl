############################################################################################
# OrbitalStructure constructor
#region

# norbs is a collection of number of orbitals, one per sublattice (or a single one for all)
function OrbitalStructure(lat::Lattice, norbs, ::Type{T} = numbertype(lat)) where {T}
    norbs´ = sanitize_Vector_of_Type(Int, nsublats(lat), norbs)
    O = SVector{maximum(norbs´),Complex{T}}
    offsets´ = offsets(lat)
    flatoffsets´ = flatoffsets(offsets, norbs´)
    return OrbitalStructure(O, norbs´, offsets´, flatoffsets´)
end

# sublat offsets after flattening (without padding zeros)
function flatoffsets(offsets, norbs)
    nsites = diff(offsets)
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

function hamiltonian(lat::Lattice, m = TightbindingModel(); orbitals = 1, type = numbertype(lat))
    orbstruct = OrbitalStructure(lat, orbitals, type)
    builder = IJVBuilder(lat, orbstruct)
    foreach(t -> applyterm!(builder, t), terms(m))
    HT = HamiltonianHarmonic{latdim(lat),blocktype(orbstruct)}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in ijvs if !isempty(e)]
    return Hamiltonian(lat, orbstruct, harmonics)
end

function applyterm!(builder, term::OnsiteTerm)
    lat = lattice(builder)
    dn0 = zerocell(builder.lat)
    ijv = builder[dn0]
    sel = appliedon(selector(term), lat)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_site(sel, dn0) do (s, i, r)
        n = norbs[s]
        v = sanitize_block(blocktype(os), term(r, r), (n, n))
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L}, term::HoppingTerm) where {L}
    lat = lattice(builder)
    sel = apply(selector(term), lat)
    kdtrees = kdtrees(builder)
    os = orbitalstructure(builder)
    norbs = norbitals(os)
    foreach_cell(sel) do dn, iter_dn
        ijv = builder[dn]
        foreach_hop!(iter_dn, sel, kdtrees, dn) do (si, sj), (i, j), (r, dr)
            ni = norbs[si]
            nj = norbs[sj]
            v = sanitize_block(blocktype(os), term(r, dr), (ni, nj))
            push!(ijv, (i, j, v))
        end
    end
    return nothing
end

#endregion
