
############################################################################################
# supercell(lattice, ...)
#region

supercell(v...; kw...) = lat -> supercell(lat, v...; kw...)

function supercell(l::Lattice{T,E,L}, v::Vararg{<:Any,L´}; kw...) where {T,E,L,L´}
    smat = sanitize_SMatrix(SMatrix{L,L´,Int}, v)
    s, m = supercell_sitelist_masklist(l, smat; kw...)
    b = Bravais{T,E,L´}(bravais_mat(l) * smat)
    n = sublatnames(l)
    o = offsets_from_masklist(m, nsublats(l))
    u = Unitcell(s, n, o)
    return Lattice(b, u)
end

function supercell_sitelist_masklist(lat::Lattice, smat::SMatrix{L,L´,Int};
                   cellseed = zero(SVector{L,Int}), kw...) where {L,L´}
    applied_selector = siteselector(lat; kw...)
    check_finite_supercell(smat, applied_selector)
    smatfull = makefull(smat)
    masklist = supercell_masklist_full(smatfull, cellseed, lat)
    smatperp = convert(SMatrix{L,L-L´,Int}, view(smatfull, :, L´+1:L))
    seedperp = zero(SVector{L-L´,Int})
    sitelist = similar(sites(lat), 0)
    supercell_sitelist!!(sitelist, masklist, smatperp, seedperp, applied_selector)
    cosort!(masklist, sitelist)  # sorted by sublat, then cell, then siteidx (i.e. masklist)
    return sitelist, masklist
end

check_finite_supercell(smat, latselector) =
    size(smat, 2) == size(smat, 1) || source(latselector).region !== missing ||
        throw(ArgumentError("Cannot reduce supercell dimensions without a bounding region."))

# build masklist = [(sublatindex, cell, siteindex)] for all sites in full supercell defined by smatfull
function supercell_masklist_full(smat::SMatrix{L,L,Int}, cellseed::SVector{L,Int}, lat) where {L}
    supercell_nsites = nsites(lat) * round(Int, abs(det(smat)))
    iszero(supercell_nsites) && throw(ArgumentError("Supercell is empty. Singular supercell matrix?"))
    masklist = Vector{Tuple{Int,SVector{L,Int},Int}}(undef, supercell_nsites)
    br = bravais_mat(lat)
    projector = pinverse(br * smat)
    counter = 0
    iter = BoxIterator(cellseed)
    for cell in iter, (i, slat) in sitesublatiter(lat)
        r = site(lat, i, cell)
        δn = projector * r
        if all(x -> 0 <= x < 1, δn)
            counter += 1
            masklist[counter] = (slat, cell, i)
        end
        acceptcell!(iter, cell)
        counter == supercell_nsites && break
    end
    counter == supercell_nsites || throw(ErrorException("Internal error: failed to find all sites in supercell, only $counter of $supercell_nsites"))
    return masklist
end

# build sitelist = [sitepositions...] and masklist´ = [(sublat, cell´, siteidx)...]
# where cell´ varies along axes smatperp not in smat, filtered by selector
function supercell_sitelist!!(sitelist, masklist, smatperp, seedperp, applied_selector)
    lat = target(applied_selector)
    masklist0 = copy(masklist)
    empty!(masklist)
    empty!(sitelist)
    iter = BoxIterator(seedperp)
    for c in iter, (s, cell, i) in masklist0
        cell´ = cell + smatperp * c
        r = site(lat, i, cell´)
        if (i, r) in applied_selector
            push!(masklist, (s, cell´, i))
            push!(sitelist, r)
            acceptcell!(iter, c)
        end
    end
    return nothing
end

function offsets_from_masklist(masklist, nsublats)
    ds = zeros(Int, nsublats)
    @simd for m in masklist
        ds[first(m)] += 1
    end
    offsets = cumsum(ds)
    prepend!(offsets, 0)
    return offsets
end

#endregion