supercell(v...; kw...) = x -> supercell(x, v...; kw...)

############################################################################################
# SupercellData (required to build lattice supercell)
#region

struct SupercellData{T,E,L,L´,LL´}
    lat::Lattice{T,E,L}
    sitelist::Vector{SVector{E,T}}                     # [sitepositions...]
    masklist::Vector{Tuple{Int,SVector{L,Int},Int}}    # [(sublatindex, cell, siteindex)...]
    bravais´::Bravais{T,E,L´}
    sm::SMatrix{L,L´,Int,LL´}
    detsm´::Int
    proj_invsm´_detsm´::SMatrix{L´,L,Int,LL´}
end

function supercell_data(lat::Lattice{<:Any,<:Any,L}, vs...; seed = missing, kw...) where {L}
    smat = sanitize_supercell(Val(L), vs)
    selector = siteselector(; kw...)
    applied_selector = apply(selector, lat)
    # IMPORTANT BEHAVIOR: if lattice dimensions are reduced and there is no bounding region
    # and no cell list, supercell is infinite. In this case: stop at a single perp unit cell
    only_one_perp = size(smat, 2) < size(smat, 1) && region(selector) === missing && cells(selector) === missing
    cellseed = seed === missing ? zero(SVector{L,Int}) : sanitize_SVector(SVector{L,Int}, seed)
    return supercell_data(lat, smat, cellseed, applied_selector, only_one_perp)
end

function supercell_data(lat::Lattice{T,E,L},
                        sm::SMatrix{L,L´,Int},
                        cellseed::SVector{L,Int},
                        applied_selector::AppliedSiteSelector{T,E},
                        only_one_perp::Bool = false) where {T,E,L,L´}
    sm´ = makefull(sm)
    detsm´ = round(Int, abs(det(sm´)))
    iszero(detsm´) && throw(ArgumentError("Supercell is empty. Singular supercell matrix?"))
    # inverse of full supercell sm´ (times det(sm´) to make it integer)
    # projected back onto superlattice axes L´
    invsm´_detsm´ = round.(Int, inv(sm´) * detsm´)
    proj_invsm´_detsm´ = SMatrix{L´,L,Bool}(I) * invsm´_detsm´
    bravais´ = Bravais{T,E,L´}(bravais_matrix(lat) * sm)
    masklist = supercell_masklist_full(invsm´_detsm´, detsm´, cellseed, lat)
    smperp = convert(SMatrix{L,L-L´,Int}, view(sm´, :, L´+1:L))
    seedperp = zero(SVector{L-L´,Int})
    sitelist = similar(sites(lat), 0)
    supercell_sitelist!!(sitelist, masklist, smperp, seedperp, lat, applied_selector, only_one_perp)
    cosort!(masklist, sitelist)  # sorted by sublat, then cell, then siteidx
    return SupercellData(lat, sitelist, masklist, bravais´, sm, detsm´, proj_invsm´_detsm´)
end

smat_projector(::SupercellData{<:Any,<:Any,L,L´}) where {L,L´} = SMatrix{L´,L,Bool}(I)

# Make matrix square by appending (or prepending) independent columns if possible
function makefull(m::SMatrix{L,L´}) where {L,L´}
    Q = qr(Matrix(m), NoPivot()).Q * I
    for i in 1:L * L´
        @inbounds Q[i] = m[i]         # overwrite first L´ cols with originals
    end
    return SMatrix{L,L}(Q)
end

# round to integers to preserve eltype
makefull(m::SMatrix{<:Any,<:Any,Int}) = round.(Int, makefull(float(m)))

# build masklist = [(sublatindex, cell, siteindex)] for all sites
# in full supercell defined by makefull(smat)
function supercell_masklist_full(smat´⁻¹N::SMatrix{L,L,Int}, N, cellseed::SVector{L,Int}, lat) where {L}
    supercell_nsites = nsites(lat) * N
    masklist = Vector{Tuple{Int,SVector{L,Int},Int}}(undef, supercell_nsites)
    counter = 0
    iter = BoxIterator(cellseed)
    for cell in iter, (i, slat) in sitesublatiter(lat)
        δn = smat´⁻¹N * cell
        if all(x -> 0 <= x < N, δn)
            counter += 1
            masklist[counter] = (slat, cell, i)
        end
        acceptcell!(iter, cell)
        counter == supercell_nsites && break
    end
    counter == supercell_nsites ||
        internalerror("supercell_masklist_full: failed to find all sites in supercell, only $counter of $supercell_nsites")
    return masklist
end

# build sitelist = [sitepositions...] and masklist´ = [(sublat, cell´, siteidx)...]
# where cell´ varies along axes smatperp not in smat, filtered by selector
function supercell_sitelist!!(sitelist, masklist, smatperp, seedperp, lat, applied_selector, only_one_perp)
    masklist´ = copy(masklist)
    empty!(masklist)
    empty!(sitelist)
    isnull(applied_selector) && return nothing
    cs = cells(applied_selector)
    csbool = zeros(Bool, length(cs))
    iter = BoxIterator(seedperp)
    for n in iter
        keepgoing = isempty(sitelist)   # if none have been found, ensure we don't stop
        for (s, cell, i) in masklist´
            cell´ = cell + smatperp * n
            r = site(lat, i, cell´)
            if (i, r, cell´) in applied_selector
                push!(masklist, (s, cell´, i))
                push!(sitelist, r)
                foundcell!(csbool, cs, cell´)
                keepgoing = true
            end
        end
        only_one_perp && break
        # If we haven't found all selector cells, accept supercell n so we keep searching
        # note that if isempty(csbool) then all(csbool) is true and we stop accepting
        foundall = all(csbool)
        keepgoing = keepgoing || !foundall
        keepgoing && acceptcell!(iter, n)
    end
    return nothing
end

function foundcell!(csbool, cs, cell)
    isempty(cs) && return true
    for (i, c) in enumerate(cs)
        if cell == c
            csbool[i] = true
            return true
        end
    end
    return false
end

#endregion

############################################################################################
# supercell(lattice, ...)
#region

supercell(lat::Lattice, vs...; kw...) =
    lattice(supercell_data(lat, vs...; kw...))

function lattice(data::SupercellData)
    n = sublatnames(data.lat)
    o = supercell_offsets(data.masklist, nsublats(data.lat))
    u = Unitcell(data.sitelist, n, o)
    lat = Lattice(data.bravais´, u)
    return lat
end

function supercell_offsets(masklist, nsublats)
    ds = zeros(Int, nsublats)
    @simd for m in masklist
        ds[first(m)] += 1
    end
    offsets = lengths_to_offsets(ds)
    return offsets
end

# function latslice(data::SupercellData)
#     # data.masklist is [(sublat, cell, siteindex)...] for each site in lat´
#     cellindsvec = sort!(Base.tail.(data.masklist))
#     subcells = splitruns(last, cellindsvec; by = first, reduce = CellSites)  # see tools.jl
#     ls = LatticeSlice(data.lat, subcells)
#     return ls
# end

#endregion

############################################################################################
# supercell(::Hamiltonian, ...)
#region

function supercell(h::Hamiltonian, v...; mincoordination = 0, kw...)
    data = supercell_data(lattice(h), v...; kw...)
    return supercell(h, data; mincoordination)
end

function supercell(h::Hamiltonian, data::SupercellData; mincoordination = 0)
    # data.sitelist may be modified to apply mincoordination
    indexlist, offset = supercell_indexlist!(data, h, mincoordination)
    lat´ = lattice(data)
    B = blocktype(h)
    bs´ = OrbitalBlockStructure{B}(norbitals(h, :), sublatlengths(lat´))
    builder = CSCBuilder(lat´, bs´)
    har´ = supercell_harmonics(h, data, builder, indexlist, offset)
    return Hamiltonian(lat´, bs´, har´)
end

function supercell_harmonics(h, data, builder, indexlist, offset)
    # Note: masklist = [(sublat, old_cell, old_siteindex)...]
    for (col´, (_, cellsrc, col)) in enumerate(data.masklist)
        for har in harmonics(h)
            celldst = cellsrc + dcell(har)
            # cell is the position of celldst within its supercell scell
            cell, scell = wrap_cell_onto_supercell(celldst, data)
            m = unflat(har)
            rows = rowvals(m)
            vals = nonzeros(m)
            for p in nzrange(m, col)
                row = rows[p]
                c = CartesianIndex((row, Tuple(cell)...)) + offset
                # If any row is out of bounds, all rows are (because cell is out of bounds)
                checkbounds(Bool, indexlist, c) || break
                # Note: indexlist[(old_site_index, old_site_cell...) + offset] = new_site_index
                row´ = indexlist[c]
                iszero(row´) && continue
                csc = builder[scell]
                # if csc is a newly created collector, advance column to current col´
                sync_columns!(csc, col´)
                # val´ = applymodifiers(vals[p], slat, (source_i, target_i), (source_dn, target_dn), modifiers´...)
                val´ = vals[p]
                pushtocolumn!(csc, row´, val´)
            end
        end
        finalizecolumn!(builder)
    end
    return sparse(builder)
end

function wrap_cell_onto_supercell(cell, data)
    # This psmat⁻¹N is the inverse of the full supercell matrix sm´ (i.e. full-rank square),
    # times N = det(sm´) to make it integer, and projected onto the actual L supercell dims
    psmat⁻¹N = data.proj_invsm´_detsm´
    N = data.detsm´
    smat = data.sm
    # `scell` is the indices of the supercell where `cell` lives
    scell = fld.(psmat⁻¹N * cell, N)
    # `cell´` is `cell` shifted back to the zero supercell
    cell´ = cell - smat * scell
    return cell´, scell
end

function supercell_indexlist(data)
    (cellmin, cellmax), (imin, imax) = mask_bounding_box(data.masklist)
    indexlist = zeros(Int, 1 + imax - imin, 1 .+ cellmax .- cellmin...)
    offset  = CartesianIndex((1 - imin, 1 .- cellmin...))
    for (inew, (_, cellold, iold)) in enumerate(data.masklist)
        c = CartesianIndex((iold, Tuple(cellold)...)) + offset
        indexlist[c] = inew
    end
    return indexlist, offset
end

# as above, but also remove sites with coordination < mincoordination
function supercell_indexlist!(data, h, mincoordination)
    indexlist, offset = supercell_indexlist(data)
    if mincoordination > 0
        indexlist, offset = remove_low_coordination_sites!(data, indexlist, offset, h, mincoordination)
    end
    return indexlist, offset
end

# more general version of boundingbox(cells)
function mask_bounding_box(masklist::Vector{Tuple{Int,SVector{L,Int},Int}}) where {L}
    cellmin = cellmax = ntuple(Returns(0), Val(L))
    imin = imax = 0
    for (s, cell, i) in masklist
        tcell = Tuple(cell)
        cellmin = min.(cellmin, tcell)
        cellmax = max.(cellmax, tcell)
        imin = min(imin, i)
        imax = max(imax, i)
    end
    return (cellmin, cellmax), (imin, imax)
end

function remove_low_coordination_sites!(data, indexlist, offset, h, mincoordination)
    # remove low-coordination sites in masklist and sitelist until they don't change
    # when a site is removed, it becomes a zero in indexlist, but the other indices
    # are not changed. Hence, at the end we must recompute indexlist with the final
    # masklist and sitelist. Note that lat´ in the calling context is updated through
    # sitelist too, since sites(lat´) === data.sitelist
    masklist  = data.masklist
    masklist´ = similar(masklist)
    sitelist  = data.sitelist
    sitelist´ = similar(sitelist)
    while true
        resize!(masklist´, 0)
        resize!(sitelist´, 0)
        for (r, (sj, cellj, j)) in zip(sitelist, masklist)
            coordination = 0
            for har in harmonics(h)
                dn = dcell(har)
                celli, _ = wrap_cell_onto_supercell(cellj + dn, data)
                m = matrix(har)
                rows = rowvals(m)
                vals = nonzeros(m)
                for p in nzrange(m, j)
                    i = rows[p]
                    c = CartesianIndex((i, Tuple(celli)...)) + offset
                    checkbounds(Bool, indexlist, c) || break
                    isonsite = i == j && iszero(dn)
                    if !isonsite && !iszero(indexlist[c]) && !iszero(vals[p])
                        coordination += 1
                    end
                end
            end
            if coordination >= mincoordination
                push!(sitelist´, r)
                push!(masklist´, (sj, cellj, j))
            else
                c = CartesianIndex((j, Tuple(cellj)...)) + offset
                indexlist[c] = 0
            end
        end
        length(sitelist´) == length(sitelist) && break
        sitelist´, sitelist = sitelist, sitelist´
        masklist´, masklist = masklist, masklist´
    end
    data.masklist .= masklist´
    data.sitelist .= sitelist´
    indexlist´, offset´ = supercell_indexlist(data)
    return indexlist´, offset´
end

#endregion

############################################################################################
# supercell(::ParametricHamiltonian, ...)
#region

function supercell(p::ParametricHamiltonian, v...; mincoordination = 0, kw...)
    h = parent(p)
    data = supercell_data(lattice(h), v...; kw...)
    h´ = supercell(h, data; mincoordination)
    shifts = supercell_shifts(data)     # allows to compute new ptrs to old r, dr
    ms = parent.(modifiers(p))          # extract unapplied modifiers to reapply them to h´
    ams = apply.(ms, Ref(h´), Ref(shifts))
    p´ = hamiltonian(h´, ams...)
    return p´
end

# For each site in new lattice, store the corresponding shift = bravais_matrix * cell
function supercell_shifts(data::SupercellData)
    b = bravais_matrix(data.lat)
    shifts = [b * cell for (_, cell, _) in data.masklist]
    return shifts
end


#endregion
