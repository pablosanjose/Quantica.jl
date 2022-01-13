############################################################################################
# band
#region

band(h::AbstractHamiltonian, mesh::Mesh; solver = ES.LinearAlgebra(), kw...) =
    band(bloch(h, solver), mesh; solver, kw...)

function band(bloch::Bloch, basemesh::Mesh{SVector{L,T}}; mapping = missing,
    solver = ES.LinearAlgebra(), showprogress = true, defects = (), patches = 0, degtol = missing, warn = true) where {T,L}
    solvers = [apply(solver, bloch, SVector{L,T}, mapping) for _ in 1:Threads.nthreads()]
    defects´ = sanitize_Vector_of_SVectors(SVector{L,T}, defects)
    degtol´ = degtol isa Number ? degtol : sqrt(eps(real(T)))
    return band_precompilable(solvers, basemesh, showprogress, defects´, patches, degtol´, warn)
end

function band_precompilable(solvers::Vector{A}, basemesh::Mesh{SVector{L,T}},
    showprogress, defects, patches, degtol, warn) where {T,L,E,O,A<:AppliedEigensolver{T,L,E,O}}

    basemesh = copy(basemesh) # will become part of Band, possibly refined
    spectra = Vector{Spectrum{E,O}}(undef, length(vertices(basemesh)))
    bandverts = BandVertex{T,L,O}[]
    bandneighs = Vector{Int}[]
    bandneideg = similar(bandneighs)
    coloffsets = Int[]
    crossed = NTuple{6,Int}[]
    frustrated = similar(crossed)
    subbands = Subband{T,L,O}[]
    data = (; basemesh, spectra, bandverts, bandneighs, bandneideg, coloffsets, solvers, L,
              crossed, frustrated, subbands, defects, patches, showprogress, degtol, warn)

    # Step 1 - Diagonalize:
    # Uses multiple AppliedEigensolvers (one per Julia thread) to diagonalize bloch at each
    # vertex of basemesh. Then, it collects each of the produced Spectrum (aka "columns")
    # into a bandverts::Vector{BandVertex}, recording the coloffsets for each column
    band_diagonalize!(data)

    # Step 2 - Knit seams:
    # Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will
    # connect to other subspaces s´ in columns of a neighboring base vertex. Connections are
    # possible if the projector ⟨s'|s⟩ has any singular value greater than 1/2
    band_knit!(data)

    # Step 3 - Patch seams:
    # Dirac points and other topological band defects will usually produce dislocations in
    # mesh connectivity that results in missing simplices.
    if L>1
        band_patch!(data)
    end

    # Step 4 - Split subbands
    band_split!(data)

    return Band(subbands, basemesh, solvers)
end

#endregion

############################################################################################
# band_diagonalize!
#region

function band_diagonalize!(data)
    baseverts = vertices(data.basemesh)
    meter = Progress(length(baseverts), "Step 1 - Diagonalizing: ")
    push!(data.coloffsets, 0) # first element
    Threads.@threads for i in eachindex(baseverts)
        vert = baseverts[i]
        solver = data.solvers[Threads.threadid()]
        data.spectra[i] = solver(vert)
        data.showprogress && ProgressMeter.next!(meter)
    end
    # Collect band vertices and store column offsets
    for (basevert, spectrum) in zip(baseverts, data.spectra)
        append_band_column!(data, basevert, spectrum)
    end
    ProgressMeter.finish!(meter)
end


# collect spectrum into a band column (vector of BandVertices for equal base vertex)
function append_band_column!(data, basevert, spectrum)
    T = eltype(basevert)
    energies´ = [maybereal(ε, T) for ε in energies(spectrum)]
    states´ = states(spectrum)
    subs = collect(approxruns(energies´, data.degtol))
    for (i, rng) in enumerate(subs)
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(data.bandverts, BandVertex(basevert, energy, state))
    end
    push!(data.coloffsets, length(data.bandverts))
    foreach(_ -> push!(data.bandneideg, Int[]), length(data.bandneideg)+1:length(data.bandverts))
    foreach(_ -> push!(data.bandneighs, Int[]), length(data.bandneighs)+1:length(data.bandverts))
    return data
end

maybereal(energy, ::Type{T}) where {T<:Real} = T(real(energy))
maybereal(energy, ::Type{T}) where {T<:Complex} = T(energy)

# Gram-Schmidt but with column normalization only when norm^2 >= threshold (otherwise zero!)
function orthonormalize!(m::AbstractMatrix, threshold = 0)
    @inbounds for j in axes(m, 2)
        col = view(m, :, j)
        for j´ in 1:j-1
            col´ = view(m, :, j´)
            norm2´ = dot(col´, col´)
            iszero(norm2´) && continue
            r = dot(col´, col)/norm2´
            col .-= r .* col´
        end
        norm2 = real(dot(col, col))
        factor = ifelse(norm2 < threshold, zero(norm2), 1/sqrt(norm2))
        col .*= factor
    end
    return m
end

#endregion

############################################################################################
# band_knit!
#region

function band_knit!(data)
    meter = Progress(length(data.spectra), "Step 2 - Knitting: ")
    for isrcbase in eachindex(data.spectra)
        for idstbase in neighbors_forward(data.basemesh, isrcbase)
            knit_seam!(data, isrcbase, idstbase)
        end
        data.showprogress && ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)
    return data
end

# Take two intervals (srcrange, dstrange) of bandverts (linked by base mesh)
# and fill bandneighs with their connections, using the projector colproj
# hascrossing signals some crossing of energies across the seam
function knit_seam!(data, ib, jb)
    srcrange = column_range(data, ib)
    dstrange = column_range(data, jb)
    colproj  = states(data.spectra[jb])' * states(data.spectra[ib])
    for i in srcrange
        src = data.bandverts[i]
        for j in dstrange
            dst = data.bandverts[j]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj)
            if connections > 0
                push!(data.bandneighs[i], j)
                push!(data.bandneighs[j], i)
                push!(data.bandneideg[i], connections)
                push!(data.bandneideg[j], connections)
                # populate crossed with all crossed links if lattice dimension > 1
                if data.L > 1
                    for i´ in first(srcrange):i-1, j´ in data.bandneighs[i´]
                        j´ in dstrange || continue
                        # if crossed, push! with ordered i´ < i, j´ > j
                        j´ > j && push!(data.crossed, (ib, jb, i´, i, j´, j))
                    end
                end
            end
        end
    end
    return data
end

# number of singular values greater than √threshold. Fast rank-1 |svd|^2 is r = tr(proj'proj)
# For higher ranks and r > 0 we must compute and count singular values
# The threshold is arbitrary, and is fixed heuristically to a high enough value to connect
# in the coarsest base lattices
function connection_rank(proj, threshold = 0.4)
    rankf = sum(abs2, proj)
    fastrank = ifelse(rankf >= threshold, 1, 0)  # For rank=1 proj: upon doubt, connect
    if iszero(fastrank) || size(proj, 1) == 1 || size(proj, 2) == 1
        return fastrank
    else
        sv = svdvals(proj)
        return count(s -> abs2(s) >= threshold, sv)
    end
end

column_range(data, ibase) = data.coloffsets[ibase]+1:data.coloffsets[ibase+1]

#endregion

############################################################################################
# band_patch!
#region

function band_patch!(data)
    data.patches > 0 || return data
    insert_defects!(data)
    queue_frustrated!(data)
    data.warn && isempty(data.defects) &&
        @warn "Trying to patch $(length(data.frustrated)) band dislocations without a list `defects` of defect positions."
    meter = data.patches < Inf ? Progress(data.patches, "Step 3 - Patching: ") :
                                 ProgressUnknown("Step 3 - Patching: ")
    newcols = 0
    done = false
    while !isempty(data.frustrated) && !done
        (ib, jb, i, i´, j, j´) = pop!(data.frustrated)
        # check edge has not been previously split
        jb in neighbors(data.basemesh, ib) || continue
        newcols += 1
        done = newcols == data.patches
        # new vertex
        k = crossing(data, (ib, jb, i´, i, j´, j))
        # insert and connect a new column into band
        insert_column!(data, (ib, jb), k)
        # From the added crossings, remove all that are non-frustrated
        queue_frustrated!(data)
        data.showprogress && ProgressMeter.next!(meter)
    end
    # If we added new columns, base edges will have split -> rebuild base simplices
    newcols > 0 && build_cliques!(simplices(data.basemesh), neighbors(data.basemesh), data.L + 1)
    data.showprogress && ProgressMeter.finish!(meter)
    ndefects = length(data.frustrated)
    data.warn && !iszero(ndefects) &&
        @warn("Warning: Band with $ndefects dislocation defects. Consider specifying topological defect locations with `defects` (or adjusting mesh) and/or increasing `patches`")
    return data
end

function insert_defects!(data)
    # insert user-provided defects as new columns in band
    foreach(k -> insert_defect_column!(data, k), data.defects)
    # detect possible defects in band and append them to data.defects
    mindeg = minimum(degeneracy, data.bandverts)
    for (v, ns) in zip(data.bandverts, data.bandneighs)
        k = base_coordinates(v)
        d = degeneracy(v)
        # exclude v if v does not increase degeneracy over minimum
        degeneracy(v) == mindeg && continue
        # only select vertices that have greater degeneracy than all its neighbors
        any(n -> degeneracy(data.bandverts[n]) >= d, ns) && continue
        # exclude v if it is already in data.defects
        any(kd -> kd ≈ k, data.defects) && continue
        push!(data.defects, base_coordinates(v))
    end
    return data
end

function insert_defect_column!(data, kdefect)
    base = data.basemesh
    for k in vertices(base)
        k ≈ kdefect && return data
    end
    # find closest edge (center) to kdefect
    (ib, jb) = argmin(((i, j) for i in eachindex(vertices(base)) for j in neighbors(base, i))) do (i, j)
        sum(abs2, 0.5*(vertices(base, i) + vertices(base, j)) - kdefect)
    end
    insert_column!(data, (ib, jb), kdefect)
    return data
end

function insert_column!(data, (ib, jb), k)
    solver = first(data.solvers)
    # remove all bandneighs in this seam
    delete_seam!(data, ib, jb)
    # create new vertex in basemesh by splitting the edge, and connect to neighbors
    split_edge!(data.basemesh, (ib, jb), k)
    # compute spectrum at new vertex
    spectrum = solver(k)
    push!(data.spectra, spectrum)
    # collect spectrum into a set of new band vertices
    append_band_column!(data, k, spectrum)
    # knit all new seams
    newbasevertex = length(vertices(data.basemesh)) # index of new base vertex
    newbaseneighs = neighbors(data.basemesh, newbasevertex)
    jb´ = newbasevertex             # destination of all new edges
    for ib´ in newbaseneighs        # neighbors of new vertex are new edge sources
        knit_seam!(data, ib´, jb´)  # also pushes new crossings into data.crossed
    end
    return data
end

# queue frustrated crossings and sort decreasing distance between crossing and defects
function queue_frustrated!(data)
    # we must check all crossed each time, as their frustration can change at any point
    for c in data.crossed
        is_frustrated_crossing(data, c) && !in(c, data.frustrated) && push!(data.frustrated, c)
    end
    isempty(data.defects) || reverse!(sort!(data.frustrated, by = i -> distance_to_defects(data, i)))
    return data
end

function distance_to_defects(data, ic)
    kc = crossing(data, ic)
    return minimum(d -> sum(abs2, kc - d), data.defects)
end

is_frustrated_crossing(data, (ib, jb, i, i´, j, j´)) =
    is_frustrated_link(data, (ib, jb, i, j)) || is_frustrated_link(data, (ib, jb, i´, j´))

function is_frustrated_link(data, (ib, jb, i, j))
    deg = degeneracy_link(data, i, j)
    for kb in neighbors(data.basemesh, ib), kb´ in neighbors(data.basemesh, jb)
        kb == kb´ || continue
        count = 0
        for k in data.bandneighs[i], k´ in data.bandneighs[j]
            k == k´ && k in column_range(data, kb) || continue
            count += min(degeneracy_link(data, i, k), degeneracy_link(data, j, k))
        end
        count < deg && return true
    end
    return false
end

function degeneracy_link(data, i, j)
    for (k, d) in zip(data.bandneighs[i], data.bandneideg[i])
        k == j && return d
    end
    return 0
end

function crossing(data, (ib, jb, i, i´, j, j´))
    εi, εi´, εj, εj´ = energy.(getindex.(Ref(data.bandverts), (i, i´, j, j´)))
    ki, kj = vertices(data.basemesh, ib), vertices(data.basemesh, jb)
    λ = (εi - εi´) / (εj´ - εi´ - εj + εi)
    k = ki + λ * (kj - ki)
    return k
end

function delete_seam!(data, isrcbase, idstbase)
    srcrange = column_range(data, isrcbase)
    dstrange = column_range(data, idstbase)
    for isrc in srcrange
        fast_setdiff!((data.bandneighs[isrc], data.bandneideg[isrc]), dstrange)
    end
    for idst in dstrange
        fast_setdiff!((data.bandneighs[idst], data.bandneideg[idst]), srcrange)
    end
    return data
end

#endregion

############################################################################################
# band_split!
#region

function band_split!(data)
    # vsinds are the subband index of each vertex index
    # svinds is lists of band vertex indices that belong to the same subband
    vsinds, svinds = subsets(data.bandneighs)
    meter = Progress(length(svinds), "Step 4 - Splitting: ")
    new2old = sortperm(vsinds)
    old2new = invperm(new2old)
    offset = 0
    for subset in svinds
        sverts  = data.bandverts[subset]
        sneighs = [ [old2new[dstold] - offset for dstold in data.bandneighs[srcold]]
                  for srcold in subset]
        offset += length(sverts)
        ssimps  = build_cliques(sneighs, data.L + 1)
        orient_simplices!(ssimps, sverts)
        push!(data.subbands, Subband(sverts, sneighs, ssimps))
        data.showprogress && ProgressMeter.next!(meter)
    end
    return data
end

#endregion