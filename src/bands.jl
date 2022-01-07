############################################################################################
# mesh methods
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs = marching_neighbors(cinds)  # sorted neighbors of i, with n[i][j] > i
    simps  = build_cliques(neighs, L+1)    # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs, simps)
end

# forward neighbors, cind is a CartesianRange over vertices
function marching_neighbors(cinds)
    linds = LinearIndices(cinds)
    matrix = [Int[] for _ in cinds]
    for cind in cinds
        forward  = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        for cind´ in forward
            cind === cind´ && continue
            push!(matrix[cind], linds[cind´])
            push!(matrix[cind´], linds[cind])
        end
    end
    neighs = vec(matrix)
    return neighs
end

# simplices are not recomputed for performance
function split_edge!(m, (i, j), k)
    i == j && return m
    if i > j
        i, j = j, i
    end
    cut_edge!(m, (i, j))
    verts = vertices(m)
    push!(verts, k)
    push!(neighbors(m), Int[])
    dst = length(verts)
    newneighs = intersect(neighbors(m, i), neighbors(m, j))
    push!(newneighs, i, j)
    for src in newneighs
        push!(neighbors(m, src), dst)
        push!(neighbors(m, dst), src)
    end
    return m
end

function cut_edge!(m, (i, j))
    i == j && return m
    if i > j
        i, j = j, i
    end
    fast_setdiff!(neighbors(m, i), j)
    fast_setdiff!(neighbors(m, j), i)
    return m
end

function fast_setdiff!(c, rng)
    i = 0
    for x in c
        x in rng && continue
        i += 1
        c[i] = x
    end
    resize!(c, i)
    return c
end

# groups of n all-to-all connected neighbors, sorted
function build_cliques(neighs, nverts)
    cliques = Vector{Int}[]
    for (src, dsts) in enumerate(neighs), ids in Combinations(length(dsts), nverts - 1)
        if all_adjacent(ids, dsts, neighs)
            clique = prepend!(dsts[ids], src)
            push!(cliques, clique)
        end
    end
    return cliques
end

# Check whether dsts[ids] are all mutual neighbors. ids are a total of nverts-1 indices of dsts = neighs[src].
function all_adjacent(ids, dsts, neighs)
    nids = length(ids)
    for (n, id) in enumerate(ids), n´ in n+1:nids
        dst = dsts[ids[n´]]
        dst in neighbors_forward(neighs, dsts[id]) || return false
    end
    return true
end

function orient_simplices!(simplices, vertices::Vector{B}) where {L,B<:BandVertex{<:Any,L}}
    for simplex in simplices
        k0 = base_coordinates(vertices[simplex[1]])
        edges = ntuple(i -> base_coordinates(vertices[simplex[i+1]])-k0, Val(L))
        volume = det(hcat(edges...))
        if volume < 0 # switch last
            simplex[end], simplex[end-1] = simplex[end-1], simplex[end]
        end
    end
    return simplices
end

#endregion

############################################################################################
# bands
#region

bands(h::AbstractHamiltonian, mesh::Mesh; solver = ES.LinearAlgebra(), kw...) =
    bands(bloch(h, solver), mesh; solver, kw...)

function bands(bloch::Bloch, basemesh::Mesh{SVector{L,T}};
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true, patchlevel = 0) where {T,L}

    S = spectrumtype(bloch)
    spectra = Vector{S}(undef, length(vertices(basemesh)))
    O = orbtype(bloch)
    bandverts = BandVertex{T,L,O}[]
    bandneighs = Vector{Int}[]
    coloffsets = Int[]
    crossed_seams = Tuple{Int,Int}[]
    basemesh = patchlevel > 0 ? copy(basemesh) : basemesh
    solvers = [apply(solver, bloch, SVector{L,T}, mapping) for _ in 1:Threads.nthreads()]
    data = (; spectra, bandverts, bandneighs, coloffsets, crossed_seams, solvers, basemesh, patchlevel, showprogress)

    # Step 1 - Diagonalize:
    band_diagonalize!(data)

    # Step 2 - Knit seams:
    # Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will connect
    # to other subspaces s´ in columns of a neighboring base vertex. Connections are
    # possible if the projector ⟨s'|s⟩ has any singular value greater than 1/2
    band_knit!(data)

    # Step 3 - Patch seams:
    if patchlevel > 0
        band_patch!(data)
    end

    # Build band simplices
    bandsimps = build_cliques(bandneighs, L+1)
    orient_simplices!(bandsimps, bandverts)

    return Mesh(bandverts, bandneighs, bandsimps)
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
        append_band_column!(data.bandverts, basevert, spectrum)
        push!(data.coloffsets, length(data.bandverts))
    end
    ProgressMeter.finish!(meter)
end


# collect spectrum into a band column (vector of BandVertices for equal base vertex)
function append_band_column!(bandverts, basevert, spectrum)
    T = eltype(basevert)
    energies´ = [maybereal(ε, T) for ε in energies(spectrum)]
    states´ = states(spectrum)
    subs = collect(approxruns(energies´))
    for (i, rng) in enumerate(subs)
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(bandverts, BandVertex(basevert, energy, state))
    end
    return bandverts
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
    foreach(_ -> push!(data.bandneighs, Int[]), data.bandverts)
    for isrcbase in eachindex(data.spectra)
        for idstbase in neighbors_forward(data.basemesh, isrcbase)
            crossed = knit_seam!(data, isrcbase, idstbase)
            crossed && push!(data.crossed_seams, (isrcbase, idstbase))
        end
        data.showprogress && ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)
end

# Take two intervals (srcrange, dstrange) of bandverts (linked by base mesh)
# and fill bandneighs with their connections, using the projector colproj
# hascrossing signals some crossing of energies across the seam
function knit_seam!(data, isrcbase, idstbase)
    srcrange = data.coloffsets[isrcbase]+1:data.coloffsets[isrcbase+1]
    dstrange = data.coloffsets[idstbase]+1:data.coloffsets[idstbase+1]
    colproj  = states(data.spectra[idstbase])' * states(data.spectra[isrcbase])
    idsthigh = 0
    hascrossing = false
    for isrc in srcrange
        src = data.bandverts[isrc]
        for idst in dstrange
            dst = data.bandverts[idst]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj)
            if connections > 0
                push!(data.bandneighs[isrc], idst)
                push!(data.bandneighs[idst], isrc)
                hascrossing = hascrossing || idst < idsthigh
                idsthigh = idst
            end
        end
    end
    return hascrossing
end

# equivalent to r = round(Int, tr(proj'proj)), but if r > 0, must compute and count singular values
function connection_rank(proj)
    frank = sum(abs2, proj)
    fastrank = frank ≈ 0.5 ? 1 : round(Int, frank)  # upon doubt, connect
    if iszero(fastrank) || size(proj, 1) == 1 || size(proj, 2) == 1
        return fastrank
    else
        sv = svdvals(proj)
        return count(s -> abs2(s) >= 0.5 || abs2(s) ≈ 0.5, sv)
    end
end

#endregion

############################################################################################
# band_patch!
#region

function band_patch!(data)
    meter = Progress(data.patchlevel, "Step 3 - Patching: ")
    baseverts = vertices(data.basemesh)
    num_original_baseverts = length(baseverts) # to distinguish new base vertices
    newcols = 0
    for (ind_cs, (isrcbase, idstbase)) in enumerate(data.crossed_seams)
        column_added = on_dislocation_add_column!(data, isrcbase, idstbase, num_original_baseverts)
        newcols += column_added
        if column_added && newcols <= data.patchlevel
            # zero-out the seam from crossed_seams, signaling it must not be processed again
            data.crossed_seams[ind_cs] = (0, 0)
            # first check if added base vertex is a neighbor of processed crossed_seams
            # if so, re-add it at the end to process again (since it is now connected to a new vertex)
            newvertex = length(data.spectra) # index of new vertex
            newneighs = neighbors(data.basemesh, newvertex)
            for ind_cs´ in 1:ind_cs-1
                (i, j) = data.crossed_seams[ind_cs´]
                i in newneighs && j in newneighs && push!(data.crossed_seams, (i, j))
            end
            # remove old seam neighbors
            cut_seam!(data, isrcbase, idstbase)
            idstbase´ = newvertex
            for isrcbase´ in newneighs  # neighbors of new vertex are new edge sources
                crossed = knit_seam!(data, isrcbase´, idstbase´)
                crossed && push!(data.crossed_seams, (isrcbase´, idstbase´))
            end
            data.showprogress && ProgressMeter.next!(meter)
        end
    end
end

function on_dislocation_add_column!(data, isrcbase, idstbase, num_original_baseverts)
    # Check for dislocation
    srcrange = data.coloffsets[isrcbase]+1:data.coloffsets[isrcbase+1]
    dstrange = data.coloffsets[idstbase]+1:data.coloffsets[idstbase+1]
    solver = first(data.solvers)
    for isrc in srcrange, isrc´ in isrc+1:last(srcrange)
        for idst  in neighbors_forward(data.bandneighs, isrc),
            idst´ in neighbors_forward(data.bandneighs, isrc´)
            # Check whether band vertices are in correct column and whether they cross
            idst in dstrange && idst´ in dstrange && idst´ < idst || continue
            # If cross neighbors (forward plus backward) have different intersections
            # there is a dislocation i.e. some simplex is frustrated
            is_near_dislocation =
                !equal_intersection(data.bandneighs, (isrc, idst´), (isrc´, idst)) ||
                seam_near_new_vertex(data.basemesh, num_original_baseverts, isrcbase, idstbase)
            # @show is_dislocation, is_near_dislocation, isrcbase, idstbase
            if is_near_dislocation
                # in such case, split base edge
                εs, εs´, εd, εd´ = energy.(getindex.(Ref(data.bandverts), (isrc, isrc´, idst, idst´)))
                ks, kd = vertices(data.basemesh, isrcbase), vertices(data.basemesh, idstbase)
                λ = (εs - εs´) / (εd´ - εs´ - εd + εs)
                k = ks + λ * (kd - ks)
                split_edge!(data.basemesh, (isrcbase, idstbase), k)
                spectrum = solver(k)
                push!(data.spectra, spectrum)
                # collect spectrum into a set of new band vertices
                append_band_column!(data.bandverts, k, spectrum)
                # add empty neighbor lists for new band vertices
                foreach(_ -> push!(data.bandneighs, Int[]), length(data.bandneighs)+1:length(data.bandverts))
                # update coloffsets
                push!(data.coloffsets, length(data.bandverts))
                # stop searching, report that new column was added
                return true
            end
        end
    end
    return false
end

function cut_seam!(data, isrcbase, idstbase)
    srcrange = data.coloffsets[isrcbase]+1:data.coloffsets[isrcbase+1]
    dstrange = data.coloffsets[idstbase]+1:data.coloffsets[idstbase+1]
    for isrc in srcrange
        fast_setdiff!(data.bandneighs[isrc], dstrange)
    end
    for idst in dstrange
        fast_setdiff!(data.bandneighs[idst], srcrange)
    end
    return data
end

# Define s = (s₁ ,s₂), p = (p₁, p₂). Is s₁ ∩ s₂ == p₁ ∩ p₂ ?
# This happens iif (s₁ ∩ s₂) ∈ pᵢ and (p₁ ∩ p₂) ∈ sᵢ
equal_intersection(ns, (i, j), (i´, j´)) =
    equal_intersection((ns[i], ns[j]), (ns[i´], ns[j´]))
equal_intersection(s, p) =
    first_inside_second(s, p) && first_inside_second(p, s)

function first_inside_second((s1, s2), (p1, p2))
    for s in s1
        if s in s2 # s ∈ (s1 ∩ s2)
            s in p1 && s in p2 || return false
        end
    end
    return true
end

function seam_near_new_vertex(basemesh, num_original_vertices, isrcbase, idstbase)
    for nsrc in neighbors(basemesh, isrcbase)
        nsrc <= num_original_vertices && continue
        nsrc in neighbors(basemesh, idstbase) && return true
    end
    return false
end

#endregion