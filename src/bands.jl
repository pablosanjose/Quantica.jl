############################################################################################
# mesh
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs_forward = marching_neighbors_forward(cinds)  # sorted neighbors of i, with n[i][j] > i
    simps  = build_cliques(neighs_forward, L+1)    # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs_forward, simps)
end

# forward neighbors, cind is a CartesianRange over vertices
function marching_neighbors_forward(cinds)
    linds = LinearIndices(cinds)
    matrix_forward = [Int[] for _ in cinds]
    for cind in cinds
        nlist = matrix_forward[cind]
        forward  = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        for cind´ in forward
            cind === cind´ && continue
            push!(nlist, linds[cind´])
        end
    end
    neighs_forward = vec(matrix_forward)
    return neighs_forward
end

# simplices are not recomputed for performance
function splitedge!(m, (i, j), k)
    i == j && return m
    if i > j
        i, j = j, i
    end
    cut_edge!(m, (i, j))
    verts = vertices(m)
    push!(verts, k)
    push!(neighbors(m), Int[])
    push!(neighbors_forward(m), Int[])
    dst = length(verts)
    newneighs = intersect(neighbors(m, i), neighbors(m, j))
    push!(newneighs, i, j)
    for src in newneighs
        push!(neighbors_forward(m, src), dst)
        push!(neighbors(m, src), dst)
        push!(neighbors(m, dst), src)
    end
    return m
end

function cut_edge!(m, (i, j))
    @assert i < j
    fast_setdiff!(neighbors_forward(m, i), j)
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
#endregion

############################################################################################
# build_cliques
#region

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
        dst in neighs[dsts[id]] || return false
    end
    return true
end

#endregion

############################################################################################
# bands
#region

bands(h::AbstractHamiltonian, mesh::Mesh; solver = ES.LinearAlgebra(), kw...) =
    bands(bloch(h, solver), mesh; solver, kw...)

function bands(bloch::Bloch, basemesh::Mesh{SVector{L,T}};
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true, cleanorder = 0) where {T,L}

    # Step 1 - Diagonalize:
    baseverts = vertices(basemesh)
    meter = Progress(length(baseverts), "Step 1 - Diagonalizing: ")
    thread_solvers = [Eigensolver{T,L}(solver, bloch, mapping) for _ in 1:Threads.nthreads()]
    S = spectrumtype(bloch)
    spectra = Vector{S}(undef, length(baseverts))
    Threads.@threads for i in eachindex(baseverts)
        vert = baseverts[i]
        solver = thread_solvers[Threads.threadid()]
        spectra[i] = solver(vert)
        showprogress && ProgressMeter.next!(meter)
    end
    # Collect band vertices and store column offsets
    O = orbtype(bloch)
    bandverts = BandVertex{T,L,O}[]
    coloffsets = [0] # offsets for intervals of bandverts corresponding to each basevertex
    for (basevert, spectrum) in zip(baseverts, spectra)
        append_band_column!(bandverts, basevert, spectrum)
        push!(coloffsets, length(bandverts))
    end
    ProgressMeter.finish!(meter)

    # Step 2 - Knit seams:
    # Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will connect
    # to other subspaces s´ in columns of a neighboring base vertex. Connections are
    # possible if the projector ⟨s'|s⟩ has any singular value greater than 1/2
    meter = Progress(length(spectra), "Step 2 - Knitting: ")
    baseneighs = neighbors_forward(basemesh)
    bandneighs = [Int[] for _ in bandverts]
    crossed_seams = Tuple{Int,Int}[]
    for isrcbase in eachindex(spectra)
        for idstbase in baseneighs[isrcbase]
            crossed = knit_seam!(bandneighs, bandverts, spectra, coloffsets, isrcbase, idstbase)
            crossed && push!(crossed_seams, (isrcbase, idstbase))
        end
        showprogress && ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)

    # Step 3 - Clean seams:
    if cleanorder > 0
        meter = Progress(cleanorder, "Step 3 - Cleaning: ")
        onesolver = first(thread_solvers)
        bandneighs_all = neighbors_from_forward(bandneighs)
        basemesh = copy(basemesh)  # to avoid changing parent mesh
        num_original_baseverts = length(baseverts) # to distinguish new base vertices
        newcols = 0
        for (ind_cs, (isrcbase, idstbase)) in enumerate(crossed_seams)
            column_added = on_dislocation_add_column!((bandverts, spectra, coloffsets, basemesh),
                        isrcbase, idstbase, bandneighs, bandneighs_all, num_original_baseverts, onesolver)
            newcols += column_added
            if column_added && newcols <= cleanorder
                # zero-out the seam from crossed_seams, signaling it must not be processed again
                crossed_seams[ind_cs] = (0, 0)
                # first check if added base vertex is a neighbor of processed crossed_seams
                # if so, re-add it at the end to process again (since it is now connected to a new vertex)
                newvertex = length(spectra) # index of new vertex
                newneighs = neighbors(basemesh, newvertex)
                for ind_cs´ in 1:ind_cs-1
                    (i, j) = crossed_seams[ind_cs´]
                    i in newneighs && j in newneighs && push!(crossed_seams, (i, j))
                end
                # remove old seam neighbors
                cut_seam!(bandneighs, isrcbase, idstbase, coloffsets)
                idstbase´ = newvertex
                for isrcbase´ in newneighs  # neighbors of new vertex are new edge sources
                    crossed = knit_seam!(bandneighs, bandverts, spectra, coloffsets, isrcbase´, idstbase´)
                    crossed && push!(crossed_seams, (isrcbase´, idstbase´))
                    # grow and fill bandneighs_all with newly added bandneighs
                    start = length(bandneighs_all) + 1
                    foreach(_ -> push!(bandneighs_all, Int[]), start:length(bandverts))
                    # append_backward_neighbors!(bandneighs_all, bandneighs, start)
                    bandneighs_all = neighbors_from_forward(bandneighs)
                end
                showprogress && ProgressMeter.next!(meter)
            end
        end
        ProgressMeter.finish!(meter)
    end

    # Build band simplices
    bandsimps = build_cliques(bandneighs, L+1)
    orient_simplices!(bandsimps, bandverts)

    return Mesh(bandverts, bandneighs, bandsimps)
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

# Take two intervals (srcrange, dstrange) of bandverts (linked by base mesh)
# and fill bandneighs with their connections, using the projector colproj
# hascrossing signals some crossing of energies across the seam
function knit_seam!(bandneighs, bandverts, spectra, coloffsets, isrcbase, idstbase)
    srcrange = coloffsets[isrcbase]+1:coloffsets[isrcbase+1]
    dstrange = coloffsets[idstbase]+1:coloffsets[idstbase+1]
    colproj  = states(spectra[idstbase])' * states(spectra[isrcbase])
    idsthigh = 0
    hascrossing = false
    for isrc in srcrange
        src = bandverts[isrc]
        # available_connections = degeneracy(src)
        for idst in dstrange
            dst = bandverts[idst]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj)
            if connections > 0
                push!(bandneighs[isrc], idst)
                hascrossing = hascrossing || idst < idsthigh
                idsthigh = idst
            end
        end
        # available_connections == 0 || @show "incomplete"
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

function on_dislocation_add_column!((bandverts, spectra, coloffsets, basemesh),
    isrcbase, idstbase, bandneighs, bandneighs_all, num_original_baseverts, solver)
    # Check for dislocation
    srcrange = coloffsets[isrcbase]+1:coloffsets[isrcbase+1]
    dstrange = coloffsets[idstbase]+1:coloffsets[idstbase+1]
    # @show isrcbase, idstbase
    for isrc in srcrange, isrc´ in isrc+1:coloffsets[isrcbase+1]
        for idst in bandneighs[isrc], idst´ in bandneighs[isrc´]
            # Check whether band vertices are in correct column and whether they cross
            idst in dstrange && idst´ in dstrange && idst´ < idst || continue
            neigh_dst, neigh_dst´ = bandneighs[idst], bandneighs[idst´]
            # If cross neighbors (forward plus backward) have different intersections
            # there is a dislocation i.e. some simplex is frustrated
            is_near_dislocation =
                !equal_intersection(bandneighs_all, (isrc, idst´), (isrc´, idst)) ||
                seam_near_new_vertex(basemesh, isrcbase, idstbase, num_original_baseverts)
            # @show is_dislocation, is_near_dislocation, isrcbase, idstbase
            if is_near_dislocation
                # in such case, split base edge
                εs, εs´, εd, εd´ = energy.(getindex.(Ref(bandverts), (isrc, isrc´, idst, idst´)))
                ks, kd = vertices(basemesh, isrcbase), vertices(basemesh, idstbase)
                λ = (εs - εs´) / (εd´ - εs´ - εd + εs)
                k = ks + λ * (kd - ks)
                splitedge!(basemesh, (isrcbase, idstbase), k)
                spectrum = solver(k)
                push!(spectra, spectrum)
                # collect spectrum into a set of new band vertices
                append_band_column!(bandverts, k, spectrum)
                # add empty neighbor lists for new band vertices
                foreach(_ -> push!(bandneighs, Int[]), length(bandneighs)+1:length(bandverts))
                # update coloffsets
                push!(coloffsets, length(bandverts))
                # stop searching, report new column added
                return true
            end
        end
    end
    return false
end

function cut_seam!(bandneighs, isrcbase, idstbase, coloffsets)
    srcrange = coloffsets[isrcbase]+1:coloffsets[isrcbase+1]
    dstrange = coloffsets[idstbase]+1:coloffsets[idstbase+1]
    for isrc in srcrange
        fast_setdiff!(bandneighs[isrc], dstrange)
    end
    return bandneighs
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

function seam_near_new_vertex(basemesh, isrcbase, idstbase, num_original_vertices)
    for nsrc in neighbors(basemesh, isrcbase)
        nsrc <= num_original_vertices && continue
        nsrc in neighbors(basemesh, idstbase) && return true
    end
    return false
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
# splitbands
#region

# splitbands(b) = b

#endregion