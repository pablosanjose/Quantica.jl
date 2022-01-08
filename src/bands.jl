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
    delete_edge!(m, (i, j))
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

function delete_edge!(m, (i, j))
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
build_cliques(neighs, nverts) = build_cliques!(Vector{Int}[], neighs, nverts)

function build_cliques!(cliques, neighs, nverts)
    empty!(cliques)
    for (src, dsts) in enumerate(neighs)
        dsts_f = filter(>(src), dsts)  # indexable forward neighbors
        for ids in Combinations(length(dsts_f), nverts - 1)
            if all_adjacent(ids, dsts_f, neighs)
                clique = prepend!(dsts_f[ids], src)
                push!(cliques, clique)
            end
        end
    end
    return cliques
end

# Check whether dsts_f[ids] are all mutual neighbors. ids are a total of nverts-1 indices
# of dsts_f = neighbor_forward(neighs, src)
function all_adjacent(ids, dsts_f, neighs)
    nids = length(ids)
    for (n, id) in enumerate(ids), n´ in n+1:nids
        dst = dsts_f[ids[n´]]
        dst in neighs[dsts_f[id]] || return false
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
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true, patchlevel = 0, degtol = missing) where {T,L}

    basemesh = copy(basemesh) # will be part of Band, possibly refined
    S = spectrumtype(bloch)
    spectra = Vector{S}(undef, length(vertices(basemesh)))
    O = orbtype(bloch)
    bandverts = BandVertex{T,L,O}[]
    bandneighs = Vector{Int}[]
    coloffsets = Int[]
    solvers = [apply(solver, bloch, SVector{L,T}, mapping) for _ in 1:Threads.nthreads()]
    crossed = NTuple{6,Int}[] # isrcbase, idstbase, isrc, isrc´, idst, idst´
    crossed_frust = similar(crossed)
    crossed_frust_neigh = similar(crossed)
    data = (; basemesh, spectra, bandverts, bandneighs, coloffsets, solvers,
              crossed, crossed_frust, crossed_frust_neigh, patchlevel, showprogress, degtol)

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
    # mesh connectivity that results in missing simplices. We patch these defects by
    # recursively refining (a copy of) basemesh to a certain patchlevel, and rediagonalizing
    # bloch at each new vertex. A sufficiently high patchlevel will usually converge to a
    # band mesh without defects
    band_patch!(data)

    # Build band simplices
    bandsimps = build_cliques(bandneighs, L+1)
    orient_simplices!(bandsimps, bandverts)
    # Rebuild basemesh simplices
    build_cliques!(simplices(basemesh), neighbors(basemesh), L+1)

    ndefects = length(data.crossed_frust)
    iszero(ndefects) || @warn "Band with $ndefects dislocation defects. Consider increasing `patchlevel`"

    bandmesh = Mesh(bandverts, bandneighs, bandsimps, )
    return Band(bandmesh, basemesh, solvers)
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
    subs = data.degtol === missing ? collect(approxruns(energies´)) :
                                     collect(approxruns(energies´, data.degtol))
    for (i, rng) in enumerate(subs)
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(data.bandverts, BandVertex(basevert, energy, state))
    end
    push!(data.coloffsets, length(data.bandverts))
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
function knit_seam!(data, isrcbase, idstbase)
    srcrange = column_range(data, isrcbase)
    dstrange = column_range(data, idstbase)
    colproj  = states(data.spectra[idstbase])' * states(data.spectra[isrcbase])
    for isrc in srcrange
        src = data.bandverts[isrc]
        for idst in dstrange
            dst = data.bandverts[idst]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj)
            if connections > 0
                push!(data.bandneighs[isrc], idst)
                push!(data.bandneighs[idst], isrc)
                # populate crossed with all crossed links
                for isrc´ in first(srcrange):isrc-1, idst´ in data.bandneighs[isrc´]
                    idst´ in dstrange || continue
                    # if crossed, push! with ordered isrc´ < isrc, idst´ > idst
                    idst´ > idst && push!(data.crossed,
                                         (isrcbase, idstbase, isrc´, isrc, idst´, idst))
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
    classify_crossed!(data)
    data.patchlevel == 0 && return data
    meter = 0 < data.patchlevel < Inf ? Progress(data.patchlevel, "Step 3 - Patching: ") :
                                        ProgressUnknown("Step 3 - Patching: ")
    baseverts = vertices(data.basemesh)
    newcols = 0
    done = false
    cf, cfn = data.crossed_frust, data.crossed_frust_neigh
    solver = first(data.solvers)
    while !isempty(cf) && !done
        (ib, jb, i, i´, j, j´) = isempty(cfn) ? popfirst!(cf) : popfirst!(cfn)
        # check edge has not been previously split
        jb in neighbors(data.basemesh, ib) || continue
        newcols += 1
        done = newcols == data.patchlevel
        # remove all bandneighs in this seam
        delete_seam!(data, ib, jb)
        # compute crossing momentum
        εi, εi´, εj, εj´ = energy.(getindex.(Ref(data.bandverts), (i, i´, j, j´)))
        ki, kj = vertices(data.basemesh, ib), vertices(data.basemesh, jb)
        λ = (εi - εi´) / (εj´ - εi´ - εj + εi)
        k = ki + λ * (kj - ki)
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
        idstbase = newbasevertex
        for isrcbase in newbaseneighs  # neighbors of new vertex are new edge sources
            knit_seam!(data, isrcbase, idstbase)
        end
        # classifies crossed band neighbors into frustrated (cf) and their neighbors (cfn)
        classify_crossed!(data)
        data.showprogress && ProgressMeter.next!(meter)
    end
    data.showprogress && ProgressMeter.finish!(meter)
    return data
end

function classify_crossed!(data)
    for (n, (ib, jb, i, i´, j, j´)) in enumerate(data.crossed)
        if is_frustrated_crossing(data, (i, i´), (j, j´))
            push!(data.crossed_frust, (ib, jb, i, i´, j, j´))
            data.crossed[n] = (0, 0, 0, 0, 0, 0)
        end
    end
    for (ib, jb, i, i´, j, j´) in data.crossed
        iszero(ib) && continue
        for (_, _, fi, fi´, fj, fj´) in data.crossed_frust
            if sorteq((i, i´), (fi, fi´)) || sorteq((i, i´), (fj, fj´)) ||
               sorteq((j, j´), (fi, fi´)) || sorteq((j, j´), (fj, fj´))
                push!(data.crossed_frust_neigh, (ib, jb, i, i´, j, j´))
                break
            end
        end
    end
    empty!(data.crossed)
    return data
end

sorteq((i, i´), (j, j´)) = (i == j && i´ == j´) || (i == j´ && i´ == j)

function delete_seam!(data, isrcbase, idstbase)
    srcrange = column_range(data, isrcbase)
    dstrange = column_range(data, idstbase)
    for isrc in srcrange
        fast_setdiff!(data.bandneighs[isrc], dstrange)
    end
    for idst in dstrange
        fast_setdiff!(data.bandneighs[idst], srcrange)
    end
    return data
end

function is_frustrated_crossing(data, (i, i´), (j, j´))
    ns = data.bandneighs
    return !equal_intersection((ns[i], ns[j´]), (ns[i´], ns[j]))
end

# Define s = (s₁ ,s₂), p = (p₁, p₂). Is s₁ ∩ s₂ == p₁ ∩ p₂ ?
# This happens iif (s₁ ∩ s₂) ∈ pᵢ and (p₁ ∩ p₂) ∈ sᵢ
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

#endregion