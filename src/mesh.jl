############################################################################################
# mesh methods
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs = marching_neighbors(cinds)          # sorted neighbors of i, with n[i][j] > i
    return Mesh{L+1}(verts, neighs)
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
    sort!(newneighs)
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

# delete elements in c if it is not in rng
function fast_setdiff!(c::Vector, rng)
    i = 0
    for x in c
        x in rng && continue
        i += 1
        c[i] = x
    end
    resize!(c, i)
    return c
end

# delete elements in c and d if the one in c is not in rng
function fast_setdiff!((c, d)::Tuple{Vector,Vector}, rng)
    i = 0
    for (i´, x) in enumerate(c)
        x in rng && continue
        i += 1
        c[i] = x
        d[i] = d[i´]
    end
    resize!(c, i)
    resize!(d, i)
    return (c, d)
end

# groups of n all-to-all connected neighbors, sorted
build_cliques(neighs, ::Val{N}) where {N} = rebuild_cliques!(NTuple{N,Int}[], neighs)

rebuild_cliques!(mesh::Mesh) = rebuild_cliques!(simplices(mesh), neighbors(mesh))

function rebuild_cliques!(cliques::Vector{NTuple{N,Int}}, neighs) where {N}
    empty!(cliques)
    for (src, dsts) in enumerate(neighs)
        dsts_f = filter(>(src), dsts)  # indexable forward neighbors
        for ids in Combinations(length(dsts_f), N - 1)
            if all_adjacent(ids, dsts_f, neighs)
                # clique = (src, dsts_f[ids]...), but non-allocating
                clique = ntuple(i -> i == 1 ? src : dsts_f[ids[i - 1]], Val(N))
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

# ensure simplex orientation has normals pointing towards positive z
function orient_simplices!(simplices, vertices::Vector{B}) where {E,B<:BandVertex{<:Any,E}}
    E <= 2 && return simplices
    for (s, simplex) in enumerate(simplices)
        if E > 2
            k0 = base_coordinates(vertices[simplex[1]])
            edges = ntuple(i -> base_coordinates(vertices[simplex[i+1]])-k0, Val(E-1))
            volume = det(hcat(edges...))
            if volume < 0
                simplices[s] = switchlast(simplex)
            end
        end
    end
    return simplices
end

switchlast(s::NTuple{N}) where {N} = ntuple(i -> i < N-1 ? s[i] : s[2N-i-1], Val(N))

# Computes connected subsets from a list of neighbors, in the form of a (vsinds, svinds)
# vsinds::Vector{Int} is the subset index for each band vertex
# svinds::Vector{Vector{Int}} is a list of vertex indices for each subset
function subsets(neighs::Vector{Vector{Int}})
    vsinds = zeros(Int, length(neighs))
    svinds = Vector{Int}[]
    sidx = 0
    vidx = 1
    while vidx !== nothing
        sidx += 1
        sv = [vidx]
        push!(svinds, sv)
        vsinds[vidx] = sidx
        for i in sv, j in neighs[i]
            iszero(vsinds[j]) || continue
            vsinds[j] = sidx
            push!(sv, j)
        end
        sort!(sv)
        vidx = findfirst(iszero, vsinds)
    end
    return vsinds, svinds
end

#endregion
