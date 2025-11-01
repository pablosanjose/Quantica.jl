############################################################################################
# spectrum
#region

function spectrum(h::AbstractHamiltonian{T}, φs; solver = ES.LinearAlgebra(), transform = missing, kw...) where {T}
    os = blockstructure(h)
    mapping = (φ...) -> ftuple(φ...; kw...)
    φs´ = sanitize_SVector(T, φs)
    S = typeof(φs´)
    asolver = apply(solver, h, S, mapping, transform)
    eigen = asolver(φs´)
    return Spectrum(eigen, os)
end

spectrum(h::AbstractHamiltonian{T,<:Any,0}; kw...) where {T} =
    spectrum(h, SVector{0,T}(); kw...)

function spectrum(b::Bandstructure, φs;)
    os = blockstructure(b)
    solver = first(solvers(b))
    eigen = solver(φs)
    return Spectrum(eigen, os)
end

#endregion

############################################################################################
# Spectrum indexing
#region

Base.first(s::Spectrum) = energies(s)
Base.last(s::Spectrum) = states(s)
Base.iterate(s::Spectrum) = first(s), Val(:states)
Base.iterate(s::Spectrum, ::Val{:states}) = last(s), Val(:done)
Base.iterate(::Spectrum, ::Val{:done}) = nothing
Base.Tuple(s::Spectrum) = (first(s), last(s))

Base.getindex(s::Spectrum, i...; around = missing) = get_around(Tuple(s), around, i...)

get_around((es, ss), ε0::Missing, i) = (es[i], ss[:,i])
get_around(s, ε0::Number) = get_around(s, ε0, 1)
get_around(s, ε0::Number, i::Integer) = get_around(s, ε0, i:i)

function get_around((es, ss), ε0::Number, inds)
    # Get indices of eachindex(es) such that if sorted by `by` will occupy `inds` positions
    if inds isa Union{Integer,OrdinalRange}
        rngs = partialsort(eachindex(es), inds, by = rng -> abs(es[rng] - ε0))
        return (es[rngs], ss[:, rngs])
    else # generic inds (cannot use partialsort)
        rngs = sort(eachindex(es), by = rng -> abs(es[rng] - ε0))
        return (es[view(rngs, inds)], ss[:, view(rngs, inds)])
    end
end

#endregion

############################################################################################
# bands(h, points...; kw...)
#   h can be an AbstractHamiltonian or a Function(vertex) -> AbstractMatrix
#region

bands(rng, rngs...; kw...) = h -> bands(h, rng, rngs...; kw...)

bands(h::Function, rng, rngs...; kw...) = bands(h, mesh(rng, rngs...); kw...)
bands(h::AbstractHamiltonian{T}, rng, rngs::Vararg{Any,L´}; kw...) where {T,L´} =
    bands(h, convert(Mesh{SVector{L´+1,T}}, mesh(rng, rngs...)); kw...)

bands(h::AbstractHamiltonian{<:Any,<:Any,L}; kw...) where {L} =
    bands(h, default_band_ticks(Val(L))...; kw...)

default_band_ticks(::Val{L}) where {L} = ntuple(Returns(subdiv(-π, π, 49)), Val(L))

function bands(h::AbstractHamiltonian, mesh::Mesh{S};
         solver = ES.LinearAlgebra(), transform = missing, mapping = missing, metadata = missing, kw...) where {S<:SVector}
    mapping´ = sanitize_mapping(mapping, h)
    metadata´ = BandMetadataGenerator(metadata)
    solvers = eigensolvers_thread_pool(solver, h, S, mapping´, transform)
    hf = apply_map(mapping´, h, S)
    mf = apply_map(mapping´, metadata´, S)
    ss = subbands(hf, mf, solvers, mesh; kw...)
    os = blockstructure(h)
    return Bandstructure(ss, solvers, os)
end

function bands(h::Function, mesh::Mesh{S};
         solver = ES.LinearAlgebra(), transform = missing, mapping = missing, metadata = missing, kw...) where {S<:SVector}
    mapping´ = sanitize_mapping(mapping, h)
    metadata´ = BandMetadataGenerator(metadata)
    solvers = eigensolvers_thread_pool(solver, h, S, mapping´, transform)
    hf = apply_map(mapping´, h, S)
    mf = apply_map(mapping´, metadata´, S)
    ss = subbands(hf, mf, solvers, mesh; kw...)
    return ss
end

function eigensolvers_thread_pool(solver, h, S, mapping, transform)
    # if h::Function we cannot be sure it is thread-safe
    nsolvers = ES.is_thread_safe(solver) && h isa AbstractHamiltonian ? Threads.nthreads() : 1
    solvers = [apply(solver, copy_if_hamiltonian(h), S, mapping, transform) for _ in 1:nsolvers]
    return solvers
end

copy_if_hamiltonian(h::AbstractHamiltonian) = minimal_callsafe_copy(h)
copy_if_hamiltonian(f) = f

function subbands(hf, mf, solvers, basemesh::Mesh{SVector{L,T}};
         showprogress = true, defects = (), patches = 0, degtol = missing, split = true, warn = true, projectors = false) where {T,L}
    defects´ = sanitize_Vector_of_SVectors(SVector{L,T}, defects)
    degtol´ = degtol isa Number ? degtol : sqrt(eps(real(T)))
    subbands = subbands_precompilable(hf, mf, solvers, basemesh, showprogress, defects´, patches, degtol´, split, warn, projectors)
    return subbands
end

sanitize_mapping(mapping, ::AbstractHamiltonian{<:Any,<:Any,L}) where {L} =
    sanitize_mapping(mapping, Val(L))
sanitize_mapping(mapping::Union{Missing,Function}, ::Function) = mapping
sanitize_mapping(_, ::Function) =
    argerror("Cannot apply this mapping with a function input")
sanitize_mapping(::Missing, ::Val) = missing
sanitize_mapping(f::Function, ::Val) = f
sanitize_mapping(pts::NTuple{N,Any}, ::Val{L}) where {N,L} =
    sanitize_mapping(ntuple(i -> i-1, Val(N)) => parsenode.(pts, Val(L)), Val(L))
sanitize_mapping((xs, nodes)::Pair, ::Val{L}) where {L} =
    polygonpath(xs, parsenode.(nodes, Val(L)))
sanitize_mapping((xs, nodes)::Pair{X,S}, ::Val{L}) where {N,L,T,X<:NTuple{N,Real},S<:NTuple{N,SVector{L,T}}} =
    polygonpath(xs, nodes)

function subbands_precompilable(hf::FunctionWrapper, mf::FunctionWrapper{M}, solvers::Vector{A}, basemesh::Mesh{SVector{L,T}},
    showprogress, defects, patches, degtol, split, warn, projectors) where {T,L,A<:AppliedEigenSolver{T,L},M}

    basemesh = copy(basemesh) # will become part of Band, possibly refined
    eigens = Vector{EigenComplex{T}}(undef, length(vertices(basemesh)))
    bandverts = BandVertex{T,L+1,M}[]
    bandneighs = Vector{Int}[]
    bandneideg = similar(bandneighs)
    coloffsets = Int[]
    crossed = NTuple{6,Int}[]
    frustrated = similar(crossed)
    subbands = Subband{T,L+1,M}[]
    data = (; hf, mf, solvers, basemesh, eigens, bandverts, bandneighs, bandneideg, coloffsets,
              L, crossed, frustrated, subbands, defects, patches, showprogress, degtol,
              split, warn, projectors)

    # Step 1 - Diagonalize:
    # Uses multiple SpectrumSolvers (one per Julia thread) to diagonalize h at each
    # vertex of basemesh. Then, it collects each of the produced Spectrum (aka "columns")
    # into a bandverts::Vector{BandVertex}, recording the coloffsets for each column
    blasthreads = BLAS.get_num_threads()
    Threads.nthreads() == 1 || BLAS.set_num_threads(1)  # One BLASthread if JULIAthreads > 1
    subbands_diagonalize!(data)
    BLAS.set_num_threads(blasthreads)                   # Restore BLAS threads

    # Step 2 - Knit seams:
    # Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will
    # connect to other subspaces s´ in columns of a neighboring base vertex. Connections are
    # possible if the projector ⟨s'|s⟩ has any singular value greater than 1/2
    subbands_knit!(data)

    # Step 3 - Patch seams:
    # Dirac points and other topological band defects will usually produce dislocations in
    # mesh connectivity that results in missing simplices.
    insert_defects!(data)
    if L>1
        subbands_patch!(data)
    end

    # Step 4 - Split subbands:
    # Split disconnected subgraphs, rebuild their neighbor lists and convert to Subbands.
    # As part of the Subband conversion, subband simplices are computed.
    subbands_split!(data)

    # Step 5 - Compute projectors:
    # Each simplex s has a continuous matrix Fₛ(k) = ψₛ(k)ψₛ(k)⁺ that interpolates between
    # the Fₛ(kᵢ) at each vertex i. We compute Pˢᵢ such that F(kᵢ) = φₛ(kᵢ)*Pˢᵢ*φₛ(kᵢ)⁺ for
    # any vertex i of s with a degenerate eigenbasis φ(kᵢ). Non-degenerate vertices have P=1
    # Required to use a Bandstructure as a GreenSolver
    subband_projectors!(data)

    return subbands
end

#endregion

############################################################################################
# parsenode
#region

parsenode(pt::SVector, ::Val{L}) where {L} = padright(pt, Val(L))
parsenode(pt::Tuple, val) = parsenode(SVector(float.(pt)), val)

function parsenode(node::Symbol, val)
    pt = get(BZpoints, node, missing)
    pt === missing && throw(ArgumentError("Unknown Brillouin zone point $pt, use one of $(keys(BZpoints))"))
    pt´ = parsenode(pt, val)
    return pt´
end

const BZpoints =
    ( Γ  = (0,)
    , X  = (pi,)
    , Y  = (0, pi)
    , Z  = (0, 0, pi)
    , K  = (2pi/3, -2pi/3)
    , K´ = (4pi/3, 2pi/3)
    , M  = (pi, 0)
    )

#endregion

############################################################################################
# polygonpath
#region

polygonpath(xs, nodes) = polygonpath(sanitize_polygonpath(xs), sanitize_polygonpath(nodes))

function polygonpath(xs::AbstractVector, nodes::AbstractVector)
    sanitize_polygonpath!(xs, nodes)
    minx, maxx = extrema(xs)
    mapping = x -> begin
        x´ = clamp(only(x), minx, maxx)
        i = argmin(i -> ifelse(x´ <= xs[i], Inf, x´ - xs[i]), eachindex(xs))
        p = nodes[i] + (nodes[i+1] - nodes[i]) * (x´ - xs[i]) / (xs[i+1] - xs[i])
        return p
    end
    return mapping
end

sanitize_polygonpath(xs::AbstractVector) = xs
sanitize_polygonpath(xs::Tuple) = collect(xs)

function sanitize_polygonpath!(xs, nodes)
    if !issorted(xs)
        p = sortperm(xs)
        permute!(xs, p)
        permute!(nodes, p)
    end
    return nothing
end

#endregion

############################################################################################
# subdiv
#region

subdiv(nodes, pts::Integer) = subdiv(nodes, [pts for _ in 1:length(nodes)-1])

subdiv(x1, x2, pts) = collect(range(x1, x2, length = pts))

function subdiv(nodes, pts)
    length(pts) == length(nodes) - 1 ||
        argerror("`subdiv(nodes, pts)` requires `length(pts) == length(nodes) - 1` or pts::Integer")
    rng = [x for (n, pt) in enumerate(pts) for x in range(nodes[n], nodes[n+1], length = pt) if x != nodes[n+1]]
    push!(rng, last(nodes))
    return rng
end

#endregion

############################################################################################
# subbands_diagonalize!
#region

function subbands_diagonalize!(data)
    baseverts = vertices(data.basemesh)
    meter = Progress(length(baseverts); desc = "Step 1 - Diagonalizing: ")
    push!(data.coloffsets, 0) # first element
    if length(data.solvers) > 1
        Threads.@threads :static for i in eachindex(baseverts)
            vert = baseverts[i]
            solver = data.solvers[Threads.threadid()]
            data.eigens[i] = solver(vert)
            data.showprogress && ProgressMeter.next!(meter)
        end
    else
        solver = first(data.solvers)
        for i in eachindex(baseverts)
            vert = baseverts[i]
            data.eigens[i] = solver(vert)
            data.showprogress && ProgressMeter.next!(meter)
        end
    end
    # Collect band vertices and store column offsets
    for (basevert, eigen) in zip(baseverts, data.eigens)
        append_bands_column!(data, basevert, eigen)
    end
    ProgressMeter.finish!(meter)
end

const degeneracy_warning = 16

# collect eigen into a band column (vector of BandVertices for equal base vertex)
function append_bands_column!(data, basevert, eigen)
    T = eltype(basevert)
    energies, states = eigen
    subs = collect(approxruns(energies, data.degtol))
    for (i, rng) in enumerate(subs)
        deg = length(rng)
        data.warn && data.projectors && deg > degeneracy_warning &&
            @warn "Encountered a highly degenerate point in bandstructure (deg = $deg), which will likely slow down the computation of projectors"
        state = orthonormalize!(view(states, :, rng))
        energy = mean(i -> energies[i], rng)
        metadata = data.mf(basevert, eigen, rng)
        push!(data.bandverts, BandVertex(basevert, energy, state, metadata))
    end
    push!(data.coloffsets, length(data.bandverts))
    foreach(_ -> push!(data.bandneideg, Int[]), length(data.bandneideg)+1:length(data.bandverts))
    foreach(_ -> push!(data.bandneighs, Int[]), length(data.bandneighs)+1:length(data.bandverts))
    return data
end

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
# subbands_knit!
#   Take two eigenpair columns connected on basemesh, and add edges between closest
#   eigenpair using projections (svd).
#region

function subbands_knit!(data)
    meter = Progress(length(data.eigens); desc = "Step 2 - Knitting: ")
    for isrcbase in eachindex(data.eigens)
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
function knit_seam!(data, ib, jb)
    srcrange = column_range(data, ib)
    dstrange = column_range(data, jb)
    _, statesib = data.eigens[ib]
    _, statesjb = data.eigens[jb]
    colproj  = statesjb' * statesib
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

# Number of singular values greater than √min_squared_overlap in proj = ψj'*ψi
# Fast rank-1 |svd|^2 is r = tr(proj'proj). For higher ranks and r > 0 we must compute and
# count singular values. The min_squared_overlap is arbitrary, and is fixed heuristically to
# a high enough value to connect in the coarsest base lattices
function connection_rank(proj)
    min_squared_overlap = 0.499
    rankf = sum(abs2, proj) # = tr(proj'proj)
    fastrank = ifelse(rankf > min_squared_overlap, 1, 0)  # For rank=1 proj: upon doubt, connect
    if iszero(fastrank) || size(proj, 1) == 1 || size(proj, 2) == 1
        return fastrank
    else
        sv = svdvals(proj)
        return count(s -> abs2(s) > min_squared_overlap, sv)
    end
end

column_range(data, ibase) = data.coloffsets[ibase]+1:data.coloffsets[ibase+1]

#endregion

############################################################################################
# subbands_patch!
#   Remove dislocations in mesh edges by edge splitting (adding patches)
#region

function subbands_patch!(data)
    data.patches > 0 || return data
    queue_frustrated!(data)
    data.warn && isempty(data.defects) &&
        @warn "Trying to patch $(length(data.frustrated)) band dislocations without a list `defects` of defect positions."
    meter = data.patches < Inf ? Progress(data.patches; desc = "Step 3 - Patching: ") :
                                 ProgressUnknown(; desc = "Step 3 - Patching: ")
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
    newcols > 0 && rebuild_cliques!(data.basemesh)
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
    (ib, jb) = find_closest_edge(kdefect, base)
    insert_column!(data, (ib, jb), kdefect)
    return data
end

function find_closest_edge(kdefect::SVector{1}, base)
    kx = only(kdefect)
    for i in eachindex(vertices(base)), j in neighbors(base, i)
        only(vertices(base, i)) < kx < only(vertices(base, j)) && return (i, j)
    end
    argerror("Defects in 1D lattices should be contained inside the lattice, but the provided $kdefect is not")
    return (0, 0)
end

function find_closest_edge(kdefect, base)
    (ib, jb) = argmin(((i, j) for i in eachindex(vertices(base)) for j in neighbors(base, i))) do (i, j)
        sum(abs2, 0.5*(vertices(base, i) + vertices(base, j)) - kdefect)
    end
    return (ib, jb)
end

function insert_column!(data, (ib, jb), k)
    solver = first(data.solvers)
    # remove all bandneighs in this seam
    delete_seam!(data, ib, jb)
    # create new vertex in basemesh by splitting the edge, and connect to neighbors
    split_edge!(data.basemesh, (ib, jb), k)
    # compute spectrum at new vertex
    spectrum = solver(k)
    push!(data.eigens, spectrum)
    # collect spectrum into a set of new band vertices
    append_bands_column!(data, k, spectrum)
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
    εi, εi´, εj, εj´ = real.(energy.(getindex.(Ref(data.bandverts), (i, i´, j, j´))))
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
# subbands_split!
#   We have vertex neighbors (data.bandneighs). Now we need to use these to cluster vertices
#   into disconnected subbands (i.e. vertex `subsets`). Once we have these, split vertices
#   into separate lists and rebuild neighbors using indices of these new lists
#region

function subbands_split!(data)
    if data.split
        # vsinds are the subband index of each vertex index
        # svinds is lists of band vertex indices that belong to the same subband
        vsinds, svinds = subsets(data.bandneighs)
        meter = Progress(length(svinds); desc = "Step 4 - Splitting: ")
        new2old = sortperm(vsinds)
        old2new = invperm(new2old)
        offset = 0
        for subset in svinds
            # avoid subbands with no simplices
            if length(subset) > data.L
                sverts  = data.bandverts[subset]
                sneighs = [ [old2new[dstold] - offset for dstold in data.bandneighs[srcold]]
                        for srcold in subset]
                sband = Subband(sverts, sneighs)
                isempty(sband) || push!(data.subbands, sband)
            end
            offset += length(subset)
            data.showprogress && ProgressMeter.next!(meter)
        end
    else
        sband = Subband(data.bandverts, data.bandneighs)
        isempty(sband) || push!(data.subbands, sband)
    end
    return data
end

#endregion

############################################################################################
# subband_projectors!(s::Subband, hf::Function)
#   Fill dictionary of projector matrices for each degenerate vertex in each simplex of s
#       Dict([(simp_index, vert_index) => φᵢ * P])
#   such that P = U PD U'. U columns are eigenstates of φᵢ hf(k_av) φᵢ⁺, i = vert_index,
#   φᵢ are vertex eigenstate matrices, k_av is the basecoordinate at the center of the
#   simplex, and PD is a Diagonal{Bool} that filters out M columns of U, so that only
#   deg_simplex = minimum(deg_verts) remain. The criterion to filter out the required
#   eigenstates is to order them is decreasing energy distance between their eigenvalue ε_av
#   and the simplex mean energy mean(ε_j). No φᵢ overlap criterion is used because we need
#   to fix the exact number of eliminations.
#region

function subband_projectors!(data)
    nsimps = sum(s -> length(simplices(s)), data.subbands)
    meter = Progress(nsimps; desc = "Step 5 - Projectors: ")
    if data.projectors
        for s in data.subbands
            subband_projectors!(s, data.hf, meter, data.showprogress)
        end
    end
    return data
end

function subband_projectors!(s::Subband{T}, hf, meter, showprogress) where {T}
    projstates = projected_states(s)
    isempty(projstates) || return s
    verts = vertices(s)
    simps = simplices(s)
    # a random symmetry-breaking perturbation common to all simplices
    perturbation = Complex{T}[]
    for (sind, simp) in enumerate(simps)
        degs = degeneracy.(getindex.(Ref(verts), simp))
        if any(>(1), degs)
            kav = mean(i -> base_coordinates(verts[i]), simp)
            εav = mean(i -> energy(verts[i]), simp)
            hkav = hf(kav)
            nzs = nonzeros(hkav)
            nnzs = length(nzs)
            nnzs == length(perturbation) || resize_perturbation!(perturbation, nnzs)
            nzs .+= perturbation     # in-place allowed since hkav gets updated on each call
            mindeg = minimum(degs)
            for (vind, deg) in zip(simp, degs)
                if deg > 1  # diagonalize vertex even if all degs are equal and > 1
                    φP = simplex_projector(hkav, verts, vind, εav, mindeg)
                    projstates[(sind, vind)] = φP
                end
            end
        end
        showprogress && ProgressMeter.next!(meter)
    end
    return projstates
end

function resize_perturbation!(p::Vector{C}, n) where {C}
    l = length(p)
    @assert n > l   # perturbation length should not decrease
    resize!(p, n)
    η = sqrt(eps(real(C)))  # makes the perturbation small
    for i in l+1:n
        p[i] = η * rand(C)
    end
    return p
end

function simplex_projector(hkav, verts, vind, εav, mindeg)
    φ = states(verts[vind])
    hproj = φ' * hkav * φ
    _, P = maybe_eigen!(Hermitian(hproj), sortby = ε -> abs(ε - εav))
    Pthin = view(P, :, 1:mindeg)
    return φ * Pthin
end

maybe_eigen!(A::AbstractMatrix{<:LinearAlgebra.BlasComplex}; kw...) = eigen!(A; kw...)
maybe_eigen!(A; kw...) = eigen(A; kw...)    # generic fallback for e.g. Complex16

#endregion

############################################################################################
# Subband slicing and indexing
#   Example: in a 2D lattice, subband[(kx,ky,:)] is a vertical slice at fixed momentum kx, ky
#region

Base.getindex(b::Bandstructure, i) = subbands(b, i)
Base.getindex(b::Bandstructure, xs::Tuple) = [s[xs] for s in subbands(b)]
Base.getindex(s::Subband, xs::Tuple) = Subband(slice(s, xs, Val(true)))

Base.lastindex(b::Bandstructure, args...) = lastindex(subbands(b), args...)

slice(b::Bandstructure, xs) = [slice(s, xs) for s in subbands(b)]
slice(s::Subband, xs::Tuple) = slice(s, xs, Val(false))

# getindex default (Val{true}): mesh with reduced embedding dimension = simplex length + 1
slice(s::Subband, xs::Tuple, ::Val{true}) = slice(s, perp_axes(s, xs...), slice_axes(s, xs...))
# slice default (Val{false}): mesh with same embedding dimension as subband and smaller simplex length
slice(s::Subband, xs::Tuple, ::Val{false}) = slice(s, perp_axes(s, xs...), all_axes(s))

function slice(subband::Subband{<:Any,E}, paxes::NTuple{N}, saxes::Tuple) where {E,N}
    isempty(saxes) && argerror("The slice must have at least one unconstrained axis")
    isempty(paxes) && return mesh(subband)
    maximum(first, paxes) <= embdim(subband) && maximum(saxes) <= embdim(subband) ||
        argerror("Cannot slice subband along more than $(embdim(subband)) axes")
    V = slice_vertex_type(subband, saxes)
    S = slice_skey_type(paxes)
    verts = V[]
    neighs = Vector{Int}[]
    vinds = Dict{S,Int}()
    vindstemp = Int[]
    subtemp = Int[]
    data = (; subband, paxes, saxes, verts, neighs, vinds, vindstemp, subtemp)

    foreach_simplex(subband, paxes) do sind
        simp = simplices(subband, sind)
        slice_simplex!(data, simp)
    end
    return Mesh{E-N}(verts, neighs)
end

perp_axes(::Subband{T}, xs...) where {T} = perp_axes(T, 1, xs...)
perp_axes(T::Type, dim, ::Colon, xs...) = perp_axes(T, dim + 1, xs...)
perp_axes(T::Type, dim, x::Number, xs...) = ((dim, T(x)), perp_axes(T, dim + 1, xs...)...)
perp_axes(T::Type, dim) = ()

slice_axes(::Subband{<:Any,E}, xs...) where {E} = slice_axes(1, padtuple(xs, :, Val(E))...)
slice_axes(dim::Int, ::Number, xs...) = slice_axes(dim + 1, xs...)
slice_axes(dim::Int, ::Colon, xs...) = (dim, slice_axes(dim + 1, xs...)...)
slice_axes(dim::Int) = ()

all_axes(::Subband{<:Any,E}) where {E} = ntuple(identity, Val(E))

slice_vertex_type(::Subband{T,<:Any}, ::NTuple{N}) where {T,N} = BandVertex{T,N,Missing}

slice_skey_type(::NTuple{N}) where {N} = SVector{N+1,Int}

function slice_simplex!(data, simp)
    empty!(data.vindstemp)
    perpaxes = SVector(first.(data.paxes))
    paraxes  = SVector(data.saxes)
    k = SVector(last.(data.paxes))
    sub = data.subtemp
    for sub´ in Combinations(length(simp), length(perpaxes) + 1)
        # subsimplex must have minimal degeneracy on first vertex
        copy!(sub, sub´)  # cannot modify sub´ because it also acts as state in Combinations
        sort!(sub, by = i -> degeneracy(vertices(data.subband, simp[i])))
        key = vindskey(data.paxes, simp, sub)
        if !haskey(data.vinds, key)
            kε0, edgemat = vertmat_simplex(data.paxes, vertices(data.subband), simp, sub)
            dvper = edgemat[perpaxes, :]
            λs = dvper \ (k - kε0[perpaxes])
            sum(λs) < 1 && all(>=(0), λs) || continue
            dvpar = edgemat[paraxes, :]
            kε = kε0[paraxes] + dvpar * λs
            φ = interpolate_state_along_edges(λs, vertices(data.subband), simp, sub)
            push!(data.verts, BandVertex(kε, φ, missing))
            push!(data.neighs, Int[])
            vind = length(data.verts)
            data.vinds[key] = vind
            push!(data.vindstemp, vind)
        else
            vind = data.vinds[key]
            push!(data.vindstemp, vind)
        end
    end
    # all-to-all among new vertices
    for i in data.vindstemp, j in data.vindstemp
        i == j || push!(data.neighs[i], j)
    end
    return data
end

# a sorted SVector of the N+1 parent vertex indices is = simp[sub] (N = num perp slice axes)
# identifying each distinct vertex in slice -> then, vinds[is] = new vertex index
vindskey(::NTuple{N}, simp, sub) where {N} = sort(SVector(ntuple(i -> simp[sub[i]], Val(N+1))))

function vertmat_simplex(::NTuple{N}, vs, simp, sub) where {N}
    kε0 = coordinates(vs[simp[sub[1]]])
    mat = hcat(ntuple(i -> coordinates(vs[simp[sub[i+1]]]) - kε0, Val(N))...)
    return kε0, mat
end

# we assume that sub is ordered and that simp is sorted so that the first vertex has minimal
# degeneracy within the simplex (note that orient_simplices! preserves first vertex)
function interpolate_state_along_edges(λs, vs, simp, sub)
    v0 = vs[simp[sub[1]]]
    φ0 = states(v0)
    deg0 = degeneracy(v0)
    φ = copy(φ0)
    φ .*= 1 - sum(λs)
    for (i, λi) in enumerate(λs)
        vi = vs[simp[sub[i+1]]]
        φi = states(vi)
        degi = degeneracy(vi)
        if  degi == deg0
            φ .+= λi .* φi
        elseif degi > deg0
            proj = φi'φ0  # size(proj) == (degi, deg0)
            Q = qr!(proj).Q * Matrix(I, degi, deg0)
            mul!(φ, φi, Q, λi, 1)
        else
            throw(ErrorException("Unexpected simplex ordering: first should be minimal degeneracy"))
        end
    end
    return φ
end

#endregion

############################################################################################
# BandMetadataGenerator
#region

abstract type BandMetadataGenerator{M} end

metadata_type(::BandMetadataGenerator{M}) where {M} = M
metadata_type(::Bandstructure{<:Any,<:Any,<:Any,<:Any,M}) where {M} = M
metadata_type(_) = argerror("metadata of unknown type")

struct MissingBandMetadata <: BandMetadataGenerator{Missing}
end

BandMetadataGenerator(::Missing) = MissingBandMetadata()
BandMetadataGenerator(m::BandMetadataGenerator) = m
BandMetadataGenerator(_) = argerror("metadata should be `missing` or a valid `BandMetadataGenerator` object")

#endregion

############################################################################################
# BerryCurvature
#   Abelian and non-Abelian Berry curvature constructors for bands metadata
#region

struct BerryCurvatureAbelian{T,H<:AbstractHamiltonian{T,<:Any,2}} <: BandMetadataGenerator{T}
    h::H
    ∂1h::SparseMatrixCSC{Complex{T},Int}
    ∂2h::SparseMatrixCSC{Complex{T},Int}
    tmp1::Vector{Complex{T}}
    tmp2::Vector{Complex{T}}
end

function BerryCurvatureAbelian(h)
    ∂1h, ∂2h = h(SA[0,0], 1), h(SA[0,0], 2)
    tmp = zeros(blockeltype(h), flatsize(h))
    return BerryCurvatureAbelian(h, ∂1h, ∂2h, tmp, copy(tmp))
end

## API ##

berry_curvature(h::AbstractHamiltonian{<:Any,<:Any, 2}) = BerryCurvatureAbelian(h)
berry_curvature(h::AbstractHamiltonian) =
    argerror("Berry curvature requires a 2D AbstractHamiltonian, got $(latdim(h))D.")

function (B::BerryCurvatureAbelian{T})(ϕs, (energies, states), rng; params...) where {T}
    length(rng) == 1 ||
        argerror("BerryCurvatureAbelian can only be constructed for non-degenerate bands, found degeneracy $(length(rng)).")
    n = only(rng)
    ψn = view(states, :, n)
    en = energies[n]
    curvature = zero(T)
    ∂1h, ∂2h = B.∂1h, B.∂2h
    copy!(nonzeros(∂1h), nonzeros(call!(B.h, ϕs, 1)))
    copy!(nonzeros(∂2h), nonzeros(call!(B.h, ϕs, 2)))
    ∂1ψn = mul!(B.tmp1, ∂1h, ψn)
    ∂2ψn = mul!(B.tmp2, ∂2h, ψn)
    for (em, ψm) in zip(energies, eachcol(states))
        em == en && continue
        denom = (en - em)^2
        term = dot(∂1ψn, ψm) * dot(ψm, ∂2ψn) / denom
        curvature += -2*imag(term)
    end
    return curvature
end

#endregion
