rdr((r1, r2)::Pair) = (0.5 * (r1 + r2), r2 - r1)

############################################################################################
# Sparse matrix transformations [involves OrbitalStructure's]
# all merged_* functions assume matching structure of sparse matrices
#region

# merge several sparse matrices onto the first as structural zeros
function merge_structure(mats::Vector{<:SparseMatrixCSC{O}}) where {O}
    n, m = size(first(mats))
    n == m || throw(ArgumentError("Internal error: matrix not square"))
    collector = CSC{O}()
    for col in 1:m
        for (i, mat) in enumerate(mats)
            for p in nzrange(mat, col)
                val = i == 1 ? nonzeros(mat)[p] : zero(O)
                row = rowvals(mat)[p]
                pushtocolumn!(collector, row, val, false) # skips repeated rows
            end
        end
        finalizecolumn!(collector)
    end
    matrix = sparse(collector, n)
    return matrix
end

# flatten a sparse matrix according to OrbitalStructures
function flatten(src::SparseMatrixCSC{O}, os::OrbitalStructure{O}, flatos::OrbitalStructure{T} = flatten(os)) where {T<:Number,O<:SMatrix}
    norbs = norbitals(os)
    collector = CSC{T}()
    for col in 1:size(src, 2)
        scol = site_to_sublat(col, os)
        for j in 1:norbs[scol]
            for p in nzrange(src, col)
                row = rowvals(src)[p]
                srow = site_to_sublat(row, os)
                rowoffset´ = site_to_flatoffset(row, os, flatos)
                val = nonzeros(src)[p]
                for i in 1:norbs[srow]
                    pushtocolumn!(collector, rowoffset´ + i, val[i, j])
                end
            end
            finalizecolumn!(collector, false)
        end
    end
    matrix = sparse(collector, nsites(flatos))
    return matrix
end

site_to_sublat(siteidx, o::OrbitalStructure) = site_to_sublat(siteidx, offsets(o))

function site_to_sublat(siteidx, offsets)
    l = length(offsets)
    for s in 2:l
        @inbounds offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

function site_to_flatoffset(siteidx, orbstruct, flatorbstruct)
    s = site_to_sublat(siteidx, orbstruct)
    N = norbitals(orbstruct)[s]
    offset = offsets(orbstruct)[s]
    offset´ = offsets(flatorbstruct)[s]
    Δi = siteidx - offset
    siteidx´ = offset´ + (Δi - 1) * N
    return siteidx´
end

# merged_mul! (mul! specializations assuming target does not need to change structure [is "merged"])

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, os, A::SparseMatrixCSC{<:Number}, b::Number, β) =
    merged_mul!(C, A, b, 1, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:SMatrix{N,N}}, os, A::SparseMatrixCSC{<:SMatrix{N,N}}, b::Number, β) where {N} =
    merged_mul!(C, A, b, 1, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, os, A::SparseMatrixCSC{<:SMatrix}, b::Number, β) =
    flatten_mul!(C, os, A, b, 1, β)

function merged_mul!(C::SparseMatrixCSC, A::SparseMatrixCSC, b::Number, α = 1, β = 0)
    nzA = nonzeros(A)
    nzC = nonzeros(C)
    if length(nzA) == length(nzC)  # assume idential structure (C has merged structure)
        @. nzC = β * nzC + α * b * nzA
    else
        for col in axes(A, 2), p in nzrange(A, col)
            row = rowvals(A)[p]
            for p´ in nzrange(C, col)
                row´ = rowvals(C)[p]
                if row == row´
                    nzC[p´] = β * nzC[p´] + α * b * nzA[p]
                    break
                end
            end
        end
    end
    return C
end

function merged_flatten_mul!(C::SparseMatrixCSC, os::OrbitalStructure, A::SparseMatrixCSC, b::Number, α , β = 0)
    cols  = axes(A, 2)
    rows  = rowvals(A)
    vals  = nonzeros(A)
    norb  = norbitals(A)
    idxC  = 0
    valsC = nonzeros(C)
    for col in cols
        scol = site_to_sublat(col, os)
        ncol = norb[scol]
        for ocol in 1:ncol, p in nzrange(A, col)
            valA = vals[p]
            row  = rows[p]
            srow = site_to_sublat(row, os)
            nrow = norb[srow]
            for orow in 1:nrow
                idxC += 1
                valsC[idxC] = β * valsC[idxC] + α * b * valA[orow, ocol]
            end
        end
    end
    idxC == length(valsC) ||
        throw(ArgumentError("Attempted to do a flattening mul! onto a matrix with incompatible structure"))
    return C
end

#endregion