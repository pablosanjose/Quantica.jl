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
                rowoffset´, nrow = site_to_flatoffset_norbs(row, os, flatos)
                val = nonzeros(src)[p]
                for i in 1:nrow
                    pushtocolumn!(collector, rowoffset´ + i, val[i, j])
                end
            end
            finalizecolumn!(collector, false)
        end
    end
    matrix = sparse(collector, nsites(flatos))
    return matrix
end

function site_to_sublat(siteidx, orbstruct)
    offsets´ = offsets(orbstruct)
    l = length(offsets´)
    for s in 2:l
        @inbounds offsets´[s] + 1 > siteidx && return s - 1
    end
    return l
end

function site_to_flatoffset_norbs(siteidx, orbstruct, flatorbstruct)
    s = site_to_sublat(siteidx, orbstruct)
    N = norbitals(orbstruct)[s]
    offset = offsets(orbstruct)[s]
    offset´ = offsets(flatorbstruct)[s]
    Δi = siteidx - offset
    flatoffset = offset´ + (Δi - 1) * N
    return flatoffset, N
end

# merged_mul! (mul! specializations assuming target does not need to change structure [is "merged"])

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, _, A::SparseMatrixCSC{<:Number}, b::Number, α, β) =
    merged_mul!(C, A, b, α, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:SMatrix{N,N}}, _, A::SparseMatrixCSC{<:SMatrix{N,N}}, b::Number, α, β) where {N} =
    merged_mul!(C, A, b, α, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, os, A::SparseMatrixCSC{<:SMatrix}, b::Number, α, β) =
    flatten_merged_mul!(C, os, A, b, α, β)

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

function flatten_merged_mul!(C::SparseMatrixCSC, (os, flatos), A::SparseMatrixCSC, b::Number, α , β = 0)
    colsA = axes(A, 2)
    rowsA = rowvals(A)
    valsA = nonzeros(A)
    rowsC = rowvals(C)
    valsC = nonzeros(C)
    for col in colsA
        coloffset´, ncol = site_to_flatoffset_norbs(col, os, flatos)
        for p in nzrange(A, col)
            valA = valsA[p]
            row  = rowsA[p]
            rowoffset´, nrow = site_to_flatoffset_norbs(row, os, flatos)
            rowfirst´ = rowoffset´ + 1
            for ocol in 1:ncol
                col´ = coloffset´ + ocol
                for p´ in nzrange(C, col´)
                    if rowsC[p´] == rowfirst´
                        for orow in 1:nrow
                            p´´ = p´ + orow - 1
                            valsC[p´´] = β * valsC[p´´] + α * b * valA[orow, ocol]
                        end
                        break
                    end
                end
            end
        end
    end
    return C
end

#endregion