############################################################################################
# IJV sparse matrix builders
#region

struct IJV{B}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{B}
end

function IJV{B}(nnzguess = missing) where {B}
    i, j, v = Int[], Int[], B[]
    if nnzguess isa Integer
        sizehint!(i, nnzguess)
        sizehint!(j, nnzguess)
        sizehint!(v, nnzguess)
    end
    return IJV(i, j, v)
end

Base.push!(ijv::IJV, (i, j, v)) =
    (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))

Base.append!(ijv::IJV, (is, js, vs)) =
    (append!(ijv.i, is); append!(ijv.j, js); append!(ijv.v, vs))

function Base.filter!(f::Function, ijv::IJV)
    ind = 0
    for (i, j, v) in zip(ijv.i, ijv.j, ijv.v)
        if f(i, j, v)
            ind += 1
            ijv.i[ind] = i
            ijv.j[ind] = j
            ijv.v[ind] = v
        end
    end
    resize!(ijv.i, ind)
    resize!(ijv.j, ind)
    resize!(ijv.v, ind)
    return ijv
end

Base.isempty(h::IJV) = length(h.i) == 0

SparseArrays.sparse(c::IJV, m::Integer, n::Integer) = sparse(c.i, c.j, c.v, m, n)

#endregion

############################################################################################
# CSC Hamiltonian builder
#region

mutable struct CSC{B}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{B}
    colcounter::Int
    rowvalcounter::Int
    cosorter::CoSort{Int,B}
end

function CSC{B}(cols = missing, nnzguess = missing) where {B}
    colptr = [1]
    rowval = Int[]
    nzval = B[]
    if cols isa Integer
        sizehint!(colptr, cols + 1)
    end
    if nnzguess isa Integer
        sizehint!(nzval, nnzguess)
        sizehint!(rowval, nnzguess)
    end
    colcounter = 1
    rowvalcounter = 0
    cosorter = CoSort(rowval, nzval)
    return CSC(colptr, rowval, nzval, colcounter, rowvalcounter, cosorter)
end

function pushtocolumn!(s::CSC, row::Int, x, skipdupcheck::Bool = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
    end
    return s
end

function appendtocolumn!(s::CSC, firstrow::Int, vals, skipdupcheck::Bool = true)
    len = length(vals)
    if skipdupcheck || !any(i->isintail(firstrow + i - 1, s.rowval, s.colptr[s.colcounter]), 1:len)
        append!(s.rowval, firstrow:firstrow+len-1)
        append!(s.nzval, vals)
        s.rowvalcounter += len
    end
    return s
end

function isintail(element, container, start::Int)
    for i in start:length(container)
        container[i] == element && return true
    end
    return false
end

function sync_columns!(s::CSC, col)
    missing_cols = col - s.colcounter
    for _ in 1:missing_cols
        finalizecolumn!(s)
    end
    return nothing
end

function finalizecolumn!(s::CSC, sortcol::Bool = true)
    if sortcol
        s.cosorter.offset = s.colptr[s.colcounter] - 1
        sort!(s.cosorter)
        isgrowing(s.cosorter) || internalerror("finalizecolumn!: repeated rows")
    end
    s.colcounter += 1
    push!(s.colptr, s.rowvalcounter + 1)
    return nothing
end

function completecolptr!(colptr, cols, lastrowptr)
    colcounter = length(colptr)
    if colcounter < cols + 1
        resize!(colptr, cols + 1)
        for col in (colcounter + 1):(cols + 1)
            colptr[col] = lastrowptr + 1
        end
    end
    return colptr
end

function SparseArrays.sparse(s::CSC, m::Integer, n::Integer)
    completecolptr!(s.colptr, n, s.rowvalcounter)
    rows, cols = isempty(s.rowval) ? 0 : maximum(s.rowval), length(s.colptr) - 1
    rows <= m && cols == n ||
        internalerror("sparse: matrix size $((rows, cols)) is inconsistent with lattice size $((m, n))")
    return SparseMatrixCSC(m, n, s.colptr, s.rowval, s.nzval)
end

Base.isempty(s::CSC) = length(s.nzval) == 0

#endregion