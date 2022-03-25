#######################################################################
# EffectiveMatrix
#######################################################################
"""
    EffectiveMatrix

A dense matrix of the form `matrix = [ZaL(ω) L' 0; L*ZaR(ω) I*ω-H R*ZrL(ω); 0 R' ZrR(ω)]`,
where H, L and R are given at construction, while ω, φL, φR are given at runtime.
"""
struct EffectiveMatrix{T}
    matrix::Matrix{T}
    L::Matrix{T}
    R::Matrix{T}
    n::Int
    r::Int
end

function EffectiveMatrix(H, L, R)
    size(L) == size(R) || throw("L and R in Effective matrix should have same `size`, but are $(size(L)) and $(size(R))")
    Hd = Matrix(H)
    Ld = Matrix(L)
    Rd = Matrix(R)
    matrix = [0I Ld' 0I; 0Ld -Hd 0Rd; 0I Rd' 0I]
    n, r = size(R)
    return EffectiveMatrix(matrix, L, R, n, r)
end

function effective_matrix!(mat, e::EffectiveMatrix, ω, (ZrL, ZrR, ZaL, ZaR))
    copy!(mat, e.matrix)
    for n in axes(mat, 1)
        mat[n, n] += ω
    end
    r, n = e.r, e.n
    i1, i2, i3 = 1:r, r+1:r+n, r+n+1:n+2r
    copy!(view(mat, i1, i1), ZaL)
    copy!(view(mat, i3, i3), ZrR)
    mul!(view(mat, i2, i1), e.L, ZaR)
    mul!(view(mat, i2, i3), e.R, ZrL)
    return mat
end

pad_effective(m::StridedMatrix{T}, e::EffectiveMatrix, ::Val{:L}) where {T} =
    [zeros(T, e.r, size(m, 2)); m]
pad_effective(m::StridedMatrix{T}, e::EffectiveMatrix, ::Val{:R}) where {T} =
    [m; zeros(T, e.r, size(m, 2))]
pad_effective(m::StridedMatrix{T}, e::EffectiveMatrix, ::Val{:LR}) where {T} =
    [zeros(T, e.r, size(m, 2)); m; zeros(T, e.r, size(m, 2))]

unpad_effective(m::StridedMatrix, e::EffectiveMatrix, ::Val{:L}) = view(m, e.r+1:e.r+e.n, :)
unpad_effective(m::StridedMatrix, e::EffectiveMatrix, ::Val{:R}) = view(m, 1:e.n, :)
unpad_effective(m::StridedMatrix, e::EffectiveMatrix, ::Val{:LR}) = view(m, e.r+1:e.r+e.n, :)

padded_lu!(invmat, e::EffectiveMatrix, ::Val{:L}) = lu!(view(invmat, 1:e.r+e.n, 1:e.r+e.n))
padded_lu!(invmat, e::EffectiveMatrix, ::Val{:R}) = lu!(view(invmat, e.r+1:2e.r+e.n, e.r+1:2e.r+e.n))
padded_lu!(invmat, e::EffectiveMatrix, ::Val{:LR}) = lu!(invmat)

padded_ldiv(invmat::AbstractMatrix, mat::AbstractMatrix, e::EffectiveMatrix, side) =
    padded_ldiv(padded_lu!(invmat, e, side), mat, e, side)
padded_ldiv(fact::Factorization, mat, e::EffectiveMatrix, side) =
    unpad_effective(ldiv!(fact, pad_effective(mat, e, side)), e, side)

Base.eltype(::EffectiveMatrix{T}) where {T} = T

#######################################################################
# EffectiveHamiltonian
#######################################################################