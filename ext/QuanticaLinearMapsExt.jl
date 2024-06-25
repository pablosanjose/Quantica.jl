module QuanticaLinearMapsExt

using LinearMaps
using LinearAlgebra
using SparseArrays
using Quantica

function Quantica.get_eigen(solver::ES.ShiftInvert, mat::AbstractSparseMatrix{T}) where {T<:Number}
    mat´ = mat - I * solver.origin
    F = lu(mat´)
    lmap = LinearMap{T}((x, y) -> ldiv!(x, F, y), size(mat)...;
        ismutating = true, ishermitian = false)
    ε, Ψ = solver.eigensolver(lmap)
    @. ε = 1 / ε + solver.origin
    return ε, Ψ
end

Quantica.get_eigen(::ES.ShiftInvert, mat) =
    Quantica.argerror("ShiftInvert requires a sparse matrix input, but received a matrix of type $(typeof(mat)).")

end # module
