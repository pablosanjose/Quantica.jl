module QuanticaKrylovKitExt

using KrylovKit
using Quantica
import Quantica: get_eigen

function Quantica.get_eigen(solver::ES.KrylovKit, mat)
    ε, Ψ, _ = KrylovKit.eigsolve(mat, solver.params...; solver.kwargs...)
    return ε, Ψ
end

end # module
