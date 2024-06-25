module QuanticaArnoldiMethodExt

using ArnoldiMethod
using Quantica

function Quantica.get_eigen(solver::ES.ArnoldiMethod, mat)
    pschur, _ = ArnoldiMethod.partialschur(mat; solver.kwargs...)
    ε, Ψ = ArnoldiMethod.partialeigen(pschur)
    return ε, Ψ
end

end # module
