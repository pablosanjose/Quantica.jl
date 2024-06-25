module QuanticaArpackExt

using Arpack
using Quantica

function Quantica.get_eigen(solver::ES.ArnoldiMethod, mat)
    ε, Ψ, _ = Arpack.eigs(mat; solver.kwargs...)
    return ε, Ψ
end

end # module
