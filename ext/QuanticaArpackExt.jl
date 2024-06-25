module QuanticaArpackExt

using Arpack
using Quantica
import Quantica: get_eigen

function Quantica.get_eigen(solver::ES.Arpack, mat)
    ε, Ψ, _ = Arpack.eigs(mat; solver.kwargs...)
    return ε, Ψ
end

end # module
