using Test
using Quantica

@testset "Quantica.jl" begin
    include("test_lattice.jl")
    include("test_hamiltonian.jl")
    include("test_bandstructure.jl")
    include("test_greenfunction.jl")
    include("test_show.jl")
end
