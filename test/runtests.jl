using Test
using Quantica

@testset "Quantica.jl" begin
    include("test_lattice.jl")
    include("test_model.jl")
    include("test_hamiltonian.jl")
    include("test_mesh.jl")
end
