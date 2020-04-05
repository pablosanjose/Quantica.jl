using Test
using Quantica
using Random

@testset "Quantica.jl" begin
    include("test_lattice.jl")
    include("test_model.jl")
    include("test_hamiltonian.jl")
    include("test_mesh.jl")
end
