using Test
using Quantica

@testset "Quantica.jl" begin
    include("test_lattice.jl")
    # include("test_model.jl")
    # include("test_ket.jl")
    include("test_hamiltonian.jl")
    # include("test_mesh.jl")
    include("test_bandstructure.jl")
    # include("test_greens.jl")
    # include("test_KPM.jl")
end
